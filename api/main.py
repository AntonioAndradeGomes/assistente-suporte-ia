"""
Fase 3 - API que serve o sistema (Componente C)
Projeto: Assistente Inteligente de Suporte
Integra o classificador (Fase 1) e o RAG (Fase 2) num serviço HTTP.
Antes de rodar:
    pip install -r requirements.txt
    (o .env com GEMINI_API_KEY já deve existir na raiz)
Rodar:
    uvicorn api.main:app --reload
Testar no navegador:
    http://127.0.0.1:8000/docs   <- documentação interativa automática
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Raiz do projeto, resolvida a partir deste arquivo (api/main.py -> ..).
# Assim os artefatos são encontrados independentemente de onde o uvicorn é chamado.
BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_PATH = BASE_DIR / "modelo_classificador.joblib"
BASE_CONHECIMENTO_PATH = BASE_DIR / "data" / "base_conhecimento.md"

load_dotenv(BASE_DIR / ".env")

# 1. Contratos de entrada e saída (validação Pydantic)
class Solicitacao(BaseModel):
    """O que a API recebe. Pydantic valida automaticamente:
    campo ausente, tipo errado ou texto vazio -> erro 422 com detalhe."""
    texto: str = Field(min_length=3, description="Texto da solicitação do cliente")
    
class Fonte(BaseModel):
    similaridade: float
    secao: str

class Resposta(BaseModel):
    """O que a API devolve."""
    categoria: str
    resposta: str
    fontes: list[Fonte]

#2. Carregamento dos artefatos UMA VEZ, no startup
# Tudo que é caro (modelo joblib, embeddings, índice) vive neste dict,
# preenchido quando o servidor sobe e reutilizado em toda requisição.
recursos = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    print("Carregando artefatos...")
    if not MODELO_PATH.exists():
        raise RuntimeError(
            f"{MODELO_PATH.name} não encontrado. Rode primeiro: python fase1_classificador.py"
        )
    if not BASE_CONHECIMENTO_PATH.exists():
        raise RuntimeError(
            f"{BASE_CONHECIMENTO_PATH} não encontrada. Rode primeiro: python fase2_rag.py"
        )
    recursos["classificador"] = joblib.load(MODELO_PATH)
    recursos["emb_model"] = SentenceTransformer("all-MiniLM-L6-v2")

    with open(BASE_CONHECIMENTO_PATH, encoding="utf-8") as f:
        conteudo = f.read()
    chunks = [c.strip() for c in conteudo.split("## ") if c.strip() and not c.startswith("#")]
    recursos["chunks"] = ["## " + c for c in chunks]
    recursos["emb_chunks"] = recursos["emb_model"].encode(
        recursos["chunks"], normalize_embeddings=True
    )
    recursos["gemini"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print(f"Pronto: classificador + {len(chunks)} chunks indexados.")
    yield
    recursos.clear()

app = FastAPI(title="Assistente Inteligente de Suporte", lifespan=lifespan)


# 3. Lógica do RAG (mesma da Fase 2)
LIMIAR_SIMILARIDADE = 0.30
MSG_NAO_ENCONTRADO = "Não encontrei essa informação na base de conhecimento."

PROMPT_RAG = """Você é um assistente de suporte ao cliente.
Responda a pergunta do cliente com base no contexto abaixo.
Se o contexto contiver informação relevante para a pergunta, mesmo que
parcial, use-a para responder da melhor forma possível.
Diga exatamente "Não encontrei essa informação na base de conhecimento."
APENAS se o contexto não tiver relação com a pergunta.
Não invente informações que não estejam no contexto. Responda em português.
CONTEXTO:
{contexto}

PERGUNTA DO CLIENTE:
{pergunta}
"""

def buscar(pergunta: str, k: int = 3):
    emb = recursos["emb_model"].encode([pergunta], normalize_embeddings=True)[0]
    sims = recursos["emb_chunks"] @ emb
    top = np.argsort(sims)[-k:][::-1]
    return [(float(sims[i]), recursos["chunks"][i]) for i in top]

def gerar_resposta_rag(pergunta: str) -> tuple[str, list[Fonte]]:
    resultados = buscar(pergunta)
    if resultados[0][0] < LIMIAR_SIMILARIDADE:
        return MSG_NAO_ENCONTRADO, []
    contexto = "\n\n---\n\n".join(chunk for _, chunk in resultados)
    resp = recursos["gemini"].models.generate_content(
        model="gemini-3.6-flash",
        contents=PROMPT_RAG.format(contexto=contexto, pergunta=pergunta),
    )
    fontes = [
        Fonte(similaridade=round(sim, 3), secao=chunk.splitlines()[0])
        for sim, chunk in resultados
    ]
    # resp.text vem None se o Gemini bloquear a geração (filtro de segurança).
    # Sem esta guarda, o response_model rejeitaria None e a API devolveria 500.
    return resp.text or MSG_NAO_ENCONTRADO, fontes

@app.post("/solicitacao", response_model=Resposta)
def processar_solicitacao(s: Solicitacao) -> Resposta:
    """Recebe o texto de uma solicitação e retorna:
    - a categoria prevista pelo classificador (Fase 1)
    - a resposta gerada pelo RAG (Fase 2), com as fontes usadas
    """
    categoria = recursos["classificador"].predict([s.texto])[0]
    resposta, fontes = gerar_resposta_rag(s.texto)
    return Resposta(categoria=categoria, resposta=resposta, fontes=fontes)

@app.get("/health")
def health():
    return {"status": "ok"}