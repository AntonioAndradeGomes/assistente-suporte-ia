"""
Fase 2 - Assistente de resposta via RAG (Componente B)
Projeto: Assistente Inteligente de Suporte

Fluxo: montar base de conhecimento -> chunking -> embeddings ->
    busca semântica -> resposta ancorada via Gemini.

Antes de rodar:
    pip install sentence-transformers google-genai python-dotenv

    Crie um arquivo .env na raiz do projeto com:
        GEMINI_API_KEY=sua_chave_aqui

Rodar:
    python fase2_rag.py
"""

import os
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# Carrega as variáveis do arquivo .env para o os.environ
load_dotenv()

# 1. Montar a base de conhecimento a partir do Bitext
# A coluna `response` traz respostas modelo de atendimento. Pegamos
# UMA resposta representativa por intenção (27 intenções) e montamos
# um "manual de atendimento": coeso, do mesmo domínio do classificador.

BASE_PATH = "data/base_conhecimento.md"

if not os.path.exists(BASE_PATH):
    os.makedirs("data", exist_ok=True)
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = ds["train"].to_pandas()[["intent", "category", "response"]].dropna()

    base_docs = []

    for (categoria, intencao), grupo in df.groupby(["category", "intent"]):
        resposta = grupo['response'].iloc[0]
        base_docs.append(f"## [{categoria}] {intencao}\n\n{resposta}\n")

    with open(BASE_PATH, "w", encoding="utf-8") as f:
        f.write("# Base de Conhecimento - Suporte ao Cliente.\n\n")
        f.write("\n".join(base_docs))
    print(f"Base de conhecimento criada em {BASE_PATH} ({len(base_docs)} seções)")
else:
    print(f"Base de conhecimento já existe em {BASE_PATH}")


# 2. Chunking: quebrar a base em pedaços recuperáveis
# Cada seção "## [categoria] intencao" já é um chunk natural e focado.
# (Se a base fosse texto corrido, quebraríamos por ~200-300 palavras.)
with open(BASE_PATH, encoding="utf-8") as f:
    conteudo = f.read()

chunks = [c.strip() for c in conteudo.split("## ") if c.strip() and not c.startswith("#")]
chunks = ["## " + c for c in chunks]
print(f"Total de chunks: {len(chunks)}")

# 3. Embeddings + índice
# Cada chunk vira um vetor que captura seu significado. Normalizando
# os vetores, o produto escalar equivale à similaridade de cosseno.
print("Gerando embeddings (primeira vez baixa o modelo ~90MB)...")
modelo_emb = SentenceTransformer("all-MiniLM-L6-v2")
emb_chunks = modelo_emb.encode(chunks, normalize_embeddings=True)
print(f"Índice pronto: {emb_chunks.shape[0]} chunks x {emb_chunks.shape[1]} dimensões")


def buscar(pergunta: str, k: int = 3) -> list[tuple[float, str]]:
    """Busca semântica: retorna os k chunks mais similares à pergunta."""
    emb_pergunta = modelo_emb.encode([pergunta], normalize_embeddings=True)[0]
    similaridades = emb_chunks @ emb_pergunta  # cosseno (vetores normalizados)
    top_idx = np.argsort(similaridades)[-k:][::-1]
    return [(float(similaridades[i]), chunks[i]) for i in top_idx]


# 4. Geração ancorada com o Gemini
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY não encontrada. Crie um arquivo .env na raiz "
        "do projeto com a linha: GEMINI_API_KEY=sua_chave_aqui"
    )

client = genai.Client(api_key=API_KEY)

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


def responder(pergunta: str, k: int = 3) -> dict:
    """Pipeline RAG completo: busca -> prompt ancorado -> Gemini."""
    resultados = buscar(pergunta, k)

    # Curto-circuito: se nem o melhor chunk é minimamente similar,
    # a pergunta está fora do domínio da base. Respondemos direto,
    # sem gastar uma chamada ao LLM.
    LIMIAR_SIMILARIDADE = 0.30
    if resultados[0][0] < LIMIAR_SIMILARIDADE:
        return {
            "pergunta": pergunta,
            "resposta": "Não encontrei essa informação na base de conhecimento.",
            "fontes": [],
        }

    contexto = "\n\n---\n\n".join(chunk for _, chunk in resultados)

    resposta = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=PROMPT_RAG.format(contexto=contexto, pergunta=pergunta),
    )
    return {
        "pergunta": pergunta,
        "resposta": resposta.text,
        # As fontes usadas: diferencial de auditabilidade
        "fontes": [(round(sim, 3), chunk.splitlines()[0]) for sim, chunk in resultados],
    }

if __name__ == "__main__":
    perguntas_teste = [
        # Perguntas cuja resposta ESTÁ na base:
        "How do I cancel my order?",
        "I want to get a refund, what should I do?",
        # Pergunta cuja resposta NÃO está na base
        # -> deve acionar a proteção contra alucinação:
        "What is the capital of France?",
    ]

    for p in perguntas_teste:
        r = responder(p)
        print(f"\n{'=' * 70}\nPERGUNTA: {r['pergunta']}\n")
        print(f"RESPOSTA: {r['resposta']}")
        print("FONTES RECUPERADAS (similaridade | seção):")
        for sim, titulo in r["fontes"]:
            print(f"  {sim} | {titulo}")

