# Assistente Inteligente de Suporte

Projeto final do Módulo 8 — Inteligência Artificial e Aprendizado de Máquina.

Dado o texto de uma solicitação de cliente, o sistema faz duas coisas:

1. **Classifica** a solicitação em uma de 11 categorias de atendimento (ML clássico).
2. **Responde** à dúvida com base numa base de conhecimento, via RAG, sem inventar informação.

Tudo servido por uma API FastAPI.

## Arquitetura

```
fase1_classificador.py         Componente A - TF-IDF + LogReg/NaiveBayes, avaliação e XAI
fase2_rag.py                   Componente B - base de conhecimento, embeddings, busca, Gemini
api/main.py                    Componente C - API FastAPI integrando A e B
monitoramento_drift.py         Diferencial  - detecção de drift com Kolmogorov-Smirnov
doc/RELATORIO.md               Relatório técnico e reflexão sobre operação (Componente D)
doc/*.png                      Gráficos da Fase 1 e capturas da documentação da API
data/base_conhecimento.md      Base do RAG (gerada pela Fase 2 a partir do Bitext)
modelo_classificador.joblib    Pipeline vencedor, carregado pela API no startup
Dockerfile                     Containerização da API (diferencial)
```

📄 **[Leia o relatório técnico completo →](doc/RELATORIO.md)**

Dados: [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(26.872 solicitações rotuladas). O mesmo dataset alimenta o classificador (coluna
`instruction`/`category`) e a base do RAG (coluna `response`), garantindo coerência de domínio.

## Instalação

Requer Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Chave de API

O componente de RAG usa o Gemini (tier gratuito). Obtenha uma chave em
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) e configure:

```bash
cp .env.example .env
# edite o .env e preencha: GEMINI_API_KEY=sua_chave_aqui
```

O `.env` está no `.gitignore` e nunca é versionado.

## Como rodar

Os três componentes são independentes e podem ser executados nesta ordem.

### 1. Classificador (Componente A)

```bash
python fase1_classificador.py
```

Baixa o dataset (cacheado em `~/.cache/huggingface`), treina e compara os dois
modelos, imprime as métricas por classe e gera:

- `modelo_classificador.joblib` — pipeline vencedor, usado pela API
- `doc/distribuicao_categorias.png`, `doc/matriz_confusao_logreg.png`,
  `doc/matriz_confusao_naive_bayes.png` — os gráficos que o relatório referencia

O modelo treinado já vem versionado no repositório, então este passo é opcional
para apenas subir a API.

### 2. RAG (Componente B)

```bash
python fase2_rag.py
```

Na primeira execução monta `data/base_conhecimento.md` a partir do dataset; nas
seguintes reaproveita o arquivo. Ao final roda três perguntas de teste, incluindo
uma fora do domínio para demonstrar a proteção contra alucinação.

### 3. API (Componente C)

```bash
uvicorn api.main:app --reload
```

Documentação interativa em <http://127.0.0.1:8000/docs>.

Os artefatos são resolvidos a partir da localização do próprio `api/main.py`, então
o comando funciona de qualquer diretório.

### 4. Monitoramento de drift (diferencial)

```bash
python monitoramento_drift.py
```

## A API

### `POST /solicitacao`

```bash
curl -X POST http://127.0.0.1:8000/solicitacao \
  -H "Content-Type: application/json" \
  -d '{"texto":"How do I cancel my order?"}'
```

```json
{
  "categoria": "ORDER",
  "resposta": "Compreendo que você tem uma dúvida sobre o cancelamento do seu pedido...",
  "fontes": [
    {"similaridade": 0.683, "secao": "## [ORDER] cancel_order"},
    {"similaridade": 0.55,  "secao": "## [CANCEL] check_cancellation_fee"},
    {"similaridade": 0.433, "secao": "## [ORDER] place_order"}
  ]
}
```

O campo `fontes` expõe quais trechos da base embasaram a resposta, com a
similaridade de cada um — auditabilidade.

Perguntas fora do domínio retornam `"Não encontrei essa informação na base de
conhecimento."` com `fontes` vazio, sem gastar chamada ao LLM.

Validação Pydantic: `texto` com menos de 3 caracteres retorna **422** com o detalhe
do erro.

### `GET /health`

```json
{"status": "ok"}
```

## Resultados

| Modelo | F1 macro |
|---|---|
| Regressão logística (TF-IDF 1-2 gramas) | **0,9964** |
| Multinomial Naive Bayes (TF-IDF 1-2 gramas) | 0,9958 |

19 erros em 5.375 exemplos de teste. A discussão desses números — incluindo por
que eles são altos demais para se levar a sério como estimativa de produção —
está no [relatório](doc/RELATORIO.md).

## Docker (diferencial)

```bash
docker build -t assistente-suporte .
docker run --rm -p 8000:8000 --env-file .env assistente-suporte
```

Build verificado (imagem de 3,29 GB); o contêiner sobe e responde em
<http://127.0.0.1:8000/docs>. A `GEMINI_API_KEY` vem do `--env-file` em tempo de
execução e nunca entra na imagem.

A imagem é grande porque o wheel padrão do `torch` para Linux embute as bibliotecas
CUDA, inúteis aqui — a inferência roda em CPU. Este e os outros pontos abertos do
projeto estão documentados, com a correção de cada um, na
[seção 7.4 do relatório](doc/RELATORIO.md#74-pontos-abertos-e-próximos-passos).
