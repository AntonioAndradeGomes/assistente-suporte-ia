FROM python:3.11-slim

WORKDIR /app

# Dependências primeiro, em camada separada: o cache do Docker só é invalidado
# quando o requirements.txt muda, não a cada alteração no código.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baixa o modelo de embeddings na imagem (~90MB). Sem isto o primeiro startup
# do contêiner faria download, deixando o /health indisponível por vários segundos.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY api/ ./api/
COPY data/ ./data/
COPY modelo_classificador.joblib .

EXPOSE 8000

# A GEMINI_API_KEY vem do ambiente (--env-file .env), nunca da imagem.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
