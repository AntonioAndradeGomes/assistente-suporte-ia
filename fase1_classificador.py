"""
Fase 1 - Classificador de solicitações (Componente A)
Projeto: Assistente Inteligente de Suporte
Dataset: Bitext Customer Support (Hugging Face)
Rode célula a célula num notebook, ou como script:
    python fase1_classificador.py
"""
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)

# Os gráficos vão para doc/, junto do relatório que os referencia.
# Resolvido a partir deste arquivo, então funciona de qualquer diretório.
DOC_DIR = Path(__file__).resolve().parent / "doc"
DOC_DIR.mkdir(exist_ok=True)

#1. Carregar o dataset
#baixa o dataset e guarda em cache local
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = ds["train"].to_pandas()


# Colunas que nos interessam:
#   instruction -> texto da solicitação do cliente (nossa feature)
#   category    -> categoria (nosso rótulo, 11 classes)
#   response    -> resposta modelo (NÃO usamos aqui; será a base do RAG na Fase 2)
df = df[["instruction", "category", "response"]].dropna()

print(f"Total de exemplos: {len(df)}")
print("\nDistribuição das categorias:")
print(df["category"].value_counts())

# Gráfico de distribuição
df["category"].value_counts().plot(kind="barh", figsize=(8, 5))
plt.title("Distribuição das categorias")
plt.tight_layout()
plt.savefig(DOC_DIR / "distribuicao_categorias.png", dpi=120)
plt.close()

# Tamanho dos textos (também vai para o relatório)
df["n_palavras"] = df["instruction"].str.split().str.len()
print("\nTamanho dos textos (palavras):")
print(df["n_palavras"].describe())

# 2. Split treino/teste — ANTES de qualquer vetorização
# stratify preserva a proporção das classes nos dois conjuntos.
# O TF-IDF será ajustado SÓ no treino, dentro do Pipeline -> sem leakage.
X = df["instruction"]
y = df["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTreino: {len(X_train)} | Teste: {len(X_test)}")

# 3. Duas abordagens para comparar
# Pipeline = vetorizador + modelo num objeto só.
# fit() ajusta o TF-IDF apenas no treino; predict() só aplica ao teste.

pipelines = {
    "logreg": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
    "naive_bayes": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3)),
        ("clf", MultinomialNB()),
    ]),
}

resultados = {}
for nome, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    resultados[nome] = f1_macro
    print(f"\n{'=' * 60}\nModelo: {nome}  (F1 macro: {f1_macro:.4f})\n{'=' * 60}")
    # precision, recall e F1 POR CLASSE — o que a rubrica exige
    print(classification_report(y_test, y_pred))
    # Matriz de confusão: onde o modelo confunde uma classe com outra?
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, xticks_rotation=90, colorbar=False
    )
    plt.title(f"Matriz de confusão - {nome}")
    plt.tight_layout()
    plt.savefig(DOC_DIR / f"matriz_confusao_{nome}.png", dpi=120)
    plt.close()
melhor = max(resultados, key=resultados.get)
print(f"\nMelhor modelo por F1 macro: {melhor} ({resultados[melhor]:.4f})")

# 4. Interpretabilidade (XAI): importância de features da LogReg
# Os coeficientes da regressão logística indicam quais palavras/bigramas
# mais "puxam" cada categoria. Isso atende o núcleo de interpretabilidade.

pipe_lr = pipelines["logreg"]
vocab = pipe_lr.named_steps["tfidf"].get_feature_names_out()
coefs = pipe_lr.named_steps["clf"].coef_
classes = pipe_lr.named_steps["clf"].classes_

print("\nTop 8 termos mais indicativos de cada categoria (LogReg):")
for i, classe in enumerate(classes):
    top = coefs[i].argsort()[-8:][::-1]
    termos = ", ".join(vocab[j] for j in top)
    print(f"  {classe}: {termos}")

# 5. Análise de erros: olhar exemplos que o melhor modelo errou

y_pred_melhor = pipelines[melhor].predict(X_test)
erros = pd.DataFrame({
    "texto": X_test,
    "real": y_test,
    "previsto": y_pred_melhor,
})
erros = erros[erros["real"] != erros["previsto"]]
print(f"\nTotal de erros no teste: {len(erros)}")
print("\nPares (real -> previsto) mais frequentes:")
print(erros.groupby(["real", "previsto"]).size().sort_values(ascending=False).head(10))
print("\nAlguns exemplos de erro para discutir no relatório:")
print(erros.sample(min(5, len(erros)), random_state=42).to_string(max_colwidth=80))

# 6. Serializar o pipeline vencedor (a API carrega este arquivo)
joblib.dump(pipelines[melhor], "modelo_classificador.joblib")
print("\nPipeline salvo em modelo_classificador.joblib")