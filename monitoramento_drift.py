"""
Diferencial - Monitoramento de drift (Componente D, implementado)
Projeto: Assistente Inteligente de Suporte

A reflexão sobre operação (RELATORIO.md) argumenta que o classificador precisa ser
monitorado porque a linguagem dos clientes muda com o tempo. Este script não só
descreve o problema: simula três cenários de drift e detecta cada um com o teste
de Kolmogorov-Smirnov, mostrando qual sinal dispara em cada caso.

Rodar (a partir da RAIZ do projeto):
    python monitoramento_drift.py

Por que KS: em produção não existe rótulo no momento da inferência. Não é possível
medir F1 em tempo real. O que se pode medir são distribuições de coisas
observáveis, e comparar com a distribuição de referência do treino. O KS é um
teste não paramétrico que compara duas distribuições contínuas sem assumir
normalidade, e devolve um p-valor: p baixo = as duas amostras dificilmente vêm da
mesma distribuição = drift.

Os dois sinais monitorados aqui:
    1. Confiança do modelo (máx. da predict_proba). Cai quando chegam textos que o
       modelo não reconhece. É o sinal mais valioso porque não depende de rótulo.
    2. Tamanho do texto em palavras. Detecta mudança de canal (chat curto -> e-mail
       longo), que altera a distribuição de entrada mesmo sem mudar o assunto.
"""

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import ks_2samp

ALPHA = 0.05  # nível de significância: p < ALPHA -> acusamos drift
RNG = np.random.default_rng(42)


# 1. Carregar o modelo em produção e os dados de referência

pipeline = joblib.load("modelo_classificador.joblib")

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = ds["train"].to_pandas()[["instruction", "category"]].dropna()

# "Referência" = a distribuição que o modelo viu no treino. Em produção real este
# perfil seria calculado uma vez, no deploy, e salvo como baseline.
referencia = df.sample(4000, random_state=42)["instruction"].tolist()

# Pool separado, sem interseção com a referência, para simular o tráfego novo.
pool_novo = df.drop(index=df.sample(4000, random_state=42).index)


def perfil(textos: list[str]) -> dict:
    """Extrai os sinais monitoráveis de um lote de tráfego."""
    probas = pipeline.predict_proba(textos)
    return {
        "confianca": probas.max(axis=1),
        "n_palavras": np.array([len(t.split()) for t in textos], dtype=float),
    }


def testar_drift(perfil_ref: dict, perfil_atual: dict, cenario: str) -> None:
    """Aplica KS sinal por sinal e imprime o veredito."""
    print(f"\n{'=' * 66}\nCENÁRIO: {cenario}\n{'=' * 66}")
    houve_drift = False
    for sinal in perfil_ref:
        ks, p = ks_2samp(perfil_ref[sinal], perfil_atual[sinal])
        media_ref = perfil_ref[sinal].mean()
        media_atual = perfil_atual[sinal].mean()
        alerta = p < ALPHA
        houve_drift |= alerta
        print(
            f"  {sinal:<12} média {media_ref:>7.3f} -> {media_atual:>7.3f} | "
            f"KS={ks:.4f} p={p:.2e} | {'DRIFT' if alerta else 'estável'}"
        )
    print(f"  --> Veredito: {'ALERTA, investigar' if houve_drift else 'sem drift detectado'}")


perfil_ref = perfil(referencia)


# 2. Cenário de controle: tráfego novo, mesma distribuição
# Serve para provar que o monitor não dá falso positivo. Se acusasse drift aqui,
# o alerta seria inútil em produção.

amostra_estavel = pool_novo.sample(2000, random_state=1)["instruction"].tolist()
testar_drift(perfil_ref, perfil(amostra_estavel), "controle - mesma distribuição")


# 3. Drift de ruído: erros de digitação
# A análise de erros da Fase 1 mostrou que os 19 erros do modelo concentram-se em
# textos com typos e palavras concatenadas ("paynents", "witj", "theaddress"):
# o token não existe no vocabulário TF-IDF, o sinal se perde e o modelo cai na
# classe majoritária. Este cenário simula um canal com mais ruído de digitação
# (ex.: migração para atendimento por celular).

def corromper(texto: str, taxa: float = 0.30) -> str:
    palavras = texto.split()
    saida = []
    for palavra in palavras:
        if len(palavra) > 3 and RNG.random() < taxa:
            i = RNG.integers(1, len(palavra) - 1)
            palavra = palavra[:i] + palavra[i + 1:]  # remove um caractere do meio
        saida.append(palavra)
    return " ".join(saida)


amostra_ruido = [corromper(t) for t in pool_novo.sample(2000, random_state=2)["instruction"]]
testar_drift(perfil_ref, perfil(amostra_ruido), "drift de ruído - erros de digitação")


# 4. Drift de canal: textos mais longos
# Cliente que escreve por e-mail é mais verboso que quem usa o chat. O assunto é o
# mesmo, mas a distribuição de entrada muda. Simulamos concatenando saudação e
# despedida ao redor da solicitação original.

def alongar(texto: str) -> str:
    return (
        "Hello, I hope this message finds you well. I am writing because "
        f"{texto} Thank you very much in advance for your kind assistance. Best regards."
    )


amostra_longa = [alongar(t) for t in pool_novo.sample(2000, random_state=3)["instruction"]]
testar_drift(perfil_ref, perfil(amostra_longa), "drift de canal - textos mais longos")


# 5. Drift de assunto: categoria nova, ausente do treino
# O caso mais perigoso e o mais realista: a empresa lança um produto e passa a
# receber um assunto que não existe em nenhuma das 11 classes. O modelo não tem
# como acertar — ele vai prever alguma classe existente, com confiança menor.
# Nenhuma métrica de acurácia acusaria isso sem rótulo; a confiança acusa.

assuntos_novos = [
    "I want to enable two-factor authentication on my mobile app",
    "How do I connect my smart watch to the loyalty program?",
    "Can I pay using cryptocurrency in the new checkout?",
    "Where do I find the carbon footprint report of my purchase?",
    "How do I join the beta program for the new dashboard?",
]
amostra_nova = [assuntos_novos[i % len(assuntos_novos)] for i in range(2000)]
testar_drift(perfil_ref, perfil(amostra_nova), "drift de assunto - categoria fora do treino")


# 6. Como isto viraria monitoramento de verdade

print(
    """
======================================================================
COMO ISTO OPERARIA EM PRODUÇÃO
======================================================================
  1. No deploy, salvar o perfil de referência (confiança e tamanho) junto
     com o .joblib, versionado com o modelo.
  2. Registrar em log, a cada requisição, a predição, a confiança e o
     tamanho do texto. Custo desprezível, e é o que torna o resto possível.
  3. Job diário: rodar KS do último dia contra a referência. p < 0.05 em
     qualquer sinal abre alerta.
  4. O alerta não retreina sozinho. Ele dispara amostragem: separar 200
     solicitações do período, rotular à mão, medir F1 de verdade. Só então
     decidir por retreino.
  5. Guardar histórico de p-valores. Uma queda gradual ao longo de semanas
     é drift real; um pico isolado costuma ser incidente pontual.
"""
)
