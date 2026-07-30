# Relatório Técnico — Assistente Inteligente de Suporte

Projeto final, Módulo 8 (IA e Aprendizado de Máquina).

---

## 1. O problema

Uma equipe de suporte que recebe centenas de mensagens por dia gasta tempo em duas
tarefas mecânicas: descobrir para qual setor cada mensagem vai, e procurar na base
de conhecimento a informação que responde o cliente. O sistema automatiza as duas.

Dado o texto de uma solicitação, ele devolve a categoria prevista (para roteamento)
e uma resposta fundamentada na base de conhecimento, com os trechos que a
embasaram. O objetivo não é substituir o atendente, e sim entregar a ele uma
sugestão auditável: ele vê a resposta e vê de onde ela saiu.

## 2. Os dados

Fonte única: [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
26.872 solicitações de clientes em inglês.

A escolha de um dataset só, para os dois componentes, foi deliberada. A coluna
`instruction` + `category` treina o classificador; a coluna `response` monta a base
do RAG. Assim os dois componentes falam do mesmo domínio — se o classificador
roteia uma mensagem para REFUND, a base do RAG tem conteúdo sobre reembolso.

**Distribuição das 11 categorias** (desbalanceada, razão de 6,3x entre extremos):

| Categoria | N | Categoria | N |
|---|---|---|---|
| ACCOUNT | 5.986 | FEEDBACK | 1.997 |
| ORDER | 3.988 | DELIVERY | 1.994 |
| REFUND | 2.992 | SHIPPING | 1.970 |
| INVOICE | 1.999 | SUBSCRIPTION | 999 |
| CONTACT | 1.999 | CANCEL | 950 |
| PAYMENT | 1.998 | | |

**Tamanho dos textos:** média de 8,7 palavras (desvio 2,6; mínimo 1; máximo 16).
São mensagens curtas e uniformes — um fato que volta a ser importante na seção 3.4.

## 3. Componente A — Classificador

### 3.1 Preparação e cuidado com leakage

A ordem das operações foi escolhida para eliminar leakage por construção:

1. `train_test_split` com `test_size=0.2` e `stratify=y` — **antes** de qualquer
   vetorização. A estratificação preserva a proporção das 11 classes nos dois
   conjuntos, o que importa num dataset desbalanceado.
2. O `TfidfVectorizer` vive **dentro** de um `Pipeline` do scikit-learn. Isso
   garante que o vocabulário e os pesos IDF são aprendidos apenas no treino: o
   `fit()` do pipeline nunca vê o teste, e o `predict()` só aplica a transformação
   já ajustada.

Se o TF-IDF fosse ajustado no dataset inteiro antes do split, as estatísticas de
frequência do teste vazariam para o treino. O `Pipeline` é o que torna isso
impossível de acontecer por descuido.

Divisão final: **21.497 treino / 5.375 teste**.

### 3.2 Vetorização

TF-IDF com `ngram_range=(1,2)` e `min_df=3`.

Bigramas porque em suporte a unidade de sentido frequentemente tem duas palavras:
`cancel order`, `delivery address`, `payment methods`. O unigrama `order` sozinho é
ambíguo entre ORDER e CANCEL; `cancel order` não é. O `min_df=3` descarta termos que
aparecem em menos de três documentos, cortando typos raros que só inflariam a
dimensionalidade sem generalizar.

### 3.3 Duas abordagens comparadas

| Modelo | F1 macro | Erros no teste |
|---|---|---|
| **Regressão logística** | **0,9964** | 19 / 5.375 |
| Multinomial Naive Bayes | 0,9958 | 23 / 5.375 |

A métrica de seleção é **F1 macro**, não acurácia. Com ACCOUNT valendo 22% do
dataset e CANCEL valendo 3,5%, a acurácia é dominada pelas classes grandes: um
modelo que ignorasse CANCEL por completo perderia pouca acurácia. O F1 macro dá
peso igual a cada classe, então penaliza justamente o que queremos evitar. Foi por
isso que a decisão de qual modelo serializar usou F1 macro.

A regressão logística venceu por margem pequena (0,0006) e foi a serializada. Numa
diferença tão estreita, o desempate real foi a interpretabilidade: os coeficientes
da LogReg são diretamente legíveis como importância de termo por classe (seção 4),
o que o Naive Bayes não oferece com a mesma clareza.

Precision, recall e F1 por classe estão na saída completa do
`classification_report`; nenhuma das 11 classes ficou abaixo de 0,98 em qualquer das
três métricas. As matrizes de confusão dos dois modelos estão em
`matriz_confusao_logreg.png` e `matriz_confusao_naive_bayes.png`.

### 3.4 Discussão honesta: por que 0,996 não é um número para se orgulhar

Um F1 macro de 0,9964 deveria gerar desconfiança, não satisfação. Investiguei.

**Primeira hipótese: leakage por duplicatas.** O dataset tem 2.237 textos
duplicados exatos. Com o split aleatório, **604 dos 5.375 exemplos de teste (11,2%)
aparecem idênticos no conjunto de treino**. O split está corretamente implementado,
mas se o dado bruto contém repetições, o modelo é parcialmente testado em frases que
literalmente memorizou.

Refiz a avaliação removendo duplicatas antes do split:

| Cenário | LogReg | Naive Bayes |
|---|---|---|
| Com duplicatas (26.872 linhas) | 0,9964 | 0,9958 |
| Sem duplicatas (24.635 linhas) | 0,9964 | 0,9963 |

A hipótese **não se confirmou**: o F1 não se move. As duplicatas não explicam o
resultado.

**Segunda hipótese, essa sim sustentada: o dataset é sintético e quase
linearmente separável.** Três evidências independentes apontam para isso:

- **Uniformidade dos textos.** Média de 8,7 palavras com desvio de 2,6, e máximo de
  16. Mensagens reais de suporte não têm essa regularidade — variam de "help" a
  parágrafos inteiros. Isso é assinatura de geração por template.
- **Os termos discriminantes são quase um dicionário.** A seção 4 mostra que
  `account` prediz ACCOUNT, `refund` prediz REFUND, `invoice` prediz INVOICE. A
  tarefa degenerou em busca de palavra-chave; quase não há ambiguidade lexical para
  o modelo resolver.
- **O modelo aprendeu artefatos do template.** Entre os 8 termos mais indicativos de
  INVOICE estão os literais `00108`, `37777`, `85632`, `12588` — números de fatura
  dos placeholders. Não é conhecimento linguístico, é decoreba de um artefato da
  geração dos dados, e não generalizaria para uma fatura com outro número.

**Conclusão.** 0,9964 mede a separabilidade do Bitext, não a dificuldade de
classificar suporte real. Em produção, com linguagem espontânea, gírias, typos,
mensagens multi-assunto e categorias mal definidas na fronteira, esse número cairia
de forma substancial. A seção 7.1 quantifica parte dessa queda: basta introduzir
30% de erros de digitação para a confiança média do modelo cair de 0,942 para
0,812.

O valor do projeto está no pipeline — split correto, métrica adequada ao
desbalanceamento, interpretabilidade, monitoramento — e não no F1.

## 4. Interpretabilidade e análise de erros

### 4.1 Importância de features

Os coeficientes da regressão logística indicam quais termos mais empurram um texto
para cada classe. Extraindo os 8 maiores por categoria:

| Categoria | Termos mais indicativos |
|---|---|
| ACCOUNT | account, signup, user, registration, profile |
| CANCEL | termination, cancellation, withdrawal, penalty, early |
| CONTACT | customer, agent, contact, speak, talk |
| DELIVERY | delivery, shipment, arrive, shipping, methods |
| FEEDBACK | feedback, claim, reclamation, complaint, file |
| INVOICE | invoice, bill, `00108`, `37777`, `85632` |
| ORDER | order, purchase, order number, several, product |
| PAYMENT | payment, payments, modalities, payment methods |
| REFUND | refund, money, reimbursement, rebate, compensation |
| SHIPPING | address, delivery address, shipping address |
| SUBSCRIPTION | newsletter, subscription, unsubscribe, corporate |

O resultado é coerente com o domínio e confirma que os bigramas pegaram sentido
real: `delivery address` e `shipping address` discriminam SHIPPING, enquanto
`delivery` sozinho puxa para DELIVERY. A distinção entre essas duas classes é
justamente onde o modelo mais erra (4.2).

Os números de fatura em INVOICE são o achado mais útil desta análise, e é um achado
negativo: sem olhar os coeficientes, não haveria como suspeitar que o modelo estava
se apoiando em placeholders.

### 4.2 Análise de erros

19 erros em 5.375. A distribuição deles não é aleatória:

| Real → Previsto | N |
|---|---|
| SHIPPING → ACCOUNT | 7 |
| PAYMENT → ACCOUNT | 5 |
| PAYMENT → CANCEL | 2 |
| CONTACT → ACCOUNT | 1 |
| ORDER → ACCOUNT | 1 |
| PAYMENT → DELIVERY | 1 |
| SUBSCRIPTION → REFUND | 1 |

**Padrão 1: 14 dos 19 erros vão para ACCOUNT**, a classe majoritária (22% do
dataset). Esse é o comportamento clássico de um classificador quando o sinal
desaparece: sem evidência no vetor, ele cai no prior.

**Padrão 2: o sinal desaparece por ruído de digitação.** Os exemplos errados
mostram exatamente isso:

| Texto | Real | Previsto |
|---|---|---|
| `help me to check the acceptedpayment modalities` | PAYMENT | CANCEL |
| `where do i report an issue with paynents` | PAYMENT | ACCOUNT |
| `i dont know what i need to do to correct theaddress` | SHIPPING | ACCOUNT |
| `can i pay witj visa` | PAYMENT | ACCOUNT |

`paynents`, `witj`, `acceptedpayment`, `theaddress`: palavra concatenada ou letra
trocada. O token não existe no vocabulário TF-IDF, é descartado, e o texto perde a
única palavra que carregava a informação. Essa é a fragilidade estrutural do TF-IDF
— ele trabalha com tokens exatos e não tem noção de similaridade ortográfica.

**Mitigação possível**, não implementada aqui: acrescentar features de n-gramas de
caracteres (`analyzer="char_wb"`, `ngram_range=(3,5)`). `paynents` e `payments`
compartilham os trigramas `pay`, `ayn`/`aym`, `nts`, então o sinal sobreviveria à
troca de letra. É a próxima coisa que eu faria neste componente.

O único erro fora desses dois padrões é `ORDER → ACCOUNT` em "would you give me
information about cancelling orders?", que é genuinamente ambíguo — um humano
poderia rotular como CANCEL.

## 5. Componente B — RAG

### 5.1 Base de conhecimento

Construída a partir da coluna `response` do Bitext. O dataset tem 27 intenções
(`cancel_order`, `get_refund`, `check_refund_policy`, …) agrupadas nas 11
categorias. Para cada par (categoria, intenção) tomei **uma** resposta
representativa, gerando um "manual de atendimento" de 27 seções e ~2.400 palavras
em `data/base_conhecimento.md`.

Cada seção recebe um cabeçalho `## [CATEGORIA] intencao`, o que serve a dois
propósitos: delimita o chunk e, como o cabeçalho é devolvido no campo `fontes`, dá
ao atendente uma referência legível de onde a resposta saiu.

**Limitação reconhecida:** 2.400 palavras é uma base enxuta. Ela cobre bem as 27
intenções do dataset, mas qualquer pergunta ligeiramente fora delas cai no
curto-circuito da seção 5.3. Uma base de produção teria políticas completas, prazos,
exceções e casos de borda. Ampliar seria simples — bastaria tomar 3 ou 4 respostas
por intenção em vez de uma, ou anexar FAQ e termos de uso reais.

### 5.2 Chunking, embeddings e busca

O chunking segue a estrutura semântica em vez de cortar por número de palavras: cada
seção `##` já é uma unidade coesa, sobre uma única intenção. Cortar em janelas de
200 palavras partiria respostas no meio; aqui não há esse risco.

Embeddings com `all-MiniLM-L6-v2` (384 dimensões), normalizados. Com vetores
normalizados o produto escalar **é** a similaridade de cosseno, então a busca inteira
é um `emb_chunks @ emb_pergunta` — uma multiplicação de matriz de 27×384 por 384.
Não há necessidade de um vector store: com 27 chunks, a busca exaustiva é
instantânea e evita uma dependência.

A base e as perguntas ficam em inglês porque o `all-MiniLM-L6-v2` tem desempenho
melhor nesse idioma; o prompt instrui o LLM a responder em português. Uma base em
português exigiria um modelo de embeddings multilíngue.

### 5.3 Proteção contra alucinação: duas camadas

**Camada 1 — curto-circuito por limiar.** Se a maior similaridade entre a pergunta
e os 27 chunks for menor que **0,30**, o sistema devolve
`"Não encontrei essa informação na base de conhecimento."` sem chamar o LLM.

O limiar de 0,30 foi calibrado observando as similaridades reais: perguntas dentro
do domínio recuperam o melhor chunk entre 0,55 e 0,70 (a pergunta
"How do I cancel my order?" recupera `[ORDER] cancel_order` a 0,683), enquanto
"What is the capital of France?" não passa de 0,30 contra nenhuma seção. A folga
entre as duas faixas é confortável.

**Camada 2 — prompt ancorado.** Quando o LLM é chamado, o prompt entrega os 3
chunks mais similares como contexto e instrui a responder apenas com base neles,
com a mesma frase de escape se o contexto não tiver relação. É a rede de segurança
para o caso de a similaridade passar do limiar mas o conteúdo não servir.

A camada 1 não é redundante em relação à camada 2 — ela é o que evita gastar
latência e cota de LLM numa pergunta que já se sabe fora de escopo. Ver 7.2.

### 5.4 Auditabilidade (diferencial)

Toda resposta vem acompanhada dos trechos que a embasaram e da similaridade de cada
um:

```json
"fontes": [
  {"similaridade": 0.683, "secao": "## [ORDER] cancel_order"},
  {"similaridade": 0.55,  "secao": "## [CANCEL] check_cancellation_fee"},
  {"similaridade": 0.433, "secao": "## [ORDER] place_order"}
]
```

Isso muda a natureza do sistema. Sem as fontes, o atendente precisa confiar na
resposta; com elas, ele confere em segundos se o RAG puxou o trecho certo. E quando
o sistema erra, as fontes dizem se o problema foi na recuperação (trecho errado) ou
na geração (trecho certo, resposta ruim) — sem isso, depurar um RAG é adivinhação.

## 6. Componente C — API

FastAPI com dois endpoints. `POST /solicitacao` recebe o texto e devolve categoria,
resposta e fontes; `GET /health` para checagem de disponibilidade.

**Validação de entrada.** Pydantic com `Field(min_length=3)`. Um texto de 2
caracteres é rejeitado com **422** e o detalhe do campo, antes de chegar ao modelo:

```json
{"detail":[{"type":"string_too_short","loc":["body","texto"],
  "msg":"String should have at least 3 characters"}]}
```

A saída também é tipada (`response_model=Resposta`), o que garante o contrato e
alimenta a documentação automática em `/docs`.

**Carregamento único no startup.** O `lifespan` do FastAPI carrega o `.joblib`,
instancia o modelo de embeddings e indexa os 27 chunks **uma vez**, guardando tudo
num dicionário reutilizado por todas as requisições. Treinar — ou mesmo só
recarregar — por requisição inviabilizaria o serviço: o `SentenceTransformer` leva
segundos para instanciar.

**Robustez.** Os caminhos dos artefatos são resolvidos a partir da localização de
`api/main.py`, não do diretório de trabalho, então o `uvicorn` pode ser chamado de
qualquer lugar. Se um artefato faltar, o startup falha com uma mensagem que diz qual
script rodar. E se o Gemini bloquear a geração por filtro de segurança (`resp.text`
vem `None`), a resposta cai na mensagem padrão em vez de estourar um 500.

### Verificação ponta a ponta

| Caso | Entrada | Resultado |
|---|---|---|
| Dentro do domínio | `How do I cancel my order?` | 200, categoria ORDER, resposta com 3 fontes |
| Fora do domínio | `What is the capital of France?` | 200, mensagem de "não encontrei", `fontes: []` |
| Entrada inválida | `ab` | 422 com detalhe do campo |
| Disponibilidade | `GET /health` | 200, `{"status":"ok"}` |

## 7. Componente D — Reflexão sobre operação

### 7.1 Monitoramento e drift

Em produção não há rótulo no momento da inferência. Não é possível calcular F1 em
tempo real, e é aí que a maioria dos sistemas de ML apodrece silenciosamente: o
modelo continua respondendo com confiança enquanto a qualidade cai.

O que **é** observável são distribuições. Implementei a detecção em
`monitoramento_drift.py`, usando o **teste de Kolmogorov-Smirnov** para comparar o
tráfego atual com o perfil de referência do treino. KS é não paramétrico, não assume
normalidade, e devolve um p-valor: p < 0,05 significa que as duas amostras
dificilmente vêm da mesma distribuição.

Dois sinais monitorados: a **confiança do modelo** (máximo da `predict_proba`) e o
**tamanho do texto** em palavras. Resultados medidos:

| Cenário simulado | Confiança (média) | KS confiança | Tamanho | Veredito |
|---|---|---|---|---|
| Controle (mesma distribuição) | 0,942 → 0,945 | p = 0,41 | p = 0,66 | sem drift |
| 30% de erros de digitação | 0,942 → **0,812** | p = 7e-134 | p = 0,96 | **drift** |
| Textos mais longos (canal novo) | 0,942 → **0,742** | p < 1e-300 | p ≈ 0 | **drift** |
| Assunto fora das 11 classes | 0,942 → **0,425** | p ≈ 0 | p = 5e-270 | **drift** |

Três leituras importam aqui:

- **O cenário de controle não dispara.** Um monitor que acusa drift em tráfego
  normal é pior que nenhum monitor, porque treina a equipe a ignorar o alerta.
- **Os dois sinais são complementares.** O drift de digitação derruba a confiança
  mas não muda o tamanho do texto; só o primeiro sinal o detecta. O drift de canal
  move os dois. Monitorar apenas o tamanho perderia o caso mais realista.
- **O caso mais perigoso é o mais detectável.** Assunto novo — a empresa lança um
  produto e chega uma solicitação que não cabe em nenhuma das 11 categorias — derruba
  a confiança de 0,942 para 0,425. O modelo vai prever alguma classe existente e o
  roteamento vai para o setor errado; nenhuma métrica sem rótulo perceberia, mas a
  confiança grita.

**Como isso viraria operação:** salvar o perfil de referência versionado junto com
o `.joblib`; logar predição, confiança e tamanho a cada requisição; job diário
rodando KS contra a referência. E o ponto que considero mais importante: **o alerta
não deve retreinar sozinho.** Ele deve disparar amostragem — separar ~200
solicitações do período, rotular à mão, medir F1 de verdade — e só então decidir por
retreino. Retreino automático em cima de um sinal indireto é como assinar cheque em
branco.

### 7.2 Custo e latência do RAG

Latências medidas na verificação da seção 6:

| Caminho | Latência |
|---|---|
| Classificador (TF-IDF + LogReg) | poucos milissegundos |
| Busca semântica (27 chunks) | ~10 ms |
| Curto-circuito (fora do domínio, sem LLM) | **0,115 s** |
| Pipeline completo com chamada ao Gemini | **9,56 s** |

A chamada ao LLM domina o tempo total em duas ordens de magnitude — o caminho com
LLM é **83x mais lento** que o caminho sem. Isso tem três consequências práticas:

**O curto-circuito por limiar é uma decisão de custo, não só de qualidade.** Toda
pergunta fora do domínio que ele intercepta é uma chamada de API não gasta e 9
segundos economizados. Numa operação real, onde parte do tráfego é ruído, isso é a
diferença entre viável e caro.

**9,5 segundos é aceitável no design escolhido, e não seria em outro.** O sistema
sugere resposta para um atendente humano, que leva mais tempo que isso só para ler a
mensagem do cliente. Se o mesmo pipeline fosse exposto direto ao cliente num chat, 9
segundos de silêncio seria inaceitável, e a saída seria streaming da resposta ou um
modelo menor.

**Quando o LLM não vale a pena.** Para as 27 intenções que a base já cobre com uma
resposta canônica, devolver o texto do chunk recuperado seria instantâneo, gratuito e
sem risco de alucinação. O LLM agrega valor quando a pergunta combina informação de
mais de um chunk, ou está fraseada de um jeito que a resposta canônica não cobre. Uma
evolução natural seria uma banda: acima de ~0,85 de similaridade devolver o chunk
direto; entre 0,30 e 0,85 chamar o LLM; abaixo de 0,30 dizer que não sabe.

O tier gratuito do Gemini tem limite de requisições por minuto. Em produção isso
exigiria fila com retry e backoff — hoje uma rajada de requisições simultâneas
receberia erro de cota.

### 7.3 Limites honestos do sistema

**O F1 não descreve a realidade.** Discutido na seção 3.4: 0,9964 mede a
separabilidade de um dataset sintético. Não tenho uma estimativa confiável do
desempenho em mensagens reais, e não é honesto apresentar esse número como se
tivesse.

**O sistema é monolíngue.** Classificador treinado em inglês, embeddings com melhor
desempenho em inglês. Uma solicitação em português seria classificada
essencialmente ao acaso.

**A base do RAG cobre 27 intenções e nada além.** Não há política de garantia,
prazo de troca, exceção regional. Perguntas legítimas de suporte fora dessas 27
recebem "não encontrei" — o comportamento é seguro, mas a cobertura é estreita.

**Não há avaliação quantitativa do RAG.** A recuperação foi verificada
qualitativamente em algumas perguntas. Não construí um conjunto de pares
pergunta-trecho-correto para medir recall@k, então "a busca funciona" é uma
afirmação sobre os casos que testei, não uma métrica.

**O classificador quebra com typo.** Consequência direta do TF-IDF por token exato
(seção 4.2). N-gramas de caracteres resolveriam boa parte.

**Não há autenticação, rate limiting ou log estruturado.** A API está no nível de
prova de conceito. O monitoramento descrito em 7.1 pressupõe log por requisição, que
é o primeiro item a implementar antes de qualquer uso real.

**Categoria nova é ponto cego por construção.** O classificador só sabe prever as 11
classes que viu. A detecção de drift avisa que algo mudou, mas a correção — definir
a nova categoria, rotular exemplos, retreinar — é trabalho humano.

## 8. Reprodutibilidade

Dependências pinadas em `requirements.txt`; instruções de execução no `README.md`;
chave de API isolada em `.env` (com `.env.example` versionado como modelo, e o
`.env` real no `.gitignore`).

O `modelo_classificador.joblib` está versionado no repositório, então a API sobe sem
precisar retreinar. Os `random_state` estão fixos em 42 no split e nas amostragens do
script de drift, então os números deste relatório são reproduzíveis.

`Dockerfile` incluído: `docker build -t assistente-suporte .` e
`docker run --rm -p 8000:8000 --env-file .env assistente-suporte`. O modelo de
embeddings é baixado durante o build, não no primeiro startup, para o contêiner
subir pronto para atender.
