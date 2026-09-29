# CreditExplain AI

<div align="center">
  <img src="credit_explain_capa.jpg" width="100%" alt="Capa do Projeto">
</div>

<div align="center">

## Sistema de explicações de risco de crédito com LLM especializado por preference fine tuning

CreditExplain AI é um projeto de IA aplicada para transformar sinais produzidos por um motor de risco de crédito em explicações claras, consistentes e auditáveis. O LLM não decide concessão de crédito. A decisão e o score permanecem em um modelo de risco separado. O LLM atua somente na camada explicativa.

O diferencial técnico é o uso de preference fine tuning em duas etapas. A primeira aprende preferência humana geral a partir de 57.477 comparações reais entre respostas de LLM. A segunda adapta o comportamento ao domínio de crédito com pares de explicações construídos por uma política explícita de qualidade, fidelidade ao score, concisão, ausência de atributos protegidos e linguagem não prescritiva.

## 1. Problema de negócio

Instituições financeiras frequentemente precisam explicar resultados produzidos por modelos de risco para times internos, atendimento, auditoria e clientes. Uma explicação útil precisa satisfazer simultaneamente quatro requisitos.

1. Fidelidade aos fatores realmente usados pelo modelo de risco.
2. Consistência de linguagem entre clientes e canais.
3. Controle de alucinação e de inferências não suportadas pelos dados.
4. Rastreabilidade suficiente para auditoria e monitoramento.

Gerar texto diretamente com um LLM base pode produzir respostas eloquentes, mas inconsistentes. O objetivo deste projeto é especializar o comportamento do modelo com dados de preferência e criar uma camada de avaliação independente.

## 2. Origem e papel do dataset

O corpus bruto vem do dataset LLM Classification Finetuning, composto por conversas do Chatbot Arena. Cada registro contém um prompt, duas respostas candidatas e a preferência humana entre resposta A, resposta B ou empate.

O dataset é usado como corpus de preferência geral, não como base de crédito. Isso é uma escolha deliberada. Renomear as colunas e fingir que as conversas são dados financeiros criaria um projeto artificial. Aqui o corpus ensina sinais gerais de qualidade de resposta, enquanto a adaptação ao domínio financeiro é feita em uma segunda etapa controlada.

Resumo observado no arquivo fornecido:

| Métrica | Valor |
| --- | ---: |
| Registros de treino | 57.477 |
| Modelos distintos | 64 |
| Preferência por A | 20.064, 34,91% |
| Preferência por B | 19.652, 34,19% |
| Empate | 17.761, 30,90% |
| Mediana de turnos por conversa | 1 |
| Percentil 99 de turnos | 5 |
| Duplicatas exatas de pares | 71 |

A licença informada na página de dados do Kaggle é CC BY NC 4.0. Portanto, este repositório deve ser tratado como projeto educacional e de portfólio enquanto utilizar esse corpus. Para uso comercial, substitua o corpus por dados com licença compatível.

Fonte: https://www.kaggle.com/competitions/llm-classification-finetuning/overview

## 3. Arquitetura

```text
Applicant features
        |
        v
Credit risk model
XGBoost or scorecard
        |
        | score + reason codes
        v
Explanation context builder
        |
        v
Fine tuned LLM
SFT + DPO with QLoRA
        |
        v
Policy validator
factuality + forbidden attributes + format
        |
        v
Preference judge
three class A / B / tie
        |
        v
API response + audit trace
```

O projeto separa claramente três responsabilidades.

1. Risk engine. Calcula score e reason codes. Não usa o LLM.
2. Explanation model. Recebe apenas fatores aprovados e produz uma explicação.
3. Preference judge. Avalia qualidade comparativa e pode ser usado em testes offline ou reranking.

## 4. Estratégia de fine tuning

### 4.1 General preference adaptation

Os pares inequívocos do dataset são convertidos para o formato prompt, chosen, rejected. Os empates não são descartados do projeto. Eles são preservados para treinamento e avaliação do judge de três classes.

Para reduzir position bias, cada par inequívoco pode ser duplicado com a ordem A e B invertida mantendo a semântica de chosen e rejected. O split é feito por hash do prompt normalizado para impedir que conversas quase idênticas apareçam em treino e validação.

### 4.2 Credit domain SFT

Casos sintéticos estruturados são criados a partir de variáveis de crédito não sensíveis. O objetivo não é simular clientes reais, e sim criar contexto controlado para treinamento da camada explicativa.

Cada exemplo contém:

```json
{
  "risk_score": 0.63,
  "risk_band": "elevated",
  "reason_codes": [
    "high_debt_to_income",
    "recent_delinquencies"
  ],
  "facts": {
    "debt_to_income": 0.47,
    "bureau_score": 612,
    "delinquencies_12m": 2
  }
}
```

A resposta alvo descreve somente os fatores presentes no contexto e nunca cria um motivo novo.

### 4.3 Credit domain DPO

Para cada cenário, o pipeline cria um par de explicações.

1. Chosen. Factual, curta, clara, sem atributos protegidos, sem promessas e sem inventar causalidade.
2. Rejected. Contém uma violação controlada, por exemplo fator inexistente, excesso de certeza, linguagem vaga, recomendação indevida ou verbosidade desnecessária.

Isso permite otimizar diretamente a preferência do LLM para o comportamento esperado no domínio.

### 4.4 Preference judge

O judge é treinado como classificador de três classes usando prompt, response A e response B. Ele preserva o sinal de empate original e serve para medir qualidade relativa, position bias e sensibilidade à verbosidade.

## 5. Modelo recomendado

A configuração padrão usa `Qwen/Qwen2.5-3B-Instruct` com QLoRA em 4 bits.

A escolha foi feita por três motivos.

1. Tamanho suficiente para demonstrar fine tuning de LLM sem exigir infraestrutura de cluster.
2. Suporte maduro em Transformers, PEFT, bitsandbytes e TRL.
3. Possibilidade de treinar adapters LoRA e manter o modelo base imutável.

Para uma GPU de maior memória, troque o modelo por uma variante de 7B ou 8B sem alterar o contrato do pipeline.

## 6. Métricas

O projeto não usa apenas loss de treinamento.

### Preference model

| Métrica | Objetivo |
| --- | --- |
| Log loss | calibração das probabilidades A, B e tie |
| Macro F1 | equilíbrio entre as três classes |
| Accuracy | leitura operacional simples |
| Expected Calibration Error | confiabilidade probabilística |
| Position flip rate | estabilidade ao inverter A e B |
| Length bias slope | dependência indevida do tamanho da resposta |

### Explanation model

| Métrica | Objetivo |
| --- | --- |
| Reason code recall | todos os fatores autorizados relevantes são mencionados |
| Unsupported claim rate | nenhuma alegação sem suporte no contexto |
| Forbidden attribute rate | zero referência a atributos protegidos |
| JSON schema pass rate | 100% quando saída estruturada estiver habilitada |
| Preference win rate | proporção de vezes em que o judge prefere o modelo ajustado ao baseline |
| Pairwise human agreement | concordância com revisão humana em amostra auditada |

## 7. Controles de risco

Este projeto deliberadamente não permite que o LLM tome decisões de crédito.

1. O score é produzido por um modelo separado.
2. O prompt contém somente reason codes aprovados.
3. Atributos protegidos não entram no contexto de geração.
4. A resposta passa por validação determinística.
5. Toda inferência recebe `trace_id`, versão do modelo, versão da política e reason codes.
6. O monitoramento acompanha drift de entrada, comprimento de resposta, falhas de schema e taxa de alegações não suportadas.

## 8. Estrutura do repositório

```text
CreditExplain_AI/
  configs/
    train.yaml
  data/
    raw/
      train.csv
      test.csv
      sample_submission.csv
    processed/
  docs/
    DATA_CARD.md
    MODEL_CARD.md
    SYSTEM_DESIGN.md
  notebooks/
    01_data_audit.ipynb
    02_build_preference_dataset.ipynb
    03_credit_domain_adaptation.ipynb
    04_qlora_dpo_training.ipynb
    05_evaluation_and_bias_audit.ipynb
  src/credit_explain/
    data/
    training/
    evaluation/
    risk/
    api/
    monitoring/
  tests/
```

## 9. Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para QLoRA é recomendada GPU NVIDIA com suporte CUDA. O audit e a geração do dataset de domínio podem ser executados em CPU.

## 10. Execução

### Auditoria do corpus

```bash
python scripts/audit_dataset.py
```

### Construção dos pares de preferência

```bash
python scripts/build_preference_data.py
```

### Construção do dataset de adaptação ao crédito

```bash
python scripts/build_credit_domain_data.py --rows 12000
```

### Fine tuning SFT

```bash
python -m credit_explain.training.train_sft
```

### Fine tuning DPO

```bash
python -m credit_explain.training.train_dpo
```

### Treino do preference judge

```bash
python -m credit_explain.training.train_judge
```

### Avaliação

```bash
python -m credit_explain.evaluation.evaluate_judge
python -m credit_explain.evaluation.evaluate_explanations
```

### API

```bash
uvicorn credit_explain.api.main:app --host 0.0.0.0 --port 8000
```

## 11. Exemplo de uso

Entrada:

```json
{
  "application_id": "demo_1024",
  "risk_score": 0.63,
  "risk_band": "elevated",
  "reason_codes": [
    "high_debt_to_income",
    "recent_delinquencies"
  ],
  "facts": {
    "debt_to_income": 0.47,
    "delinquencies_12m": 2,
    "bureau_score": 612
  }
}
```

Saída esperada:

```json
{
  "summary": "O nível de risco foi classificado como elevado.",
  "factors": [
    "A relação entre dívida e renda está acima da faixa de referência usada pelo modelo.",
    "Foram observados atrasos recentes no histórico informado ao motor de risco."
  ],
  "limitations": "A explicação descreve os fatores do modelo e não substitui revisão de política ou análise humana.",
  "trace_id": "..."
}
```

## 12. O que torna este projeto tecnicamente relevante

O projeto demonstra que fine tuning não é tratado como uma etapa isolada. Ele inclui formulação de objetivo, preparação de preference data, QLoRA, SFT, DPO, tratamento de empates, split sem leakage, calibração, auditoria de viés de posição e verbosidade, validação determinística, serving, observabilidade e governança.

A principal decisão arquitetural é simples: o LLM explica uma decisão produzida por um sistema de risco, mas não substitui esse sistema.
