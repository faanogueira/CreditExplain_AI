# CreditExplain AI 🛡️💳

### Sistema de explicabilidade de risco de crédito via LLM alinhado por preferência (SFT + DPO)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers%20%26%20TRL-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA%204--bit-blueviolet?style=for-the-badge)](https://github.com/huggingface/peft)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CC%20BY--NC%204.0-orange?style=for-the-badge)](#2-origem-e-papel-do-dataset)

---

> **TL;DR (30s):** decisão de crédito e explicação são sistemas separados. O motor de risco (scorecard/XGBoost) calcula score e reason codes. Um LLM (`Qwen2.5-3B-Instruct`) especializado via **QLoRA + SFT + DPO em duas etapas** — 57.477 comparações reais de preferência humana + 12.000 casos sintéticos de domínio de crédito — converte esses sinais em explicações auditáveis. Um validador determinístico e um preference judge de 3 classes garantem fidelidade, ausência de atributos protegidos e rastreabilidade (`trace_id`).

**Fit ideal:** Cientista de Dados Sênior · Risk Analytics · Model Risk Management · Applied AI / LLM Engineer para bancos, fintechs e bureaus.

---

## 1. Problema de negócio

Instituições financeiras precisam explicar resultados de modelos de risco para clientes, atendimento, auditoria e reguladores. Uma explicação útil precisa satisfazer **4 requisitos simultâneos**:

| # | Requisito | Por que falha com LLM base (zero-shot) |
|---|---|---|
| 1 | **Fidelidade** aos fatores realmente usados pelo modelo de risco | LLMs comerciais alucinam motivos de reprovação |
| 2 | **Consistência** de linguagem entre canais e atendentes | Respostas variam a cada chamada |
| 3 | **Zero alucinação / zero atributo protegido** | Uso indevido de idade, gênero, estado civil expõe risco jurídico (LGPD, FCRA/ECOA) |
| 4 | **Rastreabilidade** para auditoria e monitoramento | Sem versão de modelo, política ou `trace_id` |

Delegar a decisão de crédito a um LLM também viola governança de risco (**SR 11-7**). Por isso, a regra de ouro deste projeto é:

> **O LLM explica. Não decide.** Score e decisão permanecem em um modelo de risco separado.

---

## 2. Origem e papel do dataset

O corpus bruto vem do **LLM Classification Finetuning** (Kaggle), com conversas do Chatbot Arena: prompt, duas respostas candidatas e a preferência humana (A, B ou empate).

Esse corpus é usado como **preferência geral**, não como dado de crédito — renomear colunas para simular um dataset financeiro criaria um projeto artificial. Aqui ele ensina qualidade geral de resposta; a adaptação ao domínio financeiro acontece em uma segunda etapa controlada, com dados sintéticos e política explícita.

**Auditoria do corpus (`scripts/audit_dataset.py`):**

| Métrica | Valor |
| --- | ---: |
| Registros de treino | 57.477 |
| Modelos distintos (A + B) | 64 |
| Preferência por A | 20.064 (34,91%) |
| Preferência por B | 19.652 (34,19%) |
| Empate | 17.761 (30,90%) |
| Mediana de turnos por conversa | 1 |
| Percentil 99 de turnos | 5 |
| Duplicatas exatas de pares | 71 |

⚠️ Licença **CC BY-NC 4.0** — este repositório é um projeto educacional/portfólio enquanto usar esse corpus. Para uso comercial, substitua por dados com licença compatível.

Fonte: [kaggle.com/competitions/llm-classification-finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning/overview)

---

## 3. Arquitetura

```text
Applicant features
        |
        v
Risk Engine (XGBoost / scorecard) ---- fora do LLM
        |
        | score + reason codes aprovados
        v
Context Builder (whitelist de variáveis, sem atributos sensíveis)
        |
        v
LLM Explicador (Qwen2.5-3B-Instruct + QLoRA: SFT + DPO)
        |
        v
Policy Validator (schema + factuality + forbidden attributes)
        |-- falha --> Fallback determinístico + alerta de auditoria
        |
        v
Preference Judge (A / B / tie) ---- rerank e regressão offline
        |
        v
API response + trace_id + model_version
        |
        v
Monitoramento (drift, PSI, falha de schema, unsupported rate)
```

Três responsabilidades claramente separadas:

1. **Risk engine** — calcula score e reason codes. Não usa o LLM.
2. **Explanation model** — recebe apenas fatores aprovados e produz uma explicação.
3. **Preference judge** — avalia qualidade comparativa; usado em testes offline e reranking.

---

## 4. Estratégia de fine-tuning

### 4.1 Adaptação de preferência geral
Pares inequívocos do dataset viram `(prompt, chosen, rejected)`. Empates **não são descartados** — servem para treinar e avaliar o judge de 3 classes. Cada par pode ser duplicado com A/B invertidos (reduz position bias). Split por **hash SHA-256 do prompt normalizado**, impedindo leakage entre treino e validação.

### 4.2 SFT de domínio de crédito
Casos sintéticos estruturados a partir de variáveis não sensíveis:

```json
{
  "risk_score": 0.63,
  "risk_band": "elevated",
  "reason_codes": ["high_debt_to_income", "recent_delinquencies"],
  "facts": {
    "debt_to_income": 0.47,
    "bureau_score": 612,
    "delinquencies_12m": 2
  }
}
```

A resposta-alvo descreve **apenas** os fatores presentes no contexto — nunca inventa um motivo novo.

### 4.3 DPO de domínio de crédito
Para cada cenário, um par `chosen` (factual, curto, sem atributo protegido, sem causalidade inventada) vs. `rejected`, com violação controlada:

| Variante rejeitada | Violação |
|---|---|
| 0 | Certeza indevida + atributo vago |
| 1 | Fator inexistente + linguagem vaga |
| 2 | Recomendação prescritiva + atributo protegido |

### 4.4 Preference judge
Classificador de 3 classes (`prompt, response_A, response_B → A/B/tie`), usado para medir qualidade relativa, position bias e sensibilidade a verbosidade.

---

## 5. Modelo e stack

| Camada | Ferramentas |
| --- | --- |
| Core & dados | Python 3.10+, Pandas, NumPy, Scikit-learn, Pydantic v2 |
| Fine-tuning | PyTorch 2.4, Transformers, PEFT, TRL, BitsAndBytes |
| LLM base | `Qwen/Qwen2.5-3B-Instruct` — QLoRA NF4 4-bit, double quant, bfloat16 |
| Serving | FastAPI, Uvicorn |
| Observabilidade | MLflow, Prometheus Client, Pytest |
| Config | YAML centralizado (`configs/train.yaml`) |

**Por que Qwen2.5-3B:** tamanho suficiente para demonstrar fine-tuning real sem exigir cluster; suporte maduro em Transformers/PEFT/TRL; adapters LoRA mantêm o modelo base imutável. Troca para 7B/8B sem alterar o contrato do pipeline, se houver GPU disponível.

**Hiperparâmetros de treino:**

| Estágio | Configuração | Saída |
| --- | --- | --- |
| SFT domínio | 1 epoch, lr 1e-4, batch efetivo 32, warmup 0.05 | `artifacts/credit_sft_adapter` |
| DPO geral + domínio | 1 epoch, lr 5e-5, beta 0.1, batch efetivo 32, max_len 2048 | `artifacts/credit_dpo_adapter` |
| Judge 3 classes | 1 epoch, lr 5e-5, label smoothing 0.03 | classificador A/B/tie |
| LoRA | r=32, alpha=64, dropout 0.05, 7 módulos (q/k/v/o/gate/up/down), <2% params treináveis | adapter versionado |

Seed fixo `42`, `device_map=auto`, `gradient_checkpointing=True`. QLoRA roda em ~6.5 GB de VRAM (viável em GPU de entrada tipo T4/A10G). Auditoria e geração de dados de domínio rodam em CPU.

---

## 6. Métricas e gates de avaliação

> As metas abaixo são os **gates de aprovação** usados antes de subir um novo adapter — não suba um adapter que não passe nesses critérios. Ao publicar resultados de uma execução real, substitua a coluna "Resultado" pelos valores do seu MLflow/log e mantenha o link do experimento para dar credibilidade a recrutadores técnicos.

### 6.1 Preference judge (calibração e viés)

| Métrica | O que mede | Gate | Resultado (preencher com seu run) |
| --- | --- | :---: | :---: |
| Log loss | Calibração das probabilidades A/B/tie | menor é melhor | — |
| Macro F1 | Equilíbrio entre as 3 classes | > 0,75 | — |
| Accuracy | Leitura operacional simples | — | — |
| ECE (15 bins) | Confiabilidade probabilística | < 0,05 | — |
| Position flip rate | Estabilidade ao inverter A/B | < 0,10 | — |
| Length bias slope | Independência do tamanho da resposta | ≈ 0 (\|β\| < 0,05) | — |

### 6.2 Modelo de explicação (compliance)

| Métrica | O que mede | Gate | Resultado (preencher com seu run) |
| --- | --- | :---: | :---: |
| Reason code recall | Todos os fatores autorizados foram citados? | = 1,0 | — |
| Unsupported claim rate | Alegação sem suporte no contexto | = 0 | — |
| Forbidden attribute rate | Menção a atributo protegido | = 0 | — |
| JSON schema pass rate | Saída estruturada válida | = 1,0 | — |
| Preference win rate vs. baseline | Judge prefere o modelo ajustado? | > 0,55 | — |
| Human agreement (amostra auditada) | Concordância com revisor humano | > 0,75 | — |

### 6.3 Monitoramento contínuo

- **PSI (Population Stability Index)** em `monitoring/drift.py`, aplicado a score e a saídas do gerador:
  - `PSI < 0,10` → distribuição estável
  - `0,10 ≤ PSI ≤ 0,25` → alerta moderado / recalibração
  - `PSI > 0,25` → drift crítico, retraining imediato
- Taxa de falha de schema, comprimento médio de resposta, unsupported rate e latência — todos logados com `model_version`, `adapter_version` e `policy_version`.

---

## 7. Controles de risco (governança)

1. Score produzido por modelo separado — o LLM nunca decide crédito.
2. Prompt contém apenas reason codes aprovados (whitelist).
3. Atributos protegidos não entram no contexto de geração.
4. Resposta passa por validação determinística antes de sair pela API.
5. Toda inferência recebe `trace_id`, versão do modelo, versão da política e reason codes.
6. Monitoramento cobre drift de entrada, comprimento de resposta, falhas de schema e taxa de alegações não suportadas.

---

## 8. Estrutura do repositório

```text
CreditExplain_AI/
  configs/
    train.yaml
  data/
    raw/            # train.csv, test.csv, sample_submission.csv
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

---

## 9. Como rodar em 5 minutos

```bash
git clone https://github.com/faanogueira/CreditExplain_AI.git
cd CreditExplain_AI

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/audit_dataset.py
python scripts/build_preference_data.py
python scripts/build_credit_domain_data.py --rows 12000

python -m credit_explain.training.train_sft
python -m credit_explain.training.train_dpo
python -m credit_explain.training.train_judge

pytest -q
uvicorn credit_explain.api.main:app --host 0.0.0.0 --port 8000
# docs interativas: http://localhost:8000/docs
```

QLoRA requer GPU NVIDIA com CUDA. Auditoria e construção de dataset rodam em CPU.

---

## 10. Exemplo de uso da API

**Request** `POST /explain`:
```json
{
  "application_id": "APP-2026-89412",
  "debt_to_income": 0.48,
  "bureau_score": 605,
  "utilization": 0.78,
  "delinquencies_12m": 2,
  "employment_tenure_months": 8,
  "requested_amount": 25000.0
}
```

**Response** (contrato estrito, validado antes de sair):
```json
{
  "application_id": "APP-2026-89412",
  "risk": {
    "risk_score": 0.71,
    "risk_band": "elevated",
    "reason_codes": ["high_debt_to_income", "low_bureau_score", "high_credit_utilization"]
  },
  "explanation": {
    "summary": "O risco estimado pelo motor foi classificado como elevado.",
    "factors": [
      "A relação dívida/renda de 48% está acima da faixa de referência do modelo.",
      "O score de bureau informado ao modelo foi 605, sinal que contribuiu para o risco estimado.",
      "A utilização de crédito de 78% está elevada em relação ao limite disponível."
    ],
    "limitations": "Esta explicação descreve sinais do modelo e não constitui decisão autônoma de crédito."
  },
  "trace_id": "e4b98d1a-3c5e-4a6f-9817-68bcf82a0e21",
  "model_version": "qwen2.5-3b-credit-dpo-v1"
}
```

---

## 11. Limitações assumidas (transparência)

- O motor de risco atual é um **proxy determinístico** para fins de integração — próximo passo: plugar um XGBoost real com AUC/Gini/KS e calibração de PD.
- Os casos de domínio de crédito são **sintéticos, sem PII** — servem para provar governança e pipeline, não para uso em produção com dados reais sem validação adicional.
- Falta integrar SHAP / *adverse action notices* (ECOA/BACEN) e teste de fairness desagregado por subgrupo — está no roadmap.
- Pesos do adapter não publicados em `artifacts/`; reproduzível via `configs/train.yaml`.

---

## 12. O que torna este projeto tecnicamente relevante

Fine-tuning aqui não é uma etapa isolada: envolve formulação de objetivo, preparação de dado de preferência, QLoRA, SFT, DPO, tratamento de empates, split sem leakage, calibração, auditoria de viés de posição e verbosidade, validação determinística, serving, observabilidade e governança de modelo de risco.

**Decisão arquitetural central:** o LLM explica uma decisão produzida por um sistema de risco — e nunca substitui esse sistema.

---

## Autor

**Fábio Nogueira**
Cientista de Dados & Especialista em IA Aplicada, Risco e Analytics
[LinkedIn](https://www.linkedin.com/in/faanogueira) · [GitHub](https://github.com/faanogueira)
