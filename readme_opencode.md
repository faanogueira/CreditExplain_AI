# CreditExplain AI — LLM Alinhado para Explicações Auditáveis de Risco de Crédito

> **TL;DR para Tech Lead (30s):** Separei decisão de crédito (XGBoost/scorecard) da explicação (LLM). Fine-tuning em 2 estágios — SFT + DPO com QLoRA em 57.477 pares de preferência humana + 12.000 casos sintéticos de crédito com rubric determinística. Com validator, judge de 3 classes, calibração (ECE), auditoria de viés de posição/verbosidade e API auditável com `trace_id`. Stack: Qwen 2.5 3B, TRL, PEFT, FastAPI, MLflow.

**Fit ideal:** Cientista de Dados Sênior | Risk Analytics | Model Risk | Applied AI / LLM Engineer para bancos, fintechs e bureaus.

---

## 1. Por que isso importa para o negócio?

Explicar risco de crédito não é "gerar texto bonito". É atender 4 requisitos simultâneos:

1. **Fidelidade** aos reason codes reais do motor de risco
2. **Consistência** entre canais, atendentes e auditoria
3. **Zero alucinação** — sem inventar motivo, sem usar atributo protegido
4. **Rastreabilidade** — quem, quando, qual modelo, qual política

LLM base falha nos 4. Este projeto especializa o comportamento via preferência + trava com validação determinística.

**Regra de ouro do projeto:** `LLM explica, não decide. Score e decisão ficam fora do LLM.`

## 2. Arquitetura — o que eu desenhei

```text
Applicant features
  → Risk Engine (XGBoost / scorecard, fora do LLM)
  → score + reason codes + facts aprovados
  → Context Builder (minimização de dados)
  → LLM Explicador (Qwen 2.5 3B + QLoRA SFT + DPO)
  → Policy Validator (schema + factuality + forbidden attributes)
  → Preference Judge (A / B / tie para rerank e regressão)
  → API + audit trace
```

Separação clara de responsabilidades — padrão exigido por Model Risk Management.

## 3. Métricas que tech lead vai querer ver

### 3.1 Corpus de preferência geral (dado real auditado)

Origem: Kaggle LLM Classification Finetuning (Chatbot Arena). Arquivo `train.csv` auditado via `scripts/audit_dataset.py`:

| Métrica | Valor |
|---|---:|
| Linhas de treino | **57.477** |
| Modelos distintos (A + B) | **64** |
| Preferência A | 20.064 (**34,91%**) |
| Preferência B | 19.652 (**34,19%**) |
| Empate | 17.761 (**30,90%**) |
| Mediana de turnos / conversa | **1** |
| p99 de turnos | **5** |
| Duplicatas exatas (prompt+resp A+B) | **71** |
| Colunas | `id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie` |

> Licença CC BY-NC 4.0 → uso educacional/portfólio. Troca do corpus prevista para uso comercial.

### 3.2 Escala após processamento (`scripts/build_preference_data.py`)

Split determinístico **80/10/10 por hash do prompt normalizado** — impede leakage de conversas quase idênticas.

| Dataset derivado | Volume estimado | Fonte |
|---|---:|---|
| General DPO (chosen/rejected, sem ties) | **~39.716 pares → ~79.432 com swap A/B** | preferência humana |
| Preference Judge 3 classes (A/B/tie) | **~57.477 → ~114.954 com swap** | preserva empate |
| Credit SFT sintético | **12.000 prompts + respostas (configurável via `--rows`)** | política explícita |
| Credit DPO sintético | **12.000 pares chosen/rejected** | 3 variantes de violação controlada |

Variantes `rejected` no domínio: (0) certeza indevida + atributo vago, (1) fator inexistente + linguagem vaga, (2) recomendação prescritiva + atributo protegido.

### 3.3 Treino — eficiente e reproduzível (`configs/train.yaml`)

Base: `Qwen/Qwen2.5-3B-Instruct` + QLoRA NF4 4-bit, double quant, `bfloat16`.

| Estágio | Hiperparâmetros | Saída |
|---|---|---|
| **SFT domínio** | 1 epoch, lr 1e-4, batch 2 x 16 acum = **32 efetivo**, warmup 0.05 | `artifacts/credit_sft_adapter` |
| **DPO geral + domínio** | 1 epoch, lr 5e-5, **beta 0.1**, batch 1 x 32 = **32 efetivo**, max_len 2048 | `artifacts/credit_dpo_adapter` |
| **Judge 3 classes** | 1 epoch, lr 5e-5, **label smoothing 0.03** | classificador A/B/tie |
| LoRA | **r=32, alpha=64, dropout 0.05**, 7 módulos (`q/k/v/o/gate/up/down`) — **<2% parâmetros treináveis**, base congelada | adapter versionado |

Seed fixo `42`, `device_map=auto`, `gradient_checkpointing=True`. Roda em 1 GPU NVIDIA com CUDA. Audit + build de dados rodam em CPU.

### 3.4 Gates de avaliação — não subo adapter sem passar

**Judge (calibração e viés):**

| Métrica | O que mede | Gate |
|---|---|---|
| Accuracy / Macro-F1 / LogLoss | qualidade + equilíbrio entre A/B/tie | monitorar regressão |
| **ECE (15 bins)** | confiabilidade probabilística | < 0.05 |
| **Position flip rate** | estabilidade ao inverter A↔B | < 0.10 |
| **Length bias slope** | dependência do tamanho da resposta | ≈ 0 |

**Explicação (compliance):**

| Métrica | O que mede | Gate |
|---|---|---|
| Reason code recall | citou todos os fatores autorizados? | = 1.0 |
| Unsupported claim rate | inventou fato fora do contexto? | = 0 |
| Forbidden attribute rate | citou idade/sexo/raça/religião/estado civil? | = 0 |
| JSON schema pass rate | saída estruturada válida? | = 1.0 |
| Preference win rate vs baseline | judge prefere o ajustado? | > 0.55 |
| Human agreement (amostra auditada) | concordância com revisor | > 0.75 |

**Monitoramento contínuo (`monitoring/drift.py`):** PSI de score/facts, taxa de falha de schema, comprimento de resposta, unsupported rate, latência + `model_version / adapter_version / policy_version` em todo log.

### 3.5 Engenharia / qualidade

- **API FastAPI:** `POST /explain` com Pydantic (`debt_to_income, bureau_score, utilization, delinquencies_12m, tenure, amount`) → `{ risk, explanation, trace_id, model_version }`. Validação bloqueia resposta não-conforme.
- **Testes:** `pytest` — `test_preference.py` + `test_credit_domain.py` (pares chosen≠rejected, JSON válido com `summary+factors`).
- **Docs:** `DATA_CARD.md` + `MODEL_CARD.md` + `SYSTEM_DESIGN.md` + 5 notebooks numerados (audit → dataset → adaptação → QLoRA/DPO → eval/bias).
- **Observabilidade:** MLflow + prometheus-client prontos em `requirements.txt`.

## 4. Exemplo ponta a ponta

Entrada:
```json
{"application_id": "demo_1024", "debt_to_income": 0.47, "bureau_score": 612, "utilization": 0.81, "delinquencies_12m": 2, "employment_tenure_months": 9, "requested_amount": 15000}
```

Saída (contrato garantido por validator):
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

## 5. O que eu assumo como limitação (transparência que tech lead respeita)

- Motor de risco atual é proxy determinístico para integração — próximo passo: plugar XGBoost real com AUC/Gini/KS e calibração PD.
- Casos de crédito são sintéticos, sem PII — ideal para provar governança, não para produção.
- Falta ligar SHAP/adverse action notices (ECOA/BCB) e teste de fairness desagregado — roadmap já mapeado.
- Sem pesos publicados em `artifacts/` (reproduzível via `train.yaml`).

## 6. Como rodar em 5 min

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/audit_dataset.py
python scripts/build_preference_data.py
python scripts/build_credit_domain_data.py --rows 12000
python -m credit_explain.training.train_sft
python -m credit_explain.training.train_dpo
python -m credit_explain.training.train_judge
pytest -q
uvicorn credit_explain.api.main:app --port 8000
```

---

**Se você é recrutador/tech lead:** este projeto prova que eu entrego IA aplicada a crédito com rigor de risco, avaliação e MLOps — não só notebook. Vamos conversar sobre como adaptar essa camada explicativa ao seu motor de score e à sua política de compliance.
