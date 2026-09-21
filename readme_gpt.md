# CreditExplain AI

## Preference fine tuning para explicações auditáveis de risco de crédito

CreditExplain AI é uma solução de IA aplicada que especializa um LLM para explicar resultados de modelos de risco de crédito com fidelidade factual, consistência, rastreabilidade e controle de comportamento.

O princípio central é simples: **o LLM não decide crédito**. O score e os fatores de risco são produzidos por um motor independente. O LLM recebe apenas os sinais autorizados e converte esses sinais em uma explicação estruturada, clara e auditável.

Esse desenho separa decisão estatística de geração de linguagem e permite aplicar fine tuning sem transformar o modelo generativo em um componente decisório não determinístico.

## Executive snapshot

| Dimensão | Implementação |
| --- | --- |
| Caso de uso | Explicabilidade de risco de crédito |
| Modelo base | Qwen2.5 3B Instruct |
| Estratégia | SFT + DPO + Preference Judge |
| Eficiência | QLoRA 4 bit, NF4, LoRA adapters |
| Corpus geral | 57.477 comparações humanas |
| Modelos presentes no corpus | 64 |
| Credit SFT | 12.000 exemplos |
| Credit DPO | 12.000 pares chosen/rejected |
| General DPO | 79.432 pares após augmentation |
| Preference Judge | 114.954 exemplos de três classes |
| Testes automatizados | 5 de 5 aprovados |
| Serving | FastAPI |
| Observabilidade | métricas de qualidade, bias, drift e audit trace |

## Por que esse problema importa

Modelos de risco tradicionais podem gerar scores consistentes, mas a explicação desses scores para atendimento, risco, auditoria e áreas de negócio ainda costuma depender de regras rígidas, templates ou interpretação manual.

Um LLM generalista pode gerar explicações melhores linguisticamente, porém introduz novos riscos:

1. Alegações não suportadas pelos dados.
2. Uso de fatores que não participaram da decisão.
3. Linguagem excessivamente prescritiva.
4. Variação de resposta para casos equivalentes.
5. Position bias e verbosity bias em avaliações automáticas.
6. Falta de rastreabilidade entre score, reason codes e texto produzido.

CreditExplain AI trata esses problemas como requisitos de modelagem e avaliação, não apenas como prompt engineering.

## Arquitetura

```text
Applicant features
        |
        v
Credit Risk Engine
score + risk band + reason codes
        |
        v
Explanation Context Builder
somente fatos autorizados
        |
        v
Fine Tuned LLM
SFT + DPO com QLoRA
        |
        v
Policy Validator
schema + factuality + forbidden attributes
        |
        v
Preference Judge
A / B / tie
        |
        v
API Response
explanation + trace_id + model_version + policy_version
```

### Separação de responsabilidades

| Componente | Responsabilidade |
| --- | --- |
| Risk Engine | calcula score, faixa de risco e reason codes |
| Explanation Model | transforma fatores aprovados em linguagem natural |
| Policy Validator | bloqueia saídas fora do contrato de segurança |
| Preference Judge | compara respostas e mede preferência relativa |
| Monitoring Layer | acompanha drift, bias, qualidade e falhas de schema |

## Estratégia de fine tuning

### 1. Preference learning geral

O corpus original contém um prompt, duas respostas candidatas e uma preferência humana entre resposta A, resposta B ou empate.

Os pares inequívocos são convertidos para:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

Os empates são preservados para o Preference Judge de três classes.

Também é aplicado **order swap augmentation**, invertendo A e B sem alterar a semântica de preferência. Isso permite medir e reduzir position bias.

### 2. Supervised Fine Tuning no domínio de crédito

O SFT ensina o modelo a interpretar uma representação controlada do risco.

Exemplo de contexto:

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

O alvo de treinamento deve mencionar somente fatores presentes nesse contexto.

### 3. DPO específico de crédito

Cada cenário financeiro possui um par de respostas.

**Chosen** representa o comportamento desejado: factualidade, concisão, aderência aos reason codes, linguagem calibrada e ausência de atributos protegidos.

**Rejected** contém uma violação controlada: fator inventado, certeza excessiva, explicação vaga, recomendação indevida, causalidade não suportada ou verbosidade desnecessária.

Essa etapa otimiza diretamente o comportamento do modelo, em vez de depender somente de instruções no prompt.

### 4. Preference Judge independente

Um segundo adapter é treinado como classificador de três classes:

```text
A preferred
B preferred
Tie
```

O judge é usado para avaliação offline, comparação de checkpoints, análise de bias e potencial reranking.

## QLoRA

A configuração padrão utiliza `Qwen/Qwen2.5-3B-Instruct` com quantização em 4 bits.

```yaml
load_in_4bit: true
quant_type: nf4
double_quant: true
compute_dtype: bfloat16
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
```

Os adapters são aplicados às projeções de atenção e MLP:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Essa escolha reduz custo de memória e permite especialização do comportamento sem atualizar todos os parâmetros do modelo base.

## Métricas já medidas

### Corpus de preferência

| Métrica | Resultado |
| --- | ---: |
| Registros de treino | 57.477 |
| Modelos distintos | 64 |
| Preferência A | 20.064, 34,91% |
| Preferência B | 19.652, 34,19% |
| Empates | 17.761, 30,90% |
| Duplicatas exatas de pares | 71 |
| Máximo de turnos observado | 36 |

A distribuição das três classes é suficientemente equilibrada para tornar `Macro F1`, `Log Loss` e calibração métricas mais informativas do que accuracy isolada.

### Datasets derivados

| Dataset | Volume |
| --- | ---: |
| General DPO | 79.432 pares |
| Preference Judge | 114.954 exemplos |
| Credit cases | 12.000 casos |
| Credit SFT | 12.000 exemplos |
| Credit DPO | 12.000 pares |

### Qualidade de software

```text
5 passed in 0.05s
```

| Verificação | Status |
| --- | --- |
| Testes de geração do domínio | aprovado |
| Testes de transformação de preferência | aprovado |
| Validação de contratos básicos | aprovado |
| Smoke test do fluxo de inferência | aprovado |

## Métricas do modelo

O projeto foi desenhado para impedir uma prática comum em portfólio de IA: apresentar números de modelos que não foram efetivamente treinados.

As métricas abaixo fazem parte do pipeline de avaliação, mas só devem ser preenchidas como resultados após execução do fine tuning em GPU.

### Preference Judge

| Métrica | Objetivo |
| --- | --- |
| Log Loss | medir qualidade probabilística |
| Macro F1 | avaliar desempenho equilibrado em A, B e tie |
| Accuracy | leitura operacional |
| Expected Calibration Error | medir calibração das probabilidades |
| Position Flip Rate | detectar dependência da ordem A/B |
| Length Bias Slope | medir preferência indevida por respostas maiores |

### Explanation Model

| Métrica | Objetivo |
| --- | --- |
| Reason Code Recall | verificar cobertura dos fatores autorizados |
| Unsupported Claim Rate | medir alucinação factual |
| Forbidden Attribute Rate | detectar atributos não permitidos |
| JSON Schema Pass Rate | validar conformidade estrutural |
| Preference Win Rate | comparar fine tuned versus baseline |
| Pairwise Human Agreement | medir alinhamento com avaliação humana |

## Production quality gates

Os números abaixo são **critérios de aceitação propostos**, não resultados simulados.

| Gate | Critério |
| --- | ---: |
| Unsupported Claim Rate | <= 1% |
| Forbidden Attribute Rate | 0% |
| JSON Schema Pass Rate | >= 99,5% |
| Reason Code Recall | >= 98% |
| Position Flip Rate | <= 3% |
| Preference Win Rate vs baseline | >= 60% |
| ECE do judge | <= 0,05 |

Em um deployment real, qualquer checkpoint que não atinja os gates de segurança e factualidade não deve ser promovido, mesmo que apresente melhor loss ou preferência média.

## Avaliação contra baseline

A comparação recomendada é:

```text
Base model
   versus
Prompt engineered baseline
   versus
SFT
   versus
SFT + DPO
   versus
SFT + DPO + deterministic validator
```

O ganho relevante não é apenas qualidade textual. O objetivo é demonstrar melhoria simultânea em:

1. Fidelidade aos reason codes.
2. Redução de unsupported claims.
3. Consistência estrutural.
4. Preferência humana.
5. Calibração.
6. Estabilidade contra position bias.
7. Custo e latência de inferência.

## Bias audit

A avaliação inclui dois vieses particularmente importantes em sistemas que usam LLM como judge.

### Position bias

A mesma comparação é executada novamente invertendo a posição das respostas.

```text
A, B
B, A
```

Se a preferência semântica muda apenas por causa da posição, o judge apresenta instabilidade.

A métrica implementada é `Position Flip Rate`.

### Verbosity bias

A diferença de comprimento entre A e B é relacionada à probabilidade de preferência.

Uma inclinação significativa indica que o judge pode estar confundindo respostas maiores com respostas melhores.

A métrica implementada é `Length Bias Slope`.

## Calibração

O projeto calcula `Expected Calibration Error` para avaliar se confiança e acurácia estão alinhadas.

Para um sistema de decisão assistida, uma probabilidade de 0,90 deve representar aproximadamente 90% de acerto em exemplos comparáveis.

Isso permite criar políticas de abstention, por exemplo:

```text
confidence >= 0.80
    aceitar avaliação automática

confidence < 0.80
    encaminhar para revisão
```

## MLOps e governança

Cada inferência deve possuir metadados suficientes para reconstrução posterior.

```json
{
  "trace_id": "...",
  "risk_model_version": "...",
  "llm_adapter_version": "...",
  "policy_version": "...",
  "risk_score": 0.63,
  "reason_codes": ["..."],
  "validation_status": "passed"
}
```

O pipeline prevê monitoramento de:

1. Population Stability Index para variáveis de entrada.
2. Distribuição dos reason codes.
3. Comprimento das respostas.
4. Taxa de falha de schema.
5. Unsupported Claim Rate.
6. Forbidden Attribute Rate.
7. Mudança de Preference Win Rate entre versões.
8. Calibração do judge.

## Decisões técnicas relevantes

### Por que não usar apenas RAG

RAG é adequado para recuperar políticas, regras e documentação atualizada. Porém, neste caso o problema principal também é **comportamental**: como responder, quais sinais priorizar, como evitar causalidade indevida e como manter consistência de linguagem.

Por isso, RAG e fine tuning são complementares.

```text
RAG
conhecimento atualizado

Fine tuning
comportamento e preferência
```

### Por que DPO após SFT

SFT ensina a distribuição desejada de respostas. DPO adiciona sinal comparativo explícito e permite aprender diferenças sutis entre uma resposta aceitável e uma resposta preferível.

### Por que um judge separado

Usar o mesmo modelo para gerar e avaliar sua própria resposta aumenta risco de self preference. O projeto separa geração e avaliação para reduzir esse acoplamento.

### Por que não deixar o LLM decidir crédito

A decisão deve permanecer em um componente validável, versionado e apropriado ao processo de risco. O LLM recebe apenas o resultado e os fatores autorizados. Isso reduz superfície de risco e torna a explicabilidade verificável.

## Estrutura do projeto

```text
CreditExplain_AI/
  configs/
    train.yaml
  data/
    raw/
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
  scripts/
    audit_dataset.py
    build_preference_data.py
    build_credit_domain_data.py
  src/credit_explain/
    api/
    data/
    evaluation/
    monitoring/
    risk/
    training/
  tests/
```

## Pipeline de execução

```bash
python scripts/audit_dataset.py
python scripts/build_preference_data.py
python scripts/build_credit_domain_data.py --rows 12000

python -m credit_explain.training.train_sft
python -m credit_explain.training.train_dpo
python -m credit_explain.training.train_judge

python -m credit_explain.evaluation.evaluate_judge
python -m credit_explain.evaluation.evaluate_explanations
```

## O que este projeto demonstra

Para uma avaliação técnica, CreditExplain AI demonstra competências em quatro níveis.

### Data Science

Experiment design, construção de baseline, métricas probabilísticas, calibração, análise de bias, criação de datasets de preferência e definição de critérios de aceitação.

### LLM Engineering

SFT, DPO, QLoRA, PEFT, quantização 4 bit, adapters, preference modeling, reward signals, structured generation e evaluation harness.

### Machine Learning Engineering

Separação entre treinamento e serving, versionamento, testes, API, contratos de inferência, observabilidade, drift e promoção de checkpoints baseada em métricas.

### Responsible AI

Separação entre decisão e explicação, proteção contra atributos não permitidos, factualidade, traceability, abstention e auditoria de vieses do próprio sistema avaliador.

## Perguntas que este projeto permite discutir em uma entrevista técnica

1. Quando DPO é superior a apenas adicionar mais exemplos de SFT?
2. Como detectar position bias em um LLM judge?
3. Como impedir leakage entre pares de preferência no split?
4. Quando RAG deve complementar fine tuning?
5. Como medir se uma explicação é factual sem depender apenas de outro LLM?
6. Como definir um threshold de abstention usando calibração?
7. Como promover um adapter para produção sem olhar apenas para training loss?
8. Como separar drift do modelo de risco de drift do modelo explicativo?
9. Como avaliar ganho de um fine tuned model contra um prompt engineered baseline?
10. Como controlar custo, latência e memória ao escalar de 3B para 7B ou 8B?

## Status de execução

A preparação de dados, geração dos datasets derivados, testes automatizados e smoke tests foram executados.

O treinamento completo do Qwen2.5 3B exige ambiente com GPU CUDA e não foi executado neste runtime. Por esse motivo, nenhuma métrica de performance do modelo final é apresentada como se tivesse sido medida.

Essa distinção é intencional e faz parte da qualidade experimental do projeto.

## Próxima etapa recomendada

Executar SFT e DPO em GPU, registrar os experimentos com MLflow e preencher a tabela final abaixo.

| Modelo | Macro F1 Judge | ECE | Position Flip | Unsupported Claims | Reason Recall | Preference Win Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | TBD | TBD | TBD | TBD | TBD | baseline |
| SFT | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT + DPO | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT + DPO + Validator | TBD | TBD | TBD | TBD | TBD | TBD |

## Stack

`Python` · `PyTorch` · `Transformers` · `TRL` · `PEFT` · `bitsandbytes` · `QLoRA` · `DPO` · `FastAPI` · `Pydantic` · `scikit-learn` · `pytest`

## Nota de uso

O corpus original possui restrição de licença não comercial. A arquitetura, código, metodologia e pipeline podem ser reutilizados, mas um deployment comercial deve empregar dados com licença compatível e passar por validação jurídica, de risco e de governança apropriada ao contexto da instituição.

---

**CreditExplain AI foi desenhado como um sistema de IA aplicado a um problema real de produção: transformar sinais de risco em explicações consistentes sem delegar ao LLM a decisão que ele não deve tomar.**
