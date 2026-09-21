# Model Card

## Model purpose

CreditExplain AI produces natural language explanations for reason codes emitted by a separate credit risk model.

It is not a credit decision model and must not be used to approve, deny, price or rank applicants.

## Training stages

1. Supervised fine tuning on domain explanation examples.
2. Direct Preference Optimization using human preference pairs for general alignment and domain preference pairs for credit explanation style.
3. Optional three class preference judge for offline evaluation and reranking.

## Base model

Default: Qwen/Qwen2.5-3B-Instruct.

## Parameter efficient tuning

QLoRA with NF4 quantization and LoRA adapters on attention and MLP projection layers.

## Intended inputs

Only model generated score metadata, reason codes and explicitly approved facts.

## Prohibited inputs

Protected attributes, free form demographic descriptions and information not used by the underlying risk model.

## Output constraints

1. Do not invent a reason code.
2. Do not infer protected attributes.
3. Do not promise approval or denial.
4. Distinguish model explanation from policy decision.
5. Include trace metadata at serving time.

## Evaluation gates

A candidate adapter should not be promoted unless it meets all configured thresholds in `evaluate_explanations.py`.
