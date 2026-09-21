# Data Card

## Dataset A. General preference corpus

Source: Kaggle LLM Classification Finetuning, derived from Chatbot Arena conversations.

Observed training shape: 57.477 rows and 9 columns.

Columns: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie.

Target semantics:

1. `winner_model_a = 1`: response A preferred.
2. `winner_model_b = 1`: response B preferred.
3. `winner_tie = 1`: no clear preference.

The raw files remain immutable under `data/raw`.

### License

The Kaggle data page reports CC BY NC 4.0. This repository therefore treats the included corpus as noncommercial research and portfolio data.

### Known risks

1. Position bias.
2. Verbosity bias.
3. Model identity leakage through stylistic signatures.
4. Prompt duplication and near duplication.
5. Domain mismatch with financial explanations.
6. Human preference is not equivalent to factual correctness.

### Mitigations

1. Swap augmentation for A and B ordering.
2. Grouped split using normalized prompt hash.
3. Explicit length bias audit.
4. Domain adaptation dataset with deterministic credit policy rubric.
5. Factuality validator independent of the LLM.

## Dataset B. Credit explanation adaptation corpus

This dataset is generated locally from synthetic, nonidentifying credit scenarios. It contains no real applicant data.

Variables are restricted to features useful for explanation experiments, such as debt to income, bureau score, utilization, delinquencies, employment tenure and requested amount.

Protected attributes are intentionally excluded.
