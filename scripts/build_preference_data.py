from pathlib import Path
import pandas as pd

from credit_explain.data.preference import deterministic_split, to_dpo_pairs, to_judge_rows

raw = pd.read_csv("data/raw/train.csv")
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

dpo = deterministic_split(to_dpo_pairs(raw, include_swap=True))
judge = deterministic_split(to_judge_rows(raw, include_swap=True))

dpo.to_csv(out / "general_dpo.csv.gz", index=False, compression="gzip")
judge.to_csv(out / "judge.csv.gz", index=False, compression="gzip")

print("general_dpo", dpo.shape, dpo["split"].value_counts().to_dict())
print("judge", judge.shape, judge["split"].value_counts().to_dict())
