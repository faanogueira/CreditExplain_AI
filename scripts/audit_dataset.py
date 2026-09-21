from pathlib import Path
import json
import pandas as pd

PATH = Path("data/raw/train.csv")

df = pd.read_csv(PATH)
labels = ["winner_model_a", "winner_model_b", "winner_tie"]
summary = {
    "rows": len(df),
    "columns": df.columns.tolist(),
    "nulls": df.isna().sum().to_dict(),
    "label_counts": df[labels].sum().to_dict(),
    "label_share": df[labels].mean().round(6).to_dict(),
    "unique_models": len(set(df.model_a).union(df.model_b)),
    "duplicate_ids": int(df.id.duplicated().sum()),
    "duplicate_pairs": int(df.duplicated(subset=["prompt", "response_a", "response_b"]).sum()),
}
turns = df.prompt.map(lambda x: len(json.loads(x)))
summary["turns"] = turns.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict()
print(json.dumps(summary, indent=2, ensure_ascii=False))
