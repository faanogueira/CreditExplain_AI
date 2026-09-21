import argparse
from pathlib import Path

from credit_explain.data.credit_domain import build_domain_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=12000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

cases, sft, dpo = build_domain_datasets(args.rows, args.seed)
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
cases.to_csv(out / "credit_cases.csv.gz", index=False, compression="gzip")
sft.to_csv(out / "credit_sft.csv.gz", index=False, compression="gzip")
dpo.to_csv(out / "credit_dpo.csv.gz", index=False, compression="gzip")
print("cases", cases.shape)
print("sft", sft.shape)
print("dpo", dpo.shape)
