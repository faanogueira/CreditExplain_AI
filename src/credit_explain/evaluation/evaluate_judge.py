import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from .metrics import expected_calibration_error, length_bias_slope, position_flip_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate exported preference judge predictions")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    probs = df[["prob_a", "prob_b", "prob_tie"]].to_numpy()
    y = df["label"].to_numpy(dtype=int)
    pred = probs.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "ece": expected_calibration_error(y, probs),
    }

    if {"response_a_len", "response_b_len"}.issubset(df.columns):
        metrics["length_bias_slope"] = length_bias_slope(
            df["response_a_len"] - df["response_b_len"], df["prob_a"]
        )

    if {"pair_id", "is_swap"}.issubset(df.columns):
        base = df[df.is_swap == 0].set_index("pair_id")
        swap = df[df.is_swap == 1].set_index("pair_id")
        common = base.index.intersection(swap.index)
        if len(common):
            metrics["position_flip_rate"] = position_flip_rate(
                base.loc[common, ["prob_a", "prob_b", "prob_tie"]].to_numpy().argmax(axis=1),
                swap.loc[common, ["prob_a", "prob_b", "prob_tie"]].to_numpy().argmax(axis=1),
            )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
