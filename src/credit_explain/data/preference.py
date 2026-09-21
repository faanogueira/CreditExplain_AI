import hashlib
import json
import re
from dataclasses import dataclass

import pandas as pd

LABELS = {"winner_model_a": 0, "winner_model_b": 1, "winner_tie": 2}


def normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value


def group_id(prompt: str) -> str:
    return hashlib.sha256(normalize_text(prompt).encode("utf-8")).hexdigest()[:16]


def decode_conversation(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list")
    return [str(x) for x in parsed]


def format_conversation(prompt: str, response: str) -> str:
    prompts = decode_conversation(prompt)
    responses = decode_conversation(response)
    turns = []
    for idx, user_text in enumerate(prompts):
        turns.append(f"User: {user_text}")
        if idx < len(responses):
            turns.append(f"Assistant: {responses[idx]}")
    return "\n".join(turns)


def target_from_row(row: pd.Series) -> int:
    values = [int(row["winner_model_a"]), int(row["winner_model_b"]), int(row["winner_tie"])]
    if sum(values) != 1:
        raise ValueError(f"Invalid one hot target: {values}")
    return values.index(1)


def to_dpo_pairs(df: pd.DataFrame, include_swap: bool = True) -> pd.DataFrame:
    records = []
    for row in df.itertuples(index=False):
        if row.winner_tie == 1:
            continue
        prompt_turns = decode_conversation(row.prompt)
        ra = decode_conversation(row.response_a)
        rb = decode_conversation(row.response_b)
        context = "\n".join(f"User: {x}" for x in prompt_turns)
        a = "\n".join(ra)
        b = "\n".join(rb)
        chosen, rejected = (a, b) if row.winner_model_a == 1 else (b, a)
        base = {
            "id": int(row.id),
            "group_id": group_id(row.prompt),
            "prompt": context,
            "chosen": chosen,
            "rejected": rejected,
            "source": "human_preference_general",
        }
        records.append(base)
        if include_swap:
            swapped = dict(base)
            swapped["id"] = f"{row.id}_swap"
            swapped["augmentation"] = "order_swap"
            records.append(swapped)
    return pd.DataFrame(records)


def to_judge_rows(df: pd.DataFrame, include_swap: bool = True) -> pd.DataFrame:
    out = []
    for row in df.itertuples(index=False):
        target = 0 if row.winner_model_a else 1 if row.winner_model_b else 2
        base = {
            "id": str(row.id),
            "group_id": group_id(row.prompt),
            "prompt": row.prompt,
            "response_a": row.response_a,
            "response_b": row.response_b,
            "label": target,
        }
        out.append(base)
        if include_swap:
            swapped = dict(base)
            swapped["id"] = f"{row.id}_swap"
            swapped["response_a"] = row.response_b
            swapped["response_b"] = row.response_a
            swapped["label"] = 1 if target == 0 else 0 if target == 1 else 2
            out.append(swapped)
    return pd.DataFrame(out)


def deterministic_split(df: pd.DataFrame) -> pd.DataFrame:
    def split(gid: str) -> str:
        bucket = int(gid[:8], 16) % 100
        if bucket < 80:
            return "train"
        if bucket < 90:
            return "validation"
        return "test"
    result = df.copy()
    result["split"] = result["group_id"].map(split)
    return result
