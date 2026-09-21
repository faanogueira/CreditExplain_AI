import pandas as pd
from credit_explain.data.preference import deterministic_split, to_dpo_pairs, to_judge_rows


def sample():
    return pd.DataFrame([{
        "id": 1,
        "model_a": "a",
        "model_b": "b",
        "prompt": '["hello"]',
        "response_a": '["good"]',
        "response_b": '["bad"]',
        "winner_model_a": 1,
        "winner_model_b": 0,
        "winner_tie": 0,
    }])


def test_dpo_chosen():
    out = to_dpo_pairs(sample(), include_swap=False)
    assert out.iloc[0].chosen == "good"
    assert out.iloc[0].rejected == "bad"


def test_swap_label():
    out = to_judge_rows(sample(), include_swap=True)
    assert out.label.tolist() == [0, 1]


def test_split_stable():
    out1 = deterministic_split(to_judge_rows(sample(), include_swap=False))
    out2 = deterministic_split(to_judge_rows(sample(), include_swap=False))
    assert out1.split.tolist() == out2.split.tolist()
