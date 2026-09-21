import json
from credit_explain.data.credit_domain import build_domain_datasets


def test_domain_data_has_preference_pairs():
    cases, sft, dpo = build_domain_datasets(20, seed=7)
    assert len(cases) == 20
    assert len(sft) == 20
    assert len(dpo) == 20
    assert all(dpo.chosen != dpo.rejected)


def test_good_output_is_json():
    _, sft, _ = build_domain_datasets(3, seed=7)
    for value in sft.response:
        payload = json.loads(value)
        assert "summary" in payload
        assert "factors" in payload
