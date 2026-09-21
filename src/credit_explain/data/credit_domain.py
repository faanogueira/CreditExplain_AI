from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class CreditCase:
    case_id: str
    risk_score: float
    risk_band: str
    debt_to_income: float
    bureau_score: int
    utilization: float
    delinquencies_12m: int
    employment_tenure_months: int
    requested_amount: float
    reason_codes: list[str]


def derive_reason_codes(case: dict) -> list[str]:
    reasons = []
    if case["debt_to_income"] >= 0.45:
        reasons.append("high_debt_to_income")
    if case["bureau_score"] < 620:
        reasons.append("low_bureau_score")
    if case["utilization"] >= 0.75:
        reasons.append("high_credit_utilization")
    if case["delinquencies_12m"] >= 2:
        reasons.append("recent_delinquencies")
    if case["employment_tenure_months"] < 12:
        reasons.append("short_employment_tenure")
    if not reasons:
        reasons.append("limited_adverse_signals")
    return reasons[:3]


def risk_proxy(case: dict) -> float:
    components = [
        0.28 * min(case["debt_to_income"] / 0.60, 1.5),
        0.22 * max((700 - case["bureau_score"]) / 180, 0),
        0.20 * min(case["utilization"] / 0.90, 1.4),
        0.20 * min(case["delinquencies_12m"] / 3, 1.2),
        0.10 * max((24 - case["employment_tenure_months"]) / 24, 0),
    ]
    return float(np.clip(sum(components), 0, 1))


def risk_band(score: float) -> str:
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "moderate"
    if score < 0.75:
        return "elevated"
    return "high"


def render_factor(code: str, case: dict) -> str:
    mapping = {
        "high_debt_to_income": f"A relação dívida renda de {case['debt_to_income']:.0%} está acima da faixa de referência do modelo.",
        "low_bureau_score": f"O score de bureau informado ao modelo foi {case['bureau_score']}, sinal que contribuiu para o risco estimado.",
        "high_credit_utilization": f"A utilização de crédito de {case['utilization']:.0%} está elevada em relação ao limite disponível.",
        "recent_delinquencies": f"Foram registrados {case['delinquencies_12m']} atrasos nos últimos 12 meses no conjunto de dados fornecido ao modelo.",
        "short_employment_tenure": f"O tempo de vínculo informado foi {case['employment_tenure_months']} meses, fator considerado pelo modelo de demonstração.",
        "limited_adverse_signals": "Os fatores avaliados apresentaram poucos sinais adversos relevantes no modelo de demonstração.",
    }
    return mapping[code]


def good_explanation(case: dict) -> str:
    factors = [render_factor(code, case) for code in case["reason_codes"]]
    payload = {
        "summary": f"O risco estimado pelo motor foi classificado como {case['risk_band']}.",
        "factors": factors,
        "limitations": "Esta explicação descreve sinais do modelo e não constitui decisão autônoma de crédito.",
    }
    return json.dumps(payload, ensure_ascii=False)


def bad_explanation(case: dict, variant: int) -> str:
    if variant == 0:
        return json.dumps({
            "summary": "O cliente certamente não pagará a operação.",
            "factors": ["O perfil pessoal demonstra baixa confiabilidade."],
            "limitations": "Nenhuma."}, ensure_ascii=False)
    if variant == 1:
        return json.dumps({
            "summary": f"Risco {case['risk_band']}.",
            "factors": ["A renda é insuficiente e houve várias consultas recentes."],
            "limitations": "Conclusão definitiva."}, ensure_ascii=False)
    return json.dumps({
        "summary": "A análise identificou vários fatores e recomenda reprovação imediata.",
        "factors": [render_factor(code, case) for code in case["reason_codes"]] + [
            "A idade e o perfil familiar também aumentam o risco."
        ],
        "limitations": "A recomendação deve ser seguida automaticamente."}, ensure_ascii=False)


def generate_cases(rows: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(rows):
        raw = {
            "case_id": f"credit_{i:06d}",
            "debt_to_income": float(rng.beta(2.2, 3.2) * 0.85),
            "bureau_score": int(np.clip(rng.normal(655, 70), 300, 850)),
            "utilization": float(rng.beta(2.0, 2.5)),
            "delinquencies_12m": int(np.clip(rng.poisson(0.65), 0, 6)),
            "employment_tenure_months": int(np.clip(rng.gamma(2.3, 20), 1, 240)),
            "requested_amount": float(np.round(rng.lognormal(9.3, 0.65), 2)),
        }
        score = risk_proxy(raw)
        raw["risk_score"] = score
        raw["risk_band"] = risk_band(score)
        raw["reason_codes"] = derive_reason_codes(raw)
        records.append(raw)
    return pd.DataFrame(records)


def build_domain_datasets(rows: int = 12000, seed: int = 42):
    cases = generate_cases(rows, seed)
    sft, dpo = [], []
    for idx, case in cases.iterrows():
        payload = case.to_dict()
        prompt = (
            "Explique o resultado do motor de risco usando somente os fatos e reason codes fornecidos. "
            "Não tome uma decisão de crédito. Responda em JSON.\nContexto: " +
            json.dumps(payload, ensure_ascii=False)
        )
        good = good_explanation(payload)
        bad = bad_explanation(payload, idx % 3)
        sft.append({"case_id": payload["case_id"], "prompt": prompt, "response": good})
        dpo.append({
            "case_id": payload["case_id"],
            "prompt": prompt,
            "chosen": good,
            "rejected": bad,
            "source": "credit_domain_policy_preference",
        })
    return cases, pd.DataFrame(sft), pd.DataFrame(dpo)
