from credit_explain.data.credit_domain import derive_reason_codes, risk_band, risk_proxy


def score_application(features: dict) -> dict:
    score = risk_proxy(features)
    return {
        "risk_score": score,
        "risk_band": risk_band(score),
        "reason_codes": derive_reason_codes(features),
    }
