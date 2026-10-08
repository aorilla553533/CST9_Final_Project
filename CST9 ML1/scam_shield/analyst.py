"""
Rule-based scam analyst.

Implements the 3-step pipeline from the analyst spec:

    STEP 1: Analyze the scammer message.
        scam_category    one of: urgency | threat | fake_authority |
                              fake_fee | emotional_manipulation | isolation
        risk_score       integer 4-10
        risk_level       medium | high | critical

    STEP 2: Recommend the best response type from the strong types:
        refusing | verification | counter_attack | expert_mention

    STEP 3: If user_response is provided, judge it.
        effectiveness_score    integer 3-5
        user_success           1 if effectiveness_score >= 4 else 0

Output schema is fixed (field order, types) and validated before returning.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from . import config


VALID_CATEGORIES = {
    "urgency",
    "threat",
    "fake_authority",
    "fake_fee",
    "emotional_manipulation",
    "isolation",
}

VALID_RESPONSE_TYPES = [
    "refusing",
    "verification",
    "counter_attack",
    "expert_mention",
]


# ---------------------------------------------------------------------------
# STEP 1 — analyze the scammer message
# ---------------------------------------------------------------------------
def _keyword_hits(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def _money_request_signal(text: str) -> int:
    """Returns a 0-3 score for money request pressure."""
    score = 0
    t = text.lower()
    if re.search(r"\$\s*\d|\busd\b|\bdollar|\beur\b|\bgbp\b", t):
        score += 2
    elif _keyword_hits(t, config.FEE_KEYWORDS) >= 1:
        score += 1
    if re.search(r"\bwire\b|gift\s*card|\bzelle\b|apple\s*pay|bitcoin|crypto", t):
        score += 1
    return min(score, 3)


def _classify_category(text: str) -> str:
    t = text.lower()

    urgency_count = _keyword_hits(t, config.URGENCY_KEYWORDS)
    threat_count = _keyword_hits(t, config.THREAT_KEYWORDS)
    authority_count = _keyword_hits(t, config.AUTHORITY_KEYWORDS)
    fee_count = _keyword_hits(t, config.FEE_KEYWORDS)
    money_signal = _money_request_signal(text)

    # Threats (arrest, legal action, jail) are the most concrete: take priority.
    if threat_count >= 1:
        return "threat"

    # Authority impersonation with money demand = fake_authority.
    if authority_count >= 1 and (money_signal >= 1 or fee_count >= 1):
        return "fake_authority"

    # Authority impersonation on its own.
    if authority_count >= 1:
        return "fake_authority"

    # Money request without threats or authority = fake_fee.
    if money_signal >= 2 or fee_count >= 2:
        return "fake_fee"

    # Isolation cues: "only you can help", "don't tell anyone"
    isolation_signals = sum(
        1 for kw in ("only you", "trust only me", "between us", "don't tell anyone",
                     "do not tell anyone", "alone")
        if kw in t
    )
    if isolation_signals >= 1:
        return "isolation"

    # Emotional manipulation: secrecy + family framing
    secrecy_signals = sum(
        1 for kw in ("secret", "don't tell", "do not tell", "family", "mom", "dad",
                     "husband", "wife", "love", "scared", "emergency")
        if kw in t
    )
    if secrecy_signals >= 2:
        return "emotional_manipulation"

    if urgency_count >= 1:
        return "urgency"

    return "urgency"


def _compute_risk_score(text: str, category: str) -> int:
    """Return 4-10. Higher for money requests, threats, impersonation."""
    score = 4
    money_signal = _money_request_signal(text)
    score += money_signal
    score += min(_keyword_hits(text, config.THREAT_KEYWORDS), 2)
    score += min(_keyword_hits(text, config.AUTHORITY_KEYWORDS), 2)
    score += min(_keyword_hits(text, config.URGENCY_KEYWORDS), 2)

    if category in {"threat", "fake_authority"}:
        score += 1
    if category == "emotional_manipulation":
        score += 1

    return max(4, min(10, score))


def _risk_level(score: int) -> str:
    if score >= 9:
        return "critical"
    if score >= 6:
        return "high"
    return "medium"


# ---------------------------------------------------------------------------
# STEP 2 — recommend a response type
# ---------------------------------------------------------------------------
_CATEGORY_TO_RECOMMENDED = {
    "urgency":                ["refusing", "verification"],
    "threat":                 ["verification", "expert_mention"],
    "fake_authority":         ["verification"],
    "fake_fee":               ["counter_attack", "expert_mention"],
    "emotional_manipulation": ["refusing"],
    "isolation":              ["verification"],
}


def _recommend_response(category: str, risk_level: str) -> str:
    options = _CATEGORY_TO_RECOMMENDED.get(category, ["verification"])
    if risk_level == "critical" and options[0] != "refusing":
        # Critical risk — prefer a response that ends the conversation.
        # "refusing" ends; the others stall. Pick first non-empty refusing if
        # possible, otherwise fall back to the first listed recommendation.
        if "refusing" in _CATEGORY_TO_RECOMMENDED[category]:
            return "refusing"
    return options[0]


# ---------------------------------------------------------------------------
# STEP 3 — judge the user's response
# ---------------------------------------------------------------------------
def _classify_response_type(text: str) -> str:
    """Internal classification. Never returned to the user."""
    t = text.lower()

    refusal_hits = sum(
        1 for kw in ("i refuse", "i will not", "i won't", "i wont",
                     "do not send", "stop contacting", "report it", "report you",
                     "this is a scam", "this is fraud", "i am reporting",
                     "i'm reporting", "blocked", "no thanks", "do not contact")
        if kw in t
    )
    if refusal_hits >= 1:
        return "refusing"

    verification_hits = sum(
        1 for kw in (
            "call my bank", "call the bank", "call my branch", "call the ssa",
            "call the irs", "call my credit", "call official",
            "official website", "official number", "official hotline",
            "verified directly", "verified with", "verify directly",
            "verify with", "confirm with", "logged in to", "logged into",
            "official site", "official app", "check directly", "check my account",
            "call to confirm", "call to verify", "directly with my",
            "main number", "through the official", "via the official",
            "real bank", "real irs", "real ssa", "real website",
        )
        if kw in t
    )
    if verification_hits >= 1:
        return "verification"

    expert_hits = sum(
        1 for kw in ("i work in", "i am a", "i'm a", "cybersecurity",
                     "fraud prevention", "fraud analyst", "i know this is",
                     "familiar with these", "this is a phishing",
                     "this is a scam pattern", "i recognize this")
        if kw in t
    )
    if expert_hits >= 1:
        return "expert_mention"

    counter_attack_hits = sum(
        1 for kw in ("reported to", "fbi", "ftc", "fcc", "carrier", "blocked you",
                     "reported this number", "reported your number",
                     "police", "authorities", "law enforcement")
        if kw in t
    )
    if counter_attack_hits >= 1:
        return "counter_attack"

    question_marks = t.count("?")
    if question_marks >= 1:
        return "questioning"

    delay_hits = sum(
        1 for kw in ("i'll think about it", "let me think", "i will think",
                     "maybe later", "not now", "i need time", "tomorrow",
                     "next week", "call back", "later")
        if kw in t
    )
    if delay_hits >= 1:
        return "delaying"

    return "questioning"


_BAD_RESPONSE_MARKERS = [
    "okay", "ok sure", "sure thing", "of course", "yes i will",
    "yes i'll", "i will send", "i'll send", "send the", "i am sending",
    "i'll pay", "i will pay", "what's the account", "what is the account",
    "send me the link", "send me the details", "here are the codes",
    "trust you", "gift card codes", "wire the money", "transfer the money",
    "i need the code", "share the code", "verify with the code",
    "don't tell mom", "don't tell dad", "please don't tell",
]

_DISQUALIFYING_PATTERNS = [
    re.compile(r"\b(okay|ok)\b.*\b(send|wire|transfer|pay)\b", re.IGNORECASE),
    re.compile(r"\b(sure|yes)\b.*?\b(send|wire|transfer|pay)\b", re.IGNORECASE),
]


def _looks_like_compliance(text: str) -> bool:
    t = text.lower()
    if any(marker in t for marker in _BAD_RESPONSE_MARKERS):
        return True
    for pattern in _DISQUALIFYING_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _judge_response(text: str) -> tuple[int, int]:
    """
    Return (effectiveness_score, user_success) per the spec.

    5 = firmly refuses or exposes the scam, or verifies independently
    4 = clearly resists or verifies, but less decisive
    3 = weak: stalls or doubts without refusing, verifying, or reporting
    """
    response_type = _classify_response_type(text)
    compliance = _looks_like_compliance(text)

    if compliance:
        return 3, 0

    if response_type == "refusing":
        if any(kw in text.lower() for kw in ("report", "fbi", "ftc", "police", "block")):
            return 5, 1
        return 5, 1
    if response_type == "verification":
        return 5, 1
    if response_type == "counter_attack":
        return 5, 1
    if response_type == "expert_mention":
        return 4, 1
    if response_type == "questioning":
        return 3, 0
    if response_type == "delaying":
        return 3, 0
    return 3, 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze(
    scammer_message: str,
    user_response: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """
    Run the 3-step pipeline and return the strict JSON shape.

    Raises ValueError if the inputs are unusable.
    """
    if not scammer_message or not scammer_message.strip():
        raise ValueError("scammer_message is required")

    category = _classify_category(scammer_message)
    risk_score = _compute_risk_score(scammer_message, category)
    risk_level = _risk_level(risk_score)
    recommended = _recommend_response(category, risk_level)

    if user_response is None or not user_response.strip():
        verdict_user_response = None
        effectiveness = None
        user_success = None
    else:
        verdict_user_response = user_response.strip()
        effectiveness, user_success = _judge_response(user_response.strip())

    payload = {
        "scammer_message": scammer_message,
        "scam_category": category,
        "risk_score": int(risk_score),
        "risk_level": risk_level,
        "recommended_response_type": recommended,
        "user_response": verdict_user_response,
        "effectiveness_score": effectiveness,
        "user_success": user_success,
        "timestamp": timestamp if timestamp else None,
    }

    _validate_payload(payload)
    return payload


def _validate_payload(payload: dict) -> None:
    """Strict shape + type validation."""
    expected_keys = [
        "scammer_message", "scam_category", "risk_score", "risk_level",
        "recommended_response_type", "user_response", "effectiveness_score",
        "user_success", "timestamp",
    ]
    if list(payload.keys()) != expected_keys:
        raise ValueError(f"payload keys out of order: {list(payload.keys())}")

    if payload["scam_category"] not in VALID_CATEGORIES:
        raise ValueError(f"invalid scam_category: {payload['scam_category']}")
    if payload["recommended_response_type"] not in VALID_RESPONSE_TYPES:
        raise ValueError(f"invalid recommended_response_type: {payload['recommended_response_type']}")
    if not isinstance(payload["risk_score"], int) or not (4 <= payload["risk_score"] <= 10):
        raise ValueError(f"risk_score out of range: {payload['risk_score']}")
    if payload["risk_level"] not in {"medium", "high", "critical"}:
        raise ValueError(f"invalid risk_level: {payload['risk_level']}")
    if payload["user_response"] is not None and not isinstance(payload["user_response"], str):
        raise ValueError("user_response must be string or null")
    if payload["effectiveness_score"] is not None and payload["effectiveness_score"] not in (3, 4, 5):
        raise ValueError(f"invalid effectiveness_score: {payload['effectiveness_score']}")
    if payload["user_success"] is not None and payload["user_success"] not in (0, 1):
        raise ValueError(f"invalid user_success: {payload['user_success']}")


if __name__ == "__main__":
    import sys

    scam = sys.argv[1] if len(sys.argv) > 1 else "Your bank account has been locked. Verify your identity now."
    resp = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(analyze(scam, resp), indent=2))