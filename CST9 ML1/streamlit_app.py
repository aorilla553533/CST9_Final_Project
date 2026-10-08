"""
Scam Shield - single-file Streamlit app.

Bundles the rule-based analyst (originally split across scam_shield/config.py
and scam_shield/analyst.py) into one self-contained module so it can be deployed
to Streamlit Community Cloud with no package-import gymnastics.

Run locally:
    py -m pip install streamlit
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

import streamlit as st


# ===========================================================================
# 1. CONFIG - keyword vocabularies (mirrors scam_shield/config.py)
# ===========================================================================

URGENCY_KEYWORDS = [
    "urgent", "immediately", "now", "asap", "right away", "today",
    "expires", "expire", "deadline", "limited time", "act now",
    "suspend", "suspended", "locked", "lock", "final notice",
]

AUTHORITY_KEYWORDS = [
    "bank", "irs", "government", "police", "fbi", "tax", "court",
    "lawyer", "official", "agent", "officer", "authority",
    "federal", "treasury", "support team",
]

THREAT_KEYWORDS = [
    "arrest", "lawsuit", "legal action", "fine", "penalty",
    "consequences", "shut down", "close your account", "freeze",
    "warrant", "prosecute", "jail", "prison",
]

FEE_KEYWORDS = [
    "fee", "payment", "pay", "$", "usd", "wire", "transfer",
    "gift card", "bitcoin", "crypto", "deposit", "tax", "fine",
]


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


# ===========================================================================
# 2. ANALYST - 3-step rule-based pipeline (mirrors scam_shield/analyst.py)
# ===========================================================================

def _keyword_hits(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def _money_request_signal(text: str) -> int:
    """Returns a 0-3 score for money request pressure."""
    score = 0
    t = text.lower()
    if re.search(r"\$\s*\d|\busd\b|\bdollar|\beur\b|\bgbp\b", t):
        score += 2
    elif _keyword_hits(t, FEE_KEYWORDS) >= 1:
        score += 1
    if re.search(r"\bwire\b|gift\s*card|\bzelle\b|apple\s*pay|bitcoin|crypto", t):
        score += 1
    return min(score, 3)


def _classify_category(text: str) -> str:
    t = text.lower()

    urgency_count = _keyword_hits(t, URGENCY_KEYWORDS)
    threat_count = _keyword_hits(t, THREAT_KEYWORDS)
    authority_count = _keyword_hits(t, AUTHORITY_KEYWORDS)
    fee_count = _keyword_hits(t, FEE_KEYWORDS)
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
    score += min(_keyword_hits(text, THREAT_KEYWORDS), 2)
    score += min(_keyword_hits(text, AUTHORITY_KEYWORDS), 2)
    score += min(_keyword_hits(text, URGENCY_KEYWORDS), 2)

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
        if "refusing" in _CATEGORY_TO_RECOMMENDED[category]:
            return "refusing"
    return options[0]


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


def analyze(
    scammer_message: str,
    user_response: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """Run the 3-step pipeline and return the strict JSON shape."""
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


# ===========================================================================
# 3. STREAMLIT UI
# ===========================================================================

DEMO_SCAMS = [
    {
        "label": "Bank lockout (fake authority + urgency)",
        "scammer_message": (
            "Your bank account has been locked due to suspicious activity. "
            "Verify your identity immediately by replying with the 6-digit code "
            "we just texted you, or your account will be permanently suspended."
        ),
        "user_response_strong": (
            "I will call my bank branch using the number on the back of my card "
            "to confirm this before doing anything."
        ),
        "user_response_weak": "Oh no, what should I do?",
    },
    {
        "label": "IRS arrest threat",
        "scammer_message": (
            "This is the IRS. A warrant has been issued for your arrest due to "
            "unpaid taxes. To avoid jail, pay the $4,500 fine via wire transfer today."
        ),
        "user_response_strong": "I am reporting this number to the FTC and police.",
        "user_response_weak": "Okay, how do I pay?",
    },
    {
        "label": "Grandparent emergency scam",
        "scammer_message": (
            "Grandma it's me. I'm in trouble, please don't tell mom or dad. "
            "I need you to send $2,000 in gift cards right now, it's an emergency."
        ),
        "user_response_strong": "I will not send anything. I am calling your mother.",
        "user_response_weak": "Oh honey, are you okay?",
    },
    {
        "label": "Fake delivery fee",
        "scammer_message": (
            "Your package could not be delivered. Pay the $3.00 redelivery fee "
            "within 24 hours or it will be destroyed."
        ),
        "user_response_strong": "I checked the official site directly, no redelivery fee is owed.",
        "user_response_weak": "Sure, what is the link?",
    },
    {
        "label": "Romance isolation",
        "scammer_message": (
            "My love, only you can help me. Please don't tell anyone, this is "
            "between us. I need $1,500 wired today, it's an emergency."
        ),
        "user_response_strong": "I will not send money. This is a scam pattern and I am blocking you.",
        "user_response_weak": "Okay, I will send the wire.",
    },
]


st.set_page_config(
    page_title="Scam Shield",
    page_icon="🛡️",
    layout="centered",
)

st.title("Scam Shield")
st.caption("Rule-based scam analyst — classifies a message, scores risk, recommends a response, and judges your reply.")

# ---- Sidebar: demo scenarios ----
with st.sidebar:
    st.header("Try a demo scam")
    demo_choice = st.radio(
        "Pick one to autofill:",
        options=["(none)"] + [d["label"] for d in DEMO_SCAMS],
        index=0,
        label_visibility="collapsed",
    )

# Prefill from demo if chosen
prefill_scammer = ""
prefill_response = ""
if demo_choice != "(none)":
    chosen = next(d for d in DEMO_SCAMS if d["label"] == demo_choice)
    prefill_scammer = chosen["scammer_message"]
    col1, col2 = st.columns(2)
    if col1.button("Use strong reply"):
        prefill_response = chosen["user_response_strong"]
    if col2.button("Use weak reply"):
        prefill_response = chosen["user_response_weak"]

# ---- Inputs ----
scammer_message = st.text_area(
    "Scammer message",
    value=prefill_scammer,
    height=140,
    placeholder="Paste the suspicious message here...",
)

user_response = st.text_area(
    "Your planned reply (optional)",
    value=prefill_response,
    height=100,
    placeholder="What would you say back? Leave blank to skip the judgment.",
)

run = st.button("Analyze", type="primary")

# ---- Output ----
if run:
    if not scammer_message.strip():
        st.error("Please paste a scammer message first.")
    else:
        try:
            result = analyze(
                scammer_message=scammer_message,
                user_response=user_response if user_response.strip() else None,
                timestamp=datetime.now().strftime("%m/%d/%y/%H:%M:%S"),
            )
        except ValueError as exc:
            st.error(f"Validation error: {exc}")
        else:
            # Verdict cards
            risk_color = {
                "medium": "🟡",
                "high": "🟠",
                "critical": "🔴",
            }[result["risk_level"]]

            st.subheader("Verdict")
            col1, col2, col3 = st.columns(3)
            col1.metric("Category", result["scam_category"].replace("_", " ").title())
            col2.metric("Risk", f'{risk_color} {result["risk_score"]} / 10')
            col3.metric("Risk level", result["risk_level"].title())

            st.markdown(
                f"**Recommended response:** "
                f"`{result['recommended_response_type'].replace('_', ' ')}`"
            )

            if result["user_response"]:
                eff = result["effectiveness_score"]
                success = result["user_success"]
                if success == 1:
                    st.success(
                        f"Your reply is effective (score {eff}/5) — "
                        f"you successfully resisted the scam."
                    )
                else:
                    st.warning(
                        f"Your reply is weak (score {eff}/5) — "
                        f"it neither refuses, verifies, nor reports. Try again."
                    )
            else:
                st.info("No user reply provided — judgment skipped.")

            with st.expander("Raw JSON output"):
                st.code(json.dumps(result, indent=2), language="json")


with st.expander("About this app"):
    st.markdown(
        """
        **Scam Shield** is a deterministic, rule-based analyst. It does **not**
        call any ML model at runtime — the classification comes from keyword
        vocabularies and precedence rules in this single file.

        **Pipeline (3 steps):**

        1. Classify the scammer message into one of six categories and score risk (4-10).
        2. Recommend one of four *strong* response types (`refusing`, `verification`,
           `counter_attack`, `expert_mention`).
        3. If you supplied a reply, judge it as effective (4-5) or weak (3).

        **Honest limitations:** the analyst depends on keyword matches. Novel
        phrasings that hit no bucket fall back to `urgency` and a moderate score.
        See the project README for the full precedence order.
        """
    )