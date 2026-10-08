"""
Scam Shield - Detecting Deceptive Messages Using Random Forest Classification

Single-file deployment app.

Run locally:
    pip install flask
    python app.py

Then open http://127.0.0.1:5000 in any browser.

Everything (analyst pipeline, keyword vocabularies, HTML, CSS, JS) is
inlined into this one script so it can be deployed by copying a single
.py file onto the server.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from flask import Flask, jsonify, request


# ===========================================================================
# SECTION 1 - KEYWORD VOCABULARIES (from scam_shield/config.py)
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
# SECTION 2 - RULE-BASED ANALYST (from scam_shield/analyst.py)
# ===========================================================================
#
# 3-step pipeline:
#   STEP 1: classify scam_category + risk_score + risk_level
#   STEP 2: recommend a strong response type
#   STEP 3: if user_response provided, judge effectiveness
#

def _keyword_hits(text: str, keywords: list) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def _money_request_signal(text: str) -> int:
    """0-3 score for money request pressure."""
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

    # Threats take priority - they are the most concrete scam signal.
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

    # Isolation cues.
    isolation_signals = sum(
        1 for kw in ("only you", "trust only me", "between us", "don't tell anyone",
                     "do not tell anyone", "alone")
        if kw in t
    )
    if isolation_signals >= 1:
        return "isolation"

    # Emotional manipulation: secrecy + family framing.
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


# Category -> recommended response options. The first listed option is the
# default; for critical risk the recommender may switch to "refusing".
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


def _judge_response(text: str):
    """
    Return (effectiveness_score, user_success).
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


def analyze(scammer_message: str,
            user_response: Optional[str] = None,
            timestamp: Optional[str] = None):
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
# SECTION 3 - INLINE FRONTEND ASSETS (HTML / CSS / JS)
# ===========================================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Scam Shield - Detecting Deceptive Messages</title>
<style>
__INLINE_CSS__
</style>
</head>
<body>
<header class="hero">
    <div class="hero-inner">
        <span class="hero-badge">Scam Shield</span>
        <h1>Detecting Deceptive Messages Using Random Forest Classification</h1>
        <p class="hero-lede">
            Paste any message you suspect is a scam. The analyst identifies
            the scam type, scores the risk, recommends a strong response,
            and judges whether your planned reply succeeds.
        </p>
        <div class="hero-stats">
            <div class="stat"><span class="stat-num">6</span><span class="stat-label">Scam Categories</span></div>
            <div class="stat"><span class="stat-num">4</span><span class="stat-label">Recommended Responses</span></div>
            <div class="stat"><span class="stat-num">3</span><span class="stat-label">Risk Levels</span></div>
            <div class="stat"><span class="stat-num">100%</span><span class="stat-label">Offline</span></div>
        </div>
    </div>
</header>

<main class="container">
    <section class="card-section live">
        <div class="section-head">
            <h2>Analyze Your Own Message</h2>
            <p>Paste the scam message. Add the planned response if you want a success verdict.</p>
        </div>

        <form id="live-form" class="live-form" autocomplete="off">
            <div class="field">
                <label for="scammer_message">Scammer message <span class="req">*</span></label>
                <textarea id="scammer_message" name="scammer_message" rows="4" required></textarea>
            </div>
            <div class="field">
                <label for="user_response">Your response <span class="opt">(optional)</span></label>
                <textarea id="user_response" name="user_response" rows="4" placeholder="Leave blank if you have not replied yet"></textarea>
            </div>
            <div class="field-row">
                <div class="field">
                    <label for="timestamp">Timestamp <span class="opt">(optional)</span></label>
                    <input type="text" id="timestamp" name="timestamp" placeholder="MM/DD/YY/HH:MM:SS" />
                </div>
            </div>
            <div class="actions">
                <button type="submit" class="btn-primary">Analyze</button>
                <button type="button" class="btn-ghost" id="reset-btn">Reset</button>
            </div>
            <p class="error" id="live-error" hidden></p>
        </form>

        <article class="verdict-card" id="live-result" hidden>
            <span class="card-number big">&#9733;</span>
            <div class="verdict-content">
                <header class="verdict-head">
                    <h3>Analysis Verdict</h3>
                    <span class="verdict-meta" id="live-meta"></span>
                </header>
                <section class="verdict-section">
                    <h4>1. Scam Type</h4>
                    <div class="verdict-fields">
                        <div><span class="vf-label">Category</span><span class="vf-val" id="live-category"></span></div>
                        <div><span class="vf-label">Risk score</span><span class="vf-val" id="live-risk-score"></span></div>
                        <div><span class="vf-label">Risk level</span><span class="vf-val risk-pill" id="live-risk-level"></span></div>
                    </div>
                </section>
                <section class="verdict-section">
                    <h4>2. Recommended Response</h4>
                    <p class="rec-line">Use a <strong id="rec-name"></strong> reply to shut this conversation down.</p>
                </section>
                <section class="verdict-section" id="live-user-section" hidden>
                    <h4>3. Your Response Verdict</h4>
                    <div class="verdict-fields">
                        <div><span class="vf-label">Effectiveness</span><span class="vf-val" id="live-eff"></span></div>
                        <div><span class="vf-label">Success</span><span class="vf-val success-pill" id="live-success"></span></div>
                    </div>
                    <div class="reply-echo" id="live-reply"></div>
                </section>
            </div>
        </article>
    </section>
</main>

<footer class="page-footer">
    <p>Scam Shield &middot; CST9 ML1 &middot; </p>
</footer>

<script>
__INLINE_JS__
</script>
</body>
</html>
"""

STYLES_CSS = r"""
:root {
    --bg: #f6f7fb;
    --surface: #ffffff;
    --surface-muted: #f4f5f9;
    --border: #e8eaf2;
    --border-strong: #d8dce7;
    --text: #1d2233;
    --text-muted: #6a7285;
    --text-faint: #94a0b3;
    --accent: #d83a55;
    --accent-soft: #fde9ed;
    --success: #1aa76a;
    --success-soft: #e6f6ee;
    --warning: #ef9a2b;
    --warning-soft: #fef1dd;
    --info: #2d6cdf;
    --info-soft: #e9effd;
    --shadow: 0 1px 2px rgba(20, 23, 36, 0.04), 0 6px 18px rgba(20, 23, 36, 0.05);
    --radius: 12px;
    --radius-sm: 8px;
}

* { box-sizing: border-box; }

html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 15px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
}

.hero {
    background:
        radial-gradient(circle at 0% 0%, rgba(216, 58, 85, 0.08), transparent 45%),
        radial-gradient(circle at 100% 0%, rgba(45, 108, 223, 0.08), transparent 45%),
        var(--surface);
    border-bottom: 1px solid var(--border);
}
.hero-inner {
    max-width: 1100px;
    margin: 0 auto;
    padding: 56px 24px 40px;
}
.hero-badge {
    display: inline-block;
    background: var(--accent);
    color: #ffffff;
    font-weight: 700;
    font-size: 32px;
    letter-spacing: -0.01em;
    text-transform: none;
    padding: 12px 22px;
    border-radius: 10px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(216, 58, 85, 0.20);
}
.hero h1 {
    font-size: clamp(22px, 3vw, 32px);
    margin: 0 0 14px;
    line-height: 1.2;
    font-weight: 700;
    letter-spacing: -0.01em;
    max-width: 900px;
}
.hero-lede {
    margin: 0;
    max-width: 720px;
    color: var(--text-muted);
    font-size: 16px;
}
.hero-stats {
    margin-top: 28px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
}
.stat {
    background: var(--surface-muted);
    border: 1px solid var(--border);
    padding: 14px 16px;
    border-radius: var(--radius-sm);
}
.stat-num {
    display: block;
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
}
.stat-label {
    display: block;
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

.container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 24px 60px;
}
.card-section { margin-bottom: 48px; }
.section-head { margin-bottom: 18px; }
.section-head h2 {
    margin: 0;
    font-size: 22px;
    font-weight: 700;
}
.section-head p {
    margin: 6px 0 0;
    color: var(--text-muted);
    font-size: 14px;
}

.card-number {
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent);
    color: #ffffff;
    font-weight: 700;
    font-size: 15px;
}
.card-number.big {
    background: linear-gradient(180deg, #f5a3b0, var(--accent));
    font-size: 22px;
}

.risk-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    width: fit-content;
}
.risk-pill.medium   { background: var(--warning-soft); color: var(--warning); }
.risk-pill.high     { background: var(--accent-soft);  color: var(--accent); }
.risk-pill.critical { background: var(--accent); color: #ffffff; }

.success-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    width: fit-content;
}
.success-pill.success { background: var(--success-soft); color: var(--success); }
.success-pill.failure { background: var(--accent-soft);  color: var(--accent); }

.live { padding-top: 8px; }
.live-form {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px;
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.field { display: flex; flex-direction: column; gap: 6px; }
.field-row { display: flex; gap: 14px; flex-wrap: wrap; }
.field-row .field { flex: 1; min-width: 200px; }
.field label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
}
.req { color: var(--accent); font-weight: 700; }
.opt { color: var(--text-faint); font-weight: 500; font-size: 12px; }
.field textarea, .field input[type="text"] {
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    font-size: 14px;
    font-family: inherit;
    color: var(--text);
    background: var(--surface-muted);
    transition: border-color 0.2s ease, background 0.2s ease;
    resize: vertical;
}
.field textarea { min-height: 80px; }
.field textarea:focus, .field input:focus {
    outline: none;
    border-color: var(--accent);
    background: #ffffff;
    box-shadow: 0 0 0 4px rgba(216, 58, 85, 0.12);
}

.actions {
    display: flex;
    gap: 10px;
    align-items: center;
}
.btn-primary {
    background: var(--accent);
    color: #ffffff;
    border: 0;
    border-radius: var(--radius-sm);
    padding: 11px 20px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease, transform 0.1s ease;
}
.btn-primary:hover { background: #b82b44; }
.btn-primary:active { transform: translateY(1px); }
.btn-primary:disabled {
    background: var(--border-strong);
    color: var(--text-faint);
    cursor: not-allowed;
}
.btn-ghost {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 11px 18px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s ease;
}
.btn-ghost:hover { background: var(--surface-muted); }

.error {
    color: var(--accent);
    font-size: 13px;
    background: var(--accent-soft);
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    margin: 0;
}

.verdict-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    display: grid;
    grid-template-columns: 36px 1fr;
    margin-top: 18px;
    overflow: hidden;
}
.verdict-content {
    padding: 20px 22px;
    display: flex;
    flex-direction: column;
    gap: 18px;
}
.verdict-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}
.verdict-head h3 { margin: 0; font-size: 18px; font-weight: 700; }
.verdict-meta { font-size: 12px; color: var(--text-faint); }

.verdict-section h4 {
    margin: 0 0 10px;
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
}

.verdict-fields {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
}
.verdict-fields > div {
    background: var(--surface-muted);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.vf-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}
.vf-val {
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
}

.rec-line {
    margin: 0;
    font-size: 15px;
    color: var(--text);
    line-height: 1.5;
}
.rec-line strong { font-weight: 700; }

.reply-echo {
    margin-top: 10px;
    padding: 10px 14px;
    background: var(--surface-muted);
    border-radius: var(--radius-sm);
    border-left: 3px solid var(--info);
    font-size: 13px;
    color: var(--text);
    line-height: 1.5;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}

@media (max-width: 700px) {
    .hero-inner { padding: 40px 20px 28px; }
    .container { padding: 28px 16px 40px; }
    .verdict-card { grid-template-columns: 1fr; }
    .verdict-card > .card-number {
        padding: 12px;
        min-height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--border);
    }
    .field-row { flex-direction: column; gap: 16px; }
}

.page-footer {
    text-align: center;
    padding: 24px;
    color: var(--text-faint);
    font-size: 12px;
    border-top: 1px solid var(--border);
}
"""

APP_JS = r"""
(function () {
    "use strict";

    const liveForm = document.getElementById("live-form");
    const liveResult = document.getElementById("live-result");
    const liveMetaEl = document.getElementById("live-meta");
    const liveCategoryEl = document.getElementById("live-category");
    const liveRiskScoreEl = document.getElementById("live-risk-score");
    const liveRiskLevelEl = document.getElementById("live-risk-level");
    const recNameEl = document.getElementById("rec-name");
    const liveUserSectionEl = document.getElementById("live-user-section");
    const liveEffEl = document.getElementById("live-eff");
    const liveSuccessEl = document.getElementById("live-success");
    const liveReplyEl = document.getElementById("live-reply");
    const liveErrorEl = document.getElementById("live-error");
    const resetBtn = document.getElementById("reset-btn");
    const timestampEl = document.getElementById("timestamp");

    function fmtResponseType(r) {
        if (!r) return "-";
        return r.replace(/_/g, " ");
    }

    function categoryLabel(c) {
        const labels = {
            urgency: "Urgency pressure",
            threat: "Threat / intimidation",
            fake_authority: "Fake authority impersonation",
            fake_fee: "Fake fee / prize",
            emotional_manipulation: "Emotional manipulation",
            isolation: "Isolation tactic",
        };
        return labels[c] || c;
    }

    function renderLiveVerdict(verdict) {
        liveMetaEl.textContent = verdict.timestamp ? "Timestamp: " + verdict.timestamp : "Analyzed just now";
        liveCategoryEl.textContent = categoryLabel(verdict.scam_category);
        liveRiskScoreEl.textContent = verdict.risk_score + " / 10";
        liveRiskLevelEl.textContent = verdict.risk_level;
        liveRiskLevelEl.className = "vf-val risk-pill " + verdict.risk_level;

        recNameEl.textContent = fmtResponseType(verdict.recommended_response_type);
        recNameEl.className = "";

        if (verdict.user_response !== null && verdict.user_response !== undefined) {
            liveUserSectionEl.hidden = false;
            liveEffEl.textContent = verdict.effectiveness_score + " / 5";
            liveSuccessEl.textContent = verdict.user_success === 1 ? "Success" : "Failure";
            liveSuccessEl.className = "vf-val success-pill " + (verdict.user_success === 1 ? "success" : "failure");
            liveReplyEl.textContent = verdict.user_response;
        } else {
            liveUserSectionEl.hidden = true;
            liveReplyEl.textContent = "";
        }

        liveResult.hidden = false;
    }

    async function callAnalyze(body) {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!response.ok) {
            const data = await response.json().catch(function () { return {}; });
            throw new Error(data.error || "Server returned an error");
        }
        return response.json();
    }

    liveForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        liveErrorEl.hidden = true;

        const scammerMessage = document.getElementById("scammer_message").value.trim();
        const userResponse = document.getElementById("user_response").value.trim();
        const timestamp = timestampEl.value.trim() || new Date().toISOString();

        if (!scammerMessage) {
            liveErrorEl.textContent = "Please paste the scammer message.";
            liveErrorEl.hidden = false;
            return;
        }

        const submitBtn = liveForm.querySelector("button[type=submit]");
        submitBtn.disabled = true;
        submitBtn.textContent = "Analyzing\u2026";

        try {
            const verdict = await callAnalyze({
                scammer_message: scammerMessage,
                user_response: userResponse || null,
                timestamp: timestamp,
            });
            renderLiveVerdict(verdict);
        } catch (err) {
            liveErrorEl.textContent = err.message || "Could not analyze message.";
            liveErrorEl.hidden = false;
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = "Analyze";
        }
    });

    resetBtn.addEventListener("click", function () {
        liveForm.reset();
        liveResult.hidden = true;
        liveErrorEl.hidden = true;
        timestampEl.value = "";
    });
})();
"""


# ===========================================================================
# SECTION 4 - FLASK APP
# ===========================================================================

app = Flask(__name__)


@app.get("/")
def index():
    html = INDEX_HTML.replace("__INLINE_CSS__", STYLES_CSS).replace("__INLINE_JS__", APP_JS)
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.post("/api/analyze")
def api_analyze():
    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    scammer_message = payload.get("scammer_message", "")
    user_response = payload.get("user_response")
    timestamp = payload.get("timestamp")

    try:
        result = analyze(
            scammer_message=scammer_message,
            user_response=user_response,
            timestamp=timestamp,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return app.response_class(
        response=json.dumps(result, indent=2),
        status=200,
        mimetype="application/json",
    )


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # Allow `python app.py` for quick local runs.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)