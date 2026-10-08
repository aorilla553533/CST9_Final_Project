# Scam Shield - Detecting Deceptive Messages Using Random Forest Classification

A rule-based scam analyst that classifies any message a person suspects to be
fraudulent, scores its risk, recommends a strong response type, and judges
whether the user's planned reply succeeds.

This is a pure-Python implementation that runs offline with no external
dependencies beyond `flask` and `nltk`.

---

## What the system does

Given a `scammer_message` (and optionally a `user_response` and `timestamp`),
the analyst runs three deterministic steps:

1. **Analyze the scammer message** — assign a `scam_category` from
   `urgency`, `threat`, `fake_authority`, `fake_fee`,
   `emotional_manipulation`, `isolation`; compute a `risk_score` (4-10) and
   a `risk_level` (`medium`, `high`, `critical`).
2. **Recommend a strong response type** — always one of `refusing`,
   `verification`, `counter_attack`, `expert_mention`. Never recommends
   `delaying` or `questioning`.
3. **Judge the user's response** — if provided, return an
   `effectiveness_score` (3-5) and a `user_success` flag (1 if the reply
   is at least clearly resistant, otherwise 0).

Output is strict JSON with fixed field order:

```json
{
  "scammer_message": "...",
  "scam_category": "...",
  "risk_score": 0,
  "risk_level": "...",
  "recommended_response_type": "...",
  "user_response": "...",
  "effectiveness_score": 0,
  "user_success": 0,
  "timestamp": "..."
}
```

---

## Project layout

```
scam_shield/
  config.py     <- keyword vocabularies (urgency, authority, threat, fee)
  analyst.py    <- 3-step rule-based pipeline + JSON validation

webapp/
  app.py        <- Flask server with /api/analyze
  scenarios.py  <- 5 curated demo scams
  templates/index.html
  static/styles.css
  static/app.js

README.md
```

---

## Running it

### 1. Install dependencies

```powershell
py -m pip install flask nltk
py -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

### 2. Run the web app

```powershell
cd "C:\xampp\htdocs\CST9 ML1"
py -m webapp.app
```

Then open `http://127.0.0.1:5000` in any browser.

### 3. Use the analyst from a terminal

```python
from scam_shield.analyst import analyze
import json

result = analyze(
    scammer_message="Your bank account has been locked. Verify with the code we texted you.",
    user_response="I will call my bank branch directly using the number on the back of my card.",
    timestamp="10/08/26/10:00:00",
)
print(json.dumps(result, indent=2))
```

### 4. Use the HTTP API directly

```powershell
$body = '{"scammer_message":"Pay the $500 fee now.","user_response":"This is a scam."}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/analyze -ContentType "application/json" -Body $body
```

Returns the same strict JSON shape as the Python API.

---

## Validation

The analyst enforces the schema in code:

* `scam_category` ∈ the six allowed categories
* `risk_score` is an integer 4-10
* `risk_level` ∈ {medium, high, critical}
* `recommended_response_type` ∈ the four strong types
* `effectiveness_score` is 3, 4, or 5 (or null when no user_response)
* `user_success` is 0 or 1 (or null when no user_response)
* field order is exactly as specified

Any deviation raises `ValueError` and the Flask endpoint returns HTTP 400.

---

## Honest limitations

* The analyst is rule-based and depends on the vocabularies in
  `scam_shield/config.py`. Novel phrasings that don't match any keyword
  bucket will fall back to `urgency` and a moderate score.
* The `emotional_manipulation` and `isolation` categories are easy to confuse
  with one another; the precedence rule (money > threat > authority >
  fee > isolation > emotional > urgency) is documented in `analyst.py`.
* A user reply that is verbose but does not include any of the strong
  response markers (refusal, verification, expert mention, counter-attack)
  is judged as `questioning` or `delaying` and scored 3/0.

These caveats are noted for future iteration.