"""End-to-end test for the analyst webapp."""
import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webapp.app import create_app


def hit(url, *, method="GET", body=None, timeout=15.0):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def main():
    app = create_app()
    port = 5060
    t = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            hit(base + "/")
            break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.2)
    else:
        print("FAIL: server did not come up")
        sys.exit(1)
    print(f"server is up on {base}")

    r = hit(base + "/")
    body = r.read().decode("utf-8")
    assert r.status == 200
    assert "Detecting Deceptive Messages Using Random Forest Classification" in body
    assert 'id="live-form"' in body
    assert 'id="live-result"' in body
    assert "Sample Analyses" not in body
    print(f"OK homepage: {r.status}  {len(body)} bytes")

    for asset in ("/static/styles.css", "/static/app.js"):
        r = hit(base + asset)
        size = len(r.read())
        assert r.status == 200 and size > 500
        print(f"OK {asset}: {size} bytes")

    print("\n--- analyze endpoint checks ---")

    r = hit(
        base + "/api/analyze",
        method="POST",
        body={
            "scammer_message": "Officer Daniels from SSA. Warrant issued. Wire $4,800 to avoid arrest.",
            "user_response": "I will call the SSA main number to verify.",
            "timestamp": "2026-10-08T10:00:00Z",
        },
    )
    payload = json.loads(r.read())
    expected_keys = [
        "scammer_message", "scam_category", "risk_score", "risk_level",
        "recommended_response_type", "user_response", "effectiveness_score",
        "user_success", "timestamp",
    ]
    assert list(payload.keys()) == expected_keys, (
        f"keys out of order: {list(payload.keys())}"
    )
    assert payload["scam_category"] == "threat"
    assert payload["risk_level"] == "critical"
    assert payload["recommended_response_type"] == "verification"
    assert payload["effectiveness_score"] == 5
    assert payload["user_success"] == 1
    print(f"OK analyze: {payload['scam_category']} / {payload['risk_level']} / "
          f"recommend={payload['recommended_response_type']} / eff={payload['effectiveness_score']} / "
          f"success={payload['user_success']}")

    r = hit(base + "/api/analyze", method="POST", body={
        "scammer_message": "Congratulations! You won $1,000. Pay $9.99 via gift card.",
    })
    payload = json.loads(r.read())
    assert payload["user_response"] is None
    assert payload["effectiveness_score"] is None
    assert payload["user_success"] is None
    assert payload["scam_category"] == "fake_fee"
    print(f"OK analyze (no user_response): null fields, category={payload['scam_category']}")

    r = hit(base + "/api/analyze", method="POST", body={
        "scammer_message": "Pay $500 now.",
        "user_response": "Okay, send me the link.",
    })
    payload = json.loads(r.read())
    assert payload["user_success"] == 0, "compliant reply must not succeed"
    assert payload["effectiveness_score"] == 3
    print(f"OK analyze (compliance): success={payload['user_success']} eff={payload['effectiveness_score']}")

    try:
        r = hit(base + "/api/analyze", method="POST", body={
            "scammer_message": "   ",
        })
        print(f"FAIL: empty scammer_message should have been rejected, got {r.status}")
        sys.exit(1)
    except urllib.error.HTTPError as e:
        assert e.code == 400
        print("OK analyze rejects blank scammer_message with 400")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()