"""
Scam Shield - Flask webapp.

The frontend posts ``scammer_message`` and (optionally) ``user_response`` and
``timestamp`` to ``/api/analyze``; the server runs the 3-step rule-based
analyst and returns the strict JSON shape described in the analyst spec.

Flask's default ``jsonify`` sorts keys alphabetically, which would break the
spec's required field order. We disable that globally and emit raw responses
from each endpoint.
"""

from __future__ import annotations

import json
from datetime import datetime

from flask import Flask, Response, render_template, request

from scam_shield.analyst import analyze as run_analyst


def _format_timestamp(value):
    """Normalize an inbound timestamp to ``MM/DD/YY/HH:MM:SS``.

    Accepts ``None``, an already-formatted string, or any ISO 8601 variant.
    Unparseable values pass through unchanged for caller debugging.
    """
    if value is None or value == "":
        return value
    if not isinstance(value, str):
        return value
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return value
    return dt.strftime("%m/%d/%y/%H:%M:%S")


def _json_response(payload, status=200):
    return Response(
        json.dumps(payload, separators=(", ", ": ")),
        status=status,
        mimetype="application/json",
    )


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.json.sort_keys = False

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        data = request.get_json(silent=True) or {}
        scammer_message = (data.get("scammer_message") or "").strip()
        user_response = data.get("user_response")
        timestamp = _format_timestamp(data.get("timestamp"))

        if not scammer_message:
            return _json_response({"error": "scammer_message is required"}, status=400)

        if isinstance(user_response, str) and not user_response.strip():
            user_response = None

        try:
            verdict = run_analyst(scammer_message, user_response, timestamp)
        except ValueError as exc:
            return _json_response({"error": str(exc)}, status=400)

        return _json_response(verdict)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)