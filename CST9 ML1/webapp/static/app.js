/* Scam Shield - frontend logic
 * Handles the interactive form on the page and renders the analyst verdict.
 */

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