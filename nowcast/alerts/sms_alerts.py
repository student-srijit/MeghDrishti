"""Twilio SMS alerts — last-mile notification for farmers/local
administration, which the problem statement explicitly calls out as a
requirement beyond "just a dashboard".

Implemented as a direct call to Twilio's REST API via `requests` (HTTP
Basic Auth with the Account SID/Auth Token, form-encoded body) rather than
the `twilio` PyPI SDK — every other real-data integration in this project
(RainViewer, Blitzortung, Tomorrow.io) already talks to its provider this
way with no extra dependency, and Twilio's Messages resource is a single
plain POST, so pulling in the full SDK for one endpoint isn't worth it.

Off unless `USE_LIVE_ALERTS=true` AND real Twilio credentials are set (see
.env.example) — like every other USE_LIVE_* source, a misconfigured or
failing send is logged and swallowed, never allowed to take down hazard
detection itself.
"""
import requests

from nowcast.configs.settings import (
    USE_LIVE_ALERTS,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_FROM_NUMBER,
    ALERT_TO_NUMBERS,
)

_TWILIO_MESSAGES_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


def configured():
    """True only if alerts are enabled AND every credential needed to
    actually send one is present — callers use this to decide whether to
    even attempt a send, so a half-configured .env fails loudly in logs
    once, not on every hazard cycle."""
    return bool(
        USE_LIVE_ALERTS
        and TWILIO_ACCOUNT_SID
        and TWILIO_AUTH_TOKEN
        and TWILIO_FROM_NUMBER
        and ALERT_TO_NUMBERS
    )


def send_sms(body: str):
    """Sends `body` to every number in ALERT_TO_NUMBERS. Returns the list of
    numbers it actually succeeded for — a partial failure (one bad number)
    doesn't block sending to the rest."""
    if not configured():
        raise RuntimeError(
            "Twilio alerts not configured — set USE_LIVE_ALERTS=true and "
            "TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_FROM_NUMBER/ALERT_TO_NUMBERS in .env"
        )
    url = _TWILIO_MESSAGES_URL.format(sid=TWILIO_ACCOUNT_SID)
    sent = []
    for to_number in ALERT_TO_NUMBERS:
        try:
            resp = requests.post(
                url,
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                data={"From": TWILIO_FROM_NUMBER, "To": to_number, "Body": body},
                timeout=10,
            )
            resp.raise_for_status()
            sent.append(to_number)
        except requests.RequestException as exc:
            print(f"[sms_alerts] failed to send to {to_number}: {exc}")
    return sent


def format_hazard_alert(district: str, state: str, hazard_type: str, severity: str, detail: str) -> str:
    """SMS body kept short (Twilio segments at 160 chars) and actionable:
    what, where, how severe, one concrete number — not a data dump."""
    return (
        f"MeghDrishti ALERT: {severity.upper()} {hazard_type} risk in {district}, {state}. "
        f"{detail} Take shelter precautions. This is an automated nowcast alert."
    )


if __name__ == "__main__":
    print(f"configured: {configured()}")
    if configured():
        sent = send_sms(format_hazard_alert("Pune", "Maharashtra", "hail", "high", "Reflectivity 82dBZ."))
        print(f"sent to: {sent}")
