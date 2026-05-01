#!/usr/bin/env python3
"""Verify Sentry delivery end-to-end for HiCon.

Run from the ai_vision directory:
    SENTRY_DSN=<dsn> python3 scripts/verify_sentry.py

Expects: journal shows Sentry init + event_id confirmation line.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
from pathlib import Path
env = Path(__file__).parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

import sentry_config
sentry_config.init()

import sentry_sdk
event_id = sentry_sdk.capture_message(
    "HiCon Sentry verify: test event from verify_sentry.py",
    level="warning",
)
sentry_sdk.flush(timeout=5)
print(f"Sentry event sent: {event_id}")
print("Search Sentry for this event_id to confirm delivery.")
