"""Vercel's function for every HTTP route (vercel.json rewrites /api/slack/* and /api/cron/* here)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slack_events.app import create_app  # noqa: E402

app = create_app()
