"""Infra's relay for the agent watch (INF-69, `agent_watch.sh` on the control pod). The pod holds no Slack credential, so every 5
minutes this reads the lines added to /data/agent-watch/events.jsonl over ssh and posts each in #agent-alerts: a held run the watch
cancelled and continued, one held again within the hour (left for a person), a tool call running 45 minutes, a crashed run. An
agent that holds a handle (the Slack registry's `holder`) is named by its handle. The pod frees held runs whether or not this runs;
it stops when infra's VM is suspended, and resumes from the offset it last posted.

usage (tmux agent-watch-relay, research importable, $SLACK_BOT_TOKEN set, $RESEARCH_SLACK_REGISTRY when not run from a checkout):
  python3 infra/control/agent_watch_relay.py [--once] [--dry-run]"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from research import slack
from research.pods import budgets, connect

EVENTS = "/data/agent-watch/events.jsonl"
OFFSET = Path.home() / "agent-watch/offset"
EVERY_S = 5 * 60
DRY = "--dry-run" in sys.argv
REG = slack.load_registry(slack.registry_path())
ALERTS = slack.channel_id_for(REG, "agent-alerts")
HOLDERS = {a: h["handle"] for h in REG.get("handles") or []
           for a in re.findall(r"bc-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", str(h.get("holder", "")))}
SAY = {"held": "was held after a finished tool call since {since} ({minutes} min); the watch {action}",
       "held_again": "was held again since {since} ({minutes} min), within an hour of the last restart: left for a person",
       "long_call": "has had a tool call running since {since} (45+ min); not cancelled",
       "crashed": "ended in ERROR"}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def text(rec: dict) -> str:
    who = f"@{HOLDERS[rec['agent']]}'s coordinator" if rec.get("agent") in HOLDERS else (rec.get("name") or "an agent")
    line = SAY.get(rec.get("kind"), "{kind}").format(**{"since": "?", "minutes": "?", "action": "?", **rec})
    return (f"*Agent watch:* {who} (<https://cursor.com/agents/{rec.get('agent')}|{rec.get('agent')}>, run {rec.get('run')}) {line}."
            + (f" New run {rec['new_run']}." if rec.get("new_run") else ""))


def pod(cmd: str) -> str:
    return subprocess.run([*connect.ssh_base(budgets.GUARD_HOST), cmd], capture_output=True, text=True, timeout=120, check=True).stdout


def once() -> None:
    done = int(OFFSET.read_text()) if OFFSET.exists() else 0
    lines = pod(f"tail -n +{done + 1} {EVENTS} 2>/dev/null || true").splitlines()
    for ln in lines:
        try:
            rec = json.loads(ln)
        except ValueError:
            rec = {"kind": "unreadable", "agent": "?", "run": "?"}
        if DRY:
            log(f"would post: {text(rec)}")
        else:
            slack._post(slack.Slack(os.environ.get("SLACK_BOT_TOKEN")), ALERTS, "infra", text(rec))
        done += 1
        OFFSET.parent.mkdir(parents=True, exist_ok=True)
        OFFSET.write_text(str(done))


def main() -> int:
    while True:
        try:
            once()
        except (subprocess.SubprocessError, OSError) as e:
            log(f"pass failed: {type(e).__name__}: {e}")
        if "--once" in sys.argv:
            return 0
        time.sleep(EVERY_S)


if __name__ == "__main__":
    sys.exit(main())
