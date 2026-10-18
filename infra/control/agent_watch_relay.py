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
SAY = {"held": "was held after a finished tool call or the end of its turn since {since} ({minutes} min); the watch {action}",
       "held_again": "was held again since {since} ({minutes} min), within an hour of the last restart: left for a person",
       "continued": "got the continue run the API had refused after the watch cancelled its held run",
       "continue_failed": "had its held run cancelled at {since}, and the API refused the continue run for 30 min: left for "
                          "a person (send it a message)",
       "long_call": "has had a tool call running since {since} (45+ min); not cancelled",
       "crashed": "ended in ERROR; the watch {action}",
       "erroring": "ended in ERROR again after {restarts} restarts in a row: left for a person (relaunch it, or report it to Cursor)",
       "stuck": "has a run in {status} since {since} ({minutes} min), nothing streamed or started behind it; not cancelled"}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def text(rec: dict) -> str:
    who = f"@{HOLDERS[rec['agent']]}'s coordinator" if rec.get("agent") in HOLDERS else (rec.get("name") or "an agent")
    line = SAY.get(rec.get("kind"), "{kind}").format(**{"since": "?", "minutes": "?", "action": "recorded it", "restarts": "?",
                                                         **rec})
    return (f"*Agent watch:* {who} (<https://cursor.com/agents/{rec.get('agent')}|{rec.get('agent')}>, run {rec.get('run')}) {line}."
            + (f" New run {rec['new_run']}." if rec.get("new_run") else ""))


def pod(cmd: str) -> str:
    return subprocess.run([*connect.ssh_base(budgets.GUARD_HOST), cmd], capture_output=True, text=True, timeout=120, check=True).stdout


def once() -> None:
    done = int(OFFSET.read_text()) if OFFSET.exists() else 0
    *lines, _ = pod(f"cat {EVENTS} 2>/dev/null || true").split("\n")   # a line still being written waits a pass
    if len(lines) < done:
        log(f"{EVENTS} has {len(lines)} lines, fewer than the {done} posted: replaced, so posting it from the top")
        done = 0
    for ln in lines[done:]:
        try:
            rec = json.loads(ln)
        except ValueError:
            rec = {"kind": "garbled", "agent": "?", "run": "?"}
        if rec.get("kind") == "unreadable":         # a run the API won't read (a goal run's id): nothing to do
            pass
        elif DRY:
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
        except (subprocess.SubprocessError, OSError, slack.SlackError) as e:   # a refused post is posted again next pass
            log(f"pass failed: {type(e).__name__}: {e}")
        if "--once" in sys.argv:
            return 0
        time.sleep(EVERY_S)


if __name__ == "__main__":
    sys.exit(main())
