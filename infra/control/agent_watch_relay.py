"""Infra's relay for the agent watch (INF-69, `agent_watch.sh` on the control pod). The pod holds no Slack credential, so every 5
minutes this reads the lines added to /data/agent-watch/events.jsonl over ssh and posts each in #agent-alerts: a held run the watch
cancelled and continued, a tool call running 45 minutes, a crashed run, a coordinator ready to wake. An agent that holds a handle
(the Slack registry's `holder`) is named by its handle. What the watch leaves for a person (`TOP`: erroring, stuck, a refused
cancel, held again within the hour, and a run unreadable for an hour; the first unreadable line is posted nowhere) leads with
@top, and rings @top's inbox thread as `ring_decision` does for a decided card (the inbox is in the front door, so the post's
own route can't ring it). A cloud worker (a Task subagent) that went idle, or has been ACTIVE for hours, leads with its lane's
handle and rings that inbox the same way, for its coordinator to resume it with Task (the watch can't start a worker's run;
the doorbell also counts toward the coordinator's own ready-to-wake rule). A silent agent (ACTIVE 90 min with no note and no push)
and a `research run` that ended on a node lead with the record's handle and ring it the same way: a coordinator's own, a worker's
launcher's, for a run whose sender is a lane, the lane's. The pod frees held runs whether or not this runs; it stops when infra's VM is suspended, and resumes
from the offset it last posted.

Each pass also writes /data/agent-watch/doorbells.json, which the watch's ready-to-wake rule reads: each open inbox's holder (the
notes' routes.d) and its latest doorbell in the last day (a reply in the inbox thread, `from NAME, thread ...`). The watch ignores
it once 30 minutes old, so a suspended VM silences that half of the rule, never fakes it.

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
BELLS = "/data/agent-watch/doorbells.json"
BELLS_WINDOW_S = 24 * 3600
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
       "stuck": "has a run in {status} since {since} ({minutes} min), nothing streamed or started behind it; not cancelled",
       "cancel_refused": "was held since {since} ({minutes} min), and the API refused to cancel it (409) and still reads it "
                         "going: left for a person (stop it in the dashboard)",
       "unreadable": "has a latest run the API won't read ({status}) since {since} ({minutes} min): left for a person",
       "ready_to_wake": "is idle with nothing waking it: {reason}; the watch {action}",
       "worker_idle": "went idle at {at}; if it didn't report back, resume it with Task",
       "worker_long": "has been ACTIVE since {since} ({hours} h); if it isn't working, stop it and resume it with Task",
       "silent": "has been ACTIVE {active_minutes} min with no note and no push for {minutes} min; check on it",
       "run_ended": "ended rc={rc} after {after} on {node}"}
TOP = {"erroring", "stuck", "cancel_refused", "held_again", "unreadable"}
WORKER = {"worker_idle", "worker_long"}         # rung to the lane's handle: only its coordinator can resume a worker
HANDLE = WORKER | {"silent", "run_ended"}       # rung to the record's handle


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def rung(rec: dict) -> str | None:
    """The handle whose inbox the alert rings, if any."""
    if rec.get("kind") in TOP:
        return "top"
    if rec.get("kind") in HANDLE and rec.get("handle") in {h["handle"] for h in REG.get("handles") or []}:
        return rec["handle"]
    return None


def text(rec: dict) -> str:
    link = f"<https://cursor.com/agents/{rec.get('agent')}|{rec.get('agent')}>"
    line = SAY.get(rec.get("kind"), "{kind}").format(**{"since": "?", "minutes": "?", "action": "recorded it", "restarts": "?",
                                                         "reason": "?", "status": "?", "at": "?", "hours": "?", "rc": "?",
                                                         "after": "?", "node": "?", "active_minutes": "?", **rec})
    lead = f"{slack.tag(slack.entry(REG, to))} " if (to := rung(rec)) else ""
    name = rec.get("name") or "(unnamed)"
    if rec.get("kind") == "silent":
        again = " Second window: the last ring until it shows activity." if rec.get("window") == 2 else ""
        return f"{lead}*Agent watch:* agent {name} ({link}) {line}.{again}"
    if rec.get("kind") == "run_ended":
        run = f"run {rec.get('run')} ({rec.get('what') or '?'})"
        if rec.get("role") == "worker":
            return f"{lead}*Agent watch:* {run} of worker {name} ({link}) {line}; resume {name} with Task."
        if rec.get("role") == "coordinator":
            return f"{lead}*Agent watch:* your {run} ({link}) {line}; resume."
        return f"{lead}*Agent watch:* {run} of lane {rec.get('handle')} {line}."
    if rec.get("kind") in WORKER:
        lane = f" of lane {rec['lane']}" if rec.get("lane") else ""
        return f"{lead}*Agent watch:* worker {name} ({link}){lane} {line}."
    handle = rec.get("handle") or HOLDERS.get(rec.get("agent"))
    who = f"@{handle}'s coordinator" if handle else (rec.get("name") or "an agent")
    return (f"{lead}*Agent watch:* {who} ({link}, run {rec.get('run')}) {line}."
            + (f" New run {rec['new_run']}." if rec.get("new_run") else ""))


def pod(cmd: str, data: str | None = None) -> str:
    return subprocess.run([*connect.ssh_base(budgets.GUARD_HOST), cmd], input=data, capture_output=True, text=True, timeout=120,
                          check=True).stdout


def ring(api: slack.Slack, ts: str, post: str, name: str = "top") -> None:
    """Ring `name`'s inbox thread with the alert posted at `ts`, as a doorbell its reader acts on without another read."""
    to, r = slack.Routes().resolve(name)
    if not r or r.get("closed"):
        log(f"@{name} has no open inbox: the alert stands in #agent-alerts, unrung")
        return
    body = f"{post} {slack.permalink(REG, ALERTS, ts)}"
    api.call("chat.postMessage", channel=r.get("channel") or slack.channel_id_for(REG, slack.FRONT_DOOR), thread_ts=r["inbox_ts"],
             username="router", unfurl_links="false", unfurl_media="false", text=slack.doorbell("infra", ts, ts, body))
    log(f"rang @{to}")


def bells(api: slack.Slack, now: float) -> dict:
    """doorbells.json: each open inbox's holder and the ts of its latest doorbell in the last BELLS_WINDOW_S (0 for none)."""
    routes, inboxes = slack.Routes(), {}
    for name in sorted({*routes.local, *(h["handle"] for h in REG.get("handles") or [])}):
        r = routes.get_route(name, fresh=True)
        if not r or r.get("closed") or not slack.AGENT_ID_RE.fullmatch(str(r.get("holder") or "")):
            continue
        msgs = api.paged("conversations.replies", "messages", channel=r.get("channel") or slack.channel_id_for(REG, slack.FRONT_DOOR),
                         ts=r["inbox_ts"], oldest=f"{now - BELLS_WINDOW_S:.6f}")
        rung = [float(m["ts"]) for m in msgs if m.get("ts") != r["inbox_ts"]
                and (b := slack.DOORBELL_RE.match(m.get("text") or "")) and b[1] != "router"]
        inboxes[name] = {"agent": r["holder"], "last": max(rung, default=0.0)}
    return {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "inboxes": inboxes}


def once() -> None:
    try:
        b = json.dumps(bells(slack.Slack(os.environ.get("SLACK_BOT_TOKEN")), time.time()))
        if DRY:
            log(f"would write {BELLS}: {b}")
        else:
            pod(f"cat > {BELLS}.tmp && mv {BELLS}.tmp {BELLS}", b)
    except slack.SlackError as e:
        log(f"doorbells not read: {e}")
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
        if rec.get("kind") == "unreadable" and not rec.get("since"):   # a goal run's id, often brief: posted if it lasts
            pass
        elif rec.get("kind") == "ready_to_wake" and "409" in str(rec.get("action", "")):
            pass    # Cursor refuses a run on an agent waiting on background work; an hourly post of that changes nothing
        elif DRY:
            log(f"would post: {text(rec)}" + (f" and ring @{to}" if (to := rung(rec)) else ""))
        else:
            api = slack.Slack(os.environ.get("SLACK_BOT_TOKEN"))
            ts = slack._post(api, ALERTS, "infra", post := text(rec))
            if to := rung(rec):
                try:
                    ring(api, ts, post, to)
                except slack.SlackError as e:       # the alert is posted: posting it again would post it twice
                    log(f"posted, but @{to} not rung: {e}")
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
