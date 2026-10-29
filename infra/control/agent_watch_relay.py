"""Infra's relay for the agent watch (INF-69, `agent_watch.sh` on the control pod). Every 2 minutes this reads the lines added to
/data/agent-watch/events.jsonl (on the pod itself with $AGENT_WATCH_ON_POD, else over ssh) and posts each in #agent-alerts: a
held run the watch cancelled and continued, a tool call running 45 minutes, a crashed run, a coordinator ready to wake. An agent that holds a handle
(the Slack registry's `holder`) is named by its handle. What the watch leaves for a person (`TOP`: erroring, stuck, a refused
cancel, held again within the hour, and a run unreadable for an hour; the first unreadable line is posted nowhere) leads with
@top, and rings @top's inbox thread as `ring_decision` does for a decided card (the inbox is in the front door, so the post's
own route can't ring it). A cloud worker (a Task subagent) that went idle, or has been ACTIVE for hours, leads with its lane's
handle and rings that inbox the same way, for its coordinator to resume it with Task (the watch can't start a worker's run;
the doorbell also counts toward the coordinator's own ready-to-wake rule). A worker alert is never posted unrung: one whose
lane no handle owns (nor its parent's) leads with @infra and says its lane is unknown. A silent worker (ACTIVE 90 min with no note
and no push) leads with the record's handle and rings it the same way; a silent coordinator (60 min, its record's owner top)
rings @top. A `research run` that ended on a node leads with its
sender's handle (a coordinator's own, a worker's launcher's, for a run whose sender is a lane, the lane's), and its doorbell is
`research msg ring-run` (INF-140), the one sender of run-end doorbells: `from runs, run R rc N (AGENT): ...`, once per run however
often it is called, with `--to` the watch's resolved handle for all but a coordinator's own run. A worker's run that ended rc 0
is posted but rings no one: the watch resumes the worker itself, and rings its coordinator only if it stays idle (INF-216: run
ends were 47% of all doorbells, most of them workers' successes). The runs of one read whose sender
no handle owns are one post that leads with @infra, naming each run's sender, rung as an alert. A record this relay can't render
(a kind it doesn't know, or one missing what its sentence needs) is posted as its raw JSON to @infra, never as a sentence of ?s.
A record the watch marks `quiet` (a run its own lane cancelled, a worker relaunch it declined) is posted with no handle and
rings no one. Each line posted is kept by its sha256 beside the offset (`sent_path`, 3 days), so a restart, or an offset lost or
reset, never posts it again.
A failure the watch acted on (`FAIL`: held, stalled, held unread, crashed, idle after
its run ended, relaunched) leads with the record's `owner` and rings its inbox: @top for a coordinator, the launcher's handle for a
worker. Several workers of one parent that stopped together come as one `workers_failed`, posted and rung once to the lane's handle
with each worker named, and the wake-up it says it set. A handle renamed in the registry ("renamed from @OLD") is rung by its new
name. The pod frees held runs whether or not this runs; this relay stops with the pod, and resumes from the offset it last
posted.

Each pass also writes /data/agent-watch/doorbells.json, which the watch's ready-to-wake rule reads: each open inbox's holder (the
notes' routes.d) and its latest doorbell in the last day (a reply in the inbox thread, `from NAME, thread ...`, or a run's end,
`from runs, run ...`). The watch ignores it once 30 minutes old, so a stopped relay silences that half of the rule, never fakes it.

usage (the watchers' `agent-watch-relay` loop on the control pod, infra/control/watchers.toml; research importable with
`research msg ring-run` (main since #2754), $SLACK_BOT_TOKEN set, $RESEARCH_SLACK_REGISTRY when not run from a checkout,
$AGENT_WATCH_OFFSET the offset's file, default ~/agent-watch/offset):
  python3 infra/control/agent_watch_relay.py [--once] [--dry-run]"""
import hashlib
import http.client
import json
import os
import re
import shlex
import string
import subprocess
import sys
import time
from pathlib import Path

from research import slack
from research.pods import budgets, connect

EVENTS = "/data/agent-watch/events.jsonl"
BELLS = "/data/agent-watch/doorbells.json"
BELLS_WINDOW_S = 24 * 3600
OFFSET = Path(os.environ.get("AGENT_WATCH_OFFSET") or Path.home() / "agent-watch/offset")
SENT_KEEP_S = 3 * 86400     # the sent-set beside OFFSET (`sent_path`) keeps each line posted this long
EVERY_S = 2 * 60
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
       "no_step": "has shown nothing (no note, no push, nothing streamed) since {since} ({hours} h) after its two silent rings: "
                  "stop it and archive it, or resume it",
       "run_ended": "ended rc={rc} after {after} on {node}",
       "stalled": "streamed nothing for {minutes} min since {since}, its agent not moving either (a hung call or a suspended "
                  "VM); the watch {action}",
       "idle_after_run": "sat idle after its run {ended_run} ended ({result}), with nothing resuming it for {minutes} min; the "
                         "watch {action}",
       "relaunch": "needs a fresh launch ({reason}); the watch {action}",
       "not_relaunched": "was idle after its run {ended_run} ended, and the watch did not relaunch it: {why}",
       "workers_failed": "stopped within {minutes} min",
       "held_unread": "did nothing itself for {minutes} min since {since} while a worker it waits on streamed, so its "
                      "follow-ups sat unread ({reason}); the watch {action}"}
TOP = {"erroring", "stuck", "cancel_refused", "held_again", "unreadable", "no_step"}
WORKER = {"worker_idle", "worker_long"}         # rung to the lane's handle: only its coordinator can resume a worker
HANDLE = WORKER | {"silent", "run_ended", "workers_failed"}       # rung to the record's handle
FAIL = {"held", "stalled", "crashed", "idle_after_run", "relaunch", "held_unread"}  # rung to the record's owner: who follows it up
UNROUTED = "infra"      # rung for a worker whose lane no handle owns: a worker alert is never posted unrung
UNSENT_LISTED = 12      # runs of unknown senders one post names; the rest are counted
RING_RUN = [sys.executable, "-m", "research", "msg", "ring-run"]
RUNBELL_RE = re.compile(r"^from runs, run \S+ rc ")      # ring-run's doorbell
RUN_ROLES = {"coordinator", "worker", "lane", "unknown"}
RAW_MAX = 2500
#: a renamed handle's old name -> its new one, from the registry's "renamed from @OLD" (hash -> kernels, 18 Oct): lanes' notes
#: and workers.d still carry the old name
RENAMED = {old: h["handle"] for h in REG.get("handles") or []
           for old in re.findall(r"renamed from @([a-z0-9-]+)", str(h.get("charter", "")) + str(h.get("holder", "")))}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def is_worker(rec: dict) -> bool:
    return rec.get("kind") in WORKER or rec.get("kind") == "workers_failed" or rec.get("role") == "worker"


def lane_of(rec: dict) -> str | None:
    """A worker record's handle: its owner, its handle, else its lane when that is a handle; None when none is."""
    handles = {h["handle"] for h in REG.get("handles") or []}
    return next((h for h in (RENAMED.get(x, x) for x in (rec.get("owner"), rec.get("handle"), rec.get("lane"))) if h in handles),
                None)


def renderable(rec: dict) -> bool:
    """Whether `text` has a sentence for `rec`: a kind it knows, in a shape it knows, with every field that sentence reads."""
    kind = rec.get("kind")
    if kind not in SAY:
        return False
    if kind == "run_ended":
        if rec.get("role") not in RUN_ROLES:
            return False
        if rec.get("role") == "unknown":
            return bool(rec.get("runs"))
    if kind == "workers_failed":
        return bool(rec.get("members"))
    need = {f for _, f, _, _ in string.Formatter().parse(SAY[kind]) if f} - {"action"}
    return need <= {k for k, v in rec.items() if v is not None}


def rung(rec: dict) -> str | None:
    """The handle whose inbox the alert rings, if any. A worker's is always someone's: only its coordinator can resume it, and
    one whose lane is unknown goes to @infra (`UNROUTED`), who finds that coordinator; so does a record `text` can't render."""
    if not renderable(rec):
        return UNROUTED
    if rec.get("quiet"):        # the watch's own judgement that nobody must act: a run its lane cancelled, a relaunch declined
        return None
    if rec.get("kind") == "run_ended" and rec.get("sender_status") == "ACTIVE":
        return None             # its sender is in a turn and sees its own run end; if it then idles, the watch resumes it
    if rec.get("kind") in TOP or (rec.get("kind") == "silent" and rec.get("owner") == "top"):
        return "top"
    handles = {h["handle"] for h in REG.get("handles") or []}
    owner = RENAMED.get(rec.get("owner"), rec.get("owner"))
    if rec.get("kind") in FAIL and owner in handles:
        return owner
    handle = RENAMED.get(rec.get("handle"), rec.get("handle"))
    if rec.get("kind") in HANDLE and handle in handles:
        return handle
    if is_worker(rec):
        return lane_of(rec) or UNROUTED
    if rec.get("kind") == "run_ended":
        return UNROUTED
    return None


def unknown(rec: dict) -> str:
    """What a worker alert says when no handle owns the worker or its parent, so @infra knows why it was rung."""
    return (f" Its lane is unknown (no handle owns it or its parent), so @{UNROUTED} is rung to find its coordinator."
            if is_worker(rec) and lane_of(rec) is None else "")


def woke(rec: dict) -> str:
    """The wake-up a worker says it set, as the watch read it from its conversation."""
    if not rec.get("wake"):
        return ""
    due = f", due about {rec['wake_due'][11:16]}Z" if rec.get("wake_due") else ""
    return f"it set its own wake-up (\u201c{rec['wake']}\u201d{due})"


def member(rec: dict) -> str:
    name = rec.get("name") or "(unnamed)"
    what = (f"idle after its run {rec.get('ended_run')} ended" if rec.get("kind") == "idle_after_run" else
            f"idle at {str(rec.get('at') or rec.get('since') or '?')[11:16]}Z")
    wake = f"; {woke(rec)}" if rec.get("wake") else ""
    return f"{name} (<https://cursor.com/agents/{rec.get('agent')}|{str(rec.get('agent'))[:11]}>) {what}{wake}"


def unsent(u: dict) -> str:
    """One run of an unknown sender: the run, its node and result, and who `submitted_by` says sent it."""
    s = str(u.get("sender") or "")
    a = re.search(r"bc-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", s)
    who = f"<https://cursor.com/agents/{a[0]}|{a[0][:11]}>" if a else (f"`{s}`" if s else "no `submitted_by` or `--by`")
    return f"{u.get('run')} on {u.get('node')} ({u.get('what') or '?'}, {u.get('state')} rc={u.get('rc')}) sent by {who}"


def text(rec: dict) -> str:
    if not renderable(rec):
        raw = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        raw = raw if len(raw) <= RAW_MAX else raw[:RAW_MAX] + " …(cut)"
        return (f"{slack.tag(slack.entry(REG, UNROUTED))} *Agent watch:* a record this relay can't render (kind "
                f"`{rec.get('kind')}`), as the watch wrote it: ```{raw}```")
    link = f"<https://cursor.com/agents/{rec.get('agent')}|{rec.get('agent')}>"
    line = SAY.get(rec.get("kind"), "{kind}").format(**{"since": "?", "minutes": "?", "action": "recorded it", "restarts": "?",
                                                         "reason": "?", "status": "?", "at": "?", "hours": "?", "rc": "?",
                                                         "after": "?", "node": "?", "active_minutes": "?", "ended_run": "?",
                                                         "result": "?", **rec})
    lead = f"{slack.tag(slack.entry(REG, to))} " if (to := rung(rec)) else ""
    name = rec.get("name") or "(unnamed)"
    if rec.get("kind") == "silent":
        again = " Second window: the last ring until it shows activity." if rec.get("window") == 2 else ""
        return f"{lead}*Agent watch:* agent {name} ({link}) {line}.{again}"
    if rec.get("kind") == "run_ended":
        run = f"run {rec.get('run')} ({rec.get('what') or '?'})"
        if rec.get("quiet"):
            who = (f"worker {name} ({link})" if rec.get("role") == "worker" else f"{name} ({link})" if rec.get("agent")
                   else f"lane {rec.get('handle') or '?'}")
            return (f"*Agent watch:* {run} of {who} {line}, cancelled by its own lane ({rec.get('cancelled_by') or '?'}: "
                    f"\u201c{str(rec.get('cancel_reason') or '')[:160]}\u201d): nothing to resume.")
        if rec.get("sender_status") == "ACTIVE" and rec.get("role") in ("worker", "coordinator"):
            who = f"worker {name}" if rec.get("role") == "worker" else name
            return f"*Agent watch:* {run} of {who} ({link}) {line}; its sender is in a turn, so no one is rung."
        if rec.get("role") == "worker" and rec.get("listed"):
            return (f"{lead}*Agent watch:* {run} of worker {name} ({link}) {line}; the watch starts a run on it with the "
                    "result if it is still idle in 10 min.")
        if rec.get("role") == "worker":
            return f"{lead}*Agent watch:* {run} of worker {name} ({link}) {line}; resume {name} with Task.{unknown(rec)}"
        if rec.get("role") == "coordinator":
            return f"{lead}*Agent watch:* your {run} ({link}) {line}; resume."
        if rec.get("role") == "unknown":
            runs = rec.get("runs") or []
            listed = "; ".join(unsent(u) for u in runs[:UNSENT_LISTED]) + (
                f"; and {len(runs) - UNSENT_LISTED} more" if len(runs) > UNSENT_LISTED else "")
            return (f"{lead}*Agent watch:* {len(runs)} research run{'s' if len(runs) != 1 else ''} ended whose sender no handle "
                    f"owns (not a coordinator, a worker the watch knows, its parent, or a lane): {listed}. @{UNROUTED} is rung "
                    "to find who sent them.")
        return f"{lead}*Agent watch:* {run} of lane {rec.get('handle')} {line}."
    if rec.get("kind") == "workers_failed":
        members = rec.get("members") or []
        handle = rec.get("handle")
        if rec.get("agent"):
            who = " ".join(x for x in (f"@{handle}'s coordinator" if handle else "the coordinator", rec.get("name")) if x)
            parent = f"{who} ({link})"
        else:
            parent = f"lane {rec.get('lane') or '?'}"
        return (f"{lead}*Agent watch:* {len(members)} workers of {parent} {line}: "
                + "; ".join(member(m) for m in members) + f". Resume each that didn't report back with Task.{unknown(rec)}")
    if rec.get("kind") in WORKER or rec.get("role") == "worker":
        lane = f" of lane {x}" if (x := rec.get("lane") or lane_of(rec)) else ""
        new = f" New agent {rec['new_agent']}." if rec.get("new_agent") else ""
        wake = f" Before it stopped, {woke(rec)}." if rec.get("wake") else ""
        return f"{lead}*Agent watch:* worker {name} ({link}){lane} {line}.{new}{wake}{unknown(rec)}"
    handle = rec.get("handle") or HOLDERS.get(rec.get("agent"))
    who = f"@{handle}'s coordinator" if handle else (rec.get("name") or "an agent")
    return (f"{lead}*Agent watch:* {who} ({link}, run {rec.get('run')}) {line}."
            + (f" New run {rec['new_run']}." if rec.get("new_run") else "")
            + (f" New agent {rec['new_agent']}." if rec.get("new_agent") else ""))


def pod(cmd: str, data: str | None = None) -> str:
    """`cmd` on the control pod: over ssh, or in a local shell when $AGENT_WATCH_ON_POD is set (the watchers supervisor's loop)."""
    argv = ["sh", "-c", cmd] if os.environ.get("AGENT_WATCH_ON_POD") else [*connect.ssh_base(budgets.GUARD_HOST), cmd]
    return subprocess.run(argv, input=data, capture_output=True, text=True, timeout=120, check=True).stdout


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


def call(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)


def ring_run(rec: dict, to: str, ts: str) -> None:
    """A run's end, rung with `research msg ring-run` (one doorbell per run, once however often called), its text the alert's
    one line and link. `--to` is the watch's handle for all but a coordinator's own run, which ring-run finds by the agent's
    route; a former coordinator, which no route holds any more, is rung again with `--to`."""
    one = (f"{rec.get('what')} {rec.get('state') or 'ended'} after {rec.get('after')} on {rec.get('node')}: "
           f"{slack.permalink(REG, ALERTS, ts)}")
    argv = [*RING_RUN, "--run", str(rec["run"]), "--rc", str(rec["rc"] if rec.get("rc") is not None else rec.get("state")),
            "--text", one, *(("--agent", rec["agent"]) if rec.get("agent") else ())]
    to_arg = ["--to", f"@{to}"]
    own = rec.get("role") == "coordinator" and rec.get("agent")
    if DRY:
        log(f"would run: {shlex.join(argv if own else [*argv, *to_arg])}")
        return
    r = call(argv if own else [*argv, *to_arg])
    if r.returncode and own:
        r = call([*argv, *to_arg])
    if r.returncode:
        raise slack.SlackError(f"ring-run for {rec['run']} exited {r.returncode}: {(r.stderr or '').strip()[-300:]}")
    log(f"rang @{to} for run {rec['run']} (ring-run)")


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
                and ((b := slack.DOORBELL_RE.match(m.get("text") or "")) and b[1] != "router"
                     or m.get("username") == "router" and RUNBELL_RE.match(m.get("text") or ""))]
        inboxes[name] = {"agent": r["holder"], "last": max(rung, default=0.0)}
    return {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "inboxes": inboxes}


def once() -> None:
    try:
        b = json.dumps(bells(slack.Slack(os.environ.get("SLACK_BOT_TOKEN")), time.time()))
        if DRY:
            log(f"would write {BELLS}: {b}")
        else:
            pod(f"cat > {BELLS}.tmp && mv {BELLS}.tmp {BELLS}", b)
    except (slack.SlackError, http.client.HTTPException) as e:   # a page cut short (IncompleteRead) is no OSError
        log(f"doorbells not read: {type(e).__name__}: {e}")
    done = int(OFFSET.read_text()) if OFFSET.exists() else 0
    sent = {k: v for k, v in load_sent().items() if isinstance(v, (int, float)) and v >= time.time() - SENT_KEEP_S}
    *lines, _ = pod(f"cat {EVENTS} 2>/dev/null || true").split("\n")   # a line still being written waits a pass
    if len(lines) < done:
        log(f"{EVENTS} has {len(lines)} lines, fewer than the {done} posted: replaced, so posting it from the top")
        done = 0
    for ln in lines[done:]:
        try:
            rec = json.loads(ln)
        except ValueError:
            rec = {"kind": "garbled", "line": ln[:RAW_MAX]}
        key = hashlib.sha256(ln.encode()).hexdigest()
        if key in sent:
            log(f"line {done + 1} was posted before ({time.strftime('%H:%M:%SZ', time.gmtime(sent[key]))}): not again")
        elif rec.get("kind") == "unreadable" and not rec.get("since"):   # a goal run's id, often brief: posted if it lasts
            pass
        elif rec.get("kind") == "ready_to_wake" and "409" in str(rec.get("action", "")):
            pass    # Cursor refuses a run on an agent waiting on background work; an hourly post of that changes nothing
        elif DRY:
            log(f"would post: {text(rec)}" + (f" and ring @{to}" if (to := rung(rec)) else ""))
        else:
            api = slack.Slack(os.environ.get("SLACK_BOT_TOKEN"))
            ts = slack._post(api, ALERTS, "infra", post := text(rec))
            sent[key] = time.time()
            save_sent(sent)
            if (to := rung(rec)) and rec.get("kind") == "run_ended" and rec.get("role") == "worker" and rec.get("rc") == 0:
                log(f"run {rec.get('run')}: a worker's run ended rc 0, posted unrung (the watch resumes the worker)")
            elif to:
                try:
                    if rec.get("kind") == "run_ended" and rec.get("role") != "unknown" and renderable(rec):
                        ring_run(rec, to, ts)
                    else:
                        ring(api, ts, post, to)
                except (slack.SlackError, http.client.HTTPException, subprocess.SubprocessError, OSError) as e:   # posted: never twice
                    log(f"posted, but @{to} not rung: {e}")
        done += 1
        OFFSET.parent.mkdir(parents=True, exist_ok=True)
        OFFSET.write_text(str(done))


def sent_path() -> Path:
    """Each line posted (its sha256) -> when, beside OFFSET: a restart, or an offset lost or reset, never posts a line again."""
    return OFFSET.with_name(OFFSET.name + ".sent.json")


def load_sent() -> dict:
    try:
        d = json.loads(sent_path().read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def save_sent(sent: dict) -> None:
    p = sent_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(sent, sort_keys=True))
    tmp.replace(p)


def main() -> int:
    while True:
        try:
            once()
        except (subprocess.SubprocessError, OSError, slack.SlackError, http.client.HTTPException) as e:   # posted again next pass
            log(f"pass failed: {type(e).__name__}: {e}")
        if "--once" in sys.argv:
            return 0
        time.sleep(EVERY_S)


if __name__ == "__main__":
    sys.exit(main())
