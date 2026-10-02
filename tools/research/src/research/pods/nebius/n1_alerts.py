"""Node 1's stall alerts. On 1 Oct both lease pools sat blocked and node 1's dispatcher failed every tick for over two hours
before anyone heard (circuits, infra's 7th user interview). Every minute, one ssh read of node 1; each incident gets one post
addressed to its owner, and a "cleared" reply in that thread when it ends:

- a lease pool's `blocked` file exists (gpu_stray.py writes it, a person removes it): that pool's owner, with the file's text,
  which names the pid, pod and GPU;
- the dispatcher's last 3 ticks in its tmux pane failed, or it hasn't ticked for 10 minutes: circuits and infra, with the error;
- node 1 can't be read for 10 minutes: infra.

The posts go in #agent-coordination, where they ring the owners' doorbells (#agent-alerts rings none). Infra, their author, can't
be woken by its own post, so its share also goes in its inbox thread. Node 1 holds no Slack credential, so this runs where
`research msg` works (infra's VM, tmux `n1-alerts`) and stops when that VM is suspended.

  python3 n1_alerts.py [--node n1] [--every 60] [--state PATH] [--once] [--dry-run]   (research importable)
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

POOLS = {"/run/gpu-lease": "proofs", "/run/gpu-lease-circuits": "circuits"}
DISPATCH = ("circuits", "infra")
SESSION = "node1-dispatch"
FAILS = 3
STALL_S = 600
UNREAD_S = 600
STAMP = "%Y-%m-%dT%H:%M:%SZ"
TICK_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) tick(?: failed: (.*)| )")

# run on node 1 with `python3 -`: the pools' blocked files and the pane's last ticks (None when the pane can't be read)
REMOTE = f"""
import json, os, re, subprocess, time
out = {{"now": time.time(), "blocked": {{}}, "ticks": None}}
for d in {list(POOLS)!r}:
    p = os.path.join(d, "blocked")
    if os.path.exists(p):
        try:
            out["blocked"][d] = open(p, errors="replace").read(600).strip()
        except OSError as e:
            out["blocked"][d] = "(unreadable: %s)" % e
r = subprocess.run(["tmux", "capture-pane", "-pJ", "-t", {SESSION!r}, "-S", "-400"], capture_output=True, text=True)
if r.returncode == 0:
    out["ticks"] = [ln for ln in r.stdout.splitlines() if re.match({TICK_RE.pattern!r}, ln)][-{FAILS}:]
print(json.dumps(out))
"""

Post = Callable[[list, str, "str | None"], "str | None"]


def incidents(snap: dict) -> dict[str, tuple[list[str], str, str]]:
    """The incidents one read of node 1 shows: key -> (who, the alert, its "cleared" line)."""
    out = {}
    for d, text in sorted(snap["blocked"].items()):
        out[f"blocked:{d}"] = ([POOLS.get(d, "infra")],
                               f"*Node 1's lease pool `{d}` is blocked*, so it grants no new GPU and has cancelled its waiting "
                               f"holders. `{d}/blocked` says: _{text[:400]}_ Once that process is gone, remove the file "
                               f"(`ssh n1 sudo rm {d}/blocked`).",
                               f"*Cleared:* `{d}/blocked` is gone, so the pool grants GPUs again.")
    pane = f"`ssh n1 tmux capture-pane -pJ -t {SESSION}`"
    ticks = snap.get("ticks")
    if ticks is None:
        out["dispatch"] = (list(DISPATCH), f"*Node 1's dispatcher pane is gone* (tmux `{SESSION}`), so nobody can see whether it "
                           "ticks. If it was restarted elsewhere, tell infra the session's name.",
                           "*Cleared:* node 1's dispatcher ticks again.")
        return out
    ms = [m for m in map(TICK_RE.match, ticks) if m]
    if len(ms) >= FAILS and all(m[2] is not None for m in ms[-FAILS:]):
        out["dispatch"] = (list(DISPATCH), f"*Node 1's dispatcher failed its last {FAILS} ticks* ({ms[-FAILS][1]} to {ms[-1][1]}): "
                           f"`{ms[-1][2][:300]}`. Its pane: {pane}.", "*Cleared:* node 1's dispatcher ticks again.")
    elif not ms or snap["now"] - calendar.timegm(time.strptime(ms[-1][1], STAMP)) > STALL_S:
        since = f"since {ms[-1][1]}" if ms else "in its pane's last 400 lines"
        out["dispatch"] = (list(DISPATCH), f"*Node 1's dispatcher hasn't ticked {since}*; it ticks every minute. Its pane: {pane}.",
                           "*Cleared:* node 1's dispatcher ticks again.")
    return out


def step(state: dict, snap: dict | None, now: float, post: Post) -> None:
    """Post each new incident once, and a "cleared" reply for each that ended. `snap` None is a failed read, which clears
    nothing. `post(who, text, thread)` returns the thread later replies go in (None when there is none)."""
    if snap is None:
        late = now - state.setdefault("last_read", now)
        found = {k: None for k in state["open"]}
        if late > UNREAD_S:
            found["unreadable"] = (["infra"], f"*Infra's node 1 alert relay hasn't read node 1 for {late / 60:.0f} minutes*, so "
                                   "its lease pools and dispatcher aren't watched.", "*Cleared:* the relay reads node 1 again.")
    else:
        state["last_read"] = now
        found = incidents(snap)
    for key, inc in found.items():
        if inc and key not in state["open"]:
            who, text, cleared = inc
            state["open"][key] = {"who": who, "thread": post(who, text, None), "cleared": cleared}
    for key in [k for k in state["open"] if k not in found]:
        o = state["open"][key]
        post(o["who"], o["cleared"], o["thread"])
        del state["open"][key]


def slack_post(dry: bool) -> Post:
    from research import slack

    reg = slack.load_registry(slack.registry_path())
    api, ch, routes = slack.Slack(os.environ.get("SLACK_BOT_TOKEN"), dry_run=dry), slack.channel_id_for(reg, slack.FRONT_DOOR), slack.Routes()

    def post(who: list, text: str, thread: str | None) -> str | None:
        ts = None
        if others := [w for w in who if w != "infra"]:
            ts = slack._post(api, ch, "infra", f"{' '.join(slack.tag(slack.entry(reg, w)) for w in others)} {text}", thread,
                             route=(reg, routes), tags=("alert",))
        if "infra" in who:
            inbox = (routes.get_route("infra") or {}).get("inbox_ts")
            slack._post(api, ch, "infra", f"from n1-alerts: {text}", inbox)
        return ts
    return post


def read(node: str) -> dict:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, "python3", "-"], input=REMOTE,
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-300:] or f"ssh exited {r.returncode}")
    return json.loads(r.stdout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="n1_alerts", description=__doc__.split("\n\n")[0])
    ap.add_argument("--node", default="n1", help="node 1's ssh host")
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--state", default=str(Path.home() / ".research" / "n1-alerts.json"))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the posts instead (and keep no state)")
    ns = ap.parse_args(argv)
    path = Path(ns.state)
    state = json.loads(path.read_text()) if path.exists() else {"open": {}}
    post = slack_post(ns.dry_run)
    while True:
        stamp = time.strftime("%H:%M:%SZ", time.gmtime())
        try:
            snap = read(ns.node)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
            print(f"{stamp} read failed: {type(e).__name__}: {e}", flush=True)
            snap = None
        try:
            step(state, snap, time.time(), post)
        except Exception as e:  # noqa: BLE001  (a post that failed is made again at the next read)
            print(f"{stamp} post failed: {type(e).__name__}: {e}", flush=True)
        if not ns.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state))
        print(f"{stamp} open: {', '.join(sorted(state['open'])) or 'none'}", flush=True)
        if ns.once:
            return 0
        time.sleep(ns.every)


if __name__ == "__main__":
    sys.exit(main())
