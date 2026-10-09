"""Node 1's stall alerts. On 1 Oct both lease pools sat blocked and node 1's dispatcher failed every tick for over two hours
before anyone heard (circuits, infra's 7th user interview). Every minute, one ssh read of node 1; each incident gets one post
addressed to its owner, and a "cleared" reply in that thread when it ends:

- a lease pool's `blocked` file exists (gpu_stray.py writes it, a person removes it): that pool's owner, with the file's text,
  which names the pid, pod and GPU;
- the dispatcher's last 3 ticks failed, or it hasn't ticked for 10 minutes: circuits and infra, with the error. Its ticks are
  read from `loop.log`, where vy-node1-dispatch.service writes its stdout and stderr (a failed tick is on stderr), and from
  its old tmux pane while one is there;
- node 1 can't be read for 10 minutes: infra.

The same read takes each Kueue ClusterQueue's `spec.resourceGroups`. On 3 Oct node 1's quotas were changed by hand and the
steward couldn't find who or until when, so each field that changed since the last read becomes one line (the ClusterQueue,
the field, old -> new, and the managedFields manager that owns `spec.resourceGroups`, with its last write), posted to infra
and appended to node 1's `quota-changes.log` beside infra's own lines; a ClusterQueue created or deleted is a line too. The
first read with no saved baseline records it and posts nothing; a failed kubectl keeps the baseline and compares nothing.

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
SESSION = "node1-dispatch"     # its tmux pane before vy-node1-dispatch.service; read while an older loop may still run there
UNIT = "vy-node1-dispatch.service"
LOOP_LOG = "/workspace/jobs/dispatch/loop.log"
FAILS = 3
STALL_S = 600
UNREAD_S = 600
STAMP = "%Y-%m-%dT%H:%M:%SZ"
TICK_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) tick(?: failed: (.*)| )")
# a read-only kubeconfig's `kubectl` once research has no `sudo k3s` (pods/nebius/lockout: VY_KUBECTL=kubectl in its environment)
KUBECTL = os.environ.get("VY_KUBECTL", "sudo -n k3s kubectl").split()
QUOTA_LOG = "/workspace/verity-guest/quota-changes.log"
QUOTA_WHO = ["infra"]
UNSET = "(unset)"

# run on node 1 with `python3 -`: the pools' blocked files, the last ticks in the pane and in loop.log (None when unreadable),
# and the ClusterQueues with their spec's managedFields (`queues` None and `queues_error` when kubectl fails)
REMOTE = f"""
import json, os, re, subprocess, time
out = {{"now": time.time(), "blocked": {{}}, "pane": None, "log": None, "queues": None}}
try:
    r = subprocess.run({KUBECTL + ["get", "clusterqueues", "-o", "json", "--show-managed-fields"]!r}, capture_output=True,
                       text=True, timeout=30)
    if r.returncode:
        out["queues_error"] = r.stderr.strip()[-300:] or "kubectl exited %d" % r.returncode
    else:
        out["queues"] = [{{"metadata": {{"name": q["metadata"]["name"], "managedFields": [
                              f for f in q["metadata"].get("managedFields") or [] if not f.get("subresource")]}},
                          "spec": {{"resourceGroups": q["spec"].get("resourceGroups") or []}}}}
                         for q in json.loads(r.stdout)["items"]]
except Exception as e:
    out["queues_error"] = "%s: %s" % (type(e).__name__, e)
ticks = lambda text: [ln for ln in text.splitlines() if re.match({TICK_RE.pattern!r}, ln)][-{FAILS}:]
for d in {list(POOLS)!r}:
    p = os.path.join(d, "blocked")
    if os.path.exists(p):
        try:
            out["blocked"][d] = open(p, errors="replace").read(600).strip()
        except OSError as e:
            out["blocked"][d] = "(unreadable: %s)" % e
r = subprocess.run(["tmux", "capture-pane", "-pJ", "-t", {SESSION!r}, "-S", "-400"], capture_output=True, text=True)
if r.returncode == 0:
    out["pane"] = ticks(r.stdout)
try:
    with open({LOOP_LOG!r}, "rb") as f:
        f.seek(max(0, os.fstat(f.fileno()).st_size - 65536))
        out["log"] = ticks(f.read().decode(errors="replace"))
except OSError:
    pass
print(json.dumps(out))
"""


def last_ticks(snap: dict) -> list[str] | None:
    """The last FAILS tick lines of the pane and loop.log together, oldest first (both start with their UTC stamp); None when
    neither can be read."""
    if snap.get("pane") is None and snap.get("log") is None:
        return None
    return sorted({*(snap.get("pane") or []), *(snap.get("log") or [])})[-FAILS:]

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
    pane = f"`ssh n1 tail {LOOP_LOG}` (its stdout and stderr) and `ssh n1 systemctl status {UNIT}`"
    ticks = last_ticks(snap)
    if ticks is None:
        out["dispatch"] = (list(DISPATCH), f"*Node 1's dispatcher can't be seen*: `{LOOP_LOG}` (where `{UNIT}` writes) can't "
                           "be read, so nobody can see whether it ticks. If it was restarted elsewhere, tell infra where.",
                           "*Cleared:* node 1's dispatcher ticks again.")
        return out
    ms = [m for m in map(TICK_RE.match, ticks) if m]
    if len(ms) >= FAILS and all(m[2] is not None for m in ms[-FAILS:]):
        out["dispatch"] = (list(DISPATCH), f"*Node 1's dispatcher failed its last {FAILS} ticks* ({ms[-FAILS][1]} to {ms[-1][1]}): "
                           f"`{ms[-1][2][:300]}`. Its ticks: {pane}.", "*Cleared:* node 1's dispatcher ticks again.")
    elif not ms or snap["now"] - calendar.timegm(time.strptime(ms[-1][1], STAMP)) > STALL_S:
        since = f"since {ms[-1][1]}" if ms else "in its pane or the end of its log"
        out["dispatch"] = (list(DISPATCH), f"*Node 1's dispatcher hasn't ticked {since}*; it ticks every minute. Its ticks: {pane}.",
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


def flat(v, path: str) -> dict[str, str]:
    """`spec.resourceGroups` as `{field path: value}`: list items with a `name` (flavors, resources) are keyed by it, the
    rest by index; values are compared as strings, so `6` and `"6"` (as `kubectl patch` and `apply` write a quantity) agree."""
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
        out = {}
        for i, x in enumerate(v):
            named = isinstance(x.get("name"), str)
            out.update(flat({k: y for k, y in x.items() if not (named and k == "name")}, f"{path}[{x['name'] if named else i}]"))
        return out
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            out.update(flat(x, f"{path}.{k}"))
        return out
    return {path: f"[{', '.join(map(str, v))}]" if isinstance(v, list) else str(v)}


def writer(fields: list) -> tuple[str, str | None] | None:
    """The managedFields manager that last wrote a ClusterQueue's `spec.resourceGroups` and its time: the entry that owns that
    field, else the latest one that wrote any of spec. An entry keeps one time, its manager's latest write to any field it
    owns (the quiet hour's `stopPolicy` patches move `kubectl-patch`'s), and every `kubectl patch` is `kubectl-patch` unless
    it names a `--field-manager`."""
    spec = [f for f in fields if not f.get("subresource") and "f:spec" in (f.get("fieldsV1") or {})]
    owns = [f for f in spec if "f:resourceGroups" in f["fieldsV1"]["f:spec"]]
    best = max(owns or spec, key=lambda f: f.get("time") or "", default=None)
    return None if best is None else (best.get("manager") or "?", best.get("time"))


def quotas(items: list) -> dict[str, dict]:
    """ClusterQueue name -> `{"groups": flat(resourceGroups), "by": writer(managedFields)}`, from `kubectl get -o json`'s
    items (or REMOTE's trimmed copy of them)."""
    return {q["metadata"]["name"]: {"groups": flat(q["spec"].get("resourceGroups") or [], "resourceGroups"),
                                    "by": writer(q["metadata"].get("managedFields") or [])} for q in items}


def quota_lines(old: dict, new: dict) -> list[str]:
    """One line per field of a ClusterQueue that changed between two `quotas()`, and one per ClusterQueue created or deleted."""
    def by(q: dict) -> str:
        if not q["by"]:
            return " (no managedFields entry wrote its spec)"
        who, at = q["by"]
        return f" (spec.resourceGroups owner: {who}{f', last write {at}' if at else ''})"

    def had(q: dict) -> str:
        return ", ".join(f"`{p}` {v}" for p, v in q["groups"].items()) or "no resourceGroups"

    lines = []
    for name in sorted(old.keys() | new.keys()):
        o, n = old.get(name), new.get(name)
        if o is None:
            lines.append(f"ClusterQueue `{name}` is new: {had(n)}{by(n)}")
        elif n is None:
            lines.append(f"ClusterQueue `{name}` was deleted; it had {had(o)}")
        else:
            for p in [*n["groups"], *(p for p in o["groups"] if p not in n["groups"])]:
                a, b = o["groups"].get(p, UNSET), n["groups"].get(p, UNSET)
                if a != b:
                    lines.append(f"ClusterQueue `{name}`: `{p}` {a} -> {b}{by(n)}")
    return lines


def watch_quotas(state: dict, snap: dict | None, now: float, post: Post, append: Callable[[list], None]) -> str:
    """Compare this read's ClusterQueues with `state["quotas"]`; post each change to infra and `append` it to node 1's quota
    log, stamped. The first read records the baseline and posts nothing; an unread node or a failed kubectl changes nothing.
    A post or an append that fails is made again at the next read (`state["quota_unsent"]`), the other not. Returns what it
    did."""
    if snap is None:
        return "node 1 unread"
    if snap.get("queues") is None:
        return f"kubectl failed ({snap.get('queues_error') or 'no output'}); kept the baseline"
    new = quotas(snap["queues"])
    if "quotas" not in state:
        state["quotas"] = new
        return f"baseline of {len(new)} ClusterQueues"
    lines = quota_lines(state["quotas"], new)
    state["quotas"] = new
    unsent = state.setdefault("quota_unsent", {"log": [], "post": []})
    stamp = time.strftime(STAMP, time.gmtime(now))
    unsent["log"] += [f"{stamp} n1-alerts: {ln}" for ln in lines]
    unsent["post"] += lines
    failed = []
    for key, send in (("log", append), ("post", lambda ls: post(QUOTA_WHO, "\n".join(ls), None))):
        if unsent[key]:
            try:
                send(unsent[key])
                unsent[key] = []
            except Exception as e:  # noqa: BLE001
                failed.append(f"{key} failed, retried at the next read: {type(e).__name__}: {e}")
    return "; ".join([f"{len(lines)} changed" if lines else "unchanged", *failed])


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


def log_append(node: str, dry: bool) -> Callable[[list], None]:
    def append(lines: list) -> None:
        text = "".join(f"{ln}\n" for ln in lines)
        if dry:
            print(f"would append to {node}:{QUOTA_LOG}:\n{text}", end="", flush=True)
            return
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, f"cat >> {QUOTA_LOG}"], input=text,
                           capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise RuntimeError(f"appending to {QUOTA_LOG}: {r.stderr.strip()[-300:] or f'ssh exited {r.returncode}'}")
    return append


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="n1_alerts", description=__doc__.split("\n\n")[0])
    ap.add_argument("--node", default="n1", help="node 1's ssh host")
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--state", default=str(Path.home() / ".research" / "n1-alerts.json"))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the posts and quota-log lines instead (and keep no state)")
    ns = ap.parse_args(argv)
    path = Path(ns.state)
    state = json.loads(path.read_text()) if path.exists() else {"open": {}}
    post, append = slack_post(ns.dry_run), log_append(ns.node, ns.dry_run)
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
        try:
            quota = watch_quotas(state, snap, time.time(), post, append)
        except Exception as e:  # noqa: BLE001  (a ClusterQueue that doesn't parse leaves the baseline as it was)
            quota = f"compare failed: {type(e).__name__}: {e}"
        if not ns.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state))
        print(f"{stamp} open: {', '.join(sorted(state['open'])) or 'none'}; quotas: {quota}", flush=True)
        if ns.once:
            return 0
        time.sleep(ns.every)


if __name__ == "__main__":
    sys.exit(main())
