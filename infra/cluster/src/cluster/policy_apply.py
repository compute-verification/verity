"""`cluster policy apply`: the change that converges each key whose live value differs from the value in force, as a plan, and
with `--apply` that change written on the key's node and read back.

The keys and their state are `policy.diff`'s, read over the same SSH path, and a key is applied only when it is `drift` and
its surface has an applier (`applier`):

- `lean-slots.NAME`, `check-slots.NAME` and `fill.cpu-sets.OWNER`: one line of a slot file (`NAME VALUE`), as lean_slot.py,
  slot.py and fill_runner.py read it each time. The line naming it is commented out (kept, with who and when) and the value in
  force written after it, or appended when no line names it; a file where two lines name it is fixed by hand.
- `direct-gpus`, `direct-cpus`: gpu-lease's direct-run maps, written whole as the one value (gpu-lease reads the file whole, so
  no comment may stay in it).
- `user-cpuset.delegate`: the drop-in's `Delegate=` line, replaced in place (so converged it is again deploy.toml's file),
  then `systemctl daemon-reload` and the user manager's `daemon-reexec`, after which user.slice must keep the AllowedCPUs it
  had. Never `systemctl revert`, which would drop the runtime AllowedCPUs vy-device-cores sets. On vy-nebius-2 it is planned
  only (`PLAN_ONLY`): vy-device-cores owns node 2's cpusets.

Anything else (lease pools, Kueue's queues, the node's pools, booked windows) has no applier and is said so and skipped; so is
a key that is unknown (never applied: unread is not as declared) or undeclared (no value in force to converge to).

Every file is read exactly (`exact`) before it is planned, and the write (`script`) runs as one command on the node: under the
override log's lock it checks that each file and the log are byte for byte what was planned from (else it writes nothing),
writes each file, and appends one `apply` record per key to the node's hash-chained log (`policy.OVERRIDES`: key, value,
was, by, reason, path, the file's sha256 before and after). On node 2 the command starts with node 2's gate, so nothing is read
or written while a timed window is on or near. After it, the node is read again, and each applied key must read as in force.
"""
from __future__ import annotations

import base64
import difflib
import hashlib
import json
import shlex
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import ledger, nebius2, policy, status

#: a key's surface (and, for `fill`, its third part) and how its file is edited
EDITS = {"lean-slots": "line", "check-slots": "line", "fill.cpu-sets": "line", "direct-gpus": "whole", "direct-cpus": "whole",
         "user-cpuset": "delegate"}
NO_APPLIER = {
    "lease-pool": "n1_lease.py takes max, borrow and borrow-until from the policy under $VY_POOL_POLICY; the rest is its tmux "
                  "pane's environment, set by hand",
    "kueue": "a ClusterQueue is sky/kueue.yaml's, applied by cluster_up.sh",
    "pools": "the node's own `pools`, which the planner reads from the description (checks: the check slots' CPUs)",
    "fill.window": "a timed window is booked by hand in fill/windows",
}
#: (node, surface): why that surface is planned and never written there
PLAN_ONLY = {(nebius2.NODE, "user-cpuset"): "vy-device-cores owns node 2's cpusets (user.slice 0-119, its device cores "
                                            "120-127): write the drop-in by hand, never `systemctl revert`"}
#: written as root (through `sudo -n`), mode 644; every other file keeps its owner and mode
ROOT_FILES = (*policy.DIRECT.values(), policy.USER_CPUSET)
B64_MARK = "@@policy-b64 "
APPLIED = "@@policy-applied"
REEXEC = "systemctl --user daemon-reexec"
FAILED = 4


@dataclass
class Step:
    """One key's step: `change` (with the file it edits), `by-hand` (a change planned only: `PLAN_ONLY`), `skip` (why it isn't
    applied) or `unknown`."""
    key: str
    action: str
    why: str = ""
    owner: str = ""
    value: Any = None
    live: str | None = None
    path: str | None = None


@dataclass
class Plan:
    """A node's steps, and the files `--apply` writes (`files`: path to the bytes before and after) with their records, the
    commands after them, and the edits left to a person (`by_hand`)."""
    node: str
    steps: list[Step] = field(default_factory=list)
    files: dict[str, tuple[bytes | None, bytes]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    log: bytes | None = None
    then: list[str] = field(default_factory=list)
    by_hand: dict[str, tuple[bytes | None, bytes]] = field(default_factory=dict)
    refused: str | None = None


class Skip(Exception):
    pass


def surface(key: str) -> str:
    parts = key.split(".")
    return ".".join(parts[1:3]) if parts[1] == "fill" else parts[1]


def path_of(key: str) -> str:
    s = surface(key)
    return {"lean-slots": policy.LEAN_SLOTS, "check-slots": policy.CHECK_SLOTS, "fill.cpu-sets": f"{policy.FILL}/cpu-sets",
            "user-cpuset": policy.USER_CPUSET}.get(s) or policy.DIRECT[s]


def applier(key: str) -> str | None:
    """Why `key` has no applier, or None when it has one."""
    s = surface(key)
    return None if s in EDITS else "no applier: " + NO_APPLIER.get(s, f"nothing writes {s}")


def _fields(line: str) -> list[str]:
    return line.split("#")[0].split()


def _line(text: str, key: str, value: Any, note: str) -> tuple[str, str | None]:
    """A slot file with `key`'s line set to `value`: the old line commented out above the new one, or the new line appended."""
    name = key.split(".")[-1]
    lines = text.splitlines(keepends=True)
    at = [i for i, x in enumerate(lines) if _fields(x)[:1] == [name]]
    if len(at) > 1:
        raise Skip(f"{len(at)} lines of {path_of(key)} name {name}: fix it by hand")
    new = f"{name} {value}  # {note}\n"
    if not at:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return "".join(lines + [new]), None
    i = at[0]
    was = " ".join(_fields(lines[i])[1:])
    if was and policy.canon(key, was) == policy.canon(key, value):
        return text, was
    lines[i:i + 1] = [f"# {lines[i].rstrip()}   -- until {note}\n", new]
    return "".join(lines), was


def _whole(text: str, key: str, value: Any, note: str) -> tuple[str, str | None]:
    vals = [x for x in (_fields(line) for line in text.splitlines()) if x]
    return f"{value}\n", " ".join(vals[0]) if vals else None


def _delegate(text: str, key: str, value: Any, note: str) -> tuple[str, str | None]:
    """The drop-in with its one `Delegate=` line set to `value`, everything else as it was."""
    lines = text.splitlines(keepends=True)
    at = [i for i, x in enumerate(lines) if x.strip().startswith("Delegate=")]
    if len(at) > 1:
        raise Skip(f"{len(at)} Delegate= lines in {policy.USER_CPUSET} (systemd adds them up): fix it by hand")
    new = f"Delegate={value}\n"
    if at:
        was = lines[at[0]].split("=", 1)[1].strip()
        lines[at[0]] = new
        return "".join(lines), was
    if not text:
        return f"# {policy.USER_CPUSET}: written by {note}; `research deploy install` ships its comments\n[Service]\n{new}", None
    svc = [i for i, x in enumerate(lines) if x.strip() == "[Service]"]
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if svc:
        lines.insert(svc[0] + 1, new)
    else:
        lines += ["[Service]\n", new]
    return "".join(lines), None


EDITORS: dict[str, Callable[[str, str, Any, str], tuple[str, str | None]]] = {"line": _line, "whole": _whole, "delegate": _delegate}


def _post(path: str) -> list[str]:
    return ["sudo -n systemctl daemon-reload (user.slice keeps its AllowedCPUs)", REEXEC] if path == policy.USER_CPUSET else []


def stamp(t: int) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime(policy.UNTIL_FORMAT)


def _gate(node: str) -> str:
    return policy._N2_GATE if node == nebius2.NODE else "set -f\n"


def exact(node: str, paths: list[str]) -> str:
    """The command that reads `paths` byte for byte (base64), after node 2's gate there."""
    q = " ".join(shlex.quote(p) for p in paths)
    return _gate(node) + (f'for f in {q}; do if [ -e "$f" ]; then echo "{B64_MARK}$f"; base64 -w0 "$f" || exit 1; echo; '
                          f'else echo "{policy.ABSENT_MARK}$f"; fi; done\n')


def read_exact(out: str) -> tuple[str | None, dict[str, bytes | None]]:
    """`exact`'s output: why node 2's gate refused (None when it didn't) and each file's bytes, None when absent."""
    first, _, rest = out.partition("\n")
    if first.startswith("refused "):
        return first[len("refused "):], {}
    body = rest if first.startswith("status ") else out
    files: dict[str, bytes | None] = {}
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(B64_MARK):
            files[line[len(B64_MARK):]] = base64.b64decode(lines[i + 1] if i + 1 < len(lines) else "")
        elif line.startswith(policy.ABSENT_MARK):
            files[line[len(policy.ABSENT_MARK):]] = None
    return None, files


def sha(b: bytes | None) -> str:
    return "absent" if b is None else hashlib.sha256(b).hexdigest()


def steps(node: str, rows: list[dict[str, Any]]) -> list[Step]:
    """Each row that isn't as in force, as a step: `change` when it can be applied (its edit is planned from the file itself)."""
    out = []
    for r in rows:
        if r["status"] == "ok":
            continue
        s = Step(r["key"], "skip", owner=r.get("owner") or "", value=r.get("declared"), live=r.get("live"))
        if r["status"] == "unknown":
            s.action, s.why = "unknown", "never applied while unread: " + " ".join((r.get("why") or "").split())[:200]
        elif r["status"] == "undeclared":
            s.why = "undeclared: no value in force to converge to; declare it, or remove it by hand"
        else:
            s.why = applier(r["key"]) or ""
            if not s.why:
                s.path = path_of(r["key"])
                hand = PLAN_ONLY.get((node, surface(r["key"])))
                s.action, s.why = ("by-hand", f"planned only: {hand}") if hand else ("change", r.get("why") or "")
        out.append(s)
    return out


def plan(node: str, todo: list[Step], files: dict[str, bytes | None], records: list[dict[str, Any]], now: int, by: str,
         reason: str) -> Plan:
    """The edits of `todo`'s changes from the files' exact bytes, one `apply` record per key continuing the node's log."""
    p = Plan(node, todo, log=files.get(policy.OVERRIDES))
    note = f"cluster policy apply, {by or '?'} {stamp(now)}"
    after: dict[str, str] = {}
    hand: dict[str, str] = {}
    for s in todo:
        if s.action not in ("change", "by-hand") or s.path is None:
            continue
        edited = after if s.action == "change" else hand
        before = files.get(s.path)
        text = edited.get(s.path, (before or b"").decode())
        try:
            new, was = EDITORS[EDITS[surface(s.key)]](text, s.key, s.value, note)
        except Skip as e:
            s.action, s.why = "skip", str(e)
            continue
        if new == text:
            s.action, s.why = "skip", f"already as in force in {s.path} (it changed since the first read)"
            continue
        s.live = was if was is not None else s.live
        edited[s.path] = new
    p.by_hand = {path: (files.get(path), text.encode()) for path, text in hand.items()}
    seq, prev = (records[-1]["seq"], records[-1]["hash"]) if records else (0, ledger.GENESIS)
    for path, text in after.items():
        p.files[path] = (files.get(path), text.encode())
        for s in todo:
            if s.action == "change" and s.path == path:
                seq += 1
                r = {"seq": seq, "t": now, "kind": "apply", "key": s.key, "value": s.value, "was": s.live, "by": by,
                     "reason": reason, "path": path, "before": sha(files.get(path)), "after": sha(text.encode()), "prev": prev}
                r["hash"] = prev = ledger.digest(r)
                p.records.append(r)
        p.then += [c for c in _post(path) if c not in p.then]
    return p


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def script(p: Plan) -> str:
    """The one command that writes `p` on its node: node 2's gate, then under the log's lock each file and the log checked
    against the bytes planned from, each file written and its keys' records appended, then the reloads, then APPLIED."""
    log = policy.OVERRIDES
    q = shlex.quote
    # a failure prints why and exits 0, so what was written before it still reaches the caller (ssh's runner drops stdout
    # on a nonzero exit)
    out = [_gate(p.node),
           'fail() { echo "failed $1"; exit 0; }\n',
           f"mkdir -p {q(str(Path(log).parent))} || fail 'to make the log directory'\n",
           f"exec 9>>{q(log + '.lock')} || fail 'to open the log lock'\nflock -x 9 || fail 'to lock the log'\n",
           "cur() { if [ -e \"$1\" ]; then sha256sum < \"$1\" | cut -d' ' -f1; else echo absent; fi; }\n"]
    for path, (before, _) in [*p.files.items(), (log, (p.log, b""))]:
        out.append(f'[ "$(cur {q(path)})" = {sha(before)} ] || {{ echo "changed {path}"; exit 0; }}\n')
    for path, (_, new) in p.files.items():
        tmp = q(path + ".policy-new")
        if path in ROOT_FILES:
            out.append(f"t=$(mktemp) || fail 'mktemp'\nprintf %s {_b64(new)} | base64 -d > \"$t\" || fail 'to stage {path}'\n"
                       f"sudo -n cp \"$t\" {tmp} && sudo -n chmod 644 {tmp} && sudo -n mv -f {tmp} {q(path)} "
                       f"|| {{ rm -f \"$t\"; fail 'to write {path}'; }}\nrm -f \"$t\"\n")
        else:
            out.append(f"t=$(mktemp {q(path + '.policy-XXXXXX')}) || fail 'mktemp beside {path}'\n"
                       f"printf %s {_b64(new)} | base64 -d > \"$t\" || fail 'to stage {path}'\n"
                       f"chmod --reference={q(path)} \"$t\" 2>/dev/null || chmod 664 \"$t\"\n"
                       f"mv -f \"$t\" {q(path)} || fail 'to write {path}'\n")
        lines = "".join(ledger.canonical(r).decode() + "\n" for r in p.records if r["path"] == path)
        out.append(f"printf %s {_b64(lines.encode())} | base64 -d >> {q(log)} || fail 'to log {path}'\n"
                   f'echo "wrote {path}"\n')
    if REEXEC in p.then:
        out.append("a=$(systemctl show -p AllowedCPUs --value user.slice) || fail 'to read user.slice'\n"
                   "sudo -n systemctl daemon-reload || fail 'systemctl daemon-reload'\n"
                   "b=$(systemctl show -p AllowedCPUs --value user.slice) || fail 'to read user.slice'\n"
                   '[ "$a" = "$b" ] || fail "user.slice AllowedCPUs moved from $a to $b: run sudo vy-device-cores"\n'
                   f"{REEXEC} || fail '{REEXEC}'\n")
    out.append(f'echo "{APPLIED}"\n')
    return "".join(out)


def _read_node(keys: dict[str, policy.Key], node: str, run: Callable[[str], str], now: int) -> tuple[list[dict[str, Any]], policy.Live]:
    raw = policy.read(node, run)
    try:
        lv = policy.live(node, raw, now)
        return policy.diff(policy.effective(keys, lv.overrides, now, node), node, lv, now), lv
    except Exception as e:  # noqa: BLE001  (as main_diff: output of any unexpected shape is unknown, never drift)
        lv = policy.Live({}, {node: f"its live policy couldn't be parsed: {type(e).__name__}: {e}"}, {})
        return policy.diff(keys, node, lv, now), lv


def _under(key: str, prefix: str | None) -> bool:
    return prefix is None or key == prefix or key.startswith(prefix + ".") or prefix.startswith(key + ".")


def node_plan(keys: dict[str, policy.Key], node: str, run: Callable[[str], str], now: int, prefix: str | None, by: str,
              reason: str) -> Plan:
    """Read `node`, and plan the change of each of its keys (under `prefix`) that differs."""
    rows, lv = _read_node(keys, node, run, now)
    todo = steps(node, [r for r in rows if _under(r["key"], prefix)])
    paths = sorted({s.path for s in todo if s.action in ("change", "by-hand") and s.path})
    if not paths:
        return Plan(node, todo)
    try:
        refused, files = read_exact(run(exact(node, [*paths, policy.OVERRIDES])))
        if refused is None:
            records = policy.read_overrides((files.get(policy.OVERRIDES) or b"").decode(), f"{node}:{policy.OVERRIDES}")
            if records != lv.overrides:
                refused = "its override log changed between the two reads: run it again"
            elif records and records[-1]["t"] > now:
                refused = f"its log's last record is at {records[-1]['t']}, after this clock's {now}"
    except Exception as e:  # noqa: BLE001  (an exact read that fails plans nothing)
        refused, files, records = f"the files to change couldn't be read exactly: {e}", {}, []
    if refused is not None:
        for s in todo:
            if s.action in ("change", "by-hand"):
                s.action, s.why = "unknown", f"never applied while unread: {refused}"
        return Plan(node, todo, refused=refused)
    return plan(node, todo, files, records, now, by, reason)


def apply(p: Plan, keys: dict[str, policy.Key], run: Callable[[str], str], now: int, prefix: str | None) -> dict[str, Any]:
    """Write `p` on its node and read the node again: `ok` when every changed key reads as in force; `left`, each key (under
    `prefix`) that still doesn't, with its status."""
    try:
        out = run(script(p))
    except Exception as e:  # noqa: BLE001  (a write that fails is reported, then the node is read again)
        out = f"failed: {e}"
    lines = out.strip().splitlines()
    wrote = [x[len("wrote "):] for x in lines if x.startswith("wrote ")]
    done = bool(lines) and lines[-1] == APPLIED
    stop = "" if done else next((x for x in lines if x.startswith(("refused ", "changed ", "failed"))), out.strip()[-300:])
    if not done and not wrote:
        return {"ok": False, "wrote": [], "why": f"nothing written: {stop}", "left": {s.key: s.action for s in p.steps}}
    rows, _ = _read_node(keys, p.node, run, now)
    left = {r["key"]: r["status"] for r in rows if r["status"] != "ok" and _under(r["key"], prefix)}
    why = ([] if done else [f"stopped after writing {', '.join(wrote)}: {stop}"]) + [
        f"{s.key} reads {left.get(s.key, 'absent')} after the write" for s in p.steps if s.action == "change" and s.key in left]
    return {"ok": not why, "wrote": wrote, "left": left, **({"why": "; ".join(why)} if why else {})}


def _diff(path: str, before: bytes | None, after: bytes) -> list[str]:
    return list(difflib.unified_diff((before or b"").decode().splitlines(), after.decode().splitlines(),
                                     f"{path} (live{'' if before is not None else ', absent'})", f"{path} (in force)", lineterm=""))


def render(plans: list[Plan], results: dict[str, dict[str, Any]], applying: bool) -> str:
    out = []
    for p in plans:
        n = {a: sum(1 for s in p.steps if s.action == a) for a in ("change", "by-hand", "skip", "unknown")}
        out.append(f"{p.node}: {n['change']} to change, {n['by-hand']} by hand, {n['skip']} skipped, {n['unknown']} unknown")
        for s in p.steps:
            what = (f"live {policy._fmt(s.live)} -> {policy._fmt(s.value)}" if s.action in ("change", "by-hand")
                    else f"declared {policy._fmt(s.value)}  live {policy._fmt(s.live)}")
            where = "; ".join(x for x in (s.path if s.action in ("change", "by-hand") else None,
                                          s.why if s.action != "change" else None) if x)
            out.append(f"  {s.action:<8} {s.key}  {what}  ({where + '; ' if where else ''}owner {s.owner or '?'})")
        for path, (before, after) in p.files.items():
            out += [f"    {x}" for x in _diff(path, before, after)]
        for path, (before, after) in p.by_hand.items():
            out += [f"    by hand: {x}" for x in _diff(path, before, after)]
        out += [f"  then: {c}" for c in p.then]
        if p.records:
            out.append(f"  and {len(p.records)} apply record(s) appended to {p.node}:{policy.OVERRIDES}")
        r = results.get(p.node)
        if r is not None:
            out.append(f"  applied and read back as in force: {', '.join(r['wrote'])}" if r["ok"] else f"  NOT APPLIED: {r['why']}")
    if not applying and any(p.files for p in plans):
        out.append("(a dry run: --apply --as WHO --reason WHY writes it)")
    return "\n".join(out) + "\n"


def main_apply(description: str | Path, which: str, prefix: str | None, applying: bool, by: str | None, reason: str | None,
               as_json: bool, key: Path | None = status.SSH_KEY,
               runner: Callable[[str], Callable[[str], str]] | None = None) -> int:
    """Plan (and with `applying` write) the change of every key that differs: 0 nothing left to change, 1 a change planned
    or a key skipped, 3 a key unknown, 4 an apply that failed or didn't read back as in force, 2 a refused command."""
    if applying and not ((by or "").strip() and (reason or "").strip()):
        print("cluster policy apply: refused: --apply needs --as (the handle who decided it) and --reason (why)")
        return 2
    d = policy._described(description)
    if d is None:
        return 2
    keys = d[1]
    now = int(policy.clock())
    runner = runner or (lambda node: status.ssh(policy.HOSTS[node], key))
    plans, results = [], {}
    for node in policy._nodes(which):
        run = runner(node)
        p = node_plan(keys, node, run, now, prefix, by or "", reason or "")
        plans.append(p)
        if applying and p.files:
            results[node] = apply(p, keys, run, now, prefix)
    if prefix is not None and not any(p.steps for p in plans) and not any(_under(k, prefix) for k in keys):
        print(f"cluster policy apply: no key {prefix}, nor any under it, on {which}")
        return 2
    if as_json:
        print(json.dumps({"schema": "cluster/policy-apply/v1", "t": now, "description": str(description), "applied": applying,
                          "nodes": [{"node": p.node, "steps": [asdict(s) for s in p.steps], "refused": p.refused,
                                     **{name: {f: {"before": sha(b), "after": sha(a), "diff": _diff(f, b, a)}
                                               for f, (b, a) in edits.items()}
                                        for name, edits in (("files", p.files), ("by_hand", p.by_hand))},
                                     "records": p.records, "then": p.then, "result": results.get(p.node)} for p in plans]},
                         indent=1, sort_keys=True))
    else:
        print(render(plans, results, applying), end="")
    if any(not r["ok"] for r in results.values()):
        return FAILED
    left = [st for p in plans for st in (results[p.node]["left"].values() if p.node in results
                                         else (s.action for s in p.steps))]
    if "unknown" in left:
        return 3
    return 1 if left else 0


__all__ = ["EDITS", "NO_APPLIER", "PLAN_ONLY", "Step", "Plan", "applier", "exact", "read_exact", "steps", "plan", "script",
           "node_plan", "apply", "render", "main_apply"]
