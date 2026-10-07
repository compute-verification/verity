"""`cluster policy`: the policy a cluster description declares for its nodes, and every key where a node's live value differs.

Policy here is what node services read from outside git today: node 1's lease pools from their tmux panes' environment,
Kueue's ClusterQueue quotas and stop policies, the Lean audit and merge-check slot files, `gpu-lease`'s direct-run GPU and CPU
maps, the cpuset delegation drop-in, and node 2's booked timed windows and CPU sets. The description declares each under
`[policy]` as a dotted key, `NODE.SURFACE.NAME[.FIELD]` (`vy-nebius-1.lease-pool.provers.max`), and every key has an owner,
the handle who decides a change: an `owner` in a table covers every key under it, unless a table below names its own.
`NODE.pools.NAME` comes from the node's own table (`pools`, the CPU pools the planner reserves), with the owner of
`[policy.NODE.pools]`, else the node's.

`show` prints the declared policy, each key with its owner and where its live value is read (`source`). `diff` reads every
key's live value over SSH, read-only (`COMMANDS`: tmux, ps and /proc reads, `kubectl get`, `cat` and `grep`), and reports
each key where it differs: `drift` (live differs, or is absent), `undeclared` (live has a key the description lacks) or
`unknown` (it couldn't be read; unknown is never no drift). Of a process's environment only the lease pool's own variables
leave the node (`ENV`'s, and any other `VY_POOL_*`). Node 2 is read in one command whose gate, on node 2's clock and before any
other read, reads only the fill runner's `status.txt` and `windows`: unless `status.txt` says `QUIET`, is at most
`STATUS_FRESH_S` old and not ahead of the clock, and no booked window is on or starts within `WINDOW_LEAD_S`, it reads nothing
else and every node 2 key is unknown.

Values are compared in one spelling per surface (`canon`): Kubernetes quantities by amount (`384Gi` is 412316860416), CPU
lists by their CPUs, a window by its start and minutes. A window that has ended is not compared. A ClusterQueue held
(`stopPolicy: Hold`) by a declared writer (`HOLDERS`: the quiet hour, the disk guard), each of which lists the queues it holds
in its state file, is as declared; held by anyone else, it is drift.

An override (`override`) replaces one key's value until an absolute UTC time, or until a later override of the same key, and
records who and why. Each node keeps its own keys' overrides in its own log (`OVERRIDES`), hash-chained as the allocation
ledger is (`ledger.Ledger`). `show` merges the overrides in force from a local log; `diff` reads each node's log along with
its live values and compares live with the policy in force, the description's value with any override applied. A log that
doesn't verify is refused (exit 3): the policy in force can't be told from it. Recording an override changes nothing on a
node: n1_lease.py takes its pool's `max`, `borrow` and `borrow-until` from `show` when `$VY_POOL_POLICY` names the command,
and every other value is still set by hand. Those three of such a loop are read from the `LIMITS` file it writes, and while it
says they came from anywhere but the policy, they are drift whatever their values.
"""
from __future__ import annotations

import fcntl
import json
import re
import shlex
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from . import cpuset, ledger, model, nebius1, nebius2, status

DESCRIPTION = Path(__file__).parents[2] / "descriptions" / "nebius.toml"
NODES = status.NODES
HOSTS = status.HOSTS
LEAN_SLOTS = "/workspace/research/locks/lean-slots"
CHECK_SLOTS = "/workspace/research/locks/slots"
DIRECT = {"direct-gpus": "/etc/vy/direct-gpus", "direct-cpus": "/etc/vy/direct-cpus"}
USER_CPUSET = "/etc/systemd/system/user@.service.d/vy-cpuset.conf"
DISPATCH_ENV = "/workspace/jobs/dispatch/dispatch.env"
FILL = "/workspace/pouw/fill"
#: node 2 is read only while its status.txt's first line says this (the quiet form dispatch.py and n2_commit.sh wait for),
#: it is at most STATUS_FRESH_S old by node 2's clock (fill_runner.py rewrites it every 10 s), and no booked window is on or
#: starts within WINDOW_LEAD_S (status.txt lags a window's start by up to a fill tick)
QUIET = "timed False, window waiting False"
STATUS_FRESH_S = 60
WINDOW_LEAD_S = 60
HOLDERS = {"vy-quiet-hour": "/var/lib/vy-quiet-hour/held", "vy-disk-guard": "/var/lib/vy-disk-guard/held"}
#: a lease pool's tmux session, where its n1_lease.py loop runs
SESSIONS = {"provers": "n1-lease", "circuits": "n1-lease-circuits"}
#: a lease pool's keys and the environment variables n1_lease.py reads them from
ENV = {"name": "VY_POOL_NAME", "dir": "GPU_LEASE_DIR", "queue": "VY_POOL_QUEUE", "local-queue": "VY_POOL_LOCAL_QUEUE",
       "priority": "VY_POOL_PRIORITY", "max": "VY_POOL_MAX", "idle-s": "VY_POOL_IDLE_S", "borrow": "VY_POOL_BORROW",
       "borrow-until": "VY_POOL_BORROW_UNTIL", "grace-s": "VY_LEASE_EXPIRY_GRACE_S"}
#: with this set, n1_lease.py takes these keys from the policy in force and writes the values it uses to LIMITS in its lease
#: directory, so that file, not its environment, is their live value
POLICY_VAR = "VY_POOL_POLICY"
POLICY_FIELDS = ("max", "borrow", "borrow-until")
#: of those, the numbers of GPUs: whole numbers, at most the node's GPUs
GPU_FIELDS = ("max", "borrow")
LIMITS = "n1-lease.limits"
KUEUE_RESOURCES = {"gpu": "nvidia.com/gpu", "cpu": "cpu", "memory": "memory"}
KUEUE_FIELDS = {"nominal": "nominalQuota", "borrowing": "borrowingLimit", "lending": "lendingLimit"}
#: each node's surfaces, and the command each is read with
SURFACES = {nebius1.NODE: {"lease-pool": "lease", "kueue": "kueue", "lean-slots": "files", "check-slots": "files",
                           "direct-gpus": "files", "direct-cpus": "files", "user-cpuset": "files", "pools": "files"},
            nebius2.NODE: {"fill": "n2", "check-slots": "n2", "direct-gpus": "n2", "direct-cpus": "n2", "user-cpuset": "n2"}}
POOL_SOURCES = {"provers": f"{DISPATCH_ENV} VY_PROVER_CPUS", "checks": f"{CHECK_SLOTS}: every slot's CPUs"}
UNTIL_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
#: on each node, the overrides of that node's keys
OVERRIDES = "/workspace/research/policy/overrides.jsonl"


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class Key:
    """A policy key and its value in force; `override` (by, reason, until, and `was`, the description's value) while an
    override replaces it."""
    key: str
    value: Any
    owner: str
    override: dict[str, Any] | None = None

    @property
    def node(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def surface(self) -> str:
        return self.key.split(".")[1]


def _flatten(prefix: str, table: dict[str, Any], owner: str | None, keys: dict[str, tuple[Any, str | None]],
             owners: dict[str, str]) -> None:
    owner = table.get("owner", owner)
    if owner is not None:
        owners[prefix] = owner
    for k, v in table.items():
        if k == "owner":
            continue
        if isinstance(v, dict):
            _flatten(f"{prefix}.{k}", v, owner, keys, owners)
        else:
            keys[f"{prefix}.{k}"] = (v, owner)


def _owner_of(key: str, owners: dict[str, str]) -> str | None:
    parts = key.split(".")
    return next((owners[p] for p in (".".join(parts[:i]) for i in range(len(parts), 0, -1)) if p in owners), None)


def load(path: str | Path = DESCRIPTION) -> dict[str, Key]:
    """The description's policy keys, the node tables' pools among them, checked (`problems`)."""
    return _load(path)[1]


def _load(path: str | Path) -> tuple[model.Cluster, dict[str, Key]]:
    d = tomllib.loads(Path(path).read_text())
    cluster = model.from_dict(d)
    flat: dict[str, tuple[Any, str | None]] = {}
    owners: dict[str, str] = {}
    for node, table in (d.get("policy") or {}).items():
        if not isinstance(table, dict):
            raise PolicyError(f"policy.{node} is not a table")
        _flatten(node, table, None, flat, owners)
    for name, n in cluster.nodes.items():
        for pool, cpus in n.pools.items():
            key = f"{name}.pools.{pool}"
            if key in flat:
                raise PolicyError(f"{key} is the node's own `pools`; the policy table declares only its owner")
            flat[key] = (cpuset.format(cpus), _owner_of(key, owners))
    keys = {k: Key(k, v, o or "") for k, (v, o) in sorted(flat.items())}
    errs = problems(cluster, keys)
    if errs:
        raise PolicyError("; ".join(errs))
    return cluster, keys


def problems(cluster: model.Cluster, keys: dict[str, Key]) -> list[str]:
    """Why these keys can't be read or compared: no owner, an unknown node or surface, a value of the wrong shape, or a
    workstream whose GPU quota (the planner's) the Kueue keys contradict."""
    out = []
    for k in keys.values():
        parts = k.key.split(".")
        if not k.owner:
            out.append(f"{k.key}: no owner")
        if k.node not in cluster.nodes:
            out.append(f"{k.key}: unknown node {k.node!r}")
            continue
        if len(parts) < 2 or k.surface not in SURFACES.get(k.node, {}):
            out.append(f"{k.key}: no reader for {'.'.join(parts[1:2]) or k.key!r} on {k.node}")
            continue
        if isinstance(k.value, (list, dict, bool)) or k.value is None:
            out.append(f"{k.key}: a policy value is a string or a number")
            continue
        try:
            canon(k.key, k.value)
        except (ValueError, KeyError) as e:
            out.append(f"{k.key}: {e}")
            continue
        s, rest = k.surface, parts[2:]
        if s == "lease-pool" and (len(rest) != 2 or rest[0] not in SESSIONS or rest[1] not in ENV):
            out.append(f"{k.key}: a lease pool key is lease-pool.POOL.FIELD, POOL one of {sorted(SESSIONS)}, FIELD one of "
                       f"{sorted(ENV)}")
        elif s == "lease-pool" and rest[1] in GPU_FIELDS and int(canon(k.key, k.value)) > len(cluster.nodes[k.node].gpus):
            out.append(f"{k.key}: {k.value} is more GPUs than {k.node} has ({len(cluster.nodes[k.node].gpus)})")
        elif s == "lease-pool" and rest[1] == "borrow-until":
            try:
                datetime.strptime(str(k.value), UNTIL_FORMAT)
            except ValueError:
                out.append(f"{k.key}: n1_lease.py reads only {UNTIL_FORMAT} (else it never stops borrowing)")
        elif s == "kueue" and not (len(rest) == 2 and rest[1] == "stop-policy"
                                   or len(rest) == 3 and rest[1] in KUEUE_RESOURCES and rest[2] in KUEUE_FIELDS):
            out.append(f"{k.key}: a Kueue key is kueue.QUEUE.stop-policy or kueue.QUEUE.RESOURCE.FIELD, RESOURCE one of "
                       f"{sorted(KUEUE_RESOURCES)}, FIELD one of {sorted(KUEUE_FIELDS)}")
        elif s in ("lean-slots", "check-slots") and len(rest) != 1:
            out.append(f"{k.key}: a slot key is {s}.NAME")
        elif s in DIRECT and rest:
            out.append(f"{k.key}: {s} is one value")
        elif s == "user-cpuset" and rest != ["delegate"]:
            out.append(f"{k.key}: the cpuset drop-in's only key is user-cpuset.delegate")
        elif s == "pools" and (len(rest) != 1 or rest[0] not in POOL_SOURCES):
            out.append(f"{k.key}: no live source for pool {'.'.join(rest)!r} (one of {sorted(POOL_SOURCES)})")
        elif s == "fill" and not (len(rest) == 2 and rest[0] in ("window", "cpu-sets")):
            out.append(f"{k.key}: a fill key is fill.window.START or fill.cpu-sets.OWNER")
        elif s == "fill" and rest[0] == "window":
            try:
                if _minute(_start(rest[1])) != rest[1]:
                    out.append(f"{k.key}: write the start as {_minute(_start(rest[1]))}")
            except ValueError:
                out.append(f"{k.key}: {rest[1]!r} is not a UTC start (2026-10-04T06:30Z)")
    for w in cluster.workstreams:
        for field, want in (("nominal", w.gpus), ("borrowing", w.borrow_gpus)):
            k = keys.get(f"{w.node}.kueue.{w.name}.gpu.{field}")
            if k is not None and want is not None and canon(k.key, k.value) != canon(k.key, want):
                out.append(f"{k.key} = {k.value}, but workstream {w.name} says {want} (one of them is stale)")
    return out


def _quantity(q: str) -> Fraction:
    units = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50, "Ei": 2**60, "k": 10**3, "M": 10**6, "G": 10**9,
             "T": 10**12, "P": 10**15, "E": 10**18, "m": Fraction(1, 1000)}
    for u in sorted(units, key=len, reverse=True):
        if q.endswith(u) and q[:-len(u)]:
            return Fraction(q[:-len(u)]) * units[u]
    return Fraction(q)


def _start(s: str) -> int:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int((t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp())


def _minute(t: int) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%MZ" if t % 60 == 0 else "%Y-%m-%dT%H:%M:%SZ")


def canon(key: str, value: Any) -> str:
    """`value` in the one spelling `diff` compares for `key`'s surface; ValueError when it isn't that surface's shape."""
    parts = key.split(".")
    s, v = parts[1], str(value).strip()
    if s == "kueue" and parts[-1] in KUEUE_FIELDS:
        return str(_quantity(v))
    if s in ("pools", "direct-cpus"):
        return cpuset.format(cpuset.parse(v))
    if s == "check-slots":
        cpus, *opts = v.split()
        return " ".join([cpuset.format(cpuset.parse(cpus)), *sorted(opts)])
    if s == "lean-slots" or s == "lease-pool" and parts[-1] in GPU_FIELDS:
        if not (v.isascii() and v.isdigit()):
            raise ValueError(f"{v!r} is not a whole number")
        return str(int(v))
    if s == "fill" and parts[2] == "window":
        m = float(v)
        if not 0 < m < float("inf"):
            raise ValueError(f"{v!r} is not a number of minutes")
        return str(int(m)) if m.is_integer() else repr(m)
    if s == "fill" and parts[2] == "cpu-sets":
        cpus, slots, *mem = v.split()
        if not slots.isdigit() or len(mem) > 1 or (mem and not mem[0].isdigit()):
            raise ValueError(f"{v!r} is not `CPUS SLOTS [MEM_GB]`")
        return " ".join([cpuset.format(cpuset.parse(cpus)), slots, *mem])
    if s == "user-cpuset":
        return " ".join(v.split())
    return v


def window_ends(key: str, value: Any) -> int | None:
    """When a `fill.window.START` key's window ends, in epoch seconds; None for any other key."""
    parts = key.split(".", 3)
    if len(parts) < 4 or parts[1:3] != ["fill", "window"]:
        return None
    return _start(parts[3]) + int(60 * float(value))


def source(key: str) -> str:
    """Where `diff` reads `key`'s live value on its node."""
    parts = key.split(".")
    s, rest = parts[1], parts[2:]
    if s == "lease-pool":
        return (f"tmux {SESSIONS[rest[0]]}: n1_lease.py loop's {ENV[rest[1]]}"
                + (f", or its {LIMITS} under {POLICY_VAR}" if rest[1] in POLICY_FIELDS else "")) if rest[1] in ENV \
            else f"tmux {SESSIONS[rest[0]]}"
    if s == "kueue":
        return (f"ClusterQueue {rest[0]} spec.stopPolicy" if rest[1] == "stop-policy"
                else f"ClusterQueue {rest[0]} {KUEUE_RESOURCES.get(rest[1], rest[1])} {KUEUE_FIELDS.get(rest[2], rest[2])}")
    if s in DIRECT:
        return DIRECT[s]
    return {"lean-slots": LEAN_SLOTS, "check-slots": CHECK_SLOTS, "user-cpuset": f"{USER_CPUSET} Delegate=",
            "pools": POOL_SOURCES.get(rest[0], "") if rest else "",
            "fill": f"{FILL}/{'windows' if rest[:1] == ['window'] else 'cpu-sets'}"}.get(s, "")


def read_overrides(text: str, where: str) -> list[dict[str, Any]]:
    """An override log's records, the whole chain verified; PolicyError on a break."""
    try:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except ValueError as e:
        raise PolicyError(f"{where}: a record isn't JSON ({e})") from None
    errs = ledger.verify(records)
    if errs:
        raise PolicyError(f"{where}: {errs[0]}")
    return records


def overrides(path: str | Path | None) -> list[dict[str, Any]]:
    """The records of the override log at `path`; none when there is no log."""
    p = Path(path) if path else None
    return read_overrides(p.read_text(), str(p)) if p is not None and p.exists() else []


def effective(keys: dict[str, Key], records: list[dict[str, Any]], now: int, node: str | None = None) -> dict[str, Key]:
    """`keys` with every override in force at `now` applied (of `node`'s keys alone, when given): each key's latest override,
    until its `until`. An override may add a key the description lacks (a window booked ahead)."""
    latest = {r["key"]: r for r in records if r.get("kind") == "override" and (node is None or r["key"].split(".")[0] == node)}
    out = dict(keys)
    for k, r in latest.items():
        try:
            ended = ledger.epoch(r["until"]) <= now
        except (KeyError, ValueError):
            raise PolicyError(f"override record {r.get('seq')} of {k}: until {r.get('until')!r} isn't {UNTIL_FORMAT}") from None
        if ended:
            continue
        d = keys.get(k)
        out[k] = Key(k, r["value"], d.owner if d else _owner_guess(k, keys),
                     {"by": r.get("by"), "reason": r.get("reason"), "until": r["until"], "was": d.value if d else None})
    return dict(sorted(out.items()))


def override_problems(cluster: model.Cluster, keys: dict[str, Key], key: str, value: Any, until: str, by: str, reason: str,
                      now: int) -> list[str]:
    """Why an override can't be recorded: a key no node's reader knows or a value of the wrong shape (an override may differ
    from a workstream's quota: that is what it is for), a node pool (the planner reads those from the description), an
    `until` that isn't an absolute UTC time still to come, or no `by` or `reason`."""
    out = []
    if key.split(".")[1:2] == ["pools"]:
        out.append(f"{key}: a node's pools are its own `pools`, which the planner reads from the description: change them there")
    owner = keys[key].owner if key in keys else _owner_guess(key, keys) or "?"
    out += problems(replace(cluster, workstreams=()), {key: Key(key, value, owner)})
    try:
        if ledger.epoch(until) <= now:
            out.append(f"--until {until} has passed")
    except ValueError:
        out.append(f"--until {until!r}: an absolute UTC time, {UNTIL_FORMAT} (2026-10-04T10:30:00Z)")
    if not by.strip():
        out.append("--by: the handle who decided it")
    if not reason.strip():
        out.append("--reason: why")
    return out


def _value(v: str) -> int | str:
    return int(v) if v.isascii() and v.isdigit() else v


FILE_MARK = "@@policy-file "
ABSENT_MARK = "@@policy-absent "


def _dump(paths: list[str]) -> str:
    """Each file's contents after a mark line naming it, or a mark that it is absent; fails when one can't be read."""
    q = " ".join(shlex.quote(p) for p in paths)
    return (f'for f in {q}; do if [ -e "$f" ]; then echo "{FILE_MARK}$f"; cat "$f" || exit 1; echo; '
            f'else echo "{ABSENT_MARK}$f"; fi; done')


def _files(text: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    path = None
    for line in text.splitlines():
        if line.startswith(FILE_MARK):
            path = line[len(FILE_MARK):]
            out[path] = ""
        elif line.startswith(ABSENT_MARK):
            out[line[len(ABSENT_MARK):]] = None
            path = None
        elif path is not None:
            out[path] += line + "\n"
    return out


_ENV_RE = "^(" + "|".join(sorted({v for v in ENV.values() if not v.startswith("VY_POOL_")} | {"VY_POOL_[A-Z0-9_]+"})) + ")="
# the loop is the python process itself, not the pane's `bash -c` whose command line names it too; its environment is matched
# a variable at a time (`grep -z`) and each one's newlines printed as spaces, so no part of another variable's value can pass
_LEASE = (
    "set -f\n"
    'kids() { for c in $(ps --ppid "$1" -o pid=); do echo "$c"; kids "$c"; done; }\n'
    "panes=$(tmux list-panes -a -F '#{session_name} #{pane_pid}') || exit 1\n"
    'echo "$panes" | while read -r s p; do\n'
    f'  case " {" ".join(sorted(SESSIONS.values()))} " in *" $s "*) ;; *) continue ;; esac\n'
    '  echo "session $s"\n'
    '  for c in $p $(kids "$p"); do\n'
    '    set -- $(tr "\\0" " " < /proc/$c/cmdline 2>/dev/null)\n'
    '    case "${1##*/} ${2##*/} ${3:-}" in python*" n1_lease.py loop") ;; *) continue ;; esac\n'
    "    e=/proc/$c/environ\n"
    '    grep -zq "" "$e" 2>/dev/null; [ $? -le 1 ] || { echo "unreadable $s $c"; continue; }\n'
    '    echo "loop $s $c"\n'
    f"    grep -zE '{_ENV_RE}' \"$e\" | tr '\\n\\0' ' \\n'\n"
    f"    grep -zq '^{POLICY_VAR}=.' \"$e\" || continue\n"
    "    d=$(grep -z '^GPU_LEASE_DIR=' \"$e\" | tr '\\n\\0' ' \\n' | cut -d= -f2-)\n"
    f'    if l=$(cat "${{d:-/run/gpu-lease}}/{LIMITS}" 2>/dev/null); then echo "limits $l"; else echo "limits"; fi\n'
    "  done\n"
    "done\n")
# only VY_PROVER_CPUS of the dispatcher's env file; grep's 1 is no such line, anything above an error
_DISPATCH = (f'f={shlex.quote(DISPATCH_ENV)}; if [ -e "$f" ]; then echo "{FILE_MARK}$f"; '
             f'grep -E "^VY_PROVER_CPUS=" "$f" || [ $? -eq 1 ]; else echo "{ABSENT_MARK}$f"; fi')

# node 2's gate, before any other read there and on its own clock: unless status.txt is QUIET and fresh and no booked window is
# near (a windows line that doesn't parse counts as one), it prints `refused WHY: STATUS` and reads nothing else
_N2_GATE = (
    "set -f\n"
    f"s=$(head -1 {FILL}/status.txt) || exit 1\n"
    "now=$(date +%s) || exit 1\n"
    'no() { printf "refused %s: %s\\n" "$1" "$s"; exit 0; }\n'
    f'case "$s" in *"{QUIET}"*) ;; *) no "a timed window is on or waiting" ;; esac\n'
    't=$(date -d "${s%% *}" +%s 2>/dev/null) || no "status.txt does not say when it was written"\n'
    "[ \"$t\" -le \"$now\" ] || no \"status.txt is stamped $((t - now)) s ahead of node 2's clock\"\n"
    f'[ $((now - t)) -le {STATUS_FRESH_S} ] || no "status.txt is $((now - t)) s old: is the fill runner running?"\n'
    f"w={FILL}/windows\n"
    'if [ -e "$w" ]; then\n'
    '  while IFS= read -r x || [ -n "$x" ]; do\n'
    "    set -- ${x%%#*}\n"
    "    [ $# -gt 0 ] || continue\n"
    "    m=${2:-30}\n"
    "    b=$(date -d \"$1\" +%s 2>/dev/null) || no \"windows line '$x' does not parse\"\n"
    "    case \"$m\" in *[!0-9.]*|*.*.*|.) no \"windows line '$x' does not parse\" ;; esac\n"
    f"    if awk -v n=\"$now\" -v b=\"$b\" -v m=\"$m\" 'BEGIN {{ exit !(b - {WINDOW_LEAD_S} <= n && n < b + 60 * m) }}'; then\n"
    f'      no "window $1 ($m min) is on or starts within {WINDOW_LEAD_S} s"\n'
    "    fi\n"
    '  done < "$w" || exit 1\n'
    "fi\n"
    'printf "status %s\\n" "$s"\n')

COMMANDS = {
    "lease": _LEASE,
    "kueue": shlex.join([*nebius1.KUBECTL, "get", "clusterqueues", "-o", "json"]),
    "holds": _dump(sorted(HOLDERS.values())),
    "files": _dump([LEAN_SLOTS, CHECK_SLOTS, *DIRECT.values(), USER_CPUSET]) + " && " + _DISPATCH,
    "overrides": _dump([OVERRIDES]),
    "n2": _N2_GATE + _dump([f"{FILL}/windows", f"{FILL}/cpu-sets", CHECK_SLOTS, *DIRECT.values(), USER_CPUSET, OVERRIDES]),
}
READS = {nebius1.NODE: ("lease", "kueue", "holds", "files", "overrides"), nebius2.NODE: ("n2",)}


def read(node: str, run: Callable[[str], str]) -> dict[str, str | Exception]:
    """Each of the node's `READS`, run at once: its output, or the error it failed with."""
    def one(name: str) -> str | Exception:
        try:
            return run(COMMANDS[name])
        except Exception as e:  # noqa: BLE001  (an unread command makes its keys unknown, never no drift)
            return e
    with ThreadPoolExecutor(len(READS[node])) as ex:
        return dict(zip(READS[node], ex.map(one, READS[node])))


@dataclass
class Live:
    """A node's live policy: each key's value as read, the prefixes that couldn't be read (with why), notes on keys, faults
    (keys whose live value is drift whatever it is: the value in force came from the wrong place), and the records of the
    node's override log."""
    values: dict[str, str]
    unknown: dict[str, str]
    notes: dict[str, str]
    overrides: list[dict[str, Any]] = field(default_factory=list)
    faults: dict[str, str] = field(default_factory=dict)


def _read_overrides(node: str, files: dict[str, str | None], live: Live) -> None:
    try:
        live.overrides = read_overrides(files.get(OVERRIDES) or "", f"{node}:{OVERRIDES}")
    except PolicyError as e:
        live.unknown[node] = f"its override log doesn't verify, so the policy in force can't be told: {e}"


def _slots(text: str) -> list[list[str]]:
    return [f for f in (line.split("#")[0].split() for line in text.splitlines()) if f]


def _read_files(node: str, files: dict[str, str | None], live: Live, *, pools: bool) -> None:
    v = live.values
    if files.get(LEAN_SLOTS) is not None:
        for f in _slots(files[LEAN_SLOTS]):
            v[f"{node}.lean-slots.{f[0]}"] = " ".join(f[1:])
    checks: set[int] = set()
    if files.get(CHECK_SLOTS) is not None:
        for f in _slots(files[CHECK_SLOTS]):
            v[f"{node}.check-slots.{f[0]}"] = " ".join(f[1:])
            if len(f) > 1:
                try:
                    checks |= cpuset.parse(f[1])
                except cpuset.CpuSetError:
                    live.unknown[f"{node}.pools.checks"] = f"{CHECK_SLOTS}: slot {f[0]}'s CPUs {f[1]!r} don't parse"
    if pools:
        v[f"{node}.pools.checks"] = cpuset.format(checks)
        env = files.get(DISPATCH_ENV) or ""
        cpus = [line.split("=", 1)[1].strip() for line in env.splitlines() if line.startswith("VY_PROVER_CPUS=")]
        if cpus:
            v[f"{node}.pools.provers"] = cpus[-1]
    for s, p in DIRECT.items():
        text = files.get(p)
        lines = [x for x in (line.split("#")[0].strip() for line in (text or "").splitlines()) if x]
        if text is not None:
            v[f"{node}.{s}"] = lines[0] if lines else ""
    text = files.get(USER_CPUSET)
    if text is not None:
        d = [line.split("=", 1)[1] for line in text.splitlines() if line.strip().startswith("Delegate=")]
        v[f"{node}.user-cpuset.delegate"] = d[-1].strip() if d else ""


def _read_lease(node: str, text: str, live: Live) -> None:
    sessions: set[str] = set()
    unreadable: set[str] = set()
    loops: dict[str, list[dict[str, str]]] = {}
    cur: dict[str, str] | None = None
    for line in text.splitlines():
        f = line.split()
        if f[:1] == ["session"] and len(f) == 2:
            sessions.add(f[1])
            cur = None
        elif f[:1] == ["unreadable"] and len(f) == 3:
            unreadable.add(f[1])
            cur = None
        elif f[:1] == ["loop"] and len(f) == 3:
            cur = {}
            loops.setdefault(f[1], []).append(cur)
        elif f[:1] == ["limits"] and cur is not None:
            cur[f"@{LIMITS}"] = line[len("limits"):].strip()
        elif cur is not None and "=" in line:
            k, _, val = line.partition("=")
            cur[k] = val
    var_key = {var: k for k, var in ENV.items()}
    for pool, sess in SESSIONS.items():
        prefix = f"{node}.lease-pool.{pool}"
        envs = loops.get(sess, [])
        if sess not in sessions:
            live.unknown[prefix] = f"no tmux session {sess}"
        elif sess in unreadable:
            live.unknown[prefix] = f"the environment of tmux session {sess}'s n1_lease.py loop can't be read"
        elif not envs:
            live.unknown[prefix] = f"no n1_lease.py loop in tmux session {sess}"
        elif any(e != envs[0] for e in envs):
            live.unknown[prefix] = f"{len(envs)} n1_lease.py loops in tmux session {sess}, with different settings"
        else:
            env = dict(envs[0])
            limits = env.pop(f"@{LIMITS}", None)
            if env.pop(POLICY_VAR, ""):
                for var in (ENV[f] for f in POLICY_FIELDS):
                    env.pop(var, None)
                _read_limits(prefix, limits, live)
            for var, val in env.items():
                live.values[f"{prefix}.{var_key.get(var, var)}"] = val


def _read_limits(prefix: str, text: str | None, live: Live) -> None:
    """A loop under $VY_POOL_POLICY: the max, borrow and borrow-until it last wrote to its LIMITS file, those in force. An
    empty borrow-until is none, as in its environment."""
    got = dict(w.partition("=")[::2] for w in (text or "").split())
    missing = [f for f in (*POLICY_FIELDS, "from") if f not in got]
    for f in POLICY_FIELDS:
        key = f"{prefix}.{f}"
        if missing:
            live.unknown[key] = (f"n1_lease.py runs under ${POLICY_VAR} and its {LIMITS} "
                                 + (f"lacks {', '.join(missing)}" if text else "can't be read"))
            continue
        if got[f]:
            live.values[key] = got[f]
        if got["from"] != "policy":
            live.faults[key] = (f"n1_lease.py can't read ${POLICY_VAR}, so "
                                + ("the values it last read there are" if got["from"] == "last-policy"
                                   else "its environment's values are") + " in force")


def _read_kueue(node: str, raw: str | Exception, holds: str | Exception, live: Live) -> None:
    prefix = f"{node}.kueue"
    if isinstance(raw, Exception):
        live.unknown[prefix] = f"kubectl get clusterqueues failed: {raw}"
        return
    res_key = {v: k for k, v in KUEUE_RESOURCES.items()}
    held: dict[str, list[str]] | None = None
    if not isinstance(holds, Exception):
        files = _files(holds)
        held = {}
        for who, path in HOLDERS.items():
            for q in (files.get(path) or "").split():
                held.setdefault(q, []).append(who)
    for q in json.loads(raw)["items"]:
        name, spec = q["metadata"]["name"], q.get("spec") or {}
        stop = spec.get("stopPolicy") or "None"
        live.values[f"{prefix}.{name}.stop-policy"] = stop
        if stop != "None":
            key = f"{prefix}.{name}.stop-policy"
            if held is None:
                live.unknown[key] = f"held ({stop}), and the holders' state files can't be read: {holds}"
            elif held.get(name):
                live.notes[key] = f"held by {', '.join(held[name])}"
        seen: dict[str, int] = {}
        for rg in spec.get("resourceGroups") or ():
            for fl in rg.get("flavors") or ():
                for r in fl.get("resources") or ():
                    res = res_key.get(r["name"], r["name"])
                    seen[res] = seen.get(res, 0) + 1
                    for field, k8s in KUEUE_FIELDS.items():
                        if k8s in r:
                            live.values[f"{prefix}.{name}.{res}.{field}"] = str(r[k8s])
        for res, n in seen.items():
            if n > 1:
                live.unknown[f"{prefix}.{name}.{res}"] = f"{n} flavors give {res}: the description declares one"


def _read_n2(node: str, raw: str, now: int, live: Live) -> None:
    first, _, rest = raw.partition("\n")
    kind, _, line = first.partition(" ")
    if kind == "refused":
        live.unknown[node] = f"node 2 refused to be read, and nothing else there was: {line[:240]}"
        return
    if kind != "status":
        live.unknown[node] = f"node 2's gate printed {first[:80]!r}, neither a status nor a refusal"
        return
    files = _files(rest)
    # a line fill_runner.py skips is no window and no CPU set, so it is skipped here too
    for f in _slots(files.get(f"{FILL}/windows") or ""):
        try:
            start, minutes = _start(f[0]), float(f[1]) if len(f) > 1 else 30.0
        except ValueError:
            continue
        if start + 60 * minutes > now:
            key = f"{node}.fill.window.{_minute(start)}"
            live.values[key] = canon(key, minutes)
    for f in _slots(files.get(f"{FILL}/cpu-sets") or ""):
        if len(f) >= 3 and re.fullmatch(r"[\d,-]+", f[1]) and f[2].isdigit() and (len(f) < 4 or f[3].isdigit()):
            live.values[f"{node}.fill.cpu-sets.{f[0]}"] = " ".join(f[1:4])
    _read_files(node, files, live, pools=False)
    _read_overrides(node, files, live)


def live(node: str, raw: dict[str, str | Exception], now: int) -> Live:
    """The node's live policy from `read`'s outputs, at `now`."""
    out = Live({}, {}, {})
    if node == nebius2.NODE:
        if isinstance(raw["n2"], Exception):
            out.unknown[node] = f"node 2 couldn't be read: {raw['n2']}"
        else:
            _read_n2(node, raw["n2"], now, out)
        return out
    if isinstance(raw["lease"], Exception):
        out.unknown[f"{node}.lease-pool"] = f"the lease pools' tmux panes couldn't be read: {raw['lease']}"
    else:
        _read_lease(node, raw["lease"], out)
    _read_kueue(node, raw["kueue"], raw["holds"], out)
    if isinstance(raw["files"], Exception):
        for s, cmd in SURFACES[node].items():
            if cmd == "files":
                out.unknown[f"{node}.{s}"] = f"the policy files couldn't be read: {raw['files']}"
    else:
        _read_files(node, _files(raw["files"]), out, pools=True)
    if isinstance(raw["overrides"], Exception):
        out.unknown[node] = f"its override log couldn't be read, so the policy in force can't be told: {raw['overrides']}"
    else:
        _read_overrides(node, _files(raw["overrides"]), out)
    return out


def _unknown(key: str, unknown: dict[str, str]) -> str | None:
    parts = key.split(".")
    return next((unknown[p] for p in (".".join(parts[:i]) for i in range(len(parts), 0, -1)) if p in unknown), None)


def diff(keys: dict[str, Key], node: str, lv: Live, now: int) -> list[dict[str, Any]]:
    """One row per declared key of `node`, and one per live key it doesn't declare: `ok`, `drift`, `undeclared` or
    `unknown`. A window that has ended on both sides is left out."""
    rows = []
    mine = {k: v for k, v in keys.items() if v.node == node}
    for k, d in mine.items():
        row = {"key": k, "owner": d.owner, "declared": d.value, "live": lv.values.get(k), "source": source(k),
               **({"override": d.override} if d.override else {})}
        why = _unknown(k, lv.unknown)
        ends = window_ends(k, d.value)
        if ends is not None and ends <= now and k not in lv.values:
            continue
        if why is not None:
            row.update(status="unknown", why=why)
        elif k not in lv.values:
            row.update(status="drift", why="absent live")
        else:
            try:
                same, bad = canon(k, d.value) == canon(k, lv.values[k]), None
            except (ValueError, IndexError) as e:
                same, bad = False, f"the live value doesn't parse: {e}"
            if same and k not in lv.faults:
                row["status"] = "ok"
            elif k.endswith(".stop-policy") and k in lv.notes and canon(k, d.value) == "None":
                row.update(status="ok", why=lv.notes[k])
            else:
                row.update(status="drift", why=bad or lv.faults.get(k) or lv.notes.get(k) or (
                    "held by no declared writer" if k.endswith(".stop-policy") else "live differs"))
        rows.append(row)
    for k, val in lv.values.items():
        if k in mine or _unknown(k, lv.unknown) is not None:
            continue
        rows.append({"key": k, "owner": _owner_guess(k, keys), "declared": None, "live": val, "source": _source_or(k),
                     "status": "undeclared", "why": "live, and not in the description"})
    for p, why in lv.unknown.items():
        if not any(r["key"] == p or r["key"].startswith(p + ".") for r in rows):
            rows.append({"key": p, "owner": _owner_guess(p, keys), "declared": None, "live": None, "source": "",
                         "status": "unknown", "why": why})
    return sorted(rows, key=lambda r: r["key"])


def _owner_guess(key: str, keys: dict[str, Key]) -> str:
    """The owner of the declared key sharing the longest prefix with `key`: who to ask about an undeclared one."""
    parts = key.split(".")
    for i in range(len(parts), 0, -1):
        p = ".".join(parts[:i])
        for k in keys.values():
            if k.key == p or k.key.startswith(p + "."):
                return k.owner
    return ""


def _source_or(key: str) -> str:
    try:
        return source(key)
    except (IndexError, KeyError):
        return ""


def _nodes(which: str) -> list[str]:
    return list(NODES.values()) if which == "all" else [NODES[which]]


def show(keys: dict[str, Key], nodes: list[str], now: int, prefix: str | None = None) -> list[dict[str, Any]]:
    """The keys of `nodes` (only `prefix` and the keys under it, when given) with their owners and sources."""
    rows = []
    for k in keys.values():
        if k.node not in nodes or prefix is not None and not (k.key == prefix or k.key.startswith(prefix + ".")):
            continue
        ends = window_ends(k.key, k.value)
        rows.append({"key": k.key, "value": k.value, "owner": k.owner, "source": source(k.key),
                     **({"ended": True} if ends is not None and ends <= now else {}),
                     **({"override": k.override} if k.override else {})})
    return rows


def _fmt(v: Any) -> str:
    return "-" if v is None else str(v)


def _overridden(o: dict[str, Any]) -> str:
    return f"override by {o['by']} until {o['until']}, was {_fmt(o['was'])}: {o['reason']}"


def render_show(rows: list[dict[str, Any]]) -> str:
    cells = [("KEY", "VALUE", "OWNER", "SOURCE")] + [
        (r["key"], _fmt(r["value"]) + (" (ended)" if r.get("ended") else ""), r["owner"],
         r["source"] + (f"  ({_overridden(r['override'])})" if r.get("override") else "")) for r in rows]
    w = [max(len(c[i]) for c in cells) for i in range(3)]
    return "".join("  ".join(c[i].ljust(w[i]) for i in range(3)) + "  " + c[3] + "\n" for c in cells)


def render_diff(reports: list[dict[str, Any]]) -> str:
    out = []
    for rep in reports:
        n = {s: sum(1 for r in rep["keys"] if r["status"] == s) for s in ("ok", "drift", "undeclared", "unknown")}
        o = sum(1 for r in rep["keys"] if r.get("override"))
        out.append(f"{rep['node']}: {n['ok']} as declared, {n['drift']} drift, {n['undeclared']} undeclared, {n['unknown']} unknown"
                   + (f"; {o} under an override" if o else ""))
        for r in rep["keys"]:
            if r["status"] == "ok" and not r.get("override"):
                continue
            why = "; ".join(x for x in (r.get("why"), r.get("override") and _overridden(r["override"])) if x)
            out.append(f"  {r['status']:<10} {r['key']}  declared {_fmt(r['declared'])}  live {_fmt(r['live'])}"
                       f"  ({why + '; ' if why else ''}owner {r['owner'] or '?'}{'; ' + r['source'] if r['source'] else ''})")
    return "\n".join(out) + "\n"


clock = time.time


def _described(description: str | Path) -> tuple[model.Cluster, dict[str, Key]] | None:
    try:
        return _load(description)
    except (PolicyError, model.ModelError, OSError, tomllib.TOMLDecodeError) as e:
        print(f"cluster policy: {description}: {e}")
        return None


def main_show(description: str | Path, which: str, as_json: bool, prefix: str | None = None,
              log: str | Path | None = OVERRIDES) -> int:
    """Print the policy in force: the description's keys with the overrides in force from the local `log`. 2 for an unusable
    description or a `prefix` naming no key, 3 for a log that doesn't verify."""
    d = _described(description)
    if d is None:
        return 2
    now = int(clock())
    try:
        records = overrides(log)
    except (PolicyError, OSError) as e:
        print(f"cluster policy show: refused, the policy in force can't be told: {e}")
        return 3
    rows = show(effective(d[1], records, now), _nodes(which), now, prefix)
    if prefix is not None and not rows:
        print(f"cluster policy show: no key {prefix}, nor any under it, on {which}")
        return 2
    if as_json:
        print(json.dumps({"schema": "cluster/policy/v1", "t": now, "description": str(description),
                          "overrides": str(log) if log and Path(log).exists() else None, "keys": rows}, indent=1, sort_keys=True))
    else:
        print(render_show(rows), end="")
    return 0


def main_override(description: str | Path, key: str, value: str, until: str, by: str, reason: str, log: str | Path,
                  apply: bool, as_json: bool) -> int:
    """Record that `key` is `value` until `until` (`--apply`; else print what would be recorded) in the override log `log`,
    which must be the key's node's own. 2 for a refused override, 3 for a log that doesn't verify."""
    d = _described(description)
    if d is None:
        return 2
    cluster, keys = d
    now = int(clock())
    v = _value(value)
    errs = override_problems(cluster, keys, key, v, until, by, reason, now)
    if errs:
        print("cluster policy override: refused:\n" + "".join(f"  {e}\n" for e in errs), end="")
        return 2
    path = Path(log)
    try:
        before = effective(keys, overrides(path), now).get(key)
    except (PolicyError, OSError) as e:
        print(f"cluster policy override: refused, the log doesn't verify: {e}")
        return 3
    record: dict[str, Any] = {"key": key, "value": v, "until": until, "by": by, "reason": reason}
    if apply:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path.with_name(path.name + ".lock"), "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                record = ledger.Ledger(path).append(now, "override", **record)
        except (ValueError, OSError) as e:
            print(f"cluster policy override: refused: {e}")
            return 3
    was = _fmt(before.value if before else None) + (f" (override by {before.override['by']})" if before and before.override else "")
    if as_json:
        print(json.dumps({"schema": "cluster/policy-override/v1", "t": now, "log": str(path), "applied": apply,
                          "before": before.value if before else None, "record": record}, indent=1, sort_keys=True))
    else:
        print(f"{'overrode' if apply else 'would override'} {key}: {was} -> {v} until {until}, by {by}: {reason}"
              f"{'' if apply else f' (in {path}; --apply records it)'}\n"
              "Nothing carries it to the node but n1_lease.py, for its pool's max, borrow and borrow-until while $VY_POOL_POLICY "
              "is set: set any other value yourself. `cluster policy diff` compares the live value with it.")
    return 0


def main_diff(description: str | Path, which: str, as_json: bool, key: Path | None = status.SSH_KEY,
              runner: Callable[[str], Callable[[str], str]] | None = None) -> int:
    """Read `which` nodes' live policy at once and print every key where it differs: 0 none, 1 drift (or undeclared), 3 a
    value couldn't be read (whatever else was found), 2 an unusable description."""
    d = _described(description)
    if d is None:
        return 2
    keys = d[1]
    now = int(clock())
    runner = runner or (lambda node: status.ssh(HOSTS[node], key))
    nodes = _nodes(which)
    with ThreadPoolExecutor(len(nodes)) as ex:
        raws = list(ex.map(lambda n: read(n, runner(n)), nodes))
    reports = []
    for node, raw in zip(nodes, raws):
        try:
            lv = live(node, raw, now)
            rows = diff(effective(keys, lv.overrides, now, node), node, lv, now)
        except Exception as e:  # noqa: BLE001  (output of any unexpected shape is unknown, exit 3, never drift's 1)
            rows = diff(keys, node, Live({}, {node: f"its live policy couldn't be parsed: {type(e).__name__}: {e}"}, {}), now)
        reports.append({"node": node, "keys": rows})
    found = [r for rep in reports for r in rep["keys"] if r["status"] != "ok"]
    if as_json:
        print(json.dumps({"schema": "cluster/policy-diff/v1", "t": now, "description": str(description), "nodes": reports},
                         indent=1, sort_keys=True))
    else:
        print(render_diff(reports), end="")
    if any(r["status"] == "unknown" for r in found):
        return 3
    return 1 if found else 0


__all__ = ["DESCRIPTION", "OVERRIDES", "COMMANDS", "READS", "SESSIONS", "ENV", "HOLDERS", "Key", "Live", "PolicyError", "load",
           "problems", "canon", "source", "window_ends", "read_overrides", "overrides", "effective", "override_problems", "read",
           "live", "diff", "show", "render_show", "render_diff", "main_show", "main_diff", "main_override"]
