"""`cluster policy`: the policy a cluster description declares for its nodes, and every key where a node's live value differs.

Policy here is what node services read from outside git today: the merge-check slot files, `gpu-lease`'s direct-run GPU and
CPU maps, the cpuset delegation drop-in, and node 2's booked timed windows and CPU sets. The description declares each under
`[policy]` as a dotted key, `NODE.SURFACE.NAME[.FIELD]` (`vy-nebius-2.check-slots.a`), and every key has an owner, the handle
who decides a change: an `owner` in a table covers every key under it, unless a table below names its own.

`show` prints the declared policy, each key with its owner and where its live value is read (`source`). `diff` reads every
key's live value over SSH, read-only (`COMMANDS`: `cat`, `grep` and /proc reads), and reports each key where it differs:
`drift` (live differs, or is absent), `undeclared` (live has a key the description lacks) or `unknown` (it couldn't be
read; unknown is never no drift). Node 2 is read in one command whose gate, on node 2's clock and before any
other read, reads only the fill runner's `status.txt` and `windows`: unless `status.txt` says `QUIET`, is at most
`STATUS_FRESH_S` old and not ahead of the clock, and no booked window is on or starts within `WINDOW_LEAD_S`, it reads nothing
else and every node 2 key is unknown. A `status.txt` older than `STATUS_STALE_S` while no fill runner runs is read from
gpu-lease's own state instead: node 2 is read as quiet (and `diff` says so) while no lease is timed, no one pid holds every GPU,
nobody waits for every GPU and no booked window is on or starts within `STALE_WINDOW_LEAD_S`. A write on node 2 (`cluster
policy apply`) still needs the runner's fresh quiet status.

Values are compared in one spelling per surface (`canon`): CPU lists by their CPUs, a window by its start and minutes. A
window that has ended is not compared.

An override (`override`) replaces one key's value until an absolute UTC time, or until a later override of the same key, and
records who and why. Each node keeps its own keys' overrides in its own log (`OVERRIDES`), hash-chained as the allocation
ledger is (`ledger.Ledger`). `show` merges the overrides in force from a local log; `diff` reads each node's log along with
its live values and compares live with the policy in force, the description's value with any override applied. A log that
doesn't verify is refused (exit 3): the policy in force can't be told from it. Recording an override changes nothing on a
node: `cluster policy apply` (policy_apply.py) writes the values whose surface has an applier, and every other value is set
by hand.
"""
from __future__ import annotations

import fcntl
import json
import re
import shlex
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable

from . import cpuset, ledger, model, nebius2, status

DESCRIPTION = Path(__file__).parents[2] / "descriptions" / "nebius.toml"
NODES = status.NODES
CHECK_SLOTS = "/workspace/research/locks/slots"
DIRECT = {"direct-gpus": "/etc/vy/direct-gpus", "direct-cpus": "/etc/vy/direct-cpus"}
USER_CPUSET = "/etc/systemd/system/user@.service.d/vy-cpuset.conf"
FILL = "/workspace/pouw/fill"
#: node 2 is read only while its status.txt's first line says this, it is at most STATUS_FRESH_S old by node 2's clock
#: (fill_runner.py rewrites it every 10 s), and no booked window is on or starts within WINDOW_LEAD_S (status.txt lags a window's start by up to a fill tick)
QUIET = "timed False, window waiting False"
STATUS_FRESH_S = 60
WINDOW_LEAD_S = 60
#: a status.txt this old, while no process runs fill_runner.py (RUNNER_RE, as its unit and its tmux pane start it), is no
#: runner's: node 2 is read from gpu-lease's own state then, keeping every booked window STALE_WINDOW_LEAD_S away, since no
#: runner drains the GPUs for one or says it is on
STATUS_STALE_S = 900
STALE_WINDOW_LEAD_S = 1800
RUNNER_RE = r"^[^ ]*python[0-9.]*( [^ ]+)* [^ ]*fill_runner\.py( |$)"
LEASE_DIR = str(nebius2.LEASE_DIR)
#: each node's surfaces, and the command each is read with
SURFACES = {nebius2.NODE: {"fill": "n2", "check-slots": "n2", "direct-gpus": "n2", "direct-cpus": "n2", "user-cpuset": "n2"}}
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
    """The description's policy keys, checked (`problems`)."""
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
    keys = {k: Key(k, v, o or "") for k, (v, o) in sorted(flat.items())}
    errs = problems(cluster, keys)
    if errs:
        raise PolicyError("; ".join(errs))
    return cluster, keys


def problems(cluster: model.Cluster, keys: dict[str, Key]) -> list[str]:
    """Why these keys can't be read or compared: no owner, an unknown node or surface, or a value of the wrong shape."""
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
        if s == "check-slots" and len(rest) != 1:
            out.append(f"{k.key}: a slot key is {s}.NAME")
        elif s in DIRECT and rest:
            out.append(f"{k.key}: {s} is one value")
        elif s == "user-cpuset" and rest != ["delegate"]:
            out.append(f"{k.key}: the cpuset drop-in's only key is user-cpuset.delegate")
        elif s == "fill" and not (len(rest) == 2 and rest[0] in ("window", "cpu-sets")):
            out.append(f"{k.key}: a fill key is fill.window.START or fill.cpu-sets.OWNER")
        elif s == "fill" and rest[0] == "window":
            try:
                if _minute(_start(rest[1])) != rest[1]:
                    out.append(f"{k.key}: write the start as {_minute(_start(rest[1]))}")
            except ValueError:
                out.append(f"{k.key}: {rest[1]!r} is not a UTC start (2026-10-04T06:30Z)")
    return out


def _start(s: str) -> int:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int((t if t.tzinfo else t.replace(tzinfo=UTC)).timestamp())


def _minute(t: int) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%MZ" if t % 60 == 0 else "%Y-%m-%dT%H:%M:%SZ")


def canon(key: str, value: Any) -> str:
    """`value` in the one spelling `diff` compares for `key`'s surface; ValueError when it isn't that surface's shape."""
    parts = key.split(".")
    s, v = parts[1], str(value).strip()
    if s == "direct-cpus":
        return cpuset.format(cpuset.parse(v))
    if s == "check-slots":
        cpus, *opts = v.split()
        return " ".join([cpuset.format(cpuset.parse(cpus)), *sorted(opts)])
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
    if s in DIRECT:
        return DIRECT[s]
    return {"check-slots": CHECK_SLOTS, "user-cpuset": f"{USER_CPUSET} Delegate=",
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
    """Why an override can't be recorded: a key no node's reader knows or a value of the wrong shape, an `until` that isn't an
    absolute UTC time still to come, or no `by` or `reason`."""
    owner = keys[key].owner if key in keys else _owner_guess(key, keys) or "?"
    out = problems(cluster, {key: Key(key, value, owner)})
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


# the stale form of the gate, read-only, from gpu-lease's own lines and /proc (never a `flock -n` probe: nebius2.py): a live
# owner line tagged timed=1, every GPU's lock under one live pid, or a live waiter for every GPU is a timed window on or
# waiting; an owner line naming no pid can't be told from one
_N2_LEASES = (
    "set +f\n"
    "g=0; h=0; one=; same=1\n"
    f'for o in {LEASE_DIR}/[0-9]*.lock; do [ -e "$o" ] && g=$((g + 1)); done\n'
    f'for o in {LEASE_DIR}/[0-9]*.owner; do\n'
    '  [ -e "$o" ] || continue\n'
    '  l=$(head -1 "$o") || no "gpu-lease owner line $o could not be read"\n'
    '  p=${l#pid=}; p=${p%% *}\n'
    '  case "$l" in pid=*) ;; *) no "gpu-lease owner line $o names no pid" ;; esac\n'
    '  case "$p" in ""|*[!0-9]*) no "gpu-lease owner line $o names no pid" ;; esac\n'
    "  [ -d /proc/$p ] || continue\n"
    '  case " $l " in *" timed=1 "*) no "lease ${o##*/} of pid $p is timed (gpu-lease --timed)" ;; esac\n'
    '  h=$((h + 1)); [ -n "$one" ] || one=$p; [ "$p" = "$one" ] || same=0\n'
    "done\n"
    '[ "$g" -gt 0 ] && [ "$h" -ge "$g" ] && [ "$same" = 1 ] && no "pid $one holds all $g GPUs"\n'
    f'for o in {LEASE_DIR}/wait.*; do\n'
    '  [ -e "$o" ] || continue\n'
    "  p=${o##*.}\n"
    '  case "$p" in ""|*[!0-9]*) continue ;; esac\n'
    "  [ -d /proc/$p ] || continue\n"
    '  l=" $(head -1 "$o" 2>/dev/null) "; k=${l#* n=}; k=${k%% *}\n'
    '  case "$k" in ""|*[!0-9]*) no "waiter ${o##*/} does not say how many GPUs it wants" ;; esac\n'
    '  [ "$g" -gt 0 ] && [ "$k" -ge "$g" ] && no "waiter ${o##*/} wants all $g GPUs"\n'
    "done\n"
    "set -f\n")


def _n2_gate(stale_ok: bool) -> str:
    """Node 2's gate, before any other read there and on its own clock. While status.txt is fresh it must say QUIET; with
    `stale_ok`, a status.txt older than STATUS_STALE_S while no fill runner runs reads gpu-lease's state instead
    (`_N2_LEASES`) and keeps every booked window STALE_WINDOW_LEAD_S away. Unless all holds (a windows line that doesn't parse
    counts as a window near) it prints `refused WHY: STATUS` and reads nothing else; else `status STATUS`, or `stale WHY:
    STATUS` when the stale form let it read."""
    stale = (
        f'  [ "$age" -gt {STATUS_STALE_S} ] || no "status.txt is $age s old: is the fill runner running?"\n'
        f"  x=$(pgrep -f '{RUNNER_RE}'); r=$?\n"
        '  case $r in 0) no "status.txt is $age s old while a fill runner runs: it may be stuck" ;; 1) ;;\n'
        '    *) no "pgrep exited $r: whether a fill runner runs is not known" ;; esac\n'
        + "".join(f"  {x}\n" for x in _N2_LEASES.splitlines())
        + f"  lead={STALE_WINDOW_LEAD_S}\n"
        f'  q="no fill runner runs (status.txt is $age s old), and gpu-lease shows no timed lease, no holder or waiter of '
        f'every GPU and no booked window within {STALE_WINDOW_LEAD_S} s, so it is read as quiet"\n'
        if stale_ok else
        '  no "status.txt is $age s old: is the fill runner running? (a write needs its fresh quiet status)"\n')
    return (
        "set -f\n"
        f"s=$(head -1 {FILL}/status.txt) || exit 1\n"
        "now=$(date +%s) || exit 1\n"
        'no() { printf "refused %s: %s\\n" "$1" "$s"; exit 0; }\n'
        't=$(date -d "${s%% *}" +%s 2>/dev/null) || no "status.txt does not say when it was written"\n'
        "[ \"$t\" -le \"$now\" ] || no \"status.txt is stamped $((t - now)) s ahead of node 2's clock\"\n"
        f"age=$((now - t)); lead={WINDOW_LEAD_S}; q=\n"
        f'if [ "$age" -gt {STATUS_FRESH_S} ]; then\n'
        + stale +
        "else\n"
        f'  case "$s" in *"{QUIET}"*) ;; *) no "a timed window is on or waiting" ;; esac\n'
        "fi\n"
        f"w={FILL}/windows\n"
        'if [ -e "$w" ]; then\n'
        '  while IFS= read -r x || [ -n "$x" ]; do\n'
        "    set -- ${x%%#*}\n"
        "    [ $# -gt 0 ] || continue\n"
        "    m=${2:-30}\n"
        "    b=$(date -d \"$1\" +%s 2>/dev/null) || no \"windows line '$x' does not parse\"\n"
        "    case \"$m\" in *[!0-9.]*|*.*.*|.) no \"windows line '$x' does not parse\" ;; esac\n"
        "    if awk -v n=\"$now\" -v b=\"$b\" -v m=\"$m\" -v d=\"$lead\" 'BEGIN { exit !(b - d <= n && n < b + 60 * m) }'; then\n"
        '      no "window $1 ($m min) is on or starts within $lead s"\n'
        "    fi\n"
        '  done < "$w" || exit 1\n'
        "fi\n"
        'if [ -n "$q" ]; then printf "stale %s: %s\\n" "$q" "$s"; else printf "status %s\\n" "$s"; fi\n')


N2_GATE = _n2_gate(stale_ok=True)
#: what a write on node 2 runs first: the fill runner's fresh quiet status, never the stale form
N2_WRITE_GATE = _n2_gate(stale_ok=False)

COMMANDS = {
    "n2": N2_GATE + _dump([f"{FILL}/windows", f"{FILL}/cpu-sets", CHECK_SLOTS, *DIRECT.values(), USER_CPUSET, OVERRIDES]),
}
READS = {nebius2.NODE: ("n2",)}


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
    """A node's live policy: each key's value as read, the prefixes that couldn't be read (with why), and the records of the
    node's override log."""
    values: dict[str, str]
    unknown: dict[str, str]
    overrides: list[dict[str, Any]] = field(default_factory=list)
    #: why node 2's gate read it as quiet with no fresh status from its fill runner (its stale form), with that status
    quiet: str | None = None


def _read_overrides(node: str, files: dict[str, str | None], live: Live) -> None:
    try:
        live.overrides = read_overrides(files.get(OVERRIDES) or "", f"{node}:{OVERRIDES}")
    except PolicyError as e:
        live.unknown[node] = f"its override log doesn't verify, so the policy in force can't be told: {e}"


def _slots(text: str) -> list[list[str]]:
    return [f for f in (line.split("#")[0].split() for line in text.splitlines()) if f]


def _read_files(node: str, files: dict[str, str | None], live: Live) -> None:
    v = live.values
    if files.get(CHECK_SLOTS) is not None:
        for f in _slots(files[CHECK_SLOTS]):
            v[f"{node}.check-slots.{f[0]}"] = " ".join(f[1:])
    for s, p in DIRECT.items():
        text = files.get(p)
        lines = [x for x in (line.split("#")[0].strip() for line in (text or "").splitlines()) if x]
        if text is not None:
            v[f"{node}.{s}"] = lines[0] if lines else ""
    text = files.get(USER_CPUSET)
    if text is not None:
        d = [line.split("=", 1)[1] for line in text.splitlines() if line.strip().startswith("Delegate=")]
        v[f"{node}.user-cpuset.delegate"] = d[-1].strip() if d else ""


def _read_n2(node: str, raw: str, now: int, live: Live) -> None:
    first, _, rest = raw.partition("\n")
    kind, _, line = first.partition(" ")
    if kind == "refused":
        live.unknown[node] = f"node 2 refused to be read, and nothing else there was: {line[:240]}"
        return
    if kind == "stale":
        live.quiet = line
    elif kind != "status":
        live.unknown[node] = f"node 2's gate printed {first[:80]!r}, neither a status nor a refusal"
        return
    files = _files(rest)
    # a line fill_runner.py skips is no window and no CPU set, so it is skipped here too
    for f in _slots(files.get(f"{FILL}/windows") or ""):
        try:
            start, minutes = _start(f[0]), float(f[1]) if len(f) > 1 else 30.0
        except ValueError as e:
            print(f"cluster policy: {node}'s fill window {' '.join(f)[:80]!r} skipped, as fill_runner.py skips it ({e})",
                  file=sys.stderr)
            continue
        if start + 60 * minutes > now:
            key = f"{node}.fill.window.{_minute(start)}"
            live.values[key] = canon(key, minutes)
    for f in _slots(files.get(f"{FILL}/cpu-sets") or ""):
        if len(f) >= 3 and re.fullmatch(r"[\d,-]+", f[1]) and f[2].isdigit() and (len(f) < 4 or f[3].isdigit()):
            live.values[f"{node}.fill.cpu-sets.{f[0]}"] = " ".join(f[1:4])
    _read_files(node, files, live)
    _read_overrides(node, files, live)


def live(node: str, raw: dict[str, str | Exception], now: int) -> Live:
    """The node's live policy from `read`'s outputs, at `now`."""
    out = Live({}, {})
    if isinstance(raw["n2"], Exception):
        out.unknown[node] = f"node 2 couldn't be read: {raw['n2']}"
    else:
        _read_n2(node, raw["n2"], now, out)
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
            if same:
                row["status"] = "ok"
            else:
                row.update(status="drift", why=bad or "live differs")
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
        if rep.get("quiet"):
            out.append(f"  read as quiet: {rep['quiet']}")
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
              f"Nothing carries it to the node but `cluster policy apply --key {key} --apply --as WHO --reason R`: set any other "
              "value yourself. `cluster policy diff` compares the live value with it.")
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
    runner = runner or (lambda node: status.reach(node, key))
    nodes = _nodes(which)
    with ThreadPoolExecutor(len(nodes)) as ex:
        raws = list(ex.map(lambda n: read(n, runner(n)), nodes))
    reports = []
    for node, raw in zip(nodes, raws):
        try:
            lv = live(node, raw, now)
            rows = diff(effective(keys, lv.overrides, now, node), node, lv, now)
        except Exception as e:  # noqa: BLE001  (output of any unexpected shape is unknown, exit 3, never drift's 1)
            lv = Live({}, {node: f"its live policy couldn't be parsed: {type(e).__name__}: {e}"})
            rows = diff(keys, node, lv, now)
        reports.append({"node": node, "keys": rows, **({"quiet": lv.quiet} if lv.quiet else {})})
    found = [r for rep in reports for r in rep["keys"] if r["status"] != "ok"]
    if as_json:
        print(json.dumps({"schema": "cluster/policy-diff/v1", "t": now, "description": str(description), "nodes": reports},
                         indent=1, sort_keys=True))
    else:
        print(render_diff(reports), end="")
    if any(r["status"] == "unknown" for r in found):
        return 3
    return 1 if found else 0


__all__ = ["DESCRIPTION", "OVERRIDES", "COMMANDS", "READS", "Key", "Live", "PolicyError", "load",
           "problems", "canon", "source", "window_ends", "read_overrides", "overrides", "effective", "override_problems", "read",
           "live", "diff", "show", "render_show", "render_diff", "main_show", "main_diff", "main_override"]
