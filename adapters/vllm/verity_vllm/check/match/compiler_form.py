"""The fold in the registered compiler's form: its instances as `program.prune.dead_free` leaves a body's nodes.

The Build's request Programs are the registered compiler's (`prune.kept_records`): without the units no output reads, each
Definition with dead gates bound to its version without them, and each linear read only in part narrowed to the columns read.  The
fold records what the runtime ran: vLLM runs every prompt row through the last layer, with the v1 kernels.  When every component is in
that form (`components_in_form`: its derive report counts no dead computed gate, the form's defining property, or, where the form
leaves dead gates no rule removes, as an int8 row's unnarrowed linears, its Program names the dead-free versions and none of the v1s
they replace), the Match puts the fold in it before anything reads it (`compiled`), by the compiler's three rules over the instances:

* live: a sampling event, or an instance a live one reads.  Every sampling event is an output, a partial prefill chunk's discarded
  sample too: G6 counts it against the record's logits rows and the sampler geometry reads its split count.  So a chunked prefill
  keeps that row's last layer whole where its component narrows it, and compares unequal by name.
* swapped: each Definition `prune.dead_free_versions` names, at the same statics, so only the spec's `Name_vN` changes.
* narrowed: each `prune.COLUMNS` linear whose output words the live instances read only in [a, b) ⊊ [0, N) is bound at N = b - a
  over its column arguments' rows a..b-1 (the spec's N, the output's length and those arguments' runs), and every read of it moves
  down by a.

An instance keeps its node index (the fold rows' `node`), so a fold row read by its node still names its instance.  Operands are
runs, or run progressions in the compact form; an operand that changes is rebuilt, never edited, since the compact decode shares
one operand list between aliased nodes.
"""
from __future__ import annotations

import bisect
import collections
import json
import os
import re
from collections.abc import Iterable

from verity_vllm.program import compact as CP
from verity_vllm.program import prune
from verity_vllm.program.sampling_event import is_sampling_event

_INT = re.compile(r"\d+")
_WORDS = re.compile(r"Array<(\d+),(Value<\d+>)>")


def components_in_form(dirs: Iterable[str]) -> dict:
    """Whether every component dir holds a Program in the registered compiler's form: its `derive-report.json` counts no dead
    computed gate, exactly (`dead_gates`, not approximate), or its Program names only the dead-free versions (`_swapped`).  A dir
    without that record is not in the form."""
    dirs = list(dict.fromkeys(dirs))
    out = [d for d in dirs if not _in_form(d)]
    return {"in_form": bool(dirs) and not out, "components": len(dirs), "not_in_form": [os.path.basename(d.rstrip("/")) for d in out[:4]]}


def fold_in_form(fold, dirs: Iterable[str]) -> dict:
    """Put a freshly loaded fold `RootView` in the registered compiler's form when every component dir is in it; the record of both,
    also kept as `fold.fold_form` for every document that reads this fold."""
    out = components_in_form(dirs)
    if out["in_form"]:
        out.update(fold.to_compiler_form())
    fold.fold_form = out
    return out


def _in_form(d: str) -> bool:
    try:
        dg = json.load(open(os.path.join(d, "derive-report.json"))).get("dead_gates")
    except (OSError, ValueError):
        return False
    return (isinstance(dg, dict) and dg.get("dead_computed_gates") == 0 and dg.get("approximate") is False) or _swapped(d)


def _swapped(d: str) -> bool:
    """Whether the component's Program (its `result.json` `spec_histogram`, one entry per root node) names a dead-free version
    `prune.dead_free_versions` binds and none of the Definitions they replace.  `prune` swaps only inside `dead_free`, so such a
    Program is in the form whatever dead gates it counts: those no rule of the form removes (an int8 linear read in part:
    `ScaledMmS8_v1` is no `prune.COLUMNS` linear, so the last layer's qkv on the prompt rows before the last keeps its query columns)."""
    try:
        hist = json.load(open(os.path.join(d, "result.json")))["spec_histogram"]
        bases = {_base(k)[1] for k in hist}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    table = prune.dead_free_versions()
    return bool(bases & {v.id for v in table.values()}) and not bases & set(table)


def _base(spec: str) -> tuple[str, str, str]:
    """(scope prefix, `Name_vN`, statics) of a spec `[scope:]Name_vN[{...}]`."""
    head, brace, rest = spec.partition("{")
    scope, _, base = head.rpartition(":")
    return (scope + ":" if scope else ""), base, brace + rest


def _top_n(statics: str) -> tuple[int, int] | None:
    """[start, end) of the integer the top-level static `N` holds in `{K=..,N=..,...}`; None without one.  A function-valued static
    names its Definition inside a JSON string, whose own `N=` is not this one."""
    depth, quoted, escaped = 0, False, False
    for i, ch in enumerate(statics):
        if quoted:
            quoted, escaped = (quoted, False) if escaped else (ch != '"', ch == "\\")
            continue
        if ch == '"':
            quoted = True
        elif ch in "}]":
            depth -= 1
        elif ch in "{[,":
            depth += ch != ","
            if ch != "[" and depth == 1 and statics.startswith("N=", i + 1) and (m := _INT.match(statics, i + 3)):
                return m.start(), m.end()
    return None


def _visits(t: tuple, compact: bool, done: dict) -> Iterable[tuple[int, int, int]]:
    """(node, lo, hi) of each node-space run of operand item `t` not visited before under the same progression (`done`: a gather
    over the same producers and window, kept at its longest)."""
    if t[0] != "n":
        return ()
    if not compact:
        return ((t[1], t[2], t[3]),)
    (i0, lo0, hi0), (dI, dL, dH), c = t[1], t[2], t[3]
    key = (i0, lo0, hi0, dI, dL, dH) if c > 1 else None
    first = done.get(key, 0) if key else 0
    if c <= first:
        return ()
    if key:
        done[key] = c
    return ((i0 + j * dI, lo0 + j * dL, hi0 + j * dH) for j in range(first, c))


def _live(inst: list[dict], k2i: dict[int, int], compact: bool) -> list[bool]:
    live = [is_sampling_event(r["spec"]) for r in inst]
    done: dict = {}
    for i in range(len(inst) - 1, -1, -1):
        if live[i]:
            for a in inst[i]["args"]:
                for t in a:
                    for k, _lo, _hi in _visits(t, compact, done):
                        j = k2i.get(k)
                        if j is not None:
                            live[j] = True
    return live


def _cut(arg: list, x: int, y: int, compact: bool) -> list:
    """Operand `arg`'s leaves [x, y), in its own form."""
    out, at = [], 0
    for space, idx, lo, hi in (CP.expand(arg) if compact else arg):
        s, e = max(x, at), min(y, at + hi - lo)
        if e > s:
            out.append((space, idx, lo + s - at, lo + e - at))
        at += hi - lo
    return CP.from_runs(out) if compact else out


def _reads_any(t: tuple, nodes: list[int], compact: bool) -> bool:
    if t[0] != "n":
        return False
    if not compact:
        return bisect.bisect_left(nodes, t[1]) < len(nodes) and nodes[bisect.bisect_left(nodes, t[1])] == t[1]
    (i0, _, _), (dI, _, _), c = t[1], t[2], t[3]
    lo, hi = (i0, i0 + (c - 1) * dI) if dI >= 0 else (i0 + (c - 1) * dI, i0)
    j = bisect.bisect_left(nodes, lo)
    while j < len(nodes) and nodes[j] <= hi:
        if dI == 0 or (nodes[j] - i0) % dI == 0:
            return True
        j += 1
    return False


def _shifted(arg: list, by: dict[int, int], nodes: list[int], compact: bool) -> list:
    """Operand `arg` with each read of a node k in `by` moved down by `by[k]`; the same list when it reads none."""
    if not any(_reads_any(t, nodes, compact) for t in arg):
        return arg
    runs = CP.expand(arg) if compact else arg
    out = [(s, i, lo - by[i], hi - by[i]) if s == "n" and i in by else (s, i, lo, hi) for s, i, lo, hi in runs]
    return CP.from_runs(out) if compact else out


def _narrowed(inst: list[dict], compact: bool) -> tuple[list[dict], collections.Counter]:
    cand = {}
    for i, r in enumerate(inst):
        scope, base, statics = _base(r["spec"])
        if r["form"] == "call" and base in prune.COLUMNS:
            at = _top_n(statics)
            if at is None:
                raise ValueError(f"{r['spec']}: a COLUMNS linear without its N static")
            cand[r["k"]] = (int(statics[at[0]:at[1]]), i, scope, base, statics, at)
    if not cand:
        return inst, collections.Counter()
    spans = {k: [c[0], 0] for k, c in cand.items()}
    done: dict = {}
    for r in inst:
        for a in r["args"]:
            for t in a:
                for k, lo, hi in _visits(t, compact, done):
                    if k in spans:
                        sp = spans[k]
                        sp[0], sp[1] = min(sp[0], lo), max(sp[1], hi)
    by, out, made = {}, list(inst), collections.Counter()
    for k, (N, i, scope, base, statics, (s, e)) in cand.items():
        a, b = spans[k]
        if b <= a or (a, b) == (0, N):
            continue
        r = inst[i]
        w = _WORDS.fullmatch(r["out"])
        if w is None or int(w.group(1)) != N:
            raise ValueError(f"{r['spec']}: its output {r['out']} is not N words, so its columns [{a}, {b}) can't be narrowed")
        args = list(r["args"])
        for j in prune.COLUMNS[base]:
            leaves = sum(hi - lo for _s, _i, lo, hi in (CP.expand(args[j]) if compact else args[j]))
            if leaves % N:
                raise ValueError(f"{r['spec']}: argument {j} holds {leaves} leaves, not N = {N} rows")
            args[j] = _cut(args[j], a * (leaves // N), b * (leaves // N), compact)
        out[i] = {**r, "spec": scope + base + statics[:s] + str(b - a) + statics[e:], "args": args, "out": f"Array<{b - a},{w.group(2)}>"}
        by[k] = a
        made[base] += 1
    if not by:
        return inst, made
    nodes = sorted(by)
    for i, r in enumerate(out):
        args = [_shifted(a, by, nodes, compact) for a in r["args"]]
        if any(x is not y for x, y in zip(args, r["args"])):
            out[i] = {**r, "args": args}
    return out, made


def compiled(inst: list[dict], compact: bool) -> tuple[list[dict], dict]:
    """`inst` (a decoded root body's instances in node order: k, spec, form, args, out) in the registered compiler's form, and what
    that dropped, swapped and narrowed.  An instance that changes is a new dict; the rest are `inst`'s own."""
    if not any(is_sampling_event(r["spec"]) for r in inst):
        return inst, {"sampling_events": 0, "dead": {}, "swapped": {}, "narrowed": {}}
    live = _live(inst, {r["k"]: i for i, r in enumerate(inst)}, compact)
    kept = [r for r, v in zip(inst, live) if v]
    dead = collections.Counter(_base(r["spec"])[1] for r, v in zip(inst, live) if not v)
    table = {old: new.id for old, new in prune.dead_free_versions().items()}
    swaps: collections.Counter = collections.Counter()
    for i, r in enumerate(kept):
        scope, base, statics = _base(r["spec"])
        if r["form"] != "primitive" and base in table:
            kept[i] = {**r, "spec": scope + table[base] + statics}
            swaps[f"{base} -> {table[base]}"] += 1
    kept, made = _narrowed(kept, compact)
    return kept, {"sampling_events": sum(map(is_sampling_event, (r["spec"] for r in kept))), "dead": dict(sorted(dead.items())),
                  "swapped": dict(sorted(swaps.items())), "narrowed": dict(sorted(made.items()))}
