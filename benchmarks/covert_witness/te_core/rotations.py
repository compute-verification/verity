"""A run's rotation products, named for the GEMM they were computed as.

    python -m te_core.rotations RUN.json [...]    # each record's rotation products, per step structure

Three islands' update codes rotate: a fixed orthogonal matrix times a block of activations, of a gradient or of an
update, in f32 with autocast off. Each of their programs sets cuda.matmul.allow_tf32, so on sm_120 those products are
cuBLAS's TF32 GEMM (`GemmTf32_v1`), which te_core.gates prices for a row whose dtype reads `tf32`. The counter's log
carries no dtype, so te_core.gates reads every `mm` row as a bf16 GEMM. `labelled` names the rotations' rows `tf32`
(the dtype after a row's shapes, as a ledger row's key carries it), and nothing else of the record changes: the same
ops, counts and MACs.

Which rows are a rotation's is read from the record: its program, its stored arguments, its log and the start's shape.
A rotation matrix is no weight of θ and is never updated, so its products have role `theta` forward and `grad`
backward, and `param` in the optimizer's phase.

- lowrank-restart's `delta_grad.py` with `--rotate` (`GROUP`): the code packs each tensor's update in groups of g =
  `--group` (128 unless given), `x.reshape(-1, g) @ R` and back. Its products are the rows
  `other param mm [[n, g], [g, g]]`, whole: no other product of the program has a g x g operand. The log identifies
  them.
- wildcard's `train.py` with `--z-rotate` (`ADAPTER`): the adapter's r-wide code (r = `--adapter`) is rotated across
  the cut, forward and backward. Its products are the rows `mm [[T, r], [r, r]]`, whole. The log identifies them.
- wildcard's `train.py` and hybrid's `cut.py` with `--cut` (`CUTS`): each coded pass hands a token's d-vector across
  each cut as `y @ R^T`, a code, `z @ R`, and so does the backward pass of a cut whose input needs a gradient. Both
  products are `mm [[T, d], [d, d]]`, the shape of a block's four attention matrices' products, and the counter merges
  them: forward under role `theta` with a frozen block's, backward under `grad` with the trained blocks' data
  gradients. The lf model's shape separates them. Forward, a block runs four d x d products and one down product
  `[[T, h], [h, d]]` a pass, trained or not, so the rotations are the d x d count less four times the down count.
  Backward, a trained block's attention matrices each have one data gradient (d x d) and one weight gradient
  `[[d, T], [T, d]]`, so the rotations are the d x d count less the weight gradients' count. The stored arguments then
  have to agree, or the record is refused (`Unread`): the forward rotations are two a cut a coded pass, and the
  backward ones two a backward pass (the head's data gradient `[[T, V], [V, d]]` counts them) for each cut that has a
  trained weight below it (a cut at boundary B is the input of block B; the lowest trained block is read from the
  record's updated weights' names).

A code that keeps only some rotated coordinates (`p<keep>` in a cut's code) has products of another shape, and a
program this module doesn't know that takes one of the three arguments is refused too, never guessed.

`repriced` gives a ledger row ingested before this module what `ledger.ingest` writes for its run's record now: the
same row with the rotations' rows named in its `macs_log` and each part of its `formalism` at the gates the named log
has. A traced part's affected gates and token gates are the trace's shares of the part's gates (te_core.formalism's
`of_record`), so they move with its gates; the trace's shares themselves stay as traced.
"""
from __future__ import annotations

import json
import math
import re
import sys

from . import policy

#: the dtype a rotation's row is named with (te_core.gates' `DTYPES`)
TF32 = "tf32"
GROUP, ADAPTER, CUTS = "group", "adapter", "cuts"
#: (island, program) -> the arguments that make it rotate, each with its rule
PROGRAMS = {("lowrank-restart", "delta_grad.py"): {"--rotate": GROUP},
            ("wildcard", "train.py"): {"--z-rotate": ADAPTER, "--cut": CUTS},
            ("hybrid", "cut.py"): {"--cut": CUTS}}
ARGUMENTS = sorted({a for rules in PROGRAMS.values() for a in rules})
#: delta_grad.py's `--group` where the run gave none
DEFAULT_GROUP = 128
#: the lf block's attention matrices (model.py's wq, wk, wv, wo), each one d x d product a pass
ATTENTION = 4
BLOCK = re.compile(r"blocks\.(\d+)\.")


class Unread(ValueError):
    """A record whose rotations this module can't tell from its other products; the message says why."""


def argument(args: list[str], name: str) -> str | None:
    """The value of `name` in a program's arguments (`--name V` or `--name=V`), "" for a flag, None where absent."""
    for i, a in enumerate(args):
        if a == name:
            nxt = args[i + 1] if i + 1 < len(args) else ""
            return "" if nxt.startswith("--") else nxt
        if a.startswith(name + "="):
            return a[len(name) + 1:]
    return None


def island(record: dict) -> tuple[str, str]:
    ev = (record.get("evaluation") or {}).get("island") or {}
    name = record.get("island") if isinstance(record.get("island"), str) else ev.get("name")
    return name or "", record.get("program") or ev.get("program") or ""


def rules(record: dict) -> dict[str, str]:
    """The record's rotation rules, each with its argument's value; none for a program that doesn't rotate."""
    args = [str(a) for a in record.get("script_args") or []]
    given = {a: argument(args, a) for a in ARGUMENTS}
    given = {a: v for a, v in given.items() if v is not None}
    known = PROGRAMS.get(island(record))
    if known is None or not given:
        return {}
    return {known[a]: v for a, v in given.items() if a in known}


def unknown(record: dict) -> str | None:
    """Why a record that takes one of `ARGUMENTS` is of no program this module reads, else None."""
    args = [str(a) for a in record.get("script_args") or []]
    given = [a for a in ARGUMENTS if argument(args, a) is not None]
    who = island(record)
    if given and who not in PROGRAMS:
        return f"{who[0] or '?'}'s {who[1] or '?'} takes {', '.join(given)}, and te_core.rotations doesn't know the program"
    if given and not rules(record):
        return f"{who[0]}'s {who[1]} takes {', '.join(given)}, which te_core.rotations doesn't read for it"
    return None


def cuts(spec: str, layers: int) -> list[int]:
    """The boundaries of a `--cut` argument `B:CODE,...` (B from 1 to `layers`, or `all`), one per cut."""
    out = []
    for item in spec.split(","):
        b, _, code = item.partition(":")
        if not code or not re.fullmatch(r"[a-z0-9]+", code) or "q" not in code:
            raise Unread(f"--cut {spec!r}: {item!r} is no B:CODE")
        if re.search(r"p\d", code):
            raise Unread(f"--cut {spec!r}: a p<keep> code's products are [[T, d], [d, keep]], which aren't read here")
        if b != "all" and not (b.isdigit() and 1 <= int(b) <= layers):
            raise Unread(f"--cut {spec!r}: boundary {b!r} is not all or 1..{layers}")
        out += range(1, layers + 1) if b == "all" else [int(b)]
    if len(set(out)) != len(out):
        raise Unread(f"--cut {spec!r} names a boundary twice")
    return out


def lowest_trained(record: dict) -> int | None:
    """The lowest of the start's blocks the run updates, -1 where it updates a weight below every block (the
    embedding), None where it updates no named block (new weights only: where they sit isn't in their names)."""
    names = {n for s in record["steps"] for n in s.get("weights") or ()}
    blocks = [int(m[1]) for n in names if (m := BLOCK.match(n))]
    if any(not BLOCK.match(n) and n != "norm" and not n.startswith("new/") for n in names):
        return -1
    return min(blocks) if blocks else None


def _shape(row: list) -> list | None:
    try:
        s = json.loads(row[3])
    except (TypeError, json.JSONDecodeError):
        return None
    return s if (isinstance(s, list) and len(s) == 2 and all(isinstance(x, list) and len(x) == 2 and
                                                               all(isinstance(v, int) for v in x) for x in s)) else None


def _named(log: list) -> bool:
    return any(str(r[3]).endswith(" " + TF32) for r in log)


def _take(log: list, phase: str, role: str, shape: list, n: int, why: str) -> list:
    """`log` with `n` of the `mm` products of `shape` under (phase, role) named `TF32`, at their share of the row's
    MACs."""
    if not n:
        return log
    at = [i for i, r in enumerate(log) if r[0] == phase and r[1] == role and r[2] == "mm" and _shape(r) == shape]
    if len(at) != 1 or log[at[0]][4] < n or log[at[0]][5] * n % log[at[0]][4]:
        raise Unread(f"{why}: {n} rotation products of {shape} under {phase} {role}, where the log has "
                     f"{[log[i][4:6] for i in at]}")
    row = log[at[0]]
    macs = row[5] * n // row[4]
    rest = [[*row[:4], row[4] - n, row[5] - macs]] if row[4] > n else []
    return [*log[:at[0]], *rest, [*row[:3], f"{row[3]} {TF32}", n, macs], *log[at[0] + 1:]]


def _count(log: list, phase: str, shape: list) -> int:
    return sum(r[4] for r in log if r[0] == phase and r[2] == "mm" and _shape(r) == shape)


def _group(log: list, g: int) -> tuple[list, dict]:
    out, n = [], 0
    for r in log:
        s = _shape(r)
        if r[0] == "other" and r[1] == "param" and r[2] == "mm" and s and s[0][1] == g and s[1] == [g, g]:
            out.append([*r[:3], f"{r[3]} {TF32}", *r[4:]])
            n += r[4]
        else:
            out.append(r)
    return out, {"other": n}


def _adapter(log: list, r_: int) -> tuple[list, dict]:
    out, n = [], {"fwd": 0, "bwd": 0}
    for r in log:
        s = _shape(r)
        if r[2] == "mm" and s and s[0][1] == r_ and s[1] == [r_, r_] and (r[0], r[1]) in (("fwd", "theta"), ("bwd", "grad")):
            out.append([*r[:3], f"{r[3]} {TF32}", *r[4:]])
            n[r[0]] += r[4]
        else:
            out.append(r)
    if n["fwd"] % 2 or n["bwd"] % 2 or n["bwd"] > n["fwd"]:
        raise Unread(f"--z-rotate: {n['fwd']} forward and {n['bwd']} backward products of [[T, {r_}], [{r_}, {r_}]], "
                     "which are no two a pass")
    return out, n


def _cuts(log: list, shape: dict, at: list[int], low: int | None) -> tuple[list, dict]:
    d, h, v = shape["d_model"], shape["d_ff"], shape["vocab"]
    back = None if low is None else sum(b > low for b in at)
    n = {"fwd": 0, "bwd": 0}
    for t in sorted({s[0][0] for r in log if r[2] == "mm" and (s := _shape(r)) and s[0][1] == d and s[1] == [d, d]}):
        dd = [[t, d], [d, d]]
        fwd = _count(log, "fwd", dd) - ATTENTION * _count(log, "fwd", [[t, h], [h, d]])
        passes = _count(log, "bwd", [[t, v], [v, d]])
        bwd = _count(log, "bwd", dd) - _count(log, "bwd", [[d, t], [t, d]])
        why = f"--cut with {len(at)} cuts at {t} rows"
        if fwd < 0 or fwd % (2 * len(at)):
            raise Unread(f"{why}: {fwd} forward d x d products beyond the blocks' own, which are no two a cut a pass")
        if bwd < 0 or bwd % 2 or (bwd and not passes) or (passes and (bwd % (2 * passes) or bwd // (2 * passes) > len(at)
                                                                     or fwd < 2 * len(at) * passes)):
            raise Unread(f"{why}: {bwd} backward d x d products beyond the weights' own over {passes} backward passes "
                         f"and {fwd} forward ones, which are no two a cut a pass")
        if passes and back is not None and bwd != 2 * passes * back:
            raise Unread(f"{why}: {bwd} backward rotation products over {passes} backward passes, where {back} cuts "
                         f"have a trained weight below them (the lowest trained block is {low})")
        log = _take(_take(log, "fwd", "theta", dd, fwd, why), "bwd", "grad", dd, bwd, why)
        n["fwd"] += fwd
        n["bwd"] += bwd
    return log, n


def named(record: dict) -> tuple[dict, dict[str, dict]]:
    """`labelled`'s record, and per structure the rotation products it named, by phase."""
    found = rules(record)
    if (why := unknown(record)):
        raise Unread(why)
    if not found:
        return record, {}
    args = [str(a) for a in record.get("script_args") or []]
    shape = policy.SHAPES.get((record.get("start") or {}).get("parameters"))
    structures, counts = {}, {}
    for sid, st in record["structures"].items():
        log = st.get("log") or []
        if _named(log):
            structures[sid] = st
            continue
        n = {}
        if GROUP in found:
            log, got = _group(log, int(argument(args, "--group") or DEFAULT_GROUP))
            n |= got
        if ADAPTER in found:
            r = argument(args, "--adapter")
            if not r or not r.isdigit() or (shape and int(r) == shape["d_model"]):
                raise Unread(f"--z-rotate with --adapter {r!r}: the code's width is not one of its own")
            log, got = _adapter(log, int(r))
            n |= {k: n.get(k, 0) + x for k, x in got.items()}
        if CUTS in found:
            if shape is None:
                raise Unread(f"--cut on a start of {(record.get('start') or {}).get('parameters')} parameters, whose "
                             "shape te_core.policy doesn't hold")
            layers = shape["n_layer"] + int(argument(args, "--grow") or 0)
            log, got = _cuts(log, shape, cuts(found[CUTS], layers), lowest_trained(record))
            n |= {k: n.get(k, 0) + x for k, x in got.items()}
        structures[sid] = dict(st, log=log)
        if any(n.values()):
            counts[sid] = {k: x for k, x in n.items() if x}
    stepped = {s.get("struct") for s in record["steps"]}
    if not any(sid in counts for sid in stepped) and not any(_named((record["structures"].get(s) or {}).get("log") or [])
                                                             for s in stepped):
        raise Unread(f"{' and '.join(sorted(found))}: no step of the record has a rotation product")
    return dict(record, structures=structures), counts


def labelled(record: dict) -> dict:
    """The record with its rotations' rows named `TF32` in every structure's log (module docstring); the record
    itself where its program doesn't rotate. A structure already named is left as it is."""
    return named(record)[0]


def macs(record: dict, through: int | None = None) -> int:
    """The MACs of the rotation products of the record's prefix through step `through` (the whole run for None)."""
    from .ledger import _macs_log

    return sum(m for k, (_, m) in _macs_log(labelled(record), through)["ops"].items() if k.endswith(" " + TF32))


def _as_traced(record: dict, idle: bool = False) -> dict:
    """The record's steps grouped as te_core.annotate's `with_structures` groups a traced run's: a step whose structure
    updates takes that structure under a name of its own with its token count, and with `idle` a step whose structure
    updates nothing takes it under a name of its own too."""
    out = dict(record, structures=dict(record["structures"]), steps=[dict(s) for s in record["steps"]])
    for s in out["steps"]:
        st = record["structures"].get(s.get("struct"))
        if st and s.get("tokens") and (st.get("updated") or idle):
            new = f"{s['struct']}+traced{s['tokens']}" if st.get("updated") else f"{s['struct']}+traced"
            out["structures"][new] = st
            s["struct"] = new
    return out


def _views(record: dict):
    """The records `ledger.ingest` may have written a row from: the record as counted, or with its lookups priced
    (te_core.annotate's `with_lookups`, a record counted before `LOOKUPS_FROM`), each with its steps grouped as they
    are or as a traced run's."""
    from .annotate import LOOKUPS_FROM, with_lookups
    from .ledger import lookup_tables, version

    groups = (lambda r: r, _as_traced, lambda r: _as_traced(r, idle=True))
    yield from groups
    if version(record.get("te_core")) < LOOKUPS_FROM:
        tables = lookup_tables()
        for group in groups:
            yield lambda r, group=group: group(with_lookups(r, tables))


def _parts(record: dict, through: int | None) -> list[dict]:
    from .formalism import of_record

    return of_record(dict(record, formalism={}), through)["parts"]


def repriced(row: dict, record: dict) -> dict:
    """A ledger row of `record`'s run as `ledger.ingest` writes it with the rotations named (module docstring): its
    `macs_log` the named record's, and each part of its `formalism` at the named log's gates. Raises `Unread` unless
    the unnamed record, in one of `_views`, gives the row's own `macs_log` and each part's steps and gates, so a row
    of another record, or one ingested another way, is never changed."""
    from .formalism import TRACED
    from .ledger import _macs_log

    new = labelled(record)
    if new is record:
        return row
    last = record["steps"][-1]["step"] if record["steps"] else -1
    through = None if row["checkpoint_step"] >= last else row["checkpoint_step"]
    f = row.get("formalism") or {}
    stored = f.get("parts") or []
    seen = []
    for view in _views(record):
        was = view(record)
        if json.loads(json.dumps(_macs_log(was, through))) != row.get("macs_log"):
            seen.append("another macs_log")
            continue
        old = _parts(was, through) if stored else []
        if len(old) == len(stored) and all(p["steps"] == o["steps"] and math.isclose(p["gates"], o["gates"], rel_tol=1e-12)
                                           for p, o in zip(stored, old)):
            break
        seen.append(f"parts of {[(o['steps'], o['gates']) for o in old]}")
    else:
        raise Unread(f"{row.get('id')}: its macs_log and its parts' steps and gates "
                     f"{[(p['steps'], p['gates']) for p in stored]} through step {row['checkpoint_step']} are not its "
                     f"record's: {'; '.join(seen)}")
    now = view(new)
    out = dict(row, macs_log=_macs_log(now, through))
    if not stored:
        return out
    traced = (f.get("labels") or {}).get("affected_gates") == TRACED["affected_gates"]
    parts = []
    for p, o, n in zip(stored, old, _parts(now, through)):
        scale = n["gates"] / o["gates"] if o["gates"] else 1.0
        counted = math.isclose(p["token_gates"], o["token_gates"], rel_tol=1e-12, abs_tol=1e-6)
        q = dict(p, gates=n["gates"], token_gates=n["token_gates"] if counted else p["token_gates"] * scale)
        if traced:
            q["affected_gates"] = p["affected_gates"] * scale
        parts.append(q)
    return dict(out, formalism=dict(f, parts=parts))


def main(argv=None) -> int:
    from .gates import per_mac

    for path in sys.argv[1:] if argv is None else argv:
        record = json.loads(open(path).read())
        try:
            _, counts = named(record)
        except Unread as e:
            print(json.dumps({"record": path, "unread": str(e)}))
            continue
        steps: dict[str, int] = {}
        for s in record["steps"]:
            steps[s.get("struct")] = steps.get(s.get("struct"), 0) + 1
        m = macs(record) if counts else 0
        print(json.dumps({"record": path, "rules": rules(record), "rotation_macs": m,
                          "gates_added": m * (per_mac(f"gemm/{TF32}") - per_mac("gemm/bf16")),
                          "structures": {sid: n | {"steps": steps.get(sid, 0)} for sid, n in counts.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
