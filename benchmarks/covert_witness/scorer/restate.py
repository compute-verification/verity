"""A search record's tokens a step, restated, and a copy of the scorer's state that holds the restated records.

    python scorer/restate.py plan --state STATE --scope SCOPE.json --layers TREE[,TREE...] --out PLAN.json
                                  [--runs RUNS.txt] [--workers N]
    python scorer/restate.py overlay --state STATE --plan PLAN.json --overlay DIR [--only RUNS.txt] [--reuse DIR]
                                     [--drop-cache NAME] --out FILE
    python scorer/restate.py totrace --plan PLAN.json [--traced OUT ...] [--leave RUN,...] --out FILE
    python scorer/restate.py layer --base TRACED --over TRACED [--over TRACED ...] [--only RUNS.txt] --dest DIR --out FILE
    python scorer/restate.py one --record run.json --plan PLAN.json --run RUN --out FILE

What a record states. The counter that wrote a search run's record counts a batch of the data stream once a step, at
its largest read as indices (te_core.flops, `Counter._token_source`). A step that reads its batch in slices, one
forward a slice with the gradients added up, therefore states one slice's tokens in `steps[i].tokens`, and a step
that validates on tensors of one size made one after another states one of them.

What reads it. te_core's trace measures the tokens the traced step consumed; a run's factor is that over the count
its record states for the step (`formalism.trace.token_factor`). The run's token total has the factor on every step.
The uses of a trained weight a token do not (`formalism.he_of_rows` divides the traced step's token-indexed products
by the record's count, so with one slice's tokens it counts each trained weight once a forward), nor do the first
update's term (`step_tokens`) and the pilots' compute a token (`panels.dense_rate`).

The rule (`plan`, over the runs of `--runs`, the published table's, else every run of the scope). A run is restated when the trace the published tables read of it has a factor above 1 and the steps
of its most common structure each read one value of every weight: the step's forwards then all read the same weights,
and its tokens are its forwards x rows x context. In a restated run:

- the most common structure's steps state the factor times their count, the traced count;
- another structure whose steps each read one value of every weight, and whose log holds a head forward (a product
  onto the vocabulary), states its own head forwards x rows. A validation pass recorded as steps of its own is one
  forward of what its record states, where the factor would count it once per forward of a training step, and an
  evaluation of more forwards than a training step is more than the factor gives it;
- every other structure keeps the factor times its count, which is what the token total gives every step of a record
  as it is. The case this leaves at the factor is a step that reads two values of a weight: a training step followed
  by the validation of the weights it wrote. The counter counted its batch and its validation windows once each, at
  their largest read, and the factor counts both, where forwards x rows would count the windows once per weight
  value they were read at;
- a run whose most common structure's head forwards x rows is not the traced count (an adapter's own product onto
  the vocabulary counts among the head's; a head whose products are sampled) keeps the factor in every structure.

A run whose most common structure reads two values of a weight in a step (a forward-only estimator that perturbs the
weight it trains) is not restated, nor is one with no trace or a factor of 1 or less; each is listed with why.

`steps[i].weights` names each updated weight a node of the step read with the digests of its values, one per write the
counter saw; `derived/N` operands are gradients' values and aside. A run of restate.json's `one_name` records two
tensors under one name (a block grown as a copy keeps the names of the block it copies), so two digests there are no
write between passes and every structure of it counts as reading one value.

PLAN.json is `{"restated": {RUN: {record, factor, tokens: {STRUCT: {BEFORE: AFTER}}, before_sha256, after_sha256,
structures}}, "kept": {RUN: why}}`: a step of a structure listed under `tokens` states AFTER where its record states
BEFORE, and the factor times its count otherwise. The restated record is `json.dumps(record).encode()` of the record
with only `steps[i].tokens` changed, and `after_sha256` is its digest, so a trace of it names the record it read.

`overlay` writes DIR, a state whose entries are links to STATE's, except `records` (links to STATE's record
directories, a restated run's being a copy that differs in its `run/run.json`) and `trace40` (links to STATE's
caches, `--drop-cache NAME` aside). `--only` restates just the listed runs: a rebuild keeps the record as it is, and
the published trace of it, for a run whose restated record did not trace. Nothing under STATE is written.

`totrace` says which restated records a run of correct.sh traces: all but the runs of `--leave`, which keep their
record as it is and the published trace of it, and the runs an earlier run of correct.sh traced (OUT, its out/, which
holds its plan.json and traced/). An OUT's trace of a record that is not this plan's, by `after_sha256`, is refused.
FILE is `{"trace": [RUN], "held": {RUN: OUT}, "left": [RUN]}`.

`layer` makes one trace layer of several (each a trace40.py node run's out/, traced by one te_core): BASE's traces
with each OVER's over them run by run, a later OVER over an earlier, only the runs of `--only` when given. A run an
OVER failed on keeps what the layer held of it before, an earlier OVER's trace of it included. FILE lists the runs taken from BASE, those from an OVER (`from_over`: the runs whose restated record
traced, which `overlay --only` takes) and the failures. replot.py's `node` reads DIR as a `--layer`.

`one` writes one restated record, for a reader that holds the record and the plan.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent


def one_name() -> dict[str, str]:
    return json.loads((HERE / "restate.json").read_text())["one_name"]


def forwards(structure: dict, vocab: int) -> tuple[int, int | None]:
    """A structure's head forwards and their rows: its forward `mm` products onto the vocabulary."""
    n, rows = 0, None
    for phase, _role, op, shapes, count, *_ in structure.get("log") or []:
        try:
            sh = json.loads(shapes)
        except (ValueError, TypeError):
            continue
        if phase == "fwd" and op == "mm" and len(sh) == 2 and sh[0] and sh[1] and sh[1][-1] == vocab:
            n += count
            rows = sh[0][0]
    return n, rows


def values(record: dict) -> dict[str, int]:
    """Per structure, the steps that read more than one value of some weight."""
    two = collections.Counter()
    for s in record.get("steps") or []:
        if any(len(v) > 1 for k, v in (s.get("weights") or {}).items() if not k.startswith("derived/")):
            two[str(s.get("struct"))] += 1
    return dict(two)


def plan(record: dict, factor: float, same_name: bool = False) -> dict:
    """What each structure of `record` states under the rule, or `{"kept": why}` for a run that is not restated."""
    if not factor or factor <= 1:
        return {"kept": f"the traced step consumed {factor:g} times the record's count" if factor else "no trace"}
    f = Fraction(factor).limit_denominator(1 << 20)
    vocab = (record.get("start") or {}).get("vocab")
    steps = collections.defaultdict(collections.Counter)
    for s in record.get("steps") or []:
        if s.get("tokens"):
            steps[str(s.get("struct"))][s["tokens"]] += 1
    if not steps:
        return {"kept": "no step states a token count"}
    two = {} if same_name else values(record)
    most = max(steps, key=lambda k: sum(steps[k].values()))
    if two.get(most):
        return {"kept": f"{two[most]} of its most common structure's {sum(steps[most].values())} steps read two values "
                        "of a weight"}
    head = {sid: forwards((record.get("structures") or {}).get(sid) or {}, vocab) for sid in steps}
    n, rows = head[most]
    whole = not (n and rows) or any(n * rows != b * f for b in steps[most])
    tokens, structures = {}, {}
    for sid, counts in steps.items():
        n, rows = head[sid]
        own = not whole and sid != most and bool(n and rows) and not two.get(sid)
        how = ("the factor: the most common structure" if sid == most else
               "the factor: the most common structure's head forwards x rows is not the traced count" if whole else
               "the factor: no head forward" if not (n and rows) else
               "the factor: a step reads two values of a weight" if two.get(sid) else "its head forwards x rows")
        after = {}
        for b in counts:
            new = Fraction(n * rows) if own else b * f
            if new.denominator != 1:
                raise ValueError(f"{b} tokens times {f} is no whole count")
            after[b] = int(new)
        if own and any(after[b] != b * f for b in counts):
            tokens[sid] = {str(b): after[b] for b in sorted(counts)}
        structures[sid] = {"how": how, "head_forwards": n, "head_rows": rows, "steps_reading_two_values": two.get(sid, 0),
                           "counts": [[b, after[b], counts[b]] for b in sorted(counts)]}
    return {"factor": float(f), "tokens": tokens, "structures": structures}


def restate(record: dict, factor: float, tokens: dict) -> None:
    """`record` with each step's tokens restated, in place."""
    f = Fraction(factor).limit_denominator(1 << 20)
    for s in record["steps"]:
        if not s.get("tokens"):
            continue
        new = Fraction(tokens.get(str(s.get("struct")), {}).get(str(s["tokens"]), s["tokens"] * f))
        if new.denominator != 1:
            raise ValueError(f"step {s.get('step')}'s {s['tokens']} tokens times {f} is no whole count")
        s["tokens"] = int(new)


def encode(record: dict) -> bytes:
    return json.dumps(record).encode()


def traced_factor(state: Path, layers: list[str], run: str) -> tuple[float | None, str | None]:
    """The factor of the first of the caches `layers` that holds a trace of `run`, and that cache."""
    for tree in layers:
        p = state / "trace40" / tree / "records" / f"{run}.jsonl.gz"
        if p.exists() and p.stat().st_size:
            with gzip.open(p, "rt") as fh:
                line = fh.readline()
            if line:
                trace = (json.loads(line).get("formalism") or {}).get("trace") or {}
                return trace.get("token_factor"), tree
    return None, None


def _plan_one(job: tuple) -> tuple[str, dict]:
    state, layers, run, art, same = job
    factor, tree = traced_factor(Path(state), layers, run)
    if not factor or factor <= 1:
        return run, {"kept": plan({}, factor)["kept"]}
    raw = (Path(state) / "records" / art / "run" / "run.json").read_bytes()
    record = json.loads(raw)
    p = plan(record, factor, same)
    if "kept" in p:
        return run, p
    restate(record, p["factor"], p["tokens"])
    return run, {"record": art, "trace_cache": tree, "before_sha256": hashlib.sha256(raw).hexdigest(),
                 "after_sha256": hashlib.sha256(encode(record)).hexdigest(), "steps": len(record["steps"]), **p}


def cmd_plan(a: argparse.Namespace) -> int:
    scope = {r["run_id"]: r["record"] for r in json.loads(a.scope.read_text())}
    same = one_name()
    if a.runs is not None:
        listed = set(a.runs.read_text().split())
        scope = {run: art for run, art in scope.items() if run in listed}
    jobs = [(str(a.state), a.layers.split(","), run, art, run in same) for run, art in sorted(scope.items())
            if (a.state / "records" / art / "run" / "run.json").is_file()]
    # the largest records first keeps the peak early and the pool busy
    jobs.sort(key=lambda j: -(a.state / "records" / j[3] / "run" / "run.json").stat().st_size)
    restated, kept = {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for run, p in ex.map(_plan_one, jobs):
            if "kept" in p:
                kept[run] = p["kept"]
            else:
                restated[run] = p
    doc = {"layers": a.layers.split(","), "restated": restated, "kept": kept,
           "no_record": sorted(set(scope) - set(restated) - set(kept))}
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    why = collections.Counter(w if w == "no trace" or w.startswith("the traced step") else "two values" for w in kept.values())
    print(f"{len(restated)} runs restated, {sum(bool(p['tokens']) for p in restated.values())} with a structure off the "
          f"factor; {len(kept)} kept as they are: {dict(why)}; no record: {len(doc['no_record'])}")
    return 0


def write_restated(src: Path, dst: Path, entry: dict) -> None:
    """`dst`, a record directory that links `src`'s files and holds the restated run.json, its digests checked."""
    (dst / "run").mkdir(parents=True)
    for q in src.iterdir():
        if q.name != "run":
            os.symlink(q, dst / q.name)
    for q in (src / "run").iterdir():
        if q.name != "run.json":
            os.symlink(q, dst / "run" / q.name)
    raw = (src / "run" / "run.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["before_sha256"]:
        raise SystemExit(f"{src}: not the record the plan read")
    record = json.loads(raw)
    restate(record, entry["factor"], entry["tokens"])
    out = encode(record)
    if hashlib.sha256(out).hexdigest() != entry["after_sha256"]:
        raise SystemExit(f"{src}: the restated record is not the plan's")
    (dst / "run" / "run.json").write_bytes(out)


def cmd_overlay(a: argparse.Namespace) -> int:
    restated = json.loads(a.plan.read_text())["restated"]
    if a.only is not None:
        only = set(a.only.read_text().split())
        restated = {r: v for r, v in restated.items() if r in only}
    by_art = {v["record"]: r for r, v in restated.items()}
    a.overlay.mkdir(parents=True)
    for p in a.state.iterdir():
        if p.name not in ("records", "trace40"):
            os.symlink(p, a.overlay / p.name)
    (a.overlay / "trace40").mkdir()
    for p in (a.state / "trace40").iterdir():
        if p.name != a.drop_cache:
            os.symlink(p, a.overlay / "trace40" / p.name)
    (a.overlay / "records").mkdir()
    done = []
    for p in (a.state / "records").iterdir():
        run = by_art.get(p.name)
        if run is None:
            os.symlink(p, a.overlay / "records" / p.name)
        elif a.reuse is not None:
            src = a.reuse / "records" / p.name
            if src.is_symlink() or not (src / "run" / "run.json").is_file():
                raise SystemExit(f"{run}: {a.reuse} holds no restated record of it")
            os.symlink(src, a.overlay / "records" / p.name)
            done.append(run)
        else:
            write_restated(p, a.overlay / "records" / p.name, restated[run])
            done.append(run)
    missing = sorted(set(restated) - set(done))
    a.out.write_text(json.dumps({"restated": sorted(done), "no_record": missing}, indent=1))
    print(f"{len(done)} restated records in {a.overlay}; no record: {missing}")
    return 1 if missing else 0


def cmd_totrace(a: argparse.Namespace) -> int:
    plan = json.loads(a.plan.read_text())["restated"]
    leave = {r for r in a.leave.split(",") if r}
    if leave - set(plan):
        raise SystemExit(f"totrace: the plan does not restate {sorted(leave - set(plan))}")
    held = {}
    for out in a.traced:
        theirs = json.loads((out / "plan.json").read_text())["restated"]
        for run, v in json.loads((out / "traced" / "traced.json").read_text()).items():
            f = out / "traced" / "records" / f"{run}.jsonl.gz"
            if run in leave or run not in plan or v.get("error") or not f.exists() or not f.stat().st_size:
                continue
            if theirs.get(run, {}).get("after_sha256") != plan[run]["after_sha256"]:
                raise SystemExit(f"totrace: {out} traced a record of {run} that is not this plan's")
            held[run] = str(out)
    doc = {"trace": sorted(set(plan) - leave - set(held)), "held": dict(sorted(held.items())), "left": sorted(leave)}
    a.out.write_text(json.dumps(doc, indent=1))
    print(f"totrace: {len(doc['trace'])} of {len(plan)} restated records to trace, {len(held)} traced by an earlier "
          f"run, {len(leave)} left: {doc['left']}")
    return 0


def cmd_layer(a: argparse.Namespace) -> int:
    base = json.loads((a.base / "manifest.json").read_text())
    (a.dest / "records").mkdir(parents=True)
    traced, source, failed = {}, {}, {}
    for run, v in json.loads((a.base / "traced.json").read_text()).items():
        if not v.get("error"):
            traced[run], source[run] = v, "base"
            shutil.copyfile(a.base / "records" / f"{run}.jsonl.gz", a.dest / "records" / f"{run}.jsonl.gz")
    only = None if a.only is None else set(a.only.read_text().split())
    for over in a.over:
        tree = json.loads((over / "manifest.json").read_text())["te_tree"]
        if tree != base["te_tree"]:
            raise SystemExit(f"layer: {a.base} is traced by te_core tree {base['te_tree']}, {over} by {tree}")
        for run, v in json.loads((over / "traced.json").read_text()).items():
            if only is not None and run not in only:
                continue
            f = over / "records" / f"{run}.jsonl.gz"
            if v.get("error") or not f.exists() or not f.stat().st_size:
                if source.get(run) != "over":
                    failed[run] = str(v.get("error") or "no records")[-300:]
                continue
            traced[run], source[run] = v, "over"
            failed.pop(run, None)
            shutil.copyfile(f, a.dest / "records" / f"{run}.jsonl.gz")
    (a.dest / "manifest.json").write_text(json.dumps({"te_commit": base["te_commit"], "te_tree": base["te_tree"],
                                                      "runs": len(traced)}))
    (a.dest / "traced.json").write_text(json.dumps(traced, indent=1, sort_keys=True, default=str))
    doc = {"runs": len(traced), "from_base": sorted(r for r, s in source.items() if s == "base"),
           "from_over": sorted(r for r, s in source.items() if s == "over"), "over_failed": failed}
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"layer: {len(traced)} runs, {len(doc['from_over'])} laid over, {len(doc['from_base'])} as the base holds "
          f"them, {len(failed)} failed: {sorted(failed)[:8]}")
    return 0


def cmd_one(a: argparse.Namespace) -> int:
    entry = json.loads(a.plan.read_text())["restated"][a.run]
    raw = a.record.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["before_sha256"]:
        raise SystemExit(f"{a.record}: not the record the plan read")
    record = json.loads(raw)
    restate(record, entry["factor"], entry["tokens"])
    out = encode(record)
    if hashlib.sha256(out).hexdigest() != entry["after_sha256"]:
        raise SystemExit(f"{a.record}: the restated record is not the plan's")
    a.out.write_bytes(out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--scope", type=Path, required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--runs", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("overlay")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--only", type=Path, default=None)
    p.add_argument("--reuse", type=Path, default=None)
    p.add_argument("--drop-cache", default="")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_overlay)
    p = sub.add_parser("totrace")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--traced", type=Path, action="append", default=[])
    p.add_argument("--leave", default="")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_totrace)
    p = sub.add_parser("layer")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--over", type=Path, action="append", required=True)
    p.add_argument("--only", type=Path, default=None)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_layer)
    p = sub.add_parser("one")
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_one)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
