"""Collective linkage: per-rank collective launch shards from the raw logs, joined across ranks by `(communicator, step, k)`.

    python -m tests.engine.collectives.collective_link extract LOG.jsonl.gz --out shard.rankR.json [--steps 0,1,3] [--rank R]
    python -m tests.engine.collectives.collective_link link shard.rank0.json shard.rank1.json [--instances 0=instances.rank0.jsonl ...] --out link.json
    python -m tests.engine.collectives.collective_link summary link.json

`extract` is one pass over a rank's log (torch-free; 53k events in ~0.3 s): for every collective launch (`collective_record.COLLECTIVE_KINDS`)
it records the operand and result views, every earlier launch record that referenced the operand allocation (the last one is the candidate
producer), every later launch record that referenced the result allocation until its free (the consumers), and the collective's position
in its communicator's call sequence.  `link` joins the shards rank by rank on that position, checks field agreement, names the Verity
definition each launch instantiates, and — when the fold's `instances.jsonl` rows are given — joins producer / consumer launches to their
instances by event id.

The schema and its rationale are in `collective_record.py`.  Nothing is inferred about ordering across ranks beyond NCCL's contract; the
"producer" is the last launch record that referenced the operand allocation — the fold decides which launch WROTE it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from verity_vllm.observe.events import DispatchEnter, DispatchExit, Event, Root, RunHeader, StepStart, StorageAlloc, StorageFree, TritonLaunch, ViewDesc, views_in
from verity_vllm.observe.log import read_log
from verity_vllm.observe.fold.views import byte_extent
from verity_vllm.engine.collectives.collective_record import COLLECTIVE_KINDS, COMMUNICATOR_KINDS, MISSING_IN_RECORD, SCHEMA_LINK, SCHEMA_SHARD, validate_launch

GROUP_NAME_RE = re.compile(r"^[a-z_]+:\d+$")   # vLLM GroupCoordinator.unique_name, e.g. "tp:0"


def view_json(v: ViewDesc | None) -> dict | None:
    if v is None:
        return None
    lo, hi = byte_extent(v)
    return {"alloc": v.alloc, "offset": v.offset, "shape": list(v.shape), "strides": list(v.strides), "dtype": v.dtype,
            "numel": v.numel, "byte_lo": lo, "byte_hi": hi}


def _scalars(args: list) -> list:
    return [a for a in args if not isinstance(a, (ViewDesc, list, tuple, dict))]


def _communicator(args: list, kwargs: dict) -> str | None:
    g = kwargs.get("group_name")
    if isinstance(g, str):
        return g
    for a in args:
        if isinstance(a, str) and GROUP_NAME_RE.match(a):
            return a
    return None


def _touch(eid: int, kind: str, op: str, module: str | None, step: int, arg: str | None, v: ViewDesc) -> dict:
    return {"event": eid, "kind": kind, "op": op, "module": module, "step": step, "arg": arg, "view": view_json(v)}


# ---------------------------------------------------------------------------------------------------------
# extract: one rank's shard
# ---------------------------------------------------------------------------------------------------------


def extract_rank_shard(events: Iterable[Event], rank: int | None = None, steps: set[int] | None = None,
                       max_touches: int = 8, max_consumers: int = 8, source: str | None = None) -> dict[str, Any]:
    """Per-rank collective launch shard (`SCHEMA_SHARD`) from one rank's event stream.  `steps` restricts the rows kept (the touch /
    consumer bookkeeping still runs over every event so producers of the kept rows are complete)."""
    header: RunHeader | None = None
    allocs: dict[int, dict] = {}
    root_allocs: dict[int, dict] = {}        # alloc -> {role, name} of the Root event that declared it (weights / buffers / tokens)
    freed: set[int] = set()
    touches: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_touches))
    open_results: dict[int, dict] = {}       # result alloc -> launch row (consumers still being collected)
    pending: dict[int, dict] = {}            # collective DispatchEnter id -> row awaiting its DispatchExit
    enter_ops: dict[int, tuple[str, str | None, int]] = {}   # open DispatchEnter id -> (op, module, step) for the exit's touches
    rows: list[dict] = []
    ordinal: Counter = Counter()             # step -> rows so far
    k_in_step: Counter = Counter()           # (step, communicator) -> communicator collectives so far
    seq: Counter = Counter()                 # communicator -> cumulative
    steps_seen: list[int] = []
    n_events = 0
    n_launches = 0

    def note_touch(eid: int, kind: str, op: str, module: str | None, step: int, arg: str | None, v: ViewDesc) -> None:
        touches[v.alloc].append(_touch(eid, kind, op, module, step, arg, v))
        row = open_results.get(v.alloc)
        if row is not None and eid != row["event"] and v.alloc not in freed and len(row["result_consumers"]) < max_consumers:
            row["result_consumers"].append(_touch(eid, kind, op, module, step, arg, v))

    for ev in events:
        n_events += 1
        if isinstance(ev, RunHeader):
            header = ev
        elif isinstance(ev, StepStart):
            steps_seen.append(ev.step)
        elif isinstance(ev, StorageAlloc):
            allocs[ev.alloc] = {"alloc": ev.alloc, "nbytes": ev.nbytes, "origin": ev.origin}
        elif isinstance(ev, Root):
            root_allocs.setdefault(ev.view.alloc, {"role": ev.role, "name": ev.name})
        elif isinstance(ev, StorageFree):
            freed.add(ev.alloc)
            row = open_results.pop(ev.alloc, None)
            if row is not None:
                row["result_freed"] = True
        elif isinstance(ev, DispatchEnter):
            n_launches += 1
            kind = COLLECTIVE_KINDS.get(ev.op)
            arg_views = list(views_in(ev.args))
            kw_views = [(k, v) for k, v in ev.kwargs.items() if isinstance(v, ViewDesc)]
            if kind is not None and (steps is None or ev.step in steps):
                ordinal[ev.step] += 1
                comm = _communicator(ev.args, ev.kwargs) if kind in COMMUNICATOR_KINDS else None
                if kind == "vocab_shard_lookup":
                    # out-variant: arg0 = out (written), arg1 = ids, arg2 = this rank's weight shard; the shard is the rank's operand
                    operand = arg_views[2] if len(arg_views) >= 3 else (arg_views[-1] if arg_views else None)
                    result = arg_views[0] if arg_views else None
                    sc = _scalars(ev.args)
                    rank_statics = {"org_vocab_start_index": sc[0], "org_vocab_end_index": sc[1]} if len(sc) >= 2 else {}
                else:
                    operand = arg_views[0] if arg_views else None
                    result = None
                    rank_statics = {}
                if comm is not None:
                    kk = k_in_step[(ev.step, comm)]
                    k_in_step[(ev.step, comm)] += 1
                    sq = seq[comm]
                    seq[comm] += 1
                else:
                    kk, sq = None, None
                is_root = operand is not None and (operand.alloc in root_allocs or (allocs.get(operand.alloc) or {}).get("origin") in ("param", "buffer"))
                row = {"rank": rank, "world_size": None, "event": ev.id, "step": ev.step, "phase": ev.phase, "kind": kind, "op": ev.op,
                       "module": ev.module, "stream": ev.stream, "communicator": comm, "k": kk, "seq": sq, "ordinal_in_step": ordinal[ev.step],
                       "reduce_op": "sum" if kind == "all_reduce" else None, "dtype": operand.dtype if operand else None,
                       "count": operand.numel if operand else None, "operand": view_json(operand),
                       "operand_alloc": allocs.get(operand.alloc) if operand else None,
                       "operand_is_root": is_root, "operand_root": root_allocs.get(operand.alloc) if operand else None,
                       # a Root (weight shard) has no producer launch: its touches are reads (e.g. the tied lm_head Gemm), not writes
                       "operand_touches": [] if is_root else (list(touches[operand.alloc]) if operand else []),
                       "operand_last_touch": (touches[operand.alloc][-1] if operand and not is_root and touches[operand.alloc] else None),
                       "operands_all": [view_json(v) for v in arg_views] if len(arg_views) > 1 else None,
                       "result": view_json(result), "result_consumers": [],
                       "result_first_consumer": None, "result_freed": False, "rank_statics": rank_statics, "scalars": _scalars(ev.args),
                       "schema": ev.schema}
                rows.append(row)
                pending[ev.id] = row
                if kind == "vocab_shard_lookup" and result is not None:
                    open_results[result.alloc] = row     # out-variant: the exit has no outs; the written view is arg 0
            elif kind is not None:
                # not a kept step: still advance the sequence counters so `k` / `seq` of kept rows are right
                ordinal[ev.step] += 1
                comm = _communicator(ev.args, ev.kwargs) if kind in COMMUNICATOR_KINDS else None
                if comm is not None:
                    k_in_step[(ev.step, comm)] += 1
                    seq[comm] += 1
            for v in arg_views:
                note_touch(ev.id, "dispatch-arg", ev.op, ev.module, ev.step, None, v)
            for kname, v in kw_views:
                note_touch(ev.id, "dispatch-arg", ev.op, ev.module, ev.step, kname, v)
            enter_ops[ev.id] = (ev.op, ev.module, ev.step)
        elif isinstance(ev, DispatchExit):
            row = pending.pop(ev.id, None)
            outs = list(views_in(ev.outs))
            op, module, step = enter_ops.pop(ev.id, ("?", None, -1))
            if row is not None and row["kind"] != "vocab_shard_lookup" and outs:
                row["result"] = view_json(outs[0])
                open_results[outs[0].alloc] = row
            for v in outs:
                # an op's own output is a touch of that allocation (the candidate producer of a later collective's operand);
                # it is not a consumer of an open result: a view-returning op (reshape/alias) already counted as a consumer on enter
                touches[v.alloc].append(_touch(ev.id, "dispatch-out", op, module, step, None, v))
        elif isinstance(ev, TritonLaunch):
            n_launches += 1
            for name, v in ev.args.items():
                if isinstance(v, ViewDesc):
                    note_touch(ev.id, "triton-arg", ev.fn, ev.module, ev.step, name, v)

    for row in rows:
        row["result_first_consumer"] = row["result_consumers"][0] if row["result_consumers"] else None
    tp2 = ((header.notes or {}).get("tp2") or {}) if header is not None else {}
    world = tp2.get("world_size")
    rk = tp2.get("rank", rank) if rank is None else rank
    for row in rows:
        row["rank"] = rk
        row["world_size"] = world
    return {"schema": SCHEMA_SHARD, "source": source, "rank": rk, "world_size": world,
            "profile_id": header.profile_id if header is not None else None,
            "parallelism": (header.profile or {}).get("parallelism") if header is not None else None,
            "events": n_events, "launch_records": n_launches, "steps": steps_seen, "kept_steps": sorted(steps) if steps is not None else None,
            "n_rows": len(rows), "rows_by_kind": dict(Counter(r["kind"] for r in rows)),
            "rows_per_step": {str(s): n for s, n in sorted(Counter(r["step"] for r in rows).items())},
            "communicators": sorted({r["communicator"] for r in rows if r["communicator"]}),
            "launches": rows}


def extract_log(path: Path, rank: int | None = None, steps: set[int] | None = None, **kw) -> dict[str, Any]:
    return extract_rank_shard(read_log(path), rank=rank, steps=steps, source=str(path), **kw)


# ---------------------------------------------------------------------------------------------------------
# link: join the shards
# ---------------------------------------------------------------------------------------------------------


def _geom(v: dict | None) -> tuple | None:
    return None if v is None else (tuple(v["shape"]), v["dtype"], v["numel"])


def _definition_for(kind: str, count: int | None, world: int | None, rows: dict[int, dict]) -> dict:
    if kind == "all_reduce":
        from verity.ml.definitions.registry import b1_tp2 as TP2
        from verity_vllm.program.quarantine.collective.allreduce import all_reduce_sum_bf16
        R = int(world) if world else len(rows)
        if R > 2:
            sp, module = (TP2.all_reduce(R, int(count)) if count else None), "verity.ml.definitions.registry.b1_tp2"
        else:
            sp, module = (all_reduce_sum_bf16(int(count), R) if count else None), "verity_vllm.program.quarantine.collective.allreduce"
        return {"id": sp.id if sp else None, "module": module, "conformance": sp.defn.conformance if sp else None,
                "operands": [f"x{r} = rank {r}'s operand view" for r in range(R)], "output": "every rank's result view (journaled write on each rank)"}
    if kind == "all_gather":
        from verity.ml.definitions.registry import b1_tp2 as TP2
        R = int(world) if world else len(rows)
        sp = TP2.all_gather(R, int(count)) if count and R >= 2 else None       # AllGather2_v1{N} at world 2, AllGather_v1{WORLD,N} above
        return {"id": sp.id if sp else None, "module": "verity.ml.definitions.registry.b1_tp2",
                "conformance": "reference rearrangement (reshape of the per-rank parts; no gates)",
                "operands": [f"x{r} = rank {r}'s operand view" for r in sorted(rows)], "output": "the concatenation, on every rank"}
    if kind == "vocab_shard_lookup":
        per_rank = {}
        for r, row in sorted(rows.items()):
            st = row.get("rank_statics") or {}
            op = row.get("operand") or {}
            VS = op.get("shape", [None])[0]
            H = op.get("shape", [None, None])[1] if len(op.get("shape", [])) > 1 else None
            per_rank[str(r)] = f"EmbeddingShard_v1{{VS={VS},H={H},START={st.get('org_vocab_start_index')}}}"
        return {"id": per_rank, "module": "verity.ml.definitions.registry.b1_tp2", "conformance": "exact-model-tested (TA1 TP=2 commit)",
                "operands": ["tok (I32)", "this rank's table shard"], "output": "the masked row, rank-local (no communicator)"}
    return {"id": None, "module": None, "conformance": None}


def _instances_index(inst_rows: list[dict] | None) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    by_event: dict[int, list[dict]] = defaultdict(list)
    by_out_alloc: dict[int, list[dict]] = defaultdict(list)
    for row in inst_rows or []:
        if "event" in row:
            by_event[int(row["event"])].append(row)
        for ov in row.get("output_views") or []:
            a = ov.get("alloc") if isinstance(ov, dict) else None
            if a is not None:
                by_out_alloc[int(a)].append(row)
    return by_event, by_out_alloc


def _inst_ref(row: dict) -> dict:
    return {k: row.get(k) for k in ("node", "spec", "row", "rows", "pattern", "name", "event", "step") if k in row}


def link_shards(shards: dict[int, dict], instances: dict[int, list[dict]] | None = None) -> dict[str, Any]:
    """Cross-rank link record (`SCHEMA_LINK`) from `{rank: shard}` (+ optional `{rank: instances.jsonl rows}`)."""
    ranks = sorted(shards)
    worlds = {s.get("world_size") for s in shards.values() if s.get("world_size") is not None}
    world = next(iter(worlds)) if len(worlds) == 1 else (len(ranks) if not worlds else None)
    inst_idx = {r: _instances_index((instances or {}).get(r)) for r in ranks}
    by_key: dict[tuple, dict[int, dict]] = defaultdict(dict)
    invalid: list[dict] = []
    for r in ranks:
        for row in shards[r]["launches"]:
            probs = validate_launch(row)
            if probs:
                invalid.append({"rank": r, "event": row.get("event"), "problems": probs})
            if row["kind"] in COMMUNICATOR_KINDS:
                key = (row["communicator"], row["step"], row["k"])
            else:
                key = ("local", row["step"], row["ordinal_in_step"])
            by_key[key][r] = row
    links: list[dict] = []
    for key in sorted(by_key, key=lambda t: (t[1], str(t[0]), t[2] if t[2] is not None else -1)):
        rows = by_key[key]
        present = sorted(rows)
        kinds = {row["kind"] for row in rows.values()}
        first = rows[present[0]]
        agreement = {
            "op": len({row["op"] for row in rows.values()}) == 1,
            "kind": len(kinds) == 1,
            "module": len({row["module"] for row in rows.values()}) == 1,
            "operand_geometry": len({_geom(row["operand"]) for row in rows.values()}) == 1,
            "result_geometry": len({_geom(row["result"]) for row in rows.values()}) == 1,
            "count": len({row["count"] for row in rows.values()}) == 1,
        }
        if len(present) != len(ranks) or (world is not None and len(present) != world):
            status = "missing-rank"
        elif not all(agreement.values()):
            status = "rank-mismatch"
        else:
            status = "linked"
        per_rank = {}
        for r in present:
            row = rows[r]
            by_event, by_out_alloc = inst_idx[r]
            prod = row["operand_last_touch"]
            prod_inst = None
            if prod is not None:
                cands = by_event.get(prod["event"]) or (by_out_alloc.get(row["operand"]["alloc"], []) if row["operand"] else [])
                prod_inst = [_inst_ref(c) for c in cands] or None
            cons_inst = []
            for c in row["result_consumers"]:
                for ir in by_event.get(c["event"], []):
                    cons_inst.append({"consumer_event": c["event"], **_inst_ref(ir)})
            per_rank[str(r)] = {"event": row["event"], "ordinal_in_step": row["ordinal_in_step"], "operand": row["operand"], "result": row["result"],
                                "operand_root": row.get("operand_root") if row.get("operand_is_root") else None,
                                "producer": prod, "producer_instance": prod_inst, "consumers": row["result_consumers"],
                                "first_consumer": row["result_first_consumer"], "consumer_instances": cons_inst or None,
                                "rank_statics": row["rank_statics"], "result_freed": row["result_freed"]}
        kind = first["kind"]
        link = {"key": {"communicator": key[0], "step": key[1], "k": key[2]}, "kind": kind, "op": first["op"], "module": first["module"],
                "reduce_op": first["reduce_op"], "dtype": first["dtype"], "count": first["count"], "world_size": world,
                "ranks": per_rank, "definition": _definition_for(kind, first["count"], world, rows),
                "required_writes": {str(r): rows[r]["result"] for r in present} if kind in COMMUNICATOR_KINDS else {},
                "agreement": agreement, "status": status}
        links.append(link)
    by_status = Counter(l["status"] for l in links)
    by_kind = Counter(l["kind"] for l in links)
    n_prod = sum(1 for l in links for pr in l["ranks"].values() if pr["producer"] is not None)
    n_root = sum(1 for l in links for pr in l["ranks"].values() if pr["operand_root"] is not None)
    n_prod_inst = sum(1 for l in links for pr in l["ranks"].values() if pr["producer_instance"])
    n_cons_inst = sum(len(pr["consumer_instances"] or []) for l in links for pr in l["ranks"].values())
    producers_by_op = Counter((l["kind"], pr["producer"]["op"]) for l in links for pr in l["ranks"].values() if pr["producer"] is not None)
    first_cons_by_op = Counter((l["kind"], pr["first_consumer"]["op"]) for l in links for pr in l["ranks"].values() if pr["first_consumer"] is not None)
    return {"schema": SCHEMA_LINK, "ranks": ranks, "world_size": world,
            "communicators": sorted({l["key"]["communicator"] for l in links if l["key"]["communicator"] != "local"}),
            "steps": sorted({l["key"]["step"] for l in links}),
            "n_links": len(links), "by_status": dict(by_status), "by_kind": dict(by_kind),
            "links_per_step": {str(s): n for s, n in sorted(Counter(l["key"]["step"] for l in links).items())},
            "producers_found": n_prod, "root_operands": n_root, "producers_by_op": {f"{k}|{op}": n for (k, op), n in sorted(producers_by_op.items())},
            "first_consumers_by_op": {f"{k}|{op}": n for (k, op), n in sorted(first_cons_by_op.items())},
            "instances_given": {str(r): len((instances or {}).get(r) or []) for r in ranks},
            "producer_instances_joined": n_prod_inst, "consumer_instances_joined": n_cons_inst,
            "invalid_rows": invalid, "missing_in_record": list(MISSING_IN_RECORD), "links": links}


def summary(link: dict) -> str:
    lines = [f"collective link: ranks {link['ranks']} world {link['world_size']} comms {link['communicators']} steps {link['steps']}",
             f"  links {link['n_links']}  by_status {link['by_status']}  by_kind {link['by_kind']}",
             f"  per step {link['links_per_step']}",
             f"  producers found {link['producers_found']} (root operands {link['root_operands']})  by op {link['producers_by_op']}",
             f"  first consumers by op {link['first_consumers_by_op']}",
             f"  instances given {link['instances_given']}  producer instances joined {link['producer_instances_joined']}  consumer instances joined {link['consumer_instances_joined']}",
             f"  invalid rows {len(link['invalid_rows'])}"]
    for l in link["links"][:6]:
        r0 = l["ranks"][str(link["ranks"][0])]
        prod = r0["producer"]
        fc = r0["first_consumer"]
        lines.append(f"  {l['key']} {l['kind']} {l['module']} N={l['count']} -> {l['definition']['id'] if isinstance(l['definition']['id'], str) else 'per-rank'} "
                     f"[{l['status']}] rank0: event {r0['event']} operand a{(r0['operand'] or {}).get('alloc')} <- {prod['event'] if prod else None} {prod['op'] if prod else None}; "
                     f"result a{(r0['result'] or {}).get('alloc')} -> {fc['event'] if fc else None} {fc['op'] if fc else None}")
    return "\n".join(lines)


def _open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def load_shard(path: Path) -> dict:
    with _open_text(Path(path)) as f:
        return json.load(f)


def load_instances(path: Path) -> list[dict]:
    rows = []
    with _open_text(Path(path)) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if "event" in d or "output_views" in d:
                    rows.append(d)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("log")
    e.add_argument("--rank", type=int, default=None)
    e.add_argument("--steps", default=None, help="comma-separated engine steps to keep (default all)")
    e.add_argument("--max-touches", type=int, default=8)
    e.add_argument("--max-consumers", type=int, default=8)
    e.add_argument("--compact", action="store_true", help="no indentation (fixtures)")
    e.add_argument("--out", required=True)
    l = sub.add_parser("link")
    l.add_argument("shards", nargs="+")
    l.add_argument("--instances", action="append", default=[], help="RANK=instances.jsonl (repeatable)")
    l.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("link")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        steps = {int(x) for x in a.steps.split(",")} if a.steps else None
        sh = extract_log(Path(a.log), rank=a.rank, steps=steps, max_touches=a.max_touches, max_consumers=a.max_consumers)
        Path(a.out).write_text((json.dumps(sh, separators=(",", ":")) if a.compact else json.dumps(sh, indent=1)) + "\n")
        print(f"[collective_link] rank {sh['rank']} world {sh['world_size']}: {sh['n_rows']} rows {sh['rows_by_kind']} -> {a.out}", file=sys.stderr)
        return 0
    if a.cmd == "link":
        shards = {}
        for p in a.shards:
            sh = load_shard(Path(p))
            shards[int(sh["rank"])] = sh
        inst = {}
        for spec in a.instances:
            r, p = spec.split("=", 1)
            inst[int(r)] = load_instances(Path(p))
        rec = link_shards(shards, inst or None)
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
        print(summary(rec), file=sys.stderr)
        return 0
    if a.cmd == "summary":
        print(summary(json.loads(Path(a.link).read_text())))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
