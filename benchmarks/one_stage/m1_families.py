"""M1's family census: a rank's Programs of record (a deferred TP Commit's Build, as `tp.serving_rows.window_of_record`
loads them), every Program row of the given request steps by its Definition (the row's ``spec``), with what the serving
rows' partition makes of it: its unit templates (`serving_rows.unit_templates`: the row itself, and a ``CUT`` family's
batched members), whether it is a template the rows commit (`serving_rows.template`: u16 ports, else the reason), the rows
`describe` cannot address, its port words (`serving_rows.members`' rule) and its Definition's word gates. No value is read.
``--estimate`` adds each family's Flock size to a census: every distinct primitive (or Boolean Definition) of the row's
Definition lowered once on its own (`ir_lower.lower` on fresh inputs), its ANDs, table reads and assertions times how many
times the Definition holds it. Constants folded and ANDs shared across leaves are not taken off, so it is an upper bound.
With it, M0's block for the unit (`block`): the in-circuit hashing of its rows alone bounds it from below, so a family whose
``rows_log`` passes `K_MAX` is refused whatever its ANDs.  The same is computed for each family's split units (its unit
templates past the row itself: ``units``), and for each ``--also`` Definition (``candidates``: a split's unit main lacks).

    PYTHONPATH=... python benchmarks/one_stage/m1_families.py --commit ROW/commit --build-root ROW/build \\
      --workload integrations/vllm/workloads/ROW.json --rank 0 --steps 16 --out families.json
    PYTHONPATH=...:<tree>/verity/ml/flock/python python benchmarks/one_stage/m1_families.py --estimate families.json --out sized.json \\
      [--also 'CublasSplitKGemmF32Out_v1{K=6144,N=1,..}' ...] [--also-file SPECS.txt]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

#: Flock's largest block (`circuit.K_MAX`) and the statement limit the brief measures against
K_MAX, LIMIT_LOG = 27, 26


def _leaf_cost(fn, IL) -> dict:
    from verity.ml.circuits.boolean.forms import Circuit

    C = Circuit()
    params, _ret = fn.signature()
    ins = [C.input(f"a{i}.{j}", w) for i, (_n, t) in enumerate(params) for j, w in enumerate(t.leaf_widths())]
    outs = IL.lower(C, fn, ins)
    C.output("o", [b for w in outs for b in w])
    c = C.counts()
    return {"ands": c["ands"], "reads": c["lookups"], "asserts": c["asserts"]}


def estimate(fn, memo: dict, leaves: dict) -> dict:
    """fn's ANDs, table reads and assertions as the sum of its leaves' (a primitive or a Boolean Definition, each lowered once
    by `_leaf_cost`), each times how often fn holds it; ``unpieced`` the leaves Flock has no piece for."""
    from verity_flock import ir_lower as IL

    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.program import is_input

    if fn.id in memo:
        return memo[fn.id]
    boolean = callable(getattr(getattr(fn, "defn", fn), "build", None))
    if isinstance(fn, PrimitiveDefinition) or boolean:
        fam = getattr(fn, "family", None)
        if is_input(fn) or fam in ("Input", "Const"):
            r = {"ands": 0, "reads": 0, "asserts": 0, "unpieced": []}
        else:
            try:
                r = {**_leaf_cost(fn, IL), "unpieced": []}
            except IL.NoPiece:
                r = {"ands": 0, "reads": 0, "asserts": 0, "unpieced": [fn.id]}
        leaves[fn.id] = {k: v for k, v in r.items() if k != "unpieced"}
    else:
        r = {"ands": 0, "reads": 0, "asserts": 0, "unpieced": []}
        for nd in fn.body.nodes:
            if is_input(nd.fn):
                continue
            n = nd.n if nd.form in ("batch", "scan") else 1
            s = estimate(nd.fn, memo, leaves)
            for k in ("ands", "reads", "asserts"):
                r[k] += n * s[k]
            r["unpieced"] = sorted(set(r["unpieced"]) | set(s["unpieced"]))
    memo[fn.id] = r
    return r


def block(fn, ands: int | None) -> dict:
    """The log2 of M0's block for fn's unit as `m1_unit.py` stages it (a ``row/v2`` bit row per parameter, an output row per
    returned field), placed by Flock's own rule (`circuit._ranges`): its rows' compressions, the hm96 slots, the mask and
    the unit's slot, its table reads' slots left out, one unit to the block (the stager packs more while the block stays
    within 2^22). ``rows_log`` takes the unit's slot at its least, a lower bound for any number of ANDs; ``est_log`` sizes
    it at the estimate's ANDs plus the port bits. The unit fits a block at most `K_MAX`."""
    from verity_flock import circuit as FC

    params, ret = fn.signature()
    fields = [f[1] if isinstance(f, tuple) else f for f in ret.fields] if hasattr(ret, "fields") else [ret]
    bits = [sum(t.leaf_widths()) for _n, t in params] + [16 * -(-sum(f.leaf_widths()) // 16) for f in fields]
    ports = [FC.Port(f"p{i}", -(-b // 16), b) for i, b in enumerate(bits)]
    n_sha = FC._row_slots(ports)[0]

    def log(unit_rows: int) -> int:
        return FC._ranges([("unit", FC.SHA_LOG, n_sha, {"packed": True}), ("unit", FC.HM_LOG, FC._pow2(len(ports)), {}),
                           ("unit", FC._slot_log(unit_rows), 1, {}), ("mask", FC.MASK_LOG, 1, {})])[1]

    return {"row_bits": sum(bits), "compressions": sum(p.comps for p in ports), "rows_log": log(1),
            "est_log": None if ands is None else log(ands + sum(bits))}


def sized(spec: str, memo: dict) -> dict:
    """`estimate` and `block` of the Definition ``spec`` binds to (`serving_rows._bind`), or the reason it has none."""
    import math

    from verity_vllm.pipeline import serving_rows as SRW

    t = time.time()
    try:
        sp = SRW._bind(spec)
        e = estimate(sp, memo, {}) if sp is not None else {"error": "does not bind"}
    except Exception as ex:  # noqa: BLE001 -- reported, not fatal
        e = {"error": f"{type(ex).__name__}: {ex}"[:300]}
    if "ands" in e:
        e["log2_ands"] = round(math.log2(e["ands"]), 2) if e["ands"] else None
        e["under_2^26"] = e["ands"] < 2**LIMIT_LOG
        e["block"] = b = block(sp, e["ands"])
        e["fits_block"] = b["est_log"] <= K_MAX
        e["refused_block"] = b["rows_log"] > K_MAX
    e["seconds"] = round(time.time() - t, 1)
    return e


def _say(rid: str, rows: int | str, spec: str, e: dict) -> None:
    print(f"M1S {rid} {rows:>5} {spec[:90]:90s} ands={e.get('ands')} log2={e.get('log2_ands')} reads={e.get('reads')} "
          f"asserts={e.get('asserts')} unpieced={e.get('unpieced')} block={e.get('block')} {e.get('error', '')}", flush=True)


def sizes(census: str, out: str, also: list[str] = ()) -> int:
    """Each family's estimate, each of its split units' (its `unit_templates` past the row itself, ``units``: per row ``n``
    of them), and with ``also`` each of those Definitions' (``candidates``: the units of a split main doesn't have)."""
    import sys

    sys.setrecursionlimit(100000)
    doc = json.loads(Path(census).read_text())
    memo: dict = {}
    seen: dict[str, dict] = {}
    for rid, req in doc["requests"].items():
        for spec, f in req["families"].items():
            f["estimate"] = e = sized(spec, memo)
            _say(rid, f["rows"], spec, e)
            e["units"] = []
            for u, n in f.get("unit_templates", [])[1:]:
                if u not in seen:
                    seen[u] = sized(u, memo)
                    _say(rid, f"x{n}", u, seen[u])
                e["units"].append({"spec": u, "n": n, **seen[u]})
    doc["candidates"] = {}
    for s in also:
        doc["candidates"][s] = sized(s, memo)
        _say("candidate", "-", s, doc["candidates"][s])
    Path(out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", default=None, metavar="CENSUS", help="add each family's estimated size to this census's output")
    ap.add_argument("--also", action="append", default=[], metavar="SPEC",
                    help="with --estimate: also size this Definition (a split's unit that main doesn't have); repeatable")
    ap.add_argument("--also-file", default=None, metavar="PATH", help="with --estimate: --also for each line of this file")
    ap.add_argument("--commit", help="the deferred TP Commit's --out (runs.jsonl, versions.json)")
    ap.add_argument("--build-root", help="each rank's Programs of record in rank<r>/")
    ap.add_argument("--workload")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--steps", default="16", help="comma-separated request steps described per family")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.estimate:
        also = a.also + ([s for s in Path(a.also_file).read_text().splitlines() if s.strip()] if a.also_file else [])
        return sizes(a.estimate, a.out, also)
    if not (a.commit and a.build_root and a.workload):
        ap.error("--commit, --build-root and --workload, or --estimate")

    from verity_vllm.pipeline import serving_rows as SRW
    from verity_vllm.pipeline.tp import rank_replay as RR
    from verity_vllm.pipeline.tp import serving_rows as TS
    from verity_vllm.pipeline.workload_file import workload_file

    t0 = time.time()
    rec = TS.record(a.commit)
    world = int(rec["world"])
    doc = json.loads(workload_file(a.workload).read_text())
    if hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != rec["workload_sha256"]:
        raise SystemExit(f"{a.workload}: not the workload the record served")
    reqs = [{"request_id": f"r{i}", "arrive_step": int(q.get("arrive_step") or 0), "prompt_len": len(q["prompt_token_ids"]),
             "max_tokens": int(q["max_tokens"])} for i, q in enumerate(doc["requests"])]
    p = RR.load({"programs": {str(a.rank): str(Path(a.build_root) / f"rank{a.rank}")}, "of_record": rec["of_record"], "requests": reqs},
                a.rank, world, definitions=False)
    load_s = round(time.time() - t0, 1)
    print(f"M1F loaded rank {a.rank} of {world}: {len(p.progs)} request Programs in {load_s} s", flush=True)

    want = [int(s) for s in a.steps.split(",")]
    out: dict = {"rank": a.rank, "world": world, "steps": want, "programs": dict(p.digs), "load_s": load_s, "requests": {}}
    for rid, inst in p.progs.items():
        pi = p.index if rid == next(iter(p.progs)) else None
        if pi is None:
            from verity_vllm.check.replay.index import ProgramIndex
            pi = ProgramIndex(inst, program_dir=p.dirs.get(rid), attention_paths=p.attn_paths, tp=p.tp)
        all_steps: Counter = Counter()
        per_step: Counter = Counter()
        fam: dict[str, dict] = defaultdict(lambda: {"rows": 0, "undescribed": 0, "op_paths": Counter(), "steps": Counter()})
        t1 = time.time()
        for r in pi.rows:
            s = pi.step_of(int(r["i"]))
            all_steps[r["spec"]] += 1
            per_step[s] += 1
            if s not in want:
                continue
            f = fam[r["spec"]]
            f["rows"] += 1
            f["steps"][s] += 1
            d = pi.describe(r)
            if d is None:
                f["undescribed"] += 1
                continue
            f["op_paths"][re.sub(r"\.layers\.\d+\.", ".layers.*.", d["op_path"])] += 1
            if "ports" not in f:
                f["ports"] = {"in": [sum(int(sl[3]) - int(sl[2]) for sl in grp) for grp in r["args"]],
                              "members": [[m[0], int(m[1]), str(m[2])] for m in d["members"]]}
        fams = {}
        for spec, f in sorted(fam.items(), key=lambda kv: -kv[1]["rows"]):
            units = SRW.unit_templates(spec)
            try:
                t = SRW.template(spec)
                tmpl = {"ok": True, "layout": t.layout, "word": t.word}
            except (LookupError, ValueError) as e:
                tmpl = {"ok": False, "why": str(e)[:300]}
            gates = None
            try:
                sp = SRW._bind(spec)
                gates = int(sp.gates) if sp is not None else None
            except Exception as e:  # noqa: BLE001 -- reported, not fatal
                gates = f"error: {type(e).__name__}: {e}"[:200]
            fams[spec] = {"rows": f["rows"], "undescribed": f["undescribed"], "steps": dict(f["steps"]), "whole_request": all_steps[spec],
                          "unit_templates": [list(u) for u in units], "template": tmpl, "word_gates": gates, "ports": f.get("ports"),
                          "op_paths": dict(f["op_paths"].most_common(6)), "distinct_op_paths": len(f["op_paths"])}
        out["requests"][rid] = {"rows": len(pi.rows), "steps": len(pi.segs), "rows_per_step": [per_step[s] for s in range(len(pi.segs))],
                                "families": fams, "seconds": round(time.time() - t1, 1)}
        print(f"M1F {rid}: {len(pi.rows)} rows, {len(pi.segs)} steps; steps {want}: {sum(f['rows'] for f in fams.values())} rows of "
              f"{len(fams)} Definitions ({round(time.time() - t1, 1)} s)", flush=True)
        for spec, f in fams.items():
            print(f"M1F   {f['rows']:7d}  {spec[:110]:110s}  template={'ok:' + f['template']['layout'] if f['template']['ok'] else 'NO'}"
                  f"  undescribed={f['undescribed']}  gates={f['word_gates']}", flush=True)
    out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"M1F wrote {a.out}; peak RSS {out['peak_rss_gb']} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
