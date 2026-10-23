"""M1's family census: a rank's Programs of record (a deferred TP Commit's Build, as `tp.serving_rows.window_of_record`
loads them), every Program row of the given request steps by its Definition (the row's ``spec``), with what the serving
rows' partition makes of it: its unit templates (`serving_rows.unit_templates`: the row itself, and a ``CUT`` family's
batched members), whether it is a template the rows commit (`serving_rows.template`: u16 ports, else the reason), the rows
`describe` cannot address, its port words (`serving_rows.members`' rule) and its Definition's word gates. No value is read.

    PYTHONPATH=... python benchmarks/one_stage/m1_families.py --commit ROW/commit --build-root ROW/build \\
      --workload integrations/vllm/workloads/ROW.json --rank 0 --steps 16 --out families.json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True, help="the deferred TP Commit's --out (runs.jsonl, versions.json)")
    ap.add_argument("--build-root", required=True, help="each rank's Programs of record in rank<r>/")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--steps", default="16", help="comma-separated request steps described per family")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from verity_vllm.pipeline import serving_rows as SRW
    from verity_vllm.pipeline.tp import rank_replay as RR
    from verity_vllm.pipeline.tp import serving_rows as TS

    t0 = time.time()
    rec = TS.record(a.commit)
    world = int(rec["world"])
    doc = json.loads(Path(a.workload).read_text())
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
