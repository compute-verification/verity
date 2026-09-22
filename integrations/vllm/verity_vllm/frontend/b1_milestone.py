"""B1 authoring milestone: the torch-authored composites (`b1_authored`) against TA1's hand-built ones.

For each composite: the authored program's digest, TA1's, whether they are equal, and — where they are not —
the correspondence verdict (`verity_vllm.frontend.correspond`: same gates, same producers, same ordered
reference sequences, same `S.out`), plus gates, wall time, dead gates and, for `Serve`, the descriptor size.

    python -m verity_vllm.frontend.b1_milestone --config B0 --lp 32 --steps 7 --out /tmp/fe2-b0.json
    python -m verity_vllm.frontend.b1_milestone --config B1 --lp 512 --steps 15 --only Serve --out /tmp/fe2-b1.json

Torch-dependent: runs on the pod. B1-scale rows must never run on a workstation (descriptor: >6 GB RSS).
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import sys
import time
import warnings

from verity.ir import Program, bind
from verity.ir.codec import descriptor_size, program_digest
from verity_vllm.registry import b1
from verity_vllm.frontend import b1_authored as A
from verity_vllm.frontend.correspond import correspond
from verity_vllm.frontend.torch_frontend import DeadGateWarning, export_report


def _rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 ** 3) if sys.platform == "darwin" else ru / (1024 ** 2)


def row(label: str, authored, ta1_def, statics: dict, *, size: bool = False, id_limit: int = 64) -> dict:
    print(f"== {label}", flush=True)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeadGateWarning)
        rep = export_report(authored, static=statics, dead_gates="warn", id_limit=id_limit)
    t1 = time.time()
    P_a = rep.program
    P_t = Program(bind(ta1_def, **statics))
    d_t = program_digest(P_t)
    t2 = time.time()
    corr = correspond(P_a, P_t)
    t3 = time.time()
    out = {
        "composite": label, "root": rep.root, "gates": P_a.gates, "gates_ta1": P_t.gates, "input_gates": P_a.input_gates,
        "digest_authored": rep.digest, "digest_ta1": d_t, "digest_equal": rep.digest == d_t,
        "corresponds": corr.corresponds, "identical_defs": corr.identical, "same_reference_defs": corr.same_references,
        "same_reference_ids": corr.same_reference_ids[:20], "mismatches": [m.to_json() for m in corr.mismatches][:10],
        "unverified": [u.to_json() for u in corr.unverified][:10], "root_out": corr.root_out,
        "dead_gates": rep.dead.to_json(), "unreachable_gates": {"total": rep.unreachable.total, "computed": rep.unreachable.computed,
                                                                 "input": rep.unreachable.input},
        "dead_gate_warning": bool(caught),
        "wall_authored_s": round(t1 - t0, 2), "wall_ta1_s": round(t2 - t1, 2), "wall_correspond_s": round(t3 - t2, 2),
        "machine": platform.node(), "torch": rep.provenance["torch"], "registry_digest": rep.provenance["registry"]["digest"],
        "max_rss_gb": round(_rss_gb(), 2),
    }
    if size:
        t4 = time.time()
        out["descriptor_bytes_authored"] = descriptor_size(P_a)
        gc.collect()
        out["descriptor_bytes_ta1"] = descriptor_size(P_t)
        out["wall_descriptor_s"] = round(time.time() - t4, 1)
        out["max_rss_gb"] = round(_rss_gb(), 2)
    verdict = "digest-equal" if out["digest_equal"] else ("corresponds" if corr.corresponds else "MISMATCH")
    print(f"   {verdict}: gates={P_a.gates} (ta1 {P_t.gates}; inputs {P_a.input_gates}) dead={rep.dead.total} "
          f"(computed {rep.dead.computed}, input {rep.dead.input}; on S.out {rep.dead.on_s_out}) "
          f"authored {out['wall_authored_s']}s ta1 {out['wall_ta1_s']}s corr {out['wall_correspond_s']}s "
          f"same-ref defs={corr.same_references} mismatches={len(corr.mismatches)} rss={out['max_rss_gb']}GB", flush=True)
    for m in corr.mismatches[:5]:
        print(f"   mismatch {m.id}: {m.detail[:300]}", flush=True)
    del P_a, P_t, rep, corr
    gc.collect()
    return out


def rows_for(C: dict, LP: int, STEPS: int, only: set[str] | None) -> list[dict]:
    H, D, NH, KVH = C["H"], C["D"], C["NH"], C["KVH"]
    plan = [
        ("RMSNormTriton", A.RMSNormTriton, b1.RMSNormTriton, {"N": H, "EPS": C["EPS"]}, {}),
        ("RMSNormFusedCuda", A.RMSNormFusedCuda, b1.RMSNormFusedCuda, {"N": H, "EPS": C["EPS"]}, {}),
        ("AttnBlock", A.AttnBlock, b1.AttnBlock, {"D": D, "NVIS": 64, "FIRST": True}, {}),
        ("AttnBlock", A.AttnBlock, b1.AttnBlock, {"D": D, "NVIS": 37, "FIRST": False}, {}),
        ("AttentionHead", A.AttentionHead, b1.AttentionHead, {"T": 70, "D": D}, {}),
        ("Attention", A.Attention, b1.Attention, {"T": 5, "NH": NH, "KVH": KVH, "D": D}, {}),
        ("LayerPre", A.LayerPre, b1.LayerPre, {"C": C, "L0": True}, {}),
        ("LayerPre", A.LayerPre, b1.LayerPre, {"C": C, "L0": False}, {}),
        ("LayerPost", A.LayerPost, b1.LayerPost, {"C": C}, {}),
        ("Final", A.Final, b1.Final, {"C": C}, {}),
        ("Serve", A.Serve, b1.Serve, {"C": C, "LP": LP, "STEPS": STEPS}, {"size": True}),
    ]
    out = []
    for label, fa, ft, st, kw in plan:
        if only and label not in only:
            continue
        shown = ",".join(f"{k}={C['model'] if k == 'C' else v}" for k, v in st.items())
        out.append(row(f"{label}{{{shown}}}", fa, ft, st, **kw))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["B0", "B1"], default="B0")
    ap.add_argument("--lp", type=int, default=32)
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    C = b1.B0_CONFIG if a.config == "B0" else b1.B1_CONFIG
    only = set(a.only.split(",")) if a.only else None
    t0 = time.time()
    rows = rows_for(C, a.lp, a.steps, only)
    rec = {"config": a.config, "LP": a.lp, "STEPS": a.steps, "rows": rows, "wall_total_s": round(time.time() - t0, 1),
           "machine": platform.node()}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"wrote {a.out}")
    bad = [r for r in rows if not (r["digest_equal"] or r["corresponds"])]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
