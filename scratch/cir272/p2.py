"""p2.py COVER.json (v3): CIR-272's second pass over the types v3 left without a statement (refused or not run) and the 18
extras. For each: an estimate of its stager (seconds, GB), circuit (GiB) and Lean peak (4.42 x circuit GiB, the largest
ratio over the accepted circuits past 5 GiB), each from a named measurement, marked measured or inferred; whether it runs
(stager under 62 GB and Lean under 60 GB by the estimate); and, run, its place, cheapest first by the estimated time, dealt
alternately to two lists. Writes, beside this file: types-p2a.tsv, types-p2b.tsv (TAG, UNIT, 1, session, LIMIT_S) and
p2-plan.json."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
GIB = 2**30
RATIO = 4.42          # Lean peak / circuit GiB, at most, over CIR-272's accepted circuits past 5 GiB
TOTAL = 2.56          # a type's whole time / its staging, the median over CIR-272's accepted types past 2M ANDs
STAGE_GB, LEAN_GB = 62, 60

cover = json.loads(Path(sys.argv[1]).read_text())
assert cover["schema"] == "circuits/request-cover/v3", cover["schema"]
T = cover["bound_runs"]["types"]
idx = json.loads((HERE / "types.json").read_text())
todo = sorted(t for t, r in T.items() if r["outcome"] != "accepted")
assert len(todo) == 159 and sum(T[t]["extra"] for t in todo) == 18, len(todo)


def est(stage_s=None, stage_gb=None, circuit_bytes=None, total_s=None, basis="", kind="inferred", **kw):
    e = {"basis": basis, "kind": kind, **kw}
    if stage_s is not None:
        e["stage_s"] = round(stage_s)
    if stage_gb is not None:
        e["stage_gb"] = round(stage_gb, 1)
    if circuit_bytes is not None:
        e["circuit_gib"] = round(circuit_bytes / GIB, 2)
        e["lean_gb"] = round(RATIO * circuit_bytes / GIB, 1)
    if total_s is None and stage_s is not None:
        total_s = TOTAL * stage_s
    if total_s is not None:
        e["total_s"] = round(total_s)
    return e


# measured relatives, read in this session from their runs' stage.json, job.log and done files
HB1 = dict(run="r20261112-143157-9dd0", circuit=12788381484, gb=65.85, s=2545.8)      # SparseMlaHeadBlock_v1 NVIS=64 FIRST=false
HB2 = dict(run="r20261112-201828-47ac", circuit=12191652715, gb=202.63, s=5610.1)     # SparseMlaHeadBlock_v2 (its Boolean version)
LOGITS1 = dict(run="r20261112-125341-11a8", n=2, ands=1087058, circuit=328220292, gb=1.58, s=42.9)  # DsaMqaLogits_v3{N=1}
SILU256 = dict(run="r20261112-125341-11a8", n=2, ands=362496, circuit=127547077, gb=1.2, s=36.4)    # SiluMul_v2{I=256}
GREEDY0 = dict(run="r20261112-140245-1147", ands=2031531, circuit=219389233, gb=2.34, s=55.1)       # GreedyChunk_v1 FIRST=false
SUMSQ = dict(tag="t101", ands=10894748, circuit=3.87e9, gb=11.6, s=270.5, total=710)              # RmsNormMnnvlSumSq_v1{N=6144}


def plan(t: str) -> dict:
    r, b = T[t], idx[t]["block"]
    fam = b.split("{")[0]
    nd = r.get("needs") or {}
    if t in ("t019", "t031"):
        w = r["stage"]
        share = w["circuit_bytes"] / HB1["circuit"]
        lo = share * HB2["gb"]
        hi = w["maxrss_gb"] * HB2["gb"] / HB1["gb"]
        e = est(stage_s=w["wall_s"] * HB2["s"] / HB1["s"], stage_gb=hi, circuit_bytes=w["circuit_bytes"] * HB2["circuit"] / HB1["circuit"],
                basis=(f"its word form's own staging ({r['run']}: {w['circuit_bytes']} bytes, {w['maxrss_gb']} GB, {w['wall_s']} s) scaled by "
                       f"SparseMlaHeadBlock_v2 over _v1 at NVIS=64 ({HB2['run']}: {HB2['circuit']} bytes, {HB2['gb']} GB, {HB2['s']} s at 16 "
                       f"CPUs; {HB1['run']}: {HB1['circuit']} bytes, {HB1['gb']} GB, {HB1['s']} s); by its share of the head block's "
                       f"circuit instead, {lo:.0f} GB"), stage_gb_range=[round(lo, 1), round(hi, 1)],
                staged=r.get("needs", {}).get("stage"))
    elif r.get("why") == "stage-past-62GB":
        e = est(stage_gb=62.1, basis=f"its own run {r['run']}: the stager passed 62 GB at {nd.get('stage_s')} s", kind="measured")
    elif r.get("why") == "past-2700s":
        sib = nd["sibling"]
        e = est(stage_s=2725, circuit_bytes=sib["circuit_bytes"] * 3, basis=(f"its own run {r['run']}: not staged at 2,725 s [measured]; its "
                f"circuit taken as 3 x K=2048's ({sib['tag']}, {sib['circuit_bytes']} bytes) [inferred]"), kind="measured stage, inferred circuit")
    elif r.get("why") == "lean-past-64GB":
        e = est(stage_s=nd["stage_s"], stage_gb=nd["stage_gb"], circuit_bytes=nd["circuit_bytes"],
                total_s=nd["stage_s"] * 1371 / 441.3,
                basis=(f"its own run {r['run']}: staged in {nd['stage_s']} s at {nd['stage_gb']} GB to {nd['circuit_bytes']} bytes [measured]; "
                       "Lean peak 4.42 x its GiB, time 3.1 x its staging as t104's (1,371 s over 441 s) [inferred]"),
                kind="measured circuit, inferred Lean peak")
    elif "fit" in nd:
        f = nd["fit"]
        e = est(stage_s=f["stage_s"], stage_gb=f["stage_gb"], circuit_bytes=f["circuit_gib"] * GIB, total_s=f["stage_s"] * 1371 / 441.3,
                basis="linear fit of SparseMlaHeadScores_v1 FIRST=true over NVIS 10-28, CIR-272's runs; time 3.1 x staging as t104's")
    elif fam == "DsaMqaLogits_v3":
        n = int(re.search(r"N=(\d+)", b)[1])
        a = LOGITS1["ands"] / LOGITS1["n"] * n
        e = est(stage_s=LOGITS1["s"] * a / LOGITS1["ands"], stage_gb=max(1.6, LOGITS1["gb"] * a / LOGITS1["ands"]),
                circuit_bytes=LOGITS1["circuit"] * a / LOGITS1["ands"],
                basis=f"DsaMqaLogits_v3{{N=1}} at 2 instances ({LOGITS1['run']}: {LOGITS1['ands']} ANDs, {LOGITS1['circuit']} bytes, "
                      f"{LOGITS1['gb']} GB, {LOGITS1['s']} s) taken linear in N x instances", ands=round(a))
    elif fam == "SiluMul_v2":
        k = int(re.search(r"I=(\d+)", b)[1]) / 256 / SILU256["n"]
        e = est(stage_s=SILU256["s"] * k, stage_gb=max(1.2, SILU256["gb"] * k), circuit_bytes=SILU256["circuit"] * k,
                basis=f"SiluMul_v2{{I=256}} at 2 instances ({SILU256['run']}) taken linear in I x instances")
    elif fam == "GreedyChunk_v1":
        e = est(stage_s=GREEDY0["s"], stage_gb=GREEDY0["gb"], circuit_bytes=GREEDY0["circuit"],
                basis=f"GreedyChunk_v1{{N=16384,FIRST=false}} ({GREEDY0['run']}, accepted), taken equal")
    elif fam in ("Fp8GroupQuantUe8m0_v1", "FlashInferFp8Quant1x128_v1"):
        sib = nd["sibling"]
        k = int(re.search(r"K=(\d+)", b)[1]) / int(re.search(r"K=(\d+)", sib["block"])[1])
        sgb = sib["stage_gb"] if isinstance(sib.get("stage_gb"), (int, float)) else None
        e = est(stage_s=sib["seconds"] / TOTAL * k, stage_gb=sgb * k if sgb else None, circuit_bytes=sib["circuit_bytes"] * k,
                total_s=sib["seconds"] * k, basis=f"{sib['block']} ({sib['tag']}: {sib['circuit_bytes']} bytes, {sib.get('stage_gb')} GB, "
                f"{sib['seconds']} s) taken linear in K")
    elif fam == "RMSNormCudaVector8_v1":
        e = est(stage_s=SUMSQ["s"] * 2, stage_gb=SUMSQ["gb"] * 2, circuit_bytes=SUMSQ["circuit"] * 2,
                basis="RmsNormMnnvlSumSq_v1{N=6144} (t101, accepted: 3.87e9 bytes, 11.6 GB, 270.5 s) taken as half of it; no measurement of "
                      "this family")
    elif fam == "AllReduceMnnvl_v1":
        e = est(total_s=190, basis="no measurement of this family; the longest of CIR-272's 24 accepted types of card k_log 22 (190 s)")
    elif fam == "SparseMlaSplit_v1":
        e = est(total_s=1580, basis="no measurement of this family; t019's whole time (1,580 s), the one type of card k_log 27 CIR-272 staged")
    else:
        e = {"basis": "no measurement of this family", "kind": "none"}
    run = (e.get("stage_gb") or 0) <= STAGE_GB and (e.get("lean_gb") or 0) <= LEAN_GB and r.get("why") not in ("stage-past-62GB", "past-2700s")
    if t in ("t019", "t031") or fam == "FlashInferFp8Quant1x128_v1" and e.get("lean_gb", 0) > LEAN_GB:
        run = run and e.get("lean_gb", 0) <= LEAN_GB
    return {"tag": t, "block": b, "blocks": r["blocks"], "k_log": r["k_log"], "extra": r["extra"], "v3": [r["outcome"], r.get("why")],
            "estimate": e, "run": bool(run)}


P = {t: plan(t) for t in todo}
UNKNOWN_ORDER = ["AllReduceMnnvl_v1", "SparseMlaSplit_v1"]
EXTRA_ORDER = ["StopCheck_v1", "RMSNormFusedApplyCudaVector8_v1", "AllGather_v1", "DsaLatentRow_v1", "RMSNormFusedSumSqCudaVector8_v1",
               "RMSNormFusedCudaVector8_v3", "TokenSelect_v4"]


def key(p):
    fam = p["block"].split("{")[0]
    nums = [int(x) for x in re.findall(r"[=,{](\d+)(?=[,}])", p["block"])[:3]]
    if p["extra"]:
        return (2, EXTRA_ORDER.index(fam), nums)
    if "total_s" in p["estimate"]:
        return (0, p["estimate"]["total_s"], nums)
    return (1, UNKNOWN_ORDER.index(fam), nums)


runs = sorted((p for p in P.values() if p["run"]), key=key)
lists = {"a": [], "b": []}
for i, p in enumerate(runs):
    tot = p["estimate"].get("total_s")
    p["limit_s"] = min(3600, max(900, 2 * tot)) if tot else (2700 if not p["extra"] else 1800)
    p["order"], p["list"] = i + 1, "ab"[i % 2]
    lists[p["list"]].append(p)
for w, ps in lists.items():
    (HERE / f"types-p2{w}.tsv").write_text("".join(f"{p['tag']}\tcensus:{p['block']}\t1\tsession\t{p['limit_s']}\n" for p in ps))
(HERE / "p2-plan.json").write_text(json.dumps(P, indent=1) + "\n")
for w, ps in lists.items():
    print(w, len(ps), "types; estimated", sum(p["estimate"].get("total_s", 0) for p in ps), "s over", sum("total_s" in p["estimate"] for p in ps),
          "with an estimate")
print("not run:", len(P) - len(runs), [(p["tag"], p["block"][:40]) for p in P.values() if not p["run"]][:80])
