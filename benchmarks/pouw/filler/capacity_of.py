"""The certified share σ of a GPU's capacity, from a scored filler run's measured entries (FILLER, decision 77).

Each entry of a run's score.json is one served step beside its filler: the step's time, the served and filler MACs in
it, and the SM clock. Their certified load, pooled over the step and the GPU's 4 × SMs SMSPs (a receipt attests no
placement), is read by machine_model's `capacity` (note:20261017T0652Z-draft-filler §3): filler MMAs with their fold's
lane instructions, lane-chain steps if the filler ran any, and served MMAs, at f_max (decision 74) and at the run clock.

With --lfill it reads dcap's lfill lines instead (capfill_fold_probe.sh): each one's window beside the pump alone and
beside the pump and the lane filler (lines from a -DPF_LANE build, retired after r20261017-072934-dd50, carry its lane
instructions a pump MMA as pf_lane). With --dfill, dcap's dfill lines (capfill_co_probe.sh): the step beside the pump on a
split, or co-scheduled on every SM. With --fstep, dcap's fstep lines (capfill_fstep_probe.sh): the step through flin
beside the pump, co-scheduled on every SM.

  python3 benchmarks/pouw/filler/capacity_of.py RUN_OUT_DIR [--fold duo] [--lane-per-mma X] [--served-fold none] [--json]
  python3 benchmarks/pouw/filler/capacity_of.py --lfill LFILL_JSONL [--fold duo] [--json]
  python3 benchmarks/pouw/filler/capacity_of.py --dfill DFILL_JSONL [--fold duo] [--json]
  python3 benchmarks/pouw/filler/capacity_of.py --fstep FSTEP_JSONL [--fold duo] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "machine_model"))
from verity_machine_model import capacity as C  # noqa: E402
from verity_machine_model import eps as E  # noqa: E402

F_MAX_MHZ = 2430.0  # sm_120's maximum SM clock, the credit's (decision 74)
SMS = 188
MACS_A_MMA = 4096


def load(serving_macs: float, filler_macs: float, ms: float, mhz: float, sms: int = SMS, fold: str = "duo",
         lane_per_mma: float = 0.0, served_fold: str = "none") -> C.Work:
    """The pooled certified load a SMSP-cycle at `mhz` over `ms` on `sms` SMs."""
    cyc = 4 * sms * mhz * 1e6 * ms * 1e-3
    fill = filler_macs / MACS_A_MMA / cyc * C.TENSOR_CYCLES
    srv = serving_macs / MACS_A_MMA / cyc * C.TENSOR_CYCLES
    w = C.tensor_filler(fill, fold) + C.served(srv, served_fold)
    if lane_per_mma:
        w = w + C.lane_filler(lane_per_mma * fill / C.TENSOR_CYCLES)
    return w


def sigma(pi, entry: dict, mhz: float | None = None, **kw) -> C.WindowCell:
    w = load(entry["serving_macs"], entry["filler_macs"], entry["t1_ms"], mhz or entry["mhz"], **kw)
    return C.window(pi, [(1.0, w)])


def lfill_load(rec: dict, arm: str, mhz: float, fold: str = "duo", served_fold: str = "none") -> C.Work:
    """One lfill arm's pooled certified load a SMSP-cycle at `mhz`: the served step at its measured time, the pump's
    MMAs with their fold (and pf_lane lane instructions, if the line has them), and the lane filler's instructions
    (arm "lane" only)."""
    a = rec[arm]
    pump = a["mac_per_s"] if arm == "pump" else a["pump_mac_per_s"]
    lane = a.get("lane_instr_per_s", 0.0) if arm == "lane" else 0.0
    cyc = 4 * (rec["step_sms"] + rec["fill_sms"]) * mhz * 1e6
    mma = pump / MACS_A_MMA / cyc
    w = C.tensor_filler(mma * C.TENSOR_CYCLES, fold)
    w = w + C.served(rec["step_macs"] / (a["step_ms"] * 1e-3) / MACS_A_MMA / cyc * C.TENSOR_CYCLES, served_fold)
    lanes = rec.get("pf_lane", 0) * mma + lane / cyc
    if lanes:
        w = w + C.lane_filler(lanes)
    return w


def lfill_rows(pi, recs: list[dict], fold: str = "duo", served_fold: str = "none") -> list[dict]:
    rows = []
    for r in recs:
        for arm in ("pump", "lane"):
            if arm not in r:
                continue
            a = r[arm]
            run = C.window(pi, [(1.0, lfill_load(r, arm, a["mhz"], fold, served_fold))])
            top = C.window(pi, [(1.0, lfill_load(r, arm, F_MAX_MHZ, fold, served_fold))])
            rows.append({"B": r["B"], "L": r["L"], "fill_sms": r["fill_sms"], "pf_lane": r.get("pf_lane", 0), "blk": r["blk"],
                         "T": r["T"], "CH": r["CH"], "arm": arm, "slowdown": a["slowdown"], "mhz": a["mhz"],
                         "watt": a.get("watt"), "sigma": top.sigma(), "sigma_run_clock": run.sigma(),
                         "tensor": top.tensor(), "binding": list(top.pooled.binding),
                         "rec_bad": r["rec_bad_pump"] + r["rec_bad_lane"]})
    return rows


def dfill_rows(pi, recs: list[dict], fold: str = "duo", served_fold: str = "none") -> list[dict]:
    """dcap dfill lines: the step beside the pump on its split's SMs (split_fill) or on every SM at the lowest priority
    beside the step at the highest (co_fill), pooled over the GPU."""
    rows = []
    for r in recs:
        co = r["place"] == "co"
        a = r["arms"]["co_fill" if co else "split_fill"]
        sms = r["step_sms"] if co else r["step_sms"] + r["fill_sms"]

        def at(mhz):
            cyc = 4 * sms * mhz * 1e6
            w = C.tensor_filler(r["window"]["fill_mac_per_s"] / MACS_A_MMA / cyc * C.TENSOR_CYCLES, fold)
            w = w + C.served(r["step_macs"] / (a["ms"] * 1e-3) / MACS_A_MMA / cyc * C.TENSOR_CYCLES, served_fold)
            return C.window(pi, [(1.0, w)])

        top = at(F_MAX_MHZ)
        rows.append({"B": r["B"], "L": r["L"], "place": r["place"], "build": r.get("build"), "NC": r["NC"], "S": r["S"],
                     "pad_kb": r.get("pad_kb"), "fill_sms": r["fill_sms"], "slowdown": r["slowdown"], "mhz": a["mhz"],
                     "sigma": top.sigma(), "sigma_run_clock": at(a["mhz"]).sigma(), "tensor": top.tensor(),
                     "binding": list(top.pooled.binding), "rec_bad": r["rec_bad"]})
    return rows


def fstep_rows(pi, recs: list[dict], fold: str = "duo", served_fold: str = "none") -> list[dict]:
    """dcap fstep lines: the step through flin and dcap's attention beside the pump, on a split or co-scheduled on every
    SM (place co), its filler warps' MMAs (if any) with the pump's. Shares in a line are of the whole GPU's tensor peak at
    its run clock, so at `mhz` they scale by run clock / mhz. A line's own "fold" (the probe's tag for its build) wins over
    `fold`."""
    rows = []
    for r in recs:
        rf = r.get("fold", fold)

        def at(mhz):
            k = r["mhz"] / mhz
            fill = (r["pump"]["share"] + r["fill_share_lin"] + r["fill_share_att"]) * k
            srv = r["step_share_fused"] * k
            return C.window(pi, [(1.0, C.tensor_filler(fill, rf) + C.served(srv, served_fold))])

        top = at(F_MAX_MHZ)
        rows.append({"B": r["B"], "L": r["L"], "place": r.get("place", "split"), "build": r.get("build"), "M": r["M"],
                     "kind": r.get("kind", 2), "pdl": r.get("pdl", 1), "pump_nc": r["pump"].get("NC"), "fold": rf,
                     "pad_kb": r.get("pad_kb"), "fill_sms": r["fill_sms"], "slowdown": r["slowdown"],
                     "co_cost": r["fused_ms"] / r["fused0_ms"], "fused0_vs_base": r["fused0_vs_base"],
                     "pump_share": r["pump"]["share"], "pump_alone": r["pump"]["alone_share"], "mhz": r["mhz"],
                     "sigma": top.sigma(), "sigma_run_clock": at(r["mhz"]).sigma(), "tensor": top.tensor(),
                     "binding": list(top.pooled.binding),
                     "rec_bad": r["pump"]["rec_bad"] + r["lin"]["rec_bad"] + r["att"]["rec_bad"],
                     "out_bad": r["out_mismatches"] + r["lm_mismatches"]})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="capacity_of.py", description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path, nargs="?", help="a scored run's arm dir (its score.json)")
    ap.add_argument("--lfill", type=Path, help="dcap lfill lines (capfill_probe.sh's lfill.jsonl)")
    ap.add_argument("--dfill", type=Path, help="dcap dfill lines, split or co (capfill_co_probe.sh's co.jsonl)")
    ap.add_argument("--fstep", type=Path, help="dcap fstep lines (capfill_fstep_probe.sh's fstep.jsonl)")
    ap.add_argument("--fold", default="duo")
    ap.add_argument("--served-fold", default="none")
    ap.add_argument("--lane-per-mma", type=float, default=0.0, help="lane-chain warp instructions a filler MMA")
    ap.add_argument("--vector", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    pi = E.machine(a.vector)
    if a.fstep:
        recs = [json.loads(s) for s in a.fstep.read_text().splitlines() if s.startswith("{")]
        rows = fstep_rows(pi, recs, a.fold, a.served_fold)
        if a.json:
            print(json.dumps({"vector": pi.vector, "f_max_mhz": F_MAX_MHZ, "fold": a.fold, "rows": rows}, indent=1))
            return 0
        for r in rows:
            part = ("linears", "attention", "step")[r["kind"]]
            print(f"  B {r['B']:3d} L {r['L']:5d} {r['place']:5s} {r['fill_sms']:3d} {r['build'] or '':4s} M {r['M']:3d} "
                  f"{part:9s} pdl {r['pdl']} {r['fold']:5s} NC {r['pump_nc']}  slowdown "
                  f"{r['slowdown']:.3f} (flin alone {r['fused0_vs_base']:.3f}, pump's cost {r['co_cost']:.3f})  pump "
                  f"{r['pump_share']:.3f} (alone {r['pump_alone']:.3f})  σ {r['sigma']:.3f} at f_max "
                  f"({r['sigma_run_clock']:.3f} at {r['mhz']:.0f} MHz)  bind {','.join(r['binding'])}  bad {r['rec_bad']} "
                  f"out {r['out_bad']}")
        return 0
    if a.dfill:
        recs = [json.loads(s) for s in a.dfill.read_text().splitlines() if s.startswith("{")]
        rows = dfill_rows(pi, recs, a.fold, a.served_fold)
        if a.json:
            print(json.dumps({"vector": pi.vector, "f_max_mhz": F_MAX_MHZ, "fold": a.fold, "rows": rows}, indent=1))
            return 0
        for r in rows:
            print(f"  B {r['B']:3d} L {r['L']:5d} {r['place']:5s} {r['build'] or '':8s} NC {r['NC']} S {r['S']:4d} pad "
                  f"{r['pad_kb'] or 0:2d}  slowdown {r['slowdown']:.3f}  σ {r['sigma']:.3f} at f_max "
                  f"({r['sigma_run_clock']:.3f} at {r['mhz']:.0f} MHz)  tensor {r['tensor']:.3f}  "
                  f"bind {','.join(r['binding'])}  bad {r['rec_bad']}")
        return 0
    if a.lfill:
        recs = [json.loads(s) for s in a.lfill.read_text().splitlines() if s.startswith("{")]
        rows = lfill_rows(pi, recs, a.fold, a.served_fold)
        if a.json:
            print(json.dumps({"vector": pi.vector, "f_max_mhz": F_MAX_MHZ, "fold": a.fold, "rows": rows}, indent=1))
            return 0
        for r in rows:
            print(f"  B {r['B']:3d} L {r['L']:5d} pump {r['fill_sms']:3d} PF_LANE {r['pf_lane']:2d} blk {r['blk']} "
                  f"{r['arm']:4s}  slowdown {r['slowdown']:.3f}  σ {r['sigma']:.3f} at f_max ({r['sigma_run_clock']:.3f} at "
                  f"{r['mhz']:.0f} MHz)  tensor {r['tensor']:.3f}  bind {','.join(r['binding'])}  bad {r['rec_bad']}")
        return 0
    if a.out is None:
        ap.error("a run's arm dir, or --lfill")
    d = json.loads((a.out / "score.json").read_text())
    kw = dict(fold=a.fold, lane_per_mma=a.lane_per_mma, served_fold=a.served_fold)
    rows = []
    for e in d["entries"]:
        run, top = sigma(pi, e, **kw), sigma(pi, e, F_MAX_MHZ, **kw)
        rows.append({"entry": e["entry"], "kind": e["kind"], "slowdown": e["time"], "mhz": e["mhz"],
                     "filler_share": e["filler_share"], "sigma_run_clock": run.sigma(), "sigma": top.sigma(),
                     "credit": top.credit(), "tensor": top.tensor(), "binding": list(top.pooled.binding)})
    dec = [r for r in rows if r["kind"] == "decode"]
    summ = {"arm": d.get("arm"), "run": d.get("run"), "vector": pi.vector, "f_max_mhz": F_MAX_MHZ, **kw,
            "decode_mean_sigma": sum(r["sigma"] for r in dec) / len(dec),
            "decode_mean_sigma_run_clock": sum(r["sigma_run_clock"] for r in dec) / len(dec),
            "decode_min_sigma": min(r["sigma"] for r in dec), "entries": rows}
    if a.json:
        print(json.dumps(summ, indent=1))
        return 0
    print(f"{summ['arm']} ({summ['run']}), Π {pi.vector}, fold {a.fold}, lanes {a.lane_per_mma:g} a MMA")
    for r in rows:
        print(f"  {r['entry']:22s} slowdown {r['slowdown']:.3f}  filler {r['filler_share']:.3f}  σ {r['sigma']:.3f} at "
              f"f_max ({r['sigma_run_clock']:.3f} at {r['mhz']:.0f} MHz)  B {r['credit']:.3f}  bind {','.join(r['binding'])}")
    print(f"  decode mean σ {summ['decode_mean_sigma']:.3f} at f_max, {summ['decode_mean_sigma_run_clock']:.3f} at the run "
          f"clock; min {summ['decode_min_sigma']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
