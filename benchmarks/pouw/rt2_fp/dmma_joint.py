"""RT2-FP-12: read `dmma_imma120.cu`'s rows and say whether #2755's (M) holds on sm_120 beside DMMA.

(M) (`Accounting.Assumptions.TS.SlotStep`) charges each warp instruction max(MACs / (peak / 4), slots x step) SMSP-cycles,
summed over every class, and asks the sum to be at most the run's cycles. On sm_120 the step is 1/8: an IMMA.16832 s8 is
charged 4096 / 256 = 128 x 1/8 = 16, a DMMA.8x8x4 (peak 0, so its sheet term is 0) 64 x 1/8 = 8. A rate here is warp
instructions a SM-clock (4 SMSPs), so (M)'s sum over the run's cycles is (16 r_IMMA + 8 r_DMMA) / 4.

  python3 dmma_joint.py OUT_DIR    # reads OUT_DIR/rows.jsonl, writes OUT_DIR/dmma_joint.json and prints it
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CHARGE = {"IMMA": 16, "DMMA": 8}
SMSPS = 4
JOINTS = {"dmmaU_i": "IMMA fixed, DMMA until", "iU_dmma": "DMMA fixed, IMMA until"}


def rate(row: dict, op: str) -> float:
    return next(v for k, v in row["rates"].items() if k.split(":", 1)[1].startswith(op))


def m_sum(r_imma: float, r_dmma: float) -> float:
    return (CHARGE["IMMA"] * r_imma + CHARGE["DMMA"] * r_dmma) / SMSPS


def main(out: Path) -> dict:
    rows = {r["probe"]: r for r in map(json.loads, (out / "rows.jsonl").read_text().splitlines())}
    t_unit, s_unit = rate(rows["i_w8"], "IMMA"), rate(rows["s_dmma_w8"], "DMMA")
    res = {
        "alone": {
            "s_imma": {"rate": rate(rows["s_imma"], "IMMA"), "smsp_cycles": SMSPS / rate(rows["s_imma"], "IMMA"),
                       "m_sum": m_sum(rate(rows["s_imma"], "IMMA"), 0)},
            "i_w8": {"rate": t_unit, "smsp_cycles": SMSPS / t_unit, "m_sum": m_sum(t_unit, 0)},
            "s_dmma_w8": {"rate": s_unit, "smsp_cycles": SMSPS / s_unit, "macs_a_sm_clk": 256 * s_unit},
        },
        "joints": {},
        "sm_mhz": {p: r["sm_mhz"] for p, r in rows.items()},
        "tags": {p: r["tags"] for p, r in rows.items() if r["tags"]},
    }
    for name, what in JOINTS.items():
        r = rows[name]
        ri, rd = rate(r, "IMMA"), rate(r, "DMMA")
        res["joints"][name] = {
            "what": what, "imma_rate": ri, "dmma_rate": rd, "t": ri / t_unit, "s": rd / s_unit,
            "t_plus_s": ri / t_unit + rd / s_unit, "m_sum": m_sum(ri, rd), "m_sum_dmma_no_slots": m_sum(ri, 0),
            "stop_gap": r["stop_gap"], "spread": r["spread"]}
    worst = max(j["m_sum"] for j in res["joints"].values())
    res["verdict"] = (f"(M) false on sm_120 beside DMMA: its sum is {worst:.4f} of the run's cycles" if worst > 1
                      else f"(M) holds on these runs: its largest sum is {worst:.4f} of the run's cycles")
    (out / "dmma_joint.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
