"""Read `mma_joint120.cu`'s rows and say whether (M) holds on sm_120 beside DMMA and QMMA.

(M) (`Accounting.Assumptions.TS.SlotStep`) charges each warp instruction max(MACs / (peak / 4), slots x step) SMSP-cycles,
summed over every class, and asks the sum to be at most the run's cycles. On sm_120 the step is 1/8: an IMMA.16832 s8 is
charged 4096 / 256 = 128 x 1/8 = 16, a DMMA.8x8x4 (peak 0, so its sheet term is 0) 64 x 1/8 = 8, and a QMMA.16832 E4M3
at least its 128 slots x 1/8 = 16. A rate here is warp instructions a SM-clock (4 SMSPs), so (M)'s sum over the run's
cycles is (16 r_IMMA + c_X r_X) / 4. A probe set missing from the rows (a run before QMMA's) is skipped.

  python3 mma_joint.py OUT_DIR    # reads OUT_DIR/rows.jsonl, writes OUT_DIR/mma_joint.json and prints it
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CHARGE = {"IMMA": 16, "DMMA": 8, "QMMA": 16}
SMSPS = 4
MACS = {"DMMA": 256, "QMMA": 4096}
FORMS = {
    "DMMA": ("s_dmma_w8", {"dmmaU_i": "IMMA fixed, DMMA until", "iU_dmma": "DMMA fixed, IMMA until"}),
    "QMMA": ("s_qmma_w8", {"qmmaU_i": "IMMA fixed, QMMA until", "iU_qmma": "QMMA fixed, IMMA until"}),
}


def rate(row: dict, op: str) -> float:
    return next(v for k, v in row["rates"].items() if k.split(":", 1)[1].startswith(op))


def m_sum(r_imma: float, form: str, r_x: float) -> float:
    return (CHARGE["IMMA"] * r_imma + CHARGE[form] * r_x) / SMSPS


def main(out: Path) -> dict:
    rows = {r["probe"]: r for r in map(json.loads, (out / "rows.jsonl").read_text().splitlines())}
    t_unit = rate(rows["i_w8"], "IMMA")
    res = {
        "alone": {
            "s_imma": {"rate": rate(rows["s_imma"], "IMMA"), "smsp_cycles": SMSPS / rate(rows["s_imma"], "IMMA"),
                       "m_sum": m_sum(rate(rows["s_imma"], "IMMA"), "IMMA", 0)},
            "i_w8": {"rate": t_unit, "smsp_cycles": SMSPS / t_unit, "m_sum": m_sum(t_unit, "IMMA", 0)},
        },
        "joints": {},
        "verdicts": {},
        "sm_mhz": {p: r["sm_mhz"] for p, r in rows.items()},
        "tags": {p: r["tags"] for p, r in rows.items() if r["tags"]},
    }
    for form, (unit, joints) in FORMS.items():
        if unit not in rows or not all(j in rows for j in joints):
            continue
        s_unit = rate(rows[unit], form)
        res["alone"][unit] = {"rate": s_unit, "smsp_cycles": SMSPS / s_unit, "macs_a_sm_clk": MACS[form] * s_unit}
        for name, what in joints.items():
            r = rows[name]
            ri, rx = rate(r, "IMMA"), rate(r, form)
            res["joints"][name] = {
                "what": what, "form": form, "imma_rate": ri, "x_rate": rx, "t": ri / t_unit, "s": rx / s_unit,
                "t_plus_s": ri / t_unit + rx / s_unit, "m_sum": m_sum(ri, form, rx),
                "m_sum_x_no_slots": m_sum(ri, form, 0), "stop_gap": r["stop_gap"], "spread": r["spread"]}
        worst = max(res["joints"][j]["m_sum"] for j in joints)
        res["verdicts"][form] = (
            f"(M) false on sm_120 beside {form}: its sum is {worst:.4f} of the run's cycles" if worst > 1
            else f"(M) holds beside {form} on these runs: its largest sum is {worst:.4f} of the run's cycles")
    (out / "mma_joint.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
