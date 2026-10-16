"""A scored run's SCORE at each certified fraction a filler's floor could be read at (HC-PROVED, cmp-13): the locked
scorer's own arithmetic (`scoring.score`) on the run's `dscore.jsonl`, with only `certified` changed. Serving stays at the
scorer's proved 0.225; the filler is read at

- proved: Theorem E's c/m = ¼ (c = ½, m_max = 2 with OMMA on sm_120) at φ = 0.9, 0.225 an int8 slot: pfill's operands
  are fresh i.i.d. noise on {±1} for every MMA (A_M = B_M = 0) with every MMA's prefix sum committed
  (note:20261015T1845Z-draft-rs-d1e-edited-floor). Its priced form needs every head and merge priced at ≥ cL/m_max
  int8 slots; at L = 32 that is 8, a CUDA word op's real price. Each head costs the floor L products, so at that price
  the floor an int8 slot is φ·min(c·m/m_max, 8 / (slots a committed word)): every sm_120 MMA writes 128 words in 4096
  slots, so it is 0.225 for the s8 tile (L = 32) and for the e2m1 tile (L = 64, m = 2) alike;
- 0.215: Theorem M over a week at γ_zk = 0.1; 0.2496: Theorem S (φψ = 0.9984, not yet refereed);
- 0.99: the spec's provisional placeholder (the scorer's default, what `score.json` reports);
- 0: a chained filler's proved floor (chained operands are heads, note:20261016T0418Z-draft-filler-floor);
- 0.9, in-model: c/m 1.000 for {±1} on both sides under VF2 and the frozen VF3c (note:20261015T1814Z-draft-rs-machine-model
  §8.4, §13, §15) at φ = 0.9, with no margin (Four Russians ties the honest cost). Theorem E's tensor-written route
  (TW, §5 there) reads the same: every sm_120 MMA writes one committed word a 32 slots, as pfill commits them, and TW
  charges a CUDA-written word 32 slots (its real price is ≈ 8). Both are model statements, so this is a reading in
  the model, beside the chained filler's 0.99 under CH;
- priced-l64, e2m1 arms only (pfill_arm.products_per_slot, read from the run's arm.json): 0.45, Theorem E's priced form
  at L = 64 with heads priced at cL/m_max = 16 int8 slots, twice a CUDA word op's ≈ 8. That price is a claim, not a
  measurement, so the reading is conditional;
- theorem-k, pfill-bU arms only (pfill_arm.theorem_k_setting): 0.9, Theorem K's c = 1 at φ = 0.9 for straight-line,
  cell-pure runs on int MMAs with unscaled s32 reads, operands affine in their bits and every prefix word committed, as
  RS-E-REF r4 refereed it (note:20261015T1931Z-draft-theorem-e-referee Part IV, with G₂ over all bit pairs and the luck
  lemma on the integer alphabet, ≈ ½ a missing product). The guarantee isn't in the Lean lock yet, so the reading is
  projected.

  python3 benchmarks/pouw/filler/pfill_proved.py OUT_DIR [--json]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scoring as S
from pfill_arm import products_per_slot, theorem_k_setting

READINGS = {
    "proved": (0.225, "proved", "Theorem E c/m = 1/4 at phi = 0.9, heads at 8 int8 slots"),
    "theorem-m-week": (0.215, "proved", "Theorem M over a week at gamma_zk = 0.1"),
    "theorem-s": (0.2496, "conjectural", "Theorem S, phi psi = 0.9984 (not refereed)"),
    "provisional": (0.99, "provisional", "the spec's placeholder"),
    "chained-proved": (0.0, "proved", "a chained filler's floor: its operands are heads"),
    "in-model": (0.9, "in-model", "c/m 1.000 for {+-1} under VF2/VF3c at phi = 0.9, no margin; TW's output count"),
}
E2M1_READINGS = {
    "priced-l64": (0.45, "conditional", "Theorem E's priced form at L = 64: heads priced at cL/m_max = 16 int8 slots"),
}
K_READINGS = {
    "theorem-k": (0.9, "projected", "Theorem K c = 1 at phi = 0.9 (RS-E-REF r4: straight-line, cell-pure, int MMAs, "
                                    "unscaled s32 reads, affine operands, every prefix word committed)"),
}


def certified(fraction: float, status: str, source: str) -> dict:
    return {"serving": S.CERTIFIED["serving"], "filler": {"fraction": fraction, "status": status, "source": source}}


def stats_of(out: Path) -> dict:
    rows = [json.loads(ln) for ln in (out / "dscore.jsonl").read_text().splitlines() if ln.strip().startswith("{")]
    return {("decode" if r["kind"] == "step" else "prefill", r["B"], r["L"]): S.point_stats(r) for r in rows}


def lineage_of(out: Path) -> dict:
    doc = out / "arm.json"
    return (json.loads(doc.read_text()).get("lineage") or {}) if doc.is_file() else {}


def readings(out: Path) -> dict:
    st = stats_of(out)
    lin = lineage_of(out)
    table = {**READINGS, **(E2M1_READINGS if products_per_slot(lin) == 2 else {}),
             **(K_READINGS if theorem_k_setting(lin) else {})}
    res = {}
    for name, (f, status, src) in table.items():
        sc = S.score(st, certified(f, status, src))
        res[name] = {"filler_fraction": f, "status": status, "SCORE": sc.get("SCORE"), "min": sc.get("min"),
                     "min_mix": sc.get("min_mix"), "missing": sc["missing"]}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = readings(a.out)
    if a.json:
        print(json.dumps(res, indent=1))
        return 0
    print("| filler read at | c_filler | status | SCORE | min entry |")
    print("|---|---|---|---|---|")
    for name, r in res.items():
        m = r["min"] or {}
        s = "–" if r["SCORE"] is None else f"{r['SCORE']:.4f}"
        print(f"| {name} | {r['filler_fraction']:.4g} | {r['status']} | {s} | "
              f"{m.get('entry')}: {m.get('coverage', float('nan')):.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
