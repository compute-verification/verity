"""A scored run's SCORE at each certified fraction a filler's floor could be read at (HC-PROVED, cmp-13): the locked
scorer's own arithmetic (`scoring.score`) on the run's `dscore.jsonl`, with only `certified` changed. Serving stays at the
scorer's proved 0.225; the filler is read at

- proved: Theorem E's c/m = ¼ a product (c = ½, m_max = 2 with OMMA on sm_120) at φ = 0.9, 0.225: pfill's operands are
  fresh i.i.d. noise on {±1} for every MMA (A_M = B_M = 0) with every MMA's prefix sum committed
  (note:20261015T1845Z-draft-rs-d1e-edited-floor);
- 0.215: Theorem M over a week at γ_zk = 0.1; 0.2496: Theorem S (φψ = 0.9984, not yet refereed);
- 0.99: the spec's provisional placeholder (the scorer's default, what `score.json` reports);
- 0: a chained filler's proved floor (chained operands are heads, note:20261016T0418Z-draft-filler-floor).

The floor counts int8 slots, which is what the filler's rate reports. An e2m1 arm (pfill_arm.products_per_slot, read from
the run's arm.json) does two products a slot, so the three theorem readings are twice the per-product fraction for it:
0.45, 0.43 and 0.4992.

  python3 benchmarks/pouw/filler/pfill_proved.py OUT_DIR [--json]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scoring as S
from pfill_arm import products_per_slot

#: name: (certified fraction a product, status, source, scales with products a slot)
READINGS = {
    "proved": (0.225, "proved", "Theorem E c/m = 1/4 at phi = 0.9", True),
    "theorem-m-week": (0.215, "proved", "Theorem M over a week at gamma_zk = 0.1", True),
    "theorem-s": (0.2496, "conjectural", "Theorem S, phi psi = 0.9984 (not refereed)", True),
    "provisional": (0.99, "provisional", "the spec's placeholder", False),
    "chained-proved": (0.0, "proved", "a chained filler's floor: its operands are heads", False),
}


def certified(fraction: float, status: str, source: str) -> dict:
    return {"serving": S.CERTIFIED["serving"], "filler": {"fraction": fraction, "status": status, "source": source}}


def stats_of(out: Path) -> dict:
    rows = [json.loads(ln) for ln in (out / "dscore.jsonl").read_text().splitlines() if ln.strip().startswith("{")]
    return {("decode" if r["kind"] == "step" else "prefill", r["B"], r["L"]): S.point_stats(r) for r in rows}


def slot_products(out: Path) -> int:
    doc = out / "arm.json"
    return products_per_slot(json.loads(doc.read_text()).get("lineage") or {}) if doc.is_file() else 1


def readings(out: Path) -> dict:
    st, m = stats_of(out), slot_products(out)
    res = {}
    for name, (f, status, src, scales) in READINGS.items():
        f = f * m if scales else f
        sc = S.score(st, certified(f, status, src))
        res[name] = {"filler_fraction": f, "SCORE": sc.get("SCORE"), "min": sc.get("min"), "min_mix": sc.get("min_mix"),
                     "missing": sc["missing"]}
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
    print("| filler read at | c_filler | SCORE | min entry |")
    print("|---|---|---|---|")
    for name, r in res.items():
        m = r["min"] or {}
        s = "–" if r["SCORE"] is None else f"{r['SCORE']:.4f}"
        print(f"| {name} | {r['filler_fraction']:.4g} | {s} | {m.get('entry')}: {m.get('coverage', float('nan')):.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
