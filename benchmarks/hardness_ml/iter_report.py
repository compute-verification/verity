"""Design D's tables from an iter_sweep output directory: every arm's served loss against the clean int8 arm with the
same seed (gap), the untrained base served under the same draws against the base (one-shot), its alphabets' forced
work (iter_price), and the post-draw arms.

    python iter_report.py OUT_DIR [--json REPORT]
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import iter_price as P


def load(out: Path):
    runs = {}
    for f in sorted((out / "runs").glob("*.json")):
        runs[f.stem] = json.loads(f.read_text())
    return runs


def arm(name, r, ref, base, strassen=None):
    c = r["cfg"]
    ne, nr = c["n_e"], c["n_r"]
    draws = r.get("val_d0_draws") or []
    row = {"name": name, "regime": c["regime"], "n_e": ne, "n_r": nr, "seed": c["seed"],
           "val_d0": r["val_d0"], "gap": r["val_d0"] - ref,
           "sigma_draws": statistics.pstdev(draws) if len(draws) > 1 else None,
           "val_clean": r.get("val_clean"), "code_rms": r.get("code_rms"), "code_sat": r.get("code_sat_frac"),
           "coin_bits": math.log2(ne) if ne > 1 else 0.0}
    be = r.get("base_evals", {})
    if "untrained_val_d0" in be and base is not None:
        # the untrained base served under fresh draws, against the base itself: no fine-tune on either side
        row["oneshot_gap"] = be["untrained_val_d0"] - base
        row["oneshot_e"] = be.get("untrained_val_e", base) - base
        row["oneshot_r"] = be["untrained_val_r"] - base
    if c["regime"] == "designA":
        row.update({"r_norm_share": r.get("r_norm_share"), "d1_rms": r.get("d1_rms"), "rank_frac": c["rank_frac"]})
    if ne > 1 or nr > 1:
        pr = P.row(max(ne, 2), max(nr, 2), strassen) if ne > 1 and nr > 1 else None
        row["forced"] = pr["forced"] if pr else 0.0
        row["forced_mod"] = pr["forced_mod"] if pr else 0.0
        row["h200_raw"] = pr["h200_b1_raw"] if pr else 0.0
        row["sm120_raw"] = pr["sm120_raw"] if pr else 0.0
        if pr and "credit_attack" in pr:
            row["credit_attack"], row["credit_floor"] = pr["credit_attack"], pr["credit_floor"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--json", default=None)
    ap.add_argument("--strassen", default=None, help="iter_strassen.py's JSON")
    a = ap.parse_args()
    st = json.loads(Path(a.strassen).read_text()) if a.strassen else None
    runs = load(Path(a.out))
    base_json = Path(a.out) / "base.json"
    base = json.loads(base_json.read_text())["val_clean"] if base_json.exists() else None
    refs = {s: runs[n]["val_d0"] for n, s in (("clean", 0), ("clean-s1", 1)) if n in runs}
    rows = [arm(n, r, refs.get(r["cfg"]["seed"], refs.get(0)), base, st) for n, r in runs.items()]
    post = {}
    for n, r in runs.items():
        if "post" in r:
            p = r["post"]
            before = p["val_d0_draw0"]
            post[n] = {k: {**v, "gain": before - v["val_d0_draw0"]} for k, v in p.items() if isinstance(v, dict)}
            post[n]["before"] = before
    rep = {"base": base, "refs": refs, "arms": rows, "post": post}
    cols = ["name", "val_d0", "gap", "oneshot_gap", "oneshot_e", "oneshot_r", "sigma_draws", "forced", "forced_mod",
            "credit_attack", "credit_floor", "h200_raw", "sm120_raw", "coin_bits", "code_rms", "r_norm_share"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in sorted(rows, key=lambda x: x["name"]):
        print("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
    print(json.dumps({"refs": refs, "post": post}, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


if __name__ == "__main__":
    main()
