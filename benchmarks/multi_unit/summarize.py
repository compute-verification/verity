"""summarize.py OUT: job.sh's measurements per shape (OUT/k<K>-n<N>-i<n>/) as one table, written to OUT/summary.json and
printed. Per form (c, b-writer, b-reader, a): each slot type's nonzeros (the verifier's lincheck folds each type once), k_log,
the committed m, the prover's total seconds over both reps and peak memory, the verifier's seconds and peak, the proof bytes, and
the verdict (the one without the scope refusal where the verifier refuses the form); b's units padded to a power of two."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def session(p: Path) -> dict | None:
    r = load(p)
    if r is None:
        return None
    w = r.get("verdict_without_refusal") or {}
    return {"prove_total_s": r.get("prove_total_s"), "prover_peak_gb": r.get("prover_peak_gb"), "verify_s": r.get("verify_s"),
            "verifier_peak_gb": r.get("verifier_peak_gb"), "dense_m": r.get("dense_m"), "k_log": r.get("k_log"),
            "proof_bytes": sum(r.get("proof_bytes") or []) if isinstance(r.get("proof_bytes"), list) else r.get("proof_bytes"),
            "accepted": r.get("accepted"), "why": (r.get("why") or "")[:120], "without_refusal": w.get("accepted")}


def lean(p: Path) -> str | None:
    try:
        v = next(json.loads(ln[8:]) for ln in p.read_text().splitlines() if ln.startswith("VERDICT "))
    except (OSError, StopIteration, ValueError):
        return None
    return "ACCEPT" if v.get("accepted") else "REJECT " + (v.get("why") or "")[:160]


def shape(d: Path) -> dict:
    out = {}
    for f in ("c", "b-writer", "b-reader"):
        st = load(d / f"{f}-stage.json")
        if st is None:
            continue
        row = {"k_log": st.get("k_log"), "per_vu": st.get("per_vu"), "vus_per_block": st.get("vus_per_block"),
               "nnz": {k: v.get("nnz") for k, v in (st.get("counts") or {}).items()},
               "ands": {k: v.get("ands") for k, v in (st.get("counts") or {}).items()},
               "stage_s": (st.get("seconds") or {}).get("total"), "circuit_bytes": st.get("circuit_bytes"),
               "plain": session(d / f"{f}-prove.json"), "zk": session(d / f"{f}-prove-zk.json"), "lean_zk": lean(d / f"lean-{f}.out")}
        if f == "b-writer" and st.get("units") is not None:
            row["units"], row["units_padded"] = st.get("units"), st.get("units_padded")
        out[f] = row
    a, at = load(d / "a.json"), load(d / "a.time")
    if a is not None:
        out["a"] = a | {"peak_gb": round(at["maxrss_kb"] / 2**20, 2) if at else None}
    st = load(d / "c-selftest.json")
    if st is not None:
        out["c_selftest"] = {k: st.get(k) for k in ("all_pass", "cases", "failed", "wall_s")}
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[0])
    summary = {d.name: shape(d) for d in sorted(root.iterdir()) if d.is_dir() and d.name.startswith("k")}
    (root / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
