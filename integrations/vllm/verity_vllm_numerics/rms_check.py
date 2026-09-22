"""Check the Triton rms_norm CPU model against captured production outputs (CPU only, W11).

    python -m verity_vllm_numerics.rms_check <fixture_dir> --tables <tables_dir> [--variant NAME] [--json out]

Reads ``index.json`` written by ``rms_capture.py`` / ``rms_stress.py`` (``calls[]`` with ``x_file``,
``out_file``, ``weight``, ``eps``, ``n_cols``), runs :func:`rms_triton_model.rms_norm_rows` on every call and
compares the BF16 words.  Prints a JSON summary: per phase/family exact rows / outputs / mismatches, the first
mismatches (row, column, hw word, model word, fp32 scalars) and the wall time per output row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verity_vllm_numerics import rms_triton_model as rm  # noqa: E402


def load_words(p: Path) -> np.ndarray:
    a = np.load(p)
    return a.view(np.uint16) if a.dtype != np.uint16 else a


def run(fixture: Path, tables: rm.RmsTables, variant: int = 0, nfirst: int = 20, families: bool = False, kernel: str = "triton") -> dict:
    idx = json.loads((fixture / "index.json").read_text())
    weights = {k: load_words(fixture / v["file"]) for k, v in idx["weights"].items()}
    groups: dict[str, dict] = {}
    first: list[dict] = []
    t_model = 0.0
    n_rows = 0
    for c in idx["calls"]:
        x = load_words(fixture / c["x_file"]).reshape(c["rows"], c["n_cols"])
        hw = load_words(fixture / c["out_file"]).reshape(c["rows"], c["n_cols"])
        w = weights[c["weight"]]
        t0 = time.perf_counter()
        md, sc = rm.rms_norm_rows(x, w, c["eps"], tables, variant=variant, want_scalars=True, kernel=kernel)
        t_model += time.perf_counter() - t0
        n_rows += c["rows"]
        neq = md != hw
        key = c.get("family", c.get("phase", "all")) if families else c.get("phase", "all")
        g = groups.setdefault(key, {"calls": 0, "rows": 0, "outputs": 0, "mismatch_outputs": 0, "mismatch_rows": 0, "exact_rows": 0})
        g["calls"] += 1
        g["rows"] += c["rows"]
        g["outputs"] += int(neq.size)
        g["mismatch_outputs"] += int(neq.sum())
        bad_rows = neq.any(axis=1)
        g["mismatch_rows"] += int(bad_rows.sum())
        g["exact_rows"] += int((~bad_rows).sum())
        if neq.any() and len(first) < nfirst:
            for r, col in zip(*np.nonzero(neq)):
                if len(first) >= nfirst:
                    break
                first.append({
                    "seq": c["seq"], "family": c.get("family"), "case": c.get("case"), "row": int(r), "col": int(col),
                    "x": f"{int(x[r, col]):04x}", "w": f"{int(w[col]):04x}", "hw": f"{int(hw[r, col]):04x}", "model": f"{int(md[r, col]):04x}",
                    "scalars": [float(v) for v in sc[r]],
                    "scalars_hex": [f"{v:08x}" for v in sc[r].view(np.uint32)],
                })
    total = {k: sum(g[k] for g in groups.values()) for k in ("calls", "rows", "outputs", "mismatch_outputs", "mismatch_rows", "exact_rows")}
    return {
        "fixture": str(fixture), "kernel": kernel, "variant": variant,
        "variant_name": next((k for k, v in (rm.VARIANTS if kernel == "triton" else rm.CUDA_VARIANTS).items() if v == variant), None),
        "groups": groups, "total": total, "first_mismatches": first,
        "model_seconds": t_model, "model_us_per_row": 1e6 * t_model / max(n_rows, 1),
        "tables": tables.source,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--variant", default="model")
    ap.add_argument("--families", action="store_true", help="group by stress family instead of phase")
    ap.add_argument("--json")
    ap.add_argument("--kernel", default="triton", choices=["triton", "cuda"])
    args = ap.parse_args()
    tables = rm.RmsTables.load(args.tables)
    variants = rm.VARIANTS if args.kernel == "triton" else rm.CUDA_VARIANTS
    rep = run(Path(args.fixture), tables, variants[args.variant], families=args.families, kernel=args.kernel)
    s = json.dumps(rep, indent=1)
    if args.json:
        Path(args.json).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
