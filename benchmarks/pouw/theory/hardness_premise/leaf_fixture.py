"""The timing's operands: one Strassen W-form of a natural down projection, split lo + 256·hi for an s8 leaf.

`python3 leaf_fixture.py OUT [--model llama3.2-1b] [--layer 8] [--form 0]` writes, for `leaf_corr.cu`:

- `lo.bin`: the form's low digit, n × K int8, row j = output column j (the leaf's B operand, K contiguous);
- `colptr.bin`: n + 1 int32, column j's overflow entries are `ent[colptr[j]:colptr[j + 1]]`;
- `ent.bin`: int32 entries `k << 8 | (hi & 255)`, hi the overflow digit (≠ 0), sorted by (j, k);
- `fixture.json`: the shapes, the form, nnz and its row and column counts (natural.py's `hi_stats`), and every overflowing
  form's statistics, so the note can say how typical the timed one is.

The weights are natural.py's (HF's safetensors, per-row int8 to 127); the form is level 1 of `execute.SCHEMES`'s
Strassen V over Wᵀ's quadrants, the leaf `natural.py` prices.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execute import SCHEMES
from natural import HF, MODELS, fetch, hi_stats, load_tensor, quantize, safetensors_header


def down_proj(model: str, layer: int) -> np.ndarray:
    repo = MODELS[model]
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    try:
        f = json.loads(fetch(f"{HF}/{repo}/resolve/main/model.safetensors.index.json"))["weight_map"][name]
    except urllib.error.HTTPError as e:  # a single-file checkpoint has no index
        if e.code != 404:
            raise
        f = "model.safetensors"
    url = f"{HF}/{repo}/resolve/main/{f}"
    base, h = safetensors_header(url)
    return quantize(load_tensor(url, base, h[name]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="llama3.2-1b")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--form", type=int, default=0, help="Strassen V's product (0: B11 + B22)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    q = down_proj(args.model, args.layer).astype(np.int32)  # out × in
    C = q.T  # K × n, natural.py's C
    h, w = C.shape[0] // 2, C.shape[1] // 2
    Bb = [C[:h, :w], C[:h, w:], C[h:, :w], C[h:, w:]]
    _U, V, _W = SCHEMES["strassen"]
    stats = {}
    for p, v in enumerate(V):
        F = sum(c * Bb[i] for i, c in enumerate(v) if c)
        hi = (F + 128) >> 8
        stats[p] = {"coeffs": v, **(hi_stats(hi) if hi.any() else {"nnz": 0})}
    v = V[args.form]
    F = sum(c * Bb[i] for i, c in enumerate(v) if c)  # K' × n'
    hi = (F + 128) >> 8
    lo = F - 256 * hi
    assert lo.min() >= -128 and lo.max() <= 127 and np.abs(hi).max() <= 127
    Ft = np.ascontiguousarray(lo.T.astype(np.int8))  # n' × K'
    hiT = hi.T  # n' × K'
    js, ks = np.nonzero(hiT)  # row-major: sorted by (j, k)
    vals = hiT[js, ks]
    colptr = np.zeros(hiT.shape[0] + 1, np.int32)
    np.add.at(colptr, js + 1, 1)
    colptr = np.cumsum(colptr).astype(np.int32)
    ent = ((ks.astype(np.int64) << 8) | (vals.astype(np.int64) & 255)).astype(np.int32)
    Ft.tofile(args.out / "lo.bin")
    colptr.tofile(args.out / "colptr.bin")
    ent.tofile(args.out / "ent.bin")
    per_col = np.diff(colptr)
    abs_per_col = np.zeros(hiT.shape[0], np.int64)
    np.add.at(abs_per_col, js, np.abs(vals))
    meta = {"model": args.model, "layer": args.layer, "form": args.form, "coeffs": v, "K": int(Ft.shape[1]),
            "n": int(Ft.shape[0]), **hi_stats(hi), "density": float((hi != 0).mean()),
            "max_per_col": int(per_col.max()), "max_abs_per_col": int(abs_per_col.max()), "forms": stats}
    (args.out / "fixture.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in meta if k != "forms"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
