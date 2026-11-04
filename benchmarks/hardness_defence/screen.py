"""The split's screen on a GPU: each matrix's design-U codes (`quant.rotate_t`, then `quant.calibrate` to --target as
`quant.wrap` does) and the share of ‖q‖² in its top r singular directions, from the eigenvalues of the smaller Gram
matrix in float64. Within a shape the split's best price falls as that share rises (HT-PUBLIC v2: correlation −0.93 to
−0.99), so the screen picks which matrices of a large model to price with `hardness_public/rotated.py`.

    python screen.py --model DIR --match o_proj,gate_proj --layers 0,1 --out FILE.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hardness_useful"))
import quant

TOP = (1, 8, 16, 32, 64, 128, 256, 512)


def tensors(model: Path) -> dict[str, str]:
    idx = model / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    return {k: "model.safetensors" for k in json.loads((model / "config.json").read_text())}


def main(argv=None) -> int:
    from safetensors import safe_open

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--target", type=float, default=0.02)
    ap.add_argument("--match", default="o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--layers", default=None, help="comma-separated layer indices (default every layer)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    wmap = tensors(args.model)
    layers = None if args.layers is None else {int(x) for x in args.layers.split(",")}
    names = [t for t in sorted(wmap) if t.startswith("model.layers.") and t.endswith(".weight")
             and any(f".{m}." in t for m in args.match.split(","))
             and (layers is None or int(t.split(".")[2]) in layers)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for t in names:
            t0 = time.time()
            with safe_open(str(args.model / wmap[t]), "pt", device="cuda") as f:
                w = f.get_tensor(t)
            with torch.no_grad():
                wr = quant.rotate_t(w).float()
                c, ov = quant.calibrate(wr, args.target)
                q = quant.codes(wr, quant.row_scale(wr, c)).double()
                del w, wr
                g = q.T @ q if q.shape[0] >= q.shape[1] else q @ q.T
                ev = torch.linalg.eigvalsh(g).flip(0).clamp_min(0)
                tot = float(ev.sum())
                cum = torch.cumsum(ev, 0)
                rec = {"tensor": t, "shape": list(q.shape), "c": c, "overflow1": ov, "sigma_q": float(q.std()),
                       "saturated": float((q.abs() == 127).double().mean()),
                       "energy_top": {str(r): float(cum[r - 1]) / tot for r in TOP if r <= ev.numel()},
                       "seconds": time.time() - t0}
                del q, g, ev, cum
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(t, {k: round(v, 3) for k, v in rec["energy_top"].items() if k in ("8", "32", "128")},
                  f"{rec['seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
