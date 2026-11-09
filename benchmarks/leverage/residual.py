"""How much of a residual-stream fault reaches the logits along the skip path, and how much through mixing.

A fault sets coordinate i of the residual stream after layer l, at the prompt's last position, to v (the honest value
plus delta). The skip path carries delta e_i unchanged to the final norm; later layers add what their changed inputs
make them write. Per fault, with z the centered logits of the first served token:
  direct_r2   1 - ||dz - dz_direct||^2 / ||dz||^2, dz the fault's change in z and dz_direct the change when only
              delta e_i is added to the honest final residual (the skip path alone)
  direct_top1 whether the skip path alone serves the faulted first token
  kept        (h_f - h)_i / delta at the final residual: the fault's own coordinate after the later layers
  spread      ||(h_f - h) off i||_2 / |delta|: what the later layers wrote elsewhere because of it
Faults: --coords random coordinates at plus or minus their box and uniform in it, and the --top coordinates with the
largest boxes (the massive activations), at plus or minus their box, every forward clamped to clamp.py's boxes.

Also checks that no two token embedding rows differ in exactly one coordinate: an embedding fault is never exactly
a token edit.

  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/residual.py --model gpt2 --boxes boxes.pt \\
      --out "$OUT"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import LM, MODELS, Boxes, Cache
from prompts import PROMPTS


def one_coordinate_pairs(E: torch.Tensor) -> tuple[int, int]:
    """(rows differing from another in exactly one coordinate, rows duplicating another): rows are hashed with one
    coordinate dropped under random weights, and neighbours in hash order compared exactly."""
    Ed = E.double()
    r = torch.randn(E.shape[1], dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    base = Ed @ r
    ident = len(Ed) - len(torch.unique(Ed, dim=0))
    pairs = 0
    for i in range(E.shape[1]):
        s, o = (base - Ed[:, i] * r[i]).sort()
        cand = torch.nonzero((s[1:] - s[:-1]).abs() <= 1e-9 * (1 + s[1:].abs())).flatten()
        if len(cand):
            diff = Ed[o[cand]] != Ed[o[cand + 1]]
            pairs += int(((diff.sum(1) == 1) & diff[:, i]).sum())
    return pairs, ident


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", default="0:8")
    ap.add_argument("--coords", type=int, default=16)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    lm = LM(a.model)
    lm.boxes = Boxes.load(a.boxes)
    tok = lm.tokenizer()
    L, d = lm.spec.layers, lm.spec.d
    Uc = lm.E - lm.E.mean(0, keepdim=True)
    pairs, ident = one_coordinate_pairs(lm.E)
    recs = []
    p0, p1 = (int(x) for x in a.prompts.split(":"))
    for pi in range(p0, p1):
        gen = torch.Generator().manual_seed(a.seed * 1_000_003 + pi)
        ids = torch.tensor([tok.encode(PROMPTS[pi]).ids])
        t0 = ids.shape[1] - 1
        taps = {("resid", l): None for l in range(L)}
        lm.forward(ids, Cache(), taps=taps)
        pre = Cache()
        lm.forward(ids[:, :t0], pre)
        cls = lm.boxes.class_at(t0, ids[:, t0:], pre.sink)

        def zc(h: torch.Tensor) -> torch.Tensor:
            n = lm.boxes.apply(("final", L), lm.final_norm(h)[None, None], t0, torch.tensor([[cls]]))
            return Uc @ n[0, 0]

        hL = taps[("resid", L - 1)][0, t0].float()
        zh = zc(hL)
        for l in range(L):
            x = taps[("resid", l)][0, t0].float()
            hi = lm.boxes.hi[("resid", l)][cls].float()
            cands = []
            for c in torch.randperm(d, generator=gen)[:a.coords].tolist():
                sg = 1.0 if torch.rand(1, generator=gen) < 0.5 else -1.0
                cands += [(c, sg * float(hi[c]), "edge"), (c, float((torch.rand(1, generator=gen) * 2 - 1) * hi[c]), "uniform")]
            for c in hi.topk(a.top).indices.tolist():
                cands += [(c, float(hi[c]), "top"), (c, -float(hi[c]), "top")]
            b = len(cands)
            tp = {("resid", L - 1): None}
            xin = taps[("resid", l)][:, t0:].clone()
            first = l + 1
            if first < L:
                lm.forward(ids[:, t0:].expand(b, -1), pre.expand(b), taps=tp, last_only=True,
                           x_in=_faulted(xin, cands), first_layer=first, pos0=t0)
                hf = tp[("resid", L - 1)][:, -1].float()
            else:
                hf = _faulted(xin, cands)[:, -1].float()
            for j, (c, v, kind) in enumerate(cands):
                delta = float(torch.tensor(v).to(torch.bfloat16).float()) - float(x[c])
                if abs(delta) < 1e-6:
                    continue
                hd = hL.clone()
                hd[c] += delta
                zf, zd = zc(hf[j]), zc(hd)
                dz, dzd = zf - zh, zd - zh
                dh = hf[j] - hL
                off = dh.clone()
                off[c] = 0
                recs.append({"prompt": pi, "layer": l, "coord": c, "value": v, "kind": kind, "delta": delta,
                             "box": float(hi[c]), "dz": float(dz.norm()),
                             "direct_r2": float(1 - (dz - dzd).pow(2).sum() / dz.pow(2).sum().clamp_min(1e-12)),
                             "direct_top1": bool(zf.argmax() == zd.argmax()), "flip": bool(zf.argmax() != zh.argmax()),
                             "kept": float(dh[c] / delta), "spread": float(off.norm() / abs(delta))})
        print(f"prompt {pi}: {len(recs)} faults", flush=True)
    (a.out / "residual.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows = []
    for l in range(L):
        for kind in ("edge", "uniform", "top"):
            g = [r for r in recs if r["layer"] == l and r["kind"] == kind]
            if not g:
                continue
            fl = [r for r in g if r["flip"]]
            rows.append({"layer": l, "kind": kind, "n": len(g), "flip": len(fl) / len(g),
                         "direct_r2_median": float(np.median([r["direct_r2"] for r in g])),
                         "direct_top1_of_flips": (sum(r["direct_top1"] for r in fl) / len(fl)) if fl else None,
                         "kept_median": float(np.median([r["kept"] for r in g])),
                         "spread_median": float(np.median([r["spread"] for r in g]))})
    summary = {"model": a.model, "revision": MODELS[a.model]["revision"], "prompts": a.prompts,
               "embedding_rows_one_coordinate_apart": pairs, "embedding_duplicate_rows": ident, "rows": rows}
    (a.out / "residual-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def _faulted(xin: torch.Tensor, cands: list) -> torch.Tensor:
    """The residual stream into the next layer at the last position, one row per candidate (coord, value)."""
    x = xin.expand(len(cands), -1, -1).clone()
    for j, (c, v, _) in enumerate(cands):
        x[j, -1, c] = torch.tensor(v).to(torch.bfloat16).to(x.dtype)
    return x


if __name__ == "__main__":
    raise SystemExit(main())
