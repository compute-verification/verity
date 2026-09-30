"""Pearl-C4 v3 noise designs on real activations and weights (CPU): `fp4_v3_census.py`'s statistics and per-word scale
classes, with A = a linear's input rows from a Hugging Face model's forward pass and B = rows of that linear's weight.

Per decoder layer chosen, four distinct A operands: the attention input (q_proj; k_proj and v_proj read the same rows),
o_proj's input, the MLP input (gate_proj; up_proj reads the same rows) and down_proj's input. The split removes the
⌊k/100⌋ channels with the largest |x| over the captured tokens from the FP4 operand, a stand-in for the pinned channel
split (`pearlc_activations.py --split`); `--no-split` keeps them. Rows that D-SS's dead-block screen rejects are left
out, and the admitted share is reported.

    python benchmarks/pouw/fp4_v3_real.py --model Qwen/Qwen2.5-0.5B --text README.md --out qwen.json

Needs torch and transformers (CPU is enough); the cached captures go to --cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp4_v3_census import classes_of, measure  # noqa: E402

LINEARS = ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")


def capture(model_id: str, text: str, tokens: int, layers: list[int] | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.eval()
    n = model.config.num_hidden_layers
    layers = layers if layers is not None else [0, n // 2, n - 1]
    ids = tok(text, return_tensors="pt").input_ids[:, :tokens]
    got: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    hooks = []
    for li in layers:
        for name in LINEARS:
            mod = model.get_submodule(f"model.layers.{li}.{name}")
            key = f"layers.{li}.{name}"

            def hook(m, args, key=key):
                got[key] = (args[0][0].float().numpy(), m.weight.float().numpy())

            hooks.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return got


def dss_admitted(x: np.ndarray) -> np.ndarray:
    """D-SS's dead-block screen, as `fp4_v3_census.crafted`: a block with clean amax_b² > 144 ρ² is dead, and a row with
    more than 1 in 64 dead blocks is rejected."""

    rho = np.sqrt((x[:, ::8] ** 2).mean(axis=1, keepdims=True))
    amax = np.abs(x).reshape(len(x), -1, 16).max(axis=2)
    return (amax ** 2 > 144 * rho ** 2).mean(axis=1) <= 1 / 64


def split(x: np.ndarray) -> np.ndarray:
    s = x.shape[1] // 100
    if s == 0:
        return x
    out = x.copy()
    out[:, np.argsort(np.abs(x).max(axis=0))[-s:]] = 0.0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True, help="a text file for the forward pass")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--layers", default=None, help="comma-separated decoder layers (default: first, middle, last)")
    ap.add_argument("--rows", type=int, default=64, help="activation rows and weight rows per operand")
    ap.add_argument("--designs", default="v3-sign-hg4m,v3-sign-hj4m,v3-sign-nm,v1")
    ap.add_argument("--salts", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--words", type=int, default=64)
    ap.add_argument("--no-split", action="store_true")
    ap.add_argument("--cache", default="/tmp/fp4_v3_real")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cache = Path(args.cache) / (args.model.replace("/", "--") + ".npz")
    if cache.exists():
        z = np.load(cache)
        ops = {k[:-2]: (z[k[:-2] + ".x"], z[k[:-2] + ".w"]) for k in z.files if k.endswith(".x")}
    else:
        layers = [int(v) for v in args.layers.split(",")] if args.layers else None
        ops = capture(args.model, Path(args.text).read_text(), args.tokens, layers)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{f"{k}.x": v[0] for k, v in ops.items()}, **{f"{k}.w": v[1] for k, v in ops.items()})

    res = {"model": args.model, "rows": args.rows, "salts": args.salts, "seeds": args.seeds, "split": not args.no_split,
           "linears": {}}
    pick = np.random.default_rng(args.seed)
    for key, (x, w) in sorted(ops.items()):
        x = x.astype(np.float64)
        if not args.no_split:
            x = split(x)
        ok = dss_admitted(x)
        adm = x[ok]
        if len(adm) < 16:
            print(f"{key:28s} k={x.shape[1]:5d} D-SS admits {ok.sum()} of {len(x)} rows: skipped", flush=True)
            res["linears"][key] = {"k": x.shape[1], "dss_admitted": float(ok.mean())}
            continue
        xa = adm[pick.choice(len(adm), min(args.rows, len(adm)) // 16 * 16, replace=False)]
        xb = w[pick.choice(len(w), min(args.rows, len(w)), replace=False)].astype(np.float64)
        entry = {"k": x.shape[1], "dss_admitted": float(ok.mean())}
        for d in args.designs.split(","):
            runs = [measure(xa, xb, d, args.salts, np.random.default_rng(args.seed + 1000 * s)) for s in range(args.seeds)]
            cl = classes_of(xa, xb, d, args.words, np.random.default_rng(args.seed))
            entry[d] = {"debit_min": min(r["debit_2of4"] for r in runs), "debit_max": max(r["debit_2of4"] for r in runs),
                        "salt_dead_max": max(r["salt_dead"] for r in runs), "err_ratio": float(np.mean([r["err_ratio"] for r in runs])),
                        "saturated": float(np.mean([r["saturated"] for r in runs])), **cl}
            e = entry[d]
            print(f"{key:28s} k={x.shape[1]:5d} adm {100 * ok.mean():3.0f}% {d:13s} "
                  f"debit {100 * e['debit_min']:.3f}-{100 * e['debit_max']:.3f}% "
                  f"err {e['err_ratio']:.3f} sat {100 * e['saturated']:.2f}% classes pair {e['pair_median']:.0f} "
                  f"product {e['prod_median']:.0f} (min {e['prod_min']}, <37 {100 * e['prod_lt37']:.0f}%)", flush=True)
        res["linears"][key] = entry
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
