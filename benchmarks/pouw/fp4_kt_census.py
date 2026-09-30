#!/usr/bin/env python3
"""Pearl-C4's enforced rule (#580: F1′ with κ inside, F2 at c_L = 2.0, R1, the 1/400 tile reject) on Qwen2.5-7B's
linears under the keyed transforms (docs/pouw/keyed-transforms.md §5, §14), against fp4_census's activation families
(CPU; the rule has no GPU evaluation).

The adversary commits any activation rows it likes, so each family's A rows (``fp4_census.family``, 64 rows at the
linear's k, BF16) are the chain's X; the weights are the registered checkpoint's or the transformed ones, as the
keyed-transforms lane deploys them (``kt_stream_ppl.make_layer_transform`` without +voi):
  none|pc      the checkpoint as registered
  rotb8s|pc    rung 3: q/k/v/gate/up take W·diag(γ/g_keep)·Q with S's columns off the chain (zero here, and in X), o and
               down Qᵀ·W; Q is prep.npz's per-channel-position rotation
  rotb8s|al    the same with the 16-aligned placement (rank 2's)
The verifier's judgement is #580's ``real.py`` unchanged, on one npz per (variant, family): per linear, admission of
every X and W row, and ``--tiles`` seeded 64 x 64 tiles through ``PearlC4.tile_cap`` (items, debit / cap, rejected).

Shortcuts: W is 4 keyed tile-aligned 64-row blocks of each linear (a tile draw over them is a tile draw over W's
blocks); real.py's split is its stand-in (the top 1% of channels by max|x| leave the unit); the noise is keyed by labels.

    python fp4_kt_census.py prep --out DIR --prep .../s7b/prep.npz
    python fp4_kt_census.py run --out DIR --rule-tree TREE --budget-min 25

``prep`` writes DIR/npz/<variant>--<family>.npz once (restartable per variant); ``run`` works through them with real.py
(its task JSONs are the checkpoints), exits 99 once the budget is spent with work left, 0 when every summary is written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp4_census as C  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B"
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
IN_NORM = {"self_attn.q_proj": "input_layernorm", "self_attn.k_proj": "input_layernorm",
           "self_attn.v_proj": "input_layernorm", "mlp.gate_proj": "post_attention_layernorm",
           "mlp.up_proj": "post_attention_layernorm"}
VARIANTS = ("none|pc", "rotb8s|pc", "rotb8s|al")
FAMILIES = tuple(f for f in C.FAMILIES if f != "coherent")    # coherent's A is gaussian's: it only plants B
SEED = 20260930
W_BLOCKS, X_ROWS, TILE = 4, 64, 64


def tag(variant: str) -> str:
    return variant.replace("|", "_")


def bf16(x):
    import torch
    return torch.as_tensor(np.asarray(x, np.float32)).to(torch.bfloat16).float().numpy()


def keep_gamma(gamma, blk):
    import torch
    gk = torch.ones_like(gamma)
    for b in torch.unique(blk).tolist():
        m = blk == b
        gk[m] = gamma[m] if b < 0 else gamma[m].pow(2).mean().sqrt()
    return gk


def w_blocks(name: str, n: int) -> np.ndarray:
    rng = np.random.default_rng([SEED, *name.encode(), 0x57])
    b = np.sort(rng.choice(n // TILE, min(W_BLOCKS, n // TILE), replace=False))
    return (b[:, None] * TILE + np.arange(TILE)[None]).reshape(-1)


def layer_weights(shards, i: int, variant: str, P) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """{linear: (W rows sampled, as BF16 in float32; S's positions or None)} for layer i under `variant`."""
    import torch
    rot = not variant.startswith("none")
    place = variant.split("|")[1]
    get = lambda t: shards.get_tensor(f"model.layers.{i}.{t}").float()  # noqa: E731
    if rot:
        q = torch.from_numpy(P[f"q_{place}"]).float()
        blk = torch.from_numpy(P[f"blk_{place}"])
        ps = np.asarray(P[f"ps_{place}"], np.int64)
    out = {}
    for lin in LINEARS:
        w = get(f"{lin}.weight")
        rows = torch.from_numpy(w_blocks(f"{i}.{lin}", w.shape[0]))
        side = None
        if rot and lin in IN_NORM:
            gam = get(f"{IN_NORM[lin]}.weight")
            w = ((w[rows] * (gam / keep_gamma(gam, blk))[None, :]) @ q)
            if len(ps):
                side = ps
                w[:, torch.from_numpy(ps)] = 0.0
        elif rot:
            w = (q.T @ w)[rows]
        else:
            w = w[rows]
        out[lin] = (w.to(torch.bfloat16).float().numpy(), side)
    return out


class Shards:
    def __init__(self, model: str):
        import json as _j

        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        self.root = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors"]))
        self.map = _j.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.open = lambda f: safe_open(str(self.root / f), "pt")  # noqa: E731

    def get_tensor(self, name: str):
        with self.open(self.map[name]) as f:
            return f.get_tensor(name)


def cmd_prep(args) -> int:
    out = Path(args.out) / "npz"
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(v) for v in args.layers.split(",")]
    families = args.families.split(",")
    P = np.load(args.prep)
    shards = Shards(args.model)
    for variant in args.variants.split(","):
        if all((out / f"{tag(variant)}--{f}.npz").exists() for f in families):
            continue
        t0 = time.time()
        ws = {i: layer_weights(shards, i, variant, P) for i in layers}
        for fam in families:
            dst = out / f"{tag(variant)}--{fam}.npz"
            if dst.exists():
                continue
            members = {}
            for i in layers:
                for lin, (w, side) in ws[i].items():
                    k = w.shape[1]
                    rng = np.random.default_rng([SEED, i, k, *fam.encode()])
                    x = C.family(fam, X_ROWS, k, rng)[0]
                    if side is not None:
                        x[:, side] = 0.0
                    members[f"layers.{i}.{lin}.x"] = bf16(x)
                    members[f"layers.{i}.{lin}.w"] = w
            tmp = dst.with_name(dst.stem + ".tmp.npz")
            np.savez_compressed(tmp, **members)
            tmp.replace(dst)
        print(json.dumps({"variant": variant, "seconds": round(time.time() - t0)}), flush=True)
    (Path(args.out) / "prep.json").write_text(json.dumps({
        "model": args.model, "prep": str(args.prep), "layers": layers, "variants": args.variants.split(","),
        "families": families, "w_blocks": W_BLOCKS, "x_rows": X_ROWS, "seed": SEED}, indent=1))
    return 0


def cmd_run(args) -> int:
    sys.path.insert(0, str(Path(args.rule_tree) / "benchmarks/pouw/pearl_c4"))
    import real
    t0 = time.monotonic()
    npzs = sorted((Path(args.out) / "npz").glob("*--*.npz"))
    npzs = [p for p in npzs if not p.name.endswith(".tmp.npz")]
    if not npzs:
        raise SystemExit("no npz: run prep first")
    for p in npzs:
        dst = Path(args.out) / "real" / p.stem
        if (dst / "summary.json").exists():
            continue
        left = args.budget_min - (time.monotonic() - t0) / 60
        if left <= 1:
            return 99
        rc = real.main([str(p), str(dst), "--tiles", str(args.tiles), "--workers", str(args.workers),
                        "--budget-min", f"{left:.1f}", "--seed", str(args.seed)])
        if rc:
            return rc
    rows = {}
    for p in npzs:
        s = json.loads((Path(args.out) / "real" / p.stem / "summary.json").read_text())
        rows[p.stem] = {"over_cap_tiles": s["over_cap_tiles"], "rejected_tiles": s["rejected_tiles"],
                        "tiles_debited": s["tiles_debited"], "max_ratio": s["max_ratio"]}
    (Path(args.out) / "summary.json").write_text(json.dumps(rows, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["prep", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prep", default="/workspace/pouw/keyed-transforms/s7b/prep.npz")
    ap.add_argument("--layers", default="0,14,27")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--rule-tree", default="")
    ap.add_argument("--tiles", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-min", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=20261930)
    args = ap.parse_args(argv)
    return cmd_prep(args) if args.cmd == "prep" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
