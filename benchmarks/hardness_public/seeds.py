"""M2-E's seed test: does training keep the init's randomness in the int8 low bits, and does a seed change randomize them?

`train --arm NAME` trains one arm of a small GPT on HT-ML's TinyStories token cache and writes its init and final
linear weights to `--scratch`; `compare` reads every arm there and writes `seeds.json` to `--out`. The arms:

- A, A_rep: init seed 1, data seed 1, no weight decay, 4 threads (A_rep tests bit-exact replay);
- A_t2: as A on 2 threads (a different reduction order);
- C: init seed 1, data seed 2 (same public init, another data order);
- B: init seed 2, data seed 2;
- D01, D2: as A with AdamW weight decay 0.1 and 2.0.

For each linear and pair of arms it reports, on per-channel symmetric int8: agreement of each two's-complement bit
plane, exact equality and correlation; for A against B also after matching MLP hidden units (Hungarian on row
correlation). Per arm: how much of the final weight is the init (correlation, the init's spread in final LSBs, LSB
agreement with the init rounded on the final scale) and the update's spectrum.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = {
    "A": dict(init=1, data=1, wd=0.0, threads=4),
    "A_rep": dict(init=1, data=1, wd=0.0, threads=4),
    "A_t2": dict(init=1, data=1, wd=0.0, threads=2),
    "C": dict(init=1, data=2, wd=0.0, threads=4),
    "B": dict(init=2, data=2, wd=0.0, threads=4),
    "D01": dict(init=1, data=1, wd=0.1, threads=4),
    "D2": dict(init=1, data=1, wd=2.0, threads=4),
}
D, LAYERS, HEADS, CTX, VOCAB = 192, 4, 4, 128, 4096
LINEARS = ("qkv", "proj", "fc", "out")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.fc = nn.Linear(D, 4 * D, bias=False)
        self.out = nn.Linear(4 * D, D, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, HEADS, d // HEADS).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(D)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * LAYERS) if name.endswith(("proj.weight", "out.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for blk in self.blocks:
            x = blk(x)
        return self.lnf(x) @ self.emb.weight.T


def linears(model):
    return {f"{i}.{n}": getattr(b, n).weight.detach().clone().numpy() for i, b in enumerate(model.blocks) for n in LINEARS}


def train(arm, data_dir, scratch, steps, batch, save_at=(), resume=None, tail_seed=None, tag=None, mutate=None):
    """One arm; `save_at` writes resumable checkpoints, `resume` continues one, with the data order reseeded by
    `tail_seed` (None keeps the checkpoint's own stream), written under `tag`. `mutate(model)` edits the resumed
    weights in place before the first step."""
    cfg = ARMS[arm]
    tag = tag or arm
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["init"])
    model = GPT()
    w0 = linears(model)
    decay = [p for n, p in model.named_parameters() if p.dim() == 2 and "blocks" in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() == 2 and "blocks" in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["wd"]}, {"params": rest, "weight_decay": 0.0}],
                            lr=1e-3, betas=(0.9, 0.95))
    tr = np.memmap(Path(data_dir) / "train.bin", dtype=np.uint16, mode="r")
    va = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(cfg["data"])
    start = 0
    if resume:
        ck = torch.load(resume, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        if tail_seed is None:
            rng.bit_generator.state = ck["rng"]
        else:
            rng = np.random.default_rng(tail_seed)
        if mutate:
            with torch.no_grad():
                mutate(model)

    def get(arr, g):
        off = g.integers(0, len(arr) - CTX - 1, size=batch)
        x = np.stack([arr[o:o + CTX + 1] for o in off]).astype(np.int64)
        return torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])

    log = []
    for step in range(start, steps):
        if step in save_at:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "rng": rng.bit_generator.state}, Path(scratch) / f"{arm}.ckpt{step}.pt")
        lr = 1e-3 * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get(tr, rng)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            log.append([step, loss.item()])
            print(arm, step, log[-1][1], flush=True)
    vg = np.random.default_rng(999)
    with torch.no_grad():
        val = float(np.mean([F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
                             for x, y in (get(va, vg) for _ in range(32))]))
    w1 = linears(model)
    np.savez(Path(scratch) / f"{tag}.npz", **{f"init/{k}": v for k, v in w0.items()}, **{f"final/{k}": v for k, v in w1.items()})
    (Path(scratch) / f"{tag}.json").write_text(json.dumps({"arm": arm, "tag": tag, **cfg, "steps": steps, "batch": batch,
                                                           "resume": str(resume), "tail_seed": tail_seed,
                                                           "val_loss": val, "log": log}))


def tails(scratch, out, steps, lengths):
    """The tail test: A's final weights against A resumed at steps - L with a fresh data order."""
    a = dict(np.load(Path(scratch) / "A.npz"))
    names = [f"{i}.{n}" for i in range(LAYERS) for n in LINEARS]
    res = {}
    for L in lengths:
        for kind in ("fresh", "same"):
            p = Path(scratch) / f"A_tail{L}_{kind}.npz"
            if not p.exists():
                continue
            b = dict(np.load(p))
            per = {k: pair(a[f"final/{k}"], b[f"final/{k}"]) for k in names}
            res[f"{L}_{kind}"] = {"per_matrix": per,
                                  "planes_agree_mean": np.mean([per[k]["planes_agree"] for k in names], axis=0).tolist(),
                                  "val_loss": json.loads((Path(scratch) / f"A_tail{L}_{kind}.json").read_text())["val_loss"]}
    (Path(out) / "tails.json").write_text(json.dumps({"steps": steps, "lengths": lengths, "tails": res}))


def q8(w):
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    return np.clip(np.rint(w / s), -127, 127).astype(np.int16), s


def planes_agree(qa, qb):
    ua, ub = qa.astype(np.int16) & 0xFF, qb.astype(np.int16) & 0xFF
    return [float((((ua >> p) & 1) == ((ub >> p) & 1)).mean()) for p in range(8)]


def pair(wa, wb):
    qa, _ = q8(wa)
    qb, _ = q8(wb)
    return {"planes_agree": planes_agree(qa, qb), "exact_equal": float((qa == qb).mean()),
            "corr": float(np.corrcoef(wa.ravel(), wb.ravel())[0, 1]), "bit_identical_float": bool(np.array_equal(wa, wb))}


def energy_top(m, fracs=(1 / 8, 1 / 4)):
    s2 = np.linalg.svd(m.astype(np.float64), compute_uv=False) ** 2
    n = len(s2)
    return {f"{f:.3f}": float(s2[: max(1, int(n * f))].sum() / s2.sum()) for f in fracs}


def compare(scratch, out):
    from scipy.optimize import linear_sum_assignment

    import natural

    arms = {a: dict(np.load(Path(scratch) / f"{a}.npz")) for a in ARMS if (Path(scratch) / f"{a}.npz").exists()}
    meta = {a: json.loads((Path(scratch) / f"{a}.json").read_text()) for a in arms}
    res = {"arms": meta, "pairs": {}, "retention": {}, "aligned_A_B": {}, "natural_A": {}, "natural_init": {}}
    names = [f"{i}.{n}" for i in range(LAYERS) for n in LINEARS]
    for x in ("A_rep", "A_t2", "C", "B", "D01", "D2"):
        if x in arms:
            res["pairs"][f"A~{x}"] = {k: pair(arms["A"][f"final/{k}"], arms[x][f"final/{k}"]) for k in names}
    rng = np.random.default_rng(7)
    for a, w in arms.items():
        r = {}
        for k in names:
            w0, w1 = w[f"init/{k}"], w[f"final/{k}"]
            q1, s1 = q8(w1)
            q0 = np.clip(np.rint(w0 / s1), -127, 127).astype(np.int16)
            g = rng.standard_normal(w1.shape)
            r[k] = {"corr_init": float(np.corrcoef(w0.ravel(), w1.ravel())[0, 1]),
                    "init_norm_ratio": float(np.linalg.norm(w0) / np.linalg.norm(w1)),
                    "init_sigma_in_final_lsb": float((w0.std(axis=1, keepdims=True) / s1).mean()),
                    "planes_agree_with_init": planes_agree(q1, q0),
                    "update_energy_top": energy_top(w1 - w0), "gauss_energy_top": energy_top(g),
                    "final_energy_top": energy_top(w1)}
        res["retention"][a] = r
    if "B" in arms:
        for i in range(LAYERS):
            fa, fb = arms["A"][f"final/{i}.fc"], arms["B"][f"final/{i}.fc"]
            oa, ob = arms["A"][f"final/{i}.out"], arms["B"][f"final/{i}.out"]
            za = (fa - fa.mean(1, keepdims=True)) / fa.std(1, keepdims=True)
            zb = (fb - fb.mean(1, keepdims=True)) / fb.std(1, keepdims=True)
            c = za @ zb.T / fa.shape[1]
            _, perm = linear_sum_assignment(-c)
            res["aligned_A_B"][f"{i}.fc"] = {**pair(fa, fb[perm]), "mean_matched_row_corr": float(c[np.arange(len(perm)), perm].mean())}
            res["aligned_A_B"][f"{i}.out"] = pair(oa, ob[:, perm])
    for k in names:
        q, _ = q8(arms["A"][f"final/{k}"])
        rec = natural.analyse(q, np.random.default_rng(1), False)
        res["natural_A"][k] = {"best_cost": rec["best_cost"], "lsb": rec["lsb"], "sigma_q": rec["alphabet"]["sigma_q"],
                               "eckart_young_rmin": rec["eckart_young_rmin"], "tiled_r0": rec["tiled_r0"]}
        q0, _ = q8(arms["A"][f"init/{k}"])
        rec0 = natural.analyse(q0, np.random.default_rng(1), False)
        res["natural_init"][k] = {"best_cost": rec0["best_cost"], "sigma_q": rec0["alphabet"]["sigma_q"],
                                  "eckart_young_rmin": rec0["eckart_young_rmin"]}
    for a in arms:
        q, _ = q8(arms[a]["final/0.fc"])
        np.save(Path(out) / f"{a}.final_q8.0.fc.npy", q.astype(np.int8))
    (Path(out) / "seeds.json").write_text(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "compare", "tails"))
    ap.add_argument("--arm")
    ap.add_argument("--data", default=os.path.expanduser("~/hardness-ml/tinystories-bpe4096"))
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-at", default="")
    ap.add_argument("--resume")
    ap.add_argument("--tail-seed", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--lengths", default="")
    args = ap.parse_args()
    if args.cmd == "train":
        save_at = {int(x) for x in args.save_at.split(",") if x}
        train(args.arm, args.data, args.scratch, args.steps, args.batch, save_at, args.resume, args.tail_seed, args.tag)
    elif args.cmd == "tails":
        tails(args.scratch, args.out, args.steps, [int(x) for x in args.lengths.split(",")])
    else:
        compare(args.scratch, args.out)


if __name__ == "__main__":
    main()
