"""Island `ref-optim` (class `reference`, no cap on the update): the strongest honest continuation of the start within the
fixed training-FLOP budget, by its optimizer. Every knob is a flag, so one commit runs a whole sweep and each record's notes
name its setting. The launcher runs `islands/reference/train.py`, a symlink to this file.

- matrices: Muon (`--mat-opt muon`, the seed's) or AdamW (`--mat-opt adamw`, the starts' own optimizer); the embedding and the
  norms: AdamW, each with its own rate; `--freeze-embed` keeps the tied embedding at θ (its weight gradient is then not paid);
- schedule: warmup (steps, or a fraction when below 1) times `linear` (the seed's: 1 - t/T from the first step), `wsd`
  (flat, then the last `--decay-frac` decays, linear or 1 - sqrt), `cosine` or `const`, floored at `--final-frac`;
- data: `--fresh` continues the start's own data order past the windows it trained on (lf starts: seed 0 from its step),
  else the stream permuted by `--data-seed` (the seed's 1);
- EMA of the weights (`--ema-frac f`: the average's horizon is f of the steps so far); at each snapshot and at the end the
  raw or EMA weights are submitted, whichever has the lower loss on training-stream windows the run never trains on (each
  such forward is logged and priced);
- snapshots at `--snap-at` fractions of the steps, for the FLOPs-to-target curve; `--forks` adds cooldown branches: at each
  given fraction a branch of `--fork-len` times the steps so far decays the rate linearly toward 0, is snapshotted, and the
  run returns to the weights and optimizer state it forked from (the branch's passes stay in the step log, so every later
  point pays for it).

E is the trained tensors in full (`prepare.encode_full`). This file is the island's one editable file; prepare.py, evaluate.py,
guard.py and candidate.sh are fixed (README.md, the program).

    python guard.py islands/reference/train.py --start m100 --out SUB [--mat-lr 0.004] [--embed-lr 3e-4] [--batch 64] ...
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare

KLASS = "reference"


def newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).to(g.dtype)


def ns_flops(shape: tuple[int, int], steps: int = 5) -> int:
    m, n = min(shape), max(shape)
    return steps * (4 * m * m * n + 2 * m ** 3)


class Muon(torch.optim.Optimizer):
    """Muon: momentum, Newton-Schulz orthogonalized. `scale` "shape" is the seed's sqrt(max(1, rows/cols)); "rms" is
    0.2 sqrt(max(rows, cols)), an update RMS of 0.2 like AdamW's, so its rate is on AdamW's scale."""

    def __init__(self, params, lr: float, momentum: float = 0.95, nesterov: bool = True, wd: float = 0.0,
                 scale: str = "shape", ns_steps: int = 5):
        super().__init__(params, {"lr": lr, "momentum": momentum, "nesterov": nesterov, "wd": wd, "scale": scale,
                                  "ns_steps": ns_steps})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state.setdefault(p, {"buf": torch.zeros_like(p)})
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf, g["ns_steps"])
                rows, cols = p.shape
                s = max(1.0, rows / cols) ** 0.5 if g["scale"] == "shape" else 0.2 * max(rows, cols) ** 0.5
                if g["wd"]:
                    p.mul_(1 - g["lr"] * g["wd"])
                p.add_(u, alpha=-g["lr"] * s)


def lr_mult(step: int, n: int, a: argparse.Namespace) -> float:
    warm = a.warmup * n if a.warmup < 1 else a.warmup
    ramp = min(1.0, (step + 1) / warm) if warm > 0 else 1.0
    t = step / n
    if a.sched == "linear":
        d = 1 - t
    elif a.sched == "cosine":
        d = 0.5 * (1 + math.cos(math.pi * t))
    elif a.sched == "const":
        d = 1.0
    else:
        start = 1 - a.decay_frac
        p = 0.0 if t < start else (t - start) / a.decay_frac
        d = 1 - math.sqrt(p) if a.decay_shape == "sqrt" else 1 - p
    return ramp * (a.final_frac + (1 - a.final_frac) * d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mat-opt", choices=("muon", "adamw"), default="muon")
    ap.add_argument("--mat-lr", type=float, default=0.01, help="the block matrices' rate (Muon's or AdamW's)")
    ap.add_argument("--embed-lr", type=float, default=3e-4)
    ap.add_argument("--norm-lr", type=float, default=None, help="default: --embed-lr")
    ap.add_argument("--wd", type=float, default=0.0, help="decoupled weight decay on the block matrices")
    ap.add_argument("--embed-wd", type=float, default=0.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--momentum", type=float, default=0.95, help="Muon's")
    ap.add_argument("--no-nesterov", action="store_true")
    ap.add_argument("--muon-scale", choices=("shape", "rms"), default="shape")
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--freeze-embed", action="store_true")
    ap.add_argument("--freeze-norms", action="store_true")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sched", choices=("linear", "wsd", "cosine", "const"), default="linear")
    ap.add_argument("--warmup", type=float, default=50, help="steps, or a fraction of the steps when below 1")
    ap.add_argument("--decay-frac", type=float, default=0.2, help="wsd: the decaying share of the steps")
    ap.add_argument("--decay-shape", choices=("linear", "sqrt"), default="linear")
    ap.add_argument("--final-frac", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--fresh", action="store_true", help="continue the start's own data order past what it trained on")
    ap.add_argument("--ema-frac", type=float, default=0.0)
    ap.add_argument("--val-rows", type=int, default=64, help="held-back training-stream windows per validation")
    ap.add_argument("--snap-at", default="0.25,0.5,0.75", help="fractions of the steps (comma-separated; empty: none)")
    ap.add_argument("--forks", default="", help="fractions of the steps where a cooldown branch forks (comma-separated)")
    ap.add_argument("--fork-len", type=float, default=0.15, help="a branch's steps, as a fraction of the steps before it")
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    a.norm_lr = a.embed_lr if a.norm_lr is None else a.norm_lr
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = prepare.DEVICE.split(":")[0]
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.data = p.data.float()
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    mats = [n for n, p in params.items() if p.ndim == 2 and n != embed]
    norms = [n for n, p in params.items() if p.ndim == 1]
    trained = mats + ([] if a.freeze_embed else [embed]) + ([] if a.freeze_norms else norms)
    for n, p in params.items():
        p.requires_grad_(n in trained)
    opts = []
    adam_groups = []
    if a.mat_opt == "muon":
        opts.append(Muon([params[n] for n in mats], lr=a.mat_lr, momentum=a.momentum, nesterov=not a.no_nesterov,
                         wd=a.wd, scale=a.muon_scale, ns_steps=a.ns_steps))
        opts[0].param_groups[0]["base_lr"] = a.mat_lr
    else:
        adam_groups.append({"params": [params[n] for n in mats], "lr": a.mat_lr, "base_lr": a.mat_lr, "weight_decay": a.wd})
    if not a.freeze_embed:
        adam_groups.append({"params": [params[embed]], "lr": a.embed_lr, "base_lr": a.embed_lr, "weight_decay": a.embed_wd})
    if not a.freeze_norms:
        adam_groups.append({"params": [params[n] for n in norms], "lr": a.norm_lr, "base_lr": a.norm_lr, "weight_decay": 0.0})
    if adam_groups:
        opts.append(torch.optim.AdamW(adam_groups, betas=(a.beta1, a.beta2), fused=dev == "cuda"))
    trainable = {"full": sorted(trained)}
    tokens = a.batch * prepare.EVAL_CTX
    extra = sum(ns_flops(tuple(params[n].shape), a.ns_steps) for n in mats) if a.mat_opt == "muon" else 0
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    snap_at = [float(x) for x in a.snap_at.split(",") if x.strip()]
    fork_at = [float(x) for x in a.forks.split(",") if x.strip()]
    val_tokens = a.val_rows * prepare.EVAL_CTX
    n_val = ((len(snap_at) + 1) * (2 if a.ema_frac else 1) + len(fork_at)) if a.val_rows else 0
    val_flops = n_val * prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": val_tokens}])["flops"]
    room = a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - val_flops

    def fork_lens(n: int) -> dict[int, int]:
        return {max(1, round(n * f)): max(1, round(a.fork_len * n * f)) for f in fork_at if 0 < f < 1}

    n_steps = int(room // (per_step * (1 + a.fork_len * sum(fork_at))))
    while n_steps > 1 and (n_steps + sum(fork_lens(n_steps).values())) * per_step > room:
        n_steps -= 1
    forks = fork_lens(n_steps)

    if a.fresh:
        prov = rec["provenance"]
        if rec["config"]["family"] != "lf" or prov.get("data_seed") != 0:
            raise SystemExit("train: --fresh needs an lf start trained on data order 0")
        data = prepare.TrainData(rec["tokenizer"], seed=0)
        data.at = int(prov["step"]) * int(prov["batch"])
    else:
        data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    val_idx = data.order[len(data.order) - a.val_rows:] if a.val_rows else None
    if val_idx is not None and data.at + (n_steps + sum(forks.values())) * a.batch > len(data.order) - a.val_rows:
        raise SystemExit("train: the training windows would reach the validation windows")

    def loss_of(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(dev, dtype=torch.bfloat16):
            logits = net(x[:, :-1])
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1))

    lossf = loss_of if a.no_compile else torch.compile(loss_of)
    steps: list[dict] = []

    @torch.no_grad()
    def validate() -> float:
        c = prepare.EVAL_CTX
        total = 0.0
        for i in range(0, a.val_rows, a.micro):
            idx = sorted(val_idx[i: i + a.micro])
            rows = torch.stack([torch.from_numpy(data.tokens[j * c: j * c + c + 1].astype("int64")) for j in idx])
            total += float(lossf(rows.to(prepare.DEVICE))) * len(idx)
        steps.append({"op": "forward", "tokens": val_tokens, "what": "validation on held-back training windows"})
        return total / a.val_rows

    ema = [p.detach().clone() for p in params.values()] if a.ema_frac else None
    plist = list(params.values())
    names = list(params)

    def swap_ema() -> None:
        for p, e in zip(plist, ema):
            tmp = p.data.clone()
            p.data.copy_(e)
            e.copy_(tmp)

    def choose() -> tuple[str, dict]:
        """The weights to submit: raw, or EMA when its held-back loss is lower; logs both losses."""
        if not a.val_rows:
            return "raw", {}
        raw = validate()
        if ema is None:
            return "raw", {"val_raw": raw}
        swap_ema()
        ev = validate()
        if ev < raw:
            m = model_of(params, theta, trained)
            swap_ema()
            return "ema", {"val_raw": raw, "val_ema": ev, "model": m}
        swap_ema()
        return "raw", {"val_raw": raw, "val_ema": ev}

    marks = {max(1, round(n_steps * f)) for f in snap_at if 0 < f < 1}
    out = Path(a.out)
    snap_log = []
    print(json.dumps({"steps": n_steps, "forks": forks, "flops_per_step": per_step, "tokens_per_step": tokens,
                      "val_flops": val_flops, "trained": len(trained), "data_at": data.at}), flush=True)
    t0 = time.perf_counter()
    trained_params = [params[n] for n in trained]

    def train_step(m: float) -> float:
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * m
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            loss = lossf(x) * (len(x) / a.batch)
            loss.backward()
        if a.clip:
            torch.nn.utils.clip_grad_norm_(trained_params, a.clip)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if extra:
            steps.append({"op": "extra", "flops": extra, "what": "muon newton-schulz"})
        return float(loss.detach()) * a.batch / len(x)

    def snapshot(model: dict, entry: dict) -> None:
        d = out / "snap" / f"{len(snap_log):03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "E.bin").write_bytes(prepare.encode_full(model))
        (d / "meta.json").write_text(json.dumps({"steps_prefix": len(steps)}) + "\n")
        snap_log.append(entry)
        print(json.dumps({"snapshot": len(snap_log) - 1, **entry}), flush=True)

    @torch.no_grad()
    def fork(step: int, k: int) -> None:
        """A cooldown branch: k steps from the current rate linearly toward 0, snapshot, then back to where it forked."""
        saved = [p.detach().clone() for p in trained_params]
        states = [{p: {n: (v.clone() if torch.is_tensor(v) else v) for n, v in st.items()} for p, st in o.state.items()}
                  for o in opts]
        m0 = lr_mult(step - 1, n_steps, a)
        with torch.enable_grad():
            for j in range(k):
                train_step(m0 * (k - j) / k)
        entry = {"step": step, "fork_steps": k, "weights": "fork"}
        if a.val_rows:
            entry["val_raw"] = validate()
        snapshot(model_of(params, theta, trained), entry)
        for p, s in zip(trained_params, saved):
            p.copy_(s)
        for o, st in zip(opts, states):
            o.state.clear()
            o.state.update(st)

    for step in range(n_steps):
        m = lr_mult(step, n_steps, a)
        loss = train_step(m)
        if ema is not None:
            beta = 1 - 1 / max(1.0, a.ema_frac * (step + 1))
            torch._foreach_lerp_(ema, [p.detach() for p in plist], 1 - beta)
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": loss, "lr_mult": round(m, 4),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            which, info = choose()
            model = info.pop("model", None) or model_of(params, theta, trained)
            snapshot(model, {"step": step + 1, "weights": which, **info})
        if step + 1 in forks:
            fork(step + 1, forks[step + 1])
    which, info = choose()
    model = info.pop("model", None) or model_of(params, theta, trained)
    final = {"step": n_steps, "weights": which, **info}
    print(json.dumps({"final": final, "s": round(time.perf_counter() - t0, 1)}), flush=True)
    prepare.submit(out, a.start, KLASS, prepare.encode_full(model), {**theta, **model}, trainable, steps,
                   changed=sorted(trained), notes={"direction": "ref-optim", "args": vars(a), "steps": n_steps, "forks": forks,
                                                   "snapshots": snap_log, "final": final})
    return 0


def model_of(params: dict, theta: dict, trained: list[str]) -> dict:
    return {n: params[n].detach().to(theta[n].dtype).cpu() for n in trained}


if __name__ == "__main__":
    raise SystemExit(main())
