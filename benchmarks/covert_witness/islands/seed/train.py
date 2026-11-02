"""Island `seed` (E is a seed log, at most 100,000 bytes), direction seed-mezo: zeroth-order training whose update is a seed
log. Step i draws z_i from a CUDA generator seeded `seed + i` over the named tensors (`--names`), and the decoder adds
−c_i z_i to their fp32 masters (`prepare.seedlog_step`). E is the coefficients (`--coeff`, bf16 by default: 2 bytes a step, so
about 50,000 directions fit in the cap; the rounding error, 2^-9 relative, is far below the estimates' noise).

Methods (`--method`):

- `mezo`: two forwards a step on one batch at θ ± εz, c = lr · (L₊ − L₋)/2ε. `--momentum β` keeps a heavy-ball trajectory and
  stores each direction's total coefficient c_i = ĝ_i Σ_{s≥i} lr_s β^{s−i}; the submission is then decode(θ, E), replayed.
- `fo`: a backward pass a step, its gradient projected on `--dirs-per-step` seed directions, c = lr · (g · z): the seed log's
  ceiling with exact projections (not zeroth-order; for comparison).

`--kappa K` sets lr = K / tr Ĥ, tr Ĥ the mean curvature z^T H z over `--cal-dirs` directions measured at the start with three
fp32 forwards each (logged as forward passes); `--lr` sets it absolutely. Each MeZO step splits its batch in halves, so the log
reports the signal fraction |g|² / |g_B|² (the product of the halves' estimates over their squares) and a running prediction
of the gain, η ĝ_A ĝ_B − c² tr Ĥ / 2 summed over steps, in nats a token.

    python guard.py islands/seed/train.py --start s50 --out SUB [--method mezo|fo] [--names blocks|all|gains|top:K|re:REGEX]
        [--kappa K | --lr LR] [--eps 1e-3] [--batch 8] [--ctx 1024] [--coeff bf16] [--max-dirs N] [--momentum B]
        [--schedule const|linear] [--budget-fraction 0.98] [--time-budget 1000]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare

KLASS = "seed"
T_MAIN = time.perf_counter()


def select(params: dict, rec: dict, spec: str) -> list[str]:
    nodes = prepare.positions(rec)
    L = rec["config"]["n_layer"]
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    layer = {}
    for n in nodes:
        layer[n["param"]] = min(layer.get(n["param"], 10**9), n["layer"])
    names = sorted(params)
    if spec == "all":
        return names
    if spec == "blocks":
        return [n for n in names if n != embed]
    if spec == "gains":
        return [n for n in names if params[n].ndim == 1]
    if spec.startswith("top:"):
        k = int(spec[4:])
        return [n for n in names if n != embed and layer[n] >= L + 1 - k]
    if spec.startswith("re:"):
        return [n for n in names if re.search(spec[3:], n)]
    raise SystemExit(f"train: unknown --names {spec}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="mezo", choices=("mezo", "fo"))
    ap.add_argument("--names", default="blocks")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=prepare.EVAL_CTX)
    ap.add_argument("--coeff", default="bf16", choices=("bf16", "f16", "f32"))
    ap.add_argument("--max-dirs", type=int, default=None)
    ap.add_argument("--dirs-per-step", type=int, default=1, help="fo: directions per backward pass")
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--schedule", default="const", choices=("const", "linear"))
    ap.add_argument("--cal-dirs", type=int, default=8)
    ap.add_argument("--cal-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--time-budget", type=float, default=1000.0, help="seconds from start to the loop's end")
    a = ap.parse_args()
    if (a.lr is None) == (a.kappa is None):
        raise SystemExit("train: give one of --lr and --kappa")
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = prepare.DEVICE
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    names = select(params, rec, a.names)
    for p in params.values():
        p.requires_grad_(False)
    live = [params[n] for n in names]
    masters = [params[n].detach().float().clone() for n in names]
    d = sum(m.numel() for m in masters)
    trainable = {"full": names}
    tokens = a.batch * a.ctx
    fwd_tok = prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": 1, "ctx": a.ctx}])["flops"]
    if a.method == "mezo":
        per_step = 2 * tokens * fwd_tok
    else:
        per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens, "ctx": a.ctx}])["flops"]
    if a.method == "mezo" and a.batch < 2:
        raise SystemExit("train: mezo splits its batch in halves; --batch 2 at least")
    cal_tokens = a.cal_batch * a.ctx
    cal_flops = (2 * a.cal_dirs + 1) * cal_tokens * fwd_tok if a.kappa is not None else 0
    cap = prepare.CLASSES[KLASS]["max_e_bytes"]
    header = len(prepare.encode_seedlog(names, a.seed, [], a.coeff))
    width = {"bf16": 2, "f16": 2, "f32": 4}[a.coeff]
    max_dirs = (cap - header) // width
    if a.max_dirs:
        max_dirs = min(max_dirs, a.max_dirs)
    budget = a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - cal_flops
    if a.method == "mezo":
        n_steps = min(int(budget // per_step), max_dirs)
    else:
        extra = 2 * d * a.dirs_per_step
        n_steps = min(int(budget // (per_step + extra)), max_dirs // a.dirs_per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed, ctx=a.ctx)
    gen = torch.Generator(device=dev)
    steps: list[dict] = []
    vocab = rec["config"]["vocab"]
    info = {"method": a.method, "names": a.names, "tensors": len(names), "d": d, "steps": n_steps, "max_dirs": max_dirs,
            "header_bytes": header, "flops_per_step": per_step}
    print(json.dumps(info), flush=True)

    def draw(seed: int) -> list[torch.Tensor]:
        gen.manual_seed(seed)
        return [torch.randn(m.shape, generator=gen, device=dev, dtype=torch.float32) for m in masters]

    @torch.no_grad()
    def set_live(zs: list[torch.Tensor] | None, scale: float) -> None:
        for w, m, z in zip(live, masters, zs or [None] * len(live)):
            if z is None:
                w.copy_(m)
            else:
                torch.add(m, z, alpha=scale, out=w)

    # ---- calibration: tr Ĥ from three fp32 forwards a direction --------------------------------------------------------
    trh, cal = None, {}
    if a.kappa is not None:
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        xc = prepare.TrainData(rec["tokenizer"], seed=a.data_seed + 1000, ctx=a.ctx).next(a.cal_batch)

        @torch.no_grad()
        def loss32() -> float:
            logits = net(xc[:, :-1])
            return float(F.cross_entropy(logits.float().view(-1, vocab), xc[:, 1:].reshape(-1)))

        set_live(None, 0.0)
        l0 = loss32()
        curv, slopes = [], []
        for j in range(a.cal_dirs):
            zs = draw(10**12 + a.seed * 1000 + j)
            set_live(zs, a.eps)
            lp = loss32()
            set_live(zs, -a.eps)
            lm = loss32()
            curv.append((lp + lm - 2 * l0) / a.eps**2)
            slopes.append((lp - lm) / (2 * a.eps))
            del zs
        set_live(None, 0.0)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        steps += [{"op": "forward", "tokens": cal_tokens, "ctx": a.ctx}] * (2 * a.cal_dirs + 1)
        trh = sum(curv) / len(curv)
        cal = {"l0": l0, "trH": trh, "curv": curv, "slope_sq_mean": sum(s * s for s in slopes) / len(slopes)}
        print(json.dumps({"calibration": cal}), flush=True)
        if trh <= 0:
            raise SystemExit("train: calibration found no positive curvature; use --lr")
    lr0 = a.lr if a.lr is not None else a.kappa / trh
    print(json.dumps({"lr": lr0}), flush=True)

    fwd = torch.compile(net)
    marks = [0.25, 0.5, 0.75]
    snaps: list[tuple[bytes, int]] = []
    coeffs: list[float] = []
    t_loop = time.perf_counter()
    deadline = T_MAIN + a.time_budget

    def progress(i: int) -> float:
        now = time.perf_counter()
        return max(i / n_steps, (now - t_loop) / max(deadline - t_loop, 1e-9))

    def lr_at(p: float) -> float:
        return lr0 * (1 - p) if a.schedule == "linear" else lr0

    stats = {"g2": 0.0, "gab": 0.0, "n": 0, "pred": 0.0}
    ema = None

    def report(i: int, extra: dict | None = None) -> None:
        n = max(stats["n"], 1)
        row = {"step": i, "loss_ema": ema, "g2_mean": stats["g2"] / n, "gAgB_mean": stats["gab"] / n,
               "pred_gain_nats": stats["pred"], "s": round(time.perf_counter() - T_MAIN, 1)}
        if stats["g2"] > 0:
            row["signal_fraction_half"] = stats["gab"] / stats["g2"]
        print(json.dumps(row | (extra or {})), flush=True)

    if a.method == "mezo":
        ghat: list[float] = []
        lrs: list[float] = []
        mom = [torch.zeros_like(m) for m in masters] if a.momentum else None
        h = a.batch // 2

        @torch.no_grad()
        def seq_losses(x: torch.Tensor) -> torch.Tensor:
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            ce = F.cross_entropy(logits.float().view(-1, vocab), x[:, 1:].reshape(-1), reduction="none")
            return ce.view(x.shape[0], -1).mean(1)

        def total_coeffs(upto: int) -> list[float]:
            if not a.momentum:
                return coeffs[:upto]
            out, acc = [0.0] * upto, 0.0
            for s in range(upto - 1, -1, -1):
                acc = lrs[s] + a.momentum * acc
                out[s] = prepare.quantize_coefficient(ghat[s] * acc, a.coeff)
            return out

        i = 0
        while i < n_steps:
            p = progress(i)
            if p >= 1:
                break
            lr = lr_at(p)
            x = data.next(a.batch)
            zs = draw(a.seed + i)
            with torch.no_grad():
                set_live(zs, a.eps)
                lp = seq_losses(x)
                set_live(zs, -a.eps)
                lm = seq_losses(x)
                dd = (lp - lm) / (2 * a.eps)
                st = torch.stack([dd.mean(), dd[:h].mean(), dd[h:].mean(), (lp + lm).mean() / 2]).tolist()
            g, ga, gb, lmean = st
            steps += [{"op": "forward", "tokens": tokens, "ctx": a.ctx}] * 2
            stats["g2"] += (ga * ga + gb * gb) / 2
            stats["gab"] += ga * gb
            stats["n"] += 1
            ema = lmean if ema is None else 0.995 * ema + 0.005 * lmean
            if a.momentum:
                ghat.append(g)
                lrs.append(lr)
                with torch.no_grad():
                    for mb, z, m in zip(mom, zs, masters):
                        mb.mul_(a.momentum).add_(z, alpha=g)
                        m.add_(mb, alpha=-lr)
                c = lr * g
            else:
                c = prepare.quantize_coefficient(lr * g, a.coeff)
                coeffs.append(c)
                with torch.no_grad():
                    for m, z in zip(masters, zs):
                        m.add_(z, alpha=-c)
            if trh is not None:
                stats["pred"] += lr * ga * gb - c * c * trh / 2
            del zs
            i += 1
            if i % 1000 == 0:
                report(i, {"lr": lr})
            while marks and progress(i) >= marks[0]:
                marks.pop(0)
                snaps.append((prepare.encode_seedlog(names, a.seed, total_coeffs(i), a.coeff), len(steps)))
        n_done = i
        final = total_coeffs(n_done)
    else:
        for w in live:
            w.requires_grad_(True)
        i = 0
        k = a.dirs_per_step
        extra = {"op": "extra", "flops": 2 * d * k}
        t = 0
        while t < n_steps:
            p = progress(t)
            if p >= 1:
                break
            lr = lr_at(p)
            x = data.next(a.batch)
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, vocab), x[:, 1:].reshape(-1))
            loss.backward()
            steps += [{"op": "train", "tokens": tokens, "ctx": a.ctx}, extra]
            with torch.no_grad():
                grads = [w.grad.float() for w in live]
                for w in live:
                    w.grad = None
                g2 = float(sum((gr * gr).sum() for gr in grads))
                projs = []
                for _ in range(k):
                    zs = draw(a.seed + i)
                    pr = float(sum(torch.dot(gr.view(-1), z.view(-1)) for gr, z in zip(grads, zs)))
                    c = prepare.quantize_coefficient(lr * pr, a.coeff)
                    for m, z in zip(masters, zs):
                        m.add_(z, alpha=-c)
                    coeffs.append(c)
                    projs.append(pr)
                    if trh is not None:
                        stats["pred"] += lr * pr * pr - c * c * trh / 2
                    i += 1
                del zs
                set_live(None, 0.0)
            lv = float(loss)
            ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            stats["g2"] += sum(q * q for q in projs) / k
            stats["n"] += 1
            t += 1
            if t % max(1, n_steps // 20) == 0:
                report(i, {"lr": lr, "grad_norm2": g2, "pass": t})
            while marks and progress(t) >= marks[0]:
                marks.pop(0)
                snaps.append((prepare.encode_seedlog(names, a.seed, list(coeffs), a.coeff), len(steps)))
        n_done = i
        final = coeffs
    report(n_done, {"done": True})
    blob = prepare.encode_seedlog(names, a.seed, final, a.coeff)
    if a.method == "mezo" and a.momentum:
        model = prepare.decode(theta, blob)
    else:
        model = dict(theta) | {n: m.to(theta[n].dtype).cpu() for n, m in zip(names, masters)}
    notes = {"method": a.method, "args": vars(a), "directions": n_done, "lr": lr0, "calibration": cal,
             "signal_fraction_half": stats["gab"] / stats["g2"] if stats["g2"] else None,
             "pred_gain_nats": stats["pred"], "loop_seconds": time.perf_counter() - t_loop}
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, steps, changed=names, notes=notes, snapshots=snaps)
    print(json.dumps({"submitted": len(blob), "directions": n_done, "s": round(time.perf_counter() - T_MAIN, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
