"""Island `seed-mezo`: zeroth-order training whose update is a seed log (`prepare.encode_seedlog`, CWSEED1), sized for
the training formalism v2's Fit regime. Every step is forward passes only: no step computes a gradient, so the state a
step passes between token blocks is a few loss sums, not a gradient tensor.

Methods:

- MeZO (default): step i draws z_i ~ N(0, 1) from the CUDA generator seeded `seed + i` over the named tensors S
  (`--names`), measures the loss at θ_t ± εz_i on one batch (two forwards), and appends c_i = lr · ĝ_i,
  ĝ_i = (L₊ − L₋)/2ε, to the log; the decoder adds −c_i z_i to S's fp32 masters (`prepare.seedlog_step`, which the
  training also applies). E is the header plus one `--coeff` value a step (16 bits for bf16), so `--max-bits` caps the
  steps, and the decoder replays every step over S: T·|S| fp32 FMAs (`prepare.rebuild_macs`), Fit's d.
- `--subspace K`: K fixed directions z_0 … z_{K−1} (seeds `seed + j`), visited in turn: step i measures ĝ along
  z_{i mod K} and adds lr · ĝ to that direction's coefficient, rounded stochastically to `--coeff` (a generator seeded
  from `--seed`), so the update rule reads only the K coefficients E holds (Δ_t = Δ). E is the K coefficients, so Δ and
  d (K·|S|) stay fixed while the steps go on.
- `--batch 2` is the block-local online update: each step reads one token block's ~2k tokens (two sequences at ctx
  1024), updates, and hands only the log on; with `--subspace K` its Δ stays K coefficients however many blocks run.

The learning rate: `--lr` absolute, or `--lr-auto F`: lr = F · G / (E[ĝ²] · tr H_S), with G = E[ĝ_A ĝ_B] (the product of
the batch halves' estimates, an unbiased (∇L·z)²) and E[ĝ²] as EMAs of the run's own steps (`--ema`), and tr H_S the mean
three-point curvature along `--curv-dirs` Gaussian directions over S (fp32 forwards on `--curv-batch` sequences, at
`--curv-rel` × S's rms). `--warm W` steps first measure G and E[ĝ²] at θ along seeds outside the log (two forwards each,
no update, no bits). With `--warm-batch WB` ≠ `--batch`, the warm steps read WB sequences and the lr is fixed for the
run at the warm's G over E[ĝ²] projected to the run's batch, G + (E[ĝ²] − G) · WB / batch (a batch of 2's EMAs are
noise).

Checkpoints: the run plans its steps (`te_core.hooks.plan`), so its scored checkpoints (the first E encoded in each 5% of
the planned steps) spread over it, and encodes E at the first step of each 5%, or, where one of `--marks-bits` falls in
that 5%, at the step the log reaches that size instead, so a scored checkpoint sits at each such update size.

    python launch.py START --island seed-mezo --note TEXT -- --max-bits B [--marks-bits B1,B2,...] [--subspace K]
        [--names blocks|all|norms|mlp|attn|top:K|re:REGEX] [--lr LR | --lr-auto F [--warm 300] [--warm-batch WB]]
        [--batch 256]
        [--micro 32] [--ctx 1024] [--eps 1e-3 | --eps-rel R] [--coeff bf16] [--steps N] [--max-flops F]
        [--budget-fraction 0.98] [--time-budget S] [--label L --description D]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare

try:
    from te_core import hooks
except ImportError:
    hooks = None

T_MAIN = time.perf_counter()
CHECKPOINTS = 20
#: te_core.gates' prices: a bf16 GEMM MAC (HopperBF16WgmmaDot16_v2 / 16) and an fp32 FMA (F32Fma_v4), in gates
GEMM_BF16_GATES = (7872 + 15049 + 761) / 16
F32_FMA_GATES = 4485 + 7863 + 289
WIDTH = {"bf16": 2, "f16": 2, "f32": 4}


def round_stochastic(v: float, coeff: str, u: float) -> float:
    """v rounded to one of its two `coeff` neighbours, the far one with probability its share of the gap (unbiased)."""
    dt = prepare.DTYPES[coeff]
    q = torch.tensor([v], dtype=torch.float32).to(dt)
    qv = float(q.float())
    if qv == v or dt == torch.float32:
        return qv
    if qv == 0.0:
        other = float(torch.tensor([1 if v > 0 else -32767], dtype=torch.int16).view(dt).float())
    else:
        bits = q.view(torch.int16)
        other = float((bits + 1 if abs(v) > abs(qv) else bits - 1).view(dt).float())
    return other if u < (v - qv) / (other - qv) else qv


def select(params: dict, rec: dict, spec: str) -> list[str]:
    L = rec["config"]["n_layer"]
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    layer, stage = {}, {}
    for n in prepare.positions(rec):
        layer[n["param"]] = min(layer.get(n["param"], 10**9), n["layer"])
        stage[n["param"]] = n["stage"]
    names = sorted(params)
    blocks = [n for n in names if n != embed]
    if spec == "all":
        return names
    if spec == "blocks":
        return blocks
    if spec in ("gains", "norms"):
        return [n for n in names if params[n].ndim == 1]
    if spec == "mlp":
        return [n for n in blocks if stage[n] in (4, 5) and layer[n] <= L]
    if spec == "attn":
        return [n for n in blocks if stage[n] in (1, 2) and layer[n] <= L]
    if spec.startswith("top:"):
        k = int(spec[4:])
        return [n for n in blocks if layer[n] >= L + 1 - k]
    if spec.startswith("re:"):
        return [n for n in names if re.search(spec[3:], n)]
    raise SystemExit(f"train: unknown --names {spec}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="blocks")
    ap.add_argument("--subspace", type=int, default=0, help="K fixed directions visited in turn (0: a new one a step)")
    ap.add_argument("--max-bits", type=int, required=True, help="the largest 8|E| the run writes (MeZO: caps its steps)")
    ap.add_argument("--marks-bits", default="", help="update sizes in bits, each a scored checkpoint")
    ap.add_argument("--steps", type=int, default=None, help="at most this many steps")
    ap.add_argument("--max-flops", type=float, default=None, help="at most this many training FLOPs")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-auto", type=float, default=None, help="lr = this × G / (E[ĝ²] tr H_S), measured as it goes")
    ap.add_argument("--warm", type=int, default=300, help="--lr-auto: measurement steps at θ first (no update, no bits)")
    ap.add_argument("--warm-batch", type=int, default=None,
                    help="--lr-auto: the warm steps' batch; if not --batch, the lr is fixed at the warm's, projected")
    ap.add_argument("--ema", type=float, default=0.999, help="the decay of the lr rule's EMAs, a step")
    ap.add_argument("--lr-cap", type=float, default=2.0, help="lr at most this × the lr the warm steps give")
    ap.add_argument("--max-move", type=float, default=0.02,
                    help="a step moves S by at most this × its norm (|lr · ĝ| ≤ this × S's rms)")
    ap.add_argument("--schedule", default="const", choices=("const", "linear"))
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--eps-rel", type=float, default=None, help="ε = this × S's rms (overrides --eps)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--micro", type=int, default=32, help="sequences a forward")
    ap.add_argument("--ctx", type=int, default=prepare.EVAL_CTX)
    ap.add_argument("--coeff", default="bf16", choices=tuple(WIDTH))
    ap.add_argument("--curv-dirs", type=int, default=4)
    ap.add_argument("--curv-batch", type=int, default=16)
    ap.add_argument("--curv-rel", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--precision", default="bf16", choices=("bf16", "tf32", "fp32"), help="the steps' forwards")
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--time-budget", type=float, default=None, help="seconds from start to the loop's end")
    ap.add_argument("--label", default="")
    ap.add_argument("--description", default="")
    a = ap.parse_args()
    if (a.lr is None) == (a.lr_auto is None):
        raise SystemExit("train: give one of --lr and --lr-auto")
    a.warm_batch = a.warm_batch or a.batch
    if min(a.batch, a.warm_batch) < 2:
        raise SystemExit("train: a step splits its batch in halves; --batch and --warm-batch 2 at least")
    if len(a.description) > 200:
        raise SystemExit("train: --description is at most 200 characters")
    dev = prepare.DEVICE
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.requires_grad_(False)
    vocab = rec["config"]["vocab"]
    names = select(params, rec, a.names)
    if not names:
        raise SystemExit(f"train: --names {a.names} selects no tensor")
    live = [params[n] for n in names]
    masters = [params[n].detach().float().clone() for n in names]
    d = sum(m.numel() for m in masters)
    rms = math.sqrt(sum(float((m * m).sum()) for m in masters) / d)
    trainable = {"full": names}
    budget = prepare.budget(a.start)
    fwd_tok = prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": 1, "ctx": a.ctx}])["flops"]
    fwd_macs = prepare.STORE.get(a.start, {}).get("shape", {}).get("forward_macs_per_token") or fwd_tok / 2
    tokens = a.batch * a.ctx
    per_step = 2 * tokens * fwd_tok
    width = WIDTH[a.coeff]
    header = len(prepare.encode_seedlog(names, a.seed, [], a.coeff))
    n_max = max(0, (a.max_bits // 8 - header) // width)
    header = len(prepare.encode_seedlog(names, a.seed, [0.0] * n_max, a.coeff)) - width * n_max
    max_entries = (a.max_bits // 8 - header) // width
    if max_entries < 1:
        raise SystemExit(f"train: --max-bits {a.max_bits} holds no coefficient (header {header} B)")
    if a.subspace:
        if a.subspace > max_entries:
            raise SystemExit(f"train: --subspace {a.subspace} is over --max-bits' {max_entries} coefficients")
        entries = a.subspace
    curv_flops = (1 + 2 * a.curv_dirs) * a.curv_batch * a.ctx * fwd_tok
    warm = a.warm if a.lr_auto is not None else 0
    warm_flops = warm * 2 * a.warm_batch * a.ctx * fwd_tok
    flop_cap = a.budget_fraction * budget["train_flops"]
    if a.max_flops:
        flop_cap = min(flop_cap, a.max_flops)
    n_steps = int((flop_cap - curv_flops - warm_flops) // per_step)
    if not a.subspace:
        n_steps = min(n_steps, max_entries)
    if a.steps:
        n_steps = min(n_steps, a.steps)
    if n_steps < 1:
        raise SystemExit(f"train: no step fits ({flop_cap:.3g} FLOPs, {per_step:.3g} a step)")
    if not a.subspace:
        entries = n_steps
    time_budget = a.time_budget or 0.92 * budget["gpu_seconds"]
    deadline = T_MAIN + time_budget
    c_macs = 4096 * fwd_macs
    c_gates = c_macs * GEMM_BF16_GATES

    def decoder(n_entries: int) -> dict:
        macs = n_entries * d
        gates = macs * F32_FMA_GATES
        return {"macs": macs, "gates": gates, "d_over_c": gates / c_gates,
                "rho_fit": c_gates / (c_gates - gates) if gates < c_gates else None}

    info = {"names": a.names, "tensors": len(names), "d": d, "rms": rms, "steps": n_steps, "warm": warm,
            "entries": entries, "header_bytes": header, "max_entries": max_entries, "flops_per_step": per_step,
            "tier": budget["tier"], "flop_cap": flop_cap, "time_budget": time_budget, "c_macs": c_macs,
            "decoder_at_end": decoder(entries)}
    print(json.dumps(info), flush=True)

    marks = sorted({(b // 8 - header) // width for b in map(int, filter(None, a.marks_bits.split(",")))})
    if a.subspace and marks:
        raise SystemExit("train: --subspace's E has one size, K coefficients; drop --marks-bits")
    marks = [m for m in marks if 1 <= m <= n_steps]
    n_curv_steps = 1
    planned = n_curv_steps + warm + n_steps
    sess = hooks.session() if hooks is not None else None
    if hooks is not None:
        hooks.plan(planned)

    gen = torch.Generator(device=dev)
    fwd_tokens = 0

    @torch.no_grad()
    def perturb(seed: int, scale: float) -> None:
        """live = masters + scale · z_seed, by the decoder's op (the suffix log [−scale] applied to the masters)."""
        for w, m in zip(live, masters):
            w.copy_(m)
        if scale:
            prepare.seedlog_step(live, seed, -scale, gen)

    cal = prepare.TrainData(rec["tokenizer"], seed=a.data_seed + 3000, ctx=a.ctx)
    xc = cal.next(a.curv_batch)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    @torch.no_grad()
    def loss32(x: torch.Tensor) -> float:
        tot = 0.0
        for xm in x.split(a.micro):
            logits = net(xm[:, :-1])
            tot += float(F.cross_entropy(logits.float().view(-1, vocab), xm[:, 1:].reshape(-1))) * xm.shape[0]
        return tot / x.shape[0]

    t0 = time.perf_counter()
    eps_c = prepare.quantize_coefficient(a.curv_rel * rms, a.coeff)
    perturb(0, 0.0)
    l0 = loss32(xc)
    curv = []
    for j in range(a.curv_dirs):
        ls = []
        for sign in (1.0, -1.0):
            perturb(10**13 + j, sign * eps_c)
            ls.append(loss32(xc))
        curv.append((ls[0] + ls[1] - 2 * l0) / eps_c**2)
    perturb(0, 0.0)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    fwd_tokens += (1 + 2 * a.curv_dirs) * a.curv_batch * a.ctx
    trh = sum(curv) / len(curv)
    print(json.dumps({"curvature": curv, "trH": trh, "l0": l0, "eps": eps_c, "s": round(time.perf_counter() - t0, 1)}),
          flush=True)
    if a.lr_auto is not None and trh <= 0:
        raise SystemExit(f"train: S's measured curvature {trh} is not positive; --lr-auto needs tr H > 0")

    if a.eps_rel is not None:
        a.eps = a.eps_rel * rms
    eps = prepare.quantize_coefficient(a.eps, a.coeff)
    torch.backends.cuda.matmul.allow_tf32 = a.precision != "fp32"
    fwd = torch.compile(net)
    micro = a.micro
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed, ctx=a.ctx)

    @torch.no_grad()
    def seq_losses(x: torch.Tensor) -> torch.Tensor:
        out = []
        for xm in x.split(micro):
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=a.precision == "bf16"):
                logits = fwd(xm[:, :-1])
            ce = F.cross_entropy(logits.float().view(-1, vocab), xm[:, 1:].reshape(-1), reduction="none")
            out.append(ce.view(xm.shape[0], -1).mean(1))
        return torch.cat(out)

    @torch.no_grad()
    def estimate(seed: int, x: torch.Tensor) -> tuple[float, float, float, float]:
        """ĝ over the batch, over each half, and the mean of L₊ and L₋, along z_seed."""
        nonlocal fwd_tokens
        perturb(seed, eps)
        lp = seq_losses(x)
        perturb(seed, -eps)
        lm = seq_losses(x)
        fwd_tokens += 2 * x.shape[0] * a.ctx
        dd = (lp - lm) / (2 * eps)
        hx = x.shape[0] // 2
        return tuple(torch.stack([dd.mean(), dd[:hx].mean(), dd[hx:].mean(), (lp + lm).mean() / 2]).tolist())

    stats = {"w": 0.0, "G": 0.0, "E": 0.0, "n": 0, "gab": 0.0, "g2": 0.0, "pred": 0.0}

    def observe(g: float, ga: float, gb: float) -> None:
        for k, v in (("w", 1.0), ("G", ga * gb), ("E", g * g)):
            stats[k] = a.ema * stats[k] + v
        stats["n"] += 1
        stats["gab"] += ga * gb
        stats["g2"] += (ga * ga + gb * gb) / 2

    def lr_rule() -> float:
        return a.lr_auto * max(stats["G"] / stats["w"], 0.0) / (max(stats["E"] / stats["w"], 1e-30) * trh)

    ema_loss = None
    for i in range(warm):
        x = data.next(a.warm_batch)
        g, ga, gb, lmean = estimate(a.seed + 10**12 + i, x)
        observe(g, ga, gb)
        ema_loss = lmean if ema_loss is None else 0.99 * ema_loss + 0.01 * lmean
    lr_warm = lr_rule() if a.lr_auto is not None else a.lr
    fixed_lr = a.lr_auto is None or a.warm_batch != a.batch
    if a.lr_auto is not None:
        g_w, e_w = stats["G"] / stats["w"], stats["E"] / stats["w"]
        e_run = g_w + (e_w - g_w) * a.warm_batch / a.batch
        if a.warm_batch != a.batch:
            lr_warm = a.lr_auto * max(g_w, 0.0) / (max(e_run, 1e-30) * trh)
        print(json.dumps({"warm": warm, "warm_batch": a.warm_batch, "G": g_w, "E": e_w, "E_at_batch": e_run,
                          "lr": lr_warm, "lr_fixed": fixed_lr, "loss": ema_loss}), flush=True)
        if not lr_warm > 0:
            raise SystemExit(f"train: the warm steps measure G {stats['G'] / stats['w']:.3g} <= 0 along S; no lr")

    coeffs: list[float] = []
    alpha_q = [0.0] * (a.subspace or 0)
    rounding = random.Random(a.seed)
    cps: list[dict] = []
    buckets: set[int] = set()
    j0 = None

    def encoded() -> bytes:
        return prepare.encode_seedlog(names, a.seed, alpha_q if a.subspace else coeffs, a.coeff)

    def bucket_of(j: int) -> int:
        return math.ceil((j + 1) * CHECKPOINTS / planned)

    def checkpoint(i: int, j: int | None) -> None:
        blob = encoded()
        n = len(alpha_q) if a.subspace else len(coeffs)
        cps.append({"step": i, "harness_step": j, "e_bytes": len(blob), "delta_bits": 8 * len(blob),
                    "decoder": decoder(n), "loss_ema": ema_loss})

    def report(i: int, lr: float, extra: dict | None = None) -> None:
        n = max(stats["n"], 1)
        row = {"step": i, "lr": lr, "loss_ema": ema_loss, "G_ema": stats["G"] / max(stats["w"], 1e-30),
               "E_ema": stats["E"] / max(stats["w"], 1e-30), "gAgB_mean": stats["gab"] / n, "g2_mean": stats["g2"] / n,
               "pred_gain_nats": stats["pred"], "clipped": clipped, "s": round(time.perf_counter() - T_MAIN, 1)}
        print(json.dumps(row | (extra or {})), flush=True)

    t_loop = time.perf_counter()
    lr = lr_warm
    n_done = 0
    max_c = a.max_move * rms
    clipped = 0
    for i in range(n_steps):
        if time.perf_counter() > deadline:
            break
        x = data.next(a.batch)
        if sess is not None:
            j = sess.current_step()
            if j0 is None:
                j0 = j - i
                by_bucket: dict[int, list[int]] = {}
                for m in marks:
                    by_bucket.setdefault(bucket_of(j0 + m), []).append(m)
                shared = [ms for ms in by_bucket.values() if len(ms) > 1]
                if shared:
                    print(json.dumps({"marks_sharing_a_5pct": shared, "scored": [ms[0] for ms in shared]}), flush=True)
            b = bucket_of(j)
            if b not in buckets and i > 0:
                ahead = [m for m in marks if m >= i and bucket_of(j0 + m) == b]
                if not ahead or ahead[0] == i:
                    buckets.add(b)
                    checkpoint(i, j)
        elif i in marks or (i > 0 and i % max(1, n_steps // CHECKPOINTS) == 0):
            checkpoint(i, None)
        p = i / n_steps
        if not fixed_lr and stats["n"] > warm:
            lr = min(a.lr_cap * lr_warm, lr_rule())
        lr_i = lr * (1 - p) if a.schedule == "linear" else lr
        k = i % a.subspace if a.subspace else i
        g, ga, gb, lmean = estimate(a.seed + k, x)
        observe(g, ga, gb)
        ema_loss = lmean if ema_loss is None else 0.995 * ema_loss + 0.005 * lmean
        move = max(-max_c, min(max_c, lr_i * g))
        clipped += move != lr_i * g
        if a.subspace:
            new = round_stochastic(alpha_q[k] + move, a.coeff, rounding.random())
            c = new - alpha_q[k]
            alpha_q[k] = new
        else:
            c = prepare.quantize_coefficient(move, a.coeff)
            coeffs.append(c)
        if c:
            prepare.seedlog_step(masters, a.seed + k, c, gen)
        stats["pred"] += lr_i * ga * gb - c * c * trh / 2
        n_done = i + 1
        if n_done % 1000 == 0:
            report(n_done, lr_i)
    perturb(0, 0.0)
    report(n_done, lr, {"done": True, "loop_seconds": round(time.perf_counter() - t_loop, 1)})

    blob = encoded()
    model = dict(theta) | {n: m.to(theta[n].dtype).cpu() for n, m in zip(names, masters)}
    n_entries = len(alpha_q) if a.subspace else len(coeffs)
    grad_bits = 4 * 32
    ema_bits = 3 * 32
    notes = {
        "method": "mezo-subspace" if a.subspace else "mezo", "args": vars(a), "steps": n_done, "warm": warm,
        "entries": n_entries, "names": names, "d": d, "lr_warm": lr_warm, "lr_end": lr, "lr_fixed": fixed_lr,
        "eps": eps, "trH": trh,
        "curvature": curv, "l0_cal": l0, "pred_gain_nats": stats["pred"], "clipped_steps": clipped,
        "signal_fraction_half": stats["gab"] / stats["g2"] if stats["g2"] else None,
        "mechanism": {"label": a.label, "description": a.description},
        "delta_bits": 8 * len(blob), "delta_t_bits": 8 * len(blob),
        "delta_grad": {"trainable": d, "grad_bits": grad_bits, "adam_bits": 0, "muon_bits": 0, "ema_bits": ema_bits,
                       "latent_bits": 0, "bits": grad_bits + ema_bits, "bytes": (grad_bits + ema_bits) // 8,
                       "what": "a step's two batch halves' L+ and L- partial sums (fp32) and the lr rule's three fp32 "
                               "EMAs; no gradient tensor is formed"},
        "decoder": decoder(n_entries) | {"c_macs": c_macs, "c_gates": c_gates,
                                          "rule": "prepare.rebuild_macs (entries x |S| fp32 FMAs) at F32Fma_v4's gates; "
                                                  "c = 4096 x the start's forward MACs a token at bf16 GEMM gates"},
        "checkpoints": cps, "loop_seconds": time.perf_counter() - t_loop,
    }
    steps = [{"op": "forward", "tokens": fwd_tokens, "ctx": a.ctx}]
    prepare.submit(Path(a.out), a.start, blob, model, trainable, steps, changed=names, notes=notes)
    print(json.dumps({"submitted": len(blob), "steps": n_done, "entries": n_entries, "checkpoints": len(cps),
                      "s": round(time.perf_counter() - T_MAIN, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
