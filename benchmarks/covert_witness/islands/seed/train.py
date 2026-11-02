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


def groups_of(rec: dict) -> dict[str, list[str]]:
    """Tensor groups for the diagnostics: each block's (layer, stage) nodes, the final norm and the embedding."""
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    out: dict[str, list[str]] = {}
    for n in prepare.positions(rec):
        key = "embed" if n["param"] == embed else f"L{n['layer']}.s{n['stage']}"
        if n["param"] not in out.setdefault(key, []):
            out[key].append(n["param"])
    return out


def diagnose(net, rec: dict, params: dict, a, vocab: int) -> tuple[dict, list[dict]]:
    """Per tensor group: G = |g|² of the population gradient (mean product of independent batches' gradients), N = |g_b|² at
    `--diag-batch` sequences, and T = tr H (fp32 three-point curvature along `--diag-dirs` Gaussian directions, each group
    perturbed alone at 5% of its rms). The backward passes are priced as full-model train passes and logged as `extra`."""
    dev = prepare.DEVICE
    groups = groups_of(rec)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed + 2000, ctx=a.ctx)
    for p in params.values():
        p.requires_grad_(True)
    grads = []
    for _ in range(a.diag_batches):
        x = data.next(a.diag_batch)
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
            logits = net(x[:, :-1])
        F.cross_entropy(logits.float().view(-1, vocab), x[:, 1:].reshape(-1)).backward()
        grads.append({n: p.grad.detach().clone() for n, p in params.items()})
        for p in params.values():
            p.grad = None
    for p in params.values():
        p.requires_grad_(False)
    nb = len(grads)
    G, N = {}, {}
    with torch.no_grad():
        for n in params:
            prods = [float((grads[i][n] * grads[j][n]).sum()) for i in range(nb) for j in range(i + 1, nb)]
            G[n] = sum(prods) / len(prods)
            N[n] = sum(float((g[n] * g[n]).sum()) for g in grads) / nb
    del grads
    full_train = prepare.train_flops(rec, {"full": sorted(params)}, [{"op": "train", "tokens": a.diag_batch * a.ctx,
                                                                       "ctx": a.ctx}])["flops"]
    extra = [{"op": "extra", "flops": full_train}] * nb
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    xc = prepare.TrainData(rec["tokenizer"], seed=a.data_seed + 3000, ctx=a.ctx).next(a.cal_batch)

    @torch.no_grad()
    def loss32() -> float:
        logits = net(xc[:, :-1])
        return float(F.cross_entropy(logits.float().view(-1, vocab), xc[:, 1:].reshape(-1)))

    l0 = loss32()
    gen = torch.Generator(device=dev)
    table = {}
    n_fwd = 1
    for gi, (key, ns) in enumerate(sorted(groups.items())):
        base = {n: params[n].detach().clone() for n in ns}
        numel = sum(base[n].numel() for n in ns)
        rms = math.sqrt(sum(float((base[n].float() ** 2).sum()) for n in ns) / numel)
        eps = 0.05 * rms
        curv = []
        for j in range(a.diag_dirs):
            gen.manual_seed(10**13 + 1000 * gi + j)
            zs = {n: torch.randn(base[n].shape, generator=gen, device=dev) for n in ns}
            with torch.no_grad():
                for s in (1.0, -1.0):
                    for n in ns:
                        torch.add(base[n], zs[n], alpha=s * eps, out=params[n])
                    curv.append(loss32())
                for n in ns:
                    params[n].copy_(base[n])
            n_fwd += 2
        curv = [(curv[2 * j] + curv[2 * j + 1] - 2 * l0) / eps**2 for j in range(a.diag_dirs)]
        table[key] = {"names": ns, "numel": numel, "G": sum(G[n] for n in ns), "N": sum(N[n] for n in ns),
                      "T": sum(curv) / len(curv), "eps": eps}
    torch.backends.cuda.matmul.allow_tf32 = tf32
    fwd = [{"op": "forward", "tokens": a.cal_batch * a.ctx, "ctx": a.ctx}] * n_fwd
    return {"l0": l0, "diag_batch": a.diag_batch, "groups": table}, extra + fwd


def predict(diag: dict, keys: list[str], batch: int) -> dict:
    """MeZO's (or a projected gradient's) expected gain a direction on batch `batch` at its best lr, from the group sums:
    G² / (2 (G + V/b) T), V = (N − G) · diag_batch the per-sequence noise; the best lr is G / ((G + V/b) T)."""
    t = diag["groups"]
    G = sum(t[k]["G"] for k in keys)
    V = sum((t[k]["N"] - t[k]["G"]) * diag["diag_batch"] for k in keys)
    T = sum(t[k]["T"] for k in keys)
    gb = G + V / batch
    ok = G > 0 and T > 0 and gb > 0
    return {"G": G, "V": V, "T": T, "signal_fraction": G / gb if gb > 0 else None,
            "lr": G / (gb * T) if ok else None, "gain_per_dir": G * G / (2 * gb * T) if ok else None,
            "numel": sum(t[k]["numel"] for k in keys)}


def auto_groups(diag: dict, batch: int, min_numel: int) -> list[str]:
    """Greedy: add the group that most raises the predicted gain a direction, while it rises or S is below `min_numel`."""
    keys = [k for k, v in diag["groups"].items() if v["T"] > 0]
    chosen: list[str] = []
    current = 0.0
    while len(chosen) < len(keys):
        g, k = max((predict(diag, chosen + [k], batch)["gain_per_dir"] or 0.0, k) for k in keys if k not in chosen)
        if g <= current and predict(diag, chosen, batch)["numel"] >= min_numel:
            break
        chosen.append(k)
        current = g
    return chosen


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
    ap.add_argument("--diag", action="store_true", help="per-group |g|², noise and tr H first (implied by auto and --lr-auto)")
    ap.add_argument("--diag-batch", type=int, default=16)
    ap.add_argument("--diag-batches", type=int, default=4)
    ap.add_argument("--diag-dirs", type=int, default=3)
    ap.add_argument("--lr-auto", type=float, default=None, help="lr = this × the diagnostics' best lr for the subset")
    ap.add_argument("--auto-min-numel", type=int, default=0)
    ap.add_argument("--micro", type=int, default=16, help="sequences a forward (the batch is split into these)")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "tf32", "fp32"), help="mezo's forwards")
    ap.add_argument("--eps-rel", type=float, default=None, help="ε = this × the subset's rms (overrides --eps)")
    a = ap.parse_args()
    if sum(x is not None for x in (a.lr, a.kappa, a.lr_auto)) != 1:
        raise SystemExit("train: give one of --lr, --kappa and --lr-auto")
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = prepare.DEVICE
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.requires_grad_(False)
    vocab = rec["config"]["vocab"]
    diag, diag_steps = None, []
    if a.diag or a.names == "auto" or a.lr_auto is not None:
        t0 = time.perf_counter()
        diag, diag_steps = diagnose(net, rec, params, a, vocab)
        canon = {"blocks": [k for k in diag["groups"] if k != "embed"], "all": list(diag["groups"]),
                 "gains": [k for k, v in diag["groups"].items() if k.endswith((".s0", ".s3")) or k.startswith(
                     f"L{rec['config']['n_layer'] + 1}.")]}
        for k in sorted(diag["groups"]):
            v = diag["groups"][k]
            print(json.dumps({"group": k, "numel": v["numel"], "G": v["G"], "N": v["N"], "T": v["T"],
                              "pred": predict(diag, [k], a.batch)}), flush=True)
        for name, keys in canon.items():
            print(json.dumps({"subset": name, "pred": predict(diag, keys, a.batch)}), flush=True)
        print(json.dumps({"diag_seconds": round(time.perf_counter() - t0, 1), "l0": diag["l0"]}), flush=True)
    if a.names == "auto":
        keys = auto_groups(diag, a.batch, a.auto_min_numel)
        names = sorted(n for k in keys for n in diag["groups"][k]["names"])
        print(json.dumps({"auto": keys, "pred": predict(diag, keys, a.batch)}), flush=True)
    else:
        names = select(params, rec, a.names)
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
    diag_flops = prepare.train_flops(rec, trainable, diag_steps)["flops"] if diag_steps else 0
    budget = a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - cal_flops - diag_flops
    if a.method == "mezo":
        n_steps = min(int(budget // per_step), max_dirs)
    else:
        extra = 2 * d * a.dirs_per_step
        n_steps = min(int(budget // (per_step + extra)), max_dirs // a.dirs_per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed, ctx=a.ctx)
    gen = torch.Generator(device=dev)
    steps: list[dict] = list(diag_steps)
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
    sub_pred = None
    if diag is not None:
        keys = [k for k, v in diag["groups"].items() if set(v["names"]) <= set(names)]
        sub_pred = predict(diag, keys, a.batch)
        if trh is None:
            trh = sub_pred["T"]
    if a.lr_auto is not None:
        if not sub_pred["lr"]:
            raise SystemExit(f"train: the diagnostics give no positive lr for this subset: {sub_pred}")
        lr0 = a.lr_auto * sub_pred["lr"]
    else:
        lr0 = a.lr if a.lr is not None else a.kappa / trh
    if a.eps_rel is not None:
        a.eps = a.eps_rel * math.sqrt(sum(float((m * m).sum()) for m in masters) / d)
    print(json.dumps({"lr": lr0, "eps": a.eps, "subset_pred": sub_pred}), flush=True)

    torch.backends.cuda.matmul.allow_tf32 = a.precision != "fp32"
    fwd = torch.compile(net)
    micro = min(a.micro, a.batch)
    marks = [0.25, 0.5, 0.75]
    snaps: list[tuple[bytes, int]] = []
    coeffs: list[float] = []
    t_loop = time.perf_counter()
    deadline = T_MAIN + a.time_budget

    def progress(i: int) -> float:
        now = time.perf_counter()
        return max(i / n_steps, (now - t_loop) / max(deadline - t_loop, 1e-9))

    stats = {"g2": 0.0, "gab": 0.0, "gf2": 0.0, "n": 0, "pred": 0.0}

    def lr_at(p: float) -> float:
        base = lr0
        if a.lr_auto is not None and a.method == "mezo":
            n = stats["n"]
            if n < 100:
                base = 0.5 * lr0
            else:
                base = a.lr_auto * sub_pred["G"] / (max(stats["gf2"] / n, sub_pred["G"]) * sub_pred["T"])
        return base * (1 - p) if a.schedule == "linear" else base
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
            out = []
            for xm in x.split(micro):
                with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=a.precision == "bf16"):
                    logits = fwd(xm[:, :-1])
                ce = F.cross_entropy(logits.float().view(-1, vocab), xm[:, 1:].reshape(-1), reduction="none")
                out.append(ce.view(xm.shape[0], -1).mean(1))
            return torch.cat(out)

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
            stats["gf2"] += g * g
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
            lv = 0.0
            for xm in x.split(micro):
                with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                    logits = fwd(xm[:, :-1])
                loss = F.cross_entropy(logits.float().view(-1, vocab), xm[:, 1:].reshape(-1)) * (xm.shape[0] / a.batch)
                loss.backward()
                lv += float(loss)
                steps.append({"op": "train", "tokens": xm.shape[0] * a.ctx, "ctx": a.ctx})
            steps.append(extra)
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
             "subset_pred": sub_pred, "names": names,
             "diag": {k: {x: v[x] for x in ("numel", "G", "N", "T")} for k, v in diag["groups"].items()} if diag else None,
             "signal_fraction_half": stats["gab"] / stats["g2"] if stats["g2"] else None,
             "pred_gain_nats": stats["pred"], "loop_seconds": time.perf_counter() - t_loop}
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, steps, changed=names, notes=notes, snapshots=snaps)
    print(json.dumps({"submitted": len(blob), "directions": n_done, "s": round(time.perf_counter() - T_MAIN, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
