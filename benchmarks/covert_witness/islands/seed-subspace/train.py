"""Island `seed-subspace` (class `seed`: E a seed log, at most 100,000 bytes): updates fully described by a seed and a few
coefficients, and the loss they buy per training FLOP.

The seed-log decoder replays Δ = −Σ_i c_i z_i over the named tensors S, z_i ~ N(0, 1) drawn in `names` order from a CUDA
generator seeded `seed + i`. E holds one coefficient per direction, not per step, so a fixed set of d directions can be
trained for as many steps as the budget allows. Modes:

- `subspace` (intrinsic dimension): θ_S = θ0_S + Σ_i α_i z_i over d fixed directions, Z cached on the GPU (bf16, d x |S|);
  each step backprops the loss to θ_S, projects the gradient onto the directions (g_α = Z g) and takes an Adam step on α.
  Two GEMVs a step with Z are logged as `extra` FLOPs. E is c = −α. `--redraw N` splits each arm into N equal blocks, each
  over d fresh directions (block b's are z_{bd}..z_{bd+d−1}) from where the blocks before it left θ, so an arm explores N·d
  directions in the memory of d; each redraw's d·|S| draws are logged as `extra`. `--exact` caches Z in fp32 and builds
  each step's weights with the decoder's own sequential adds (θ_S, then `add_(z_i, alpha=−c_i)` in order), so every step
  reads decode(θ, E_t) bit for bit and declares E_t (`prepare.declare_step`) for the harness to credit; its d adds a step
  cost d·|S| MACs, so keep d·|S| to a few 1e9.
- `fit` (a low-dimensional reparameterization solved exactly): train S itself with AdamW, then solve the minimum-norm c with
  −Zᵀc = Δ over d ≥ |S| directions (S small: the norm gains). The Gram matrix's FLOPs are logged as `extra`.
- `mezo` (zeroth-order, ES with shared seeds): per coefficient two forwards at θ ± εz on one batch, c = lr·(L₊ − L₋)/2ε;
  `--pop K` draws K seeds a step at the same θ (antithetic ES) and applies them all. The perturbations go through
  `prepare.seedlog_step` from the masters, so each step declares the log so far plus a ∓ε suffix per forward.

`--probe` (on by default on a GPU) first backprops one micro-batch with every tensor trained and records each tensor's mean
squared gradient (a d-direction subspace over S captures about d/|S| of ‖g_S‖², so the dense tensors are where directions
buy the most); its pass is logged as `extra`.

The submitted model is `prepare.decode(θ, E)` itself, so it is the decoder's bit for bit.

    python benchmarks/covert_witness/launch.py seed s50 --note "MECHANISM ..." -- --mode fit --perturb norms --lrs 3e-4,1e-3
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

KLASS = "seed"
EMBED = {"lf": "embed", "hf-llama": "model.embed_tokens.weight"}


def leaf(name: str) -> str:
    parts = name.removesuffix(".weight").split(".")
    if parts[0] == "blocks":
        return ".".join(parts[2:])
    if parts[:2] == ["model", "layers"]:
        return ".".join(parts[3:])
    return parts[-1]


def block_of(name: str) -> int | None:
    parts = name.split(".")
    if parts[0] == "blocks":
        return int(parts[1])
    if parts[:2] == ["model", "layers"]:
        return int(parts[2])
    return None


def select(rec: dict, params: dict, spec: str, layers: str) -> list[str]:
    """S: `all`, `blocks` (every block tensor and the final norm), `norms` (every gain), `mats` (every block matrix), or
    leaf names (`wo,w_down`); `--layers a-b` keeps blocks a..b (and the final norm when S names it)."""
    embed = EMBED[rec["config"]["family"]]
    if spec == "all":
        names = list(params)
    elif spec == "blocks":
        names = [n for n in params if n != embed]
    elif spec == "norms":
        names = [n for n, p in params.items() if p.ndim == 1]
    elif spec == "mats":
        names = [n for n, p in params.items() if p.ndim == 2 and n != embed]
    else:
        want = set(spec.split(","))
        names = [n for n in params if leaf(n) in want]
    if layers:
        lo, hi = (int(x) for x in layers.split("-"))
        names = [n for n in names if block_of(n) is None or lo <= block_of(n) <= hi]
    if not names:
        raise SystemExit(f"seed-subspace: --perturb {spec} --layers {layers} names no tensor")
    return sorted(names)


@torch.no_grad()
def fill_directions(Z: torch.Tensor, gen: torch.Generator, seed: int, shapes: list[torch.Size], first: int = 0) -> None:
    """Row j of Z is z_{first + j}, drawn exactly as `prepare.seedlog_step` draws it (cast to Z's dtype)."""
    for j in range(Z.shape[0]):
        gen.manual_seed(seed + first + j)
        at = 0
        for s in shapes:
            n = math.prod(s)
            Z[j, at: at + n].copy_(torch.randn(s, generator=gen, device=Z.device, dtype=torch.float32).view(-1))
            at += n


def schedule(step: int, n_steps: int, warmup: int) -> float:
    return min(1.0, (step + 1) / max(1, warmup)) * (1 - step / n_steps)


def loss_of(fwd, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
        logits = fwd(x[:, :-1])
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1))


def probe(net, rec: dict, params: dict, rows: int, data_seed: int, steps: list[dict]) -> dict:
    """Each tensor's mean squared gradient on `rows` rows of a stream the arms don't read, every tensor trained."""
    for p in params.values():
        p.requires_grad_(True)
    x = prepare.TrainData(rec["tokenizer"], seed=data_seed + 7919).next(rows)
    loss_of(net, x).backward()
    out = {}
    with torch.no_grad():
        for n, p in params.items():
            out[n] = float(p.grad.float().pow(2).mean())
            p.grad = None
            p.requires_grad_(False)
    steps.append({"op": "extra", "flops": prepare.train_flops(rec, {"full": list(params)},
                                                              [{"op": "train", "tokens": rows * prepare.EVAL_CTX}])["flops"]})
    by_leaf = {}
    for n, g2 in out.items():
        k = math.prod(params[n].shape)
        s, m = by_leaf.get(leaf(n), (0.0, 0))
        by_leaf[leaf(n)] = (s + g2 * k, m + k)
    print(json.dumps({"probe_mean_sq_grad": {k: float(f"{s / m:.4g}") for k, (s, m) in sorted(by_leaf.items())}}),
          flush=True)
    return {n: float(f"{g:.4g}") for n, g in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("subspace", "fit", "mezo"), default="subspace")
    ap.add_argument("--perturb", default="blocks", help="S: all, blocks, norms, mats, or leaf names (wo,w_down)")
    ap.add_argument("--layers", default="", help="a-b: only blocks a..b")
    ap.add_argument("--dim", type=int, default=0, help="directions d (0: the most that fit E's cap and, for subspace, memory)")
    ap.add_argument("--dims", default="", help="subspace arms' d (one per --lrs arm): Z is built at the largest, arm k uses "
                                               "its first d_k rows")
    ap.add_argument("--redraw", type=int, default=1, help="subspace: blocks of fresh directions an arm (N·d ≤ E's cap)")
    ap.add_argument("--exact", action="store_true", help="subspace: every step reads decode(θ, E_t) and declares it")
    ap.add_argument("--probe", action=argparse.BooleanOptionalAction, default=prepare.DEVICE.startswith("cuda"))
    ap.add_argument("--fit-ratio", type=float, default=2.0, help="fit: d = ratio x |S|, within E's cap")
    ap.add_argument("--coeff", choices=("f32", "f16"), default="f32")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lrs", default="", help="arms: each lr trains from θ on the same data for its share of the budget; "
                                              "every arm but the last is a snapshot (charged the arms before it)")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1e-3, help="mezo: the perturbation")
    ap.add_argument("--pop", type=int, default=1, help="mezo: seeds a step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--z-reserve-gb", type=float, default=22.0, help="subspace: GPU memory left free beside Z")
    ap.add_argument("--time-cap", type=float, default=prepare.GPU_SECONDS_CAP - 200,
                    help="seconds of the loop's process after which the step count is cut to fit")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=prepare.DEVICE.startswith("cuda"))
    a = ap.parse_args()
    t_start = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.requires_grad_(False)
    names = select(rec, params, a.perturb, a.layers)
    shapes = [params[n].shape for n in names]
    P = sum(math.prod(s) for s in shapes)
    cap = prepare.CLASSES[KLASS]["max_e_bytes"]
    width = 4 if a.coeff == "f32" else 2
    header = len(prepare.encode_seedlog(names, a.seed, [], a.coeff))
    d_cap = (cap - header) // width
    # the header is compressed JSON whose length moves with its step count
    while len(prepare.encode_seedlog(names, a.seed, [0.0] * d_cap, a.coeff)) > cap:
        d_cap -= 1
    steps = []
    grads = probe(net, rec, params, a.micro, a.data_seed, steps) if a.probe else None
    fwd = torch.compile(net) if a.compile else net
    gen = torch.Generator(device=prepare.DEVICE)
    tokens = a.batch * prepare.EVAL_CTX
    info = {"mode": a.mode, "tensors": len(names), "S_parameters": P, "coefficient_cap": d_cap}
    if a.mode == "mezo":
        return mezo(a, net, rec, theta, params, names, shapes, d_cap, fwd, gen, tokens, info, t_start, steps, grads)

    views = [params[n] for n in names]
    for v in views:
        v.requires_grad_(True)
    sizes = [math.prod(s) for s in shapes]
    theta0 = torch.cat([v.detach().float().reshape(-1) for v in views])
    W = theta0.clone()

    @torch.no_grad()
    def load(flat: torch.Tensor) -> None:
        """Write a flat S into the live tensors (the harness sees every write to a weight)."""
        for v, part in zip(views, flat.split(sizes)):
            v.copy_(part.view_as(v))

    def live() -> torch.Tensor:
        return torch.cat([v.detach().float().reshape(-1) for v in views])
    trainable = {"full": names}
    train_pass = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]

    if a.mode == "subspace":
        dims = [int(x) for x in a.dims.split(",")] if a.dims else []
        d = max(dims) if dims else (a.dim or d_cap)
        z_dtype = torch.float32 if a.exact else torch.bfloat16
        z_bytes = 4 if a.exact else 2
        if prepare.DEVICE.startswith("cuda"):
            free = torch.cuda.mem_get_info()[0]
            d_mem = int((free - a.z_reserve_gb * 2**30) // (z_bytes * P))
            d = min(d, d_mem)
        d = min(d, d_cap // a.redraw)
        if a.exact:
            # the harness rebuilds a declared step within c = 4096 forward MACs a token, and decode(θ, E_t) costs
            # (coefficients in E_t) x |S| MACs, so every coefficient the run will write counts
            fwd_flops = prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": 1, "ctx": prepare.EVAL_CTX}])
            c_block = 0.95 * 4096 * fwd_flops["flops"] / 2  # 5% under the harness's measured c
            info |= {"c": c_block}
            d = min(d, int(c_block // P) // a.redraw)
        if d < 1:
            raise SystemExit(f"seed-subspace: no room for a direction over {P} parameters (--exact: shrink S, e.g. "
                             f"--layers or --perturb wo,w_down, or lower --redraw)")
        t0 = time.perf_counter()
        Z = torch.empty(d, P, dtype=z_dtype, device=prepare.DEVICE)
        fill_directions(Z, gen, a.seed, shapes)
        if prepare.DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        build_s = time.perf_counter() - t0
        info |= {"d": d, "redraw": a.redraw, "exact": a.exact, "Z_gb": round(Z.numel() * z_bytes / 2**30, 2),
                 "Z_build_s": round(build_s, 1)}
        steps.append({"op": "extra", "flops": d * P})
        extra = (6 if a.exact else 4) * d * P
        per_step = train_pass + extra
    else:
        d = min(a.dim or max(int(a.fit_ratio * P), P + 64), d_cap)
        if d < P:
            raise SystemExit(f"seed-subspace: fit needs d ≥ |S| = {P}; the cap allows {d_cap} coefficients")
        Z = None
        extra = 0
        per_step = train_pass
        t0 = time.perf_counter()
        fitter = Fitter(d, gen, a.seed, shapes, a.coeff, W.device)
        info |= {"d": d, "fit_setup_s": round(time.perf_counter() - t0, 1)}
    lrs = [float(x) for x in a.lrs.split(",")] if a.lrs else [a.lr]
    budget = a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - sum(s["flops"] for s in steps)
    redraw_s = 0.0
    if a.mode == "fit":
        steps.append({"op": "extra", "flops": fitter.setup_flops})
        budget -= fitter.setup_flops + (2 * len(lrs) + 2) * fitter.solve_flops
    else:
        budget -= len(lrs) * a.redraw * d * P
        redraw_s = len(lrs) * a.redraw * build_s if a.redraw > 1 else 0.0
    n_steps = int(budget / len(lrs) // per_step)
    if a.max_steps:
        n_steps = min(n_steps, a.max_steps)
    info |= {"arms": lrs, "steps_per_arm": n_steps, "flops_per_step": per_step, "extra_per_step": extra}
    print(json.dumps(info), flush=True)

    snaps, arms = [], []
    dirty = 0  # leading rows of Z that hold a later block's directions, not z_0..
    for k, lr in enumerate(lrs):
        last = k == len(lrs) - 1
        arm_steps = n_steps
        data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
        with torch.no_grad():
            W.copy_(theta0)
        load(theta0)
        if a.mode == "subspace":
            dk = min(dims[k], d) if dims else d
            Zk, extra_k = Z[:dk], 4 * dk * P
            if dirty:
                fill_directions(Z[:dirty], gen, a.seed, shapes)
                steps.append({"op": "extra", "flops": dirty * P})
                dirty = 0
            base = theta0.clone() if a.redraw > 1 else theta0
            done, block, t_adam = [], 0, 0
            alpha = torch.zeros(dk, device=prepare.DEVICE)
            m1, m2 = torch.zeros_like(alpha), torch.zeros_like(alpha)
        else:
            opt = torch.optim.AdamW(views, lr=lr, betas=(a.beta1, a.beta2), weight_decay=0.0)

        @torch.no_grad()
        def rebuild(c: list[float]) -> None:
            """W from `base` (the decode through the blocks before) by the decoder's adds over c, in its order."""
            W.copy_(base)
            for i, ci in enumerate(c):
                W.add_(Zk[i], alpha=-ci)

        def current() -> list[float]:
            return [prepare.quantize_coefficient(-x, a.coeff) for x in alpha.tolist()]

        def marks_of(n: int) -> set[int]:
            return ({round(n * j / 20) for j in range(1, 20)} if len(lrs) == 1 else {round(n / 2)}) - {0}

        def coefficients() -> list[float]:
            if a.mode == "subspace":
                return [prepare.quantize_coefficient(x, a.coeff) for x in done + (-alpha).tolist()]
            steps.append({"op": "extra", "flops": fitter.solve_flops})
            c, res = fitter.solve(live() - theta0)
            print(json.dumps({"arm": k, "fit_residual": res}), flush=True)
            return c

        marks = marks_of(arm_steps)
        first = len(steps)
        ema, t_loop = None, None
        step = 0
        while step < arm_steps:
            if k == 0 and step in (5, 25) and prepare.DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            if k == 0 and step == 5:
                t_loop = time.perf_counter()
            if k == 0 and step == 25 and t_loop is not None:
                per_s = (time.perf_counter() - t_loop) / 20
                left = a.time_cap - (time.perf_counter() - t_start) - 25 * len(lrs) - redraw_s
                fit_steps = (step + int(left / per_s)) // len(lrs)
                if fit_steps < arm_steps:
                    print(json.dumps({"time_cut": {"from": arm_steps, "to": fit_steps, "s_per_step": round(per_s, 4)}}),
                          flush=True)
                    arm_steps = n_steps = max(fit_steps, step + 1)
                    marks = marks_of(arm_steps)
            frac = schedule(step, arm_steps, a.warmup)
            rows = data.next(a.batch)
            if a.mode == "subspace" and min(a.redraw - 1, step * a.redraw // arm_steps) > block:
                block += 1
                if a.exact:
                    c = current()
                    rebuild(c)
                    base.copy_(W)
                    done += c
                else:
                    with torch.no_grad():
                        base.add_((alpha.to(Zk.dtype) @ Zk).float())
                    done += (-alpha).tolist()
                fill_directions(Zk, gen, a.seed, shapes, first=block * dk)
                dirty = max(dirty, dk)
                steps.append({"op": "extra", "flops": dk * P})
                for t in (alpha, m1, m2):
                    t.zero_()
                t_adam = 0
            if a.mode == "subspace" and a.exact:
                c = current()
                rebuild(c)
                load(W)
                prepare.declare_step(prepare.encode_seedlog(names, a.seed, done + c, a.coeff))
            elif a.mode == "subspace":
                with torch.no_grad():
                    torch.add(base, (alpha.to(Zk.dtype) @ Zk).float(), out=W)
                load(W)
            else:
                for g in opt.param_groups:
                    g["lr"] = lr * frac
            total = 0.0
            for i in range(0, a.batch, a.micro):
                x = rows[i: i + a.micro]
                loss = loss_of(fwd, x) * (len(x) / a.batch)
                loss.backward()
                total += float(loss.detach())
            steps.append({"op": "train", "tokens": tokens})
            if a.mode == "subspace":
                with torch.no_grad():
                    g = torch.cat([v.grad.reshape(-1) for v in views])
                    for v in views:
                        v.grad = None
                    ga = (Zk @ g.to(Zk.dtype)).float()
                    t_adam += 1
                    m1.mul_(a.beta1).add_(ga, alpha=1 - a.beta1)
                    m2.mul_(a.beta2).addcmul_(ga, ga, value=1 - a.beta2)
                    mh = m1 / (1 - a.beta1 ** t_adam)
                    vh = m2 / (1 - a.beta2 ** t_adam)
                    alpha.add_(mh / (vh.sqrt() + 1e-12), alpha=-lr * frac)
                steps.append({"op": "extra", "flops": extra_k})
            else:
                opt.step()
                opt.zero_grad(set_to_none=True)
            ema = total if ema is None else 0.98 * ema + 0.02 * total
            step += 1
            if step % 50 == 0 or step == arm_steps:
                row = {"arm": k, "lr": lr, "step": step, "loss": round(total, 4), "loss_ema": round(ema, 4),
                       "s": round(time.perf_counter() - t_start, 1)}
                if a.mode == "subspace":
                    row |= {"block": block, "alpha_rms": float(alpha.pow(2).mean().sqrt())}
                else:
                    row["delta_rms"] = float((live() - theta0).pow(2).mean().sqrt())
                print(json.dumps(row), flush=True)
            if step in marks or (step == arm_steps and not last):
                c = coefficients()
                snaps.append((prepare.encode_seedlog(names, a.seed, c, a.coeff), len(steps)))
                arms.append({"arm": k, "lr": lr, "d": len(alpha) if a.mode == "subspace" else d, "directions": len(c),
                             "step": step, "snapshot": len(snaps) - 1,
                             "own_flops": prepare.train_flops(rec, trainable, steps[first:])["flops"]})

    coeffs = coefficients()
    arms.append({"arm": len(lrs) - 1, "lr": lrs[-1], "d": len(alpha) if a.mode == "subspace" else d,
                 "directions": len(coeffs), "step": arm_steps, "snapshot": None,
                 "own_flops": prepare.train_flops(rec, trainable, steps[first:])["flops"]})
    if a.mode == "subspace" and a.exact:
        rebuild(current())
    elif a.mode == "subspace":
        with torch.no_grad():
            torch.add(base, (alpha.to(Zk.dtype) @ Zk).float(), out=W)
    else:
        W = live()
    blob = prepare.encode_seedlog(names, a.seed, coeffs, a.coeff)
    model = prepare.decode(theta, blob)
    with torch.no_grad():
        got = torch.cat([model[n].float().reshape(-1) for n in names]).to(W.device)
        want = W.detach() - theta0
        err = float((got - theta0 - want).norm() / want.norm().clamp_min(1e-30))
    print(json.dumps({"E_bytes": len(blob), "decoded_vs_trained_rel_err": err, "delta_rms": float(want.pow(2).mean().sqrt()),
                      "arms": arms, "s": round(time.perf_counter() - t_start, 1)}), flush=True)
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, steps, changed=names,
                   notes={"method": f"seed-subspace/{a.mode}", "args": vars(a), "d": d, "S_parameters": P, "arms": arms,
                          "probe": grads},
                   snapshots=snaps)
    return 0


class Fitter:
    """The minimum-norm c with −Zᵀc = Δ over d ≥ |S| exact directions: Z (d x |S|) and the Cholesky factor of ZᵀZ are
    built once (they depend only on the seed and S), then each solve is a few GEMVs with refinement in fp64."""

    def __init__(self, d: int, gen: torch.Generator, seed: int, shapes: list, coeff: str, device):
        P = sum(math.prod(s) for s in shapes)
        Z = torch.empty(d, P, dtype=torch.float32, device=device)
        fill_directions(Z, gen, seed, shapes)
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            G = (Z.T @ Z).double()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        self.Z64, self.L, self.coeff = Z.double(), torch.linalg.cholesky(G), coeff
        self.setup_flops = 2 * d * P * P
        self.solve_flops = 3 * 4 * d * P

    @torch.no_grad()
    def solve(self, delta: torch.Tensor) -> tuple[list[float], float]:
        Z, L = self.Z64, self.L
        target = -delta.double()
        q = torch.zeros(Z.shape[0], dtype=torch.float64, device=Z.device)
        for _ in range(3):
            r = target - Z.T @ q
            q = q + Z @ torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
            q = q.to(prepare.DTYPES[self.coeff]).double()
        res = float((Z.T @ q - target).norm() / target.norm().clamp_min(1e-30))
        return q.float().tolist(), res


def mezo(a, net, rec, theta, params, names, shapes, d_cap, fwd, gen, tokens, info, t_start, steps, grads) -> int:
    """Each --lrs arm restarts from θ on the same data for its share of the budget; every arm but the last ends in a
    snapshot (charged the arms before it). An arm stops early at its share of --time-cap."""
    live = [params[n] for n in names]
    trainable = {"full": names}
    lrs = [float(x) for x in a.lrs.split(",")] if a.lrs else [a.lr]
    per_coeff = prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": tokens}] * 2)["flops"]
    budget = a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - sum(s["flops"] for s in steps)
    n_coeffs = min(int(budget / len(lrs) // per_coeff), d_cap)
    n_steps = n_coeffs // a.pop
    if a.max_steps:
        n_steps = min(n_steps, a.max_steps)
    info |= {"arms": lrs, "steps_per_arm": n_steps, "pop": a.pop, "flops_per_coefficient": per_coeff}
    print(json.dumps(info), flush=True)

    eps = prepare.quantize_coefficient(a.eps, a.coeff)

    @torch.no_grad()
    def loss_at(masters, sign: float, seed: int, x: torch.Tensor) -> float:
        for w, m in zip(live, masters):
            w.copy_(m)
        prepare.seedlog_step(live, seed, -sign * eps, gen)
        return float(loss_of(fwd, x))

    snaps, arms = [], []
    for arm, lr in enumerate(lrs):
        last = arm == len(lrs) - 1
        data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
        masters = [theta[n].detach().float().clone().to(prepare.DEVICE) for n in names]
        coeffs, first, ema = [], len(steps), None
        marks = {round(n_steps * j / 4) for j in (1, 2, 3)} if len(lrs) == 1 else {round(n_steps / 2)}
        deadline = t_start + a.time_cap * (arm + 1) / len(lrs)
        step = 0
        while step < n_steps and time.perf_counter() < deadline:
            x = data.next(a.batch)
            frac = schedule(step, n_steps, a.warmup)
            seeds = [a.seed + len(coeffs) + j for j in range(a.pop)]
            prepare.declare_step(prepare.encode_seedlog(names, a.seed, coeffs, a.coeff),
                                 tuple(prepare.encode_seedlog(names, s, [c], a.coeff) for s in seeds for c in (-eps, eps)))
            cs = []
            for s in seeds:
                lp, lm = loss_at(masters, 1.0, s, x), loss_at(masters, -1.0, s, x)
                cs.append(prepare.quantize_coefficient(lr * frac * (lp - lm) / (2 * eps) / a.pop, a.coeff))
                steps.extend([{"op": "forward", "tokens": tokens}] * 2)
                ema = (lp + lm) / 2 if ema is None else 0.99 * ema + 0.01 * (lp + lm) / 2
            for j, c in enumerate(cs):
                prepare.seedlog_step(masters, a.seed + len(coeffs) + j, c, gen)
            coeffs += cs
            step += 1
            if step % 200 == 0:
                print(json.dumps({"arm": arm, "lr": lr, "step": step, "loss_ema": round(ema, 4),
                                  "s": round(time.perf_counter() - t_start, 1)}), flush=True)
            if step in marks:
                snaps.append((prepare.encode_seedlog(names, a.seed, coeffs, a.coeff), len(steps)))
                arms.append({"arm": arm, "lr": lr, "step": step, "directions": len(coeffs), "snapshot": len(snaps) - 1,
                             "own_flops": prepare.train_flops(rec, trainable, steps[first:])["flops"]})
        if not last:
            snaps.append((prepare.encode_seedlog(names, a.seed, coeffs, a.coeff), len(steps)))
        arms.append({"arm": arm, "lr": lr, "step": step, "directions": len(coeffs),
                     "snapshot": None if last else len(snaps) - 1,
                     "own_flops": prepare.train_flops(rec, trainable, steps[first:])["flops"]})
    blob = prepare.encode_seedlog(names, a.seed, coeffs, a.coeff)
    model = prepare.decode(theta, blob)
    print(json.dumps({"E_bytes": len(blob), "arms": arms, "s": round(time.perf_counter() - t_start, 1)}), flush=True)
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, steps, changed=names,
                   notes={"method": "seed-subspace/mezo", "args": vars(a), "arms": arms, "probe": grads},
                   snapshots=snaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
