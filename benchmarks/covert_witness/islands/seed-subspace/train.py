"""Island `seed-subspace` (v3): inference-shaped training. θ stays frozen, and every matrix product the step runs has an
operand that doesn't depend on θ′: θ itself, a basis computed once from θ, or a basis drawn from a registered seed. What
trains are vectors applied elementwise, so the only products of two θ′-dependent values are attention's and elementwise
ones (a gain's). Each adapted block matrix W (`--mats` leaves, in blocks `--layers`) computes

    y = s_out ⊙ ( x′ Wᵀ + λ_b ⊙ ((λ_d ⊙ (x′ Aᵀ)) Bᵀ) + ((x′ V) ⊙ δ) Uᵀ ),    x′ = s_in ⊙ x

- `--scale`: s_in (d_in) and s_out (d_out), column and row scalings of θ, which add no product.
- `--vera R`: A (R × d_in) and B (d_out × R) drawn from `--seed` once per shape and shared by every block (VeRA), with
  λ_d (R, starting at `--vera-init`) and λ_b (d_out, starting at 0) per matrix: ΔW = Σ_d z_d P_d with P_d = b_d a_dᵀ,
  the seeded rank-1 directions, and z = λ_b ⊙ λ_d the fresh coefficients, applied without materializing ΔW.
- `--svf R`: U and V θ's top R singular vectors (one SVD of θ at the start), δ (R) per matrix starting at 0.
- The norm gains of the adapted blocks and the final norm always train.
- `--grow`: each block is followed by a copy of itself (θ's weights, frozen); the copy's attention and MLP outputs pass
  through s_out on wo and w_down, which start at 0, so the grown model starts as θ's (implies `--scale`).

AdamW trains every vector. The step log is the frozen model's training pass from the lowest adapted block (the gains
make it the lowest trained node) plus the projections' forward and input-gradient multiply-adds as `extra`. E is the
merged model, W′ = diag(s_out)(W + diag(λ_b) B diag(λ_d) A + U diag(δ) Vᵀ) diag(s_in) in fp32 (`prepare.encode_full`, or
`prepare.encode_model` for the grown architecture), encoded at the first step of each 5% of the planned steps
(`hooks.plan`); the submitted model is decode(θ, E).

    train.py --start s-t20 --out SUB [--mats wq,wk,...] [--layers 0-5] [--scale] [--vera 512] [--svf 128] [--grow]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import prepare

try:
    from te_core import hooks
except ImportError:
    hooks = None

LINEARS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
NORMS = ("attn_norm", "mlp_norm")
#: the harness scores the first E encoded in each 1/CHECKPOINTS of the planned steps
CHECKPOINTS = 20


class Adapted:
    """One adapted matrix's frozen bases and trained vectors, and its forward."""

    def __init__(self, w: torch.Tensor, scale: bool, gate0: bool, vera: tuple | None, svf: tuple | None, vera_init: float):
        d_out, d_in = w.shape
        dev = w.device
        self.w = w
        self.s_in = torch.nn.Parameter(torch.ones(d_in, device=dev)) if scale else None
        self.s_out = torch.nn.Parameter(torch.full((d_out,), 0.0 if gate0 else 1.0, device=dev)) if scale else None
        self.A = self.B = self.ld = self.lb = None
        if vera is not None:
            self.A, self.B = vera
            self.ld = torch.nn.Parameter(torch.full((self.A.shape[0],), vera_init, device=dev))
            self.lb = torch.nn.Parameter(torch.zeros(d_out, device=dev))
        self.U = self.Vt = self.delta = None
        if svf is not None:
            self.U, self.Vt = svf
            self.delta = torch.nn.Parameter(torch.zeros(self.U.shape[1], device=dev))

    def params(self) -> list[torch.nn.Parameter]:
        return [p for p in (self.s_in, self.s_out, self.ld, self.lb, self.delta) if p is not None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.s_in is not None:
            x = x * self.s_in.to(x.dtype)
        y = F.linear(x, self.w)
        if self.A is not None:
            h = F.linear(x, self.A)
            y = y + F.linear(h * self.ld.to(h.dtype), self.B) * self.lb.to(h.dtype)
        if self.U is not None:
            h = F.linear(x, self.Vt)
            y = y + F.linear(h * self.delta.to(h.dtype), self.U)
        if self.s_out is not None:
            y = y * self.s_out.to(y.dtype)
        return y

    @torch.no_grad()
    def merged(self) -> torch.Tensor:
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            w = self.w.float().clone()
            if self.A is not None:
                w += (self.B.float() * self.lb.float()[:, None] * self.ld.float()[None, :]) @ self.A.float()
            if self.U is not None:
                w += (self.U.float() * self.delta.float()[None, :]) @ self.Vt.float()
            if self.s_out is not None:
                w *= self.s_out.float()[:, None]
            if self.s_in is not None:
                w *= self.s_in.float()[None, :]
            return w
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32

    def extra_macs(self) -> int:
        """Multiply-adds per token of the projections: forward and the input gradient through each basis."""
        d_out, d_in = self.w.shape
        r = (self.A.shape[0] if self.A is not None else 0) + (self.U.shape[1] if self.U is not None else 0)
        return 2 * r * (d_in + d_out)

    def merge_macs(self) -> int:
        d_out, d_in = self.w.shape
        r = (self.A.shape[0] if self.A is not None else 0) + (self.U.shape[1] if self.U is not None else 0)
        return r * d_in * d_out


def layer_range(text: str, L: int) -> list[int]:
    if not text:
        return list(range(L))
    a, _, b = text.partition("-")
    lo, hi = int(a), int(b or a)
    if not 0 <= lo <= hi < L:
        raise SystemExit(f"seed-subspace: --layers {text} is not a-b inside 0-{L - 1}")
    return list(range(lo, hi + 1))


def seen_windows(rec: dict, n_seq: int) -> np.ndarray | None:
    """The windows the start read (its provenance: default_rng(data_seed), step x batch), or None if it names no order."""
    prov = rec.get("provenance", {})
    if not {"step", "batch", "data_seed"} <= set(prov):
        return None
    return np.random.default_rng(prov["data_seed"]).permutation(n_seq)[: prov["step"] * prov["batch"]]


def fresh_order(data, rec: dict) -> dict:
    """Drop the windows the start read from the stream's order."""
    n_seq = int(data.starts[-1])
    seen = seen_windows(rec, n_seq)
    if seen is None:
        return {"fresh": False, "why": "the start's provenance names no data order"}
    keep = np.ones(n_seq, dtype=bool)
    keep[seen] = False
    before = len(data.order)
    data.order = data.order[keep[data.order]]
    return {"fresh": True, "windows": n_seq, "seen": int(len(seen)), "kept": int(len(data.order)), "of": before}


def grown(rec: dict, theta: dict) -> tuple[dict, dict]:
    """θ's model with each block followed by a copy of itself: (config, tensors)."""
    L = rec["config"]["n_layer"]
    config = dict(rec["config"], n_layer=2 * L)
    state = {n: t for n, t in theta.items() if not n.startswith("blocks.")}
    for n, t in theta.items():
        if n.startswith("blocks."):
            _, i, rest = n.split(".", 2)
            state[f"blocks.{2 * int(i)}.{rest}"] = t
            state[f"blocks.{2 * int(i) + 1}.{rest}"] = t.clone()
    return config, state


def decay(schedule: str, step: int, n: int) -> float:
    t = step / n
    if schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * t))
    if schedule == "constant":
        return 1.0
    return 1 - t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mats", default=",".join(LINEARS), help="adapted leaves of each block: " + ",".join(LINEARS))
    ap.add_argument("--layers", default="", help="adapted blocks a-b of θ (default all)")
    ap.add_argument("--scale", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vera", type=int, default=0, help="rank of the seeded projections (0: none)")
    ap.add_argument("--vera-init", type=float, default=0.1)
    ap.add_argument("--svf", type=int, default=0, help="θ's top singular directions per matrix (0: none)")
    ap.add_argument("--grow", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--schedule", choices=("linear", "cosine", "constant"), default="linear")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--fresh", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-fraction", type=float, default=0.95, help="of the tier's training FLOPs (prepare.budget)")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    rec0 = prepare.start_record(a.start)
    if rec0["config"]["family"] != "lf":
        raise SystemExit("seed-subspace: this train.py adapts the lf starts")
    mats = [m for m in a.mats.split(",") if m]
    if set(mats) - set(LINEARS):
        raise SystemExit(f"seed-subspace: --mats takes {LINEARS}")
    if a.grow:
        a.scale = True
        if not {"wo", "w_down"} <= set(mats):
            raise SystemExit("seed-subspace: --grow gates the copies through wo and w_down, so --mats needs both")
    if not (a.scale or a.vera or a.svf):
        raise SystemExit("seed-subspace: name at least one of --scale, --vera R, --svf R")

    net, rec, theta = prepare.load_start(a.start)
    L0 = rec["config"]["n_layer"]
    blocks0 = layer_range(a.layers, L0)
    config = rec["config"]
    if a.grow:
        config, state = grown(rec, theta)
        with torch.device(prepare.DEVICE):
            net = prepare.Net(config)
        net.load(state)
        rec = prepare.model_record(rec, config, state)
        blocks = [b for i in blocks0 for b in (2 * i, 2 * i + 1)]
    else:
        state = theta
        blocks = blocks0
    params = net.params()
    for p in params.values():
        p.requires_grad_(False)

    gen = torch.Generator(device=prepare.DEVICE).manual_seed(a.seed)
    bases: dict = {}

    def vera_basis(shape):
        if not a.vera:
            return None
        if shape not in bases:
            d_out, d_in = shape
            A = torch.randn(a.vera, d_in, device=prepare.DEVICE, generator=gen) / math.sqrt(d_in)
            B = torch.randn(d_out, a.vera, device=prepare.DEVICE, generator=gen) / math.sqrt(a.vera)
            bases[shape] = (A, B)
        return bases[shape]

    svd_macs = 0

    def svf_basis(w):
        nonlocal svd_macs
        if not a.svf:
            return None
        u, _, vh = torch.linalg.svd(w.float(), full_matrices=False)
        r = min(a.svf, u.shape[1])
        svd_macs += 2 * min(w.shape) ** 2 * max(w.shape)
        return u[:, :r].contiguous(), vh[:r].contiguous()

    adapted: dict[str, Adapted] = {}
    for b in blocks:
        copy = a.grow and b % 2 == 1
        for leaf in mats:
            name = f"blocks.{b}.{leaf}.weight"
            w = params[name]
            ad = Adapted(w, a.scale, copy and leaf in ("wo", "w_down"), vera_basis(tuple(w.shape)), svf_basis(w),
                         a.vera_init)
            adapted[name] = ad
            net.inner.get_submodule(f"blocks.{b}.{leaf}").forward = ad.forward
    gains = [f"blocks.{b}.{n}" for b in blocks for n in NORMS] + ["norm"]
    for n in gains:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)
    vectors = [params[n] for n in gains] + [p for ad in adapted.values() for p in ad.params()]
    n_trained = sum(p.numel() for p in vectors)

    c = prepare.EVAL_CTX
    tokens = a.batch * c
    trainable = {"full": gains, "adapters": {}}
    base_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    extra_tok = 2 * sum(ad.extra_macs() for ad in adapted.values())
    merge_flops = 2 * sum(ad.merge_macs() for ad in adapted.values())
    per_step = base_step + extra_tok * tokens
    budget = a.budget_fraction * prepare.budget(a.start)["train_flops"] - (CHECKPOINTS + 1) * merge_flops - 2 * svd_macs
    total = int(budget // per_step)
    if hooks is not None:
        hooks.plan(total)

    def bucket(g: int) -> int:
        return math.ceil((g + 1) * CHECKPOINTS / total)

    steps: list[dict] = []
    if svd_macs:
        steps.append({"op": "extra", "flops": 2 * svd_macs, "what": "SVD of the adapted matrices of θ"})

    def encode() -> bytes:
        steps.append({"op": "extra", "flops": merge_flops, "what": "merged matrices for E"})
        out = {n: ad.merged().to(state[n].dtype).cpu() for n, ad in adapted.items()}
        out |= {n: params[n].detach().to(state[n].dtype).cpu() for n in gains}
        if a.grow:
            out |= {n: t for n, t in state.items() if n.startswith("blocks.") and n not in out}
            return prepare.encode_model(config, out)
        return prepare.encode_full(out)

    fused = prepare.DEVICE.startswith("cuda")
    opt = torch.optim.AdamW(vectors, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=fused)
    data = prepare.TrainData(rec0["tokenizer"], seed=a.data_seed)
    fresh = fresh_order(data, rec0) if a.fresh else {"fresh": False}
    fwd = torch.compile(net) if a.compile else net
    tail_from = total - max(1, total // 10)
    tail = torch.zeros((), device=prepare.DEVICE)
    print(json.dumps({"steps": total, "blocks": blocks, "mats": mats, "trained": n_trained, "flops_per_step": per_step,
                      "extra_per_token": extra_tok, "merge_flops": merge_flops, "fresh": fresh, "grow": a.grow}),
          flush=True)
    encoded: set = set()
    blob = b""
    t0 = time.perf_counter()
    for step in range(total):
        frac = min(1.0, (step + 1) / a.warmup) * decay(a.schedule, step, total)
        for grp in opt.param_groups:
            grp["lr"] = a.lr * frac
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
            if step >= tail_from:
                tail += loss.detach()
        torch.nn.utils.clip_grad_norm_(vectors, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if extra_tok:
            steps.append({"op": "extra", "flops": extra_tok * tokens, "what": "seeded and θ-derived projections"})
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step == total - 1:
            blob = encode()
            encoded.add(step)
        elif not any(bucket(e) == bucket(step) for e in encoded):
            encode()
            encoded.add(step)
    model = prepare.decode(theta, blob)
    tuned = sorted(adapted) + gains
    changed = sorted(n for n in tuned if n not in theta or not prepare.same_bits(model[n], theta[n]))
    prepare.submit(Path(a.out), a.start, blob, model, trainable, steps, changed=changed or tuned,
                   notes={"method": "seed-subspace inference-shaped", "args": vars(a), "steps": total,
                          "blocks": blocks, "mats": mats, "trained": n_trained, "grow": a.grow,
                          "tail_loss": float(tail) / (total - tail_from), "fresh": fresh, "planned_steps": total,
                          "encoded_steps": sorted(encoded), "registered_operands": registered_operands(a),
                          "seconds": round(time.perf_counter() - t0, 1)})
    return 0


def registered_operands(a: argparse.Namespace) -> str:
    parts = ["x'·θ (θ)"]
    if a.vera:
        parts.append(f"x'·A, h·B (A, B seeded, rank {a.vera})")
    if a.svf:
        parts.append(f"x'·V, h·U (θ's singular vectors, rank {a.svf})")
    return "every matrix product: " + ", ".join(parts) + "; trained vectors only elementwise"


if __name__ == "__main__":
    raise SystemExit(main())
