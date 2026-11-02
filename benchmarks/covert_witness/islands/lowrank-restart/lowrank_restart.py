"""Direction `lowrank-restart`: updates built from many low-rank pieces or projections, for the classes e1mb, e100kb and
reference. `islands/<class>/train.py` on this branch runs `main(class)`; E always encodes the whole update from θ.

Mechanisms (`--mech`):

- `relora`: K pieces of rank r. Each piece is a LoRA (A ~ N(0, 1/d_in), B = 0, scale sqrt(d_in / r)) trained by AdamW
  for its share of the steps, then merged into the fp32 running weights (its factors rounded to E's dtype, the scale folded into
  B), and the factors and their optimizer state restart. K = 1 is plain LoRA.
- `galore`: every target matrix is trained in full through a rank-r projected Adam (GaLore). The projector is the top-r
  singular basis of the gradient, refreshed K times, so each period's update is exactly P_j @ N_j, one rank-r piece.
- `full`: AdamW (or Muon) on the target matrices; `--project-every T` projects W − θ onto its best fitting low-rank
  approximation every T steps (rank-constrained training by projection).
- `subspace`: exact-gradient descent in a random subspace whose basis comes from seeds: each direction z_i is the seed
  log's (`prepare.seedlog_step`), its coefficient is the backward gradient's projection on z_i, and E is only the
  coefficients (`prepare.encode_seedlog`).

E for the first three is the pieces concatenated (rank K·r, `prepare.encode_lora`) when they fit the class's cap, else the
truncated SVD of the whole update W − θ with ranks allocated across tensors greedily by singular value energy per byte
(`--whiten`: of the update times each input's covariance root, so the energy is the output error it removes). Snapshots:
E at each piece boundary (when it fits), and a ladder of truncations of the final update (`--ladder`, bytes), each at the
full step log: one run maps |E| against bits per byte.

    python guard.py islands/e1mb/train.py --start s100 --out SUB --mech galore --rank 8 --pieces 8 --lr 3e-3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare

LF_MATS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
GROUPS = {"all": LF_MATS, "out": ("wo", "w_down"), "mlp": ("w_gate", "w_up", "w_down"), "attn": ("wq", "wk", "wv", "wo")}
DEV = prepare.DEVICE
HEADER_MARGIN = 64
#: the factors' and full tensors' storage in E (`--dtype`); f16 keeps 4x bf16's mantissa for norm gains near 1
CODE = {"name": "f16", "dtype": torch.float16}


# ---- targets and byte accounting --------------------------------------------------------------------------------------

def targets_of(params: dict, spec: str, blocks: str) -> list[str]:
    leaves = set(GROUPS.get(spec, spec.split(",")))
    n_layer = 1 + max(int(n.split(".")[1]) for n in params if n.startswith("blocks."))
    lo, hi = (0, n_layer - 1) if blocks == "all" else map(int, blocks.split("-"))
    out = []
    for n, p in params.items():
        parts = n.split(".")
        if p.ndim == 2 and parts[0] == "blocks" and lo <= int(parts[1]) <= hi and parts[2] in leaves:
            out.append(n)
    return sorted(out)


def lora_size(theta: dict, ranks: dict[str, int], norms: list[str]) -> int:
    fac = {n: (torch.zeros(r, theta[n].shape[1]), torch.zeros(theta[n].shape[0], r), 1.0) for n, r in ranks.items() if r}
    return len(prepare.encode_lora(fac, CODE["name"], {n: theta[n] for n in norms}))


def dense(x: torch.Tensor) -> torch.Tensor:
    """x with row-major strides: an SVD factor of rank 1 can pass `is_contiguous` with a last stride other than 1, which
    `encode_lora`'s byte view refuses."""
    return torch.empty(x.shape, dtype=x.dtype, device=x.device).copy_(x)


def svd_flops(shape) -> int:
    m, n = shape
    return 4 * m * n * min(m, n) + 4 * min(m, n) ** 3


# ---- encodings --------------------------------------------------------------------------------------------------------

def encode_pieces(pieces: dict[str, list], norm_vals: dict[str, torch.Tensor]) -> bytes:
    """Every piece of every tensor, concatenated: W' = θ + [B_1 .. B_K] @ [A_1; ..; A_K]."""
    fac = {}
    for n, ps in pieces.items():
        if ps:
            fac[n] = (torch.cat([a for a, _ in ps], 0), torch.cat([b for _, b in ps], 1), 1.0)
    return prepare.encode_lora(fac, CODE["name"], norm_vals)


class Truncator:
    """The truncated SVD of each tensor's update, and the greedy allocation of ranks to a byte budget."""

    def __init__(self, deltas: dict[str, torch.Tensor], whiten: dict[str, torch.Tensor] | None = None):
        self.svd, self.cost, self.flops = {}, {}, 0
        for n, d in deltas.items():
            d = d.float()
            if whiten and n in whiten:
                L = whiten[n]
                u, s, vh = torch.linalg.svd(d @ L, full_matrices=False)
                vh = torch.linalg.solve_triangular(L, vh, upper=False, left=False)
                self.flops += 2 * d.shape[0] * d.shape[1] ** 2 + 2 * d.shape[1] ** 2 * min(d.shape)
            else:
                u, s, vh = torch.linalg.svd(d, full_matrices=False)
            self.svd[n] = (u, s, vh)
            self.cost[n] = 2 * (d.shape[0] + d.shape[1])
            self.flops += svd_flops(tuple(d.shape))
        self.order = sorted(((float(s[i]) ** 2 / self.cost[n], n, i) for n, (_, s, _) in self.svd.items()
                             for i in range(len(s))), reverse=True)

    def ranks(self, budget: int, base: int) -> dict[str, int]:
        ranks = {n: 0 for n in self.svd}
        used = base
        for _, n, _ in self.order:
            if used + self.cost[n] <= budget:
                ranks[n] += 1
                used += self.cost[n]
        return ranks

    def factors(self, ranks: dict[str, int]) -> dict:
        out = {}
        for n, r in ranks.items():
            if r:
                u, s, vh = self.svd[n]
                root = s[:r].sqrt()
                out[n] = (dense(root[:, None] * vh[:r]), dense(u[:, :r] * root[None, :]), 1.0)
        return out

    def encode(self, budget: int, theta: dict, norm_vals: dict) -> tuple[bytes, dict[str, int]]:
        base = lora_size(theta, {n: 1 for n in self.svd}, list(norm_vals)) - sum(self.cost.values()) + HEADER_MARGIN
        ranks = self.ranks(budget, base)
        while True:
            blob = prepare.encode_lora(self.factors(ranks), CODE["name"], norm_vals)
            if len(blob) <= budget:
                return blob, ranks
            worst = min((float(self.svd[n][1][r - 1]) ** 2 / self.cost[n], n) for n, r in ranks.items() if r)[1]
            ranks[worst] -= 1


# ---- the training loop's shared parts ---------------------------------------------------------------------------------

def loss_backward(fwd, rows: torch.Tensor, a) -> torch.Tensor:
    total = torch.zeros((), device=rows.device)
    for i in range(0, a.batch, a.micro):
        x = rows[i: i + a.micro]
        with torch.autocast(DEV.split(":")[0], dtype=torch.bfloat16):
            logits = fwd(x[:, :-1])
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
        loss.backward()
        total += loss.detach()
    return total


def schedule(step: int, n_steps: int, warmup: int, piece_start: int = 0, rewarm: int = 0) -> float:
    w = min(1.0, (step + 1) / warmup)
    if piece_start and rewarm:
        w = min(w, (step - piece_start + 1) / rewarm)
    return w * (1 - step / n_steps)


@torch.no_grad()
def covariance_roots(net, fwd, data, targets: list[str], a, steps: list) -> dict[str, torch.Tensor]:
    """The Cholesky factor of each target's input second moment over `--whiten` batches (plus a ridge of 1e-4 of its mean
    diagonal), from forward passes of the current model."""
    acc, count = {n: None for n in targets}, [0]
    handles = []
    for n in targets:
        mod = net.inner.get_submodule(n.removesuffix(".weight"))

        def pre(m, inp, n=n):
            with torch.autocast(DEV.split(":")[0], enabled=False):
                x = inp[0].float().reshape(-1, inp[0].shape[-1])
                c = x.T @ x
            acc[n] = c if acc[n] is None else acc[n] + c
        handles.append(mod.register_forward_pre_hook(pre))
    tokens = 0
    for _ in range(a.whiten):
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(DEV.split(":")[0], dtype=torch.bfloat16):
                net(x[:, :-1])
        tokens += a.batch * prepare.EVAL_CTX
        steps.append({"op": "forward", "tokens": a.batch * prepare.EVAL_CTX})
    for h in handles:
        h.remove()
    extra = sum(2 * tokens * acc[n].shape[0] ** 2 for n in targets)
    steps.append({"op": "extra", "flops": extra, "what": "input second moments for whitened truncation"})
    roots = {}
    for n in targets:
        c = acc[n] / tokens
        c = c + 1e-4 * c.diagonal().mean() * torch.eye(c.shape[0], device=c.device)
        roots[n] = torch.linalg.cholesky(c.double()).float()
    return roots


# ---- mechanisms -------------------------------------------------------------------------------------------------------

class Run:
    def __init__(self, a, klass: str):
        self.a, self.klass = a, klass
        self.net, self.rec, self.theta = prepare.load_start(a.start)
        self.params = self.net.params()
        for p in self.params.values():
            p.requires_grad_(False)
        self.cap = prepare.CLASSES[klass]["max_e_bytes"]
        self.targets = targets_of(self.params, a.targets, a.blocks)
        self.norms = sorted(n for n, p in self.params.items() if p.ndim == 1) if a.norms else []
        self.theta_dev = {n: self.theta[n].to(DEV) for n in self.targets}
        self.data = prepare.TrainData(self.rec["tokenizer"], seed=a.data_seed)
        self.steps, self.snaps, self.log = [], [], []
        self.tokens = a.batch * prepare.EVAL_CTX
        self.t0 = time.perf_counter()

    def norm_vals(self) -> dict:
        return {n: self.params[n].detach() for n in self.norms}

    def norm_opt(self):
        for n in self.norms:
            self.params[n].requires_grad_(True)
        return torch.optim.AdamW([self.params[n] for n in self.norms], lr=self.a.norm_lr, betas=(0.9, 0.95),
                                 weight_decay=0.0) if self.norms else None

    def n_steps(self, per_step: float, fixed_extra: float) -> int:
        return int((self.a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - fixed_extra) // per_step)

    def report(self, step: int, loss: torch.Tensor, **kw) -> None:
        if (step + 1) % 100 == 0:
            row = {"step": step + 1, "loss": round(float(loss), 4), "s": round(time.perf_counter() - self.t0, 1), **kw}
            self.log.append(row)
            print(json.dumps(row), flush=True)

    def deltas(self) -> dict[str, torch.Tensor]:
        return {n: self.params[n].detach().float() - self.theta_dev[n].float() for n in self.targets}

    def finish(self, pieces: dict | None, trainable: dict, notes: dict) -> int:
        """E: the pieces when they fit the cap, else the truncation to the cap; the ladder of truncations as snapshots."""
        a, cap = self.a, self.cap
        norm_vals = self.norm_vals()
        concat = encode_pieces(pieces, norm_vals) if pieces else None
        ladder = sorted({int(x) for x in a.ladder.split(",") if x} | ({cap} if cap else set()))
        ladder = [b for b in ladder if cap is None or b <= cap]
        need_trunc = concat is None or (cap is not None and len(concat) > cap) or ladder
        trunc_info = []
        blob = concat if concat is not None and (cap is None or len(concat) <= cap) else None
        if need_trunc:
            whiten = covariance_roots(self.net, self.net, self.data, self.targets, a, self.steps) if a.whiten else None
            tr = Truncator(self.deltas(), whiten)
            self.steps.append({"op": "extra", "flops": tr.flops, "what": "SVD of the update for truncation"})
            for b in ladder:
                if concat is not None and len(concat) <= b:
                    continue
                tb, ranks = tr.encode(b, self.theta, norm_vals)
                trunc_info.append({"budget": b, "bytes": len(tb), "rank_total": sum(ranks.values()),
                                   "ranks": {n: r for n, r in ranks.items() if r}})
                if b == cap and blob is None:
                    blob = tb
                else:
                    self.snaps.append((tb, len(self.steps)))
            if blob is None and cap is not None:
                blob = tr.encode(cap, self.theta, norm_vals)[0]
        if blob is None:
            blob = prepare.encode_full({n: self.params[n].detach().to(self.theta[n].dtype)
                                        for n in self.targets + self.norms})
        model = prepare.decode(self.theta, blob)
        changed = sorted(n for n in self.theta if not prepare.same_bits(model[n], self.theta[n]))
        src = Path(__file__).read_bytes()
        notes = {"direction": "lowrank-restart", "mech": a.mech, "args": vars(a), "E_bytes": len(blob),
                 "concat_bytes": len(concat) if concat is not None else None, "truncations": trunc_info,
                 "program": "islands/lowrank-restart/lowrank_restart.py",
                 "program_sha256": hashlib.sha256(src).hexdigest(), "train_log": self.log[-20:], **notes}
        prepare.submit(Path(a.out), a.start, self.klass, blob, model, trainable, self.steps, changed=changed,
                       notes=notes, snapshots=self.snaps)
        print(json.dumps({"E_bytes": len(blob), "changed": len(changed), "snapshots": len(self.snaps),
                          "truncations": [(t["budget"], t["bytes"], t["rank_total"]) for t in trunc_info]}), flush=True)
        return 0

    def snapshot_pieces(self, pieces: dict) -> None:
        if self.a.no_piece_snaps:
            return
        b = encode_pieces(pieces, self.norm_vals())
        if self.cap is None or len(b) <= self.cap:
            self.snaps.append((b, len(self.steps)))


def run_relora(run: Run) -> int:
    a, params, targets = run.a, run.params, run.targets
    r, K = a.rank, a.pieces
    nopt = run.norm_opt()
    g = torch.Generator(device=DEV).manual_seed(a.seed)
    fac = {}
    for n in targets:
        w = params[n]
        A = torch.nn.Parameter(torch.zeros(r, w.shape[1], device=DEV))
        B = torch.nn.Parameter(torch.zeros(w.shape[0], r, device=DEV))
        s = a.alpha * math.sqrt(w.shape[1] / r)
        fac[n] = (A, B, s)
        mod = run.net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], A), B))

    def restart():
        with torch.no_grad():
            for A, B, _ in fac.values():
                A.copy_(torch.randn(A.shape, device=DEV, generator=g) / math.sqrt(A.shape[1]))
                B.zero_()
        return torch.optim.AdamW([p for A, B, _ in fac.values() for p in (A, B)], lr=a.lr, betas=(0.9, 0.95),
                                 weight_decay=0.0)

    trainable = {"full": run.norms, "adapters": {n: r for n in targets}}
    per_step = prepare.train_flops(run.rec, trainable, [{"op": "train", "tokens": run.tokens}])["flops"]
    merge = sum(2 * r * params[n].shape[0] * params[n].shape[1] for n in targets)
    trunc = sum(svd_flops(tuple(params[n].shape)) for n in targets)
    n_steps = run.n_steps(per_step, K * merge + trunc)
    bounds = [round(n_steps * j / K) for j in range(K + 1)]
    pieces = {n: [] for n in targets}
    fwd = torch.compile(run.net) if a.compile else run.net
    print(json.dumps({"mech": "relora", "steps": n_steps, "rank": r, "pieces": K, "targets": len(targets),
                      "norms": len(run.norms), "piece_bytes": lora_size(run.theta, {n: r for n in targets}, []),
                      "flops_per_step": per_step}), flush=True)
    opt, piece = restart(), 0
    for step in range(n_steps):
        frac = schedule(step, n_steps, a.warmup, bounds[piece], a.rewarm)
        for grp in opt.param_groups:
            grp["lr"] = a.lr * frac
        if nopt:
            for grp in nopt.param_groups:
                grp["lr"] = a.norm_lr * schedule(step, n_steps, a.warmup)
        loss = loss_backward(fwd, run.data.next(a.batch), a)
        torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp["params"]] +
                                       [params[n] for n in run.norms], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if nopt:
            nopt.step()
            nopt.zero_grad(set_to_none=True)
        run.steps.append({"op": "train", "tokens": run.tokens})
        run.report(step, loss, piece=piece)
        if step + 1 == bounds[piece + 1]:
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.no_grad():
                for n, (A, B, s) in fac.items():
                    aq, bq = A.detach().to(CODE["dtype"]), (s * B.detach()).to(CODE["dtype"])
                    params[n].data += bq.float() @ aq.float()
                    pieces[n].append((aq, bq))
            torch.backends.cuda.matmul.allow_tf32 = tf32
            run.steps.append({"op": "extra", "flops": merge, "what": f"merge piece {piece}"})
            piece += 1
            if piece < K:
                run.snapshot_pieces(pieces)
                opt = restart()
    for A, B, _ in fac.values():
        with torch.no_grad():
            B.zero_()
    return run.finish(pieces, trainable, {"pieces": K, "rank": r, "steps": n_steps})


def run_stack(run: Run) -> int:
    """K pieces of rank r kept as adapters on θ, never merged: every base W x stays θ's, and the forward adds the finished
    pieces (frozen, preallocated at rank K·r) and the active one. `--basis lora`: the active piece is a LoRA (ReLoRA's
    restarts without the merge, the same function). `--basis galore`: GaLore in adapter form; at each piece's first step one
    backward also takes the target matrices' gradient G, the active piece's frozen factor is G's top-r singular basis (P on
    the left when d_out <= d_in, Q on the right otherwise) and only the other factor N trains, so dL/dN = P^T G is GaLore's
    projected gradient and AdamW on N is its projected Adam."""
    a, params, targets = run.a, run.params, run.targets
    r, K, galore = a.rank, a.pieces, a.basis == "galore"
    nopt = run.norm_opt()
    g = torch.Generator(device=DEV).manual_seed(a.seed)
    st = {}
    for n in targets:
        d_out, d_in = params[n].shape
        left = d_out <= d_in
        s = 1.0 if galore else a.alpha * math.sqrt(d_in / r)
        A_all = torch.zeros(K * r, d_in, device=DEV)
        B_all = torch.zeros(d_out, K * r, device=DEV)
        A = torch.nn.Parameter(torch.zeros(r, d_in, device=DEV), requires_grad=not galore or left)
        B = torch.nn.Parameter(torch.zeros(d_out, r, device=DEV), requires_grad=not galore or not left)
        st[n] = {"A_all": A_all, "B_all": B_all, "A": A, "B": B, "s": s, "left": left}
        mod = run.net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, A_all=A_all, B_all=B_all, A=A, B=B, s=s:
                                  out + F.linear(F.linear(inp[0], A_all), B_all) + s * F.linear(F.linear(inp[0], A), B))
    active = [p for v in st.values() for p in (v["A"], v["B"]) if p.requires_grad]

    def restart():
        with torch.no_grad():
            for v in st.values():
                if galore:
                    v["A"].zero_()
                    v["B"].zero_()
                else:
                    v["A"].copy_(torch.randn(v["A"].shape, device=DEV, generator=g) / math.sqrt(v["A"].shape[1]))
                    v["B"].zero_()
        return torch.optim.AdamW(active, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    trainable = {"full": run.norms, "adapters": {n: (K + 1) * r for n in targets}}
    per_step = prepare.train_flops(run.rec, trainable, [{"op": "train", "tokens": run.tokens}])["flops"]
    wgrad = sum(2 * params[n].shape[0] * params[n].shape[1] for n in targets) * run.tokens if galore else 0
    svd = sum(svd_flops(tuple(params[n].shape)) for n in targets)
    n_steps = run.n_steps(per_step, K * (wgrad + (svd if galore else 0)) + svd)
    bounds = [round(n_steps * j / K) for j in range(K + 1)]
    fwd = torch.compile(run.net) if a.compile else run.net
    print(json.dumps({"mech": "stack", "basis": a.basis, "steps": n_steps, "rank": r, "pieces": K, "targets": len(targets),
                      "norms": len(run.norms), "piece_bytes": lora_size(run.theta, {n: r for n in targets}, []),
                      "flops_per_step": per_step}), flush=True)

    def pieces_so_far(j: int) -> dict:
        return {n: [(v["A_all"][: j * r].to(CODE["dtype"]), v["B_all"][:, : j * r].to(CODE["dtype"]))] if j else []
                for n, v in st.items()}

    opt, piece = restart(), 0
    for step in range(n_steps):
        refresh = galore and step == bounds[piece]
        frac = schedule(step, n_steps, a.warmup, bounds[piece], a.rewarm)
        for grp in opt.param_groups:
            grp["lr"] = a.lr * frac
        if nopt:
            for grp in nopt.param_groups:
                grp["lr"] = a.norm_lr * schedule(step, n_steps, a.warmup)
        if refresh:
            for n in targets:
                params[n].requires_grad_(True)
        loss = loss_backward(fwd, run.data.next(a.batch), a)
        if refresh:
            with torch.no_grad():
                for n, v in st.items():
                    u, _, vh = torch.linalg.svd(params[n].grad.float(), full_matrices=False)
                    if v["left"]:
                        v["B"].copy_(u[:, :r])
                    else:
                        v["A"].copy_(vh[:r])
                    params[n].grad = None
                    params[n].requires_grad_(False)
            run.steps.append({"op": "extra", "flops": wgrad + svd, "what": f"target weight gradients and SVDs, piece {piece}"})
            opt.zero_grad(set_to_none=True)
        else:
            torch.nn.utils.clip_grad_norm_(active + [params[n] for n in run.norms], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if nopt:
            nopt.step()
            nopt.zero_grad(set_to_none=True)
        run.steps.append({"op": "train", "tokens": run.tokens})
        run.report(step, loss, piece=piece)
        if step + 1 == bounds[piece + 1]:
            with torch.no_grad():
                for v in st.values():
                    sl = slice(piece * r, (piece + 1) * r)
                    v["A_all"][sl] = v["A"].to(CODE["dtype"]).float()
                    v["B_all"][:, sl] = (v["s"] * v["B"]).to(CODE["dtype"]).float()
            piece += 1
            if piece < K:
                run.snapshot_pieces(pieces_so_far(piece))
                opt = restart()
    with torch.no_grad():
        for n, v in st.items():
            v["A"].zero_()
            v["B"].zero_()
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            params[n].data = run.theta_dev[n].float() + v["B_all"] @ v["A_all"]
            torch.backends.cuda.matmul.allow_tf32 = tf32
    return run.finish(pieces_so_far(K), trainable, {"pieces": K, "rank": r, "basis": a.basis, "steps": n_steps,
                                                    "merged": False})


class GaloreState:
    def __init__(self, w: torch.Tensor, r: int):
        self.left = w.shape[0] <= w.shape[1]
        self.r = r
        self.P = None
        self.m = self.v = self.N = None
        self.t = 0

    def refresh(self, grad: torch.Tensor, keep: bool) -> None:
        u, _, vh = torch.linalg.svd(grad.float(), full_matrices=False)
        self.P = u[:, :self.r].contiguous() if self.left else vh[:self.r].T.contiguous()
        shape = (self.r, grad.shape[1]) if self.left else (grad.shape[0], self.r)
        if not keep or self.m is None:
            self.m, self.v, self.t = torch.zeros(shape, device=grad.device), torch.zeros(shape, device=grad.device), 0
        self.N = torch.zeros(shape, device=grad.device)

    def piece(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.left:
            return dense(self.N.to(CODE["dtype"])), dense(self.P.to(CODE["dtype"]))
        return dense(self.P.T.to(CODE["dtype"])), dense(self.N.to(CODE["dtype"]))

    @torch.no_grad()
    def step(self, w: torch.Tensor, grad: torch.Tensor, lr: float, b1=0.9, b2=0.95, eps=1e-8) -> None:
        R = self.P.T @ grad if self.left else grad @ self.P
        self.t += 1
        self.m.mul_(b1).add_(R, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(R, R, value=1 - b2)
        d = (self.m / (1 - b1 ** self.t)) / ((self.v / (1 - b2 ** self.t)).sqrt() + eps)
        d.mul_(-lr)
        self.N.add_(d)
        w.add_(self.P @ d if self.left else d @ self.P.T)


def run_galore(run: Run) -> int:
    a, params, targets = run.a, run.params, run.targets
    r, K = a.rank, a.pieces
    nopt = run.norm_opt()
    for n in targets:
        params[n].requires_grad_(True)
    st = {n: GaloreState(params[n], r) for n in targets}
    trainable = {"full": sorted(targets + run.norms)}
    per_step = prepare.train_flops(run.rec, trainable, [{"op": "train", "tokens": run.tokens}])["flops"]
    proj = sum(4 * r * params[n].shape[0] * params[n].shape[1] for n in targets)
    svd = sum(svd_flops(tuple(params[n].shape)) for n in targets)
    n_steps = run.n_steps(per_step + proj, (K + 1) * svd)
    bounds = [round(n_steps * j / K) for j in range(K + 1)]
    pieces = {n: [] for n in targets}
    fwd = torch.compile(run.net) if a.compile else run.net
    print(json.dumps({"mech": "galore", "steps": n_steps, "rank": r, "pieces": K, "targets": len(targets),
                      "norms": len(run.norms), "piece_bytes": lora_size(run.theta, {n: r for n in targets}, []),
                      "flops_per_step": per_step + proj}), flush=True)
    piece = 0
    for step in range(n_steps):
        frac = schedule(step, n_steps, a.warmup, bounds[piece], a.rewarm)
        loss = loss_backward(fwd, run.data.next(a.batch), a)
        torch.nn.utils.clip_grad_norm_([params[n] for n in targets + run.norms], 1.0)
        if step == bounds[piece]:
            for n in targets:
                st[n].refresh(params[n].grad, a.keep_moments)
            run.steps.append({"op": "extra", "flops": svd, "what": f"projector SVDs, piece {piece}"})
        for n in targets:
            st[n].step(params[n].data, params[n].grad, a.lr * frac)
            params[n].grad = None
        if nopt:
            for grp in nopt.param_groups:
                grp["lr"] = a.norm_lr * schedule(step, n_steps, a.warmup)
            nopt.step()
            nopt.zero_grad(set_to_none=True)
        run.steps += [{"op": "train", "tokens": run.tokens}, {"op": "extra", "flops": proj, "what": "projections"}]
        run.report(step, loss, piece=piece)
        if step + 1 == bounds[piece + 1]:
            for n in targets:
                pieces[n].append(st[n].piece())
            piece += 1
            if piece < K:
                run.snapshot_pieces(pieces)
    return run.finish(pieces, trainable, {"pieces": K, "rank": r, "steps": n_steps})


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


def ns_flops(shape, steps: int = 5) -> int:
    m, n = min(shape), max(shape)
    return steps * (4 * m * m * n + 2 * m ** 3)


def run_full(run: Run) -> int:
    a, params, targets = run.a, run.params, run.targets
    nopt = run.norm_opt()
    for n in targets:
        params[n].requires_grad_(True)
    trainable = {"full": sorted(targets + run.norms)}
    per_step = prepare.train_flops(run.rec, trainable, [{"op": "train", "tokens": run.tokens}])["flops"]
    muon_extra = sum(ns_flops(tuple(params[n].shape)) for n in targets) if a.opt == "muon" else 0
    svd = sum(svd_flops(tuple(params[n].shape)) for n in targets)
    est = run.n_steps(per_step + muon_extra, svd)
    n_proj = est // a.project_every if a.project_every else 0
    n_steps = run.n_steps(per_step + muon_extra, (n_proj + 1) * svd)
    if a.opt == "muon":
        bufs = {n: torch.zeros_like(params[n]) for n in targets}
    else:
        opt = torch.optim.AdamW([params[n] for n in targets], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    fwd = torch.compile(run.net) if a.compile else run.net
    print(json.dumps({"mech": "full", "opt": a.opt, "steps": n_steps, "targets": len(targets), "norms": len(run.norms),
                      "project_every": a.project_every, "project_bytes": a.project_bytes or run.cap,
                      "flops_per_step": per_step + muon_extra}), flush=True)
    for step in range(n_steps):
        frac = schedule(step, n_steps, a.warmup)
        loss = loss_backward(fwd, run.data.next(a.batch), a)
        torch.nn.utils.clip_grad_norm_([params[n] for n in targets + run.norms], 1.0)
        if a.opt == "muon":
            with torch.no_grad():
                for n in targets:
                    p = params[n]
                    buf = bufs[n].mul_(0.95).add_(p.grad)
                    p.add_(newton_schulz(p.grad.add(buf, alpha=0.95)),
                           alpha=-a.lr * frac * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)
                    p.grad = None
            run.steps.append({"op": "extra", "flops": muon_extra, "what": "muon newton-schulz"})
        else:
            for grp in opt.param_groups:
                grp["lr"] = a.lr * frac
            opt.step()
            opt.zero_grad(set_to_none=True)
        if nopt:
            for grp in nopt.param_groups:
                grp["lr"] = a.norm_lr * frac
            nopt.step()
            nopt.zero_grad(set_to_none=True)
        run.steps.append({"op": "train", "tokens": run.tokens})
        run.report(step, loss)
        if a.project_every and (step + 1) % a.project_every == 0 and step + 1 < n_steps:
            tr = Truncator(run.deltas())
            budget = a.project_bytes or run.cap
            ranks = tr.ranks(budget, lora_size(run.theta, {}, run.norms) + HEADER_MARGIN + 2048)
            fac = tr.factors(ranks)
            with torch.no_grad():
                for n in targets:
                    lo = (fac[n][1].to(CODE["dtype"]).float() @ fac[n][0].to(CODE["dtype"]).float()) if n in fac else 0
                    params[n].data.copy_(run.theta_dev[n].float() + lo)
            run.steps.append({"op": "extra", "flops": tr.flops, "what": "rank projection"})
    return run.finish(None, trainable, {"steps": n_steps, "opt": a.opt})


def run_subspace(run: Run) -> int:
    """Random-subspace descent with exact directional derivatives: each step one backward, then `--dirs` seed-log
    directions, each moved by c_i = lr · <G, z_i> / (|G| sqrt(P)) (quantized to f16), applied as the decoder applies it."""
    a, params = run.a, run.params
    names = sorted(run.targets + run.norms)
    for n in names:
        params[n].requires_grad_(True)
    P = sum(params[n].numel() for n in names)
    header = len(prepare.encode_seedlog(names, a.seed, [], "f16"))
    max_dirs = ((run.cap or 10 ** 9) - header) // 2
    trainable = {"full": names}
    per_step = prepare.train_flops(run.rec, trainable, [{"op": "train", "tokens": run.tokens}])["flops"]
    dir_flops = 4 * P
    n_steps = min(run.n_steps(per_step + a.dirs * dir_flops, 0), max_dirs // a.dirs)
    fwd = torch.compile(run.net) if a.compile else run.net
    gen = torch.Generator(device=DEV)
    masters = [params[n].data for n in names]
    coeffs: list[torch.Tensor] = []
    print(json.dumps({"mech": "subspace", "steps": n_steps, "dirs_per_step": a.dirs, "P": P, "names": len(names),
                      "flops_per_step": per_step + a.dirs * dir_flops}), flush=True)
    marks = {round(n_steps * k / 4) for k in (1, 2, 3)}
    for step in range(n_steps):
        frac = schedule(step, n_steps, a.warmup)
        loss = loss_backward(fwd, run.data.next(a.batch), a)
        grads = [params[n].grad for n in names]
        gnorm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
        with torch.no_grad():
            for _ in range(a.dirs):
                gen.manual_seed(a.seed + len(coeffs))
                zs = [torch.randn(w.shape, generator=gen, device=w.device, dtype=torch.float32) for w in masters]
                gi = sum(torch.dot(g.reshape(-1), z.reshape(-1)) for g, z in zip(grads, zs))
                c = (a.lr * frac * gi / (gnorm * math.sqrt(P) + 1e-12)).to(torch.float16).float()
                coeffs.append(c)
                for w, z in zip(masters, zs):
                    w.add_(z * (-c))
        for n in names:
            params[n].grad = None
        run.steps += [{"op": "train", "tokens": run.tokens},
                      {"op": "extra", "flops": a.dirs * dir_flops, "what": "directional derivatives and steps"}]
        run.report(step, loss)
        if step + 1 in marks:
            run.snaps.append((prepare.encode_seedlog(names, a.seed, torch.stack(coeffs).tolist(), "f16"),
                              len(run.steps)))
    blob = prepare.encode_seedlog(names, a.seed, torch.stack(coeffs).tolist(), "f16")
    model = prepare.decode(run.theta, blob)
    changed = sorted(n for n in run.theta if not prepare.same_bits(model[n], run.theta[n]))
    src = Path(__file__).read_bytes()
    prepare.submit(Path(a.out), a.start, run.klass, blob, model, trainable, run.steps, changed=changed,
                   notes={"direction": "lowrank-restart", "mech": "subspace", "args": vars(a), "E_bytes": len(blob),
                          "directions": len(coeffs), "steps": n_steps, "program_sha256": hashlib.sha256(src).hexdigest(),
                          "train_log": run.log[-20:]}, snapshots=run.snaps)
    print(json.dumps({"E_bytes": len(blob), "directions": len(coeffs), "changed": len(changed)}), flush=True)
    return 0


def main(klass: str) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mech", choices=("relora", "stack", "galore", "full", "subspace"), required=True)
    ap.add_argument("--basis", choices=("lora", "galore"), default="lora", help="stack: the active piece's form")
    ap.add_argument("--targets", default="all", help="all, out, mlp, attn, or leaf names (wq,wo,...)")
    ap.add_argument("--blocks", default="all", help="all, or LO-HI (0-based, inclusive)")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--pieces", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0, help="relora: the scale's multiplier on sqrt(d_in / r)")
    ap.add_argument("--norms", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--norm-lr", type=float, default=3e-4)
    ap.add_argument("--opt", choices=("adam", "muon"), default="adam")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rewarm", type=int, default=20, help="relora/galore: steps of warmup after each restart")
    ap.add_argument("--keep-moments", action="store_true", help="galore: keep the projected moments across refreshes")
    ap.add_argument("--project-every", type=int, default=0, help="full: project W − θ to the byte budget every T steps")
    ap.add_argument("--project-bytes", type=int, default=0)
    ap.add_argument("--whiten", type=int, default=0, help="truncate by output error, from this many batches' inputs")
    ap.add_argument("--ladder", default="", help="byte budgets for truncation snapshots of the final update")
    ap.add_argument("--no-piece-snaps", action="store_true")
    ap.add_argument("--dirs", type=int, default=64, help="subspace: directions per step")
    ap.add_argument("--dtype", choices=("f16", "bf16"), default="f16", help="E's factors and norm gains")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    CODE.update(name=a.dtype, dtype=prepare.DTYPES[a.dtype])
    run = Run(a, klass)
    return {"relora": run_relora, "stack": run_stack, "galore": run_galore, "full": run_full,
            "subspace": run_subspace}[a.mech](run)
