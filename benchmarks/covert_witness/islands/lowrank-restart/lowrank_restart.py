"""Direction `lowrank-restart`: updates built from many low-rank pieces or projections, run by this island's `train.py`,
E within `--max-e-bytes` (0: no cap); E always encodes the whole update from θ.

Mechanisms (`--mech`):

- `relora`: K pieces of rank r. Each piece is a LoRA (A ~ N(0, 1/d_in), B = 0, scale sqrt(d_in / r)) trained by AdamW
  for its share of the steps, then merged (its factors rounded to E's dtype, the scale folded into B), and the factors and
  their optimizer state restart. K = 1 is plain LoRA. A merge rebuilds each weight from θ with every piece so far
  (`prepare.lora_merge`), so the running weights are decode(θ, E_j) bit for bit. Its steps can't be declared: each also
  reads the active piece's factors, new tensors that no fixed format decodes.
- `stack`: the same K pieces kept as adapters on θ, never merged, so every base W x stays θ's (`--basis lora`: ReLoRA's
  function; `--basis galore`: GaLore in adapter form, each piece's frozen factor the top-r singular basis of the
  gradient at its first step).
- `galore`: every target matrix is trained in full through a rank-r projected Adam (GaLore; `--opt muon`: Muon's
  orthogonalized momentum step on the projected gradient instead). The projector is the top-r singular basis of the
  gradient, refreshed K times, so each period's update is exactly P_j @ N_j, one rank-r piece (`--alloc 1`: each
  refresh gives the tensors ranks greedily by the gradient's σ² per byte, as many bytes as rank r on every tensor).
  After every step each W is rebuilt from θ with the pieces so far as E stores them (N's and the norm gains' fp32
  masters kept apart), so the next step reads decode(θ, E_t) bit for bit and declares E_t (`prepare.declare_step`):
  merged steps the harness credits as a counterexample family.
- `full`: AdamW (or Muon) on the target matrices; `--project-every T` projects W − θ onto its best fitting low-rank
  approximation every T steps (rank-constrained training by projection).
- `subspace`: exact-gradient descent in a random subspace whose basis comes from seeds: each direction z_i is the seed
  log's (`prepare.seedlog_step`), its coefficient is the backward gradient's projection on z_i, and E is only the
  coefficients (`prepare.encode_seedlog`). Each step declares the log so far.

E for the piece mechanisms is the pieces concatenated (rank K·r, `prepare.encode_lora`) when they fit `--max-e-bytes`,
else the truncated SVD of the whole update W − θ with ranks allocated across tensors greedily by singular value energy
per byte (`--whiten`: of the update times each input's covariance root, so the energy is the output error it removes).

`--arms` runs several settings in one candidate, each from θ on the same data order with its share of the budget, e.g.
`--arms lr=1e-3/lr=3e-3/lr=1e-2` (a 3-point learning-rate sweep) or `--arms pieces=1,rank=8/pieces=8,rank=1`. The harness
scores the first E encoded in each 5% of the planned steps (`hooks.plan`), so each arm encodes E at each piece boundary
whose 5% is still free and at its end; the arm with the least final training loss is submitted. A later arm's
checkpoints are charged every arm before it.

    python benchmarks/covert_witness/launch.py s-t20 --island lowrank-restart --note "stack r2x4 lr sweep" -- \
        --max-e-bytes 1000000 --mech stack --rank 2 --pieces 4 --arms lr=1e-3/lr=3e-3/lr=1e-2 --budget-fraction 0.6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare
from te_core import hooks

LF_MATS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
GROUPS = {"all": LF_MATS, "out": ("wo", "w_down"), "mlp": ("w_gate", "w_up", "w_down"), "attn": ("wq", "wk", "wv", "wo")}
DEV = prepare.DEVICE
#: the harness scores the first E encoded in each 1/CHECKPOINTS of the planned steps (its run.py's CHECKPOINTS)
CHECKPOINTS = 20
#: the factors' and full tensors' storage in E (`--dtype`); f16 keeps 4x bf16's mantissa for norm gains near 1
CODE = {"name": "f16", "dtype": torch.float16}
ARM_KEYS = {"mech": str, "lr": float, "rank": int, "pieces": int, "alpha": float, "norm_lr": float, "warmup": int, "rewarm": int,
            "dirs": int, "targets": str, "blocks": str, "basis": str, "opt": str, "project_every": int, "alloc": int}


# ---- targets and byte accounting --------------------------------------------------------------------------------------

def targets_of(params: dict, spec: str, blocks: str) -> list[str]:
    leaves = set(GROUPS.get(spec, spec.split("+")))
    n_layer = 1 + max(int(n.split(".")[1]) for n in params if n.startswith("blocks."))
    lo, hi = (0, n_layer - 1) if blocks == "all" else map(int, blocks.split("-"))
    out = []
    for n, p in params.items():
        parts = n.split(".")
        if p.ndim == 2 and parts[0] == "blocks" and lo <= int(parts[1]) <= hi and parts[2] in leaves:
            out.append(n)
    return sorted(out)


def lora_bytes(ranks: dict[str, int], shapes: dict[str, tuple], norm_vals: dict) -> int:
    """len(encode_lora(...)) for these ranks, without encoding: the harness scores every E encoded during the steps."""
    names = sorted(n for n, r in ranks.items() if r)
    header = {"dtype": CODE["name"], "factors": [[n, int(ranks[n]), 1.0] for n in names], "full": sorted(norm_vals)}
    entries = sum(ranks[n] * (shapes[n][0] + shapes[n][1]) for n in names) + sum(v.numel() for v in norm_vals.values())
    return len(prepare._pack(b"CWLORA1\n", header, b"")) + entries * torch.finfo(CODE["dtype"]).bits // 8


def seedlog_bytes(names: list[str], seed: int, steps: int) -> int:
    header = {"names": list(names), "seed": int(seed), "steps": int(steps), "coeff": "f16"}
    return len(prepare._pack(b"CWSEED1\n", header, b"")) + 2 * steps


def dense(x: torch.Tensor) -> torch.Tensor:
    """x with row-major strides: an SVD factor of rank 1 can pass `is_contiguous` with a last stride other than 1, which
    `encode_lora`'s byte view refuses."""
    return torch.empty(x.shape, dtype=x.dtype, device=x.device).copy_(x)


def svd_flops(shape) -> int:
    m, n = shape
    return 4 * m * n * min(m, n) + 4 * min(m, n) ** 3


def no_tf32_matmul(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return b.float() @ a.float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


# ---- encodings --------------------------------------------------------------------------------------------------------

def concat_factors(pieces: dict[str, list]) -> dict:
    """Every piece of every tensor, concatenated: W' = θ + [B_1 .. B_K] @ [A_1; ..; A_K]."""
    return {n: (torch.cat([a for a, _ in ps], 0), torch.cat([b for _, b in ps], 1), 1.0) for n, ps in pieces.items() if ps}


class Truncator:
    """The truncated SVD of each tensor's update, and the greedy allocation of ranks to a byte budget."""

    def __init__(self, deltas: dict[str, torch.Tensor], whiten: dict[str, torch.Tensor] | None = None):
        self.svd, self.cost, self.shapes, self.flops = {}, {}, {}, 0
        for n, d in deltas.items():
            d = d.float()
            if whiten and n in whiten:
                L = whiten[n]
                # L, computed from this step's tokens, goes first: the counter takes the first token-derived operand's
                # position with list.index, which compares tensors by value and fails when it isn't the first input
                u, s, vh = torch.linalg.svd((L.T @ d.T).T, full_matrices=False)
                vh = torch.linalg.solve_triangular(L, vh, upper=False, left=False)
                self.flops += 2 * d.shape[0] * d.shape[1] ** 2 + 2 * d.shape[1] ** 2 * min(d.shape)
            else:
                u, s, vh = torch.linalg.svd(d, full_matrices=False)
            self.svd[n] = (u, s, vh)
            self.shapes[n] = tuple(d.shape)
            self.cost[n] = 2 * (d.shape[0] + d.shape[1])
            self.flops += svd_flops(tuple(d.shape))
        self.order = sorted(((float(s[i]) ** 2 / self.cost[n], n, i) for n, (_, s, _) in self.svd.items()
                             for i in range(len(s))), reverse=True)

    def ranks(self, budget: int, norm_vals: dict) -> dict[str, int]:
        """The greedy allocation, then the least valuable rank dropped until the exact size fits."""
        ranks = {n: 0 for n in self.svd}
        used = lora_bytes({n: 9999 for n in self.svd}, {n: (0, 0) for n in self.svd}, norm_vals)
        for _, n, _ in self.order:
            if used + self.cost[n] <= budget:
                ranks[n] += 1
                used += self.cost[n]
        while lora_bytes(ranks, self.shapes, norm_vals) > budget:
            worst = min((float(self.svd[n][1][r - 1]) ** 2 / self.cost[n], n) for n, r in ranks.items() if r)[1]
            ranks[worst] -= 1
        return ranks

    def factors(self, ranks: dict[str, int]) -> dict:
        out = {}
        for n, r in ranks.items():
            if r:
                u, s, vh = self.svd[n]
                root = s[:r].sqrt()
                out[n] = (dense(root[:, None] * vh[:r]), dense(u[:, :r] * root[None, :]), 1.0)
        return out


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
def covariance_roots(run, a) -> dict[str, torch.Tensor]:
    """The Cholesky factor of each target's input second moment over `--whiten` batches (plus a ridge of 1e-4 of its mean
    diagonal), from forward passes of the current model."""
    acc = {n: None for n in run.targets}
    handles = []
    for n in run.targets:
        mod = run.net.inner.get_submodule(n.removesuffix(".weight"))

        def pre(m, inp, n=n):
            with torch.autocast(DEV.split(":")[0], enabled=False):
                x = inp[0].float().reshape(-1, inp[0].shape[-1])
                c = x.T @ x
            acc[n] = c if acc[n] is None else acc[n] + c
        handles.append(mod.register_forward_pre_hook(pre))
    tokens = 0
    for _ in range(a.whiten):
        rows = run.next_batch()
        for i in range(0, a.batch, a.micro):
            with torch.autocast(DEV.split(":")[0], dtype=torch.bfloat16):
                run.net(rows[i: i + a.micro, :-1])
        tokens += run.tokens
        run.steps.append({"op": "forward", "tokens": run.tokens})
    for h in handles:
        h.remove()
    run.steps.append({"op": "extra", "flops": sum(2 * tokens * acc[n].shape[0] ** 2 for n in run.targets),
                      "what": "input second moments for whitened truncation"})
    roots = {}
    for n in run.targets:
        c = acc[n] / tokens
        c = c + 1e-4 * c.diagonal().mean() * torch.eye(c.shape[0], device=c.device)
        roots[n] = torch.linalg.cholesky(c.double()).float()
    return roots


# ---- the run: arms, checkpoints, submission ---------------------------------------------------------------------------

class Run:
    def __init__(self, base, arms: list[argparse.Namespace]):
        self.base, self.arms = base, arms
        self.net, self.rec, self.theta = prepare.load_start(base.start)
        self.params = self.net.params()
        for p in self.params.values():
            p.requires_grad_(False)
        self.cap = base.max_e_bytes or None
        self.theta_dev = {n: self.theta[n].to(DEV) for n in self.params}
        self.tokens = base.batch * prepare.EVAL_CTX
        self.steps, self.log, self.results = [], [], []
        self.k, self.n_total, self.used, self.handles = 0, None, set(), []
        self.t0 = time.perf_counter()

    # -- an arm's setup

    def begin(self, a) -> None:
        """θ into every tensor, the hooks of the arm before removed, and the data stream from its start."""
        for h in self.handles:
            h.remove()
        self.handles = []
        with torch.no_grad():
            for n, p in self.params.items():
                p.grad = None
                p.requires_grad_(False)
                p.data.copy_(self.theta_dev[n])
        self.a = a
        self.targets = targets_of(self.params, a.targets, a.blocks)
        self.norms = sorted(n for n, p in self.params.items() if p.ndim == 1) if a.norms else []
        self.data = prepare.TrainData(self.rec["tokenizer"], seed=a.data_seed)
        self.losses = []
        self.arm_start = self.k

    def hook(self, name: str, fn) -> None:
        mod = self.net.inner.get_submodule(name.removesuffix(".weight"))
        self.handles.append(mod.register_forward_hook(fn))

    def norm_vals(self) -> dict:
        return {n: self.params[n].detach() for n in self.norms}

    def norm_opt(self):
        for n in self.norms:
            self.params[n].requires_grad_(True)
        return torch.optim.AdamW([self.params[n] for n in self.norms], lr=self.a.norm_lr, betas=(0.9, 0.95),
                                 weight_decay=0.0) if self.norms else None

    def shapes(self) -> dict:
        return {n: tuple(self.params[n].shape) for n in self.targets}

    # -- steps and the harness's 5% checkpoints

    def plan(self, n_steps: int) -> None:
        """The arm's step count; the first arm's plans the run: every arm takes as many batches."""
        self.arm_end = self.arm_start + n_steps + self.a.whiten
        if self.n_total is None:
            self.n_total = len(self.arms) * (n_steps + self.a.whiten)
            hooks.plan(self.n_total)

    def next_batch(self) -> torch.Tensor:
        self.k += 1
        return self.data.next(self.a.batch)

    def bucket(self, k: int | None = None) -> int:
        return math.ceil((self.k if k is None else k) * CHECKPOINTS / self.n_total)

    def free(self) -> bool:
        """Whether an E encoded now is scored without taking the 5% the arm's final E will be encoded in."""
        return self.bucket() not in self.used and self.bucket() != self.bucket(self.arm_end)

    def encoded(self) -> None:
        self.used.add(self.bucket())

    def n_steps(self, per_step: float, fixed_extra: float) -> int:
        return int((self.a.budget_fraction / len(self.arms) * prepare.TRAIN_FLOP_BUDGET - fixed_extra) // per_step)

    def report(self, step: int, loss: torch.Tensor, **kw) -> None:
        self.losses.append(loss)
        if (step + 1) % 100 == 0:
            row = {"arm": len(self.results), "step": step + 1, "loss": round(float(loss), 4),
                   "s": round(time.perf_counter() - self.t0, 1), **kw}
            self.log.append(row)
            print(json.dumps(row), flush=True)

    def snapshot_pieces(self, pieces: dict) -> None:
        if self.a.no_piece_snaps or not self.free():
            return
        fac = concat_factors(pieces)
        nv = self.norm_vals()
        if self.cap is None or lora_bytes({n: f[0].shape[0] for n, f in fac.items()}, self.shapes(), nv) <= self.cap:
            prepare.encode_lora(fac, CODE["name"], nv)
            self.encoded()

    # -- an arm's E

    def finish(self, pieces: dict | None, deltas, trainable: dict, notes: dict) -> None:
        """E: the pieces when they fit the cap, else the truncation of the update (`deltas()`) to the cap, else (no cap)
        the full tensors."""
        a, cap = self.a, self.cap
        nv = self.norm_vals()
        whiten = covariance_roots(self, a) if a.whiten else None
        fac = concat_factors(pieces) if pieces else None
        concat = lora_bytes({n: f[0].shape[0] for n, f in fac.items()}, self.shapes(), nv) if fac else None
        trunc = None
        if fac is not None and (cap is None or concat <= cap):
            blob = prepare.encode_lora(fac, CODE["name"], nv)
        elif cap is not None:
            tr = Truncator(deltas(), whiten)
            self.steps.append({"op": "extra", "flops": tr.flops, "what": "SVD of the update for truncation"})
            ranks = tr.ranks(cap, nv)
            blob = prepare.encode_lora(tr.factors(ranks), CODE["name"], nv)
            trunc = {"rank_total": sum(ranks.values()), "ranks": {n: r for n, r in ranks.items() if r}}
        else:
            blob = prepare.encode_full({n: self.params[n].detach().to(self.theta[n].dtype)
                                        for n in self.targets + self.norms})
        self.encoded()
        self.close_arm(blob, trainable, {"concat_bytes": concat, "truncation": trunc, **notes})

    def close_arm(self, blob: bytes, trainable: dict, notes: dict) -> None:
        losses = torch.stack(self.losses).float().cpu()
        tail = float(losses[-max(1, len(losses) // 10):].mean())
        arm = {"arm": len(self.results), "args": {k: getattr(self.a, k) for k in ARM_KEYS}, "E_bytes": len(blob),
               "E_sha256": hashlib.sha256(blob).hexdigest(), "steps": self.k - self.arm_start, "end_batch": self.k,
               "tail_loss": round(tail, 5), **notes}
        print(json.dumps(arm), flush=True)
        self.results.append({"blob": blob, "trainable": trainable, "row": arm})

    def submit(self) -> int:
        best = min(self.results, key=lambda r: r["row"]["tail_loss"])
        blob = best["blob"]
        model = prepare.decode(self.theta, blob)
        changed = sorted(n for n in self.theta if not prepare.same_bits(model[n], self.theta[n]))
        src = Path(__file__).read_bytes()
        notes = {"direction": "lowrank-restart", "mech": self.base.mech, "args": vars(self.base),
                 "submitted_arm": best["row"]["arm"], "arms": [r["row"] for r in self.results], "E_bytes": len(blob),
                 "program": "islands/lowrank-restart/lowrank_restart.py",
                 "program_sha256": hashlib.sha256(src).hexdigest(), "train_log": self.log[-40:]}
        prepare.submit(Path(self.base.out), self.base.start, blob, model, best["trainable"], self.steps,
                       changed=changed, notes=notes)
        print(json.dumps({"submitted_arm": best["row"]["arm"], "E_bytes": len(blob), "changed": len(changed)}), flush=True)
        return 0


# ---- mechanisms -------------------------------------------------------------------------------------------------------

def run_relora(run: Run) -> None:
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
        run.hook(n, lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], A), B))

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
    n_steps = run.n_steps(per_step, K * (K + 1) // 2 * merge + trunc)
    run.plan(n_steps)
    bounds = [round(n_steps * j / K) for j in range(K + 1)]
    pieces = {n: [] for n in targets}
    fwd = torch.compile(run.net) if a.compile else run.net
    print(json.dumps({"mech": "relora", "arm": len(run.results), "lr": a.lr, "steps": n_steps, "rank": r, "pieces": K,
                      "targets": len(targets), "norms": len(run.norms), "flops_per_step": per_step}), flush=True)
    opt, piece = restart(), 0
    for step in range(n_steps):
        frac = schedule(step, n_steps, a.warmup, bounds[piece], a.rewarm)
        for grp in opt.param_groups:
            grp["lr"] = a.lr * frac
        if nopt:
            for grp in nopt.param_groups:
                grp["lr"] = a.norm_lr * schedule(step, n_steps, a.warmup)
        loss = loss_backward(fwd, run.next_batch(), a)
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
            with torch.no_grad():
                for n, (A, B, s) in fac.items():
                    pieces[n].append((A.detach().to(CODE["dtype"]), (s * B.detach()).to(CODE["dtype"])))
                so_far = concat_factors(pieces)
                for n, (aq, bq, _) in so_far.items():
                    params[n].data.copy_(prepare.lora_merge(run.theta_dev[n], aq, bq, 1.0))
            run.steps.append({"op": "extra", "flops": (piece + 1) * merge, "what": f"merge pieces 0-{piece} from θ"})
            piece += 1
            if piece < K:
                run.snapshot_pieces(pieces)
                opt = restart()
    with torch.no_grad():
        for _, B, _ in fac.values():
            B.zero_()
    run.finish(pieces, run_deltas(run), trainable, {"pieces": K, "rank": r, "steps": n_steps, "merged": True})


def run_deltas(run: Run):
    return lambda: {n: run.params[n].detach().float() - run.theta_dev[n].float() for n in run.targets}


def run_stack(run: Run) -> None:
    """K pieces of rank r kept as adapters on θ, never merged: every base W x stays θ's, and the forward adds the finished
    pieces (frozen; only those finished are multiplied, so the updated MACs grow piece by piece) and the active one. `--basis lora`: the active piece is a LoRA (ReLoRA's
    restarts without the merge, the same function). `--basis galore`: GaLore in adapter form; at each piece's first step one
    backward also takes the target matrices' gradient G, the active piece's frozen factor is G's top-r singular basis (P on
    the left when d_out <= d_in, Q on the right otherwise) and only the other factor N trains, so dL/dN = P^T G is GaLore's
    projected gradient and AdamW on N is its projected Adam."""
    a, params, targets = run.a, run.params, run.targets
    r, K, galore = a.rank, a.pieces, a.basis == "galore"
    nopt = run.norm_opt()
    g = torch.Generator(device=DEV).manual_seed(a.seed)
    st, done = {}, [0]

    def forward_hook(A_all, B_all, A, B, s):
        def fn(m, inp, out):
            out = out + s * F.linear(F.linear(inp[0], A), B)
            if done[0]:
                out = out + F.linear(F.linear(inp[0], A_all[: done[0] * r]), B_all[:, : done[0] * r])
            return out
        return fn

    for n in targets:
        d_out, d_in = params[n].shape
        left = d_out <= d_in
        s = 1.0 if galore else a.alpha * math.sqrt(d_in / r)
        A_all = torch.zeros(K * r, d_in, device=DEV)
        B_all = torch.zeros(d_out, K * r, device=DEV)
        A = torch.nn.Parameter(torch.zeros(r, d_in, device=DEV), requires_grad=not galore or left)
        B = torch.nn.Parameter(torch.zeros(d_out, r, device=DEV), requires_grad=not galore or not left)
        st[n] = {"A_all": A_all, "B_all": B_all, "A": A, "B": B, "s": s, "left": left}
        run.hook(n, forward_hook(A_all, B_all, A, B, s))
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
    run.plan(n_steps)
    bounds = [round(n_steps * j / K) for j in range(K + 1)]
    fwd = torch.compile(run.net) if a.compile else run.net
    print(json.dumps({"mech": "stack", "basis": a.basis, "arm": len(run.results), "lr": a.lr, "steps": n_steps, "rank": r,
                      "pieces": K, "targets": len(targets), "norms": len(run.norms), "flops_per_step": per_step}),
          flush=True)

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
        loss = loss_backward(fwd, run.next_batch(), a)
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
            done[0] = piece
            if piece < K:
                run.snapshot_pieces(pieces_so_far(piece))
                opt = restart()
    run.finish(pieces_so_far(K), lambda: {n: no_tf32_matmul(v["B_all"], v["A_all"]) for n, v in st.items()}, trainable,
               {"pieces": K, "rank": r, "basis": a.basis, "steps": n_steps, "merged": False})


class GaloreState:
    def __init__(self, w: torch.Tensor, r: int):
        self.left = w.shape[0] <= w.shape[1]
        self.r = r
        self.scale = max(1.0, w.shape[0] / w.shape[1]) ** 0.5
        self.P = None
        self.m = self.v = self.N = None
        self.t = 0

    def refresh(self, svd: tuple, r: int, keep: bool) -> None:
        u, _, vh = svd
        keep = keep and r == self.r
        self.r = r
        self.P = u[:, :r].contiguous() if self.left else vh[:r].T.contiguous()
        shape = (r, vh.shape[1]) if self.left else (u.shape[0], r)
        if not keep or self.m is None:
            self.m, self.v, self.t = torch.zeros(shape, device=u.device), torch.zeros(shape, device=u.device), 0
        self.N = torch.zeros(shape, device=u.device)

    def piece(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.left:
            return dense(self.N.to(CODE["dtype"])), dense(self.P.to(CODE["dtype"]))
        return dense(self.P.T.to(CODE["dtype"])), dense(self.N.to(CODE["dtype"]))

    @torch.no_grad()
    def step(self, grad: torch.Tensor, lr: float, b1=0.9, b2=0.95, eps=1e-8, muon: bool = False) -> None:
        if not self.r:
            return
        R = self.P.T @ grad if self.left else grad @ self.P
        if muon:
            # P has orthonormal columns, so P @ dN has dN's spectral norm: a full-matrix Muon step's size at the same lr
            self.m.mul_(0.95).add_(R)
            self.N.add_(newton_schulz(R.add(self.m, alpha=0.95)), alpha=-lr * self.scale)
            return
        self.t += 1
        self.m.mul_(b1).add_(R, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(R, R, value=1 - b2)
        d = (self.m / (1 - b1 ** self.t)) / ((self.v / (1 - b2 ** self.t)).sqrt() + eps)
        self.N.add_(d, alpha=-lr)


def allocate(svds: dict, r: int) -> dict[str, int]:
    """One piece's ranks: greedy by the gradient's σ² per byte of factors, within the bytes of rank r on every tensor."""
    cost = {n: 2 * (u.shape[0] + vh.shape[1]) for n, (u, _, vh) in svds.items()}
    budget = r * sum(cost.values())
    order = sorted(((s_i ** 2 / cost[n], n) for n, (_, s, _) in svds.items() for s_i in s.cpu().tolist()), reverse=True)
    ranks = dict.fromkeys(svds, 0)
    for _, n in order:
        if cost[n] <= budget:
            ranks[n] += 1
            budget -= cost[n]
    return ranks


def run_galore(run: Run) -> None:
    a, params, targets = run.a, run.params, run.targets
    r, K = a.rank, a.pieces
    nopt = run.norm_opt()
    for n in targets:
        params[n].requires_grad_(True)
    st = {n: GaloreState(params[n], r) for n in targets}
    masters = {n: params[n].detach().clone() for n in run.norms}
    trainable = {"full": sorted(targets + run.norms)}
    per_step = prepare.train_flops(run.rec, trainable, [{"op": "train", "tokens": run.tokens}])["flops"]
    proj = sum(4 * r * params[n].shape[0] * params[n].shape[1] for n in targets)
    svd = sum(svd_flops(tuple(params[n].shape)) for n in targets)
    rebuild = sum(2 * K * r * params[n].shape[0] * params[n].shape[1] for n in targets)
    if a.alloc:
        # allocated ranks can sit where a byte buys the most MACs: plan for the worst tensor's MACs per byte
        worst = max(params[n].shape[0] * params[n].shape[1] / sum(params[n].shape) for n in targets)
        uniform = sum(params[n].shape[0] * params[n].shape[1] for n in targets) / sum(sum(params[n].shape) for n in targets)
        proj, rebuild = proj * worst / uniform, rebuild * worst / uniform
    muon = a.opt == "muon"
    ns = sum(ns_flops((r, max(params[n].shape))) for n in targets) if muon else 0
    proj += ns
    n_steps = run.n_steps(per_step + proj + rebuild, (K + 1) * svd)
    run.plan(n_steps)
    bounds = [round(n_steps * j / K) for j in range(K + 1)]
    pieces = {n: [] for n in targets}
    fwd = torch.compile(run.net) if a.compile else run.net
    print(json.dumps({"mech": "galore", "opt": a.opt, "arm": len(run.results), "lr": a.lr, "steps": n_steps, "rank": r, "pieces": K,
                      "targets": len(targets), "norms": len(run.norms), "flops_per_step": per_step + proj + rebuild}),
          flush=True)
    piece, base = 0, b""
    for step in range(n_steps):
        frac = schedule(step, n_steps, a.warmup, bounds[piece], a.rewarm)
        rows = run.next_batch()
        if base:
            prepare.declare_step(base)
        loss = loss_backward(fwd, rows, a)
        torch.nn.utils.clip_grad_norm_([params[n] for n in targets + run.norms], 1.0)
        if step == bounds[piece]:
            svds = {n: torch.linalg.svd(params[n].grad.float(), full_matrices=False) for n in targets}
            ranks = allocate(svds, r) if a.alloc else dict.fromkeys(targets, r)
            for n in targets:
                st[n].refresh(svds[n], ranks[n], a.keep_moments)
            run.steps.append({"op": "extra", "flops": svd, "what": f"projector SVDs, piece {piece}"})
            if a.alloc:
                print(json.dumps({"arm": len(run.results), "piece": piece, "ranks": {n: k for n, k in ranks.items() if k}}),
                      flush=True)
        for n in targets:
            st[n].step(params[n].grad, a.lr * frac, muon=muon)
            params[n].grad = None
        with torch.no_grad():
            if nopt:
                for grp in nopt.param_groups:
                    grp["lr"] = a.norm_lr * schedule(step, n_steps, a.warmup)
                for n in run.norms:
                    params[n].data.copy_(masters[n])
                nopt.step()
                nopt.zero_grad(set_to_none=True)
                for n in run.norms:
                    masters[n].copy_(params[n].data)
                    params[n].data.copy_(masters[n].to(CODE["dtype"]))
            if step + 1 == bounds[piece + 1]:
                for n in targets:
                    if st[n].r:
                        pieces[n].append(st[n].piece())
                piece += 1
                current = pieces
            else:
                current = {n: pieces[n] + ([st[n].piece()] if st[n].r else []) for n in targets}
            # each step reads W = decode(θ, E_t) bit for bit: rebuilt from θ with the factors as E stores them
            so_far = concat_factors(current)
            for n, (aq, bq, _) in so_far.items():
                params[n].data.copy_(prepare.lora_merge(run.theta_dev[n], aq, bq, 1.0))
        run.steps += [{"op": "train", "tokens": run.tokens}, {"op": "extra", "flops": proj, "what": "projections"},
                      {"op": "extra", "flops": rebuild, "what": "rebuild from θ"}]
        run.report(step, loss, piece=piece)
        if step + 1 < n_steps:
            base = prepare.encode_lora(so_far, CODE["name"], run.norm_vals())
            run.used.add(run.bucket())
    run.finish(pieces, run_deltas(run), trainable, {"pieces": K, "rank": r, "steps": n_steps, "declared": True})


def newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """bf16 on the GPU; fp32 on the CPU, where bf16 inside a training process now and then blew up (CPU smokes)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16() if g.is_cuda else g.float()
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


def run_full(run: Run) -> None:
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
    run.plan(n_steps)
    if a.opt == "muon":
        bufs = {n: torch.zeros_like(params[n]) for n in targets}
    else:
        opt = torch.optim.AdamW([params[n] for n in targets], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    fwd = torch.compile(run.net) if a.compile else run.net
    print(json.dumps({"mech": "full", "opt": a.opt, "arm": len(run.results), "lr": a.lr, "steps": n_steps,
                      "targets": len(targets), "norms": len(run.norms), "project_every": a.project_every,
                      "project_bytes": a.project_bytes or run.cap, "flops_per_step": per_step + muon_extra}), flush=True)
    for step in range(n_steps):
        frac = schedule(step, n_steps, a.warmup)
        loss = loss_backward(fwd, run.next_batch(), a)
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
            tr = Truncator(run_deltas(run)())
            fac = tr.factors(tr.ranks(a.project_bytes or run.cap, run.norm_vals()))
            with torch.no_grad():
                for n in targets:
                    lo = no_tf32_matmul(fac[n][1].to(CODE["dtype"]), fac[n][0].to(CODE["dtype"])) if n in fac else 0
                    params[n].data.copy_(run.theta_dev[n].float() + lo)
            run.steps.append({"op": "extra", "flops": tr.flops, "what": "rank projection"})
    run.finish(None, run_deltas(run), trainable, {"steps": n_steps, "opt": a.opt})


def run_subspace(run: Run) -> None:
    """Random-subspace descent with exact directional derivatives: each step one backward, then `--dirs` seed-log
    directions, each moved by c_i = lr · <G, z_i> / (|G| sqrt(P)) (quantized to f16) with the decoder's own op, so each
    step reads decode(θ, the log so far) bit for bit and declares it. The log is cut where its rebuild (P MACs a
    coefficient) would pass half of c, beyond which the harness can't credit a step."""
    a, params = run.a, run.params
    names = sorted(run.targets + run.norms)
    for n in names:
        params[n].requires_grad_(True)
    P = sum(params[n].numel() for n in names)
    c_macs = 4096 * prepare.train_flops(run.rec, {}, [{"op": "forward", "tokens": 1}])["flops"] / 2
    max_dirs = min(((run.cap or 10 ** 9) - seedlog_bytes(names, a.seed, 0)) // 2, int(c_macs / 2 // P))
    trainable = {"full": names}
    per_step = prepare.train_flops(run.rec, trainable, [{"op": "train", "tokens": run.tokens}])["flops"]
    dir_flops = 4 * P
    n_steps = min(run.n_steps(per_step + a.dirs * dir_flops, 0), max_dirs // a.dirs)
    run.plan(n_steps)
    fwd = torch.compile(run.net) if a.compile else run.net
    gen = torch.Generator(device=DEV)
    masters = [params[n].data for n in names]
    coeffs: list[float] = []
    print(json.dumps({"mech": "subspace", "arm": len(run.results), "lr": a.lr, "steps": n_steps, "dirs_per_step": a.dirs,
                      "P": P, "names": len(names), "max_dirs": max_dirs, "flops_per_step": per_step + a.dirs * dir_flops}),
          flush=True)
    base = b""
    for step in range(n_steps):
        frac = schedule(step, n_steps, a.warmup)
        rows = run.next_batch()
        if base:
            prepare.declare_step(base)
        loss = loss_backward(fwd, rows, a)
        grads = [params[n].grad for n in names]
        gnorm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
        with torch.no_grad():
            for _ in range(a.dirs):
                gen.manual_seed(a.seed + len(coeffs))
                zs = [torch.randn(w.shape, generator=gen, device=w.device, dtype=torch.float32) for w in masters]
                gi = sum(torch.dot(g.reshape(-1), z.reshape(-1)) for g, z in zip(grads, zs))
                c = float((a.lr * frac * gi / (gnorm * math.sqrt(P) + 1e-12)).to(torch.float16))
                coeffs.append(c)
                for w, z in zip(masters, zs):
                    w.add_(z, alpha=-c)
        for n in names:
            params[n].grad = None
        run.steps += [{"op": "train", "tokens": run.tokens},
                      {"op": "extra", "flops": a.dirs * dir_flops, "what": "directional derivatives and steps"}]
        run.report(step, loss)
        base = prepare.encode_seedlog(names, a.seed, coeffs, "f16")
        run.used.add(run.bucket())
    run.close_arm(base, trainable, {"directions": len(coeffs), "steps": n_steps})


def parse_arms(base: argparse.Namespace) -> list[argparse.Namespace]:
    arms = []
    for spec in (base.arms.split("/") if base.arms else [""]):
        over = {}
        for kv in filter(None, spec.split(",")):
            k, v = kv.split("=", 1)
            k = k.replace("-", "_")
            if k not in ARM_KEYS:
                raise SystemExit(f"--arms: {k!r} is not one of {sorted(ARM_KEYS)}")
            over[k] = ARM_KEYS[k](v)
        arms.append(argparse.Namespace(**(vars(base) | over)))
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-e-bytes", type=int, default=0, help="E's size cap, the island's own choice (0: none)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mech", choices=("relora", "stack", "galore", "full", "subspace"), required=True)
    ap.add_argument("--arms", default="", help="settings separated by '/', each key=value pairs separated by ','")
    ap.add_argument("--basis", choices=("lora", "galore"), default="lora", help="stack: the active piece's form")
    ap.add_argument("--targets", default="all", help="all, out, mlp, attn, or leaf names joined by '+' (wq+wo)")
    ap.add_argument("--blocks", default="all", help="all, or LO-HI (0-based, inclusive)")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--pieces", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0, help="relora/stack: the scale's multiplier on sqrt(d_in / r)")
    ap.add_argument("--norms", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--norm-lr", type=float, default=3e-4)
    ap.add_argument("--opt", choices=("adam", "muon"), default="adam")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rewarm", type=int, default=20, help="relora/stack/galore: steps of warmup after each restart")
    ap.add_argument("--keep-moments", action="store_true", help="galore: keep the projected moments across refreshes")
    ap.add_argument("--alloc", type=int, default=0, help="galore: 1 allocates each piece's ranks across tensors by the "
                    "gradient's σ² per byte, within the bytes of rank r on every tensor")
    ap.add_argument("--project-every", type=int, default=0, help="full: project W − θ to the byte budget every T steps")
    ap.add_argument("--project-bytes", type=int, default=0)
    ap.add_argument("--whiten", type=int, default=0, help="truncate by output error, from this many batches' inputs")
    ap.add_argument("--no-piece-snaps", action="store_true")
    ap.add_argument("--dirs", type=int, default=64, help="subspace: directions per step")
    ap.add_argument("--dtype", choices=("f16", "bf16"), default="f16", help="E's factors and norm gains")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--budget-fraction", type=float, default=0.98, help="of the budget, shared by the arms")
    base = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    CODE.update(name=base.dtype, dtype=prepare.DTYPES[base.dtype])
    arms = parse_arms(base)
    run = Run(base, arms)
    mech = {"relora": run_relora, "stack": run_stack, "galore": run_galore, "full": run_full, "subspace": run_subspace}
    try:
        for a in arms:
            run.begin(a)
            mech[a.mech](run)
        return run.submit()
    except Exception:
        traceback.print_exc()
        raise
