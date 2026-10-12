"""The audited decode step on node 2, and its γ, for one workload and one choice of parameters.

A workload is a Llama-class shape in FP8 (KV in FP8, lm_head in BF16 and encoded like every linear with a BF16
payload, so the logits never change) at a decode batch and a context, laid out at the smallest tensor-parallel degree
in {1, 2, 4, 8} whose weights and KV fit the 93.9 GB stored per GPU; the node runs 8/tp replicas. Times are per GPU
per decode step.

The claim is Conservation (Daniel, 21:00Z): receipts over registered data worth ≥ 99% of the declared capacity were
paid for by reads, held state or rebuild compute. The declared capacity is HBM's R·T on node 2's 8 GPUs; a helper
only spends resources of its own, so no host tier, no μ_link and no identity term enters γ, and binding receipts to z
is the optional PoUW join (``bound``).

The step, per GPU (``step``):
- plain: the linears take max(W·e_mem(m)/R, FLOPs/Φ), attention KV·e_attn/R, the ephemeral traffic (reads, and
  writes at ``write_bus`` read-times a byte, measured by ra-ephemeral, ``eph_split`` of it divided across a step's
  cards) its bytes/R, and a tensor-parallel step 2L all-reduces over PCIe;
- audited: the linears' CUDA-core work (decode o_w and receipt τ_w per weight pass, and binding τ_bind if ``bound``)
  is exposed except for what hides under tensor-bound time (β·(FLOPs/Φ − reads)); the KV append's encode (k·x₁_h
  read-times a byte, for the step's tokens and P/O prompt tokens a sequence) hides in what tensor slack is left;
  attention pays o_kv + τ_kv (+ τ_bind_kv) a KV pass;
- each pass waits ``t_gap`` for the previous pass's root (its seed), less S_chip/R with ``prefetch``; with ``copies``
  c differently keyed copies of the weights and KV, a pass spans c steps, each reading its own copy (so no block twice
  in a pass), and the KV append encodes c times;
- under ``cap``, memory left idle is filled with receipt-only reads of the store (τ_fill a pass, no decode: top-up),
  and the step may be padded with more of them (``pad``) when its ephemeral traffic is too large a share of it;
  under ``pass`` neither: the pass is the step's own reads, W + KV.
- the word: ``bits`` wide, with ra-codec-mix's σ_K between rounds if ``masked`` (its XOR a decoded byte, and partial
  completion at ``theta_masked``). x₁(b) is kv.cu's root time a byte at b against p512's (ra-squaring-ceiling's
  ``wide.x1_ref``), the honest append's x₁_h scales with it, a decode's cost a byte scales by ``dec_ratio`` (its
  floor and exposed share alike), and ``sq_speed`` is the honest kernels' squaring against kv.cu's at b.

γ (``gamma_cap``, ``gamma_pass``), as fractions of HBM's R·T:
- ``cap`` (99% of read bandwidth to registered data): μ_chip = S_chip/(R·t_pass), the state carried across each pass
  boundary (a pass is one step's plan, since a plan reads a block at most once, so short steps pay it most),
  μ_rebuild = s(b)·(1+h(b))/(k·x₁(b)) (WorkHardness at the adversary's squaring, ``wide.adversaries``), μ_partial =
  μ_rebuild·(1/θ − 1), μ_unreg = the step's ephemeral bus time over its capacity (μ_eph), μ_timing = (RTT +
  pass)/(nT) + gap/pass, μ_sample = ln(1/δ)/(s·horizon) for the hybrid's s openings a window (each window's own
  sampling at most ``window_max``), μ_serve = the honest reads that serve them, λ = (N + log₂(1/δ))/(8nN) named (else
  (log₂q + log₂e − log₂N)/(8n)), μ_sketch = 2^-k_s, and against R·T (``eta_counted``) μ_eta = ra-kv-filler's phases
  (decode, prefill, the engine's gaps, idle; on the GPU's clock). ``eta_allowance`` is what the rest of γ leaves for η.
- ``pass`` (the design's and the red team's): μ_chip = S_chip/(W + KV), μ_rebuild = d·s·(1+h)/(k·x₁) with d = the
  audited step in read passes of W + KV, no μ_unreg, and the capacity left idle, the chaining gap's included,
  reported as ``residual`` = 1 − (1 − γ)/d (F9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

GAMMA_MAX = 0.01


@dataclass(frozen=True)
class Shape:
    name: str
    layers: int
    hidden: int
    inter: int
    linear_params: float
    kv_heads: int = 8
    head_dim: int = 128
    vocab: int = 128_256

    @property
    def lm_head_bytes(self) -> float:
        return 2.0 * self.hidden * self.vocab

    @property
    def weight_bytes(self) -> float:
        return self.linear_params + self.lm_head_bytes

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim

    @property
    def flops_per_token(self) -> float:
        return 2.0 * (self.linear_params + self.hidden * self.vocab)

    def ephemeral(self, table: dict[str, tuple[float, float]], batch: int, tp: int, split: float) -> float:
        fixed, per_seq = table[self.name]
        return self.layers * self.hidden * (fixed + per_seq * batch) * ((1.0 - split) + split / tp)


SHAPES = {
    "8B": Shape("8B", 32, 4096, 14336, 6.98e9),
    "70B": Shape("70B", 80, 8192, 28672, 68.45e9),
    "405B": Shape("405B", 126, 16384, 53248, 401.65e9),
}


@dataclass(frozen=True)
class Workload:
    shape: Shape
    batch: int
    ctx: int
    tp: int | None = None

    @property
    def label(self) -> str:
        return f"{self.shape.name} B{self.batch} ctx{self.ctx // 1024}k"


@dataclass(frozen=True)
class Params:
    k: int
    block: int
    k_s: int
    binding: int
    copies: int = 1

    @property
    def binding_label(self) -> str:
        return "tile" if self.binding == 0 else f"k-slice {self.binding}"


def at_width(v: dict[str, Any], name: str, bits: int | None = None) -> float:
    return interp(v[name], v["bits"] if bits is None else bits)


def margins(bits: int) -> dict[str, float]:
    """The red team's partial-completion margin at k = 2 (cost over the per-bit claim), an estimate: its table at 512,
    640 and 768 bits (fplll 12.3 / 8.7 / 6.7; the GPU floor's centre 1.2 / 0.8 / 0.6 and low end 0.75 / 0.50 / 0.37),
    fitted as b^-1.55 and b^-1.8."""
    r = 512 / bits
    return {"fplll": 12.3 * r ** 1.55, "floor_centre": 1.2 * r ** 1.8, "floor_low": 0.75 * r ** 1.8}


def decode(o: float, k: int, v: dict[str, Any]) -> float:
    """A decode form's cost a pass at ``v``'s word width and squaring speed, from its cost ``o`` at 512 bits on kv.cu's
    squaring: the floor c·k − 1 where the decode outlasts the read, plus the share of c·k the form exposes beyond it,
    c·k scaled by the width's decode a byte (``dec_ratio``). ra-codec-mix's σ_K adds its XOR a byte (``mask_cost``)."""
    x0 = 1.0 + v["decode_floor"][k]
    f = max(0.0, o - max(0.0, x0 - 1.0)) / x0
    x = x0 * at_width(v, "dec_ratio") / at_width(v, "sq_speed")
    return max(0.0, x - 1.0) + f * x + (v["mask_cost"][k] if v["masked"] else 0.0)


def interp(table: Any, m: float) -> float:
    if isinstance(table, (int, float)):
        return float(table)
    xs = [x for x, _ in table]
    ys = [y for _, y in table]
    if m <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if m <= x1:
            return y0 + (y1 - y0) * (m - x0) / (x1 - x0)
    return ys[-1]


def layout(w: Workload, v: dict[str, Any]) -> int | None:
    if w.tp is not None:
        return w.tp
    need = w.shape.weight_bytes + w.batch * w.ctx * w.shape.kv_bytes_per_token
    for tp in (1, 2, 4, 8):
        if need / tp <= v["H"]:
            return tp
    return None


def _allreduce(w: Workload, tp: int, v: dict[str, Any]) -> float:
    if tp == 1:
        return 0.0
    msg = w.batch * w.shape.hidden * 2.0
    return 2 * w.shape.layers * (v["t_allreduce"] + 2 * (tp - 1) / tp * msg / (v["b_p2p"] * 1e9))


def z_ratio(w: Workload, p: Params) -> float:
    """|z|/|E| for a weight block: 4m/K with tile-output binding (K the hidden size), 4m/K_slice with k-slice
    binding; above 1 the receipt folds the surplus z into its |E| bytes, at that many XORs a byte."""
    return 4.0 * w.batch / (w.shape.hidden if p.binding == 0 else p.binding)


def step(w: Workload, p: Params, v: dict[str, Any]) -> dict[str, float]:
    tp = layout(w, v)
    if tp is None:
        raise ValueError(f"{w.label} does not fit the node")
    s, m = w.shape, w.batch
    R = v["R"] * 1e9
    W = s.weight_bytes / tp
    KV = m * w.ctx * s.kv_bytes_per_token / tp
    act = s.ephemeral(v["eph_read"], m, tp, v["eph_split"])
    wr = (s.ephemeral(v["eph_write"], m, tp, v["eph_split"]) + m * s.kv_bytes_per_token / tp) * v["write_bus"]
    e_mem = interp(v["e_mem"], m)
    t_rd = W * e_mem / R
    t_tc = s.flops_per_token * m / tp / v["phi"]
    t_kv = KV * v["e_attn"] / R
    t_act = (act + wr) / R
    t_ar = _allreduce(w, tp, v)
    plain = max(t_rd, t_tc) + t_kv + t_act + t_ar

    blk = 0.75 + 0.25 * 4096 / p.block
    ksr = v["ks_ratio"][p.k_s]
    tau_w = interp(v["tau_w"][p.k], m) * ksr * blk
    tau_kv = interp(v["tau_kv"][p.k], m) * ksr * blk
    tau_fill = v["tau_fill"][p.k_s] * blk
    o_w = decode(interp(v["o_w"][p.k], m), p.k, v)
    o_kv = decode(interp(v["o_kv"][p.k], m), p.k, v)
    x1_h = v["x1_h"] * at_width(v, "x1") / at_width(v, "x1", 512) / at_width(v, "sq_speed")
    bound = 1.0 if v["bound"] else 0.0

    tau_bind = bound * v["tau_bind"] * max(1.0, z_ratio(w, p))
    cuda_w = t_rd * (o_w + tau_w + tau_bind)
    slack = v["beta"] * max(0.0, t_tc - t_rd)
    lin = max(t_rd, t_tc) + max(0.0, cuda_w - slack)
    slack = max(0.0, slack - cuda_w)
    t_enc = m * s.kv_bytes_per_token / tp * (1.0 + v["p_over_o"]) * p.k * x1_h / R * p.copies
    enc = max(0.0, t_enc - slack)
    slack = max(0.0, slack - t_enc)
    attn = t_kv * (1.0 + o_kv) + KV / R * (tau_kv + bound * v["tau_bind_kv"][p.k])
    gap = max(0.0, v["t_gap"] - (v["s_chip"] / R if v["prefetch"] else 0.0))

    core = lin + enc + attn + t_act + t_ar
    busy = t_rd + t_kv + t_act
    idle = max(0.0, core - busy)
    free = slack + t_ar
    topup = max(0.0, tau_fill * idle - free) / (1.0 - tau_fill)
    cap = core + topup + gap / p.copies
    return {
        "tp": tp, "plain": plain, "cap": cap, "pass": core + gap / p.copies, "copies": p.copies,
        "t_rd": t_rd, "t_tc": t_tc,
        "t_kv": t_kv, "t_act": t_act, "t_ar": t_ar, "lin": lin, "enc": enc, "attn": attn, "idle": idle,
        "topup": topup, "pad": 0.0, "gap": gap,
        "o_w": o_w, "o_kv": o_kv, "bits": v["bits"], "W": W, "KV": KV,
        "act": act + wr, "eph_read": act, "eph_write": wr,
        "f_kv_hbm": KV / v["H"],
        "f_kv_reads": t_kv / plain,
    }


def _per_pass(v: dict[str, Any], st: dict[str, float]) -> dict[str, float]:
    """γ's terms under ``cap`` that fall as the step grows, as B in μ = B/t_step (s): the state carried across a pass
    boundary on chip and the chaining gap, once a pass of ``copies`` steps, and the step's ephemeral traffic."""
    R = v["R"] * 1e9
    c = st["copies"]
    return {"mu_chip": v["s_chip"] / R / c, "mu_unreg": st["act"] / R, "gap": st["gap"] / c}


def pad(w: Workload, p: Params, v: dict[str, Any], st: dict[str, float], limit: float) -> dict[str, float]:
    """The step padded with receipt-only reads to the shortest pass whose γ under ``cap`` is ``limit``: the per-pass
    terms fall as 1/t and the session's tail grows as t/(nT); unchanged if γ is already within it or no length is."""
    t0 = st["cap"]
    g = gamma_cap(w, p, v, st)["gamma"]
    if g <= limit:
        return st
    b = sum(_per_pass(v, st).values())
    c = st["copies"] / (v["n_session"] * v["t_window"])
    room = limit - (g - b / t0 - c * t0)
    disc = room * room - 4 * b * c
    if room <= 0 or disc < 0:
        return st
    t = (room - math.sqrt(disc)) / (2 * c)
    if t <= t0:
        return st
    return {**st, "cap": t, "pad": t - t0}


def _openings(w: Workload, p: Params, v: dict[str, Any]) -> tuple[float, float]:
    """The hybrid's openings a window (each window's sampling at ``window_max``), or fewer if the verifier's budget
    can't take them, and the store's blocks. An opening carries z only for a bound receipt."""
    n_blocks = v["gpus"] * v["H"] / p.block
    k_bind = w.shape.hidden if p.binding == 0 else p.binding
    x_share = p.block * min(1.0, 4.0 * w.batch / k_bind) if v["bound"] and not v["x_from_transcript"] else 0.0
    hybrid = math.log(2) * v["log2_inv_delta"] / v["window_max"]
    budget = v["v_budget"] * 1e9 / 8 * v["t_window"] / (p.block + v["opening_bytes"] + x_share)
    return min(hybrid, budget), n_blocks


def _lambda(p: Params, v: dict[str, Any], n_blocks: float) -> float:
    if v["lambda_named"]:
        return (n_blocks + v["log2_inv_delta"]) / (8 * p.block * n_blocks)
    return max(0.0, v["log2_q"] + math.log2(math.e) - math.log2(n_blocks)) / (8 * p.block)


def _common(w: Workload, p: Params, v: dict[str, Any]) -> dict[str, float]:
    s, n_blocks = _openings(w, p, v)
    ln = math.log(2) * v["log2_inv_delta"]
    return {
        "mu_sample": ln / (s * v["horizon"]),
        "mu_serve": v["mu_serve"],
        "lambda": _lambda(p, v, n_blocks),
        "mu_sketch": 2.0 ** -p.k_s,
        "openings": s,
        "window_sample": ln / s,
    }


def _rebuild(p: Params, v: dict[str, Any]) -> float:
    return at_width(v, "s_adv") * (1 + at_width(v, "h_host_sq")) / (p.k * at_width(v, "x1"))


def _partial(p: Params, v: dict[str, Any]) -> float:
    """What completing a word's dropped tail adds to the rebuild: WorkHardness at θ·c_root divides the rebuild term by
    θ (ra-partial-lean), with θ at this width from the red team's floor; with σ_K, ra-codec-mix's θ at every width."""
    if v["masked"]:
        theta = v["theta_masked"]
    else:
        theta = v["theta"] * margins(v["bits"])["floor_low"] / margins(512)["floor_low"]
    return _rebuild(p, v) * max(0.0, 1.0 / theta - 1.0)


def _timing(t_pass: float, gap: float, v: dict[str, Any]) -> float:
    """ra-remote-driver's μ_timing for a session of chained windows: its start and tail, (RTT + pass)/(nT), and the
    chaining gap's share of each pass."""
    return (v["rtt"] + t_pass) / (v["n_session"] * v["t_window"]) + gap / t_pass


def gamma_cap(w: Workload, p: Params, v: dict[str, Any], st: dict[str, float]) -> dict[str, float]:
    once = _per_pass(v, st)
    t = {
        "mu_chip": once["mu_chip"] / st["cap"],
        "mu_rebuild": _rebuild(p, v),
        "mu_partial": _partial(p, v),
        "mu_unreg": once["mu_unreg"] / st["cap"],
        "mu_timing": _timing(st["cap"] * st["copies"], st["gap"], v),
        "mu_eta": eta(v) if v["eta_counted"] else 0.0,
        **_common(w, p, v),
    }
    t["gamma"] = sum(x for n, x in t.items() if n.startswith("mu_") or n == "lambda")
    t["eta_allowance"] = GAMMA_MAX - t["gamma"] + t["mu_eta"]
    return t


def eta(v: dict[str, Any]) -> float:
    return v["eta_decode"] + v["eta_prefill"] + v["eta_gaps"] + v["eta_idle"]


def gamma_pass(w: Workload, p: Params, v: dict[str, Any], st: dict[str, float]) -> dict[str, float]:
    store = (st["W"] + st["KV"]) * st["copies"]
    d = st["pass"] * v["R"] * 1e9 / (st["W"] + st["KV"])
    t = {
        "mu_chip": min(1.0, v["s_chip"] / store),
        "mu_rebuild": d * _rebuild(p, v),
        "mu_partial": d * _partial(p, v),
        **_common(w, p, v),
    }
    t["gamma"] = sum(x for n, x in t.items() if n.startswith("mu_") or n == "lambda")
    t["d"] = d
    t["residual"] = 1 - (1 - t["gamma"]) / d
    return t


def evaluate(w: Workload, p: Params, v: dict[str, Any]) -> dict[str, Any]:
    st = step(w, p, v)
    return {
        "step": st,
        "cap": {"overhead": st["cap"] / st["plain"] - 1, **gamma_cap(w, p, v, st)},
        "pass": {"overhead": st["pass"] / st["plain"] - 1, **gamma_pass(w, p, v, st)},
    }
