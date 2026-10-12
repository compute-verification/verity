"""The audited decode step on node 2, and its γ, for one workload and one choice of parameters.

A workload is a Llama-class shape in FP8 (KV in FP8, lm_head in BF16) at a decode batch and a context, laid out at the
smallest tensor-parallel degree in {1, 2, 4, 8} whose weights and KV fit the 93.9 GB stored per GPU; the node runs
8/tp replicas. Times are per GPU per decode step.

The step, per GPU (``step``):
- plain: the linears take max(W·e_mem(m)/R, FLOPs/Φ), attention KV·e_attn/R, activations their HBM traffic/R, and a
  tensor-parallel step 2L all-reduces over PCIe;
- audited: the linears' CUDA-core work (decode o_w, receipt τ_w, binding τ_bind, per weight pass) is exposed except
  for what hides under tensor-bound time (β·(FLOPs/Φ − reads)); the KV append's encode (k·x₁_h read-times a byte) hides
  in what tensor slack is left; attention pays o_kv + τ_kv + τ_bind a KV pass; the host tier slows the step by
  τ_host per b_zero_copy of host stream per GPU, and the all-reduce gets what the host stream leaves of each GPU's
  PCIe receive side, min(b_p2p, b_pcie − stream).
- under ``cap``, memory left idle is filled with receipt-only reads of the store (τ_fill a pass, no decode: top-up),
  and the step may be padded with more of them (``pad``) when its activation traffic is too large a share of it;
  under ``pass`` neither: the pass is the step's own reads, W + KV.

γ (``gamma_cap``, ``gamma_pass``), as fractions of the node's read capacity ΣR = 8·R + b_host:
- ``cap`` (the ruling: 99% of read bandwidth goes to registered data): μ_chip = S_chip/H, μ_rebuild = s·(1+h)/(k·x₁),
  μ_partial, μ_host = (1 − host share)·b_host/ΣR, μ_unreg = the step's activation traffic over its capacity,
  μ_sample = ln(1/δ)/s, λ = (log₂q + log₂e − log₂N)/(8n), μ_sketch = 2^-k_s, μ_link (0 under the identity
  statement ``C``; ``link_n`` prices memory-only helpers over the NIC). The honest read shortfall η is unmeasured:
  the frontier holds ``eta_reserve`` back for it and reports what is left (``eta_allowance``).
- ``pass`` (the design's and the red team's): μ_chip = S_chip/(W + KV), μ_rebuild = d·s·(1+h)/(k·x₁) and
  μ_host = d·(1 − share)·b_host/(8R) with d = the audited step in read passes of W + KV, no μ_unreg, and the capacity
  left idle reported as ``residual`` = 1 − (1 − γ)/d (F9).
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

    def act_bytes_per_token(self, tp: int) -> float:
        kv = self.kv_heads * self.head_dim
        return 2.0 * self.layers * (12 * self.hidden + (4 * (self.hidden + 2 * kv) + 6 * self.inter) / tp)


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
    host_share: float

    @property
    def binding_label(self) -> str:
        return "tile" if self.binding == 0 else f"k-slice {self.binding}"


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


def _allreduce(w: Workload, tp: int, b_p2p: float, v: dict[str, Any]) -> float:
    if tp == 1:
        return 0.0
    msg = w.batch * w.shape.hidden * 2.0
    return 2 * w.shape.layers * (v["t_allreduce"] + 2 * (tp - 1) / tp * msg / (b_p2p * 1e9))


def step(w: Workload, p: Params, v: dict[str, Any]) -> dict[str, float]:
    tp = layout(w, v)
    if tp is None:
        raise ValueError(f"{w.label} does not fit the node")
    s, m = w.shape, w.batch
    R = v["R"] * 1e9
    W = s.weight_bytes / tp
    KV = m * w.ctx * s.kv_bytes_per_token / tp
    act = m * s.act_bytes_per_token(tp) * (1.0 - v["rho_l2"])
    e_mem = interp(v["e_mem"], m)
    t_rd = W * e_mem / R
    t_tc = s.flops_per_token * m / tp / v["phi"]
    t_kv = KV * v["e_attn"] / R
    t_act = act / R
    plain = max(t_rd, t_tc) + t_kv + t_act + _allreduce(w, tp, v["b_p2p"], v)

    blk = 0.75 + 0.25 * 4096 / p.block
    ksr = v["ks_ratio"][p.k_s]
    tau_w = interp(v["tau_w"][p.k], m) * ksr * blk
    tau_kv = interp(v["tau_kv"][p.k], m) * ksr * blk
    tau_fill = v["tau_fill"][p.k_s] * blk
    o_w = interp(v["o_w"][p.k], m)
    o_kv = interp(v["o_kv"][p.k], m)

    cuda_w = t_rd * (o_w + tau_w + v["tau_bind"])
    slack = v["beta"] * max(0.0, t_tc - t_rd)
    lin = max(t_rd, t_tc) + max(0.0, cuda_w - slack)
    slack = max(0.0, slack - cuda_w)
    t_enc = m * s.kv_bytes_per_token / tp * p.k * v["x1_h"] / R
    enc = max(0.0, t_enc - slack)
    slack = max(0.0, slack - t_enc)
    attn = t_kv * (1.0 + o_kv) + KV / R * (tau_kv + v["tau_bind"])

    r_h = min(p.host_share * v["b_host"] / v["gpus"], v["b_zero_copy"])
    host = v["tau_host"] * r_h / v["b_zero_copy"]
    t_ar = _allreduce(w, tp, max(1.0, min(v["b_p2p"], v["b_pcie"] - r_h)), v)

    core = lin + enc + attn + t_act + t_ar
    busy = t_rd + t_kv + t_act
    idle = max(0.0, core - busy)
    free = slack + t_ar
    topup = max(0.0, tau_fill * idle - free) / (1.0 - tau_fill)
    cap = (core + topup) * (1.0 + host)
    return {
        "tp": tp, "plain": plain, "cap": cap, "pass": core * (1.0 + host), "t_rd": t_rd, "t_tc": t_tc,
        "t_kv": t_kv, "t_act": t_act, "t_ar": t_ar, "lin": lin, "enc": enc, "attn": attn, "idle": idle,
        "topup": topup, "pad": 0.0, "host": host, "host_stream": v["gpus"] * r_h, "W": W, "KV": KV, "act": act,
        "f_kv_hbm": KV / v["H"],
        "f_kv_reads": t_kv / plain,
    }


def pad(st: dict[str, float], unreg_budget: float, v: dict[str, Any]) -> dict[str, float]:
    """The step padded with receipt-only reads until its activation traffic is ``unreg_budget`` of its capacity
    (μ_unreg under ``cap`` falls as the step grows); unchanged if it already is."""
    sigma = v["gpus"] * v["R"] + v["b_host"]
    hbm = v["gpus"] * v["R"] / sigma
    need = st["act"] * hbm / (v["R"] * 1e9 * unreg_budget) if unreg_budget > 0 else math.inf
    if need <= st["cap"]:
        return st
    return {**st, "cap": need, "pad": need - st["cap"]}


def _openings(w: Workload, p: Params, v: dict[str, Any]) -> tuple[float, float]:
    n_blocks = v["gpus"] * v["H"] / p.block
    path = 2 * 64 * math.ceil(math.log2(n_blocks))
    k_bind = w.shape.hidden if p.binding == 0 else p.binding
    x_share = 0.0 if v["x_from_transcript"] else p.block * min(1.0, 4.0 * w.batch / k_bind)
    s = v["v_budget"] * 1e9 / 8 * v["t_window"] / (p.block + path + x_share)
    return s, n_blocks


def _common(w: Workload, p: Params, v: dict[str, Any]) -> dict[str, float]:
    s, n_blocks = _openings(w, p, v)
    return {
        "mu_partial": v["mu_partial"][p.k],
        "mu_sample": math.log(2) * v["log2_inv_delta"] / s,
        "lambda": max(0.0, v["log2_q"] + math.log2(math.e) - math.log2(n_blocks)) / (8 * p.block),
        "mu_sketch": 2.0 ** -p.k_s,
        "openings": s,
    }


def _rebuild(p: Params, v: dict[str, Any]) -> float:
    return v["s_adv"] * (1 + v["h_host_sq"]) / (p.k * v["x1"])


def gamma_cap(w: Workload, p: Params, v: dict[str, Any], st: dict[str, float]) -> dict[str, float]:
    sigma = v["gpus"] * v["R"] + v["b_host"]
    hbm = v["gpus"] * v["R"] / sigma
    t = {
        "mu_chip": v["s_chip"] / v["H"] * hbm,
        "mu_rebuild": _rebuild(p, v) * hbm,
        "mu_host": (v["b_host"] - st["host_stream"]) / sigma,
        "mu_unreg": st["act"] / (v["R"] * 1e9 * st["cap"]) * hbm,
        **_common(w, p, v),
    }
    t["gamma"] = sum(x for n, x in t.items() if n.startswith("mu_") or n == "lambda")
    t["eta_allowance"] = GAMMA_MAX - t["gamma"]
    return t


def gamma_pass(w: Workload, p: Params, v: dict[str, Any], st: dict[str, float]) -> dict[str, float]:
    store = st["W"] + st["KV"]
    d = st["pass"] * v["R"] * 1e9 / store
    t = {
        "mu_chip": min(1.0, v["s_chip"] / store),
        "mu_rebuild": d * _rebuild(p, v),
        "mu_host": d * (v["b_host"] - st["host_stream"]) / (v["gpus"] * v["R"]),
        **_common(w, p, v),
    }
    t["gamma"] = sum(x for n, x in t.items() if n.startswith("mu_") or n == "lambda")
    t["d"] = d
    t["residual"] = 1 - (1 - t["gamma"]) / d
    return t


def link_n(w: Workload, p: Params, v: dict[str, Any], st: dict[str, float]) -> float:
    """μ_link for memory-only helpers over the NIC: each class of reads a helper can serve from a copy, given what
    the relay must move a byte (z for a weight block, q and the output for a sequence's KV, nothing for top-up)."""
    sigma = v["gpus"] * v["R"] + v["b_host"]
    hbm = v["gpus"] * v["R"] / sigma
    k_bind = w.shape.hidden if p.binding == 0 else p.binding
    shares = {
        "weights": (st["t_rd"] / st["cap"] * hbm, min(1.0, 4.0 * w.batch / k_bind)),
        "kv": (st["t_kv"] / st["cap"] * hbm, min(1.0, 4.0 * w.shape.hidden / (w.ctx * w.shape.kv_bytes_per_token))),
        "topup": (max(0.0, st["cap"] - st["t_rd"] - st["t_kv"] - st["t_act"]) / st["cap"] * hbm, 0.0),
    }
    nic = v["b_nic"] / sigma
    return sum(share if r == 0 else min(share, nic / r) for share, r in shares.values())


def evaluate(w: Workload, p: Params, v: dict[str, Any]) -> dict[str, Any]:
    st = step(w, p, v)
    cap = gamma_cap(w, p, v, st)
    return {
        "step": st,
        "cap": {"overhead": st["cap"] / st["plain"] - 1, **cap},
        "pass": {"overhead": st["pass"] / st["plain"] - 1, **gamma_pass(w, p, v, st)},
        "mu_link_n": link_n(w, p, v, st),
    }
