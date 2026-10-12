"""The audited decode step on node 2, and its γ, for one workload and one choice of parameters.

A workload is a Llama-class shape in FP8 (KV in FP8, lm_head in BF16 and encoded like every linear) at a decode batch
and a context, laid out at the smallest tensor-parallel degree in {1, 2, 4, 8} whose weights and KV fit the 93.9 GB
stored per GPU; the node runs 8/tp replicas. Times are per GPU per decode step.

The step, per GPU (``step``):
- plain: the linears take max(W·e_mem(m)/R, FLOPs/Φ), attention KV·e_attn/R, the ephemeral traffic (reads, and
  writes at ``write_bus`` read-times a byte, measured by ra-ephemeral) its bytes/R, and a tensor-parallel step 2L
  all-reduces over PCIe;
- audited: the linears' CUDA-core work (decode o_w, receipt τ_w, binding τ_bind, per weight pass) is exposed except
  for what hides under tensor-bound time (β·(FLOPs/Φ − reads)); the KV append's encode (k·x₁_h read-times a byte) hides
  in what tensor slack is left; attention pays o_kv + τ_kv + τ_bind_kv a KV pass; the host tier slows the step by
  τ_host per b_zero_copy of host stream per GPU, and the all-reduce gets what the host stream leaves of each GPU's
  PCIe receive side, min(b_p2p, b_pcie − stream); the host's loop (its CPUs' receipts beside the GPUs' stream,
  ra-host-tier's) covers ``host_cpu`` of the DRAM ceiling at no cost to the step, and the GPUs' zero-copy stream
  adds only what exceeds it;
- each pass waits ``t_gap`` for the previous pass's root (its seed), less S_chip/R with ``prefetch``; with ``copies``
  c differently keyed copies of the weights and KV, a pass spans c steps, each reading its own copy (so no block twice
  in a pass), and the KV append encodes c times;
- under ``cap``, memory left idle is filled with receipt-only reads of the store (τ_fill a pass, no decode: top-up),
  and the step may be padded with more of them (``pad``) when its ephemeral traffic is too large a share of it;
  under ``pass`` neither: the pass is the step's own reads, W + KV.
- the word width ``bits`` scales the squaring by ``sq_ratio``: the encode a byte (x₁, the adversary's and the honest
  append's) as the squaring, the decode a byte as the squaring over the word's bytes (``decode``); ``sq_speed`` is
  the honest kernels' squaring against kv.cu's. ``masked`` is ra-codec-mix's σ_K between rounds: its XOR a decoded
  byte, and partial completion at ``theta_masked`` at every width.

γ (``gamma_cap``, ``gamma_pass``), as fractions of the node's read capacity ΣR = 8·R + b_host:
- ``cap`` (the ruling: 99% of read bandwidth goes to registered data): μ_chip = S_chip/(R·t_pass) and μ_hostcache =
  the host's caches over ΣR·t_pass, the state carried across each pass boundary (a pass is one step's plan, since a
  plan reads a block at most once, so short steps pay them most), μ_rebuild = s·(1+h)/(k·x₁(b)), μ_partial = μ_rebuild·(1/θ(b) − 1), μ_host = the host DRAM the loop
  leaves over ΣR, μ_unreg = the step's ephemeral bus time over its capacity, μ_timing = (RTT + pass)/(nT) + gap/pass,
  μ_sample = ln(1/δ)/s, λ = (log₂q + log₂e − log₂N)/(8n), μ_sketch = 2^-k_s, μ_cpu = the host CPUs' tile work by batch if
  this audit must count it (0: under C it buys nothing without bytes from off the node), μ_link (0 under the identity statement
  ``C``; ``link_n`` prices memory-only helpers over the NIC). The step reads at R by construction, so the honest
  engine's read shortfall η is what γ leaves (``eta_allowance``) when exhaustion is against a declared load; against
  R·T (``eta_counted``) μ_eta adds ra-kv-filler's phases (decode, prefill, the engine's gaps, idle).
- ``pass`` (the design's and the red team's): μ_chip = S_chip/(W + KV), μ_rebuild = d·s·(1+h)/(k·x₁) and
  μ_host = d·(b_host − covered)/(8R) with d = the audited step in read passes of W + KV, no μ_unreg, and the capacity
  left idle, the chaining gap's included, reported as ``residual`` = 1 − (1 − γ)/d (F9).
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

    def ephemeral(self, table: dict[str, tuple[float, float]], batch: int, tp: int) -> float:
        fixed, per_seq = table[self.name]
        return self.layers * self.hidden * (fixed + per_seq * batch) / tp


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
    copies: int = 1

    @property
    def binding_label(self) -> str:
        return "tile" if self.binding == 0 else f"k-slice {self.binding}"


def sq_ratio(bits: int) -> float:
    """A modular squaring's cost at ``bits`` over its cost at 512, by ra-squaring-ceiling's cycle model at n 32-bit limbs:
    n(n+1)/2 products with their carries at 4.63 cycles, an n-product fold at 4.02 and the doubling's 2n − 1 shifts at
    0.6 (713 cycles at 512, which FOLD 5 runs at 774)."""
    def cycles(b: int) -> float:
        n = b / 32
        return n * (n + 1) / 2 * 4.63 + 4.02 * n + 0.6 * (2 * n - 1)
    return cycles(bits) / cycles(512)


def margins(bits: int) -> dict[str, float]:
    """The red team's partial-completion margin at k = 2 (cost over the per-bit claim), an estimate: its table at 512,
    640 and 768 bits (fplll 12.3 / 8.7 / 6.7; the GPU floor's centre 1.2 / 0.8 / 0.6 and low end 0.75 / 0.50 / 0.37),
    fitted as b^-1.55 and b^-1.8, extrapolated below 512."""
    r = 512 / bits
    return {"fplll": 12.3 * r ** 1.55, "floor_centre": 1.2 * r ** 1.8, "floor_low": 0.75 * r ** 1.8}


def decode(o: float, k: int, v: dict[str, Any]) -> float:
    """A decode form's cost a pass at ``v``'s word width and squaring speed, from its cost ``o`` at 512 bits on kv.cu's
    squaring: the floor c·k − 1 where the decode outlasts the read, plus the share of c·k the form exposes beyond it.
    A round decodes a word with one squaring, so c scales as the squaring's cost over the word's bytes. ra-codec-mix's
    σ_K adds its XOR a byte (``mask_cost``)."""
    x0 = 1.0 + v["decode_floor"][k]
    f = max(0.0, o - max(0.0, x0 - 1.0)) / x0
    x = x0 * sq_ratio(v["bits"]) * 512 / v["bits"] / v["sq_speed"]
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


def _allreduce(w: Workload, tp: int, b_p2p: float, v: dict[str, Any]) -> float:
    if tp == 1:
        return 0.0
    msg = w.batch * w.shape.hidden * 2.0
    return 2 * w.shape.layers * (v["t_allreduce"] + 2 * (tp - 1) / tp * msg / (b_p2p * 1e9))


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
    act = s.ephemeral(v["eph_read"], m, tp)
    wr = (s.ephemeral(v["eph_write"], m, tp) + m * s.kv_bytes_per_token / tp) * v["write_bus"]
    e_mem = interp(v["e_mem"], m)
    t_rd = W * e_mem / R
    t_tc = s.flops_per_token * m / tp / v["phi"]
    t_kv = KV * v["e_attn"] / R
    t_act = (act + wr) / R
    plain = max(t_rd, t_tc) + t_kv + t_act + _allreduce(w, tp, v["b_p2p"], v)

    blk = 0.75 + 0.25 * 4096 / p.block
    ksr = v["ks_ratio"][p.k_s]
    tau_w = interp(v["tau_w"][p.k], m) * ksr * blk
    tau_kv = interp(v["tau_kv"][p.k], m) * ksr * blk
    tau_fill = v["tau_fill"][p.k_s] * blk
    o_w = decode(interp(v["o_w"][p.k], m), p.k, v)
    o_kv = decode(interp(v["o_kv"][p.k], m), p.k, v)
    x1_h = v["x1_h"] * sq_ratio(v["bits"]) / v["sq_speed"]

    tau_bind = v["tau_bind"] * max(1.0, z_ratio(w, p))
    cuda_w = t_rd * (o_w + tau_w + tau_bind)
    slack = v["beta"] * max(0.0, t_tc - t_rd)
    lin = max(t_rd, t_tc) + max(0.0, cuda_w - slack)
    slack = max(0.0, slack - cuda_w)
    t_enc = m * s.kv_bytes_per_token / tp * (1.0 + v["p_over_o"]) * p.k * x1_h / R * p.copies
    enc = max(0.0, t_enc - slack)
    slack = max(0.0, slack - t_enc)
    attn = t_kv * (1.0 + o_kv) + KV / R * (tau_kv + v["tau_bind_kv"][p.k])

    r_h = min(p.host_share * v["b_host"] / v["gpus"], v["b_zero_copy"])
    host = v["tau_host"] * r_h / v["b_zero_copy"]
    stream = v["gpus"] * r_h
    host_cpu = max(0.0, v["host_cpu"] * v["b_host"] - stream)
    t_ar = _allreduce(w, tp, max(1.0, min(v["b_p2p"], v["b_pcie"] - r_h)), v)
    gap = max(0.0, v["t_gap"] - (v["s_chip"] / R if v["prefetch"] else 0.0))

    core = lin + enc + attn + t_act + t_ar
    busy = t_rd + t_kv + t_act
    idle = max(0.0, core - busy)
    free = slack + t_ar
    topup = max(0.0, tau_fill * idle - free) / (1.0 - tau_fill)
    cap = (core + topup) * (1.0 + host) + gap / p.copies
    return {
        "tp": tp, "plain": plain, "cap": cap, "pass": core * (1.0 + host) + gap / p.copies, "copies": p.copies,
        "t_rd": t_rd, "t_tc": t_tc,
        "t_kv": t_kv, "t_act": t_act, "t_ar": t_ar, "lin": lin, "enc": enc, "attn": attn, "idle": idle,
        "topup": topup, "pad": 0.0, "gap": gap, "host": host, "host_stream": stream, "host_cpu": host_cpu,
        "o_w": o_w, "o_kv": o_kv, "bits": v["bits"], "W": W, "KV": KV,
        "act": act + wr, "eph_read": act, "eph_write": wr,
        "f_kv_hbm": KV / v["H"],
        "f_kv_reads": t_kv / plain,
    }


def _per_pass(v: dict[str, Any], st: dict[str, float]) -> dict[str, float]:
    """γ's terms under ``cap`` that fall as the step grows, as B in μ = B/t_step (s): the state carried across a pass
    boundary on chip and in the host's caches and the chaining gap, once a pass of ``copies`` steps, and the step's
    ephemeral traffic."""
    sigma = (v["gpus"] * v["R"] + v["b_host"]) * 1e9
    hbm = v["gpus"] * v["R"] * 1e9 / sigma
    R = v["R"] * 1e9
    c = st["copies"]
    return {"mu_chip": v["s_chip"] / R * hbm / c, "mu_hostcache": v["s_hostcache"] / sigma / c,
            "mu_unreg": st["act"] / R * hbm, "gap": st["gap"] / c}


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
    n_blocks = v["gpus"] * v["H"] / p.block
    k_bind = w.shape.hidden if p.binding == 0 else p.binding
    x_share = 0.0 if v["x_from_transcript"] else p.block * min(1.0, 4.0 * w.batch / k_bind)
    s = v["v_budget"] * 1e9 / 8 * v["t_window"] / (p.block + v["opening_bytes"] + x_share)
    return s, n_blocks


def _lambda(p: Params, v: dict[str, Any], n_blocks: float) -> float:
    if v["lambda_named"]:
        g = GAMMA_MAX
        return -(g * math.log2(g) + (1 - g) * math.log2(1 - g)) / (8 * p.block)
    return max(0.0, v["log2_q"] + math.log2(math.e) - math.log2(n_blocks)) / (8 * p.block)


def _common(w: Workload, p: Params, v: dict[str, Any]) -> dict[str, float]:
    s, n_blocks = _openings(w, p, v)
    return {
        "mu_sample": math.log(2) * v["log2_inv_delta"] / (s * v["horizon"]),
        "lambda": _lambda(p, v, n_blocks),
        "mu_sketch": 2.0 ** -p.k_s,
        "openings": s,
    }


def _rebuild(p: Params, v: dict[str, Any]) -> float:
    return v["s_adv"] * (1 + v["h_host_sq"]) / (p.k * v["x1"] * sq_ratio(v["bits"]))


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
    sigma = v["gpus"] * v["R"] + v["b_host"]
    hbm = v["gpus"] * v["R"] / sigma
    once = _per_pass(v, st)
    t = {
        "mu_chip": once["mu_chip"] / st["cap"],
        "mu_hostcache": once["mu_hostcache"] / st["cap"],
        "mu_rebuild": _rebuild(p, v) * hbm,
        "mu_partial": _partial(p, v) * hbm,
        "mu_host": (v["b_host"] - st["host_stream"] - st["host_cpu"]) / sigma,
        "mu_unreg": once["mu_unreg"] / st["cap"],
        "mu_timing": _timing(st["cap"] * st["copies"], st["gap"], v),
        "mu_cpu": interp(v["mu_cpu"], w.batch),
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
        "mu_hostcache": v["s_hostcache"] / (v["gpus"] * store),
        "mu_rebuild": d * _rebuild(p, v),
        "mu_partial": d * _partial(p, v),
        "mu_host": d * (v["b_host"] - st["host_stream"] - st["host_cpu"]) / (v["gpus"] * v["R"]),
        "mu_cpu": interp(v["mu_cpu"], w.batch),
        **_common(w, p, v),
    }
    t["gamma"] = sum(x for n, x in t.items() if n.startswith("mu_") or n == "lambda")
    t["d"] = d
    t["residual"] = 1 - (1 - t["gamma"]) / d
    return t


def link_n(w: Workload, p: Params, v: dict[str, Any], st: dict[str, float]) -> float:
    """μ_link for memory-only helpers over the NIC: the reads a helper can serve from a copy, given what the relay must
    move a byte (ra-bound-receipt's ½·min(H(z), 8|E|) for a block bound to z, H(z) at ``z_entropy`` of |z|; q and the
    output for a sequence's KV; nothing for a block bound to no z), with the NIC shared across them. A helper that can
    sketch at rate computes NCP's z itself, so this prices memory-only helpers only.

    A weight block is bound at ``z_ratio``. Each block needs its own z, so top-up and host blocks are bound (at 1) only
    by the surplus of the step's z beyond its weights, and a block the host CPUs receipt only by z shipped down PCIe:
    F30's window, with the surplus folded to |E| bytes before it ships."""
    sigma = v["gpus"] * v["R"] + v["b_host"]
    hbm = v["gpus"] * v["R"] / sigma
    per_s = hbm / (v["R"] * 1e9 * st["cap"])
    r = z_ratio(w, p)
    surplus = max(0.0, r - 1.0) * st["W"] * per_s
    topup = max(0.0, st["cap"] - st["t_rd"] - st["t_kv"] - st["t_act"]) / st["cap"] * hbm
    host_gpu, host_cpu = st["host_stream"] / sigma, st["host_cpu"] / sigma
    bound_gpu = min(topup + host_gpu, surplus)
    bound_cpu = min(host_cpu, surplus - bound_gpu, v["gpus"] * v["b_pcie_down"] / sigma)
    rho, h = v["rho"], v["z_entropy"]
    classes = [
        (st["t_rd"] / st["cap"] * hbm, rho * h * min(1.0, r)),
        (st["t_kv"] / st["cap"] * hbm, rho * min(1.0, 4.0 * w.shape.hidden / (w.ctx * w.shape.kv_bytes_per_token))),
        (bound_gpu + bound_cpu, rho * h),
        (topup + host_gpu + host_cpu - bound_gpu - bound_cpu, 0.0),
    ]
    budget, relayed = v["b_nic"] / sigma, 0.0
    for share, ratio in sorted(classes, key=lambda c: c[1]):
        take = share if ratio == 0 else min(share, budget / ratio)
        budget -= take * ratio
        relayed += take
    return relayed


def evaluate(w: Workload, p: Params, v: dict[str, Any]) -> dict[str, Any]:
    st = step(w, p, v)
    cap = gamma_cap(w, p, v, st)
    return {
        "step": st,
        "cap": {"overhead": st["cap"] / st["plain"] - 1, **cap},
        "pass": {"overhead": st["pass"] / st["plain"] - 1, **gamma_pass(w, p, v, st)},
        "mu_link_n": link_n(w, p, v, st),
    }
