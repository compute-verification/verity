"""`sampler.select` relation (contract §7) for vLLM Model Runner V2 (0.28.0 and 0.28.1rc1.dev472).

What vLLM's V2 sampler does (vllm/v1/worker/gpu/sample/{sampler,gumbel,states}.py):

  logits (bf16 [B, V]) -> (fp32 copy if any request in the batch needs processing)
    -> logit bias / penalties / bad words / thinking budget (none in our profile)
    -> temperature: logits / T in fp32 (skipped when T in {0, 1})
    -> min_p (none) -> top_k / top_p masking to -inf (vllm.v1.sample.ops.topk_topp_sampler.
       apply_top_k_top_p: Triton kernel for every batch on the nightly; on 0.28.0 Triton when
       B >= 8 rows, torch sort otherwise.  The two differ at ties on the k-th logit, see
       `nucleus_masks`.)
    -> gumbel_sample: per row, per token v:
         T == 0:  argmax(logits) (fp32, first index wins ties, per 1024-block then across blocks)
         T != 0:  gumbel_seed = tl.randint(seed, pos)            # Philox4x32-10, key = seed
                  u        = tl.rand(gumbel_seed, v)              # Philox4x32-10, key = gumbel_seed, ctr = v
                  u        = max(u, 2^-31 * (1 - 2^-24))          # _TL_RAND_MIN clamp
                  g        = -log(-log1p(-u))                     # fp32 libdevice
                  argmax_v (logits[v] + g[v])                    # first index wins ties

  `seed` is SamplingParams.seed (int64; a fresh np.random int when None), `pos` is the
  absolute position of the input token whose logits are being sampled (= prompt_len - 1 + k
  for the k-th output token, k >= 0).  There is NO stateful generator: the noise is a pure
  function of (seed, pos, token id).  FlashInfer sampling is bypassed whenever a request in the
  batch has an explicit seed or is greedy.

Two checker classes are provided (contract §7):

  * `independent_select`: numpy re-derivation of the whole chain (Philox in integer arithmetic,
    fp32 IEEE arithmetic for the uniform mapping and the additions, float64 for the nucleus
    boundary).  The only pieces that cannot be reproduced bit-exactly on a CPU are the CUDA
    libdevice `logf`/`log1pf` (<= 1-2 ulp) and the top-p boundary reduction order, so the checker
    computes a *certified* answer: it returns the argmax together with `certified=True` only
    when no other token could overtake under those error bounds; otherwise `certified=False`
    and the reason.  An uncertified prediction is still compared, but counted separately.
  * `same_runtime_select`: calls vLLM's own `apply_top_k_top_p` and `gumbel_sample` on the
    committed row on the GPU in a fresh process.

Both are relations over (logits row bytes + dtype, SamplingParams, seed, pos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Triton philox constants (32-bit variant), python/triton/language/random.py
PHILOX_KEY_A = np.uint32(0x9E3779B9)
PHILOX_KEY_B = np.uint32(0xBB67AE85)
PHILOX_ROUND_A = np.uint32(0xD2511F53)
PHILOX_ROUND_B = np.uint32(0xCD9E8D57)
N_ROUNDS = 10

# tl.uint_to_uniform_float scale (fp32 0x2FFFFFFF) and vLLM's _TL_RAND_MIN clamp
RAND_SCALE = np.float32(4.6566127342e-10)
TL_RAND_MIN = np.float32(4.6566127342e-10)

# libdevice error budget used for certification (ulps of fp32).  CUDA math API documents
# logf / log1pf at <= 1 ulp; we allow 2 to be conservative about the composition.
LOG_ULP_BUDGET = 2

_MASK32 = np.uint64(0xFFFFFFFF)


# ----------------------------------------------------------------------------- philox
def _umulhi32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a.astype(np.uint64) * b.astype(np.uint64)) >> np.uint64(32)).astype(np.uint32)


def _mullo32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint64) * b.astype(np.uint64)).astype(np.uint32)


def philox4x32_10(c0, c1, c2, c3, k0: np.uint32, k1: np.uint32):
    """Triton `philox_impl` (32-bit words), vectorised over the counter arrays."""
    c0 = np.asarray(c0, dtype=np.uint32)
    c1 = np.asarray(c1, dtype=np.uint32)
    c2 = np.asarray(c2, dtype=np.uint32)
    c3 = np.asarray(c3, dtype=np.uint32)
    k0 = np.uint32(k0)
    k1 = np.uint32(k1)
    with np.errstate(over="ignore"):
        for _ in range(N_ROUNDS):
            _c0, _c2 = c0, c2
            c0 = _umulhi32(PHILOX_ROUND_B, _c2) ^ c1 ^ k0
            c2 = _umulhi32(PHILOX_ROUND_A, _c0) ^ c3 ^ k1
            c1 = _mullo32(PHILOX_ROUND_B, _c2)
            c3 = _mullo32(PHILOX_ROUND_A, _c0)
            k0 = np.uint32((int(k0) + int(PHILOX_KEY_A)) & 0xFFFFFFFF)
            k1 = np.uint32((int(k1) + int(PHILOX_KEY_B)) & 0xFFFFFFFF)
    return c0, c1, c2, c3


def _seed_words(seed: int) -> tuple[np.uint32, np.uint32]:
    """`seed.to(tl.uint64)` then split: works for int64 (two's complement) and uint32 seeds."""
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    return np.uint32(s & 0xFFFFFFFF), np.uint32((s >> 32) & 0xFFFFFFFF)


def tl_randint(seed: int, offset) -> np.ndarray:
    """`tl.randint(seed, offset)` for int64 offsets (c0 = low 32, c1 = high 32 of the offset)."""
    off = np.asarray(offset, dtype=np.int64)
    lo = (off & np.int64(0xFFFFFFFF)).astype(np.uint32)
    hi = ((off >> np.int64(32)) & np.int64(0xFFFFFFFF)).astype(np.uint32)
    z = np.zeros_like(lo)
    k0, k1 = _seed_words(seed)
    return philox4x32_10(lo, hi, z, z, k0, k1)[0]


def tl_randint_32(seed: int, offset) -> np.ndarray:
    """`tl.randint(seed, offset)` for int32 offsets (c1 = 0)."""
    off = np.asarray(offset, dtype=np.int64)
    lo = (off & np.int64(0xFFFFFFFF)).astype(np.uint32)
    z = np.zeros_like(lo)
    k0, k1 = _seed_words(seed)
    return philox4x32_10(lo, z, z, z, k0, k1)[0]


def uint_to_uniform_float(x: np.ndarray) -> np.ndarray:
    """Triton `uint_to_uniform_float` for 32-bit words: |int32| (via ~x for negatives) * scale."""
    xi = x.astype(np.uint32).view(np.int32)
    xi = np.where(xi < 0, ~xi, xi)
    return xi.astype(np.float32) * RAND_SCALE


def tl_rand32(seed: int, offset) -> np.ndarray:
    return uint_to_uniform_float(tl_randint_32(seed, offset))


def gumbel_seed_for(seed: int, pos: int, seed_is_signed_int32: bool = False) -> int:
    """`tl.randint(seed, pos)` with `pos` int64.  Returns the python int that Triton feeds to
    the second philox as the *seed* (`.to(tl.uint64)`), i.e. zero-extended uint32 unless the
    installed Triton types randint's result as int32 (then sign-extended)."""
    g = int(tl_randint(seed, np.asarray([pos], dtype=np.int64))[0])
    if seed_is_signed_int32 and g >= 2**31:
        g -= 2**32  # sign-extend to int64 (two's complement into uint64 in _seed_words)
    return g


def gumbel_noise(seed: int, pos: int, vocab_size: int, *, use_fp64: bool = False,
                 seed_is_signed_int32: bool = False, dtype=np.float32) -> np.ndarray:
    """The noise vector vLLM adds to `logits[pos]` for request `seed` (fp32 model)."""
    gs = gumbel_seed_for(seed, pos, seed_is_signed_int32)
    keys = np.arange(vocab_size, dtype=np.int32)
    if use_fp64:
        raise NotImplementedError("use_fp64_gumbel=True is not part of the frozen profile")
    u = tl_rand32(gs, keys)
    u = np.maximum(u, TL_RAND_MIN)
    with np.errstate(divide="ignore"):
        if dtype is np.float32:
            return (-np.log(-np.log1p(-u))).astype(np.float32)
        return -np.log(-np.log1p(-u.astype(np.float64)))


# ----------------------------------------------------------------------------- helpers
def logits_from_bytes(raw: bytes, dtype: str) -> np.ndarray:
    """Committed row bytes -> fp32 numpy (bf16 widening is exact)."""
    if dtype in ("bfloat16", "bf16"):
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
        return u16.view(np.float32)
    if dtype in ("float32", "fp32"):
        return np.frombuffer(raw, dtype=np.float32).copy()
    if dtype in ("float16", "fp16"):
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    raise ValueError(f"unsupported logits dtype {dtype!r}")


def effective_params(temperature: float, top_k: int, top_p: float, vocab_size: int) -> tuple[float, int, float]:
    """SamplingParams normalisation + SamplingStates.add_request."""
    if temperature < 1e-5:  # vllm.sampling_params._SAMPLING_EPS -> greedy
        return 0.0, vocab_size, 1.0
    if top_k <= 0 or top_k > vocab_size:
        top_k = vocab_size
    return float(temperature), int(top_k), float(top_p)


def fp32_ulp(x: np.ndarray) -> np.ndarray:
    return np.spacing(np.abs(x).astype(np.float32)).astype(np.float32)


def noise_error_budget(noise: np.ndarray, y: np.ndarray, log_ulp_budget: int = LOG_ULP_BUDGET) -> np.ndarray:
    """Absolute bound on |libdevice noise - numpy noise| for noise = -log(y), y = -log1p(-u).

    `log_ulp_budget` ulps of the outer result, plus the inner log1p error propagated through
    d(-log y)/dy = 1/y (ulp(y)/y ~ 2^-24), plus a 2^-24 absolute floor for results near zero
    where "ulp of the result" is not a meaningful error measure for a libm-style implementation.
    Validated on the pod against the Triton kernel (pod/validate_rng.py: `budget_violations` = 0).
    """
    unc = log_ulp_budget * fp32_ulp(noise)
    unc = unc + log_ulp_budget * fp32_ulp(y) / np.maximum(y, np.float32(1e-38))
    unc = unc + np.float32(log_ulp_budget * 2.0**-24)
    return unc.astype(np.float32)


@dataclass
class SelectResult:
    predicted: int
    certified: bool
    cls: str  # "independent" | "same_runtime"
    margin: float | None = None  # fp32 gap between winner and runner-up score (after masking)
    runner_up: int | None = None
    reason: str | None = None  # why not certified
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = {"predicted": int(self.predicted), "certified": bool(self.certified), "class": self.cls,
             "margin": None if self.margin is None else float(self.margin),
             "runner_up": None if self.runner_up is None else int(self.runner_up),
             "reason": self.reason}
        d.update(self.details)
        return d


# ----------------------------------------------------------------------------- greedy
def greedy_select(logits_f32: np.ndarray) -> SelectResult:
    """vLLM greedy = first argmax over fp32(logits) (block-wise tl.max tie_break_left, then
    torch.argmax over block maxima which also returns the first)."""
    x = logits_f32.astype(np.float32, copy=False)
    tok = int(np.argmax(x))
    top = x[tok]
    others = np.where(x == top)[0]
    x2 = x.copy()
    x2[tok] = -np.inf
    ru = int(np.argmax(x2))
    return SelectResult(predicted=tok, certified=True, cls="independent",
                        margin=float(top - x2[ru]), runner_up=ru,
                        details={"ties_at_max": int(len(others)), "mode": "greedy"})


# ----------------------------------------------------------------------------- nucleus
TOPK_TIES = ("exact_k", "keep_all")


def nucleus_masks(logits_f32: np.ndarray, top_k: int, top_p: float, tol_p: float = 1e-4,
                  topk_ties: str = "exact_k"):
    """Return (keep, ambiguous) boolean masks for top-k then top-p over the row.

    Two vLLM implementations exist and they differ at ties on the k-th largest logit — common with
    bf16 logits (quantised to 1/16..1/8 at these magnitudes): ties at the 50th logit occurred at
    93 % of the positions of this workload.

    * `topk_ties="exact_k"` — the Triton kernel (`topk_topp_triton.py`, `num_keep` duplicate
      handling): exactly k tokens survive; surplus tokens equal to the k-th value are dropped in
      vocabulary order (lowest ids kept).  Used by vLLM 0.28.1rc1.dev472 for every batch size and
      by 0.28.0 for batches of >= 8 rows.
    * `topk_ties="keep_all"` — `apply_top_k_top_p_pytorch`: every token >= the k-th value is
      kept (the set can exceed k).  0.28.0 for batches of < 8 rows.

    The softmax is renormalised over the surviving set — so the tie rule moves the nucleus
    boundary by whole tokens, far from any rounding tolerance — and a token is kept by top-p iff
    the probability mass strictly above it (descending logit; equal logits in vocabulary order) is
    < p (the minimal set whose mass reaches p; both implementations agree on this).  `ambiguous`
    marks tokens whose fate depends on rounding or on an equal-logit group straddling the p
    boundary: equal to the k-th logit (exact_k: only the ones actually at the cut), mass-above
    within tol_p of p, or members of a tie group that the p cut splits.
    """
    if topk_ties not in TOPK_TIES:
        raise ValueError(f"topk_ties must be one of {TOPK_TIES}")
    V = logits_f32.shape[0]
    x = logits_f32.astype(np.float64)
    keep = np.ones(V, dtype=bool)
    amb = np.zeros(V, dtype=bool)
    if top_k < V:
        kth = np.partition(x, V - top_k)[V - top_k]
        ties = np.flatnonzero(x == kth)
        if topk_ties == "keep_all" or len(ties) == 1:
            keep &= x >= kth
        else:
            above = x > kth
            n_keep_ties = top_k - int(above.sum())
            keep = above
            keep[ties[:n_keep_ties]] = True
            # the tokens on either side of the cut inside the tie group are order-defined
            amb[ties[max(0, n_keep_ties - 1):n_keep_ties + 1]] = True
        if len(ties) > 1 and topk_ties == "keep_all":
            amb |= x == kth
    if top_p < 1.0:
        xk = np.where(keep, x, -np.inf)
        m = np.max(xk)
        e = np.exp(xk - m)
        probs = e / e.sum()
        order = np.argsort(-xk, kind="stable")  # descending; equal logits in vocabulary order
        p_sorted = probs[order]
        mass_above_sorted = np.concatenate(([0.0], np.cumsum(p_sorted)[:-1]))
        mass_above = np.empty(V)
        mass_above[order] = mass_above_sorted
        keep_p = mass_above < top_p
        # boundary tolerance: fp32 cumsum / pivot search error
        amb |= keep & (np.abs(mass_above - top_p) <= tol_p)
        # an equal-logit group split by the p cut: which members survive is order-defined
        cut = keep & ~keep_p
        if cut.any():
            boundary_val = np.max(xk[cut])
            group = keep & (xk == boundary_val)
            if group.sum() > 1 and (group & keep_p).any():
                amb |= group
        keep &= keep_p
    return keep, amb


# ----------------------------------------------------------------------------- independent
def independent_select(
    logits_f32: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int | None,
    pos: int,
    *,
    seed_is_signed_int32: bool = False,
    log_ulp_budget: int = LOG_ULP_BUDGET,
    topk_ties: str | None = "exact_k",
) -> SelectResult:
    """Predict the token vLLM V2 samples at `pos` from the committed logits row.

    `topk_ties` pins which top-k tie rule production used (see `nucleus_masks`): "exact_k" (Triton
    kernel; the POC pin 0.28.1rc1.dev472 at every batch size), "keep_all" (torch sort path; 0.28.0
    below 8 rows), or None when the path is not known — then both are evaluated and the prediction
    is certified only if they agree on the token.
    """
    V = logits_f32.shape[0]
    T, k, p = effective_params(temperature, top_k, top_p, V)
    if T == 0.0:
        return greedy_select(logits_f32)
    if seed is None:
        return SelectResult(predicted=-1, certified=False, cls="independent",
                            reason="no seed recorded: vLLM drew np.random seed; not reproducible")
    if topk_ties is None:
        a = independent_select(logits_f32, temperature, top_k, top_p, seed, pos, seed_is_signed_int32=seed_is_signed_int32,
                               log_ulp_budget=log_ulp_budget, topk_ties="exact_k")
        b = independent_select(logits_f32, temperature, top_k, top_p, seed, pos, seed_is_signed_int32=seed_is_signed_int32,
                               log_ulp_budget=log_ulp_budget, topk_ties="keep_all")
        a.details["topk_ties"] = "unpinned"
        a.details["keep_all_prediction"] = b.predicted
        if a.predicted != b.predicted:
            a.certified = False
            a.reason = ("; ".join(filter(None, [a.reason, f"top-k tie rule not pinned: exact_k -> {a.predicted}, "
                                                          f"keep_all -> {b.predicted}"])))
        elif not b.certified and a.certified:
            a.certified = False
            a.reason = f"keep_all path uncertified: {b.reason}"
        return a

    x = logits_f32.astype(np.float32, copy=True)
    details: dict[str, Any] = {"mode": "gumbel", "pos": int(pos), "seed": int(seed),
                               "temperature": T, "top_k": k, "top_p": p, "topk_ties": topk_ties}
    scale_unc = np.zeros(V, dtype=np.float32)
    if T != 1.0:
        # Triton fp32 division; allow 2 ulp for div.full-style approximation.
        x = (x / np.float32(T)).astype(np.float32)
        scale_unc += 2 * fp32_ulp(x)
        details["temperature_division"] = "fp32 with 2-ulp budget"

    keep, amb = nucleus_masks(x, k, p, topk_ties=topk_ties)
    u = np.maximum(tl_rand32(gumbel_seed_for(seed, pos, seed_is_signed_int32),
                             np.arange(V, dtype=np.int32)), TL_RAND_MIN)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = (-np.log1p(-u)).astype(np.float32)  # in (2^-31, ~21.5]
        noise = (-np.log(y)).astype(np.float32)
    noise_unc = noise_error_budget(noise, y, log_ulp_budget)

    score = (x + noise).astype(np.float32)
    unc = (scale_unc + noise_unc + fp32_ulp(score)).astype(np.float32)  # + final add rounding
    masked_score = np.where(keep, score, -np.inf).astype(np.float32)
    win = int(np.argmax(masked_score))
    s_win = masked_score[win]
    tmp = masked_score.copy()
    tmp[win] = -np.inf
    ru = int(np.argmax(tmp))
    margin = float(s_win - tmp[ru]) if np.isfinite(tmp[ru]) else float("inf")

    reasons = []
    # (1) another kept token could overtake within the error budget
    lo_win = s_win - unc[win]
    overtake = keep & (score + unc >= lo_win)
    overtake[win] = False
    if overtake.any():
        reasons.append(f"kept token(s) {np.where(overtake)[0][:5].tolist()} within fp32 error budget of winner")
    # (2) a masked token that would beat the winner sits at an ambiguous boundary
    beats = (~keep) & (score + unc >= lo_win)
    if (beats & amb).any():
        reasons.append(f"masked token(s) {np.where(beats & amb)[0][:5].tolist()} at nucleus boundary outscore winner")
    # (3) the winner itself is at an ambiguous boundary
    if amb[win]:
        reasons.append("winner is at the top-k/top-p boundary")
    certified = not reasons
    details.update({"kept": int(keep.sum()), "ambiguous_boundary_tokens": int(amb.sum()),
                    "noise_at_winner": float(noise[win]), "logit_at_winner": float(x[win]),
                    "n_masked_outscoring": int(((~keep) & (score > s_win)).sum())})
    return SelectResult(predicted=win, certified=certified, cls="independent", margin=margin,
                        runner_up=ru, reason="; ".join(reasons) or None, details=details)


# ----------------------------------------------------------------------------- same runtime
def same_runtime_select(
    logits_f32: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int | None,
    pos: int,
    *,
    topk_topp_impl: str = "auto",  # "auto" (batch-size rule, i.e. torch for 1 row) | "triton" | "torch"
    device: str = "cuda",
) -> SelectResult:
    """Run vLLM's own kernels on the committed row (needs torch + vllm + the GPU)."""
    import torch
    from vllm.v1.sample.ops.topk_topp_sampler import (
        apply_top_k_top_p,
        apply_top_k_top_p_pytorch,
    )
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    V = logits_f32.shape[0]
    T, k, p = effective_params(temperature, top_k, top_p, V)
    logits = torch.as_tensor(np.ascontiguousarray(logits_f32), dtype=torch.float32, device=device).view(1, V)
    mapping = torch.zeros(1, dtype=torch.int32, device=device)
    temp_t = torch.tensor([T], dtype=torch.float32, device=device)
    if T == 0.0:
        seed_t = torch.zeros(1, dtype=torch.int64, device=device)
    else:
        if seed is None:
            return SelectResult(predicted=-1, certified=False, cls="same_runtime", reason="no seed recorded")
        seed_t = torch.tensor([int(seed)], dtype=torch.int64, device=device)
    pos_t = torch.tensor([int(pos)], dtype=torch.int64, device=device)
    if T not in (0.0, 1.0):
        from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
        apply_temperature(logits, mapping, temp_t)
    kt = torch.tensor([k], dtype=torch.int32, device=device) if k < V else None
    pt = torch.tensor([p], dtype=torch.float32, device=device) if p < 1.0 else None
    impl_used = "none"
    if T != 0.0 and (kt is not None or pt is not None):
        if topk_topp_impl == "triton":
            from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
            logits = apply_top_k_top_p_triton(logits, kt, pt)
            impl_used = "triton"
        elif topk_topp_impl == "torch":
            logits = apply_top_k_top_p_pytorch(logits, kt, pt)
            impl_used = "torch_sort"
        else:
            logits = apply_top_k_top_p(logits, kt, pt)
            impl_used = "auto(torch_sort for 1 row)"
    sampled = _gumbel_sample_compat(gumbel_sample, logits, mapping, temp_t, seed_t, pos_t)
    tok = int(sampled.item())
    return SelectResult(predicted=tok, certified=True, cls="same_runtime",
                        details={"topk_topp_impl": impl_used, "pos": int(pos), "seed": seed})


def _gumbel_sample_compat(gumbel_sample, logits, mapping, temp_t, seed_t, pos_t):
    """0.28.0: gumbel_sample(..., apply_temperature); nightly 0.28.1rc1.dev472 adds a required
    `is_drafting` (speculative-decoding noise salt; False for the target model)."""
    import inspect
    params = inspect.signature(gumbel_sample).parameters
    kw = {"apply_temperature": False}
    if "is_drafting" in params:
        kw["is_drafting"] = False
    return gumbel_sample(logits, mapping, temp_t, seed_t, pos_t, **kw)
