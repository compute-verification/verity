"""The stable import surface of the packed sumcheck + logUp GPU kernels (for the ``a-gpu`` prover).

Everything here is bit-exact against the pure-Python references ``reference.py`` (checker sumcheck) and
``logup_reference.py`` (logUp fractional GKR): same coins in, same messages out.

Conventions (read these before calling anything)
------------------------------------------------
* **Field**: BabyBear ``p = 2^31 - 2^27 + 1``; extension ``F_{p^6} = F_p[X] / (X^6 - W)``, ``W = reference.W``.
* **Base table**: ``torch.int32`` of shape ``(t,)``, values in ``[0, p)`` (plain, NOT Montgomery form).
* **Extension table**: ``torch.int32`` of shape ``(t, 6)``, row = the 6 coefficients in ``[0, p)``, plain.  This is
  the *interleaved* layout (``x[i, s]`` = coefficient ``s`` of element ``i``), not planar ``6 x (t,)``; use
  ``x.t().contiguous()`` / ``x.view(6, t).t()`` at the boundary if your prover is planar.
* **Montgomery**: internal only.  The Triton passes multiply in Montgomery form (``R = 2^32``) and compensate through
  ``R``-pre-scaled challenges / eq-weight limbs; every tensor that crosses a function boundary here is plain.
* **Binding order**: a sumcheck over ``2^n`` points binds variable 0 (bit 0 of the index) first: round ``i`` pairs
  ``(2y, 2y+1)`` of the current table.  ``eq(gamma, b)`` puts ``gamma[0]`` on bit 0 (``reference.eq_table``).
  ``logup.rs`` binds the top bit first; the two are the same protocol under bit-reversal of the table index
  (``logup_reference`` docstring).
* **Extension elements on the host**: ``reference.Ext`` = a 6-tuple of Python ints.
* **Shapes**: base tables need ``t`` a multiple of ``2^k . 2^13`` for the fused packed rounds (the inner-product depth
  ``K1 = 8192``; smaller ``K1`` via ``k1_log2``); levels below ``128`` pairs per eq chunk run in torch.
* **Device**: CUDA only (Triton 3.2, torch 2.6 / cu124 tested; H100).  The first call compiles the generated kernels
  (~1-2 min; cached under ``$PACKED_KERNEL_CACHE`` or the temp dir).

Test vectors: ``python -m backends.gkr.packed.kernels --vectors`` prints the reference transcripts of two fixed small
cases (checker sumcheck at ``t = 2^16, k = 3``; logUp at ``Nw = 12288, Nt = 2048, k = 3``) as SHA-256 digests; the
GPU path must reproduce them (``tests/test_kernel_agreement.py``, ``tests/test_logup_gpu.py`` check message by message).
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable

import torch

from . import logup_gpu as _lg
from . import logup_graph as _lgr
from . import logup_reference as logup_reference
from . import reference as reference
from . import sumcheck_packed as _sp

Ext = reference.Ext
P = reference.P

__all__ = ["Ext", "P", "SumcheckKernel", "sumcheck_layer", "sumcheck_layer_device", "prepare_inputs", "fold", "LogUpProver",
           "GraphLogUp", "logup_prove", "logup_prove_graphed", "logup_round", "logup_fold", "multiplicities", "leaf_level",
           "tree_level", "reference", "logup_reference"]


# ---------------------------------------------------------------------------------------------------------------------
# checker sumcheck (one GKR layer)
# ---------------------------------------------------------------------------------------------------------------------

SumcheckKernel = _sp.Kernel


def make_kernel(device: str = "cuda", k1_log2: int = _sp.K1_LOG2_DEFAULT, round_fusion: bool = True) -> SumcheckKernel:
    """The kernel object (holds the Montgomery constants and the generated Triton module); make one and reuse it.

    ``round_fusion`` (a-fusion, default on): the ``k`` packed rounds are served by ONE read of ``g, h`` (the ``U`` tensor,
    ``kernels_fused``; 3.48 vs 7.83 ms at 2^28 k=3) -- same messages, different summation order.  ``fold_fusion``
    (``Kernel(fold_fusion=True)``) fuses the fold into the next Hadamard pass and is OFF: measured slower (SIMT-bound)."""
    return _sp.Kernel(device, k1_log2=k1_log2, mode="triton", round_fusion=round_fusion)


def sumcheck_layer(kernel: SumcheckKernel, gammas: list[Ext], g: torch.Tensor, h: torch.Tensor, k: int,
                   challenger: Callable[[int, list[Ext]], Ext]) -> tuple[list[list[Ext]], list[Ext], tuple[Ext, Ext, Ext]]:
    """One eq-weighted degree-3 layer sumcheck ``sum_b eq(gammas, b) g(b) h(b)`` with ``k`` packed base rounds.

    ``g, h``: base tables ``(t,) int32`` plain, ``t = 2^len(gammas)``.  ``challenger(i, S_i)`` returns ``r_i`` given the
    round-``i`` message ``S_i = [S(0), S(1), S(2), S(3)]`` (4 ``Ext``): this is where Fiat-Shamir goes (one host
    round trip per round).  Returns ``(rounds, rs, (eq(gammas, rs), g(rs), h(rs)))`` -- identical to
    ``reference.packed_prover``.  ``k = 3`` is the measured optimum on the H100 (``README.md``)."""
    return _sp.prove(kernel, gammas, g, h, k, challenger)


prepare_inputs = _sp.prepare_inputs


def sumcheck_layer_device(kernel: SumcheckKernel, gammas: list[Ext], coins: list[Ext], g: torch.Tensor, h: torch.Tensor, k: int,
                          inputs: list[_sp.RoundInputs] | None = None) -> list[torch.Tensor]:
    """:func:`sumcheck_layer` without host synchronisation: the coins are known ahead (``coins[i-1] = r_i``; benchmarks,
    or a prover that derives the layer's coins from an earlier commitment) and every host-side operand comes from
    :func:`prepare_inputs` (eq tables, fold operands, ``R``-scaled challenges -- reusable across proofs with the same coins).
    Returns the ``n`` messages ``S_i (4, 6) int32`` plain as device tensors, bit-identical to ``sumcheck_layer``'s.  The
    whole call is CUDA-graph capturable (``bench_fused.py`` times it as one graph: 12.65 ms at 2^28 k=3)."""
    if inputs is None:
        inputs = _sp.prepare_inputs(kernel, gammas, coins, k)
    return _sp.prove_device(kernel, gammas, coins, g, h, k, inputs)


def fold(kernel: SumcheckKernel, table: torch.Tensor, rs: list[Ext]) -> torch.Tensor:
    """Bind the low ``len(rs)`` variables of a table by ``rs``: ``out(b) = sum_a eq(rs, a) table(a, b)``.

    ``table`` base ``(t,)`` -> extension ``(t / 2^k, 6)`` (``fold_k``, one pass), or extension ``(t, 6)`` -> ``(t / 2^k, 6)``
    (``len(rs)`` SIMT passes).  Plain in, plain out."""
    tk = kernel._tk
    if table.dim() == 1:
        k = len(rs)
        coeffs_R = _sp.pre_scale(_sp.to_dev(reference.eq_table(rs), kernel.device))
        return tk.fold_k(table, k, coeffs_R)
    out = table
    for r in rs:
        m = out.shape[0] // 2
        r_R = _sp.pre_scale(_sp.to_dev([r], kernel.device))[0].contiguous()
        out = tk.fold_ext(out.view(m, 12), r_R)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# logUp fractional GKR
# ---------------------------------------------------------------------------------------------------------------------

LogUpProver = _lg.LogUpProver
multiplicities = _lg.multiplicities


def logup_prove(prover: LogUpProver, w: torch.Tensor, table: torch.Tensor, z: Ext, coins: logup_reference.Coins,
                mult: torch.Tensor | None = None):
    """The whole logUp GKR for one table: ``w (Nw,) int32`` queries (natural order; ``Nw`` a multiple of ``2^(k+1) . 128``
    and aligned so every region is a multiple of 128 eq chunks through round ``k`` -- ``Nw = units x 306`` and
    ``Nt = 65536`` are), ``table (Nt,) int32``, ``z`` the logUp challenge.  ``coins.next()`` supplies ``lambda_0`` then
    per level ``r_1..r_k, mu, lambda`` (replace with the Fiat-Shamir sponge).  Returns
    ``(Transcript, LeafClaim, Timing)``; ``Transcript.q_root`` and per level ``rounds`` (4 ``Ext`` each) / ``vals``
    (``p0, p1, q0, q1``) are exactly ``logup_reference.prove_table``'s."""
    return prover.prove(w, table, z, coins, mult)


GraphLogUp = _lgr.GraphLogUp


def logup_prove_graphed(prover: GraphLogUp, w: torch.Tensor, table: torch.Tensor, z: Ext, coins: logup_reference.Coins):
    """:func:`logup_prove` on CUDA graphs (a-fusion; v1 shape 427 -> 89 ms device / 92 ms wall).  ``prover =
    GraphLogUp("cuda", k=3, rounds_per_graph=None | 1)``: ``None`` = one graph per level with the level's coins written
    into the static device coin buffer before replay (coins must be known ahead of the level: a transcript-independent
    coin source, or a sponge fed only by earlier levels); ``1`` = one graph per round, the message ``S_i`` synced to the
    host and ``r_i`` written between replays (drop-in for Fiat-Shamir; 103.5 / 107.1 ms).  Both keep the transcript
    identical: the round constants (``eq(r, .)`` tables, ``prefix``, ``eq_x``, ``r R``) are computed ON DEVICE from the
    coin buffer (``kernels_graph.py``), so replaying with new coins evaluates exactly the eager prover's arithmetic.  The
    first call at a shape compiles and captures (~1 min); reuse the prover for that shape.  Returns ``(Transcript,
    LeafClaim, Timing)`` with ``logup_reference.prove_table``'s ``q_root`` / ``rounds`` / ``vals``."""
    return prover.prove(w, table, z, coins)


def logup_round(prover: LogUpProver, T4: torch.Tensor, rho: list[Ext], rs: list[Ext], i: int, lam: Ext) -> torch.Tensor:
    """One generic logUp level round: ``T4 (4, t_i, 6) int32`` = ``p0, p1, q0, q1`` (children de-interleaved by parity),
    the level's claim point ``rho`` (``len(rho)`` rounds), the challenges so far ``rs = [r_1..r_{i-1}]``; returns
    ``S_i(0..3)`` as ``(4, 6) int32`` (plain) for ``sum_y eq(rho, y) [p0 q1 + p1 q0 + lambda q0 q1]``."""
    lam6 = _sp.to_dev([lam], prover.device)
    return prover.generic_round(T4, rho, rs, i, lam6, _sp.pre_scale(lam6)[0].contiguous())


def logup_fold(prover: LogUpProver, T4: torch.Tensor, r: Ext) -> torch.Tensor:
    """``(4, t_i, 6) -> (4, t_i / 2, 6)``: all four tables bound by ``r`` (pairs ``(2y, 2y+1)``)."""
    return prover._fold4(T4, r)


def leaf_level(prover: LogUpProver, w: torch.Tensor, z: Ext, n: int) -> torch.Tensor:
    """Level ``n - 1`` of the fraction tree straight from base queries: ``(2, 2^(n-1), 6)`` = ``p, q`` de-interleaved
    (``[evens | odds]``), witness pairs filled, the rest ``(0, 1)``."""
    dev = prover.device
    z6 = _sp.to_dev([z], dev)
    pq = torch.empty((2, 1 << (n - 1), 6), dtype=torch.int32, device=dev)
    pq[0] = 0
    pq[1] = prover.one6
    zR = _sp.pre_scale(z6)[0].contiguous()
    z2 = _sp.to_dev([reference.mul(z, z)], dev)[0].contiguous()
    prover.tk.leaf_level(w, z6[0].contiguous(), zR, z2, pq[0], pq[1], 1 << (n - 2))
    return pq


def tree_level(prover: LogUpProver, pq: torch.Tensor) -> torch.Tensor:
    """``(2, 2m, 6)`` de-interleaved level -> ``(2, m, 6)`` de-interleaved parents ``(p0 q1 + p1 q0, q0 q1)``."""
    return prover.tk.tree_level(pq)


# ---------------------------------------------------------------------------------------------------------------------
# test vectors (reference side; runs on any machine)
# ---------------------------------------------------------------------------------------------------------------------


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def test_vectors() -> dict:
    """Deterministic small cases and the SHA-256 of their reference transcripts."""
    out = {}
    # checker sumcheck: t = 2^12, k = 3, seed 7 (keep the pure-Python reference under a second)
    rng = random.Random(7)
    n, k = 12, 3
    gammas = [reference.random_ext(rng) for _ in range(n)]
    g = [rng.randrange(P) for _ in range(1 << n)]
    h = [rng.randrange(P) for _ in range(1 << n)]
    coins = [reference.random_ext(rng) for _ in range(n)]
    rounds, rs, final = reference.packed_prover(gammas, g, h, k, lambda i, S: coins[i - 1])
    out["sumcheck_t4096_k3_seed7"] = {"inputs": {"n": n, "k": k, "seed": 7, "note": "gammas, g, h, coins drawn in that order from random.Random(7) via reference.random_ext / randrange(P)"},
                                      "sha256(rounds, rs, final)": _digest([rounds, rs, list(final)])}
    # logUp: Nw = 12288, Nt = 2048 (range table), z from Random(1), coins Coins(1)
    table = list(range(2048))
    rng = random.Random(1)
    queries = [rng.randrange(2048) for _ in range(12288)]
    z = reference.random_ext(rng)
    mult = logup_reference.multiplicities(queries, table)
    p, q, nn = logup_reference.leaves(queries, table, mult, z)
    tr, claim = logup_reference.prove_table(p, q, nn, logup_reference.Coins(1))
    out["logup_Nw12288_Nt2048_seed1"] = {"inputs": {"Nw": 12288, "Nt": 2048, "seed": 1, "note": "queries then z from random.Random(1); coins Coins(1); k = 3 on the GPU"},
                                          "sha256(q_root, levels, claim)": _digest([tr.q_root, tr.levels, [claim.rho, claim.lam, claim.value]])}
    return out


if __name__ == "__main__":
    import sys

    if "--vectors" in sys.argv:
        print(json.dumps(test_vectors(), indent=1))
