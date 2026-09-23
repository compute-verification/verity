"""Ligerito PCS (torch prover, numpy verifier) -- lane ligerito-proto.  Public surface for the other Ligerito lanes:

    prove(f, z, dims, timer=None, coins=None) -> Proof     f (N,) int32 device tensor (multilinear coefficients over {0,1}^n,
                                                           bit j of the index <-> variable j), z (n, 6) F_{p^6} point, dims from
                                                           ``dims.choose(n, queries)`` (or ligerito-design's params); commit is
                                                           round 1 (``proof.roots[0]``), the proof carries v = f(z) (``proof.v``)
    verify(proof, z, v=None, coins=None) -> (bool, reason)
    Proof.to_bytes() / Proof.from_bytes()

Randomness only through ``transcript.py`` (Coins protocol).  Implementation in ``proto/``.
"""
from .proto.dims import Dims, choose                       # noqa: F401
from .proto.pcs import Proof, Reject, Timer, prove, verify  # noqa: F401

__all__ = ["Dims", "Proof", "Reject", "Timer", "choose", "prove", "verify"]
