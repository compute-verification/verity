"""The chain test's field term (``chain.py``, "More than three linked rows"; red-team-bligero-real-k finding A1), without
torch, so that ``protocol.soundness`` / ``protocol.config_for`` (which size and book it) and ``cell register`` (which
books it on a result produced before it was booked) compute the same number.

Extra constraint ``e`` of a column (link extras, then start extras, then fingerprint extras) has the monomial coefficient
``u_(e mod 6) * u_6^(e div 6 + 1)``, of total degree ``e div 6 + 2``.  With ``E`` extras the combination ``r^T (A w - b)``
is a nonzero polynomial of degree ``1 + ceil(E / 6)`` in the column coefficients whenever ``A w != b``.  The coefficients
are uint32 words reduced mod p (``protocol._expand``, no rejection), so no value has probability above 3 / 2^32 (the
fingerprint term's ``FP_COIN_MAX_PROB``).  Schwartz-Zippel over independent coins then bounds a vanishing coordinate by
``degree x 3 / 2^32``, and the ``D`` coordinates are independent.  With uniform coins the bound would be
``(degree / p)^D``: that is the red team's per-shape table, 0.49 bits per coordinate below this one.
"""
from __future__ import annotations

from fractions import Fraction

__all__ = ["COIN_MAX_PROB", "N_BASE_LINKS", "OPERANDS", "degree", "extra_constraints", "field_term", "hashed_extras"]

N_BASE_LINKS = 3                      # linked rows with their own uniform link / start families (chain.N_BASE_LINKS)
COIN_MAX_PROB = Fraction(3, 2 ** 32)  # a coefficient is a uint32 reduced mod p: no residue has more than 3 preimages
OPERANDS = 2                          # hashchain.ROLES: the x and W operand rows each carry the leaf's state along the chain


def extra_constraints(n_links: int, n_fp: int = 0) -> int:
    """``E``: the constraints with monomial coefficients at a chain of ``n_links`` linked rows -- one link and one start
    extra per row beyond the base three -- plus ``n_fp`` fingerprint constraints (``n_op x coords``)."""
    if n_links < N_BASE_LINKS or n_fp < 0:
        raise ValueError(f"a chain links {N_BASE_LINKS} or more rows ({n_links}) and carries n_fp >= 0 ({n_fp})")
    return 2 * (n_links - N_BASE_LINKS) + n_fp


def degree(n_extra: int) -> int:
    """The combination's degree in the column coefficients: 1 (uniform coefficients) without extras, else
    ``1 + ceil(E / 6)`` (the last extra's monomial, ``(E - 1) div 6 + 2``)."""
    return 1 if n_extra <= 0 else 1 + -(-n_extra // 6)


def field_term(n_extra: int, D: int) -> Fraction:
    """``(degree x 3 / 2^32)^D`` per proof; 0 without extras (the uniform combination is the booked ``linear_field``)."""
    return Fraction(0) if n_extra <= 0 else (degree(n_extra) * COIN_MAX_PROB) ** D


def hashed_extras(relation: str) -> int:
    """``E`` of the hashed statement ``<relation>[+<leaf>]`` (``hashchain.compose``: the three accumulator rows, plus the
    leaf's carried rows for each operand, the leaf sized to the relation's rows)."""
    from .leaf import registry
    scheme = registry.for_relation(relation)
    return extra_constraints(N_BASE_LINKS + OPERANDS * scheme.carry_elems)
