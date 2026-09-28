"""Closed-form forged witnesses that ``relation.check_step`` accepts over a prime field, with the wrong output.

Three predicates of the relation are ``a * b = c`` with ``a`` range-checked to ``m`` bits and ``b`` a power of two of
up to ``n`` bits fetched from a table; the remainder side is pinned only through a second product ``r * 2^(n-d)`` that
is range-checked.  Over the integers that pins ``a`` to the true quotient.  Over F_p it pins ``a`` only if
``2^(m+n) <= p``: otherwise ``(a + k) * 2^d`` wraps and the wrapped remainder still has a small ``r * invpow``.

    gadget      predicate (relation.py)                   m + n              spurious solutions exist iff
    align       aligned * pow  = scaled - r,  rinv < 2^A   A + A = 2A         p < 2^(2A)      (52 Ampere, 54 Hopper)
    final       mag24 * pow_s  = magW - r_s,  rsinv < 2^K  24 + K             p < 2^(24+K)    (59 Ampere, 60 Hopper)
    norm        normB = mag_t * normpow, normB in [2^(B-1), 2^B)   B + B      p < 2^(2B)      (60 Ampere, 64 Hopper)

A fourth hole does not depend on a small field.  ``magW`` is never range-checked: ``magW * 2^(B-W) + r_n = normB``
with ``r_n < 2^(B-W)`` pins ``magW`` over the integers (divisibility) but over F_p every other ``r_n`` gives a
``magW' = (normB - r_n') / 2^(B-W) mod p``, a large field element.  The final shift then has to solve
``mag24 * 2^sh + r_s = magW'`` with ``r_s * 2^(K-sh) < 2^K``: writing ``T = magW' * 2^(K-sh) mod p``, the pair
``mag24 = T >> K``, ``rsinv = T mod 2^K`` works whenever ``T < 2^(24+K)`` -- probability ``2^(24+K) / p`` per
``r_n'``, i.e. essentially always below 2^59 and for a fair share of instances at 2^64.  For Goldilocks
(``p = 2^64 - 2^32 + 1``) it is not even probabilistic on Ampere: ``2^64 = 2^32 - 1 (mod p)`` makes ``r_n' = r_n + 1``
land on ``mag24 - 1`` whenever the honest ``r_s`` is 0, i.e. the output is one ulp low on most normal results.

The module docstring of ``relation.py`` counts predicates "for a field of at least 40 bits".  ``forge_*`` build full
step witnesses from an honest one: the forged gadget, then the rest of the group re-derived honestly from the forged
value (``_tail``), then the following group re-derived honestly from the forged state.  Everything downstream of the
forgery is *exactly* what an honest prover would write for the forged intermediate, so only the forged gadget's
identities rely on the wrap.  ``relation_target.FieldCensus`` is the judge.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from verity.ml.circuits.silicon import relation as R
from verity.ml.circuits.silicon.silicon import Pipeline

__all__ = ["FORGERIES", "Forgery", "forge_align", "forge_final", "forge_leading_bit", "forge_split", "max_products",
           "thresholds"]


@dataclass(frozen=True)
class Forgery:
    gadget: str
    field: int
    group: int
    detail: dict
    witness: dict
    honest_output: int
    forged_output: int


def thresholds(pipe: Pipeline) -> dict[str, int]:
    """log2 of the smallest field size at which each gadget is pinned (the wrap needs ``p < 2^that``); for ``split``
    the exponent of the per-``r_n'`` success probability ``2^that / p`` (no field size pins it)."""
    p = R.params(pipe)
    return {"align": 2 * p.A, "final": 24 + p.K, "norm": 2 * p.B, "split": 24 + p.K}


def max_products(pipe: Pipeline) -> dict[str, int]:
    """The largest value each wrap gadget's product can take with its ranged factor maximal: a field ``p`` strictly
    above it cannot wrap (Goldilocks, ``2^64 - 2^32 + 1``, clears Hopper's ``norm`` bound ``(2^32 - 1) * 2^32`` by one)."""
    p = R.params(pipe)
    return {"align": ((1 << p.A) - 1) << p.A, "final": ((1 << 24) - 1) << p.K, "norm": ((1 << p.B) - 1) << p.B}


# ---------------------------------------------------------------------------------------------------------------------
# honest re-derivation of a group from an intermediate (a transcription of step_witness's tail, kept separate on purpose)
# ---------------------------------------------------------------------------------------------------------------------

def _tail_from_total(pipe: Pipeline, g: dict, total: int) -> None:
    p = R.params(pipe)
    norm_t, exp_t = R.T_norm(p), R.T_exp(p)
    Estar = g["Estar"]
    neg_t = int(total < 0)
    mag_t = -total if neg_t else total
    L = mag_t.bit_length()
    normpow, z_t = norm_t(L)
    normB = mag_t * normpow
    g.update(total=total, neg_t=neg_t, mag_t=mag_t, L=L, normpow=normpow, z_t=z_t, normB=normB)
    _tail_from_normB(pipe, g, L, normB)


def _tail_from_normB(pipe: Pipeline, g: dict, L: int, normB: int) -> None:
    p = R.params(pipe)
    exp_t = R.T_exp(p)
    magW = normB >> (p.B - p.W)
    e_out = g["Estar"] + L - p.W
    e_fin, pow_s, invpow_s, ovf = exp_t(e_out + R.E_OFF)
    mag24 = magW // pow_s
    r_s = magW - mag24 * pow_s
    g.update(magW=magW, r_n=normB - (magW << (p.B - p.W)), e_out=e_out, e_fin=e_fin, pow_s=pow_s, invpow_s=invpow_s,
             ovf=ovf, mag24=mag24, r_s=r_s, rsinv=r_s * invpow_s)
    _tail_from_mag24(pipe, g, mag24)


def _tail_from_mag24(pipe: Pipeline, g: dict, mag24: int) -> None:
    """The special-value logic and the output state from ``mag24`` and the already-set flags of ``g``."""
    Z = pipe.zero_exponent
    z_out = int(mag24 == 0)
    ovf_eff = g["ovf"] * (1 - g["z_t"])
    any_nan, any_pinf, any_ninf = g["any_nan"], g["any_pinf"], g["any_ninf"]
    any_inf = any_pinf + any_ninf - any_pinf * any_ninf
    nan_out = any_nan + (1 - any_nan) * any_pinf * any_ninf
    inf_out = (1 - nan_out) * (any_inf + (1 - any_inf) * ovf_eff)
    neg_inf = any_inf * any_ninf + (1 - any_inf) * g["neg_t"]
    fin = 1 - nan_out - inf_out
    neg_fin = g["neg_t"] * (1 - z_out)
    e_state = g["e_fin"] + z_out * (Z - g["e_fin"])
    out = R.State(nan_out, inf_out, inf_out * neg_inf + fin * neg_fin, Z + fin * (e_state - Z), fin * mag24)
    g.update(mag24=mag24, z_out=z_out, ovf_eff=ovf_eff, any_inf=any_inf, nan_out=nan_out, inf_out=inf_out,
             neg_inf=neg_inf, fin=fin, neg_fin=neg_fin, e_state=e_state, out=out)


def _regroup_following(pipe: Pipeline, w: dict, gi: int) -> None:
    """Re-derive groups ``gi+1..`` honestly from group ``gi``'s (forged) output state, with the generator itself run
    on a one-group pipeline of the same parameters (equal group sizes, as in every registered BF16 pipeline)."""
    if len(set(pipe.groups)) != 1:
        raise NotImplementedError("unequal group sizes")
    size = pipe.groups[0]
    sub = Pipeline(name=f"{pipe.name}/one-group", arch=pipe.arch, dtype=pipe.dtype, operand_bits=pipe.operand_bits,
                   groups=(size,), width=pipe.width, zero_exponent=pipe.zero_exponent, validated="forge")
    cur = w["groups"][gi]["out"]
    for g2 in range(gi + 1, len(pipe.groups)):
        start = g2 * size
        fresh = R.step_witness(sub, cur, w["a"][start:start + size], w["b"][start:start + size], w.get("align", R.ALIGN_DIV))
        w["groups"][g2] = fresh["groups"][0]
        cur = w["groups"][g2]["out"]
    w["out"] = cur


def _finish(pipe: Pipeline, w: dict, gi: int, gadget: str, field: int, detail: dict, honest_out: int) -> Forgery:
    _regroup_following(pipe, w, gi)
    return Forgery(gadget, field, gi, detail, w, honest_out, R.pack_state(w["out"]))


def _honest_output(w: dict) -> int:
    return R.pack_state(w["groups"][-1]["out"])


def _wrap_k(inv_ranged: int, width: int, field: int, cap: int) -> int | None:
    """Smallest ``k >= 1`` with ``(inv_ranged - k * 2^width) mod field`` back in ``[0, 2^width)`` and ``k < cap``."""
    k = (inv_ranged + field) // (1 << width)
    while k * (1 << width) > inv_ranged + field:
        k -= 1
    if k < 1 or k >= cap:
        return None
    return k


# ---------------------------------------------------------------------------------------------------------------------
# the three forgeries
# ---------------------------------------------------------------------------------------------------------------------

def forge_align(pipe: Pipeline, w: dict, field: int, gi: int = 0, term: int | None = None) -> Forgery | None:
    """Group ``gi``: replace one term's truncated alignment by ``aligned + k`` with the wrapped remainder."""
    p = R.params(pipe)
    honest = _honest_output(w)
    w = copy.deepcopy(w)
    g = w["groups"][gi]
    st = w["in"] if gi == 0 else w["groups"][gi - 1]["out"]
    prods = w["prod"][sum(pipe.groups[:gi]):sum(pipe.groups[:gi + 1])]
    ins = R._terms(p, st, prods)
    candidates = [term] if term is not None else range(len(ins))
    for i in candidates:
        neg, e, scaled = ins[i]
        t = g["terms"][i]
        if scaled == 0 or (w.get("align") == R.ALIGN_TABLE and i > 0):
            continue
        k = _wrap_k(t["rinv"], p.A, field, (1 << p.A) - t["aligned"])
        if k is None:
            continue
        aligned2 = t["aligned"] + k
        r2 = (scaled - aligned2 * t["pow"]) % field
        rinv2 = (r2 * t["invpow"]) % field
        assert 0 <= rinv2 < 1 << p.A
        total2 = g["total"] + (k if not neg else -k)
        t.update(aligned=aligned2, r=r2, rinv=rinv2, signed=neg * aligned2)
        _tail_from_total(pipe, g, total2)
        detail = {"term": i, "d": t["d"], "aligned_honest": aligned2 - k, "aligned_forged": aligned2, "k": k,
                  "r_forged": r2, "rinv_forged": rinv2, "missing_bound": f"aligned * pow < 2^{p.A} (p >= 2^{2 * p.A})"}
        return _finish(pipe, w, gi, "align", field, detail, honest)
    return None


def forge_final(pipe: Pipeline, w: dict, field: int, gi: int = 0) -> Forgery | None:
    """Group ``gi``: replace the 24-bit result magnitude by ``mag24 + k`` with the wrapped final-shift remainder."""
    p = R.params(pipe)
    honest = _honest_output(w)
    w = copy.deepcopy(w)
    g = w["groups"][gi]
    if g["magW"] == 0 or g["fin"] == 0:
        return None
    k = _wrap_k(g["rsinv"], p.K, field, (1 << 24) - g["mag24"])
    if k is None:
        return None
    mag24_2 = g["mag24"] + k
    r_s2 = (g["magW"] - mag24_2 * g["pow_s"]) % field
    rsinv2 = (r_s2 * g["invpow_s"]) % field
    assert 0 <= rsinv2 < 1 << p.K
    g.update(r_s=r_s2, rsinv=rsinv2)
    _tail_from_mag24(pipe, g, mag24_2)
    detail = {"mag24_honest": mag24_2 - k, "mag24_forged": mag24_2, "k": k, "sh": g["pow_s"].bit_length() - 1,
              "r_s_forged": r_s2, "rsinv_forged": rsinv2, "missing_bound": f"mag24 * pow_s < 2^{p.W} (p >= 2^{24 + p.K})"}
    return _finish(pipe, w, gi, "final", field, detail, honest)


def forge_leading_bit(pipe: Pipeline, w: dict, field: int, gi: int = 0) -> Forgery | None:
    """Group ``gi``: claim a leading-bit position ``L' < L`` whose wrapped ``mag_t * 2^(B-L')`` lands in the window."""
    p = R.params(pipe)
    honest = _honest_output(w)
    w = copy.deepcopy(w)
    g = w["groups"][gi]
    mag_t, L = g["mag_t"], g["L"]
    if mag_t == 0:
        return None
    for L2 in range(1, L):
        normB2 = (mag_t << (p.B - L2)) % field
        if (1 << (p.B - 1)) <= normB2 < (1 << p.B):
            normpow2, z_t2 = R.T_norm(p)(L2)
            g.update(L=L2, normpow=normpow2, z_t=z_t2, normB=normB2)
            _tail_from_normB(pipe, g, L2, normB2)
            detail = {"L_honest": L, "L_forged": L2, "mag_t": mag_t, "normB_forged": normB2,
                      "missing_bound": f"mag_t * normpow < 2^{p.B} (p >= 2^{2 * p.B})"}
            return _finish(pipe, w, gi, "norm", field, detail, honest)
    return None


def forge_split(pipe: Pipeline, w: dict, field: int, gi: int = 0) -> Forgery | None:
    """Group ``gi``: mis-split ``normB`` into an unranged ``magW'`` and another ``r_n``, then pick the ``mag24`` the
    wrapped final shift allows.  Works at any field size with probability about ``2^(B-W) * 2^(24+K) / p``."""
    p = R.params(pipe)
    honest = _honest_output(w)
    w = copy.deepcopy(w)
    g = w["groups"][gi]
    if g["fin"] == 0 or g["z_t"]:
        return None
    shift = 1 << (p.B - p.W)
    inv_shift = pow(shift, -1, field)
    r_n1, magW1, mag24_1 = g["r_n"], g["magW"], g["mag24"]
    for r_n2 in range(shift):
        if r_n2 == r_n1:
            continue
        magW2 = ((g["normB"] - r_n2) * inv_shift) % field
        T = (magW2 * g["invpow_s"]) % field                     # = r_s' * 2^(K-sh) + mag24' * 2^K  (mod p)
        mag24_2 = T >> p.K
        if mag24_2 >= 1 << 24 or mag24_2 == mag24_1:
            continue
        r_s2 = (magW2 - mag24_2 * g["pow_s"]) % field
        rsinv2 = (r_s2 * g["invpow_s"]) % field
        assert rsinv2 == T & ((1 << p.K) - 1) and 0 <= rsinv2 < 1 << p.K
        g.update(magW=magW2, r_n=r_n2, r_s=r_s2, rsinv=rsinv2)
        _tail_from_mag24(pipe, g, mag24_2)
        detail = {"r_n_honest": r_n1, "r_n_forged": r_n2, "magW_honest": magW1, "magW_forged": magW2,
                  "mag24_honest": mag24_1, "mag24_forged": mag24_2, "r_s_forged": r_s2, "rsinv_forged": rsinv2,
                  "missing_bound": f"RANGE(magW, {p.W}) -- magW is never range-checked"}
        return _finish(pipe, w, gi, "split", field, detail, honest)
    return None


FORGERIES = {"align": forge_align, "final": forge_final, "norm": forge_leading_bit, "split": forge_split}
