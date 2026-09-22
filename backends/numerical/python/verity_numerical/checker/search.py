"""Adversarial search for soundness: exhaustive at reduced widths, z3 at the real widths, per gadget and for a
full group step; and the same gadgets over a prime field to show which additional bounds an embedding needs.

    exhaustive(spec, p)        every input in the contract x every joint hint assignment; all accepted assignments
                               must agree on the outputs and equal the mathematical reference; the honest hints
                               must be accepted on admissible inputs
    single_hint_search(p, x)   the composed unit on one input: every hint in turn over its whole contract range
    z3_gadget(spec, p)         symbolic inputs and hints at the real widths: SAT = an accepted output that differs
                               from the reference (a finding), UNSAT within the timeout = evidence, UNKNOWN = open
    z3_group_step(p)           one full group step, two hint copies on one input, outputs required to differ
    modulus=...                the same predicates read modulo a prime: every identity mod p, every range check
                               on a representative -- SAT here names a bound the field embedding must add

``python -m verity_numerical.checker.search`` runs the suite and prints the table; ``--json PATH`` records it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field

from .ctx import IntCtx, Table, Z3Ctx
from .gadgets import (
    AlignProduct,
    AlignState,
    Bf16Decode,
    Contract,
    Epilogue,
    Fp32Decode,
    GroupMax,
    LeadingBit,
    Normalise,
    Pack,
    Product,
    SignedSum,
    StateOut,
)
from .params import REAL, TINY, Params, Reject
from .step import check_step_expr, gen_step

__all__ = ["SPECS", "Spec", "exhaustive", "single_hint_search", "z3_gadget", "z3_group_step", "main"]

P40 = 2**40 - 87                 # a 40-bit prime (relation.py's stated field size)
M31 = 2**31 - 1                  # Mersenne31
GOLDILOCKS = 2**64 - 2**32 + 1


# ---------------------------------------------------------------------------------------------------------------------
# Gadget specifications: how to call each gadget on a flat dict of inputs and hints, and what it must compute
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    name: str
    contract: Callable[[Params], Contract]
    call: Callable[..., tuple]                 # (ctx, p, inputs, hints) -> outputs
    gen: Callable[[Params, dict], dict]        # honest flat hints
    ref: Callable[[Params, dict], tuple]       # the mathematical outputs on Python ints
    zref: Callable[[object, Params, dict], tuple]   # the same as z3 expressions (z3 module passed first)
    admissible: Callable[[Params, dict], bool] = lambda p, x: True
    inputs_of: Callable[[Params, dict], dict] | None = None   # derive the ``derived`` inputs from the others
    derived: tuple[str, ...] = ()                              # inputs not enumerated by the exhaustive search
    zadmissible: Callable[[object, Params, dict], object] | None = None   # the premise as a z3 constraint


def _sel_names(p: Params) -> list[str]:
    return [f"sel[{i}]" for i in range(p.n + 2)]


def _bf16_ref(p, x):
    w = x["w"]
    t = (w >> p.frac) & (2**p.exp - 1)
    return (w >> (p.word_bits - 1), (w & (2**p.frac - 1)) + (2**p.frac if t else 0), max(t, 1) - p.bias)


def _bf16_zref(z3, p, x):
    w = x["w"]
    t = (w / 2**p.frac) % 2**p.exp
    return (w / 2 ** (p.word_bits - 1), w % 2**p.frac + z3.If(t == 0, 0, 2**p.frac), z3.If(t == 0, 1, t) - p.bias)


def _fp32_ref(p, x):
    c = x["c"]
    t = (c >> p.acc_frac) & (2**p.exp - 1)
    M = (c & (2**p.acc_frac - 1)) + (2**p.acc_frac if t else 0)
    return (c >> (p.acc_word_bits - 1), p.floor if M == 0 else max(t, 1) - p.bias, M)


def _fp32_zref(z3, p, x):
    c = x["c"]
    t = (c / 2**p.acc_frac) % 2**p.exp
    M = c % 2**p.acc_frac + z3.If(t == 0, 0, 2**p.acc_frac)
    return (c / 2 ** (p.acc_word_bits - 1), z3.If(M == 0, p.floor, z3.If(t == 0, 1, t) - p.bias), M)


def _prod_call(ctx, p, x, h):
    return Product.check(ctx, p, (x["s_a"], x["m_a"], x["e_a"]), (x["s_b"], x["m_b"], x["e_b"]), h)


def _prod_ref(p, x):
    u = x["m_a"] * x["m_b"]
    return (u, x["s_a"] ^ x["s_b"], p.floor if u == 0 else x["e_a"] + x["e_b"])


def _prod_zref(z3, p, x):
    u = x["m_a"] * x["m_b"]
    return (u, z3.If(x["s_a"] == x["s_b"], 0, 1), z3.If(u == 0, p.floor, x["e_a"] + x["e_b"]))


def _cands(p, x):
    return [x[f"cand[{i}]"] for i in range(p.n + 1)]


def _max_call(ctx, p, x, h):
    E, ds = GroupMax.check(ctx, p, _cands(p, x), {"E": h["E"], "sel": [h[s] for s in _sel_names(p)]})
    return (E, *ds)


def _max_gen(p, x):
    g = GroupMax.gen(p, _cands(p, x))
    return {"E": g["E"], **{s: b for s, b in zip(_sel_names(p), g["sel"])}}


def _max_ref(p, x):
    E = max(p.floor, *_cands(p, x))
    return (E, *[E - c for c in _cands(p, x)])


def _max_zref(z3, p, x):
    E = z3.IntVal(p.floor)
    for c in _cands(p, x):
        E = z3.If(c > E, c, E)
    return (E, *[E - c for c in _cands(p, x)])


def _shift_ref(P: int, d: int) -> int:
    return P >> d


def _zdiv_pow2(z3, value, d, width: int):
    """floor(value / 2^d) for a symbolic shift d >= 0 and value < 2^width, as a nested If (0 once d >= width)."""
    out = z3.IntVal(0)
    for k in range(width - 1, -1, -1):
        out = z3.If(d == k, value / 2**k, out)
    return out


def _align_state_ref(p, x):
    return ((x["M"] << p.acc_shift) >> x["d"],)


def _align_state_zref(z3, p, x):
    return (_zdiv_pow2(z3, x["M"] * 2**p.acc_shift, x["d"], p.acc_bits),)


def _align_prod_ref(p, x):
    return ((x["u"] << p.prod_shift) >> x["d"],)


def _align_prod_zref(z3, p, x):
    return (_zdiv_pow2(z3, x["u"] * 2**p.prod_shift, x["d"], p.prod_bits),)


def _terms(p, x):
    return [(x[f"s[{i}]"], x[f"q[{i}]"]) for i in range(p.n + 1)]


def _sum_call(ctx, p, x, h):
    return SignedSum.check(ctx, p, _terms(p, x), h)


def _sum_ref(p, x):
    S = sum(-q if s else q for s, q in _terms(p, x))
    return (int(S < 0), abs(S))


def _sum_zref(z3, p, x):
    S = sum(q - 2 * s * q for s, q in _terms(p, x))
    return (z3.If(S < 0, 1, 0), z3.If(S < 0, -S, S))


def _lead_ref(p, x):
    return (x["v"].bit_length(),)


def _lead_zref(z3, p, x):
    v = x["v"]
    out = z3.IntVal(p.l_hi)
    for l in range(p.l_hi - 1, -1, -1):
        out = z3.If(v < 2**l, l, out)
    return (out,)


def _norm_ref(p, x):
    E, l, v = x["E"], x["l"], x["v"]
    e1 = max(E + l - p.width, p.e_min)
    k = E - e1 + p.acc_sig - p.width
    return (e1, (v << k) if k >= 0 else (v >> -k))


def _norm_zref(z3, p, x):
    E, l, v = x["E"], x["l"], x["v"]
    e_raw = E + l - p.width
    e1 = z3.If(e_raw < p.e_min, p.e_min, e_raw)
    k = E - e1 + p.acc_sig - p.width
    out = z3.IntVal(-1)
    for kk in range(p.k_lo, p.k_hi + 1):
        out = z3.If(k == kk, v * 2**kk if kk >= 0 else v / 2 ** (-kk), out)
    return (e1, out)


def _norm_inputs(p, x):
    return {**x, "l": x["v"].bit_length()}


def _out_call(ctx, p, x, h):
    return StateOut.check(ctx, p, x["sgn"], x["e1"], x["M"], h)


def _out_ref(p, x):
    z = int(x["M"] == 0)
    return (0 if z else x["sgn"], p.floor if z else x["e1"], x["M"], int(not z and x["e1"] > p.e_max), z)


def _out_zref(z3, p, x):
    M, e1, sgn = x["M"], x["e1"], x["sgn"]
    return (z3.If(M == 0, 0, sgn), z3.If(M == 0, p.floor, e1), M,
            z3.If(z3.And(M != 0, e1 > p.e_max), 1, 0), z3.If(M == 0, 1, 0))


def _pack_call(ctx, p, x, h):
    G = len(p.groups)
    return (Pack.check(ctx, p, (x["s"], x["e1"], x["M"], x["z"]), [(x[f"ovf[{g}]"], x[f"sgn[{g}]"]) for g in range(G)], h),)


def _pack_ref(p, x):
    G = len(p.groups)
    for g in range(G):
        if x[f"ovf[{g}]"]:
            return ((x[f"sgn[{g}]"] << (p.acc_word_bits - 1)) | (p.t_finite << p.acc_frac),)
    if x["z"]:
        return (0,)
    hidden = x["M"] >> p.acc_frac
    fld = x["e1"] + p.bias - 1 + hidden
    return ((x["s"] << (p.acc_word_bits - 1)) + fld * 2**p.acc_frac + (x["M"] & (2**p.acc_frac - 1)),)


def _pack_zref(z3, p, x):
    G = len(p.groups)
    hidden = x["M"] / 2**p.acc_frac
    fld = x["e1"] + p.bias - 1 + hidden
    fin = x["s"] * 2 ** (p.acc_word_bits - 1) + fld * 2**p.acc_frac + x["M"] % 2**p.acc_frac
    y = z3.If(x["z"] == 1, 0, fin)
    for g in range(G - 1, -1, -1):
        y = z3.If(x[f"ovf[{g}]"] == 1, x[f"sgn[{g}]"] * 2 ** (p.acc_word_bits - 1) + p.t_finite * 2**p.acc_frac, y)
    return (y,)


def _pack_admissible(p, x):
    # the state's own invariants: a zero state has M = 0; a nonzero state without hidden bit sits at e_min
    if x["z"] != int(x["M"] == 0):
        return False
    return x["M"] == 0 or x["M"] >> p.acc_frac == 1 or x["e1"] == p.e_min


def _cast_ref(p, x):
    L = p.acc_frac - p.frac
    y = x["y32"]
    return ((((y >> L) & 1) + y + (2 ** (L - 1) - 1)) >> L,)


def _cast_zref(z3, p, x):
    L = p.acc_frac - p.frac
    y = x["y32"]
    return (((y / 2**L) % 2 + y + (2 ** (L - 1) - 1)) / 2**L,)


def _cast_admissible(p, x):
    # a word whose rounding carries out of the sign bit is refused (only a NaN with a large payload does)
    return _cast_ref(p, x)[0] < 2**p.word_bits


SPECS: dict[str, Spec] = {
    "bf16_decode": Spec("bf16_decode", Bf16Decode.contract, lambda ctx, p, x, h: Bf16Decode.check(ctx, p, x["w"], h),
                        lambda p, x: Bf16Decode.gen(p, x["w"]), _bf16_ref, _bf16_zref,
                        admissible=lambda p, x: (x["w"] >> p.frac) & (2**p.exp - 1) != p.t_finite),
    "fp32_decode": Spec("fp32_decode", Fp32Decode.contract, lambda ctx, p, x, h: Fp32Decode.check(ctx, p, x["c"], h),
                        lambda p, x: Fp32Decode.gen(p, x["c"]), _fp32_ref, _fp32_zref,
                        admissible=lambda p, x: (x["c"] >> p.acc_frac) & (2**p.exp - 1) != p.t_finite),
    "product": Spec("product", Product.contract, _prod_call, lambda p, x: Product.gen(p, x["m_a"], x["m_b"]),
                    _prod_ref, _prod_zref),
    "group_max": Spec("group_max", GroupMax.contract, _max_call, _max_gen, _max_ref, _max_zref),
    "align_state": Spec("align_state", AlignState.contract,
                        lambda ctx, p, x, h: (AlignState.check(ctx, p, x["M"], x["d"], h),),
                        lambda p, x: AlignState.gen(p, x["M"], x["d"]), _align_state_ref, _align_state_zref),
    "align_product": Spec("align_product", AlignProduct.contract,
                          lambda ctx, p, x, h: (AlignProduct.check(ctx, p, x["u"], x["d"], h),),
                          lambda p, x: AlignProduct.gen(p, x["u"], x["d"]), _align_prod_ref, _align_prod_zref),
    "signed_sum": Spec("signed_sum", SignedSum.contract, _sum_call, lambda p, x: SignedSum.gen(p, _terms(p, x)),
                       _sum_ref, _sum_zref),
    "leading_bit": Spec("leading_bit", LeadingBit.contract, lambda ctx, p, x, h: (LeadingBit.check(ctx, p, x["v"], h),),
                        lambda p, x: LeadingBit.gen(p, x["v"]), _lead_ref, _lead_zref),
    "normalise": Spec("normalise", Normalise.contract,
                      lambda ctx, p, x, h: Normalise.check(ctx, p, x["E"], x["l"], x["v"], h),
                      lambda p, x: Normalise.gen(p, x["E"], x["l"], x["v"]), _norm_ref, _norm_zref,
                      admissible=lambda p, x: x["l"] == x["v"].bit_length(), inputs_of=_norm_inputs, derived=("l",),
                      zadmissible=lambda z3, p, x: x["l"] == _lead_zref(z3, p, x)[0]),
    "state_out": Spec("state_out", StateOut.contract, _out_call, lambda p, x: StateOut.gen(p, x["e1"], x["M"]),
                      _out_ref, _out_zref),
    "pack": Spec("pack", Pack.contract, _pack_call, lambda p, x: Pack.gen(p, x["M"]), _pack_ref, _pack_zref,
                 admissible=_pack_admissible,
                 zadmissible=lambda z3, p, x: z3.And(x["z"] == z3.If(x["M"] == 0, 1, 0),
                                                     z3.Or(x["M"] == 0, x["M"] >= 2**p.acc_frac, x["e1"] == p.e_min))),
    "epilogue": Spec("epilogue", Epilogue.contract, lambda ctx, p, x, h: (Epilogue.check(ctx, p, x["y32"], h),),
                     lambda p, x: Epilogue.gen(p, x["y32"]), _cast_ref, _cast_zref, admissible=_cast_admissible,
                     zadmissible=lambda z3, p, x: _cast_zref(z3, p, x)[0] < 2**p.word_bits),
}


# ---------------------------------------------------------------------------------------------------------------------
# Exhaustive search at reduced widths
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class ExhaustiveResult:
    gadget: str
    params: str
    inputs: int = 0                 # input assignments enumerated
    admissible: int = 0             # of which satisfy the gadget's premise (the honest hints must be accepted)
    hint_space: int = 0             # joint hint assignments per input
    accepted: int = 0               # accepted (input, hints) pairs
    max_accepted_per_input: int = 0
    inputs_with_two_outputs: int = 0    # soundness failures: an input with two accepted, different outputs
    inputs_off_reference: int = 0       # soundness failures: an accepted output differing from the reference
    incomplete: int = 0                 # completeness failures: admissible input whose honest hints were rejected
    seconds: float = 0.0
    examples: list = field(default_factory=list)

    @property
    def sound(self) -> bool:
        return self.inputs_with_two_outputs == 0 and self.inputs_off_reference == 0

    @property
    def complete(self) -> bool:
        return self.incomplete == 0


def _ranges(d: dict[str, tuple[int, int]]) -> tuple[list[str], list[range]]:
    names = list(d)
    return names, [range(lo, hi) for lo, hi in d.values()]


def exhaustive(spec: Spec, p: Params = TINY, *, budget: int = 5_000_000) -> ExhaustiveResult:
    """Every input in the contract x every joint hint assignment.  Raises if the joint space exceeds ``budget``."""
    contract = spec.contract(p)
    in_names = [n for n in contract.inputs if n not in spec.derived]
    in_ranges = [range(*contract.inputs[n]) for n in in_names]
    h_names, h_ranges = _ranges(contract.hints)
    hint_space = 1
    for r in h_ranges:
        hint_space *= len(r)
    n_inputs = 1
    for r in in_ranges:
        n_inputs *= len(r)
    if hint_space * n_inputs > budget:
        raise ValueError(f"{spec.name} at {p}: {n_inputs} inputs x {hint_space} hints exceeds the budget {budget}")
    res = ExhaustiveResult(spec.name, repr(p), inputs=n_inputs, hint_space=hint_space)
    t0 = time.perf_counter()
    ctx = IntCtx()
    for values in itertools.product(*in_ranges):
        x = dict(zip(in_names, values))
        if spec.inputs_of is not None:
            x = spec.inputs_of(p, x)
        adm = spec.admissible(p, x)
        res.admissible += adm
        expected = spec.ref(p, x) if adm else None
        honest = spec.gen(p, x) if adm else None
        outputs: set = set()
        n_acc = 0
        honest_ok = False
        for hv in itertools.product(*h_ranges):
            h = dict(zip(h_names, hv))
            try:
                out = spec.call(ctx, p, x, h)
            except Reject:
                continue
            n_acc += 1
            outputs.add(tuple(out))
            if honest is not None and all(h[k] == honest[k] for k in h_names):
                honest_ok = True
        res.accepted += n_acc
        res.max_accepted_per_input = max(res.max_accepted_per_input, n_acc)
        if len(outputs) > 1:
            res.inputs_with_two_outputs += 1
            res.examples.append({"input": x, "outputs": sorted(outputs)})
        if adm:
            if not honest_ok:
                res.incomplete += 1
                res.examples.append({"input": x, "honest_rejected": honest})
            if outputs and outputs != {tuple(expected)}:
                res.inputs_off_reference += 1
                res.examples.append({"input": x, "outputs": sorted(outputs), "reference": expected})
        elif outputs and len(outputs) == 1 and spec.name in ("bf16_decode", "fp32_decode", "epilogue"):
            # an inadmissible word (non-finite) must have no accepted hints at all
            res.inputs_off_reference += 1
            res.examples.append({"input": x, "outputs": sorted(outputs), "reference": "rejected"})
    res.seconds = time.perf_counter() - t0
    res.examples = res.examples[:5]
    return res


# ---------------------------------------------------------------------------------------------------------------------
# The composed unit: every hint, one at a time, over its whole range (reduced widths) or a sample (real widths)
# ---------------------------------------------------------------------------------------------------------------------

def _flatten(h, prefix=""):
    if isinstance(h, dict):
        for k, v in h.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(h, list):
        for i, v in enumerate(h):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, h


def _set(h, path: str, value) -> None:
    node = h
    tokens = path.replace("]", "").replace("[", ".").split(".")
    for tok in tokens[:-1]:
        node = node[int(tok)] if isinstance(node, list) else node[tok]
    last = tokens[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _hint_range(p: Params, path: str) -> tuple[int, int]:
    leaf = path.split(".")[-1].split("[")[0]
    section = path.split(".")[-2].split("[")[0]
    table = {
        "acc": Fp32Decode.contract(p).hints, "a": Bf16Decode.contract(p).hints, "b": Bf16Decode.contract(p).hints,
        "prod": Product.contract(p).hints, "max": GroupMax.contract(p).hints, "acc_align": AlignState.contract(p).hints,
        "prod_align": AlignProduct.contract(p).hints, "sum": SignedSum.contract(p).hints,
        "lead": LeadingBit.contract(p).hints, "norm": Normalise.contract(p).hints, "out": StateOut.contract(p).hints,
        "pack": Pack.contract(p).hints, "epilogue": Epilogue.contract(p).hints,
    }
    hints = table[section]
    key = leaf if leaf in hints else f"{leaf}[0]"
    return hints[key]


@dataclass
class SingleHintResult:
    params: str
    inputs: int = 0
    hints: int = 0                  # hint columns tried
    alternatives: int = 0           # alternative values tried
    accepted_same_output: int = 0   # accepted alternatives with the same output (selector ties)
    accepted_other_output: int = 0  # soundness failures
    seconds: float = 0.0
    examples: list = field(default_factory=list)


def single_hint_search(p: Params, cases: Iterable[tuple[int, list[int], list[int]]], *, sample: int | None = None,
                       seed: int = 0) -> SingleHintResult:
    """For each input, each hint of the honest witness is replaced by every other value of its contract range
    (or ``sample`` values when the range is large); the checker must reject, or accept with the same output."""
    from .reference import dot

    rng = random.Random(seed)
    res = SingleHintResult(repr(p))
    t0 = time.perf_counter()
    ctx = IntCtx()
    for c, a, b in cases:
        y = dot(p, c, a, b)
        h = gen_step(p, c, a, b)
        assert check_step_expr(ctx, p, c, a, b, h) == y
        res.inputs += 1
        for path, honest in _flatten(h):
            lo, hi = _hint_range(p, path)
            if sample is None or hi - lo <= sample:
                candidates = [v for v in range(lo, hi) if v != honest]
            else:
                candidates = {lo, hi - 1, honest - 1, honest + 1, honest ^ 1, honest * 2, honest // 2}
                candidates |= {rng.randrange(lo, hi) for _ in range(sample)}
                candidates = [v for v in candidates if lo <= v < hi and v != honest]
            res.hints += 1
            for v in candidates:
                _set(h, path, v)
                res.alternatives += 1
                try:
                    got = check_step_expr(ctx, p, c, a, b, h)
                except Reject:
                    continue
                if got == y:
                    res.accepted_same_output += 1
                else:
                    res.accepted_other_output += 1
                    if len(res.examples) < 5:
                        res.examples.append({"c": c, "a": a, "b": b, "hint": path, "value": v, "y": y, "got": got})
            _set(h, path, honest)
    res.seconds = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------------------------------------------------
# z3 at the real widths
# ---------------------------------------------------------------------------------------------------------------------

class ModZ3Ctx(Z3Ctx):
    """The predicates read over the prime field F_p: identities modulo p, range checks on some representative in
    the range (a fresh witness), lookups matching the key modulo p.  Hints are field elements, i.e. Ints in [0, p).
    A congruence a = b (mod p) is stated as a - b = m p with a fresh integer m (linear in m; z3 handles it far
    better than ``%``)."""

    def __init__(self, modulus: int, *, ideal_bounds: bool = False) -> None:
        super().__init__()
        self.modulus = modulus
        self.ideal_bounds = ideal_bounds

    def _cong(self, a, b):
        m = self.fresh("wrap")
        return a - b == m * self.modulus

    def lt_pow(self, name, x, ipow, bits):
        if not self.ideal_bounds:
            super().lt_pow(name, x, ipow, bits)         # the reference realisation: the product, read mod p
            return
        # an ideal variable-bound range check (e.g. a lookup on (value, bound)): some representative of x in
        # [0, 2^bits) satisfies rep * ipow < 2^bits over the integers
        rep = self.fresh(f"{name}.rep")
        self._add(self.z3.And(rep >= 0, rep * ipow < 2**bits, self._cong(x, rep)))

    def rng(self, name, x, lo, hi):
        if isinstance(x, int):
            self._add(lo <= x % self.modulus < hi or lo <= x < hi)
            return
        rep = self.fresh(f"{name}.rep")
        self._add(self.z3.And(rep >= lo, rep < hi, self._cong(x, rep)))

    def eq(self, name, a, b):
        if isinstance(a, int) and isinstance(b, int):
            self._add((a - b) % self.modulus == 0)
            return
        self._add(self._cong(a, b))

    def boolean(self, name, b):
        if isinstance(b, int):
            self._add(b % self.modulus in (0, 1))
            return
        self._add(self.z3.Or(self._cong(b, 0), self._cong(b, 1)))

    def lookup(self, name, table: Table, key):
        z3 = self.z3
        if isinstance(key, int):
            return super().lookup(name, table, key)
        outs = tuple(self.fresh(f"{name}.{o}") for o in table.outs)
        rows = [z3.And(self._cong(key, k), *[o == v for o, v in zip(outs, table.fn(k))])
                for k in range(table.lo, table.hi)]
        self._add(z3.Or(rows))
        return outs


@dataclass
class QueryResult:
    gadget: str
    query: str
    status: str
    seconds: float
    model: dict | None = None
    note: str = ""


def _symbolic_inputs(ctx: Z3Ctx, contract: Contract) -> dict:
    x = {}
    for name, (lo, hi) in contract.inputs.items():
        v = ctx.fresh(f"in.{name}")
        ctx.solver.add(v >= lo, v < hi)
        x[name] = v
    return x


def _symbolic_hints(ctx: Z3Ctx, contract: Contract, modulus: int | None) -> dict:
    h = {}
    for name in contract.hints:
        v = ctx.fresh(f"hint.{name}")
        if modulus is not None:
            ctx.solver.add(v >= 0, v < modulus)
        h[name] = v
    return h


def _model_dict(ctx: Z3Ctx, named: dict) -> dict:
    m = ctx.solver.model()
    out = {}
    for k, v in named.items():
        if isinstance(v, int):
            out[k] = v
        else:
            val = m.eval(v, model_completion=True)
            out[k] = val.as_long() if hasattr(val, "as_long") else str(val)
    return out


def z3_gadget(spec: Spec, p: Params = REAL, *, timeout_ms: int = 20_000, modulus: int | None = None,
              inputs: dict | None = None, drop: Iterable[str] = (), label: str | None = None,
              ideal_bounds: bool = False) -> QueryResult:
    """Is there an accepted hint assignment whose outputs differ from the reference?  ``inputs`` fixes some inputs
    to constants; ``drop`` names predicates to omit (to show which bound a SAT depends on); ``modulus`` reads the
    predicates over F_p, with ``ideal_bounds`` realising every variable-bound range check wrap-free."""
    ctx = ModZ3Ctx(modulus, ideal_bounds=ideal_bounds) if modulus else Z3Ctx()
    z3 = ctx.z3
    if drop:
        drop = set(drop)
        orig = {"rng": ctx.rng, "eq": ctx.eq, "lt_pow": ctx.lt_pow}

        def rng(name, x, lo, hi):
            if name.split(".")[-1] not in drop:
                orig["rng"](name, x, lo, hi)

        def eq(name, a, b):
            if name.split(".")[-1] not in drop:
                orig["eq"](name, a, b)

        def lt_pow(name, x, ipow, bits):
            if name.split(".")[-1] not in drop:
                orig["lt_pow"](name, x, ipow, bits)
        ctx.rng, ctx.eq, ctx.lt_pow = rng, eq, lt_pow
    contract = spec.contract(p)
    x = _symbolic_inputs(ctx, contract)
    if inputs:
        for k, v in inputs.items():
            ctx.solver.add(x[k] == v)
    h = _symbolic_hints(ctx, contract, modulus)
    try:
        outs = spec.call(ctx, p, x, h)
    except Reject as exc:
        return QueryResult(spec.name, label or "differs-from-reference", "rejected-statically", 0.0, note=str(exc))
    refs = spec.zref(z3, p, x)
    if spec.zadmissible is not None:
        ctx.solver.add(spec.zadmissible(z3, p, x))
    if modulus:
        # the field embedding is sound iff every output, a field element, is the reference integer's residue
        ctx.solver.add(z3.Or([(o - r) % modulus != 0 for o, r in zip(outs, refs)]))
    else:
        ctx.solver.add(z3.Or([o != r for o, r in zip(outs, refs)]))
    ctx.solver.set("timeout", timeout_ms)
    t0 = time.perf_counter()
    status = str(ctx.solver.check())
    seconds = time.perf_counter() - t0
    model = None
    if status == "sat":
        model = {"inputs": _model_dict(ctx, x), "hints": _model_dict(ctx, h),
                 "outputs": _model_dict(ctx, {f"out[{i}]": o for i, o in enumerate(outs)}),
                 "reference": _model_dict(ctx, {f"ref[{i}]": r for i, r in enumerate(refs)})}
    note = f"mod {modulus}" if modulus else ""
    if ideal_bounds:
        note += ", ideal variable-bound checks"
    if drop:
        note += (" " if note else "") + f"without {sorted(drop)}"
    return QueryResult(spec.name, label or "differs-from-reference", status, round(seconds, 3), model, note)


def z3_group_step(p: Params = REAL, *, timeout_ms: int = 60_000, concrete: dict | None = None) -> QueryResult:
    """One full group step at the real widths: a state and n products (symbolic unless ``concrete`` gives them),
    two independent hint copies, and the two output states required to differ."""
    ctx = Z3Ctx()
    z3 = ctx.z3
    n = p.n
    last = p.e_hi(len(p.groups) - 1)

    def sym(name, lo, hi):
        v = ctx.fresh(name)
        ctx.solver.add(v >= lo, v < hi)
        return v

    s_c, e_c, M_c = sym("s_c", 0, 2), sym("e_c", p.exp_lo, last + 1), sym("M_c", 0, 2**p.acc_sig)
    prods = [(sym(f"u{i}", 0, 2 ** (2 * p.sig)), sym(f"sp{i}", 0, 2), sym(f"ep{i}", p.exp_lo, p.ep_hi + 1)) for i in range(n)]
    x = {"s_c": s_c, "e_c": e_c, "M_c": M_c, **{f"u{i}": u for i, (u, _, _) in enumerate(prods)},
         **{f"sp{i}": s for i, (_, s, _) in enumerate(prods)}, **{f"ep{i}": e for i, (_, _, e) in enumerate(prods)}}
    if concrete:
        for k, v in concrete.items():
            ctx.solver.add(x[k] == v)

    def copy_(tag):
        h = {"max": {"E": ctx.fresh(f"{tag}.E"), "sel": [ctx.fresh(f"{tag}.sel{i}") for i in range(n + 2)]},
             "acc_align": {"q": ctx.fresh(f"{tag}.qa"), "r": ctx.fresh(f"{tag}.ra")},
             "prod_align": [{"q": ctx.fresh(f"{tag}.q{i}"), "r": ctx.fresh(f"{tag}.r{i}")} for i in range(n)],
             "sum": {"sgn": ctx.fresh(f"{tag}.sgn"), "z": ctx.fresh(f"{tag}.zs")},
             "lead": {"l": ctx.fresh(f"{tag}.l")},
             "norm": {"clamp": ctx.fresh(f"{tag}.clamp"), "M": ctx.fresh(f"{tag}.M"), "r": ctx.fresh(f"{tag}.rn")},
             "out": {"z": ctx.fresh(f"{tag}.z"), "ovf": ctx.fresh(f"{tag}.ovf")}}
        cands = [e_c] + [e for _, _, e in prods]
        E, ds = GroupMax.check(ctx, p, cands, h["max"], f"{tag}.max")
        q_acc = AlignState.check(ctx, p, M_c, ds[0], h["acc_align"], f"{tag}.align_state")
        qs = [AlignProduct.check(ctx, p, u, d, hq, f"{tag}.align_prod[{i}]")
              for i, ((u, _, _), d, hq) in enumerate(zip(prods, ds[1:], h["prod_align"]))]
        terms = [(s_c, q_acc)] + [(s, q) for (_, s, _), q in zip(prods, qs)]
        sgn, v = SignedSum.check(ctx, p, terms, h["sum"], f"{tag}.sum")
        l = LeadingBit.check(ctx, p, v, h["lead"], f"{tag}.lead")
        e1, M = Normalise.check(ctx, p, E, l, v, h["norm"], f"{tag}.norm")
        s_out, e_out, M_out, ovf, z = StateOut.check(ctx, p, sgn, e1, M, h["out"], f"{tag}.out")
        return (s_out, e_out, M_out, ovf, z), h

    out1, h1 = copy_("h1")
    out2, h2 = copy_("h2")
    ctx.solver.add(z3.Or([a != b for a, b in zip(out1, out2)]))
    ctx.solver.set("timeout", timeout_ms)
    t0 = time.perf_counter()
    status = str(ctx.solver.check())
    seconds = time.perf_counter() - t0
    model = None
    if status == "sat":
        model = {"inputs": _model_dict(ctx, x),
                 "h1": _model_dict(ctx, dict(_flatten(h1))), "h2": _model_dict(ctx, dict(_flatten(h2))),
                 "out1": _model_dict(ctx, {f"o{i}": o for i, o in enumerate(out1)}),
                 "out2": _model_dict(ctx, {f"o{i}": o for i, o in enumerate(out2)})}
    return QueryResult("group_step", "two-witnesses-two-outputs" + (" (concrete x)" if concrete else " (symbolic x)"),
                       status, round(seconds, 3), model)


# ---------------------------------------------------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------------------------------------------------

def modular_queries(p: Params = REAL, *, timeout_ms: int = 20_000) -> list[QueryResult]:
    """Which bounds a field embedding must add: the same gadgets over F_p, as stated (the reference realisation of
    every variable-bound check as a product) and with ideal variable-bound checks, with and without the tight
    quotient bound."""
    S = SPECS
    T = timeout_ms
    return [
        # F1/F2: the product realisation of a variable bound wraps once 2^(2 bits) > p
        z3_gadget(S["align_state"], p, timeout_ms=T, modulus=P40, label="F_p40, as stated"),
        z3_gadget(S["align_state"], p, timeout_ms=T, modulus=GOLDILOCKS, label="F_goldilocks, as stated"),
        z3_gadget(S["leading_bit"], p, timeout_ms=T, modulus=P40, label="F_p40, as stated"),
        z3_gadget(S["leading_bit"], p, timeout_ms=T, modulus=P40, ideal_bounds=True, label="F_p40, ideal bounds"),
        z3_gadget(S["leading_bit"], p, timeout_ms=T, modulus=GOLDILOCKS, label="F_goldilocks, as stated"),
        # F3: with ideal variable bounds in a 31-bit field the tight quotient bound is what pins the quotient
        z3_gadget(S["align_state"], p, timeout_ms=T, modulus=M31, ideal_bounds=True, label="F_m31, ideal bounds"),
        z3_gadget(S["align_state"], p, timeout_ms=T, modulus=M31, ideal_bounds=True, drop=("quot",),
                  label="F_m31, ideal bounds, without the tight quotient bound"),
        z3_gadget(S["align_product"], p, timeout_ms=T, modulus=M31, ideal_bounds=True, label="F_m31, ideal bounds"),
        z3_gadget(S["align_product"], p, timeout_ms=T, modulus=M31, ideal_bounds=True, drop=("quot",),
                  label="F_m31, ideal bounds, without the tight quotient bound"),
        z3_gadget(S["align_state"], p, timeout_ms=T, drop=("quot",), label="integers, without the tight quotient bound"),
        z3_gadget(S["normalise"], p, timeout_ms=T, modulus=M31, ideal_bounds=True, label="F_m31, ideal bounds"),
        # small decompositions and flags do not wrap even in a 31-bit field
        z3_gadget(S["bf16_decode"], p, timeout_ms=T, modulus=M31, label="F_m31, as stated"),
        z3_gadget(S["fp32_decode"], p, timeout_ms=T, modulus=M31, label="F_m31, as stated"),
        z3_gadget(S["product"], p, timeout_ms=T, modulus=M31, label="F_m31, as stated"),
        z3_gadget(S["group_max"], p, timeout_ms=T, modulus=M31, label="F_m31, as stated"),
        z3_gadget(S["signed_sum"], p, timeout_ms=T, modulus=M31, label="F_m31, as stated"),
        z3_gadget(S["state_out"], p, timeout_ms=T, modulus=M31, label="F_m31, as stated"),
        z3_gadget(S["pack"], p, timeout_ms=T, modulus=M31, label="F_m31, as stated"),
        z3_gadget(S["epilogue"], p, timeout_ms=T, modulus=M31, label="F_m31, as stated"),
    ]


def integer_queries(p: Params = REAL, *, timeout_ms: int = 20_000, targeted: bool = True) -> list[QueryResult]:
    out = [z3_gadget(spec, p, timeout_ms=timeout_ms) for spec in SPECS.values()]
    if targeted:
        # the handoff's named targets, as dropped predicates: each must turn the UNSAT into a SAT (the bound matters)
        out.append(z3_gadget(SPECS["group_max"], p, timeout_ms=timeout_ms, drop=("attained",), label="excessive E: without `attained`"))
        out.append(z3_gadget(SPECS["group_max"], p, timeout_ms=timeout_ms, drop=("onehot",), label="false selector: without `onehot`"))
        out.append(z3_gadget(SPECS["align_state"], p, timeout_ms=timeout_ms, drop=("rem",), label="remainder: without `rem`"))
        out.append(z3_gadget(SPECS["align_product"], p, timeout_ms=timeout_ms, drop=("rem",), label="remainder: without `rem`"))
        out.append(z3_gadget(SPECS["leading_bit"], p, timeout_ms=timeout_ms, drop=("window",), label="wrong l: without `window`"))
        out.append(z3_gadget(SPECS["signed_sum"], p, timeout_ms=timeout_ms, drop=("v", "z1"), label="sign: without the ranges of v (`v`, `z1`)"))
        out.append(z3_gadget(SPECS["signed_sum"], p, timeout_ms=timeout_ms, drop=("zero_sign",), label="sign of zero: without `zero_sign`"))
        out.append(z3_gadget(SPECS["normalise"], p, timeout_ms=timeout_ms, drop=("clamp_proof",), label="clamp: without `clamp_proof`"))
        out.append(z3_gadget(SPECS["state_out"], p, timeout_ms=timeout_ms, drop=("ovf_proof",), label="overflow: without `ovf_proof`"))
    return out


def concrete_group_inputs() -> dict:
    """A concrete group-1 input for the full-step query: a normal state and eight mixed products."""
    x = {"s_c": 0, "e_c": 3, "M_c": (1 << 23) | 0x123456}
    for i in range(8):
        x[f"u{i}"] = (128 + 3 * i) * (128 + 5 * (i + 1))
        x[f"sp{i}"] = i & 1
        x[f"ep{i}"] = [0, 3, -5, 3, 7, -20, 3, 1][i]
    return x


def render(rows: list[QueryResult]) -> str:
    lines = ["| gadget | query | status | seconds | note |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.gadget} | {r.query} | {r.status} | {r.seconds} | {r.note} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timeout", type=float, default=20.0, help="z3 timeout per query, seconds")
    ap.add_argument("--step-timeout", type=float, default=60.0, help="z3 timeout for the full group step, seconds")
    ap.add_argument("--json", default=None, help="write the results here (default: $RESEARCH_RUN_DIR/result.json if set)")
    ap.add_argument("--skip-exhaustive", action="store_true")
    ap.add_argument("--skip-modular", action="store_true")
    ap.add_argument("--skip-step", action="store_true")
    ap.add_argument("--single-hint-cases", type=int, default=20)
    args = ap.parse_args(argv)
    report: dict = {"params": repr(REAL), "exhaustive": [], "single_hint": [], "z3": []}
    if not args.skip_exhaustive:
        for spec in SPECS.values():
            try:
                r = exhaustive(spec, TINY)
            except ValueError as exc:
                print(f"exhaustive {spec.name}: skipped ({exc})", flush=True)
                continue
            report["exhaustive"].append(asdict(r) | {"sound": r.sound, "complete": r.complete})
            print(f"exhaustive {spec.name}: inputs={r.inputs} admissible={r.admissible} hints={r.hint_space} "
                  f"accepted={r.accepted} max/input={r.max_accepted_per_input} two_outputs={r.inputs_with_two_outputs} "
                  f"off_ref={r.inputs_off_reference} incomplete={r.incomplete} {r.seconds:.2f}s", flush=True)
        rng = random.Random(1)

        def finite(frac_bits: int) -> int:           # a random finite word: any sign, finite exponent, any fraction
            return (rng.getrandbits(1) << (TINY.exp + frac_bits)) | (rng.randrange(TINY.t_finite) << frac_bits) \
                | rng.getrandbits(frac_bits)
        k = sum(TINY.groups)
        cases = [(finite(TINY.acc_frac), [finite(TINY.frac) for _ in range(k)], [finite(TINY.frac) for _ in range(k)])
                 for _ in range(args.single_hint_cases)]
        r = single_hint_search(TINY, cases)
        report["single_hint"].append(asdict(r))
        print(f"single-hint TINY: inputs={r.inputs} hints={r.hints} alternatives={r.alternatives} "
              f"same={r.accepted_same_output} other={r.accepted_other_output} {r.seconds:.2f}s", flush=True)
    rows = integer_queries(timeout_ms=int(args.timeout * 1000))
    if not args.skip_modular:
        rows += modular_queries(timeout_ms=int(args.timeout * 1000))
    if not args.skip_step:
        rows.append(z3_group_step(timeout_ms=int(args.step_timeout * 1000), concrete=concrete_group_inputs()))
        rows.append(z3_group_step(timeout_ms=int(args.step_timeout * 1000)))
    report["z3"] = [asdict(r) for r in rows]
    print(render(rows), flush=True)
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    path = args.json or (os.path.join(run_dir, "search.json") if run_dir else None)
    if path:
        with open(path, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print(f"wrote {path}")
    if run_dir:
        # the research run-result envelope: the searches' verdict, one measurement per query, the report as artifact
        problems = [f"exhaustive {e['gadget']}" for e in report["exhaustive"] if not (e["sound"] and e["complete"])]
        problems += [f"single-hint alternatives accepted: {s['accepted_other_output']} other, {s['accepted_same_output']} same"
                     for s in report["single_hint"] if s["accepted_other_output"] or s["accepted_same_output"]]
        # a SAT at the integer widths with every predicate in place is a soundness finding; modular SATs and
        # dropped-predicate SATs are the expected demonstrations
        problems += [f"SAT: {r.gadget} / {r.query}" for r in rows
                     if r.status == "sat" and not (r.note.startswith("mod") or "without" in r.note)]
        unknown = [f"{r.gadget} / {r.query}" for r in rows if r.status == "unknown"]
        validation = {"status": "passed" if not problems else "failed",
                      "detail": "; ".join(problems) or "no accepted deviation from the reference at the integer widths",
                      "unresolved": unknown}
        result = {
            "schema": "research/result/v0.1",
            "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
            "workload_fingerprint": {"params": repr(REAL), "timeout_s": args.timeout, "step_timeout_s": args.step_timeout},
            "validation": validation,
            "measurements": [{"name": f"z3.{r.gadget}.{r.query}", "value": r.seconds, "unit": "s"} for r in rows],
            "measurement_files": ["search.json"],
            "artifacts": ["search.json"],
        }
        with open(os.path.join(run_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
