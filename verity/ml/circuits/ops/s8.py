"""The exact s8 matmul as one Definition family, which PoUW, memory accounting and vLLM all bind: every arm, plain and edited,
is the same chain of exact s8 IMMA steps (`prims.BlackwellS8ImmaDot32_v1`, `prims.BlackwellU8S8ImmaDot32_v1`).

    S8Dot_v1{K, L, T0, C, DOT, A, B, TAIL, TDOT, NOISE, LIMBS, COMMIT}
        ([acc: Value<32>,] a, b: Array<K, Value<8>>[, ta, tb: Array<32 m, Value<8>>])
        -> Value<32>                                                   COMMIT "final" and no tail
         | RecordType(T: Array<n, Value<32>>[, D: Array<g, Value<32>>])     otherwise
    S8Matmul_v1{N, K, L, T0, C, DOT, A, B, TAIL, TDOT, NOISE, LIMBS, COMMIT}
        ([acc: Array<N, Value<32>>,] a: Array<K, Value<8>>, b: Array<N, Array<K, Value<8>>>
         [, ta: Array<32 m, Value<8>>, tb: Array<N, Array<32 m, Value<8>>>])
        -> Array<N, S8Dot's return>: `a` (and `ta`) against the N rows of `b` (and `tb`), S8Dot batched
    LimbSplit_v1{FORM, N, Y}   (y: Value<32>) -> Array<N, Value<8>>: a runtime s32 word split into s8 limbs, least significant first

**The chain.** `a` against `b` in K/32 `DOT` steps in ascending k, from T_0: zero (``T0 = "zero"``) or the incoming
accumulator `acc` (``T0 = "input"``, whose interval `C` its producer declares). T_t is the accumulator after t·L products, for
t = 1 .. K/L (the k block `L` a multiple of 32 dividing K).

**The tail.** `TAIL` is a tuple of groups ``(atoms, mode, A_g, B_g)``: `atoms` more `TDOT` steps (tail atoms, 32 k-steps each)
over the next 32·atoms codes of `ta` and `tb`, in order. A group whose mode is ``"chain"`` continues the main chain from
T_{K/L} (the first group only: egemm's and co32c's peel on the same accumulator); a ``"fresh"`` group starts from zero (a
separate word, as egemm's S_0·U and P·V's digit products). Each group's last accumulator is one D word.

**Committed words.** `COMMIT` is the layout of the words the arm commits: ``"final"`` (L = K, T = [T_1]: the plain arm, whose
one word is its return) or ``"prefix"`` (T_1 .. T_{K/L}: the running sums PoUW's receipts fold). PIN may add layouts; each is
a new value here. The receipt fold over T (`PouwDuoAbsorb`) and the noise derivation (SHA-512 over key ‖ tag ‖ j) are
Definitions of their own: the family is a pure exact matmul, and the coin, the noise and the limbs are its input values.

**Bounds, and no wrap.** `A` and `B` are the main chain's operand intervals ``(lo, hi)`` and each group's ``A_g``, ``B_g`` its
tail's, in the values the codes denote (s8 or u8 by the step's format), from the producing Definition (the quantizer's
clamp: steps.py's dither reaches -128). Binding refuses any interval outside its format, and any binding under which some
step's exact sum could leave s32: so every step's clamp (``.satfinite``) or wrap is the identity, and every T and D word is
the exact integer sum (`no_wrap`). A runtime incoming accumulator with no derivable interval is not this family.

**Arms.** `NOISE` says where the challenger's noise enters (``"none"``, ``"weight"``: W' = W + U or + c; ``"P"``: on the
probabilities' side; ``"both"``), and `LIMBS` how the tail's operands were split (``"none"``; ``"halves"``: cohalf's
X[:64] + X[64:]; ``"bytes"``: P·V's Y0 / Y1 / Y2; ``"byteclamp"``: egemm's S_0 / S_1 / S_2), each by `LimbSplit_v1` (or the
halves' add) on its producer's side. The plain serving step is the family at NOISE "none", no tail, LIMBS "none" and COMMIT
"final": b = c = U = 0, with rint in the quantizer.

**Limb splits.** A runtime s32 word is never read back as s8 codes in place: `LimbSplit_v1` is the registered split, a carry
per limb beyond the first. ``FORM = "bytes"``: l_i = sext8(r_i), r_{i+1} = (r_i - l_i) / 256, the last limb r_{N-1} itself, so
y = sum 256^i l_i. ``FORM = "byteclamp"`` (N = 3): s_0 = sext8(y), h = (y - s_0) / 256, s_1 = clamp(h, -127, 127),
s_2 = h - s_1, so y = s_0 + 256 (s_1 + s_2). `Y` is y's interval from its producer, and binding refuses one whose last limb
could leave s8.

**Census.** One class per binding (the descriptor id), the arm read from the statics. `counts` is census/v1's rule for the
family (`census.FAMILY_COUNTS`): per coordinate K served MACs and 32 per tail atom apart, no receipt folds; a LimbSplit
instance is N - 1 carries (`carries`).
"""
from __future__ import annotations

from typing import Any, Callable

from verity.ml.circuits.census import FAMILY_COUNTS
from verity.ml.circuits.defs import CompositeDefinition, bind, composite, primitive
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.silicon.integer import S32_MAX, S32_MIN, s32_value, s32_word
from verity.ml.circuits.types import Array, RecordType, Type, Value

from . import prims as P

__all__ = ["COMMITS", "FORMS", "LIMBS", "LimbSplit", "NOISES", "S32ByteCarry", "S32ClampRest", "S32ClampS8", "S32LowByte",
           "S8Dot", "S8Matmul", "SMOKE_EDITED", "SMOKE_PLAIN", "carries", "counts", "last_limb", "no_wrap"]

S32, V8 = Value(32), Value(8)
S8_RANGE, U8_RANGE = (-128, 127), (0, 255)
NOISES = ("none", "weight", "P", "both")
LIMBS = ("none", "halves", "bytes", "byteclamp")
COMMITS = ("final", "prefix")
FORMS = ("bytes", "byteclamp")

#: each step's operand formats (a, b), by its id: the word steps and the Boolean Definitions that compute them
_STEP_FORMATS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "BlackwellS8ImmaDot32_v1": (S8_RANGE, S8_RANGE),
    "BlackwellU8S8ImmaDot32_v1": (U8_RANGE, S8_RANGE),
}


def _formats(step: Any, boolean: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    word = getattr(getattr(step, "defn", step), "word", None)
    fid = getattr(word if boolean else step, "id", None)
    if fid not in _STEP_FORMATS:
        raise ValueError(f"an s8 family step is one of {sorted(_STEP_FORMATS)}{' as a Boolean Definition' if boolean else ''}, "
                         f"not {getattr(step, 'id', step)!r}")
    return _STEP_FORMATS[fid]


def _interval(v: Any, what: str, within: tuple[int, int]) -> tuple[int, int]:
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
            and v[0] <= v[1]):
        raise ValueError(f"{what} is an interval (lo, hi) of ints with lo <= hi, not {v!r}")
    if not (within[0] <= v[0] and v[1] <= within[1]):
        raise ValueError(f"{what} = {tuple(v)} leaves its format's {within}: declare the producer's interval, inside the format")
    return int(v[0]), int(v[1])


def _products(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    corners = [x * y for x in a for y in b]
    return min(corners), max(corners)


def _groups(S: Any, boolean: bool) -> list[tuple[int, str, tuple[int, int], tuple[int, int]]]:
    tail = S.TAIL
    if not isinstance(tail, (list, tuple)):
        raise ValueError(f"TAIL is a tuple of groups (atoms, mode, A_g, B_g), not {tail!r}")
    if tail and S.TDOT is None:
        raise ValueError("a tail names its step: bind TDOT to the s8 step its atoms run")
    if not tail and S.TDOT is not None:
        raise ValueError("TDOT is None without a tail")
    fa, fb = _formats(S.TDOT, boolean) if tail else (S8_RANGE, S8_RANGE)
    out = []
    for i, g in enumerate(tail):
        if not (isinstance(g, (list, tuple)) and len(g) == 4):
            raise ValueError(f"TAIL[{i}] is (atoms, mode, A_g, B_g), not {g!r}")
        atoms, mode, ga, gb = g
        if isinstance(atoms, bool) or not isinstance(atoms, int) or atoms < 1:
            raise ValueError(f"TAIL[{i}]'s atom count is an int >= 1, not {atoms!r}")
        if mode not in ("chain", "fresh") or (mode == "chain" and i):
            raise ValueError(f"TAIL[{i}]'s mode is 'fresh', or 'chain' for the first group only, not {mode!r}")
        out.append((atoms, mode, _interval(ga, f"TAIL[{i}]'s A_g", fa), _interval(gb, f"TAIL[{i}]'s B_g", fb)))
    return out


def no_wrap(S: Any, boolean: bool = False) -> dict[str, tuple[int, int]]:
    """The interval of every word the binding's chain computes ({"T_t" | "D_g": (lo, hi)}), each within s32: the theorem that
    no step clamps or wraps, so every word is the exact sum.  Raises ValueError, naming the first word that could leave s32,
    and on any binding outside the family (``boolean``: the steps are the Boolean Definitions, as `S8Dot_v2` binds)."""
    K, L = S.K, S.L
    if isinstance(K, bool) or not isinstance(K, int) or K < 32 or K % 32:
        raise ValueError(f"K is a positive multiple of 32, not {K!r}")
    if isinstance(L, bool) or not isinstance(L, int) or L < 32 or L % 32 or K % L:
        raise ValueError(f"L is a multiple of 32 dividing K = {K}, not {L!r}")
    if S.COMMIT not in COMMITS:
        raise ValueError(f"COMMIT is one of {COMMITS}, not {S.COMMIT!r}")
    if S.COMMIT == "final" and L != K:
        raise ValueError(f"COMMIT 'final' commits T_1 only: bind L = K = {K}, not {L}")
    if S.NOISE not in NOISES:
        raise ValueError(f"NOISE is one of {NOISES}, not {S.NOISE!r}")
    if S.LIMBS not in LIMBS:
        raise ValueError(f"LIMBS is one of {LIMBS}, not {S.LIMBS!r}")
    groups = _groups(S, boolean)
    if bool(groups) != (S.LIMBS != "none"):
        raise ValueError(f"a tail carries split operands and only a tail does: TAIL {S.TAIL!r} with LIMBS {S.LIMBS!r}")
    if S.NOISE == "none" and groups:
        raise ValueError("the plain arm (NOISE 'none') has no tail")
    if S.T0 == "zero":
        if S.C is not None:
            raise ValueError("T0 'zero' has no incoming accumulator: bind C = None")
        start = (0, 0)
    elif S.T0 == "input":
        start = _interval(S.C, "C, the incoming accumulator's interval", (S32_MIN, S32_MAX))
    else:
        raise ValueError(f"T0 is 'zero' or 'input', not {S.T0!r}")
    fa, fb = _formats(S.DOT, boolean)
    lo_p, hi_p = _products(_interval(S.A, "A", fa), _interval(S.B, "B", fb))
    words: dict[str, tuple[int, int]] = {}

    def run(name: str, at: tuple[int, int], steps: int, p: tuple[int, int]) -> tuple[int, int]:
        lo, hi = at
        for s in range(1, steps + 1):
            lo, hi = lo + 32 * p[0], hi + 32 * p[1]
            if lo < S32_MIN or hi > S32_MAX:
                raise ValueError(f"{name}'s step {s} could leave s32 ([{lo}, {hi}]): the bounds don't prove no wrap; narrow the "
                                 f"producers' intervals or K")
        return lo, hi

    at = start
    for t in range(1, K // L + 1):
        at = words[f"T_{t}"] = run(f"T_{t}", at, L // 32, (lo_p, hi_p))
    for g, (atoms, mode, ga, gb) in enumerate(groups):
        words[f"D_{g}"] = run(f"D_{g}", at if mode == "chain" else (0, 0), atoms, _products(ga, gb))
    return words


def _signature(S: Any, acc_t: Type, op_t: Type) -> tuple[tuple[tuple[str, Type], ...], Type]:
    no_wrap(S, boolean=acc_t != S32)
    m = 32 * sum(g[0] for g in S.TAIL)
    params: list[tuple[str, Type]] = [("acc", acc_t)] if S.T0 == "input" else []
    params += [("a", Array(S.K, op_t)), ("b", Array(S.K, op_t))]
    if m:
        params += [("ta", Array(m, op_t)), ("tb", Array(m, op_t))]
    n_t, n_d = S.K // S.L, len(S.TAIL)
    if S.COMMIT == "final" and not n_d:
        return tuple(params), acc_t
    fields: list[tuple[str, Type]] = [("T", Array(n_t, acc_t))] + ([("D", Array(n_d, acc_t))] if n_d else [])
    return tuple(params), RecordType(*fields)


def _dot_body(B: Any, S: Any, args: tuple, zero: Callable[[Any], Coll], acc_t: Type, op_t: Type) -> Coll:
    _, ret = _signature(S, acc_t, op_t)
    args = list(args)
    acc = args.pop(0) if S.T0 == "input" else zero(B)
    a, b = args[0], args[1]
    ta, tb = (args[2], args[3]) if len(args) > 2 else (None, None)
    T: list[Coll] = []
    for s in range(S.K // 32):
        acc = B.call(S.DOT, acc, a[32 * s:32 * s + 32], b[32 * s:32 * s + 32])
        if (32 * (s + 1)) % S.L == 0:
            T.append(acc)
    D: list[Coll] = []
    j = 0
    for atoms, mode, _ga, _gb in S.TAIL:
        d = T[-1] if mode == "chain" else zero(B)
        for _ in range(atoms):
            d = B.call(S.TDOT, d, ta[32 * j:32 * j + 32], tb[32 * j:32 * j + 32])
            j += 1
        D.append(d)
    if not isinstance(ret, RecordType):
        return T[0]
    return Coll(ret, concat([w.refs for w in T + D]))


def _word_zero(B: Any) -> Coll:
    return B.call(P.ZERO32)


@composite("S8Dot", 1, ["K", "L", "T0", "C", "DOT", "A", "B", "TAIL", "TDOT", "NOISE", "LIMBS", "COMMIT"],
           lambda S: _signature(S, S32, V8), conformance="exact-model-tested")
def S8Dot(B, S, *args):
    """One output coordinate of the exact s8 matmul: K/32 `DOT` steps from T_0 in ascending k, the committed T words, then the
    tail groups (the module docstring)."""
    return _dot_body(B, S, args, _word_zero, S32, V8)


def _matmul_signature(S: Any, dot: Any) -> tuple[tuple[tuple[str, Type], ...], Type]:
    if isinstance(S.N, bool) or not isinstance(S.N, int) or S.N < 1:
        raise ValueError(f"N is an int >= 1, not {S.N!r}")
    params, ret = dot.signature()
    rows = {"acc", "b", "tb"}
    return tuple((n, Array(S.N, t) if n in rows else t) for n, t in params), Array(S.N, ret)


_DOT_STATICS = ("K", "L", "T0", "C", "DOT", "A", "B", "TAIL", "TDOT", "NOISE", "LIMBS", "COMMIT")


def _coordinate(S: Any, dot: CompositeDefinition) -> Any:
    return bind(dot, **{k: getattr(S, k) for k in _DOT_STATICS})


def _matmul_body(B: Any, S: Any, args: tuple, dot: CompositeDefinition) -> Coll:
    coord = _coordinate(S, dot)
    names = [n for n, _ in coord.signature()[0]]
    return B.batch(coord, *args, axes=[0 if n in ("acc", "b", "tb") else None for n in names])


@composite("S8Matmul", 1, ["N"] + list(_DOT_STATICS), lambda S: _matmul_signature(S, _coordinate(S, S8Dot)),
           conformance="exact-model-tested")
def S8Matmul(B, S, *args):
    """`a` (and `ta`) against the N rows of `b` (and `tb`): `S8Dot_v1` batched over the rows, one incoming accumulator a row."""
    return _matmul_body(B, S, args, S8Dot)


# -- limb splits ---------------------------------------------------------------------------------------------------------


def _sext8(v: int) -> int:
    return ((v & 0xFF) ^ 0x80) - 0x80


@primitive("S32LowByte", 1, [("y", S32)], V8, conformance="protocol-defined")
def S32LowByte(y: int) -> int:
    """The s8 code of y's low byte (sext8(y) as two's complement): y & 0xFF."""
    return y & 0xFF


@primitive("S32ByteCarry", 1, [("y", S32)], S32, conformance="protocol-defined")
def S32ByteCarry(y: int) -> int:
    """The carry past y's low byte read as s8: (y - sext8(y)) / 256, exact, an s32 word."""
    v = s32_value(y)
    return s32_word((v - _sext8(v)) >> 8)


@primitive("S32ClampS8", 1, [("h", S32)], V8, conformance="protocol-defined")
def S32ClampS8(h: int) -> int:
    """clamp(h, -127, 127) as an s8 code."""
    return min(max(s32_value(h), -127), 127) & 0xFF


@primitive("S32ClampRest", 1, [("h", S32)], V8, conformance="protocol-defined")
def S32ClampRest(h: int) -> int:
    """h - clamp(h, -127, 127) as an s8 code (its low byte: binding `LimbSplit_v1` keeps it in s8)."""
    v = s32_value(h)
    return (v - min(max(v, -127), 127)) & 0xFF


def _carry(v: int) -> int:
    return (v - _sext8(v)) >> 8  # floor((v + 128) / 256): nondecreasing in v


def _clamp127(v: int) -> int:
    return min(max(v, -127), 127)


def last_limb(S: Any) -> tuple[int, int]:
    """The interval of the last limb of a y in `Y` (every other limb is an s8 code by construction).  Raises ValueError on a
    binding outside the family, or one whose last limb could leave s8."""
    if S.FORM not in FORMS:
        raise ValueError(f"FORM is one of {FORMS}, not {S.FORM!r}")
    if isinstance(S.N, bool) or not isinstance(S.N, int) or S.N < 2 or (S.FORM == "byteclamp" and S.N != 3):
        raise ValueError(f"N is the limb count: >= 2 for 'bytes', 3 for 'byteclamp'; not {S.N!r}")
    lo, hi = _interval(S.Y, "Y, y's interval", (S32_MIN, S32_MAX))
    if S.FORM == "bytes":
        for _ in range(S.N - 1):
            lo, hi = _carry(lo), _carry(hi)
        last = (lo, hi)
    else:
        h_lo, h_hi = _carry(lo), _carry(hi)
        last = (h_lo - _clamp127(h_lo), h_hi - _clamp127(h_hi))  # h - clamp(h) is nondecreasing in h
    if not (S8_RANGE[0] <= last[0] and last[1] <= S8_RANGE[1]):
        raise ValueError(f"LimbSplit{{FORM={S.FORM}, N={S.N}}}: y in {tuple(S.Y)} puts its last limb in {list(last)}, outside "
                         f"s8: bind more limbs or narrow Y to the producer's interval")
    return last


def _limb_signature(S: Any, word_t: Type, code_t: Type) -> tuple[tuple[tuple[str, Type], ...], Type]:
    last_limb(S)
    return (("y", word_t),), Array(S.N, code_t)


@composite("LimbSplit", 1, ["FORM", "N", "Y"], lambda S: _limb_signature(S, S32, V8), conformance="exact-model-tested")
def LimbSplit(B, S, y):
    """y as N s8 limbs, least significant first (the module docstring's forms)."""
    if S.FORM == "byteclamp":
        h = B.call(S32ByteCarry, y)
        limbs = [B.call(S32LowByte, y), B.call(S32ClampS8, h), B.call(S32ClampRest, h)]
    else:
        limbs, r = [], y
        for _ in range(S.N - 1):
            limbs.append(B.call(S32LowByte, r))
            r = B.call(S32ByteCarry, r)
        limbs.append(B.call(S32LowByte, r))
    return Coll(Array(S.N, V8), concat([c.refs for c in limbs]))


# -- census -------------------------------------------------------------------------------------------------------------


def counts(bindings: dict) -> dict[str, int]:
    """census/v1's counters for one instance of `S8Dot_v1` or `S8Matmul_v1` from its statics: K served MACs a coordinate (the
    served product's m·n·k summed over coordinates), the tail's k-steps apart (32 an atom), and no receipt folds (a fold is
    `PouwDuoAbsorb`'s, over the named T words)."""
    n = bindings.get("N", 1)
    return {"served_macs": n * bindings["K"], "tail_atoms": n * 32 * sum(g[0] for g in bindings["TAIL"]), "receipt_folds": 0}


def carries(bindings: dict) -> int:
    """The carries one `LimbSplit_v1` instance computes: one a limb beyond the first."""
    return bindings["N"] - 1


S8Dot.counts = counts  # type: ignore[attr-defined]
S8Matmul.counts = counts  # type: ignore[attr-defined]
LimbSplit.carries = carries  # type: ignore[attr-defined]
FAMILY_COUNTS["S8Dot_v1"] = FAMILY_COUNTS["S8Matmul_v1"] = counts

#: the plain serving arm at K = 64 (vLLM's int8 linear: rint quantizer, b = c = U = 0, ±127 operands)
SMOKE_PLAIN: dict = {"K": 64, "L": 64, "T0": "zero", "C": None, "DOT": P.BlackwellS8ImmaDot32, "A": (-127, 127),
                     "B": (-127, 127), "TAIL": (), "TDOT": None, "NOISE": "none", "LIMBS": "none", "COMMIT": "final"}
#: an edited arm at K = 64 in egemm's shape: the dithered X in [-127, 127] against W' = W + U (|W| <= 63, U in ±1..±64), T_t
#: every 32, and the peel's byteclamp limbs against U: (S_1 | S_2) on one fresh word, S_0 on another
SMOKE_EDITED: dict = {"K": 64, "L": 32, "T0": "zero", "C": None, "DOT": P.BlackwellS8ImmaDot32, "A": (-127, 127),
                      "B": (-127, 127), "TAIL": ((2, "fresh", (-127, 127), (-64, 64)), (1, "fresh", (-128, 127), (-64, 64))),
                      "TDOT": P.BlackwellS8ImmaDot32, "NOISE": "weight", "LIMBS": "byteclamp", "COMMIT": "prefix"}
