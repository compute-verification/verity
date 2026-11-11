"""A Pearl-C -h3 call's rows as Definitions, and a Definition laid out in the units a proof takes.

A call C = A·B under ``-h3`` forms each row of A and each row of the fixed operand B against noise nothing supplies
(`verity.ml.protocols.pouw.schemes.pearl_c`: `PearlC.encode`):

    PearlCRowA{K, R, DOT, G}
        (salt, x: f32[K], root, unit, row, fa: Array<K, e4m3[32]>, fb: Array<K, e4m3[R]>)
            -> (codes: e4m3[K], P_A: bf16[2R], s, ρ, α, β_row)
        the row's line E_A from the salt, the row itself, B's root and the row's place (`pearl_c_noise.PearlCRowLine`),
        its forming against E_A and F_A, and its peel factors over F_B (`pearl_c_forming.PearlCForm`, `PearlCPeelA`)
    PearlCRowB{K, R, DOT, G}
        (x: f32[K], e: e4m3[32], fb: Array<K, e4m3[32]>, fa: Array<K, e4m3[R]>, g: Array<R, f32[R]>)
            -> (codes: e4m3[K], P_B: bf16[2R], s, ρ, α, β_row)
        the row's forming against its line E_B and F_B, and its peel factors over F_A and G (`PearlCForm`, `PearlCPeelB`)

Row A is the one place a call's noise is derived from a value the call keeps private, the row: E_B, F_A, F_B and G come
from seed_B = `PearlCSeedB` of the salt and B's root, both public, so whoever holds them derives those
(`PearlCLineKey`, `PearlCNoiseLine` at norm 16 for a basis' line, `pearl_c_forming.PearlCGram`), and here they are
parameters. On the codes and the factors a row is PoUW's row unit (`verity.ml.protocols.pouw.circuit.pc8`: `Pc8RowA`,
`Pc8RowB` at the same atom and promotion period, R = 32), row A with its line derived; a row's admission (the unit's
``ok``) is not here, and reads the statistics a row returns. R, the basis' rank, is 32 in a call and a static so that
circuit-check binds a row small.

Each has a word form (``_v1``) and a Boolean version (``_v2``), the σ of the word form. A row is far over the 65,536 ANDs a
Boolean Definition is lowered whole at, and is no proof unit: `units` lays a Boolean Definition out in the Definitions
under that size its Calls reach, the units a proof of it takes, and refuses one that has an AND outside them.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition, bind, composite
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.definitions.boolean import sigma_lift
from verity.ml.definitions.registry.pearl_c_forming import LINE, SLICE, PearlCForm, PearlCPeelA, PearlCPeelB
from verity.ml.definitions.registry.pearl_c_noise import BYTES32, BYTES64, PearlCRowLine

__all__ = ["LOWERED_WHOLE", "PearlCRowA", "PearlCRowAV2", "PearlCRowB", "PearlCRowBV2", "ands", "bindings", "units"]

F32, BF16, E4M3, U32 = Value(32), Value(16), Value(8), Value(32)
#: the ANDs circuit-check lowers a Boolean Definition whole up to (`circuit_check.checks.Options.lower_max_ands`): a unit's size
LOWERED_WHOLE = 1 << 16

_TESTS = "verity/ml/definitions/tests/test_pearl_c_call.py"
_HELD = f"protocol-defined; equal to `verity.ml.protocols.pouw.schemes.pearl_c.PearlC.encode` on its vectors ({_TESTS})"


def _refuse(b: dict) -> str | None:
    if b.get("K", SLICE) < SLICE or b.get("K", SLICE) % SLICE:
        return f"the chain multiplies in atoms of k = {SLICE}: K is a positive multiple of it"
    if b.get("G", 0) < 0 or not 1 <= b.get("R", 1) <= SLICE:
        return f"G is a number of atoms (0: no promotion) and R a rank of 1 to {SLICE}, the codes of a noise line"
    return None


def _row(S, factors: Array):
    return Tuple(Array(S.K, E4M3), factors, F32, F32, F32, F32)


def _sig_a(S):
    return ((("salt", BYTES32), ("x", Array(S.K, F32)), ("root", BYTES64), ("unit", U32), ("row", U32),
             ("fa", Array(S.K, LINE)), ("fb", Array(S.K, Array(S.R, E4M3)))), _row(S, Array(2 * S.R, BF16)))


def _sig_b(S):
    return ((("x", Array(S.K, F32)), ("e", LINE), ("fb", Array(S.K, LINE)), ("fa", Array(S.K, Array(S.R, E4M3))),
             ("g", Array(S.R, Array(S.R, F32)))), _row(S, Array(2 * S.R, BF16)))


def _formed(B, S, x, e, f):
    form = B.call(bind(PearlCForm, K=S.K, DOT=S.DOT), x, e, f)
    return [form[i] for i in range(6)]


@composite("PearlCRowA", 1, ["K", "R", "DOT", "G"], _sig_a, conformance=_HELD, refuse=_refuse)
def PearlCRowA(B, S, salt, x, root, unit, row, fa, fb):
    """Row ``row`` of A in unit ``unit`` of a -h3 call: its line E_A derived from the salt, the row and B's root
    (`PearlCRowLine_v1`), the row formed against E_A and the basis F_A (`PearlCForm_v1`), and row i of P_A over F_B's R
    columns (`PearlCPeelA_v1`). The row's codes, its peel factors, and its statistics s, ρ, α, β_row."""
    codes, ep, s, rho, alpha, beta = _formed(B, S, x, B.call(bind(PearlCRowLine, K=S.K), salt, x, root, unit, row), fa)
    factors = B.call(bind(PearlCPeelA, K=S.K, R=S.R, DOT=S.DOT, G=S.G), codes, ep[:S.R], fb)
    return tuple_of(codes, factors, s, rho, alpha, beta)


@composite("PearlCRowB", 1, ["K", "R", "DOT", "G"], _sig_b, conformance=_HELD, refuse=_refuse)
def PearlCRowB(B, S, x, e, fb, fa, g):
    """A row of the fixed operand B (a column of the product) in a call: the row formed against its line E_B and the basis
    F_B (`PearlCForm_v1`), and row j of P_B over F_A's R columns and the Gram matrix G (`PearlCPeelB_v1`). The row's
    codes, its peel factors, and its statistics s, ρ, α, β_row."""
    codes, ep, s, rho, alpha, beta = _formed(B, S, x, e, fb)
    factors = B.call(bind(PearlCPeelB, K=S.K, R=S.R, DOT=S.DOT, G=S.G), codes, ep[:S.R], fa, g)
    return tuple_of(codes, factors, s, rho, alpha, beta)


PearlCRowAV2 = sigma_lift(PearlCRowA, 2)
PearlCRowBV2 = sigma_lift(PearlCRowB, 2)


# ---- a Definition in units ---------------------------------------------------------------------------------------------------

def ands(fn: SpecializedDefinition | PrimitiveDefinition, _memo: dict | None = None) -> int:
    """The ANDs of a Boolean Definition: its `And_v1` gates through every Call, each batch member and scan step counted."""
    if isinstance(fn, PrimitiveDefinition):
        return int(fn is BL.AND)
    memo = {} if _memo is None else _memo
    if fn.id not in memo:
        memo[fn.id] = sum(n.n * ands(n.fn, memo) for n in fn.body.nodes)
    return memo[fn.id]


def units(fn: SpecializedDefinition, limit: int = LOWERED_WHOLE) -> dict[str, tuple[int, int]]:
    """The Boolean Definition ``fn`` laid out in units: each Definition of at most ``limit`` ANDs its Calls reach from
    Definitions over it, as ``{id: (Calls, ANDs of one)}``, so ``fn``'s ANDs are the sum of Calls · ANDs. A Definition
    without an AND (a constant, a rewiring) is no unit and is left out. Refused: a Definition over ``limit`` with an AND of
    its own, which no unit holds."""
    memo: dict = {}
    out: dict[str, tuple[int, int]] = {}

    def walk(d, times: int) -> None:
        n_ands = ands(d, memo)
        if n_ands <= limit:
            if n_ands:
                out[d.id] = (out.get(d.id, (0, n_ands))[0] + times, n_ands)
            return
        for node in d.body.nodes:
            if node.fn is BL.AND:
                raise ValueError(f"{d.id} is {n_ands} ANDs, over {limit}, and has ANDs of its own: it is not laid out in units")
            if not isinstance(node.fn, PrimitiveDefinition):
                walk(node.fn, times * node.n)

    walk(fn, 1)
    return dict(sorted(out.items()))


def bindings() -> list:
    """The bindings circuit-check takes (the sidecar): each row on words and on bits, on the H100's atom at G = 4 over one
    atom of the row and rank 1, the least a row is."""
    from verity.ml.circuits.ops.prims import HopperE4m3QgmmaDot32
    from verity.ml.definitions.boolean import boolean_version

    statics = {"K": SLICE, "R": 1, "G": 4}
    return [bind(PearlCRowA, DOT=HopperE4m3QgmmaDot32, **statics), bind(PearlCRowB, DOT=HopperE4m3QgmmaDot32, **statics),
            bind(PearlCRowAV2, DOT=boolean_version(HopperE4m3QgmmaDot32), **statics),
            bind(PearlCRowBV2, DOT=boolean_version(HopperE4m3QgmmaDot32), **statics)]
