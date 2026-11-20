"""Pearl+'s -h3 tile as one node that writes its tile of the linear's y: a tile's rows formed against noise from the live
salt, their admission, the tile's inert-work bound and its digest, and the served y the core computes from its U, so that a
reader of the linear takes y by wire from the node that wrote it and the verifier holds the node's ``ok`` to 1.

    PearlPlusTileCore_v3{K,TM,TR,TN,N,RHO,RHOR,DEV}    on bits: the σ of the core's word body
        (a: Array<TM, f32[K]>, b: Array<TN, f32[K]>, cert, salt: u8[32], root: u8[64], unit: u32, rows: Array<TM, u32>,
         eb: Array<TN, e4m3[32]>, fa, fb: Array<K, e4m3[32]>, g: Array<32, f32[32]>)
        -> (digest: u8[64], y: Array<TR, bf16[TN]>, ok: bit)
    PearlPlusTileSeed_v4{K,TN}                          on 16-bit words
        (salt: u16[16], root: u16[32], col: u32)
        -> (key: u16[16], eb_0 … eb_{TN-1}: Array<1, u16[16]>, f_0 … f_7: Array<K/8, u16[2]>)
        key = `PearlCLineKey_v1`(`PearlCSeedB_v1`(salt, root)), eb's lines one a field (E_B's line at col·TN + j), and
        F's line indices t in `F_ROWS` = 8 fields of K/8 (the σ of its unregistered word body and E_B's line's, behind
        `WordBits` and `BitsWord{16}`)
    PearlPlusTileLine_v4{SIDE,FACTOR,G}                 on 16-bit words: the σ of one line's word body on each index
        (key: u16[16], index: Array<G, u16[2]>) -> Array<G, u16[16]>   `PearlCNoiseLine_v1`(key, SIDE, FACTOR, index, norm)
    PearlPlusGramRow_v5{K,DEV,P}                        on 16-bit words: the σ of one Gram row's word body
        (fb, fa: Array<K, u16[16]>) -> Array<32, u16[2]>       row P of `PearlCGram_v1`, for F_B's column P

Each port and each output of these units is in 16-bit words, the width C commits a value in: a byte string two bytes a
word, the first the low byte, and a value on bits (a line's 32 E4M3 codes, an index's or an FP32 word's 32 bits) its
bits in order, low bit first, 16 a word; each unit takes its words' bits inside (`WordBits{16}`) and writes its bits
back in words (`BitsWord{16}`), so no reader reads a writer's field at a width other than the writer's.

The node (`pearl_plus_credited.PearlPlusTileCoreWords_v5`) Calls the seed, which writes eb's TN lines (SIDE 1, FACTOR 0)
itself, maps the line over F's 8 groups of K/8 indices (SIDE 0 and 1, FACTOR 1, G K/8), and Calls the Gram row for each
of F_B's 32 columns: each line unit and each row is independent given the key, so each is a unit of its own when C is
laid (`verity.circuits.laying`), and eb, fa, fb and g reach the core as wired values. Every unit reads each of its
writers' output fields whole: a line unit its key and its own indices (the seed writes one field a unit), a Gram row
F_B's and F_A's lines (its column P a static, so F_B's column is no part of a line's field), and the core the lines and
the rows. A reader of F reads 8 rows of each side whatever K, and a reader of eb one row a line, so that a row of B that
the laying opens the core into reads its own line whole; Flock gives each row a block reads its own hm96 slot. eb's
lines are the seed's and no unit of their own: a unit of one line holds no row with a compression before its last
block (its key, its index and its line are each shorter), which a Flock block needs.

The core, from the registry's call Definitions (`verity.protocols.accounting.compute.pouw.prover.circuit.pearl_c_call`) and PoUW's inert-work bound (`inert_bound`):

  - **real rows.** TR, a static of the Definition (1 <= TR <= TM), is the tile's real rows: the first TR of its TM rows
    are the call's, and rows TR .. TM − 1 are the zero rows a call's A is padded with to a multiple of 16 (filler).  It is
    bound from the call's record (`UnitRecord.real_rows`), never an input: a count the prover chose could call a live row
    filler.  Lean's tile rule (`PouwPearlPlus/Rule.lean` `rowsFailure`, `Records.lean` `credited`, `Tile.lean`) is the
    same: a filler row is held to zero, word for word (−0 is not zero), and is neither admitted, in the inert-work bound,
    credited nor served; it is formed and in the tile's digest like any row.
  - **rows.** Row i of A (``a[i]``, the tile's row ``rows[i]`` of unit ``unit``) by `PearlCRowA_v1`: its line E_A from the
    salt, the row's own digest and B's root (``root``), its codes A′_i, its peel factors P_A,i and its statistics; row j
    of B by `PearlCRowB_v1` against its line ``eb[j]``.  R = 32, the device's atom and promotion period.
  - **admission.** Each real row's and each B row's `PearlPlusAdmit_v1`, `Pc8RowA`'s ``ok`` on the statistics the row
    returned: every word finite, the noise floor ρ·α >= 1 and at most k/64 dead words (`pc8`'s module docstring,
    `PearlC.admissible`).
  - **inert-work bound.** A's salt-dead counts (`PearlPlusSaltDead_v1`), then `Pc8TileCap_v2` on the codes, the factors,
    the admission words (a filler row's the constant 0, so its words are outside the bound's debit) and the prover's
    certificates (``cert``): C̃ and U, and the verdict of the tile's bound at ρ = 1/RHO and each admitted row's at
    ρr = 1/RHOR (`inert_bound.row_rho`: 4ρ at a census k, else ρ).
  - **leaf.** The tile's ``sha512/row-seg/v1`` digest of its messages (`leaves.tile_messages`, `PearlCRowDigest_v1`): the
    preimage of its leaf in the transcript tree.
  - **y.** Each real row's TN served BF16 words from U (`PearlCUnscale_v1`, no bias): the tile's TR × TN words of the
    linear's y, its rows ``rows[:TR]`` and its column tile's TN columns, which C's readers of the linear take by wire.
  - ``ok`` is every real row's and B row's admission, the bound's verdict, each served word finite (`pearl_c_u.y` refuses
    a non-finite word and a BF16 overflow) and each filler row's every word zero, and-ed.

The core reads no column index: E_B's lines ``eb`` are the column tile's, which the noise derives from ``col``.  The node
(`pearl_plus_credited.PearlPlusTileCoreWords_v5`) derives the noise and Calls the core on the salt and B's root of the public row
the verifier fills itself (`pearl_plus_credited.public`), so the salt and the root reach the tile only through that row.

The core is one registered Definition, its σ: its word body is an unregistered Definition the σ lifts and names as its
word view, through which a fixture tile is evaluated on words against the scheme (`reference`).  Every Definition the core
Calls is a σ, a Boolean gadget or a Call of them, so it is laid out in units under 65,536 ANDs (`pearl_c_call.units`).
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any, Sequence

from verity.circuits import boolean as BL
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.circuits.ops.fp32 import F32MulV2
from verity.circuits.ops.prims import Bf16ToF32, F32ToE4m3Sat
from verity.circuits.ops.scalar import Bf16GtStrict, BitAnd, F32IsFinite, I32Add, I32Eq, I32Le
from verity.circuits.population import BitsWord, WordBits
from verity.circuits.refs import Coll, array_of, concat, tuple_of
from verity.circuits.silicon import fp32
from verity.circuits.types import Array, Tuple, Value
from verity.circuits.ops.boolean.versions import lift_body
from verity.protocols.accounting.compute.pouw.prover.circuit.pearl_c_call import PearlCRowA, PearlCRowB
from verity.protocols.accounting.compute.pouw.prover.circuit.pearl_c_forming import LINE, PearlCChain
from verity.protocols.accounting.compute.pouw.prover.circuit.pearl_c_noise import (
    BYTES32, BYTES64, NORM_E, NORM_F, PearlCLineKey, PearlCNoiseLine, PearlCRowDigest, PearlCSeedB)
from verity.protocols.accounting.compute.pouw.prover.circuit.pearl_c_tile import PearlCUnscale

from verity.protocols.accounting.compute.pouw.verifier.protocol import DomainError, Shape
from ..schemes import pearl_c as C
from ..schemes import pearl_c_inert_work as D
from ..schemes import pearl_c_u as CU
from ..schemes import pearl_kw as P
from ..schemes.pearl_c_device import DEVICES
from verity.circuits.ops.boolean import words as _boolean
from . import inert_bound, pc8
from . import tile_units as U
from . import leaves as L
from verity.circuits.ops import words as W

__all__ = ["LINES", "PUBLIC", "PearlPlusAdmit", "PearlPlusAdmitV2", "PearlPlusGramRowV5", "PearlPlusSaltDead",
           "PearlPlusSaltDeadV2", "PearlPlusTileCoreV3", "PearlPlusTileLineV4", "PearlPlusTileSeed", "core", "gram_row", "line",
           "noise", "reference", "seed", "served"]

F32, U32, BIT, I32, BF16, E4M3, W16 = Value(32), Value(32), Value(1), Value(32), Value(16), Value(8), Value(16)
R = pc8.R
SLICE = C.SLICE


# ---- σ with the cap and PoUW's words ------------------------------------------------------------------------------------

def _bits_of(fn: Any) -> Any:
    """The Boolean Definition a σ here Calls for ``fn`` where `boolean_version` may not hold it: PoUW's words (`boolean`,
    which `boolean_version`'s table holds only when imported before its first use) and the cap's word and tile, whose
    Boolean versions are this module's and unregistered, so that no other Program's σ changes."""
    if isinstance(fn, PrimitiveDefinition):
        return bind(_WORDS[id(fn)]) if id(fn) in _WORDS else None
    if isinstance(fn, SpecializedDefinition) and id(fn.defn) in _LIFTED:
        return bind(_LIFTED[id(fn.defn)], **fn.bindings)
    return None


def _on_bits(word: CompositeDefinition, version: int, register: bool = True) -> CompositeDefinition:
    """σ of ``word`` (`verity.circuits.ops.boolean.versions.sigma_lift`'s, whose statics here are integers and names), each Call the
    Boolean version of its Definition or `_bits_of`'s."""
    statics = tuple(word.statics)

    def at(S) -> SpecializedDefinition:
        return bind(word, **{k: getattr(S, k) for k in statics})

    def signature(S):
        params, ret = at(S).signature()
        return tuple((n, BL.bit_type(t)) for n, t in params), BL.bit_type(ret)

    d = CompositeDefinition(word.name, version, statics, signature, lambda B, S, *a: lift_body(at(S), _bits_of)(B, S, *a),
                            doc=f"σ({word.name}_v{word.version}): its body on bits, each Call the Boolean version of its "
                                "Definition", conformance=word.conformance, refuse=word.refuse, register=register)
    d.word = (lambda b: bind(word, **b)) if statics else bind(word)  # type: ignore[attr-defined]
    d.sigma_of = word  # type: ignore[attr-defined]
    _LIFTED[id(word)] = d
    return d


_WORDS = {id(d.word): d for d in vars(_boolean).values() if isinstance(d, CompositeDefinition) and isinstance(
    getattr(d, "word", None), PrimitiveDefinition)}
_LIFTED: dict[int, CompositeDefinition] = {}
_on_bits(inert_bound.Pc8CapWordDef, 2, register=False)
_on_bits(inert_bound.Pc8TileCapDefV2, 3, register=False)


# ---- a row's admission and salt-dead counts -----------------------------------------------------------------------------

def _refuse_k(b: dict) -> str | None:
    k = b.get("K", 256)
    return None if isinstance(k, int) and k >= 64 and k % 64 == 0 else f"a row is whole 64-word blocks: K = {k!r}"


def _admit_body(B, S, x, s, rho, alpha):
    e, k = U._Emit(B), S.K
    positions = range(0, k, C.S5_STRIDE)
    total = 0
    for p in positions:
        total = e.call(W.PouwSqAcc, total, x[p])
    dead = pc8._dead(e, x, e.call(W.PouwSqCeil, total, len(positions)), 0, k)
    floor, finite = pc8._bounds(e, s, rho, alpha)
    return e.call(BitAnd, finite, e.call(BitAnd, floor, e.call(I32Le, dead, k // 64)))


PearlPlusAdmit = CompositeDefinition(
    "PearlPlusAdmit", 1, ("K",), lambda S: ((("x", Array(S.K, F32)), ("s", F32), ("rho", F32), ("alpha", F32)), BIT),
    _admit_body, refuse=_refuse_k, conformance="protocol-defined",
    doc="A row's admission from its K FP32 words and the s, ρ and α its row Definition returned: s and ρ finite, "
        "fma(ρ, α, −1) >= +0 and at most K/64 dead words (`Pc8RowA`'s ok, `PearlC.admissible`).")
PearlPlusAdmitV2 = _on_bits(PearlPlusAdmit, 2)


def _salt_dead_body(B, S, x, alpha, beta):
    e, k = U._Emit(B), S.K
    dead = inert_bound.SALT_DEAD[DEVICES[S.DEV].atom.name]
    cb = e.call(F32ToE4m3Sat, e.call(F32MulV2, beta, inert_bound.FLAT_E))
    bits = [e.call(dead, e.call(F32MulV2, alpha, x[t]), cb) for t in range(k)]
    return array_of([inert_bound._wire(e, 32, inert_bound._count(e, bits[s:s + SLICE])) for s in range(0, k, SLICE)])


def _salt_dead_sig(S):
    if DEVICES[S.DEV].atom.name not in inert_bound.SALT_DEAD:
        raise DomainError(f"Pearl-C's cap circuit has no salt-dead bit for {S.DEV!r}'s atom")
    return (("x", Array(S.K, F32)), ("alpha", F32), ("beta", F32)), Array(S.K // SLICE, I32)


PearlPlusSaltDead = CompositeDefinition(
    "PearlPlusSaltDead", 1, ("K", "DEV"), _salt_dead_sig, _salt_dead_body, refuse=_refuse_k, conformance="protocol-defined",
    doc="A row of A's salt-dead counts per block of 32 elements, from its FP32 words and its α and β_row: `Pc8RowACap`'s, "
        "the elements whose support's two ends agree on f32(α·x) and e4m3(f32(β·44)) (`SaltDead`'s `EndsAgree`).")
PearlPlusSaltDeadV2 = _on_bits(PearlPlusSaltDead, 2)


# ---- the core -----------------------------------------------------------------------------------------------------------

def _refuse_core(b: dict) -> str | None:
    tm, tr = b.get("TM", 1), b.get("TR", 1)
    if not (isinstance(tr, int) and isinstance(tm, int) and 1 <= tr <= tm):
        return f"a tile's real rows are 1 to its TM rows: TR = {tr!r}, TM = {tm!r}"
    return _refuse_k(b)


def _check(S) -> Any:
    d, ng = inert_bound._promoted(S.DEV, S.K)
    if S.N % S.TN or S.N < S.TN:
        raise DomainError(f"a row of y holds whole column tiles: TN = {S.TN} does not divide N = {S.N}")
    inert_bound._inert_bound_constants(S.K, S.TM, S.TN, S.N, S.RHO, S.DEV)
    inert_bound._row_constants(S.K, S.TN, S.N, S.RHOR, S.DEV)
    return d, ng


def _core_params(S) -> tuple:
    k, tm, tn = S.K, S.TM, S.TN
    _d, ng = _check(S)
    return (("a", Array(tm, Array(k, F32))), ("b", Array(tn, Array(k, F32))),
            ("cert", Array(tm, Array(tn, Array(ng - 1, Array(2, F32))))),
            ("salt", BYTES32), ("root", BYTES64), ("unit", U32), ("rows", Array(tm, U32)),
            ("eb", Array(tn, LINE)), ("fa", Array(k, LINE)), ("fb", Array(k, LINE)), ("g", pc8.GRAM))


def _core_ret(S) -> Tuple:
    return Tuple(BYTES64, Array(S.TR, Array(S.TN, BF16)), BIT)


def _core_body(B, S, a, b, cert, salt, root, unit, rows, eb, fa, fb, g):
    e, k, tm, tr, tn, dev = U._Emit(B), S.K, S.TM, S.TR, S.TN, S.DEV
    d = DEVICES[dev]
    at = dict(K=k, R=R, DOT=pc8.ATOMS[d.atom.name], G=d.g)
    admit = bind(PearlPlusAdmit, K=k)
    ra = [B.call(bind(PearlCRowA, **at), salt, a[i], root, unit, rows[i], fa, fb) for i in range(tm)]
    rb = [B.call(bind(PearlCRowB, **at), b[j], eb[j], fb, fa, g) for j in range(tn)]
    ok_a = [B.call(admit, a[i], ra[i][2], ra[i][3], ra[i][4]) for i in range(tr)]
    ok_b = [B.call(admit, b[j], rb[j][2], rb[j][3], rb[j][4]) for j in range(tn)]
    sd = [B.call(bind(PearlPlusSaltDead, K=k, DEV=dev), a[i], ra[i][4], ra[i][5]) for i in range(tm)]
    oka = [U.ok_word(e, o) for o in ok_a] + [e.c(L.OK.w, 0)] * (tm - tr)
    tile_inert_bound = bind(inert_bound.Pc8TileCapDefV2, K=k, TM=tm, TN=tn, N=S.N, RHO=S.RHO, RHOR=S.RHOR, DEV=dev)
    capped = B.call(tile_inert_bound, array_of([r[0] for r in ra]), array_of([r[0] for r in rb]), array_of([r[1] for r in ra]),
                    array_of([r[1] for r in rb]), array_of(oka), array_of(sd), cert)
    c, u, verdict = capped[0], capped[1], capped[2]
    digest = B.call(bind(PearlCRowDigest, K=2 * tm * tn), L.tile_messages(c, u, tm, tn))
    served = B.call(bind(PearlCUnscale, TM=tr, TN=tn, BIAS=False), u[:tr], array_of([r[4] for r in ra[:tr]]),
                    array_of([r[4] for r in rb]))
    finite = [e.call(F32IsFinite, e.call(Bf16ToF32, served[i][j])) for i in range(tr) for j in range(tn)]
    zero = [e.call(I32Eq, a[i][t], 0) for i in range(tr, tm) for t in range(k)]
    ok = e.call(Bf16GtStrict, verdict, 0)
    for bit in ok_a + ok_b + finite + zero:
        ok = e.call(BitAnd, ok, bit)
    return tuple_of(digest, served, ok)


_CORE = CompositeDefinition(
    "PearlPlusTileCore", 2, ("K", "TM", "TR", "TN", "N", "RHO", "RHOR", "DEV"),
    lambda S: (_core_params(S), _core_ret(S)),
    _core_body, refuse=_refuse_core, register=False,
    conformance="protocol-defined; equal to `pearl_plus_tile.reference` on its fixture tiles "
                "(tests/verity/protocols/accounting/compute/pouw/verifier/test_circuit_pearl_plus_tile.py)",
    doc="A drawn -h3 tile on words: its rows formed against the salt's noise (`PearlCRowA`, `PearlCRowB`), and (the "
        "sha512/row-seg/v1 digest of its C̃ and U over all TM rows, y: its first TR rows' (the real rows') served BF16 "
        "words, ok: those rows and B's admitted, the tile's and each real row's inert-work bound, each served word finite, "
        "and each later row (filler) zero word for word).")
PearlPlusTileCoreV3 = _on_bits(_CORE, 3)


# ---- the tile's noise ---------------------------------------------------------------------------------------------------

def _times(e: U._Emit, x, n: int):
    """``x`` · ``n`` (u32, wrapping) for a static n >= 1: `I32Add` of x's doublings at n's set bits."""
    total = None
    while n:
        if n & 1:
            total = x if total is None else e.call(I32Add, total, x)
        n >>= 1
        if n:
            x = e.call(I32Add, x, x)
    return total


def _refuse_slices(b: dict) -> str | None:
    k = b.get("K", 256)
    return None if isinstance(k, int) and k >= SLICE and k % SLICE == 0 else f"F is whole {SLICE}-code slices: K = {k!r}"


def _refuse_gram(b: dict) -> str | None:
    dev, p = b.get("DEV", "h100"), b.get("P", 0)
    return _refuse_slices(b) or (None if dev in DEVICES else f"no device record {dev!r}") or \
        (None if isinstance(p, int) and 0 <= p < R else f"P is F_B's column, 0 <= P < {R}: {p!r}")


_NOISE_CONF = ("protocol-defined; equal to `pearl_plus_credited`'s row references and the scheme's noise on their fixtures "
               "(tests/verity/protocols/accounting/compute/pouw/prover/test_pearl_plus_public_row.py)")


def _seed_body(B, S, salt, root, col):
    e = U._Emit(B)
    key = B.call(PearlCLineKey, B.call(PearlCSeedB, salt, root))
    base = _times(e, col, S.TN)
    eb = [base if j == 0 else e.call(I32Add, base, j) for j in range(S.TN)]
    return tuple_of(key, array_of(eb), array_of([e.c(32, t) for t in range(S.K)]))


_SEED = CompositeDefinition(
    "PearlPlusTileKey", 1, ("K", "TN"),
    lambda S: ((("salt", BYTES32), ("root", BYTES64), ("col", U32)), Tuple(BYTES32, Array(S.TN, U32), Array(S.K, U32))),
    _seed_body, refuse=_refuse_slices, register=False, conformance=_NOISE_CONF,
    doc="A -h3 tile's line key and its lines' indices on words: key = `PearlCLineKey_v1`(`PearlCSeedB_v1`(salt, root)), "
        "eb's col·TN + j for j < TN (u32, wrapping) and F's t for t < K")
_SEED_BITS = _on_bits(_SEED, 2, register=False)


def _words(t) -> Array:
    """The 16-bit words (`W16`) that hold the bits of bit type ``t``, low bit first: the width C commits a value in."""
    n = t.leaves
    if n % 16 or t.width_set() != {1}:
        raise DomainError(f"{t!r} is no whole 16-bit words of bits")
    return Array(n // 16, W16)


def _unpack(B, x, t):
    """Words ``x`` as the bit type ``t`` they hold, low bit first (`WordBits{16}`)."""
    return Coll(t, B.batch(bind(WordBits, W=16), Coll(Array(t.leaves // 16, W16), x.refs), axes=(0,)).refs)


_PACK = CompositeDefinition(
    "PearlPlusPackWords", 1, ("N",), lambda S: ((("bits", Array(S.N, Array(16, BIT))),), Array(S.N, W16)),
    lambda B, S, bits: B.batch(bind(BitsWord, W=16), bits, axes=(0,)), register=False,
    doc="N 16-bit words from their N·16 bits, low bit first: `BitsWord{16}` mapped over the parameter, 16 bits a word")


def _pack(B, x, t):
    """Bits ``x`` as the 16-bit words of type ``t``, low bit first (`BitsWord{16}`), by one Call of `_PACK` on them
    whole: Lean lowers a map whose argument steps through a parameter's leaves, and refuses one that steps through the
    copies of an earlier node other than by whole copies, so the words are packed in a Definition of their own."""
    n = x.type.leaves // 16
    return Coll(t, B.call(bind(_PACK, N=n), Coll(Array(n, Array(16, BIT)), x.refs)).refs)


def _seed_params(S) -> tuple:
    bits = dict(bind(_SEED_BITS, K=S.K, TN=S.TN).signature()[0])
    return ("salt", _words(bits["salt"])), ("root", _words(bits["root"])), ("col", U32)


#: F's K lines in this many groups of K/F_ROWS: the fields a reader of F_A or F_B reads, whatever K
F_ROWS = 8


def _seed_ret(S) -> Tuple:
    key, _eb, f = bind(_SEED_BITS, K=S.K, TN=S.TN).signature()[1].fields
    line = _words(bind(_LINE_BITS, SIDE=1, FACTOR=0).signature()[1])
    return Tuple(_words(key), *[Array(1, line)] * S.TN, *[Array(S.K // F_ROWS, _words(f.elem))] * F_ROWS)


def _seed_on_words(B, S, salt, root, col):
    sigma = bind(_SEED_BITS, K=S.K, TN=S.TN)
    bits = dict(sigma.signature()[0])
    cb = Coll(bits["col"], B.batch(bind(WordBits, W=32), Coll(Array(1, U32), col.refs), axes=(0,)).refs)
    out = B.call(sigma, _unpack(B, salt, bits["salt"]), _unpack(B, root, bits["root"]), cb)
    line = bind(_LINE_BITS, SIDE=1, FACTOR=0)
    (_k, kt), (_i, it) = line.signature()[0]
    eb = B.batch(line, Coll(kt, out[0].refs), Coll(Array(S.TN, it), out[1].refs), axes=(None, 0))
    held = concat([out[0].refs, eb.refs, out[2].refs])
    return _pack(B, Coll(Array(held.count, BIT), held), _seed_ret(S))


PearlPlusTileSeed = CompositeDefinition(
    "PearlPlusTileSeed", 4, ("K", "TN"), lambda S: (_seed_params(S), _seed_ret(S)),
    _seed_on_words, refuse=_refuse_slices, conformance=_NOISE_CONF,
    doc="`PearlPlusTileKey_v1` on 16-bit words with E_B's lines: the salt's and the root's words and col's bits "
        "(`WordBits`), its σ (`PearlPlusTileKey_v2`), `PearlPlusTileLine_v2`{SIDE=1, FACTOR=0} (σ of E_B's "
        "`PearlCNoiseLine_v1`) on the key and each of eb's TN indices, and the key, eb's TN lines and F's indices back "
        "in 16-bit words (`BitsWord{16}`), one field each for the key, each of eb's lines and each of F's `F_ROWS` "
        "groups of K/`F_ROWS` indices, so that each reader reads one field whole by wire, at the width C commits it in")


def _on_words16(bits: CompositeDefinition, version: int, doc: str) -> CompositeDefinition:
    """``bits`` (a σ whose every port and return is bits) with each port and its return in the 16-bit words that hold
    those bits, low bit first: each port's words its bits (`WordBits{16}`), then ``bits``, then its bits in words
    (`BitsWord{16}`)."""
    statics = tuple(bits.statics)

    def sigma(S) -> SpecializedDefinition:
        return bind(bits, **{k: getattr(S, k) for k in statics})

    def signature(S):
        params, ret = sigma(S).signature()
        return tuple((n, _word_type(t)) for n, t in params), _word_type(ret)

    def body(B, S, *xs):
        f = sigma(S)
        out = B.call(f, *(_unpack(B, x, t) for x, (_n, t) in zip(xs, f.signature()[0])))
        return _pack(B, out, _word_type(f.signature()[1]))

    return CompositeDefinition(bits.name, version, statics, signature, body, refuse=bits.refuse,
                               conformance=bits.conformance, doc=doc)


def _word_type(t) -> Any:
    """A bit type's 16-bit words, an Array of them its elements' words (`_words`)."""
    if isinstance(t, Array) and t.elem.leaves % 16 == 0 and t.elem.leaves > 16:
        return Array(t.n, _word_type(t.elem))
    return _words(t)


def _line_body(B, S, key, index):
    e = U._Emit(B)
    return B.call(PearlCNoiseLine, key, e.c(8, S.SIDE), e.c(8, S.FACTOR), index, e.c(16, NORM_F if S.FACTOR else NORM_E))


def _refuse_line(b: dict) -> str | None:
    side, factor = b.get("SIDE", 1), b.get("FACTOR", 1)
    return None if (side, factor) in ((1, 0), (0, 1), (1, 1)) else \
        f"a tile's lines are E_B's (SIDE 1, FACTOR 0), F_A's (0, 1) and F_B's (1, 1): {side, factor!r}"


_LINE = CompositeDefinition(
    "PearlPlusTileLine", 1, ("SIDE", "FACTOR"), lambda S: ((("key", BYTES32), ("index", U32)), LINE),
    _line_body, refuse=_refuse_line, register=False, conformance=_NOISE_CONF,
    doc="One of a -h3 tile's noise lines on words: `PearlCNoiseLine_v1`(key, SIDE, FACTOR, index, norm), norm bf16 8192 "
        "for E_B's (FACTOR 0) and bf16 512 for F's")
_LINE_BITS = _on_bits(_LINE, 2, register=False)


def _lines_signature(S):
    sigma = bind(_LINE_BITS, SIDE=S.SIDE, FACTOR=S.FACTOR)
    (_k, kt), (_i, it) = sigma.signature()[0]
    return (("key", _words(kt)), ("index", Array(S.G, _words(it)))), Array(S.G, _words(sigma.signature()[1]))


def _lines_body(B, S, key, index):
    sigma = bind(_LINE_BITS, SIDE=S.SIDE, FACTOR=S.FACTOR)
    (_k, kt), (_i, it) = sigma.signature()[0]
    lines = B.batch(sigma, _unpack(B, key, kt), _unpack(B, index, Array(S.G, it)), axes=(None, 0))
    return _pack(B, lines, Array(S.G, _words(sigma.signature()[1])))


def _refuse_lines(b: dict) -> str | None:
    g = b.get("G", 1)
    if (b.get("SIDE", 1), b.get("FACTOR", 1)) == LINES["eb"]:
        return "E_B's lines (SIDE 1, FACTOR 0) are the seed's (`PearlPlusTileSeed_v4`), no line unit's"
    return _refuse_line(b) or (None if isinstance(g, int) and g >= 1 else f"G is a count of lines, G >= 1: {g!r}")


PearlPlusTileLineV4 = CompositeDefinition(
    "PearlPlusTileLine", 4, ("SIDE", "FACTOR", "G"), _lines_signature, _lines_body, refuse=_refuse_lines,
    conformance=_NOISE_CONF,
    doc="G of a -h3 tile's F lines (F_A's, SIDE 0, or F_B's, SIDE 1; FACTOR 1) on 16-bit words: the key's words and each of the G indices' words their bits, "
        "`PearlPlusTileLine_v2` (σ of `PearlCNoiseLine_v1`(key, SIDE, FACTOR, index, norm)) on each index, and each "
        "line's 32 E4M3 codes back in 16 words, low bit first")


def _gram_row_body(B, S, fb, fa):
    d = DEVICES[S.DEV]
    chain = bind(PearlCChain, K=S.K, DOT=pc8.ATOMS[d.atom.name], G=d.g or 0)
    column = fb.take(1, S.P)
    return array_of([B.call(chain, column, fa.take(1, q)) for q in range(R)])


_GRAM_ROW = CompositeDefinition(
    "PearlPlusGramRow", 2, ("K", "DEV", "P"),
    lambda S: ((("fb", Array(S.K, LINE)), ("fa", Array(S.K, LINE))), Array(R, F32)),
    _gram_row_body, refuse=_refuse_gram, register=False, conformance=_NOISE_CONF,
    doc="Row P of a -h3 tile's Gram matrix on words, from F_B's lines whole (``fb``) and F_A's (``fa``): "
        "`PearlCChain_v1`{K, the device's atom, its window (0: none)} of F_B's column P and each of F_A's 32 columns, "
        "`PearlCGram_v1`'s row P")
_GRAM_ROW_BITS = _on_bits(_GRAM_ROW, 3, register=False)
PearlPlusGramRowV5 = _on_words16(
    _GRAM_ROW_BITS, 5, "Row P of a -h3 tile's Gram matrix on 16-bit words: F_B's and F_A's lines' words their bits, "
                       "`PearlPlusGramRow_v3` (σ of row P of `PearlCGram_v1`, F_B's column P against each of F_A's 32), "
                       "and the row's 32 FP32 words back in 16-bit words, two each, low half first")


def seed(k: int, tn: int) -> SpecializedDefinition:
    return bind(PearlPlusTileSeed, K=k, TN=tn)


#: a tile's three kinds of line, (SIDE, FACTOR): E_B's, F_A's, F_B's
LINES = {"eb": (1, 0), "fa": (0, 1), "fb": (1, 1)}


def line(kind: str, g: int) -> SpecializedDefinition:
    side, factor = LINES[kind]
    return bind(PearlPlusTileLineV4, SIDE=side, FACTOR=factor, G=g)


def gram_row(k: int, dev: str, p: int) -> SpecializedDefinition:
    return bind(PearlPlusGramRowV5, K=k, DEV=dev, P=p)


def noise(B, k: int, tn: int, dev: str, salt, root, col) -> tuple:
    """A tile's noise in the body under construction from C's words of its ``salt``, ``root`` and ``col``: (eb, fa, fb, g)
    in 16-bit words, `seed`'s Call (eb's TN lines among its fields), then `line` mapped over F's `F_ROWS` groups of
    indices K/`F_ROWS` lines a unit, and a `gram_row` Call for each of F_B's 32 columns, each reading F_B's and F_A's
    lines whole."""
    s = B.call(seed(k, tn), salt, root, col)
    key = s[0]

    def run(at: int, n: int):
        t = s[1 + at].type
        return Coll(Array(n, t), s.refs.slice(s.type.field_offset(1 + at), s.type.field_offset(1 + at) + n * t.leaves))

    def lines(kind: str, n: int, at: int, members: int):
        out = B.batch(line(kind, n // members), key, run(at, members), axes=(None, 0))
        return Coll(Array(n, out.type.elem.elem), out.refs)

    eb = Coll(Array(tn, s[1].type.elem), run(0, tn).refs)
    fa, fb = (lines(kind, k, tn, F_ROWS) for kind in ("fa", "fb"))
    return eb, fa, fb, array_of([B.call(gram_row(k, dev, p), fb, fa) for p in range(R)])


# ---- the unit -------------------------------------------------------------------------------------------------------

def core(scheme: C.PearlC, k: int, tm: int, tn: int, n: int, tr: int | None = None) -> SpecializedDefinition:
    """`PearlPlusTileCore_v3` for ``scheme`` (forming v1, -h3) at depth ``k``, a TM × TN tile of a linear of ``n`` columns
    whose first ``tr`` rows are real (the call's record's; every row when None): ρ the device's bound, ρr the scheme's row
    bound at k (`inert_bound.row_rho`)."""
    d = scheme.device
    if scheme.forming != "v1" or scheme.hashing != "h3" or DEVICES.get(d.name) != d:
        raise DomainError(f"{scheme.name}: Pearl+'s tile core is forming v1 under -h3 on a device record")
    return bind(PearlPlusTileCoreV3, K=k, TM=tm, TR=tm if tr is None else tr, TN=tn, N=n,
                RHO=inert_bound._rho(d.name, None), RHOR=inert_bound.row_rho(scheme, k), DEV=d.name)


#: the core's ports the verifier fills itself (its public row), in its port order
PUBLIC = ("salt", "root", "unit", "rows", "eb", "fa", "fb", "g")


# ---- the reference ------------------------------------------------------------------------------------------------------

def reference(dev: str, k: int, n: int, a: Sequence[Sequence[int]], b: Sequence[Sequence[int]],
              cert: Sequence[Sequence[Sequence[int]]], salt: bytes, root: bytes, unit: int, rows: Sequence[int],
              cols: Sequence[int], rho: int, rhor: int, tr: int | None = None
              ) -> tuple[bytes | None, list[list[int]] | None, int]:
    """The core on the scheme (`pearl_c`): (the tile's digest, its served y, ok) for A's rows ``a`` (tile rows ``rows`` of
    unit ``unit``, the first ``tr`` real, every row when None, the rest filler) and B's ``b`` (columns ``cols``, which give
    E_B's lines), the certificates, the salt and B's root.  y is each real row's TN served BF16 words (`pearl_c_u.y`).
    ok: the tile encodes (`PearlC.encode`), every real row and B row is admissible (`PearlC.admissible`'s per-row rule),
    every filler row's every word is 0, the real rows' certificates are the prover's (`inert_bound.certificates`), the
    debit of the real rows and each real row's is within ρ = 1/rho and ρr = 1/rhor of their cells' credit (rev1's
    inertBoundOK on `pearl_c_inert_work`'s replay), and every served word is finite.  The digest, of every row, is None
    where the tile doesn't encode; y is None there and where a served word is not finite."""
    d = DEVICES[dev]
    tm, tn = len(a), len(b)
    tr = tm if tr is None else tr
    scheme = C.PearlC(k_min=k, tile=(tm, tn), forming="v1", device=d, hashing="h3")
    ctx = C.Context(salt, C.Matmul(unit, C.Shape(max(rows) + 1, k, max(cols) + 1), 0), bytes(64), root)
    ops = scheme.encode(ctx, scheme.noise(ctx), [C.f32_row(x) for x in a], [C.f32_row(x) for x in b],
                        C.Tile(tuple(rows), tuple(cols)))
    if ops is None:
        return None, None, 0
    checked = scheme.checked(ctx, ops)
    filler = all(w == 0 for x in a[tr:] for w in x)
    admitted = all(P.f32_of_bits(r) * P.f32_of_bits(al) >= P.SIGMA_MIN and C.live_row(x)
                   for side, real in ((ops.a, tr), (ops.b, tn))
                   for r, al, x in zip(side.rho[:real], side.alpha[:real], side.x[:real]))
    certs = all([w for g in cert[i][j] for w in g] == inert_bound.certificates(ops.a.codes[i], ops.b.codes[j], dev)
                for i in range(tr) for j in range(tn))
    sd = [_salt_dead(d, ops.a.alpha[i], ops.a.beta[i], a[i]) for i in range(tr)]
    capped = (_cap_ok(d, k, n, rho, ops.a.codes[:tr], ops.b.codes, sd)
              and all(_cap_ok(d, k, n, rhor, [ops.a.codes[i]], ops.b.codes, [sd[i]]) for i in range(tr)))
    y = served(checked, ops, tr, tn)
    finite = not isinstance(y, str)
    return C.leaf(checked, "h3"), y if finite else None, int(filler and admitted and certs and capped and finite)


#: `served`'s refusal where a served word is not finite
NOT_FINITE = "tile-served-word-not-finite"


def served(checked: Any, ops: Any, tr: int, tn: int) -> list[list[int]] | str:
    """The tile's served y (`pearl_c_u.y`): each of its first ``tr`` rows' TN BF16 words; where a word is not finite
    (`pearl_c_u.NonFinite`: an FP32 product or the BF16 rounding overflows), the refusal naming it (`NOT_FINITE`)."""
    try:
        return [[CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j]) for j in range(tn)] for i in range(tr)]
    except (CU.NonFinite, P.Reject) as e:
        return f"{NOT_FINITE}: {type(e).__name__}: {e}"


def _salt_dead(d, alpha: int, beta: int, x: Sequence[int]) -> list[int]:
    """A row's salt-dead counts per block of 32: the device's salt-dead bit on f32(α·x_t) and e4m3(f32(β·44))."""
    bit = inert_bound.SALT_DEAD[d.atom.name]
    cb = P.f32_to_fp8(P.f32_of_bits(fp32.mul(beta, inert_bound.FLAT_E)))
    dead = [bit.evaluate(fp32.mul(alpha, xt), cb) for xt in x]
    return [sum(dead[s:s + SLICE]) for s in range(0, len(x), SLICE)]


def _cap_ok(d, k: int, n: int, rho: int, ac, bc, sd) -> bool:
    """rev1's inertBoundOK over every row of ``ac`` and column of ``bc`` at ρ = 1/rho on a unit of ``n`` columns
    (`pearlCTilesDevRev1`), False where the replay raises."""
    try:
        atoms, zeros = D.chain_inert_work(list(ac), list(bc), d)
        flags = {(i, j): D.word_flags(ac[i], bc[j], d) for i in range(len(ac)) for j in range(len(bc))}
    except P.Reject:
        return False
    elems = sum(SLICE if all(flags[i, j][t] for j in range(len(bc))) else sd[i][t]
                for i in range(len(ac)) for t in range(k // SLICE))
    debit = D.inert_work_units(atoms, Fraction(len(bc) * elems, n), device=d, zeros=zeros)
    at, m = replace(d, cap=Fraction(1, rho)), 64
    bound = at.cap * C.PearlC(forming="v1", device=at, hashing="h0").credit_of(Shape(m, k, n)) * len(ac) * len(bc) / (m * n)
    return debit <= bound

