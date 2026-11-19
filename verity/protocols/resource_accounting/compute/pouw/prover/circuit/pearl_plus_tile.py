"""Pearl+'s -h3 tile as one node that writes its tile of the linear's y: a tile's rows formed against noise from the live
salt, their admission, the tile's inert-work bound and its digest, and the served y the core computes from its U, so that a
reader of the linear takes y by wire from the node that wrote it and the verifier holds the node's ``ok`` to 1.

    PearlPlusTileCore_v3{K,TM,TR,TN,N,RHO,RHOR,DEV}    on bits: the σ of the core's word body
        (a: Array<TM, f32[K]>, b: Array<TN, f32[K]>, cert, salt: u8[32], root: u8[64], unit: u32, rows: Array<TM, u32>,
         eb: Array<TN, e4m3[32]>, fa, fb: Array<K, e4m3[32]>, g: Array<32, f32[32]>)
        -> (digest: u8[64], y: Array<TR, bf16[TN]>, ok: bit)
    PearlPlusTileNoise_v2{K,TN,DEV}                     on bits: the σ of the noise's word body
        (salt: u8[32], root: u8[64], col: u32) -> (eb: Array<TN, e4m3[32]>, fa, fb: Array<K, e4m3[32]>, g: Array<32, f32[32]>)
        the core's eb, fa, fb and g for column tile col, from seed_B = H_"seed-B"(salt ‖ root) (`PearlCSeedB_v1`,
        `PearlCLineKey_v1`, `PearlCNoiseLine_v1`, `PearlCGram_v1`)

The core, from the registry's call Definitions (`verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_call`) and PoUW's inert-work bound (`inert_bound`):

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

Where the tile sits in the linear (its column tile) is C's wiring, so the core reads no column index: E_B's lines ``eb``
are the column tile's, in the public row the verifier fills itself (`pearl_plus_credited.public`), with the salt, B's
root and the rest of the noise, so the salt and the root reach the core only through that row.

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
from verity.circuits.refs import Coll, array_of, tuple_of
from verity.circuits.silicon import fp32
from verity.circuits.types import Array, Tuple, Value
from verity.circuits.ops.boolean.versions import lift_body
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_call import PearlCRowA, PearlCRowB
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_forming import LINE, PearlCGram
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_noise import (
    BYTES32, BYTES64, NORM_E, NORM_F, PearlCLineKey, PearlCNoiseLine, PearlCRowDigest, PearlCSeedB)
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_tile import PearlCUnscale

from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import DomainError, Shape
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

__all__ = ["PUBLIC", "PearlPlusAdmit", "PearlPlusAdmitV2", "PearlPlusSaltDead", "PearlPlusSaltDeadV2", "PearlPlusTileCoreV3",
           "PearlPlusTileNoiseV2", "core", "reference", "served"]

F32, U32, BIT, I32, BF16 = Value(32), Value(32), Value(1), Value(32), Value(16)
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
                "(tests/verity/protocols/resource_accounting/compute/pouw/verifier/test_circuit_pearl_plus_tile.py)",
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


def _refuse_noise(b: dict) -> str | None:
    k, dev = b.get("K", 256), b.get("DEV", "h100")
    if not (isinstance(k, int) and k >= SLICE and k % SLICE == 0):
        return f"the Gram is over whole {SLICE}-code slices: K = {k!r}"
    return None if dev in DEVICES else f"no device record {dev!r}"


def _noise_ret(S) -> Tuple:
    return Tuple(Array(S.TN, LINE), Array(S.K, LINE), Array(S.K, LINE), pc8.GRAM)


def _noise_body(B, S, salt, root, col):
    e, k, tn, d = U._Emit(B), S.K, S.TN, DEVICES[S.DEV]
    key = B.call(PearlCLineKey, B.call(PearlCSeedB, salt, root))

    def line(side: int, factor: int, t, norm: int):
        return B.call(PearlCNoiseLine, key, e.c(8, side), e.c(8, factor), e.c(32, t) if isinstance(t, int) else t,
                      e.c(16, norm))

    base = _times(e, col, tn)
    eb = [line(1, 0, base if j == 0 else e.call(I32Add, base, j), NORM_E) for j in range(tn)]
    fa = [line(0, 1, t, NORM_F) for t in range(k)]
    fb = [line(1, 1, t, NORM_F) for t in range(k)]
    g = B.call(bind(PearlCGram, K=k, R=R, DOT=pc8.ATOMS[d.atom.name], G=d.g or 0), array_of(fb), array_of(fa))
    return tuple_of(array_of(eb), array_of(fa), array_of(fb), g)


_NOISE = CompositeDefinition(
    "PearlPlusTileNoise", 1, ("K", "TN", "DEV"),
    lambda S: ((("salt", BYTES32), ("root", BYTES64), ("col", U32)), _noise_ret(S)),
    _noise_body, refuse=_refuse_noise, register=False,
    conformance="protocol-defined; equal to `pearl_plus_credited`'s row references and the scheme's noise on their "
                "fixtures (tests/verity/protocols/resource_accounting/compute/pouw/prover/test_pearl_plus_credited.py)",
    doc="A -h3 tile's noise on words from the live salt and the weights' root: key = `PearlCLineKey_v1` of seed_B = "
        "`PearlCSeedB_v1`(salt, root); eb, E_B's lines of the column tile col, `PearlCNoiseLine_v1`(key, 1, 0, col·TN + j, "
        "bf16 8192) for j < TN; fa and fb, F_A's and F_B's K lines, (key, 0, 1, t, bf16 512) and (key, 1, 1, t, bf16 512) "
        "for t < K; g = `PearlCGram_v1`{K, 32, the device's atom, its window (0: none)}(fb, fa)")
PearlPlusTileNoiseV2 = _on_bits(_NOISE, 2)


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
    return C.leaf(checked, "h3"), y, int(filler and admitted and certs and capped and y is not None)


def served(checked: Any, ops: Any, tr: int, tn: int) -> list[list[int]] | None:
    """The tile's served y (`pearl_c_u.y`): each of its first ``tr`` rows' TN BF16 words; None where a word is not finite
    (`pearl_c_u.NonFinite`: an FP32 product or the BF16 rounding overflows)."""
    try:
        return [[CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j]) for j in range(tn)] for i in range(tr)]
    except (CU.NonFinite, P.Reject):
        return None


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

