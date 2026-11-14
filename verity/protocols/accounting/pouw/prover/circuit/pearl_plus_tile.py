"""Pearl+'s -h3 tile check as one unit whose one output ``ok`` the verifier holds to 1: a drawn tile's rows formed against
noise from the live salt, their admission, the tile's caps, its leaf under the transcript root and its served y, so that an
auditor reads only the verifier's own registered rows and the proof, never a row's values, the salt key or the tile.

    PearlPlusTile_v1{K,TM,TN,N,D,RHO,RHOR,DEV}         on bits
        (a: Array<TM, f32[K]>, b: Array<TN, f32[K]>, y: Array<TM, f32[N]>, cert, hsalt: u8[192], path: Array<D, u8[64]>,
         salt: u8[32], root: u8[64], unit: u32, rows: Array<TM, u32>, col: u32, eb: Array<TN, e4m3[32]>,
         fa, fb: Array<K, e4m3[32]>, g: Array<32, f32[32]>, troot, tdomain: u8[64], tpos: Array<D, bit>) -> ok: bit
        `PearlPlusTileCore_v2`'s ok and `PearlPlusTileLeaf_v1` of its digest, and-ed
    PearlPlusTileCore_v1{K,TM,TN,N,RHO,RHOR,DEV}       on words; _v2 its σ
        (a, b, y, cert, salt, root, unit, rows, col, eb, fa, fb, g) -> (digest: u8[64], ok: bit)
    PearlPlusTileLeaf_v1{D}                             on bits
        (x: u8[64], y: u8[192], root, domain: u8[64], pos: Array<D, bit>, path: Array<D, u8[64]>) -> bit
        1 iff the hm96-sha512 leaf of digest x under salt y opens at pos to root (`Hm96Sha512Leaf_v1`, `FrameV3Read_v1`)

The core, from the registry's call Definitions (`verity.definitions.registry.pearl_c_call`) and PoUW's cap (`cap`):

  - **rows.** Row i of A (``a[i]``, the tile's row ``rows[i]`` of unit ``unit``) by `PearlCRowA_v1`: its line E_A from the
    salt, the row's own digest and B's root (``root``), its codes A′_i, its peel factors P_A,i and its statistics; row j
    of B by `PearlCRowB_v1` against its line ``eb[j]``.  R = 32, the device's atom and promotion period.
  - **admission.** Each row's `PearlPlusAdmit_v1`, `Pc8RowA`'s ``ok`` on the statistics the row returned: every word
    finite, the noise floor ρ·α >= 1 and at most k/64 dead words (`pc8`'s module docstring, `PearlC.admissible`).
  - **caps.** A's salt-dead counts (`PearlPlusSaltDead_v1`, `Pc8RowACap`'s), then `Pc8TileCap_v2` on the codes, the
    factors, the admission words and the prover's certificates (``cert``): C̃ and U, and the verdict of the tile's cap at
    ρ = 1/RHO and each admitted row's at ρr = 1/RHOR (`cap.row_rho`: 4ρ at a census k, else ρ).
  - **leaf.** The tile's ``sha512/row-seg/v1`` digest of its messages (`leaves.tile_messages`, `PearlCRowDigest_v1`): the
    preimage of its leaf in the transcript tree.
  - **y.** Each row's TN served words from U (`PearlCUnscale_v1`, no bias), held to the row's committed y (``y[i]``, N FP32
    words, each a BF16 word widened: `served_y.f32_rows`) at column tile ``col``: words col·TN .. col·TN + TN − 1.
  - ``ok`` is every row's admission (A's and B's), the cap's verdict and each row's y, and-ed.

The unit then opens the digest's hm96 leaf, under the private salt ``hsalt``, at ``tpos`` in the transcript tree of root
``troot`` and domain id ``tdomain`` along the private ``path`` (`audit.transcript_domain`: digests, so x is the digest).

**No public input port.**  `flock-verify` refuses any statement with public input ports (`Flock/ProvedScope.lean`
`statementRules`), so the unit has none: every input is a registered read of the verifier's own ``--registered`` file at the
position the Program fixes (`Flock/Registered.lean`), as H1's tile unit is (`benchmarks/pouw/h1_prove`, the route the
one-tile measurement takes, ``flock-verify --zk --registered`` a statement).  `statement_rows` groups them:

  - **the formed rows** ``a``, ``b`` and ``y``, each a registered read opening the serve's own root at the position the
    Program fixes (`statement_rows`: one row a port).  Their roots are registered values reused from the serve's record
    (R7, `Spot/Registration.lean`), not inputs here; that the roots are the serve's and were committed before the draw coin
    is Pearl+'s own Lean protocol's, outside this measurement.
  - **``pub``, a supplied row of known content** (`PUBLIC`): the salt, B's root, the unit, the rows' indices, the column
    tile, the public noise (E_B's lines, F_A, F_B and G, which anyone derives from seed_B = s512("seed-B", salt ‖ root):
    `pearl_c`'s lines and Gram), and the transcript root, its domain id and the tile's position.  Its two shared roots, B's
    ``root`` and the transcript ``troot``, sit together in this one row, so the three sessions' shared root reads are in one
    place.
  - **private rows**: the certificates ``cert``, the hm96 salt ``hsalt`` and the path ``path``.
  - **``ok``**, the one output, registered at known content 1, so a proof the verifier accepts is one whose unit computed 1.

The core has a word form, so a fixture tile is evaluated on words against the scheme (`reference`); the unit is on bits
only, as H1's tile unit is (`h1tile`).  Every Definition the unit Calls is a σ, a Boolean gadget or a Call of them, so the
unit is laid out in units under 65,536 ANDs (`pearl_c_call.units`).
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any, Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.circuits.ops.fp32 import F32MulV2
from verity.circuits.ops.prims import Bf16ToF32, F32ToE4m3Sat
from verity.circuits.ops.scalar import Bf16GtStrict, BitAnd, BitOr, I32Eq, I32Le
from verity.circuits.refs import Coll, array_of, tuple_of
from verity.circuits.silicon import fp32
from verity.circuits.types import Array, Tuple, Value
from verity.commitments.gates import frame_v3 as F3
from verity.commitments.gates.hm96 import Hm96Sha512Leaf
from verity.commitments.gates.hm96 import reference as hm96_reference
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG
from verity.definitions.boolean import boolean_version, lift_body
from verity.definitions.registry.pearl_c_call import PearlCRowA, PearlCRowB
from verity.definitions.registry.pearl_c_forming import LINE
from verity.definitions.registry.pearl_c_noise import BYTES32, BYTES64, PearlCRowDigest
from verity.definitions.registry.pearl_c_tile import PearlCUnscale

from verity.protocols.accounting.pouw.verifier.protocol import DomainError, Shape
from ..schemes import pearl_c as C
from ..schemes import pearl_c_debit as D
from ..schemes import pearl_c_u as CU
from ..schemes import pearl_kw as P
from ..schemes.pearl_c_device import DEVICES
from . import boolean as _boolean
from . import cap, pc4, pc8
from . import leaves as L
from . import words as W

__all__ = ["PUBLIC", "PearlPlusAdmit", "PearlPlusAdmitV2", "PearlPlusSaltDead", "PearlPlusSaltDeadV2", "PearlPlusTile",
           "PearlPlusTileCore", "PearlPlusTileCoreV2", "PearlPlusTileLeaf", "leaf_reference", "reference", "statement_rows",
           "tile"]

F32, U32, BIT, I32 = Value(32), Value(32), Value(1), Value(32)
HM96_SALT = Array(192, Value(8))
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
    """σ of ``word`` (`verity.definitions.boolean.sigma_lift`'s, whose statics here are integers and names), each Call the
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
_on_bits(cap.Pc8CapWordDef, 2, register=False)
_on_bits(cap.Pc8TileCapDefV2, 3, register=False)


# ---- a row's admission and salt-dead counts -----------------------------------------------------------------------------

def _refuse_k(b: dict) -> str | None:
    k = b.get("K", 256)
    return None if isinstance(k, int) and k >= 64 and k % 64 == 0 else f"a row is whole 64-word blocks: K = {k!r}"


def _admit_body(B, S, x, s, rho, alpha):
    e, k = pc4._Emit(B), S.K
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
    e, k = pc4._Emit(B), S.K
    dead = cap.SALT_DEAD[DEVICES[S.DEV].atom.name]
    cb = e.call(F32ToE4m3Sat, e.call(F32MulV2, beta, cap.FLAT_E))
    bits = [e.call(dead, e.call(F32MulV2, alpha, x[t]), cb) for t in range(k)]
    return array_of([cap._wire(e, 32, cap._count(e, bits[s:s + SLICE])) for s in range(0, k, SLICE)])


def _salt_dead_sig(S):
    if DEVICES[S.DEV].atom.name not in cap.SALT_DEAD:
        raise DomainError(f"Pearl-C's cap circuit has no salt-dead bit for {S.DEV!r}'s atom")
    return (("x", Array(S.K, F32)), ("alpha", F32), ("beta", F32)), Array(S.K // SLICE, I32)


PearlPlusSaltDead = CompositeDefinition(
    "PearlPlusSaltDead", 1, ("K", "DEV"), _salt_dead_sig, _salt_dead_body, refuse=_refuse_k, conformance="protocol-defined",
    doc="A row of A's salt-dead counts per block of 32 elements, from its FP32 words and its α and β_row: `Pc8RowACap`'s, "
        "the elements whose support's two ends agree on f32(α·x) and e4m3(f32(β·44)) (`SaltDead`'s `EndsAgree`).")
PearlPlusSaltDeadV2 = _on_bits(PearlPlusSaltDead, 2)


# ---- the core -----------------------------------------------------------------------------------------------------------

def _check(S) -> Any:
    d, ng = cap._promoted(S.DEV, S.K)
    if S.N % S.TN or S.N < S.TN:
        raise DomainError(f"a row of y holds whole column tiles: TN = {S.TN} does not divide N = {S.N}")
    cap._cap_constants(S.K, S.TM, S.TN, S.N, S.RHO, S.DEV)
    cap._row_constants(S.K, S.TN, S.N, S.RHOR, S.DEV)
    return d, ng


def _core_params(S) -> tuple:
    k, tm, tn = S.K, S.TM, S.TN
    _d, ng = _check(S)
    return (("a", Array(tm, Array(k, F32))), ("b", Array(tn, Array(k, F32))), ("y", Array(tm, Array(S.N, F32))),
            ("cert", Array(tm, Array(tn, Array(ng - 1, Array(2, F32))))),
            ("salt", BYTES32), ("root", BYTES64), ("unit", U32), ("rows", Array(tm, U32)), ("col", U32),
            ("eb", Array(tn, LINE)), ("fa", Array(k, LINE)), ("fb", Array(k, LINE)), ("g", pc8.GRAM))


def _y_ok(e: pc4._Emit, yrow, served, col, tn: int, n: int):
    """1 iff the committed row ``yrow``'s words col·TN .. col·TN + TN − 1 are the served BF16 words widened, and col < N/TN."""
    widened = [e.call(Bf16ToF32, served[j]) for j in range(tn)]
    hit = 0
    for w in range(n // tn):
        eq = e.call(I32Eq, col, w)
        for j in range(tn):
            eq = e.call(BitAnd, eq, e.call(I32Eq, yrow[w * tn + j], widened[j]))
        hit = eq if isinstance(hit, int) else e.call(BitOr, hit, eq)
    return hit


def _core_body(B, S, a, b, y, cert, salt, root, unit, rows, col, eb, fa, fb, g):
    e, k, tm, tn, dev = pc4._Emit(B), S.K, S.TM, S.TN, S.DEV
    d = DEVICES[dev]
    at = dict(K=k, R=R, DOT=pc8.ATOMS[d.atom.name], G=d.g)
    admit = bind(PearlPlusAdmit, K=k)
    ra = [B.call(bind(PearlCRowA, **at), salt, a[i], root, unit, rows[i], fa, fb) for i in range(tm)]
    rb = [B.call(bind(PearlCRowB, **at), b[j], eb[j], fb, fa, g) for j in range(tn)]
    ok_a = [B.call(admit, a[i], ra[i][2], ra[i][3], ra[i][4]) for i in range(tm)]
    ok_b = [B.call(admit, b[j], rb[j][2], rb[j][3], rb[j][4]) for j in range(tn)]
    sd = [B.call(bind(PearlPlusSaltDead, K=k, DEV=dev), a[i], ra[i][4], ra[i][5]) for i in range(tm)]
    tile_cap = bind(cap.Pc8TileCapDefV2, K=k, TM=tm, TN=tn, N=S.N, RHO=S.RHO, RHOR=S.RHOR, DEV=dev)
    capped = B.call(tile_cap, array_of([r[0] for r in ra]), array_of([r[0] for r in rb]), array_of([r[1] for r in ra]),
                    array_of([r[1] for r in rb]), array_of([pc4.ok_word(e, o) for o in ok_a]), array_of(sd), cert)
    c, u, verdict = capped[0], capped[1], capped[2]
    digest = B.call(bind(PearlCRowDigest, K=2 * tm * tn), L.tile_messages(c, u, tm, tn))
    served = B.call(bind(PearlCUnscale, TM=tm, TN=tn, BIAS=False), u, array_of([r[4] for r in ra]),
                    array_of([r[4] for r in rb]))
    ok = e.call(Bf16GtStrict, verdict, 0)
    for bit in ok_a + ok_b + [_y_ok(e, y[i], served[i], col, tn, S.N) for i in range(tm)]:
        ok = e.call(BitAnd, ok, bit)
    return tuple_of(digest, ok)


PearlPlusTileCore = CompositeDefinition(
    "PearlPlusTileCore", 1, ("K", "TM", "TN", "N", "RHO", "RHOR", "DEV"), lambda S: (_core_params(S), Tuple(BYTES64, BIT)),
    _core_body, refuse=_refuse_k, conformance="protocol-defined; equal to `pearl_plus_tile.reference` on its fixture tiles "
                                              "(verity/protocols/accounting/pouw/verifier/tests/test_circuit_pearl_plus_tile.py)",
    doc="A drawn -h3 tile on words: its rows formed against the salt's noise (`PearlCRowA`, `PearlCRowB`), and (the "
        "sha512/row-seg/v1 digest of its C̃ and U, ok: every row admitted, the tile's and each row's cap, and each row's "
        "served y its committed y at column tile col).")
PearlPlusTileCoreV2 = _on_bits(PearlPlusTileCore, 2)


# ---- the leaf and the unit ----------------------------------------------------------------------------------------------

def _refs(c: Coll) -> list:
    return [c.refs[t] for t in range(c.refs.count)]


def _words(c: Coll) -> Coll:
    """A byte string's bits (bytes of 8 bits, LSB first) as hm96's and frame-v3's 64-bit words: LSB first, bytes big-endian."""
    data = _refs(c)
    out = [data[8 * (8 * w + 7 - b) + j] for w in range(len(data) // 64) for b in range(8) for j in range(8)]
    return Coll(Array(len(data) // 64, F3.WORD), TR._refs(out))


def _leaf_sig(S):
    F3._check_depth(S.D)
    byte = BL.bit_type(BYTES64)
    return ((("x", byte), ("y", BL.bit_type(HM96_SALT)), ("root", byte), ("domain", byte), ("pos", Array(S.D, BIT)),
             ("path", Array(S.D, byte))), BIT)


def _leaf_body(B, S, x, y, root, domain, pos, path):
    leaf = B.call(Hm96Sha512Leaf, _words(x), _words(y))[2]
    sibs = Coll(Array(S.D, F3.DIGEST), TR._refs([r for k in range(S.D) for r in _refs(_words(path[k]))]))
    return B.call(F3.read(S.D, SCHEMA_HM96_SHA512_ROW_SEG), _words(root), _words(domain), leaf, pos, sibs)


PearlPlusTileLeaf = CompositeDefinition(
    "PearlPlusTileLeaf", 1, ("D",), _leaf_sig, _leaf_body, conformance="exact-model-tested",
    doc="1 iff the hm96-sha512 leaf of the 64-byte digest x under the 192-byte salt y, framed under hm96-sha512/row-seg/v1 "
        "at rank = position = pos, opens along path to root in the depth-D frame-v3-sha512 tree of domain id domain "
        "(`audit.Hm96Domain` with digests)")


def _tile_sig(S):
    core, _ret = bind(PearlPlusTileCoreV2, **_core_statics(S)).signature()
    leaf = dict(_leaf_sig(S)[0])
    params = core[:4] + (("hsalt", leaf["y"]), ("path", leaf["path"])) + core[4:] + (
        ("troot", leaf["root"]), ("tdomain", leaf["domain"]), ("tpos", leaf["pos"]))
    return params, BIT


def _core_statics(S) -> dict:
    return {k: getattr(S, k) for k in PearlPlusTileCore.statics}


def _tile_body(B, S, a, b, y, cert, hsalt, path, salt, root, unit, rows, col, eb, fa, fb, g, troot, tdomain, tpos):
    out = B.call(bind(PearlPlusTileCoreV2, **_core_statics(S)), a, b, y, cert, salt, root, unit, rows, col, eb, fa, fb, g)
    opened = B.call(bind(PearlPlusTileLeaf, D=S.D), out[0], hsalt, troot, tdomain, tpos, path)
    return B.call(boolean_version(BitAnd), out[1], opened)


def _refuse_tile(b: dict) -> str | None:
    return _refuse_k(b) or (None if 1 <= b.get("D", 1) <= F3.MAX_D else f"a tree's depth is 1 to {F3.MAX_D}: D = {b['D']!r}")


PearlPlusTile = CompositeDefinition(
    "PearlPlusTile", 1, ("K", "TM", "TN", "N", "D", "RHO", "RHOR", "DEV"), _tile_sig, _tile_body, refuse=_refuse_tile,
    conformance="protocol-defined; `pearl_plus_tile.reference` and `leaf_reference` on its fixture tiles "
                "(verity/protocols/accounting/pouw/verifier/tests/test_circuit_pearl_plus_tile.py)",
    doc="Pearl+'s -h3 tile unit: `PearlPlusTileCore_v2`'s ok, and its digest's hm96 leaf under hsalt opening at tpos to the "
        "transcript root troot (`PearlPlusTileLeaf_v1`); the verifier holds ok to 1.")


def tile(scheme: C.PearlC, k: int, tm: int, tn: int, n: int, depth: int) -> SpecializedDefinition:
    """`PearlPlusTile_v1` for ``scheme`` (forming v1, -h3) at depth ``k``, a TM × TN tile of a unit of ``n`` columns and a
    transcript tree of depth ``depth``: ρ the device's cap, ρr the scheme's row cap at k (`cap.row_rho`)."""
    d = scheme.device
    if scheme.forming != "v1" or scheme.hashing != "h3" or DEVICES.get(d.name) != d:
        raise DomainError(f"{scheme.name}: Pearl+'s tile unit is forming v1 under -h3 on a device record")
    return bind(PearlPlusTile, K=k, TM=tm, TN=tn, N=n, D=depth, RHO=cap._rho(d.name, None),
                RHOR=cap.row_rho(scheme, k), DEV=d.name)


# ---- the statement's ports ----------------------------------------------------------------------------------------------

#: the unit's supplied row ``pub``, a registered read of known content, in this order; its two shared roots (B's ``root``
#: and the transcript ``troot``) sit together here, so the tile's three sessions' shared root reads are in one place
PUBLIC = ("salt", "root", "unit", "rows", "col", "eb", "fa", "fb", "g", "troot", "tdomain", "tpos")


def statement_rows(d: SpecializedDefinition) -> list[tuple[str, list[str]]]:
    """The unit's input rows (``d`` `PearlPlusTile_v1`), each ``(port, params)``, all registered reads (no public input
    port): A's rows ``a0`` .. by tile row, B's ``b0`` .., y's ``y0`` .., the certificates ``cert``, the tile leaf's opening
    ``tile`` (its hm96 salt and path), and ``pub`` (`PUBLIC`).  Each row of A, B and y opens the serve's own root at the
    position the Program fixes; ``pub`` is a supplied row of known content."""
    if d.defn is not PearlPlusTile:
        raise ValueError(f"Pearl+'s tile statement is PearlPlusTile_v1, not {d.defn.id}")
    ty = dict(d.signature()[0])
    rows = [(f"{n}{i}", [f"{n}[{i}]"]) for n in ("a", "b", "y") for i in range(ty[n].n)]
    rows += [("cert", ["cert"]), ("tile", ["hsalt", "path"]), ("pub", list(PUBLIC))]
    if {p.partition("[")[0] for _port, ps in rows for p in ps} != set(ty):
        raise AssertionError(f"{d.id}: the ports {rows} are not its parameters {list(ty)}")
    return rows


# ---- the reference ------------------------------------------------------------------------------------------------------

def reference(dev: str, k: int, n: int, a: Sequence[Sequence[int]], b: Sequence[Sequence[int]],
              y: Sequence[Sequence[int]], cert: Sequence[Sequence[Sequence[int]]], salt: bytes, root: bytes, unit: int,
              rows: Sequence[int], cols: Sequence[int], col: int, rho: int, rhor: int) -> tuple[bytes | None, int]:
    """The core on the scheme (`pearl_c`): (the tile's digest, ok) for A's rows ``a`` (tile rows ``rows`` of unit
    ``unit``) and B's ``b`` (columns ``cols``, which give E_B's lines), the rows of y, the certificates, the salt and B's
    root.  ok: the tile encodes (`PearlC.encode`), every row is admissible (`PearlC.admissible`'s per-row rule), the
    certificates are the prover's (`cap.certificates`), the tile's and each row's debit within ρ = 1/rho and ρr = 1/rhor of
    their cells' credit (rev1's capOK on `pearl_c_debit`'s replay), and each row's committed y at column tile ``col`` is
    the served y (`pearl_c_u.y`).  The digest is None where the tile doesn't encode."""
    d = DEVICES[dev]
    tm, tn = len(a), len(b)
    scheme = C.PearlC(k_min=k, tile=(tm, tn), forming="v1", device=d, hashing="h3")
    ctx = C.Context(salt, C.Matmul(unit, C.Shape(max(rows) + 1, k, max(cols) + 1), 0), bytes(64), root)
    ops = scheme.encode(ctx, scheme.noise(ctx), [C.f32_row(x) for x in a], [C.f32_row(x) for x in b],
                        C.Tile(tuple(rows), tuple(cols)))
    if ops is None:
        return None, 0
    checked = scheme.checked(ctx, ops)
    admitted = all(P.f32_of_bits(r) * P.f32_of_bits(al) >= P.SIGMA_MIN and C.live_row(x)
                   for side in (ops.a, ops.b) for r, al, x in zip(side.rho, side.alpha, side.x))
    certs = all([w for g in cert[i][j] for w in g] == cap.certificates(ops.a.codes[i], ops.b.codes[j], dev)
                for i in range(tm) for j in range(tn))
    sd = [_salt_dead(d, ops.a.alpha[i], ops.a.beta[i], a[i]) for i in range(tm)]
    capped = (_cap_ok(d, k, n, rho, ops.a.codes, ops.b.codes, sd)
              and all(_cap_ok(d, k, n, rhor, [ops.a.codes[i]], ops.b.codes, [sd[i]]) for i in range(tm)))
    try:
        served = 0 <= col < n // tn and all(y[i][col * tn + j] == CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j]) << 16
                                            for i in range(tm) for j in range(tn))
    except P.Reject:
        served = False
    return C.leaf(checked, "h3"), int(admitted and certs and capped and served)


def _salt_dead(d, alpha: int, beta: int, x: Sequence[int]) -> list[int]:
    """A row's salt-dead counts per block of 32: the device's salt-dead bit on f32(α·x_t) and e4m3(f32(β·44))."""
    bit = cap.SALT_DEAD[d.atom.name]
    cb = P.f32_to_fp8(P.f32_of_bits(fp32.mul(beta, cap.FLAT_E)))
    dead = [bit.evaluate(fp32.mul(alpha, xt), cb) for xt in x]
    return [sum(dead[s:s + SLICE]) for s in range(0, len(x), SLICE)]


def _cap_ok(d, k: int, n: int, rho: int, ac, bc, sd) -> bool:
    """rev1's capOK over every row of ``ac`` and column of ``bc`` at ρ = 1/rho on a unit of ``n`` columns
    (`pearlCTilesDevRev1`), False where the replay raises."""
    try:
        atoms, zeros = D.chain_debit(list(ac), list(bc), d)
        flags = {(i, j): D.word_flags(ac[i], bc[j], d) for i in range(len(ac)) for j in range(len(bc))}
    except P.Reject:
        return False
    elems = sum(SLICE if all(flags[i, j][t] for j in range(len(bc))) else sd[i][t]
                for i in range(len(ac)) for t in range(k // SLICE))
    debit = D.debit_units(atoms, Fraction(len(bc) * elems, n), device=d, zeros=zeros)
    at, m = replace(d, cap=Fraction(1, rho)), 64
    bound = at.cap * C.PearlC(forming="v1", device=at, hashing="h0").credit_of(Shape(m, k, n)) * len(ac) * len(bc) / (m * n)
    return debit <= bound


def leaf_reference(digest: bytes, hsalt: bytes, root: bytes, domain: bytes, pos: int, path: Sequence[bytes]) -> int:
    """`PearlPlusTileLeaf_v1` on bytes: 1 iff the digest's hm96 leaf under ``hsalt`` opens at ``pos`` to ``root``."""
    return F3.opens(SCHEMA_HM96_SHA512_ROW_SEG, root, domain, hm96_reference(digest, hsalt)[2], pos, path)
