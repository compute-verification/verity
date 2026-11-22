"""A credited linear of C as Pearl+'s product: its grid's units of `PearlPlusTileCoreWords_v5`, the tile on C's words with
its noise derived inside, which are the linear's nodes in C and write its y, and each unit's public row.

    PearlPlusTileCoreWords_v5{K,TM,TR,TN,N,RHO,RHOR,DEV}   on words
        (a: Array<TM, f32[K]>, b: Array<TN, bf16[K]>, cert, salt: u16[16], root: u16[32], unit: u32, rows: Array<TM, u32>,
         col: u32)
        -> (digest: u8[64] as bits, y: Array<TR, bf16[TN]>, ok: u16)
        maps of the noise (`pearl_plus_tile.noise`: eb, fa, fb, g of salt, root and col) and the wired core
    PearlPlusTileCoreWired_v2{K,TM,TR,TN,N,RHO,RHOR,DEV}  on words
        (a, b, cert, salt, root, unit, rows as above, eb: Array<TN, u16[16]>, fa, fb: Array<K, u16[16]>,
         g: Array<32, Array<32, u16[2]>>) -> as above
        `PearlPlusTileCore_v3` behind `WordBits` batches, b's words widened to FP32 first (`Bf16ToF32`), so each port is a
        row of the words C holds, the salt's and the root's bytes and the noise's bits in 16-bit words, low bit first;
        y's bits back to BF16 words (`BitsWord{16}`), ok one word, bit 0 the core's ok and the rest zero
    PearlPlusTileCoreWords_v6 / PearlPlusTileCoreWired_v3 as _v5 / _v2, with a: Array<TM, bf16[K]> widened inside
        (`Bf16ToF32`) as b is: the tile in the place of a served linear's unit, x read by wire from its writer's BF16
        row (`served_product`)
    PearlPlusTileCoreWords_v7 / PearlPlusTileCoreWired_v4 as _v6 / _v3, the same function, the core as TN + 1 units on C's
        words, each a proof block within the composer's limit at K = 256 (one, about 46M ANDs, is not):
        PearlPlusTileRowBWords_v1{HEAD,K,DEV,TN}: one B row's `PearlCRowB` and admission (about 9.5M ANDs), HEAD 1 on
        the TN rows whole, the one registered weight row a block reads, passing rows 1 .. TN − 1 on;
        PearlPlusTileCoreRestWords_v1{as the core}: A's rows, the bound, the digest, y and ok (about 7.8M ANDs)
    cut(t)                                -> [(Definition, n)] the units C's laying makes of node t (`CUTS`)
    tiles(scheme, k, m, n, tile)          -> [Tile]          the linear's TM × TN grid, row-major: one unit of C a tile
    public(scheme, k, t, unit, salt, root) -> {port: value}  tile t's public row (`PUBLIC`), in the node's port order
    row_eb(salt, root_b, col, tn) / row_fa(salt, root_b, k) / row_fb(salt, root_b, k) / row_gram(salt, root_b, k, dev)
                                          -> bytes           the bytes of one noise value the node derives inside, on bytes:
                                                             the references its tests hold it to

At a credited linear of C, y = x · Wᵀ over M rows of x (K words) and the weight's N columns, each grid tile is one unit
`PearlPlusTileCoreWords_v5{K, TM, TR, TN, N, RHO, RHOR, DEV}` (`credited_product`), TR its rows below M, the rest filler
(rows a call's A is padded with to whole tiles, held to zero by the core). Its ports read:

  - ``a``: the tile's TM rows of A, the linear's x rows widened to FP32 as their writers hold them (a filler row zero);
  - ``b``: its TN rows of B, the weight's columns ``cols``, the registered read of the weights' record (BF16);
  - ``cert``: the prover's inert-work certificates, a committed value;
  - ``salt``, ``root``, ``unit``, ``rows``, ``col`` (`PUBLIC`): its public row, which the verifier fills itself: the
    epoch's live salt, the weights' registered root (B's root, which seeds the noise), the call's index and the tile's
    place in C (its rows and its column tile).

The node derives its noise from seed_B = H_"seed-B"(salt ‖ root) inside: E_B's lines are its column tile's, F_A, F_B and G
the call's; the core derives each row's E_A from the row's own digest.  Its top level is the noise's units and the core,
each a Call or a map member: `PearlPlusTileSeed_v4` (the line key, eb's TN lines and each of F's lines' index),
`PearlPlusTileLine_v4` mapped over F's K for fa and for fb, `PearlPlusGramRow_v5` Called for each of F_B's 32 columns, and
`PearlPlusTileCoreWired_v2`, eb, fa, fb and g crossing as values C wires from each producer's output row to its consumer's
input row, each in the 16-bit words C commits its values in, which every reader reads at that width.  No K that the core takes fits the node whole in one proof block (the noise alone is about 370k · K ANDs), so
C's laying (`verity.circuits.laying.lay`) opens it into those units (`cut`), each its own block: a line is 73,459 ANDs and
a Gram row 32 chains (about 7,000 · K), in range at every K up to the served 2048; the wired core is in range whole at
K = 256 (about 16.6M ANDs), and at K = 2048 (about 5.1G ANDs at a 16 × 64 tile) laying opens it too.  It writes ``y``,
its TR real rows' TN words of the linear's y (Pearl+'s served product, `pearl_plus_tile.served`): the one y C holds,
which every reader of the linear takes by wire, a row of y the concatenation of its column tiles' words in column order.
Its ``ok`` is one u16 word held to 1; its digest is a committed value no unit reads. A unit's amount is its credited
MACs, TR · TN · K (`scheme.tile_macs` over its statics, `PEARL_PLUS.blocks`), credited to the tile as a whole over every
proof block laying makes of it (`scheme.laid_tiles`; the credit is the Lean's `creditAtAssumption`). The core and the noise's lines
and Gram rows are registered on bits only (their word bodies are unregistered), and C's values are words, so this is the
tile's one word form C names.

The noise the node derives is, on bytes, the references below (`row_eb`, `row_fa`, `row_fb`, `row_gram`), each value's
bytes in the order and ``le`` layout its core port reads; no verifier reads them. Paths are under ``pouw/prover/schemes/``
unless named.

  - H_ℓ(m) = SHA-512(L_ℓ ‖ m)[0:32], L_ℓ = ASCII ``pearl-c/h3s/<ℓ>`` then one 0x00, zero-padded to 128 bytes
    (`pearl_c.s512`, pearl_c.py:148; `S512_LABELS` :145).
  - seed_B = H_"seed-B"(salt ‖ root_B): the epoch's 32-byte salt and the weights record's 64-byte registered root
    (`pearl_c.seed_b`, pearl_c.py:163). key = H_"noise-line"(seed_B) (`_s512_line_key`, pearl_c.py:208).
  - line(side, factor, t, norm), 32 E4M3 bytes (`pearl_c.sample_line`, pearl_c.py:196): raw = H_"line"(key ‖ side byte ‖
    factor byte ‖ t as u32 LE); x_i = s·((raw_i & 0x7F) + 1), s = −1 if raw_i ≥ 0x80 else +1; scale =
    bf16_div(bf16(32·norm), bf16(isqrt(1024·Σx_i²))); byte_i = e4m3(bf16_to_f32(bf16_mul(bf16(x_i), scale))). bf16 is FP32's
    nearest-even rounding to BF16, refusing non-finite and overflow (`pearl_kw.f32_to_bf16`, pearl_kw.py:136); bf16_mul and
    bf16_div take the FP32 product or quotient of the two BF16 values, rounded to FP32 nearest-even, then bf16 (:198, :202);
    e4m3 is nearest-even saturating to ±448, 0x7E (`pearl_kw.f32_to_fp8`, :175). bf16(32·256) = 0x4600, bf16(32·16) = 0x4400.
  - eb {col, tn}: line(1, 0, j, 256) for j = col·tn … col·tn + tn − 1 in order, 32·tn bytes (norm 256 is
    `pearl_kw.NOISE_NORM`, pearl_kw.py:69).
  - fa {k}: line(0, 1, t, 16) for t = 0 … k − 1, 32·k bytes; fb {k}: line(1, 1, t, 16) (16 is `pearl_c.LINE_NORM_V1`,
    pearl_c.py:131; `pearl_c._basis`, :295). F_A[t][p] is byte p of fa's line t.
  - gram {k, dev}: G[p][q] = chain_dev(F_B[·][p], F_A[·][q]) for p, q = 0 … 31, p-major, each FP32 word 4 bytes LE, 4,096
    bytes (`pearl_c.gram`, pearl_c.py:305), k a multiple of 32.
  - chain_dev(a, b) over k E4M3 codes (`pearl_c.chain`, pearl_c.py:218), with dev = `pearl_c_device.DEVICES[dev]`
    (pearl_c_device.py:120), atom its E4M3 k32 step on FP32 words (`GroupSum.step`, circuits/silicon/models.py:215) and G
    its window in atoms: total = +0; per window of G·32 products, acc = +0, then acc = atom(acc, a[s:s+32], b[s:s+32]) for
    each 32-slice s in it, then total = FP32 nearest-even total + acc (`circuits.silicon.fp32.add`), and a total of −0
    (0x80000000) becomes +0. With G None: acc = +0 over every slice in order, and acc is the word. A non-finite word
    (exponent 0xFF) after any step or add refuses.
  - Devices: ``h100``, atom `HOPPER_E4M3_K32` (models.py:299: groups (32,), adder width 14, zero exponent −139), G = 4;
    ``sm120``, atom `BLACKWELL_SM120_E4M3_M16N8K32` (models.py:389: (32,), 26, −133), G = 4; ``sm120-unpromoted``, that
    atom, G None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.ops.prims import BF16, Bf16ToF32
from verity.circuits.population import BitsWord, WordBits
from verity.circuits.refs import Coll, concat, tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.protocols.accounting.compute.pouw.prover import schemes as PS
from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT
from verity.protocols.accounting.compute.pouw.prover.circuit import tile_units as U
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.accounting.compute.pouw.prover.schemes.pearl_c_device import DEVICES

U32 = Value(32)

#: the node's ports the verifier fills itself (its public row), in its port order
PUBLIC = ("salt", "root", "unit", "rows", "col")
_CORE_OWN = ("a", "b", "cert", "salt", "root", "unit", "rows")


_NOISE_PORTS = ("eb", "fa", "fb", "g")


def _core(S) -> SpecializedDefinition:
    return bind(PT.PearlPlusTileCoreV3, **{s: getattr(S, s) for s in _STATICS})


def _wired(S) -> SpecializedDefinition:
    return bind(PearlPlusTileCoreWired, **{s: getattr(S, s) for s in _STATICS})


def _bit_params(S) -> dict:
    return dict(_core(S).signature()[0])


#: the core's ports C holds in 16-bit words, its bits low bit first: the salt and the root (bytes) and the noise (bits)
_IN_WORDS16 = ("salt", "root") + _NOISE_PORTS


#: the core's ports C holds in BF16 words, which the wired core widens to FP32 first (`Bf16ToF32`): B's rows in
#: `PearlPlusTileCoreWired_v2`, and A's too in `_v3`, whose x is a served linear's input as its writer holds it
_WIDENED = {2: ("b",), 3: ("a", "b")}


def _word_params(S, widened: tuple[str, ...] = _WIDENED[2]) -> tuple:
    params = dict(bind(PT.PearlPlusTileCoreV3.sigma_of, **{s: getattr(S, s) for s in _STATICS}).signature()[0])
    bits = _bit_params(S)
    rows = {"a": S.TM, "b": S.TN}
    return tuple((p, Array(rows[p], Array(S.K, BF16)) if p in widened else PT._word_type(bits[p]) if p in _IN_WORDS16
                  else params[p]) for p in _CORE_OWN)


def _noise_params(S) -> tuple:
    return tuple((p, PT._word_type(_bit_params(S)[p])) for p in _NOISE_PORTS)


def _word_ret(S):
    digest, _y, _ok = _core(S).signature()[1].fields
    return Tuple(digest, Array(S.TR, Array(S.TN, BF16)), BF16)


def _wired_body(B, S, *xs, widened: tuple[str, ...] = _WIDENED[2], y_calls: bool = False):
    bits, bt = {}, _bit_params(S)
    for (p, t), x in zip(_word_params(S, widened) + _noise_params(S), xs):
        if p in widened:
            x = B.batch(Bf16ToF32, Coll(Array(t.leaves, BF16), x.refs), axes=(0,))
        w = 32 if p in widened else t.leaf_widths()[0]
        bits[p] = Coll(bt[p], B.batch(bind(WordBits, W=w), Coll(Array(t.leaves, Value(w)), x.refs), axes=(0,)).refs)
    out = B.call(_core(S), *(bits[p] for p in _CORE_OWN + _NOISE_PORTS))
    n = S.TR * S.TN
    ys = Coll(Array(n, Array(16, Value(1))), out[1].refs)
    if y_calls:
        # a batch reading y out of the core's one output steps through part of a node, which the door's lowering refuses
        y = concat([B.call(bind(BitsWord, W=16), ys[i]).refs for i in range(n)])
    else:
        y = B.batch(bind(BitsWord, W=16), ys, axes=(0,)).refs
    return tuple_of(out[0], Coll(Array(S.TR, Array(S.TN, BF16)), y), U.ok_word(U._Emit(B), out[2]))


_STATICS = tuple(PT.PearlPlusTileCoreV3.statics)
PearlPlusTileCoreWired = CompositeDefinition(
    "PearlPlusTileCoreWired", 2, _STATICS,
    lambda S: (_word_params(S) + _noise_params(S), _word_ret(S)),
    _wired_body, refuse=PT.PearlPlusTileCoreV3.refuse, conformance=PT.PearlPlusTileCoreV3.conformance,
    doc="`PearlPlusTileCore_v3` on C's words with its noise by wire: b's BF16 words widened (`Bf16ToF32`), then each "
        "w-bit leaf its w bits by `WordBits{w}`, the salt, the root, eb, fa, fb and g each in the 16-bit words their "
        "writers wrote, its bits low bit first; y's 16-bit "
        "leaves back to BF16 words by `BitsWord{16}`, and ok one 16-bit word, bit 0 the core's ok and bits 1-15 zero "
        "(`tile_units.ok_word`)")


PearlPlusTileCoreWiredV3 = CompositeDefinition(
    "PearlPlusTileCoreWired", 3, _STATICS,
    lambda S: (_word_params(S, _WIDENED[3]) + _noise_params(S), _word_ret(S)),
    lambda B, S, *xs: _wired_body(B, S, *xs, widened=_WIDENED[3], y_calls=True), refuse=PT.PearlPlusTileCoreV3.refuse,
    conformance=PT.PearlPlusTileCoreV3.conformance,
    doc="`PearlPlusTileCoreWired_v2` with a's rows in BF16 words too, widened (`Bf16ToF32`) as b's are: a served linear's "
        "x as its writer holds it, read by wire; y's words each one `BitsWord{16}` call on its 16 bits of the core's "
        "output")


def _row_b_sigma(S) -> SpecializedDefinition:
    return bind(PT.PearlPlusTileRowBV2, K=S.K, DEV=S.DEV)


def _row_words(S) -> int:
    """The 16-bit words of one B row's unit output: its codes', factors' and statistics' bits, then its ok word."""
    return (_row_b_sigma(S).signature()[1].leaves - 1) // 16 + 1


def _row_b_words_sig(S):
    bt = dict(_row_b_sigma(S).signature()[0])
    x = Array(S.TN, Array(S.K, BF16)) if S.HEAD else Array(S.K, BF16)
    params = (("b", x),) + tuple((p, PT._word_type(bt[p])) for p in ("eb", "fb", "fa", "g"))
    return params, Tuple(Array(_row_words(S), PT.W16), *[Array(S.K, BF16)] * ((S.TN - 1) if S.HEAD else 0))


def _row_b_words_body(B, S, b, eb, fb, fa, g):
    sigma, k = _row_b_sigma(S), S.K
    bt = dict(sigma.signature()[0])
    rest = B.batch(bind(WordBits, W=16), Coll(Array((S.TN - 1) * k, Value(16)), b.refs.slice(k, S.TN * k)),
                   axes=(0,)) if S.HEAD and S.TN > 1 else None
    x = B.batch(Bf16ToF32, Coll(Array(k, BF16), b.refs.slice(0, k)), axes=(0,))
    xb = Coll(bt["b"], B.batch(bind(WordBits, W=32), Coll(Array(k, Value(32)), x.refs), axes=(0,)).refs)
    out = B.call(sigma, xb, *(PT._unpack(B, v, bt[p]) for p, v in (("eb", eb), ("fb", fb), ("fa", fa), ("g", g))))
    n = sigma.signature()[1].leaves - 1
    words = PT._pack(B, Coll(Array(n, Value(1)), out.refs.slice(0, n)), Array(n // 16, PT.W16))
    held = [concat([words.refs, U.ok_word(U._Emit(B), out[6]).refs])]
    for j in range(S.TN - 1 if rest is not None else 0):
        held.append(PT._pack(B, Coll(Array(16 * k, Value(1)), rest.refs.slice(16 * k * j, 16 * k * (j + 1))),
                             Array(k, BF16)).refs)
    return Coll(_row_b_words_sig(S)[1], concat(held))


def _refuse_row_b_words(b: dict) -> str | None:
    head, tn = b.get("HEAD", 0), b.get("TN", 1)
    if head not in (0, 1) or not (isinstance(tn, int) and tn >= 1):
        return f"HEAD is 0 or 1 and TN a count of B rows: HEAD = {head!r}, TN = {tn!r}"
    return PT._refuse_k(b)


PearlPlusTileRowBWords = CompositeDefinition(
    "PearlPlusTileRowBWords", 1, ("HEAD", "K", "DEV", "TN"), _row_b_words_sig, _row_b_words_body,
    refuse=_refuse_row_b_words, conformance=PT.PearlPlusTileCoreV3.conformance,
    doc="One of a -h3 tile's B rows on C's words, a unit of `PearlPlusTileCoreWired_v4`: its BF16 words widened "
        "(`Bf16ToF32`), its line and the noise in 16-bit words their bits, `PearlPlusTileRowB_v2` (σ of the row's "
        "`PearlCRowB` and `PearlPlusAdmit`), and its codes', factors' and statistics' bits back in 16-bit words, then its "
        "admission's ok word. HEAD 1: ``b`` is the tile's TN rows whole, the one registered row of the weights a block "
        "reads; it forms row 0 and writes rows 1 to TN − 1 back in BF16 words, one field each, for the other B row units")


def _rest_words_sig(S):
    rest = bind(PT.PearlPlusTileCoreRestV2, **{s: getattr(S, s) for s in _STATICS})
    bt, wp = dict(rest.signature()[0]), dict(_word_params(S, _WIDENED[3]))
    params = tuple((p, wp[p]) for p in _CORE_OWN if p != "b") + tuple((p, PT._word_type(bt[p])) for p in ("fa", "fb"))
    return params + (("rb", Array(S.TN, Array(_row_words(S), PT.W16))),), _word_ret(S)


def _rest_words_body(B, S, a, cert, salt, root, unit, rows, fa, fb, rb):
    rest = bind(PT.PearlPlusTileCoreRestV2, **{s: getattr(S, s) for s in _STATICS})
    bt, wt = dict(rest.signature()[0]), dict(_rest_words_sig(S)[0])
    bits = {}
    for p, x in zip(PT._REST_OWN, (a, cert, salt, root, unit, rows, fa, fb)):
        t = wt[p]
        if p in _WIDENED[3]:
            x = B.batch(Bf16ToF32, Coll(Array(t.leaves, BF16), x.refs), axes=(0,))
        w = 32 if p in _WIDENED[3] else t.leaf_widths()[0]
        bits[p] = Coll(bt[p], B.batch(bind(WordBits, W=w), Coll(Array(t.leaves, Value(w)), x.refs), axes=(0,)).refs)
    nw = _row_words(S)
    per, n = 16 * nw, 16 * (nw - 1)
    rbits = B.batch(bind(WordBits, W=16), Coll(Array(S.TN * nw, Value(16)), rb.refs), axes=(0,))
    rows_b = Coll(bt["rb"], concat([rbits.refs.slice(j * per, j * per + n) for j in range(S.TN)]))
    ok_b = Coll(bt["ok_b"], concat([rbits.refs.slice(j * per + n, j * per + n + 1) for j in range(S.TN)]))
    out = B.call(rest, *(bits[p] for p in PT._REST_OWN), rows_b, ok_b)
    m = S.TR * S.TN
    ys = Coll(Array(m, Array(16, Value(1))), out[1].refs)
    y = concat([B.call(bind(BitsWord, W=16), ys[i]).refs for i in range(m)])
    return tuple_of(out[0], Coll(Array(S.TR, Array(S.TN, BF16)), y), U.ok_word(U._Emit(B), out[2]))


PearlPlusTileCoreRestWords = CompositeDefinition(
    "PearlPlusTileCoreRestWords", 1, _STATICS, _rest_words_sig, _rest_words_body, refuse=PT.PearlPlusTileCoreV3.refuse,
    conformance=PT.PearlPlusTileCoreV3.conformance,
    doc="`PearlPlusTileCoreWired_v3` past its B rows, a unit of `PearlPlusTileCoreWired_v4`: its own ports and F as "
        "`_v3` takes them, and each B row unit's output words (``rb``, `PearlPlusTileRowBWords_v1`'s first field) their "
        "bits, `PearlPlusTileCoreRest_v2` (σ of the core past its B rows), y's bits back in BF16 words and ok one word")


def _wired4_body(B, S, a, b, cert, salt, root, unit, rows, eb, fa, fb, g):
    def row_b(head: int) -> SpecializedDefinition:
        return bind(PearlPlusTileRowBWords, HEAD=head, K=S.K, DEV=S.DEV, TN=S.TN)

    head = B.call(row_b(1), b, eb[0], fb, fa, g)
    rbs = [head[0]] + [B.call(row_b(0), head[j], eb[j], fb, fa, g)[0] for j in range(1, S.TN)]
    rest = B.call(bind(PearlPlusTileCoreRestWords, **{s: getattr(S, s) for s in _STATICS}), a, cert, salt, root, unit,
                  rows, fa, fb, Coll(Array(S.TN, Array(_row_words(S), PT.W16)), concat([r.refs for r in rbs])))
    return tuple_of(rest[0], rest[1], rest[2])


PearlPlusTileCoreWiredV4 = CompositeDefinition(
    "PearlPlusTileCoreWired", 4, _STATICS,
    lambda S: (_word_params(S, _WIDENED[3]) + _noise_params(S), _word_ret(S)),
    _wired4_body, refuse=PT.PearlPlusTileCoreV3.refuse, conformance=PT.PearlPlusTileCoreV3.conformance,
    doc="`PearlPlusTileCoreWired_v3`, the same function, as TN + 1 units on C's words so that each is a proof block "
        "within the composer's limit: `PearlPlusTileRowBWords_v1`{HEAD 1} on B's TN rows whole (row 0, and rows 1 to "
        "TN − 1 passed on), `PearlPlusTileRowBWords_v1`{HEAD 0} on each of rows 1 to TN − 1, and "
        "`PearlPlusTileCoreRestWords_v1` on the rest of the core's ports and every B row unit's output")


def _node_params(S, widened: tuple[str, ...] = _WIDENED[2]) -> tuple:
    return _word_params(S, widened) + (("col", U32),)


def _on_words(B, S, a, b, cert, salt, root, unit, rows, col, wired=None, seed_version: int = 4):
    eb, fa, fb, g = PT.noise(B, S.K, S.TN, S.DEV, salt, root, col, seed_version)
    return B.call(wired or _wired(S), a, b, cert, salt, root, unit, rows, eb, fa, fb, g)


PearlPlusTileCoreWords = CompositeDefinition(
    "PearlPlusTileCoreWords", 5, _STATICS, lambda S: (_node_params(S), _word_ret(S)),
    _on_words, refuse=PT.PearlPlusTileCoreV3.refuse, conformance=PT.PearlPlusTileCoreV3.conformance,
    doc="A -h3 tile on C's words, its noise derived inside as maps (`pearl_plus_tile.noise`): `PearlPlusTileSeed_v4`(salt, "
        "root, col) with eb's TN lines, `PearlPlusTileLine_v4` over F's K eight units of K/8, `PearlPlusGramRow_v5` for each of F_B's 32 columns, and "
        "`PearlPlusTileCoreWired_v2` on the tile's words and that noise by wire; the salt and the root in 16-bit words, two "
        "bytes a word, low byte first")
PearlPlusTileCoreWordsV6 = CompositeDefinition(
    "PearlPlusTileCoreWords", 6, _STATICS, lambda S: (_node_params(S, _WIDENED[3]), _word_ret(S)),
    lambda B, S, *xs: _on_words(B, S, *xs, wired=bind(PearlPlusTileCoreWiredV3, **{s: getattr(S, s) for s in _STATICS})),
    refuse=PT.PearlPlusTileCoreV3.refuse, conformance=PT.PearlPlusTileCoreV3.conformance,
    doc="`PearlPlusTileCoreWords_v5` with a's rows in BF16 words, its core `PearlPlusTileCoreWired_v3`: the tile in the "
        "place of a served linear's unit, x read by wire from its writer's BF16 row and B a registered read of the "
        "weight's BF16 rows")
PearlPlusTileCoreWordsV7 = CompositeDefinition(
    "PearlPlusTileCoreWords", 7, _STATICS, lambda S: (_node_params(S, _WIDENED[3]), _word_ret(S)),
    lambda B, S, *xs: _on_words(B, S, *xs, wired=bind(PearlPlusTileCoreWiredV4, **{s: getattr(S, s) for s in _STATICS}),
                                seed_version=5),
    refuse=PT.PearlPlusTileCoreV3.refuse, conformance=PT.PearlPlusTileCoreV3.conformance,
    doc="`PearlPlusTileCoreWords_v6`, its core `PearlPlusTileCoreWired_v4` (the same function, its core in TN + 1 units) "
        "and its seed `PearlPlusTileSeed_v5` (E_B's lines a call each, which the door's lowering takes)")


def cut(t: SpecializedDefinition) -> list[tuple[SpecializedDefinition, int]]:
    """The units C's laying (`verity.circuits.laying.lay`) makes of a binding of `PearlPlusTileCoreWords_v5`, which no
    binding fits whole (its noise alone is about 370k · K ANDs), each Definition bound with how many of it: its body's
    Calls and each map's members."""
    s = t.bindings
    g = s["K"] // PT.F_ROWS
    noise = [(PT.seed(s["K"], s["TN"], 5 if t.defn is PearlPlusTileCoreWordsV7 else 4), 1), (PT.line("fa", g), PT.F_ROWS), (PT.line("fb", g), PT.F_ROWS),
             *((PT.gram_row(s["K"], s["DEV"], p), 1) for p in range(PT.R))]
    if t.defn is PearlPlusTileCoreWordsV7:
        rows = [(bind(PearlPlusTileRowBWords, HEAD=h, K=s["K"], DEV=s["DEV"], TN=s["TN"]), n)
                for h, n in ((1, 1), (0, s["TN"] - 1)) if n]
        return noise + rows + [(bind(PearlPlusTileCoreRestWords, **s), 1)]
    wired = PearlPlusTileCoreWiredV3 if t.defn is PearlPlusTileCoreWordsV6 else PearlPlusTileCoreWired
    return noise + [(bind(wired, **s), 1)]


#: staged as its cut (`circuit_check.checks.cut_of`)
CUTS = {"PearlPlusTileCoreWords_v5": cut, "PearlPlusTileCoreWords_v6": cut, "PearlPlusTileCoreWords_v7": cut}


def credited_product(scheme: str, k: int, tm: int, tn: int, n: int, real: int | None = None) -> SpecializedDefinition:
    """`PearlPlusTileCoreWords_v5` for ``scheme`` at depth ``k`` over a TM × TN tile of a linear of ``n`` columns, its first
    ``real`` rows real (every row when None): the core's statics (`pearl_plus_tile.core`)."""
    core = PT.core(PS.get(scheme), k, tm, tn, n, real)
    return bind(PearlPlusTileCoreWords, **{s: core.bindings[s] for s in _STATICS})


def served_product(scheme: str, k: int, tm: int, tn: int, n: int, real: int | None = None) -> SpecializedDefinition:
    """`PearlPlusTileCoreWords_v6`, the tile reading x in BF16 words, at `credited_product`'s statics."""
    return bind(PearlPlusTileCoreWordsV6, **credited_product(scheme, k, tm, tn, n, real).bindings)


@dataclass(frozen=True)
class Tile:
    """One unit of a credited linear: ``core`` its Definition, ``rows`` its TM rows of the linear (those from M on are
    filler), ``cols`` its TN columns of the weight and ``col`` its column tile."""

    core: SpecializedDefinition
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    col: int

    @property
    def real(self) -> int:
        return self.core.bindings["TR"]


def tiles(scheme: str, k: int, m: int, n: int, tile: tuple[int, int] | None = None) -> list[Tile]:
    """The units of a credited linear of C over ``m`` rows of x, depth ``k`` and ``n`` columns, under ``scheme`` at its tile
    (``tile`` in its place: TM × TN): row tiles of TM rows, each column tile of TN columns, row-major, a row tile past
    ``m`` rows filler. Refused (ValueError) where TN doesn't divide N, or the core refuses the shape."""
    tm, tn = tile or PS.get(scheme).tile
    if m < 1 or n % tn:
        raise ValueError(f"a credited linear is at least one row and whole column tiles: m = {m}, TN = {tn}, N = {n}")
    out = []
    for i in range(0, m, tm):
        core = credited_product(scheme, k, tm, tn, n, min(tm, m - i))
        out += [Tile(core, tuple(range(i, i + tm)), tuple(range(c * tn, (c + 1) * tn)), c) for c in range(n // tn)]
    return out


def public(scheme: str, k: int, t: Tile, unit: int, salt: bytes, root: bytes) -> dict[str, Any]:
    """Tile ``t``'s public row (`PUBLIC`) for call ``unit`` under the epoch's ``salt`` and the weights' root ``root``, each
    field as the node's port holds it: the epoch's salt and the weights' root in 16-bit words (`le16`), and the tile's
    place (``unit``, ``rows``, ``col``)."""
    s = PS.get(scheme, k_min=k, tile=(len(t.rows), len(t.cols)))
    if s.hashing != "h3":
        raise ValueError(f"{scheme}: the node derives -h3's noise, not {s.hashing}'s")
    return {"salt": le16(salt), "root": le16(root), "unit": unit, "rows": list(t.rows), "col": t.col}


def le16(data: bytes) -> list[int]:
    """``data``'s bytes as the 16-bit words the node's ports hold them in: two bytes a word, the first the low byte."""
    if len(data) % 2:
        raise ValueError(f"{len(data)} bytes are no whole 16-bit words")
    return [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]


def _seed(salt: bytes, root_b: bytes) -> bytes:
    if len(salt) != 32 or len(root_b) != 64:
        raise ValueError(f"the noise reads a 32-byte salt and a 64-byte root_B, not {len(salt)} and {len(root_b)} bytes")
    return C.seed_b(salt, root_b, "h3")


def row_eb(salt: bytes, root_b: bytes, col: int, tn: int) -> bytes:
    """The node's eb at column tile ``col`` of ``tn`` columns: E_B's lines of columns col·TN … col·TN + TN − 1, 32 E4M3
    bytes each, in column order (32·TN bytes): line(seed_B, 1, 0, j, 256)."""
    sb = _seed(salt, root_b)
    return b"".join(bytes(C.sample_line(sb, 1, 0, j)) for j in range(col * tn, (col + 1) * tn))


def row_fa(salt: bytes, root_b: bytes, k: int) -> bytes:
    """The node's fa at depth ``k``: F_A's k lines, line(seed_B, 0, 1, t, 16) for t < k, 32 E4M3 bytes each (32·k bytes)."""
    return b"".join(bytes(ln) for ln in C._basis(_seed(salt, root_b), 0, k, C.LINE_NORM_V1))


def row_fb(salt: bytes, root_b: bytes, k: int) -> bytes:
    """The node's fb at depth ``k``: F_B's k lines, line(seed_B, 1, 1, t, 16) for t < k (32·k bytes)."""
    return b"".join(bytes(ln) for ln in C._basis(_seed(salt, root_b), 1, k, C.LINE_NORM_V1))


def row_gram(salt: bytes, root_b: bytes, k: int, dev: str) -> bytes:
    """The node's g at depth ``k`` on device ``dev``: G = F_Bᵀ·F_A on device ``dev``'s chain, G[p][q] = chain(F_B column p, F_A
    column q) for p, q < 32, p-major, each FP32 word 4 bytes little-endian (4,096 bytes)."""
    return b"".join(w.to_bytes(4, "little")
                    for row in C.gram(_seed(salt, root_b), k, C.LINE_NORM_V1, DEVICES[dev]) for w in row)


__all__ = ["CUTS", "PUBLIC", "PearlPlusTileCoreWired", "PearlPlusTileCoreWiredV3", "PearlPlusTileCoreWords",
           "PearlPlusTileCoreWordsV6", "Tile", "served_product", "cut", "credited_product", "le16",
           "public", "row_eb", "row_fa",
           "row_fb", "row_gram", "tiles"]
