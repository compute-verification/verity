"""`verity.ml.commitments.gates.block_check`: the references of the remote bandwidth audit's check-3 units are its Python
decode and receipts (`verity.core.protocols.remote_bandwidth.codec`, `receipts`) on random blocks at p512/k2 and
p1024/k3/masked, refusing what they refuse: a tampered block, a wrong append coin c_a, a payload that is not the committed
one, a word at or above p, a wrong receipt leaf, a plan without the block at the drawn position. The Boolean units equal
their references on toy blocks with those negatives, and the parts equal theirs at the deployed fields and cost the gates
pinned here."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from verity.core.protocols.remote_bandwidth import codec as CO
from verity.core.protocols.remote_bandwidth import receipts as RC
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.commitments.gates import block_check as BC

P512 = SimpleNamespace(N=512, C=569, K=2, MASKED=0, BYTES=4096, WORDS=64)
P512_SLICED = SimpleNamespace(N=512, C=569, K=2, MASKED=0, BYTES=4096, WORDS=16)
P1024 = SimpleNamespace(N=1024, C=105, K=3, MASKED=1, BYTES=4096, WORDS=32)
TOY = SimpleNamespace(N=32, C=5, K=2, MASKED=0, BYTES=64, WORDS=16)
TOY_SLICED = SimpleNamespace(N=32, C=5, K=2, MASKED=1, BYTES=64, WORDS=4)
TOY_RECEIPT = SimpleNamespace(ROWS=2, LEAF=3, P=4, BYTES=64)
#: (part, its statics) -> (And, Xor, Not) gates of its traced circuit
PARTS = {
    ("RowCompress", ()): (57947, 286724, 4948),
    ("CodecBinding", (512, 569)): (112569, 545189, 11366),
    ("CodecModAdd", (512, 569)): (1534, 4091, 4),
    ("CodecLess", (512, 569)): (511, 1022, 5),
    ("CodecRound", (512, 569, 0)): (104912, 303366, 8202),
    ("CodecRound", (1024, 105, 1)): (310507, 901490, 23798),
    ("CodecMatch", (512,)): (512, 512, 512),
    ("CodecPick", (512, 16, 4)): (32795, 24576, 16),
    ("SketchRow", (4096,)): (36863, 214292, 4099),
    ("LeafMatch", (16, 32)): (9493, 8890, 561),
}


def _codec(S) -> CO.Codec:
    return CO.Codec(CO.FIELDS[BC.FIELDS[S.N, S.C]], S.K, bool(S.MASKED))


def _payload(rng: random.Random, S) -> bytes:
    """Random bytes whose every word is below p (its top byte below 0x80)."""
    out = bytearray(rng.randbytes(S.BYTES))
    out[S.N // 8 - 1::S.N // 8] = bytes(b & 0x7F for b in out[S.N // 8 - 1::S.N // 8])
    return bytes(out)


def _flip(words: list[int], at: int = 0, bit: int = 1) -> list[int]:
    return words[:at] + [words[at] ^ bit] + words[at + 1:]


def _toy_encode(S, seed: bytes, j: int, payload: bytes, coin: bytes | None) -> bytes:
    """`codec.encode` over the toy field: core's ρ on each round, the reference's key and binding."""
    f = CO.Field(BC.FIELDS[S.N, S.C], S.N, S.C)
    c, tau, mask = BC.key(S, seed)
    w, W, p = S.N // 8, S.BYTES // (S.N // 8), f.p
    b = BC._binding(S, seed, j, BC.sha512_row_v2_digest(payload, 8 * S.BYTES), coin) if coin is not None else 0
    out = b""
    for i in range(W):
        x, t = int.from_bytes(payload[w * i:w * (i + 1)], "little"), (tau * (W * j + i) + b) % p
        for r, cr in enumerate(c):
            if r and mask is not None:
                x = x ^ mask if x ^ mask < p else x
            x = CO.rho((x + cr + t) % p, f)
        out += x.to_bytes(w, "little")
    return out


def _decode_cases(rng: random.Random, S, encode) -> tuple[list[dict], list[str]]:
    """An appended block (bound under its coin) and a registered one (the seed's key), each honest, then with one thing
    wrong, at a slice q drawn at random (the block whole when WORDS is W); a sliced unit's honest instances are every slice."""
    rows, names = [], []
    slices = BC._words_of(S) // S.WORDS
    for coin in (rng.randbytes(32), None):
        seed, j, q = rng.randbytes(32), rng.randrange(1 << 20), rng.randrange(slices)
        pay = _payload(rng, S)
        blk = encode(S, seed, j, pay, coin)
        a = BC.decode_args(S, seed, j, blk, pay, coin, q)
        at, w = q * S.WORDS * S.N // 16, S.N // 8
        top = BC._p(S) + rng.randrange(S.C)
        over = blk[:w * q * S.WORDS] + top.to_bytes(w, "little") + blk[w * (q * S.WORDS + 1):]
        cases = {"honest": a, "tampered block": dict(a, block=_flip(a["block"], at + 3)),
                 "other payload": dict(a, payload=_flip(a["payload"], at + 1, 4)),
                 "word at or above p": dict(a, block=BC._u16s(over)),
                 "other block index": BC.decode_args(S, seed, j + 1, blk, pay, coin, q),
                 "slice past the block": dict(a, q=[slices])}
        if coin is not None:
            cases["wrong coin"] = dict(a, coin=_flip(a["coin"], 7))
            cases["unbound"] = dict(a, bound=[0])
        else:
            cases["bound"] = BC.decode_args(S, seed, j, blk, pay, bytes(32), q)
        if slices > 1:
            cases["another slice's tweak"] = dict(a, t0=BC.decode_args(S, seed, j, blk, pay, coin, (q + 1) % slices)["t0"])
            for r in range(slices):
                if r != q:
                    cases[f"honest slice {r}"] = BC.decode_args(S, seed, j, blk, pay, coin, r)
        tag = "appended" if coin is not None else "registered"
        for name, args in cases.items():
            rows.append(args)
            names.append(f"{tag}: {name}")
    return rows, names


def test_the_key_is_the_codecs() -> None:
    rng = random.Random(1)
    for S in (P512, P1024):
        seed = rng.randbytes(32)
        k = CO.key(seed, _codec(S))
        assert BC.key(S, seed) == (k.c, k.tau, k.mask)


@pytest.mark.parametrize("S", [P512, P512_SLICED, P1024], ids=["p512-k2", "p512-k2-sliced", "p1024-k3-masked"])
def test_the_decode_reference_is_the_audits_decode(S) -> None:
    """On random blocks the reference accepts exactly what `codec.decode` decodes to the instance's payload, under the
    payload's own digest and the append's coin (an appended block) or the seed's key (a registered one), slice by slice;
    each negative alters the instance's own slice."""
    rng = random.Random(S.N)

    def encode(S, seed, j, pay, coin):
        return CO.encode(pay, j, CO.key(seed, _codec(S)), bound=coin is not None, coin=coin)

    rows, names = _decode_cases(rng, S, encode)
    for a, name in zip(rows, names):
        blk, pay = BC._bytes_of(a["block"]), BC._bytes_of(a["payload"])
        k = CO.key(BC._bytes_of(a["seed"]), _codec(S))
        j = BC._int(a["j"])
        if a["bound"][0]:
            got = CO.decode(blk, j, k, CO.digest(pay), BC._bytes_of(a["coin"]))
        else:
            got = CO.decode(blk, j, k)
        assert BC.decode_check(S, a) == int("honest" in name), name
        if name.partition(": ")[2] not in ("another slice's tweak", "slice past the block"):
            assert int(got == pay) == int("honest" in name), name


def test_the_receipt_reference_is_the_sketch_receipter() -> None:
    """The reference's receipt is `SketchReceipter`'s at 16 rows, and it accepts a receipt leaf holding it at k with the
    block at the plan's drawn position, and nothing else."""
    rng = random.Random(3)
    S = SimpleNamespace(ROWS=RC.SKETCH_ROWS, LEAF=RC.PER_LEAF, P=8, BYTES=CO.BLOCK_BYTES)
    seed = rng.randbytes(64)
    signs, prefix = RC.sign_bits(seed, S.ROWS), RC.sketch_prefix(seed)
    rec = RC.SketchReceipter(seed, S.ROWS)
    blk, other = rng.randbytes(S.BYTES), rng.randbytes(S.BYTES)
    plan = rng.sample(range(1 << 16), S.P)
    pos, k = 5, 9
    j = plan[pos]
    assert BC.receipt(prefix, j, blk, signs) == rec(j, blk)
    assert BC.sketch(blk, signs) == rec.sketch(blk)
    leaf = b"".join(rec(j + 1 + i, other) for i in range(S.LEAF))
    leaf = leaf[:64 * k] + rec(j, blk) + leaf[64 * (k + 1):]
    a = BC.receipt_args(S, prefix, signs, j, k, pos, blk, leaf, plan)
    wrong_leaf = leaf[:64 * k] + rec(j, other) + leaf[64 * (k + 1):]
    cases = {"honest": a, "wrong receipt leaf": dict(a, leaf=BC._u16s(wrong_leaf)), "wrong k": dict(a, k=[k + 1]),
             "k past the leaf": dict(a, k=[S.LEAF]), "tampered block": dict(a, block=_flip(a["block"], 11, 0x80)),
             "plan without j at pos": dict(a, pos=[pos + 1]), "pos past the plan": dict(a, pos=[S.P]),
             "other pass's signs": dict(a, signs=BC.receipt_args(S, prefix, RC.sign_bits(rng.randbytes(64), S.ROWS), j, k,
                                                                  pos, blk, leaf, plan)["signs"])}
    assert {n: BC.receipt_check(S, c) for n, c in cases.items()} == {n: int(n == "honest") for n in cases}


def test_the_rows_hold_every_parameter() -> None:
    for S, sig, rows in ((P512, BC._decode_signature, BC.decode_rows), (P1024, BC._decode_signature, BC.decode_rows),
                         (P512_SLICED, BC._decode_signature, BC.decode_rows),
                         (TOY_RECEIPT, BC._receipt_signature, BC.receipt_rows)):
        params, _ = sig(S)
        laid = rows(S)
        assert [p for _r, _b, ps in laid for p in ps] == [n for n, _ in params]
        assert sum(b for _r, b, _ps in laid) == sum(t.leaves for _, t in params)


def test_the_public_rows_are_the_verifiers_own() -> None:
    """`decode_public` and `receipt_public` are exactly the public rows' params of an instance's arguments, so a verifier
    holding only the public inputs derives the words its public rows open to."""
    rng = random.Random(4)
    for S, coin in ((P512_SLICED, rng.randbytes(32)), (P1024, None)):
        seed, j, q = rng.randbytes(32), rng.randrange(1 << 20), 1
        a = BC.decode_args(S, seed, j, rng.randbytes(S.BYTES), rng.randbytes(S.BYTES), coin, q)
        pub = BC.decode_public(S, seed, j, coin, q)
        assert list(pub) == BC.decode_rows(S)[0][2] and pub == {p: a[p] for p in pub}
    S = SimpleNamespace(ROWS=RC.SKETCH_ROWS, LEAF=RC.PER_LEAF, P=4, BYTES=CO.BLOCK_BYTES)
    seed = rng.randbytes(64)
    prefix, signs = RC.sketch_prefix(seed), RC.sign_bits(seed, S.ROWS)
    a = BC.receipt_args(S, prefix, signs, 7, 2, 1, rng.randbytes(S.BYTES), rng.randbytes(64 * S.LEAF), [3, 7, 5, 1])
    pub = BC.receipt_public(S, prefix, signs, 7, 2, 1)
    assert list(pub) == [p for _r, _b, ps in BC.receipt_rows(S)[:2] for p in ps] and pub == {p: a[p] for p in pub}


@pytest.mark.parametrize("bad", [dict(N=64, C=59), dict(BYTES=64), dict(K=1), dict(MASKED=2), dict(BYTES=96), dict(WORDS=24),
                                 dict(WORDS=0)])
def test_a_decode_off_the_codec_is_refused(bad) -> None:
    S = SimpleNamespace(**{**vars(P512), **bad})
    with pytest.raises(ValueError):
        BC.decode_rows(S)


def _u16_bits(words) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint64)
    return ((w[:, None] >> np.arange(16, dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1)


def _gates(defn, rows: list[dict]) -> list[int]:
    params, _ = defn.signature()
    cols = [np.stack([_u16_bits(a[n]) for a in rows]) for n, _ in params]
    out = np.asarray(evaluate_batch(defn, *cols)).reshape(len(rows), 16)
    assert not out[:, 1:].any()
    return out[:, 0].tolist()


@pytest.mark.slow
@pytest.mark.parametrize("S", [TOY, TOY_SLICED], ids=["whole", "sliced-masked"])
def test_the_toy_decode_unit_is_its_reference(S) -> None:
    """About 25 s (the compressions' first trace): the only evaluation of the unit's wiring, the slice picked by q, the
    tweak chain, rounds and masks, the payload's digest and the binding's message, which the deployed fields' parts never
    reach."""
    rows, names = _decode_cases(random.Random(10 + S.WORDS), S, _toy_encode)
    want = [BC.decode_check(S, a) for a in rows]
    assert want == [int("honest" in n) for n in names]
    assert _gates(bind(BC.BlockDecodeCheck, **vars(S)), rows) == want


@pytest.mark.slow
def test_the_toy_receipt_unit_is_its_reference() -> None:
    S = TOY_RECEIPT
    rng = random.Random(4)
    rows = []
    for _ in range(2):
        prefix, signs, blk = rng.randbytes(128), [rng.randbytes(S.BYTES // 8) for _ in range(S.ROWS)], rng.randbytes(S.BYTES)
        plan = rng.sample(range(1 << 16), S.P)
        pos, k = rng.randrange(S.P), rng.randrange(S.LEAF)
        j = plan[pos]
        y = BC.receipt(prefix, j, blk, signs)
        leaf = b"".join(y if i == k else rng.randbytes(64) for i in range(S.LEAF))
        a = BC.receipt_args(S, prefix, signs, j, k, pos, blk, leaf, plan)
        rows += [a, dict(a, leaf=_flip(a["leaf"], 32 * k + 3)), dict(a, k=[(k + 1) % S.LEAF]), dict(a, k=[S.LEAF]),
                 dict(a, block=_flip(a["block"], 2, 0x8000)), dict(a, pos=[(pos + 1) % S.P]), dict(a, pos=[S.P]),
                 dict(a, j=_flip(a["j"], 2), plan=a["plan"])]
    want = [BC.receipt_check(S, a) for a in rows]
    assert want == [1, 0, 0, 0, 0, 0, 0, 0] * 2
    assert _gates(bind(BC.BlockReceiptCheck, **vars(S)), rows) == want


def _bits(x: int, n: int) -> np.ndarray:
    return np.array([(x >> i) & 1 for i in range(n)], dtype=np.uint8)


def _value(bits) -> int:
    return sum(int(b) << i for i, b in enumerate(bits))


@pytest.mark.slow
@pytest.mark.parametrize("S", [P512, P1024], ids=["p512-k2", "p1024-k3-masked"])
def test_the_rounds_and_the_binding_at_the_deployed_fields_are_the_codecs(S) -> None:
    """About 40 s at p1024 (its round's trace): the only evaluation of the Karatsuba squares and the folds by C at the
    deployed widths, and of the binding's reduction mod p, which a toy word (one schoolbook square) never reaches."""
    rng = random.Random(S.N + 1)
    f = CO.FIELDS[BC.FIELDS[S.N, S.C]]
    p, n = f.p, S.N
    mask = rng.getrandbits(n) if S.MASKED else None
    xs = [0, 1, 2, p - 1, p - 2, (p - 1) // 2] + [rng.randrange(p) for _ in range(10)]
    cs = [p - 1, 0] + [rng.randrange(p) for _ in range(len(xs) - 2)]
    ts = [p - 1] + [rng.randrange(p) for _ in range(len(xs) - 1)]
    rnd = bind(BC.CodecRound, N=S.N, C=S.C, MASK=S.MASKED)
    cols = [np.stack([_bits(v, n) for v in vs]) for vs in (xs, cs, ts)]
    if S.MASKED:
        cols.append(np.stack([_bits(mask, n)] * len(xs)))
    got = [_value(r) for r in np.asarray(evaluate_batch(rnd, *cols)).reshape(len(xs), n)]
    want = []
    for x, c, t in zip(xs, cs, ts):
        y = (CO.rho_inv(x, f) - c - t) % p
        want.append((y ^ mask if y ^ mask < p else y) if S.MASKED else y)
    assert got == want

    k = CO.key(rng.randbytes(32), _codec(S))
    d, coin, j = rng.randbytes(64), rng.randbytes(32), rng.randrange(1 << 40)
    b = bind(BC.CodecBinding, N=S.N, C=S.C)
    dw = [int.from_bytes(d[8 * i:8 * i + 8], "big") for i in range(8)]
    args = [np.array([[1], [0]], dtype=np.uint8), np.stack([np.unpackbits(np.frombuffer(k.seed, np.uint8), bitorder="little")] * 2),
            np.stack([_bits(j, 64)] * 2), np.stack([np.concatenate([_bits(w, 64) for w in dw])] * 2),
            np.stack([np.unpackbits(np.frombuffer(coin, np.uint8), bitorder="little")] * 2)]
    out = np.asarray(evaluate_batch(b, *args)).reshape(2, n)
    assert [_value(r) for r in out] == [CO.binding(d, j, k, coin), 0]


@pytest.mark.slow
def test_the_parts_cost_the_pinned_gates() -> None:
    """About a minute alone (every part's trace): the deployed units' cost is these parts' (`block_check`'s Cost), and no
    lowering at the deployed sizes runs anywhere else to pin it."""
    for (name, statics), (ands, xors, nots) in PARTS.items():
        assert getattr(BC, name).traced(*statics).counts == {"and": ands, "xor": xors, "not": nots}, name


@pytest.mark.slow
def test_the_toy_units_are_boolean() -> None:
    assert BL.is_boolean(bind(BC.BlockDecodeCheck, **vars(TOY)))
    assert BL.is_boolean(bind(BC.BlockReceiptCheck, **vars(TOY_RECEIPT)))
