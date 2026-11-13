"""`verity.protocols.poub.verifier.commitments.gates.block_check`: the reference of PoUB's opening unit is its Python
checks (`verity.protocols.poub.verifier.codec`, `receipts`, `audit`'s trees and `merkle.verify_opening`) on random
openings at p512/k2 and p1024/k3/masked, appended and registered, and names the check each wrong opening fails: a tampered
block, a payload not the committed one or not the committed KV, a word at or above p, a wrong append coin, a wrong receipt
leaf, salt or path, a plan without the block at the sampled position, a receipt the pass's tree commits but the block doesn't
give. The Boolean unit equals its reference on toy blocks with those negatives, and the parts equal theirs at the deployed
fields and cost the gates pinned here. The second version's reference gives v1's verdict on every opening under the identity
plan, opens a block past 2^16 in a store past 2^16 blocks and refuses one a position off, and its unit equals it."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from verity.protocols.poub.verifier import audit as AU
from verity.protocols.poub.verifier import codec as CO
from verity.protocols.poub.verifier import receipts as RC
from verity.circuits import boolean as BL
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.commitments import CommitmentDomain, Opening, verify_opening
from verity.protocols.poub.verifier.commitments.gates import block_check as BC
from verity.commitments.gates import frame_v3 as FV
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import MerkleTree, merkle_depth

FIELD = dict(BYTES=4096, ROWS=16, LEAF=16, P=40, DR=2, DP=3, DV=3)
P512 = SimpleNamespace(N=512, C=569, K=2, MASKED=0, **FIELD)
P1024 = SimpleNamespace(N=1024, C=105, K=3, MASKED=1, **FIELD)
TOY = SimpleNamespace(N=32, C=5, K=2, MASKED=0, BYTES=64, ROWS=2, LEAF=3, P=20, DR=1, DP=2, DV=1)
TOY_MASKED = SimpleNamespace(**{**vars(TOY), "MASKED": 1})
#: (part, its statics) -> (And, Xor, Not) gates of its traced circuit
PARTS = {
    ("RowCompress", ()): (57947, 286724, 4948),
    ("CodecBinding", (512, 569)): (112569, 545189, 11366),
    ("CodecModAdd", (512, 569)): (1534, 4091, 4),
    ("CodecLess", (512, 569)): (511, 1022, 5),
    ("CodecRound", (512, 569, 0)): (104912, 303366, 8202),
    ("CodecRound", (1024, 105, 1)): (310507, 901490, 23798),
    ("CodecMatch", (512,)): (512, 512, 512),
    ("SketchRow", (4096,)): (36863, 214292, 4099),
    ("LeafMatch", (16, 40)): (9646, 9034, 561),
    ("LeafMatchV2", (16,)): (8848, 8316, 561),
    ("BlockVerdict", (64,)): (69, 2, 5),
}
V2 = "hm96-sha512/row/v2"
PER_LEAF = BC.PER_LEAF


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


def _core_encode(S, seed: bytes, j: int, payload: bytes, coin: bytes | None) -> bytes:
    return CO.encode(payload, j, CO.key(seed, _codec(S)), bound=coin is not None, coin=coin)


def _hidden(row: bytes, salt: bytes) -> bytes:
    return BC.hm96_leaf(BC.sha512_row_v2_digest(row, 8 * len(row)), salt)[2]


def _domain(rng: random.Random, first: int, count: int) -> CommitmentDomain:
    return CommitmentDomain(rng.randbytes(64), 0, RangeIndexedDomain(first, first + count), hash="sha512")


def _opening(rng: random.Random, S, bound: bool, encode, pass_seed: bytes | None = None) -> SimpleNamespace:
    """An honest opening: block j (appended at first .. first + count - 1 under a coin, or registered among N blocks) sampled
    at position pos of a pass whose receipt tree's leaf pos >> 4 holds its receipt; its payload committed in its append's
    tree (or the registration's) and, appended, in a value commitment of at most 2^DV rows, under one salt."""
    N, seed = 4, rng.randbytes(32)
    pos = rng.randrange(PER_LEAF, min(S.P, 2 * PER_LEAF)) if S.P > PER_LEAF else rng.randrange(S.P)
    length = rng.randrange(pos + 1, S.P + 1)
    if bound:
        first, count = N + rng.randrange(3), 1 << S.DP
        j = first + rng.randrange(count)
        coin, start = rng.randbytes(32), first
    else:
        first, count, coin = 0, N, None
        j, start = rng.randrange(N), 0
    pay = _payload(rng, S)
    block = encode(S, seed, j, pay, coin)
    salt = rng.randbytes(BC.SALT_BYTES)
    pays = {start + u: _payload(rng, S) for u in range(count)}
    pays[j] = pay
    salts = {i: rng.randbytes(BC.SALT_BYTES) for i in pays}
    salts[j] = salt
    pdom = _domain(rng, start, count)
    ptree = MerkleTree(pdom, {i: _hidden(pays[i], salts[i]) for i in pays}, lambda _i: V2)
    values, vat = None, 0
    if bound:
        vcount = rng.randrange(1, (1 << S.DV) + 1)
        vat = rng.randrange(vcount)
        vdom = _domain(rng, 0, vcount)
        vals = {i: _hidden(_payload(rng, S), rng.randbytes(BC.SALT_BYTES)) for i in range(vcount)}
        vals[vat] = _hidden(pay, salt)
        vtree = MerkleTree(vdom, vals, lambda _i: V2)
        values = (vtree.commitment.root, vdom.domain_id, vat, merkle_depth(vcount))
    if S.BYTES == CO.BLOCK_BYTES:
        pass_seed = pass_seed or rng.randbytes(64)
        prefix, signs = RC.sketch_prefix(pass_seed), RC.sign_bits(pass_seed, S.ROWS)
    else:
        prefix, signs = rng.randbytes(128), [rng.randbytes(S.BYTES // 8) for _ in range(S.ROWS)]
    plan = sorted(rng.sample([x for x in range(1 << 12) if x != j], length - 1))
    plan.insert(pos, j)
    ys = [rng.randbytes(64) for _ in range(length)]
    ys[pos] = BC.receipt(prefix, j, block, signs)
    leaves = -(-length // PER_LEAF)
    rdom = _domain(rng, 0, leaves)
    rsalts = [rng.randbytes(BC.SALT_BYTES) for _ in range(leaves)]
    rows = [b"".join(ys[PER_LEAF * c:PER_LEAF * (c + 1)]) for c in range(leaves)]
    rtree = MerkleTree(rdom, {c: _hidden(rows[c], rsalts[c]) for c in range(leaves)}, lambda _c: V2)
    c = pos // PER_LEAF
    S = SimpleNamespace(**{**vars(S), "LEAF": len(rows[c]) // 64})
    pub = BC.opening_public(S, seed, j, coin, prefix, signs, pos, (rtree.commitment.root, rdom.domain_id, merkle_depth(leaves)),
                            (ptree.commitment.root, pdom.domain_id, j - start, merkle_depth(count)), values)
    vpath = vtree.open(vat).path if bound else ()
    args = BC.opening_args(S, pub, block, pay, rows[c], plan + [0] * (S.P - length), rsalts[c], salt, rtree.open(c).path,
                           ptree.open(j).path, vpath)
    return SimpleNamespace(S=S, args=args, seed=seed, j=j, pos=pos, coin=coin, block=block, pay=pay, salt=salt, rows=rows, c=c,
                           rsalts=rsalts, rtree=rtree, ptree=ptree, rdom=rdom, pdom=pdom, prefix=prefix, signs=signs,
                           pass_seed=pass_seed, plan=plan, ys=ys)


def _cases(rng: random.Random, S, encode) -> list[tuple[str, SimpleNamespace, dict, str | None]]:
    """An appended opening and a registered one, each honest, then with one thing wrong: (name, opening, args, the check
    it fails, None for accepted)."""
    out = []
    for bound in (True, False):
        o = _opening(rng, S, bound, encode)
        a, S_ = o.args, o.S
        k = o.pos % BC.PER_LEAF
        top = BC._p(S) + rng.randrange(S.C)
        over = top.to_bytes(S.N // 8, "little") + o.block[S.N // 8:]
        wrong_y = o.rows[o.c][:64 * k] + bytes(64) + o.rows[o.c][64 * (k + 1):]
        cases = {"honest": (a, None), "tampered block": (dict(a, block=_flip(a["block"], 3)), "codec"),
                 "other payload": (dict(a, payload=_flip(a["payload"], 1, 4)), "codec"),
                 "word at or above p": (dict(a, block=BC._u16s(over)), "codec"),
                 "receipt leaf altered": (dict(a, leaf=_flip(a["leaf"], 32 * k + 2)), "receipt-path"),
                 "another receipt of the leaf altered": (dict(a, leaf=_flip(a["leaf"], 32 * ((k + 1) % S_.LEAF))),
                                                         "receipt-path" if S_.LEAF > 1 else "receipt-path"),
                 "receipt leaf's salt": (dict(a, salts=_flip(a["salts"], 5)), "receipt-path"),
                 "receipt path": (dict(a, rpath=_flip(a["rpath"], 7)), "receipt-path" if a["rdepth"][0] else None),
                 "receipt tree deeper than its path": (dict(a, rdepth=[S.DR + 1]), "receipt-path"),
                 "plan without j at pos": (dict(a, plan=_flip(a["plan"], o.pos)), "receipt-path"),
                 "payload's salt": (dict(a, salts=_flip(a["salts"], 96 + 5)), "payload-path"),
                 "payload path": (dict(a, ppath=_flip(a["ppath"], 9)), "payload-path"),
                 "payload at another rank": (dict(a, prank=[a["prank"][0] ^ 1, 0]), "payload-path"),
                 "other pass's signs": (dict(a, signs=[x ^ 0x5A5A for x in a["signs"]]), "receipt"),
                 "a receipt the tree commits but the block doesn't give": (_committed_wrong(o, wrong_y), "receipt")}
        if bound:
            cases["wrong coin"] = (dict(a, coin=_flip(a["coin"], 7)), "codec")
            cases["unbound"] = (dict(a, bound=[0]), "codec")
            cases["value path"] = (dict(a, vpath=_flip(a["vpath"], 3)), "payload-path" if a["vdepth"][0] else None)
            cases["value at another position"] = (dict(a, vrank=[a["vrank"][0] ^ 1, 0]), "payload-path")
        else:
            cases["bound"] = (dict(a, bound=[1]), "codec")
            cases["value tree unread"] = (dict(a, vroot=[1] * 32, vpath=[2] * len(a["vpath"])), None)
        tag = "appended" if bound else "registered"
        out += [(f"{tag}: {n}", o, x, want) for n, (x, want) in cases.items()]
    return out


def _committed_wrong(o: SimpleNamespace, row: bytes) -> dict:
    """The opening of a pass whose receipt tree commits ``row`` as the sampled leaf: its path holds, its receipt isn't the
    block's."""
    rows = list(o.rows)
    rows[o.c] = row
    tree = MerkleTree(o.rdom, {c: _hidden(rows[c], o.rsalts[c]) for c in range(len(rows))}, lambda _c: V2)
    return dict(o.args, leaf=BC._u16s(row), rroot=BC._u16s(tree.commitment.root), rpath=BC._u16s(b"".join(
        tree.open(o.c).path) + bytes(64 * (o.S.DR - len(tree.open(o.c).path)))))


def test_the_key_is_the_codecs() -> None:
    rng = random.Random(1)
    for S in (P512, P1024):
        seed = rng.randbytes(32)
        k = CO.key(seed, _codec(S))
        assert BC.key(S, seed) == (k.c, k.tau, k.mask)


@pytest.mark.parametrize("S", [P512, P1024], ids=["p512-k2", "p1024-k3-masked"])
def test_the_reference_is_the_audits_opening(S) -> None:
    """The honest openings pass the audit's own checks (`codec.decode`, `SketchReceipter`, `receipts.hidden` and
    `verify_opening` on each tree) and the reference; each wrong one is refused, by the check it names."""
    rng = random.Random(S.N)
    for name, o, a, want in _cases(rng, S, _core_encode):
        got = BC.opening_check(o.S, a)
        assert BC.verdict(got) == want, (name, got)
        if name.endswith(": honest"):
            k = CO.key(o.seed, _codec(S))
            d = CO.digest(o.pay)
            assert (CO.decode(o.block, o.j, k, d, o.coin) if o.coin else CO.decode(o.block, o.j, k)) == o.pay
            assert RC.SketchReceipter(o.pass_seed, S.ROWS)(o.j, o.block) == o.ys[o.pos]
            assert RC.hidden(o.pay, o.salt) == _hidden(o.pay, o.salt) == RC.hidden_digest(d, o.salt)
            for tree, at, value in ((o.rtree, o.c, RC.hidden(o.rows[o.c], o.rsalts[o.c])), (o.ptree, o.j, RC.hidden(o.pay, o.salt))):
                assert verify_opening(tree.domain, tree.commitment, Opening(at, value, tree.open(at).path), V2, RC.LIMITS)


def _runs(blocks: list[int]) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for b in blocks:
        if out and sum(out[-1]) == b:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((b, 1))
    return tuple(out)


def test_the_core_trees_open_in_the_unit() -> None:
    """The audit's own trees (a pass's receipt tree under its seed and plan, an append's tree, the registration's payload
    tree) at their domains, ranks and depths, as `opening_public` takes them, are accepted."""
    rng = random.Random(7)
    S = P512
    reg = rng.randbytes(64)
    o = _opening(rng, S, True, _core_encode)
    first = o.ptree.domain.positions.start
    count = o.ptree.domain.count
    digests = {i: CO.digest(_payload(rng, S)) for i in range(first, first + count)}
    digests[o.j] = CO.digest(o.pay)
    salts = {i: rng.randbytes(BC.SALT_BYTES) for i in digests}
    salts[o.j] = o.salt
    app = AU.append_tree(reg, first, [digests[i] for i in sorted(digests)], [salts[i] for i in sorted(salts)])
    plan = RC.Plan(_runs(o.plan), 1 << 12)
    assert [plan.block(p) for p in range(plan.length)] == o.plan
    rtree = RC.tree(o.pass_seed, plan, o.ys, o.rsalts)
    rdom = RC.domain(o.pass_seed, plan)
    pub = BC.opening_public(o.S, o.seed, o.j, o.coin, o.prefix, o.signs, o.pos,
                            (rtree.commitment.root, rdom.domain_id, merkle_depth(rdom.count)),
                            (app.commitment.root, app.domain.domain_id, o.j - first, merkle_depth(count)),
                            (BC._bytes_of(o.args["vroot"]), BC._bytes_of(o.args["vdom"]), BC._int(o.args["vrank"]),
                             o.args["vdepth"][0]))
    a = dict(o.args, **{k: pub[k] for k in ("rroot", "rdom", "rdepth", "proot", "pdom", "prank", "pdepth")})
    a["rpath"] = BC._u16s(b"".join(rtree.open(o.c).path) + bytes(64 * (S.DR - len(rtree.open(o.c).path))))
    a["ppath"] = BC._u16s(b"".join(app.open(o.j).path) + bytes(64 * (S.DP - len(app.open(o.j).path))))
    assert BC.opening_check(o.S, a) == 1
    assert BC.verdict(BC.opening_check(o.S, dict(a, prank=o.args["prank"]))) is None
    assert BC.verdict(BC.opening_check(o.S, dict(a, pdom=o.args["pdom"]))) == "payload-path"


def test_the_rows_hold_every_parameter() -> None:
    for S in (P512, P1024, TOY):
        params, _ = BC._opening_signature(S)
        laid = BC.opening_rows(S)
        assert [p for _r, _b, ps in laid for p in ps] == [n for n, _ in params]
        assert sum(b for _r, b, _ps in laid) == sum(t.leaves for _, t in params)
        assert [r for r, _b, _ps in laid] == ["pub", "signs", "block", "payload", "leaf", "plan", "aux"]


def test_the_public_rows_are_the_verifiers_own() -> None:
    """`opening_public` is exactly the public rows' params of an instance's arguments, so a verifier holding only the public
    inputs derives the words its public rows open to."""
    rng = random.Random(4)
    for S, bound in ((P512, True), (P1024, False)):
        o = _opening(rng, S, bound, _core_encode)
        rows = BC.opening_rows(o.S)
        pub = {k: o.args[k] for _r, _b, ps in rows[:2] for k in ps}
        assert BC.row_words(rows[:2], pub) == BC.row_words(rows[:2], o.args)


@pytest.mark.parametrize("bad", [dict(N=64, C=59), dict(BYTES=64), dict(K=1), dict(MASKED=2), dict(BYTES=96), dict(LEAF=17),
                                 dict(DR=13), dict(DP=0), dict(DV=25), dict(P=1 << 16)])
def test_an_opening_off_the_codec_is_refused(bad) -> None:
    S = SimpleNamespace(**{**vars(P512), **bad})
    with pytest.raises(ValueError):
        BC.opening_rows(S)


def _u16_bits(words) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint64)
    return ((w[:, None] >> np.arange(16, dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1)


def _gates(defn, rows: list[dict]) -> list[int]:
    params, _ = defn.signature()
    cols = [np.stack([_u16_bits(a[n]) for a in rows]) for n, _ in params]
    out = np.asarray(evaluate_batch(defn, *cols)).reshape(len(rows), 16)
    return [int(sum(int(b) << i for i, b in enumerate(r))) for r in out]


@pytest.mark.slow
@pytest.mark.parametrize("S", [TOY, TOY_MASKED], ids=["toy", "toy-masked"])
def test_the_toy_unit_is_its_reference(S) -> None:
    """A few minutes (the compressions' and reads' first trace): the only evaluation of the unit's wiring, the tweak chain,
    rounds and masks, the payload's digest into the binding and its hm96 leaf, the sketch into the receipt, the receipt
    leaf's digest and hm96 leaf, each read's rank, position and depth, and the verdict's flags, which the deployed fields'
    parts never reach."""
    cases = _cases(random.Random(10 + S.MASKED), S, _toy_encode)
    by_leaf: dict[int, list] = {}
    for name, o, a, want in cases:
        by_leaf.setdefault(o.S.LEAF, []).append((name, o, a, want))
    for leaf, group in by_leaf.items():
        S_ = group[0][1].S
        refs = [BC.opening_check(S_, a) for _n, _o, a, _w in group]
        assert [BC.verdict(r) for r in refs] == [w for *_x, w in group], [n for n, *_ in group]
        assert _gates(bind(BC.BlockOpeningCheck, **vars(S_)), [a for _n, _o, a, _w in group]) == refs, leaf


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
    """About a minute alone (every part's trace): the deployed unit's cost is these parts' (`block_check`'s Cost), and no
    lowering at the deployed sizes runs anywhere else to pin it."""
    for (name, statics), counts in PARTS.items():
        got = getattr(BC, name).traced(*statics).counts
        assert got == dict(zip(("and", "xor", "not"), counts)), (name, got)


@pytest.mark.slow
def test_the_toy_unit_is_boolean() -> None:
    assert BL.is_boolean(bind(BC.BlockOpeningCheck, **vars(TOY)))


# --- the second version: the identity plan, u32 positions -------------------------------------------------------------------

def _v2(S, **kw) -> SimpleNamespace:
    """v1's statics as v2's: no P, and ``kw`` over them."""
    return SimpleNamespace(**{**{k: v for k, v in vars(S).items() if k != "P"}, **kw})


#: v2's statics where v1's are also defined: a store of up to 64 blocks, a pass of up to 4 receipt leaves
SMALL = dict(DR=2, DP=6, DV=2)
TOY_V2 = _v2(TOY, **SMALL)
#: a store past 2^16 blocks: 70,003, so a receipt tree of 13 levels, a payload tree of 17 and a last leaf of 3 receipts
DEEP, DEEP_N = _v2(TOY_V2, DR=13, DP=17), (1 << 16) + 4467


def _identity_opening(rng: random.Random, S, bound: bool, encode, n: int, j: int | None = None) -> SimpleNamespace:
    """An honest opening of block j, sampled at position j of a pass whose plan is the identity over a store of n blocks
    (appended under a coin past the registration's first 4, or registered), every tree sparse: the opened path's siblings
    random and the root the one they give (`frame_v3.root_at_ref`)."""
    first = 4 if bound else 0
    j = rng.randrange(first, n) if j is None else j
    seed, coin = rng.randbytes(32), (rng.randbytes(32) if bound else None)
    pay, salt = _payload(rng, S), rng.randbytes(BC.SALT_BYTES)
    block = encode(S, seed, j, pay, coin)
    if S.BYTES == CO.BLOCK_BYTES:
        pass_seed = rng.randbytes(64)
        prefix, signs = RC.sketch_prefix(pass_seed), RC.sign_bits(pass_seed, S.ROWS)
    else:
        prefix, signs = rng.randbytes(128), [rng.randbytes(S.BYTES // 8) for _ in range(S.ROWS)]
    c = j // PER_LEAF
    ys = [rng.randbytes(64) for _ in range(min(PER_LEAF, n - PER_LEAF * c))]
    ys[j % PER_LEAF] = BC.receipt(prefix, j, block, signs)
    row, rsalt = b"".join(ys), rng.randbytes(BC.SALT_BYTES)

    def tree(value: bytes, rank: int, position: int, count: int) -> tuple:
        dom, depth = rng.randbytes(64), merkle_depth(count)
        path = [rng.randbytes(64) for _ in range(depth)]
        return FV.root_at_ref(dom, V2, value, rank, position, path), dom, depth, path

    rroot, rdom, rdepth, rpath = tree(_hidden(row, rsalt), c, c, -(-n // PER_LEAF))
    proot, pdom, pdepth, ppath = tree(_hidden(pay, salt), j - first, j, n - first)
    values, vpath = None, []
    if bound:
        vcount = rng.randrange(2, (1 << S.DV) + 1)
        vat = rng.randrange(vcount)
        vroot, vdom, vdepth, vpath = tree(_hidden(pay, salt), vat, vat, vcount)
        values = (vroot, vdom, vat, vdepth)
    S = SimpleNamespace(**{**vars(S), "LEAF": len(ys)})
    trees = ((rroot, rdom, rdepth), (proot, pdom, j - first, pdepth), values)
    pub = BC.opening_public_v2(S, seed, j, coin, prefix, signs, *trees)
    return SimpleNamespace(S=S, j=j, first=first, seed=seed, coin=coin, prefix=prefix, signs=signs, trees=trees,
                           args=BC.opening_args_v2(S, pub, block, pay, row, rsalt, salt, rpath, ppath, vpath))


def _as_v1(S, a: dict) -> tuple[SimpleNamespace, dict]:
    """v1's binding and arguments for v2's (S, a) at a position below 2^16: the plan the identity row of 64 entries."""
    assert a["pos"][1] == 0
    return SimpleNamespace(**vars(S), P=64), dict(a, pos=a["pos"][:1], plan=list(range(64)))


def _v2_cases(o: SimpleNamespace) -> dict[str, dict]:
    """The honest opening, then each with one thing wrong; a position or a sample off by one where the store has one."""
    a, j = o.args, o.j
    k = j % PER_LEAF
    out = {"honest": a, "tampered block": dict(a, block=_flip(a["block"], 3)),
           "other payload": dict(a, payload=_flip(a["payload"], 1, 4)),
           "receipt leaf altered": dict(a, leaf=_flip(a["leaf"], 32 * k + 2)),
           "receipt path": dict(a, rpath=_flip(a["rpath"], 7)), "payload path": dict(a, ppath=_flip(a["ppath"], 9)),
           "receipt leaf's salt": dict(a, salts=_flip(a["salts"], 5)), "other pass's signs": dict(a, signs=[x ^ 0x5A5A for x in a["signs"]]),
           "position one past": dict(a, pos=BC._u16s(j + 1, 2)), "position one before": dict(a, pos=BC._u16s(j - 1, 2)),
           "the sample one past": dict(a, **BC.opening_public_v2(o.S, o.seed, j + 1, o.coin, o.prefix, o.signs, o.trees[0],
                                                                 (*o.trees[1][:2], j + 1 - o.first, o.trees[1][3]), o.trees[2]))}
    if o.coin is not None:
        out["value path"] = dict(a, vpath=_flip(a["vpath"], 3))
    return out


@pytest.mark.parametrize("S", [TOY_V2, _v2(P512, **SMALL), _v2(P1024, **SMALL)], ids=["toy", "p512-k2", "p1024-k3-masked"])
def test_v2_gives_v1s_verdict_under_the_identity_plan(S) -> None:
    """Where both are defined (positions below 2^16, v1's plan the identity row), v2's reference returns v1's verdict word
    on the honest opening and on each wrong one: a wrong block, receipt path, payload path or position among them."""
    encode = _toy_encode if (S.N, S.C) == BC.TOY else _core_encode
    rng = random.Random(S.N + 2)
    for bound in (False, True):
        o = _identity_opening(rng, S, bound, encode, 40, rng.randrange(17, 39))
        for name, a in _v2_cases(o).items():
            got = BC.opening_check_v2(o.S, a)
            assert got == BC.opening_check(*_as_v1(o.S, a)), (bound, name)
            assert (got == 1) == (name == "honest"), (bound, name, got)


def test_a_position_past_2_16_opens() -> None:
    """A store of 70,003 blocks opened at positions above 65,535: the honest opening is accepted, and the same block at a
    position one off, or 2^16 off (which a u16 position would not tell apart), or the next block's sample, refused."""
    rng = random.Random(17)
    for bound in (False, True):
        o = _identity_opening(rng, DEEP, bound, _toy_encode, DEEP_N, rng.randrange(1 << 16, DEEP_N - 1))
        assert o.args["pos"][1] == 1 and o.args["rdepth"] == [13] and o.args["pdepth"] == [17]
        cases = _v2_cases(o)
        cases["position 2^16 before"] = dict(o.args, pos=BC._u16s(o.j - (1 << 16), 2))
        for name, a in cases.items():
            got = BC.opening_check_v2(o.S, a)
            assert (got == 1) == (name == "honest"), (bound, name, got)
            if name.startswith("position"):
                assert BC.verdict(got) == "receipt-path", (bound, name, got)


def test_v2s_rows_hold_every_parameter_and_its_public_rows_are_the_verifiers() -> None:
    rng = random.Random(18)
    for S in (_v2(P512), TOY_V2, DEEP):
        params, _ = BC._opening_signature_v2(S)
        laid = BC.opening_rows_v2(S)
        assert [p for _r, _b, ps in laid for p in ps] == [n for n, _ in params]
        assert sum(b for _r, b, _ps in laid) == sum(t.leaves for _, t in params)
        assert [r for r, _b, _ps in laid] == ["pub", "signs", "block", "payload", "leaf", "aux"]
    o = _identity_opening(rng, TOY_V2, True, _toy_encode, 40)
    rows = BC.opening_rows_v2(o.S)
    pub = {k: o.args[k] for _r, _b, ps in rows[:2] for k in ps}
    assert BC.row_words(rows[:2], pub) == BC.row_words(rows[:2], o.args)


@pytest.mark.parametrize("bad", [dict(N=64, C=59), dict(BYTES=64), dict(K=1), dict(K=4), dict(MASKED=2), dict(ROWS=17),
                                 dict(ROWS=64), dict(LEAF=17), dict(DR=29), dict(DP=33), dict(DV=33), dict(DR=0),
                                 dict(DR=28, DP=32, DV=5), dict(DR=24, DP=28, DV=13)])
def test_a_v2_binding_past_its_range_is_refused_by_name(bad) -> None:
    with pytest.raises(ValueError, match="^BlockOpeningCheck_v2: "):
        BC.opening_rows_v2(_v2(P1024, **bad))


def test_v2s_costliest_legal_bindings_are_admitted() -> None:
    for field in ((1024, 105), (512, 569), BC.TOY):
        S = SimpleNamespace(N=field[0], C=field[1], K=3, MASKED=1, BYTES=4096, ROWS=16, LEAF=16, DR=28, DP=32, DV=BC.LEVELS - 60)
        assert BC.opening_rows_v2(S)


@pytest.mark.slow
def test_the_toy_unit_v2_is_its_reference() -> None:
    """A few minutes (the compressions' and the deep reads' first trace): the only evaluation of v2's wiring, its u32
    position into the receipt leaf's rank, position and entry and pos = j, at a store past 2^16 blocks as at a small one."""
    rng = random.Random(19)
    for S, n in ((TOY_V2, 35), (_v2(TOY_V2, MASKED=1), 35), (DEEP, DEEP_N)):
        group = [a for bound in (False, True) for a in _v2_cases(_identity_opening(rng, S, bound, _toy_encode, n, n - 2)).values()]
        S_ = SimpleNamespace(**{**vars(S), "LEAF": 3})
        refs = [BC.opening_check_v2(S_, a) for a in group]
        assert refs.count(1) == 2
        assert _gates(bind(BC.BlockOpeningCheckV2, **vars(S_)), group) == refs
