"""Openings of PoUB's opening unit (`verity.ml.commitments.gates.block_check`), honest and each with one thing wrong, at the
deployed fields and the toy one, for `test_block_check.py` and PoUB's own tests (`test_block_check_codec.py`, which loads
this file). A block's encoding and a pass's sketch come from the caller: the codec is PoUB's."""
from __future__ import annotations

import random
from types import SimpleNamespace

from verity.ml.commitments import CommitmentDomain
from verity.ml.commitments.gates import block_check as BC
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import MerkleTree, merkle_depth

FIELD = dict(BYTES=4096, ROWS=16, LEAF=16, P=40, DR=2, DP=3, DV=3)
P512 = SimpleNamespace(N=512, C=569, K=2, MASKED=0, **FIELD)
P1024 = SimpleNamespace(N=1024, C=105, K=3, MASKED=1, **FIELD)
TOY = SimpleNamespace(N=32, C=5, K=2, MASKED=0, BYTES=64, ROWS=2, LEAF=3, P=20, DR=1, DP=2, DV=1)
TOY_MASKED = SimpleNamespace(**{**vars(TOY), "MASKED": 1})
V2 = "hm96-sha512/row/v2"
PER_LEAF = BC.PER_LEAF


def payload(rng: random.Random, S) -> bytes:
    """Random bytes whose every word is below p (its top byte below 0x80)."""
    out = bytearray(rng.randbytes(S.BYTES))
    out[S.N // 8 - 1::S.N // 8] = bytes(b & 0x7F for b in out[S.N // 8 - 1::S.N // 8])
    return bytes(out)


def flip(words: list[int], at: int = 0, bit: int = 1) -> list[int]:
    return words[:at] + [words[at] ^ bit] + words[at + 1:]


def hidden(row: bytes, salt: bytes) -> bytes:
    return BC.hm96_leaf(BC.sha512_row_v2_digest(row, 8 * len(row)), salt)[2]


def domain(rng: random.Random, first: int, count: int) -> CommitmentDomain:
    return CommitmentDomain(rng.randbytes(64), 0, RangeIndexedDomain(first, first + count), hash="sha512")


def opening(rng: random.Random, S, bound: bool, encode, pass_seed: bytes | None = None, sketch=None) -> SimpleNamespace:
    """An honest opening: block j (appended at first .. first + count - 1 under a coin, or registered among N blocks) sampled
    at position pos of a pass whose receipt tree's leaf pos >> 4 holds its receipt; its payload committed in its append's
    tree (or the registration's) and, appended, in a value commitment of at most 2^DV rows, under one salt. ``encode(S, seed,
    j, payload, coin)`` is the block's encoding; ``sketch(pass_seed, rows)`` is the pass's sketch prefix and sign bits from
    its seed, and without it they are random."""
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
    pay = payload(rng, S)
    block = encode(S, seed, j, pay, coin)
    salt = rng.randbytes(BC.SALT_BYTES)
    pays = {start + u: payload(rng, S) for u in range(count)}
    pays[j] = pay
    salts = {i: rng.randbytes(BC.SALT_BYTES) for i in pays}
    salts[j] = salt
    pdom = domain(rng, start, count)
    ptree = MerkleTree(pdom, {i: hidden(pays[i], salts[i]) for i in pays}, lambda _i: V2)
    values, vat = None, 0
    if bound:
        vcount = rng.randrange(1, (1 << S.DV) + 1)
        vat = rng.randrange(vcount)
        vdom = domain(rng, 0, vcount)
        vals = {i: hidden(payload(rng, S), rng.randbytes(BC.SALT_BYTES)) for i in range(vcount)}
        vals[vat] = hidden(pay, salt)
        vtree = MerkleTree(vdom, vals, lambda _i: V2)
        values = (vtree.commitment.root, vdom.domain_id, vat, merkle_depth(vcount))
    if sketch is not None:
        pass_seed = pass_seed or rng.randbytes(64)
        prefix, signs = sketch(pass_seed, S.ROWS)
    else:
        prefix, signs = rng.randbytes(128), [rng.randbytes(S.BYTES // 8) for _ in range(S.ROWS)]
    plan = sorted(rng.sample([x for x in range(1 << 12) if x != j], length - 1))
    plan.insert(pos, j)
    ys = [rng.randbytes(64) for _ in range(length)]
    ys[pos] = BC.receipt(prefix, j, block, signs)
    leaves = -(-length // PER_LEAF)
    rdom = domain(rng, 0, leaves)
    rsalts = [rng.randbytes(BC.SALT_BYTES) for _ in range(leaves)]
    rows = [b"".join(ys[PER_LEAF * c:PER_LEAF * (c + 1)]) for c in range(leaves)]
    rtree = MerkleTree(rdom, {c: hidden(rows[c], rsalts[c]) for c in range(leaves)}, lambda _c: V2)
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


def cases(rng: random.Random, S, encode, sketch=None) -> list[tuple[str, SimpleNamespace, dict, str | None]]:
    """An appended opening and a registered one, each honest, then with one thing wrong: (name, opening, args, the check
    it fails, None for accepted)."""
    out = []
    for bound in (True, False):
        o = opening(rng, S, bound, encode, sketch=sketch)
        a, S_ = o.args, o.S
        k = o.pos % BC.PER_LEAF
        top = BC._p(S) + rng.randrange(S.C)
        over = top.to_bytes(S.N // 8, "little") + o.block[S.N // 8:]
        wrong_y = o.rows[o.c][:64 * k] + bytes(64) + o.rows[o.c][64 * (k + 1):]
        named = {"honest": (a, None), "tampered block": (dict(a, block=flip(a["block"], 3)), "codec"),
                 "other payload": (dict(a, payload=flip(a["payload"], 1, 4)), "codec"),
                 "word at or above p": (dict(a, block=BC._u16s(over)), "codec"),
                 "receipt leaf altered": (dict(a, leaf=flip(a["leaf"], 32 * k + 2)), "receipt-path"),
                 "another receipt of the leaf altered": (dict(a, leaf=flip(a["leaf"], 32 * ((k + 1) % S_.LEAF))),
                                                         "receipt-path" if S_.LEAF > 1 else "receipt-path"),
                 "receipt leaf's salt": (dict(a, salts=flip(a["salts"], 5)), "receipt-path"),
                 "receipt path": (dict(a, rpath=flip(a["rpath"], 7)), "receipt-path" if a["rdepth"][0] else None),
                 "receipt tree deeper than its path": (dict(a, rdepth=[S.DR + 1]), "receipt-path"),
                 "plan without j at pos": (dict(a, plan=flip(a["plan"], o.pos)), "receipt-path"),
                 "payload's salt": (dict(a, salts=flip(a["salts"], 96 + 5)), "payload-path"),
                 "payload path": (dict(a, ppath=flip(a["ppath"], 9)), "payload-path"),
                 "payload at another rank": (dict(a, prank=[a["prank"][0] ^ 1, 0]), "payload-path"),
                 "other pass's signs": (dict(a, signs=[x ^ 0x5A5A for x in a["signs"]]), "receipt"),
                 "a receipt the tree commits but the block doesn't give": (committed_wrong(o, wrong_y), "receipt")}
        if bound:
            named["wrong coin"] = (dict(a, coin=flip(a["coin"], 7)), "codec")
            named["unbound"] = (dict(a, bound=[0]), "codec")
            named["value path"] = (dict(a, vpath=flip(a["vpath"], 3)), "payload-path" if a["vdepth"][0] else None)
            named["value at another position"] = (dict(a, vrank=[a["vrank"][0] ^ 1, 0]), "payload-path")
        else:
            named["bound"] = (dict(a, bound=[1]), "codec")
            named["value tree unread"] = (dict(a, vroot=[1] * 32, vpath=[2] * len(a["vpath"])), None)
        tag = "appended" if bound else "registered"
        out += [(f"{tag}: {n}", o, x, want) for n, (x, want) in named.items()]
    return out


def toy_cases(S, encode) -> list[tuple[str, SimpleNamespace, dict, str | None]]:
    """The toy unit's cases at ``S`` (`TOY` or `TOY_MASKED`): the openings whose blocks `fixtures/block_check_toy.json`
    holds."""
    return cases(random.Random(10 + S.MASKED), S, encode)


def committed_wrong(o: SimpleNamespace, row: bytes) -> dict:
    """The opening of a pass whose receipt tree commits ``row`` as the sampled leaf: its path holds, its receipt isn't the
    block's."""
    rows = list(o.rows)
    rows[o.c] = row
    tree = MerkleTree(o.rdom, {c: hidden(rows[c], o.rsalts[c]) for c in range(len(rows))}, lambda _c: V2)
    return dict(o.args, leaf=BC._u16s(row), rroot=BC._u16s(tree.commitment.root), rpath=BC._u16s(b"".join(
        tree.open(o.c).path) + bytes(64 * (o.S.DR - len(tree.open(o.c).path)))))
