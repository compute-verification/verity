"""The remote bandwidth audit's pinned vectors (`vectors.json`): `python remote_bandwidth_vectors.py` rewrites them, and
`test_remote_bandwidth_vectors.py` holds the code to them.

Payload blocks are generated, not stored: payload `j` is 64 pieces of 64 bytes, piece u the first 64 bytes of
`SHA512("verity/remote-bandwidth/test-payload/v0\\0" ‖ u64be j ‖ u32be u)` with its top byte cleared, so every word is
below p (two pieces a word at p1024, one at p512), and payload 0 is all zeros (the weights may be).  The store is under
the default codec, p1024 masked at k = 3.  Every coin and salt comes from `entropy(label)`, SHA-512 in counter mode:
call c returns the first n bytes of `SHA512(label ‖ u64be c) ‖ SHA512(label ‖ u64be c + 1) ‖ …`, the counter running on
across calls (`Entropy`).  The challenger (`coins.local`, session 0's epoch of the one challenger) draws each coin of its
layout and then its salt, in layout order: window 0's challenge, its openings coin, then the append coins; a coin is 64
bytes and the audit's is its first 32, so one SHA-512 block a coin.  `coins.window` is R, the session's window over its
registration, which σ_0 is opened after; the commitment names no registration, so every variant's coins are the same
coins under the same root, each opened after its own R.  The payload salts are drawn in block order and the receipt
leaves' salts pass by pass, leaf by leaf.  The
three passes are one window's chain: pass 0 follows the registration's digest and each later pass the root before it.
A GPU receipter fills its store with the same payloads, encodes them under the pinned codec seed, and checks its blocks,
receipts and roots against these.  `codec.bound` holds blocks 0 and 20 encoded under their bound keys, as blocks
written after the codec seed are: d_j, b_j, the first word and the block's SHA-512; `codec.coin` the same blocks under
the coin form of their bound keys, an append's coin c_a being the first 32 bytes of `SHA512(APPEND_COIN_LABEL)`, with
b_j(c_a) in place of b_j, so a device that takes b_j as its input is checked on both forms.  The middle pass reads a
partial plan out of store order, so its receipt for position p is block plan[p]'s and its tree has 10 positions.  The main
chain is the default registration's: sketch-sha512, the default receipt kind, at 16 rows, of the block alone.
`variants` holds every other: the same store registered under each other kind and `bind`, and sketch-sha512 at 64 rows
(named "kind", "kind/bind" for receipts that bind the stub tiles' z, "sketch-sha512/rows64"), with the same nonce, so
the same codec seed and blocks (none of them is in the seed's context) under another digest, so another window R, and
other seeds, signs and roots.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
from pathlib import Path

from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts
from verity.core.protocols.sampled_proofs.challenger import live_domain, live_leaf
from verity.ml.commitments.rowleaf import sha512_row_v2_digest
from verity.ml.randomness import derive

PATH = Path(__file__).with_name("vectors.json")
PAYLOAD_DOMAIN = b"verity/remote-bandwidth/test-payload/v0\0"
SOURCE = b"verity remote bandwidth vector source"
NODE = b"verity remote bandwidth vector node"
NONCE = bytes(range(32))
COIN_LABEL = b"verity remote bandwidth vector coins"
PAYLOAD_SALT_LABEL = b"verity remote bandwidth vector payload salts"
LEAF_SALT_LABEL = b"verity remote bandwidth vector receipt salts"
APPEND_COIN_LABEL = b"verity remote bandwidth vector append coin"
APPEND_COIN = hashlib.sha512(APPEND_COIN_LABEL).digest()[:codec.COIN_BYTES]
BLOCKS = 21
BOUND_BLOCKS = (0, 20)
PLANS = (receipts.Plan.whole(BLOCKS), receipts.Plan(((16, 5), (3, 4), (0, 1)), BLOCKS), receipts.Plan.whole(BLOCKS))
PARAMS = audit.Params(blocks=BLOCKS, deadline_ns=10, windows=1, samples=7, delta=1e-6, window_ns=10)
TILES = audit.NoiseTiles(96)
DEFAULT = (receipts.DEFAULT, False, receipts.SKETCH_ROWS)
VARIANTS = (*((k, b, receipts.sketch_rows(k)) for k in receipts.KINDS for b in (False, True)
              if (k, b, receipts.sketch_rows(k)) != DEFAULT), (receipts.SKETCH_SHA512, False, 64))


def name(kind: str, bind: bool, rows: int) -> str:
    return kind + ("/bind" if bind else "") + ("" if rows == receipts.sketch_rows(kind) else f"/rows{rows}")


class Entropy:
    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


def payload(j: int) -> bytes:
    if j == 0:
        return bytes(codec.BLOCK_BYTES)
    pieces = (hashlib.sha512(PAYLOAD_DOMAIN + j.to_bytes(8, "big") + u.to_bytes(4, "big")).digest()[:63] + b"\0"
              for u in range(codec.BLOCK_BYTES // 64))
    return b"".join(pieces)


def payload_salts() -> list[bytes]:
    e = Entropy(PAYLOAD_SALT_LABEL)
    return [e(receipts.SALT_BYTES) for _ in range(BLOCKS)]


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


@functools.cache
def store() -> audit.Store:
    """The vectors' store, encoded once a process: its 21 blocks take about 6 s under the default codec."""
    payloads, salts = [payload(j) for j in range(BLOCKS)], payload_salts()
    root = audit.payload_tree(NODE, payloads, salts).commitment.root
    return audit.Store(audit.register(SOURCE, NODE, BLOCKS, root, nonce=NONCE), payloads, salts)


@functools.cache
def registered(receipt: str, bind: bool, rows: int) -> audit.Store:
    """`store()` registered under `receipt`, `bind` and `rows` with the same nonce: its codec seed, so its blocks, are
    `store()`'s."""
    st = store()
    reg = audit.register(SOURCE, NODE, BLOCKS, st.registration.payload_root, st.registration.codec, receipt, bind=bind,
                         rows=rows, nonce=NONCE)
    assert reg.codec_seed == st.registration.codec_seed
    out = copy.copy(st)
    out.registration = reg
    return out


def chain(st: audit.Store) -> tuple[coins.LiveOpening, list[dict], coins.LiveOpening, coins.SessionCoins]:
    """Window 0's challenge, opened after R, its three passes under the registration's receipt kind and `bind` (the
    stub tiles' z when it binds them), its openings coin, opened after the last pass's root, and the session's coins."""
    reg, blocks = st.registration, st.blocks
    tiles = TILES if reg.bind else None
    challenger = coins.local(reg.digest, 0, PARAMS.windows, Entropy(COIN_LABEL))
    leaf_salts = Entropy(LEAF_SALT_LABEL)
    sketched = reg.receipt == receipts.SKETCH_SHA512

    challenge = challenger.challenge(0, challenger.window)
    passes, previous = [], reg.digest
    for i, plan in enumerate(PLANS):
        s = receipts.seed(coins.coin(challenge), i, previous)
        receipter = receipts.receipter(reg.receipt, s, reg.rows)
        zs = {j: None if tiles is None else tiles.output(s, j) for j in plan}
        ys = [receipter(j, blocks[j], zs[j]) for j in plan]
        ss = [leaf_salts(receipts.SALT_BYTES) for _ in range(receipts.leaves(plan.length))]
        leaf0 = b"".join(ys[:receipts.PER_LEAF])
        root = receipts.tree(s, plan, ys, ss).commitment.root
        ends = (0, 1, plan.length - 1)
        passes.append({"i": i, "previous": previous.hex(), "seed": s.hex(),
                       "prefix": (receipts.sketch_prefix if sketched else receipts.prefix)(s).hex(),
                       "plan": [list(r) for r in plan.runs], "plan_encoding": plan.encode().hex(),
                       "plan_digest": plan.digest.hex(), "binding": receipts.binding(s, plan).hex(),
                       "y": {str(p): ys[p].hex() for p in ends}, "receipts_sha512": _sha(b"".join(ys)),
                       "leaf_0": {"x": sha512_row_v2_digest(leaf0, 8 * len(leaf0)).hex(), "salt": ss[0].hex(),
                                  "value": receipts.hidden(leaf0, ss[0]).hex()},
                       "root": root.hex()})
        if tiles is not None:
            passes[-1]["z_1"] = tiles.output(s, 1).hex()
        if sketched:
            sigma = {str(p): receipter.sketch(blocks[j], zs[j]).hex() for p, j in zip(ends, map(plan.block, ends))}
            passes[-1] |= {"sign_bits_sha512": _sha(b"".join(receipts.sign_bits(s, reg.rows))), "sigma": sigma}
        previous = root
    return challenge, passes, challenger.openings(0, previous), challenger


def _passes(st: audit.Store, challenge: coins.LiveOpening, passes: list[dict]) -> dict:
    out = {"window": 0, "challenge": coins.coin(challenge).hex(), "passes": passes}
    return out | {"z_bytes": TILES.z_bytes} if st.registration.bind else out


def variant(kind: str, bind: bool, rows: int) -> dict:
    st = registered(kind, bind, rows)
    reg = st.registration
    challenge, passes, opening, challenger = chain(st)
    return {"registration": {"receipt": reg.receipt, "bind": reg.bind, "layout": reg.layout, "rows": reg.rows,
                             "codec_seed": reg.codec_seed.hex(), "digest": reg.digest.hex()},
            "coins_root": challenger.commitment.root.hex(), "window": challenger.window.hex(),
            "passes": _passes(st, challenge, passes),
            "openings": {"coin": coins.coin(opening).hex(), "draws": draws(reg, opening)}}


def draws(reg: audit.Registration, opening: coins.LiveOpening) -> list[int]:
    key = derive(coins.coin(opening), audit.OPENINGS_DOMAIN, {"registration": reg.digest, "window": 0})
    return list(key.subset(PARAMS.samples, sum(p.length for p in PLANS), "openings"))


def bound(payload: bytes, j: int, key: codec.Key, coin: bytes | None = None) -> dict:
    """Block `j` encoded under its bound key, as a block written after the codec seed is, under its append's `coin` if
    given."""
    d, s = codec.digest(payload), key.codec.field.word_bytes
    block = codec.encode(payload, j, key, bound=True, coin=coin)
    return {"j": j, "digest": d.hex(), "b": format(codec.binding(d, j, key, coin), "x"), "word_0": block[:s].hex(),
            "block_sha512": _sha(block)}


def generate() -> dict:
    st = store()
    payloads, salts, tree, reg, blocks = st.payloads, st.salts, st.payload_tree, st.registration, st.blocks
    key = reg.key
    challenge, passes, opening, challenger = chain(st)
    c = challenger.commitment
    z_short, z_long = bytes(range(100)), bytes((7 * u) % 251 for u in range(codec.BLOCK_BYTES + 1000))
    s, last = reg.codec.field.word_bytes, reg.codec.field.block_words - 1
    return {
        "codec": {"domain": codec.DOMAIN.decode(), "name": reg.codec.name, "seed": reg.codec_seed.hex(),
                  "c": [format(c, "x") for c in key.c], "tau": format(key.tau, "x"), "mask": format(key.mask, "x"),
                  "words": [{"j": j, "w": w, "payload": payloads[j][s * w:s * (w + 1)].hex(),
                             "word": blocks[j][s * w:s * (w + 1)].hex()} for j, w in ((0, 0), (0, last), (1, 0), (20, 5))],
                  "blocks_sha512": _sha(b"".join(blocks)),
                  "bound": [bound(payloads[j], j, key) for j in BOUND_BLOCKS],
                  "coin": {"c_a": APPEND_COIN.hex(), "index": codec.COIN_INDEX,
                           "bound": [bound(payloads[j], j, key, APPEND_COIN) for j in BOUND_BLOCKS]}},
        "payload": {"schema": audit.PAYLOAD_SCHEMA, "binding": audit.payload_domain(NODE, BLOCKS).binding.hex(),
                    "leaf_0": {"x": sha512_row_v2_digest(payloads[0], 8 * codec.BLOCK_BYTES).hex(), "salt": salts[0].hex(),
                               "value": receipts.hidden(payloads[0], salts[0]).hex()},
                    "root": tree.commitment.root.hex()},
        "registration": {"source": SOURCE.hex(), "node": NODE.hex(), "blocks": BLOCKS, "nonce": NONCE.hex(),
                         "payload_root": reg.payload_root.hex(), "codec": reg.codec.name, "receipt": reg.receipt,
                         "bind": reg.bind, "layout": reg.layout, "rows": reg.rows, "codec_seed": reg.codec_seed.hex(),
                         "digest": reg.digest.hex()},
        "coins": {"scheme": c.scheme, "epoch": c.epoch, "families": [list(f) for f in c.families],
                  "binding": live_domain(c.scheme, c.epoch, c.families).binding.hex(), "window": challenger.window.hex(),
                  "coin_0": {"label": challenge.label, "index": challenge.index, "after": challenge.after.hex(),
                             "coin": challenge.coin.hex(), "salt": challenge.salt.hex(),
                             "value": live_leaf(c.scheme, challenge.coin, challenge.salt).hex(),
                             "path": [x.hex() for x in challenge.path]},
                  "root": c.root.hex()},
        "spread": {"short": {"z": z_short.hex(), "sha512": _sha(receipts.spread(z_short))},
                   "long": {"z_len": len(z_long), "sha512": _sha(receipts.spread(z_long))}},
        "passes": _passes(st, challenge, passes),
        "openings": {"coin": coins.coin(opening).hex(), "label": opening.label, "index": opening.index,
                     "position": c.position(opening.label, opening.index), "after": opening.after.hex(),
                     "units": sum(p.length for p in PLANS), "draws": draws(reg, opening)},
        "sketch": {"sign_domain": receipts.SIGN_DOMAIN, "receipt_domain": receipts.SKETCH_RECEIPT_DOMAIN.decode(),
                   "rows": receipts.SKETCH_ROWS, "rows_allowed": list(receipts.ROWS)},
        "variants": {name(*v): variant(*v) for v in VARIANTS},
    }


if __name__ == "__main__":
    PATH.write_text(json.dumps(generate(), indent=1) + "\n")
