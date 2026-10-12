"""The remote bandwidth audit's pinned vectors (`vectors.json`): `python remote_bandwidth_vectors.py` rewrites them, and
`test_remote_bandwidth_vectors.py` holds the code to them.

Payload blocks are generated, not stored: payload `j` is 64 pieces of 64 bytes, piece u the first 64 bytes of
`SHA512("verity/remote-bandwidth/test-payload/v0\\0" ‖ u64be j ‖ u32be u)` with its top byte cleared, so every word is
below p (two pieces a word at p1024, one at p512), and payload 0 is all zeros (the weights may be).  The store is under
the default codec, p1024 masked at k = 3.  Every coin and salt comes from `entropy(label)`, SHA-512 in counter mode:
call c returns the first n bytes of `SHA512(label ‖ u64be c) ‖ SHA512(label ‖ u64be c + 1) ‖ …`, the counter running on
across calls (`Entropy`).  The challenger draws coin k and then its salt, k = 0, 1 (window 0's challenge, then its
openings coin); the payload salts are drawn in block order and the receipt leaves' salts pass by pass, leaf by leaf.  The
three passes are one window's chain: pass 0 follows the registration's digest and each later pass the root before it.
A GPU receipter fills its store with the same payloads, encodes them under the pinned codec seed, and checks its blocks,
receipts and roots against these.  The middle pass reads a partial plan out of store order, so its receipt for position
p is block plan[p]'s and its tree has 10 positions.  The main chain is the default registration's: the default receipt
kind, of the block alone.  `variants` holds every other: the same store registered under each other kind and `bind`
(named "kind", or "kind/bind" for receipts that bind the stub tiles' z) with the same nonce, so the same codec seed and
blocks (neither is in the seed's context) under another digest, so other coins' binding, seeds, signs and roots.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
from pathlib import Path

from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts
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
BLOCKS = 21
PLANS = (receipts.Plan.whole(BLOCKS), receipts.Plan(((16, 5), (3, 4), (0, 1)), BLOCKS), receipts.Plan.whole(BLOCKS))
PARAMS = audit.Params(blocks=BLOCKS, deadline_ns=10, windows=1, samples=7, delta=1e-6)
TILES = audit.NoiseTiles(96)
VARIANTS = tuple((k, b) for k in receipts.KINDS for b in (False, True) if (k, b) != (receipts.DEFAULT, False))


def name(kind: str, bind: bool) -> str:
    return f"{kind}/bind" if bind else kind


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
def registered(receipt: str, bind: bool) -> audit.Store:
    """`store()` registered under `receipt` and `bind` with the same nonce: its codec seed, so its blocks, are
    `store()`'s."""
    st = store()
    reg = audit.register(SOURCE, NODE, BLOCKS, st.registration.payload_root, st.registration.codec, receipt, bind=bind,
                         nonce=NONCE)
    assert reg.codec_seed == st.registration.codec_seed
    out = copy.copy(st)
    out.registration = reg
    return out


def chain(st: audit.Store) -> tuple[coins.CoinOpening, list[dict], coins.CoinOpening, coins.Challenger]:
    """Window 0's challenge, its three passes under the registration's receipt kind and `bind` (the stub tiles' z when
    it binds them), its openings coin and the challenger."""
    reg, blocks = st.registration, st.blocks
    tiles = TILES if reg.bind else None
    challenger = coins.Challenger(reg.digest, 0, PARAMS.coins, entropy=Entropy(COIN_LABEL))
    leaf_salts = Entropy(LEAF_SALT_LABEL)
    sketched = reg.receipt == receipts.SKETCH_SHA512

    challenge = challenger.open(0)
    passes, previous = [], reg.digest
    for i, plan in enumerate(PLANS):
        s = receipts.seed(challenge.coin, i, previous)
        receipter = receipts.receipter(reg.receipt, s)
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
            passes[-1] |= {"sign_bits_sha512": _sha(b"".join(receipts.sign_bits(s))), "sigma": sigma}
        previous = root
    return challenge, passes, challenger.open(PARAMS.windows), challenger


def _passes(st: audit.Store, challenge: coins.CoinOpening, passes: list[dict]) -> dict:
    out = {"window": 0, "challenge": challenge.coin.hex(), "passes": passes}
    return out | {"z_bytes": TILES.z_bytes} if st.registration.bind else out


def variant(kind: str, bind: bool) -> dict:
    st = registered(kind, bind)
    reg = st.registration
    challenge, passes, opening, challenger = chain(st)
    return {"registration": {"receipt": reg.receipt, "bind": reg.bind, "codec_seed": reg.codec_seed.hex(),
                             "digest": reg.digest.hex()},
            "coins_root": challenger.commitment.root.hex(), "passes": _passes(st, challenge, passes),
            "openings": {"coin": opening.coin.hex(), "draws": draws(reg, opening)}}


def draws(reg: audit.Registration, opening: coins.CoinOpening) -> list[int]:
    key = derive(opening.coin, audit.OPENINGS_DOMAIN, {"registration": reg.digest, "window": 0})
    return list(key.subset(PARAMS.samples, sum(p.length for p in PLANS), "openings"))


def generate() -> dict:
    st = store()
    payloads, salts, tree, reg, blocks = st.payloads, st.salts, st.payload_tree, st.registration, st.blocks
    key = reg.key
    challenge, passes, opening, challenger = chain(st)
    first = coins.Challenger(reg.digest, 0, PARAMS.coins, entropy=Entropy(COIN_LABEL)).open(0)
    z_short, z_long = bytes(range(100)), bytes((7 * u) % 251 for u in range(codec.BLOCK_BYTES + 1000))
    s, last = reg.codec.field.word_bytes, reg.codec.field.block_words - 1
    return {
        "codec": {"domain": codec.DOMAIN.decode(), "name": reg.codec.name, "seed": reg.codec_seed.hex(),
                  "c": [format(c, "x") for c in key.c], "tau": format(key.tau, "x"), "mask": format(key.mask, "x"),
                  "words": [{"j": j, "w": w, "payload": payloads[j][s * w:s * (w + 1)].hex(),
                             "word": blocks[j][s * w:s * (w + 1)].hex()} for j, w in ((0, 0), (0, last), (1, 0), (20, 5))],
                  "blocks_sha512": _sha(b"".join(blocks))},
        "payload": {"schema": audit.PAYLOAD_SCHEMA, "binding": audit.payload_domain(NODE, BLOCKS).binding.hex(),
                    "leaf_0": {"x": sha512_row_v2_digest(payloads[0], 8 * codec.BLOCK_BYTES).hex(), "salt": salts[0].hex(),
                               "value": receipts.hidden(payloads[0], salts[0]).hex()},
                    "root": tree.commitment.root.hex()},
        "registration": {"source": SOURCE.hex(), "node": NODE.hex(), "blocks": BLOCKS, "nonce": NONCE.hex(),
                         "payload_root": reg.payload_root.hex(), "codec": reg.codec.name, "receipt": reg.receipt,
                         "bind": reg.bind, "codec_seed": reg.codec_seed.hex(), "digest": reg.digest.hex()},
        "coins": {"scheme": coins.SCHEME, "session": 0, "count": PARAMS.coins,
                  "binding": coins.domain(reg.digest, 0, PARAMS.coins).binding.hex(),
                  "coin_0": {"coin": first.coin.hex(), "salt": first.salt.hex(),
                             "value": coins.leaf_value(first.coin, first.salt).hex(), "path": [x.hex() for x in first.path]},
                  "root": challenger.commitment.root.hex()},
        "spread": {"short": {"z": z_short.hex(), "sha512": _sha(receipts.spread(z_short))},
                   "long": {"z_len": len(z_long), "sha512": _sha(receipts.spread(z_long))}},
        "passes": _passes(st, challenge, passes),
        "openings": {"coin": opening.coin.hex(), "index": opening.index, "units": sum(p.length for p in PLANS),
                     "draws": draws(reg, opening)},
        "sketch": {"sign_domain": receipts.SIGN_DOMAIN, "receipt_domain": receipts.SKETCH_RECEIPT_DOMAIN.decode(),
                   "rows": receipts.SKETCH_ROWS},
        "variants": {name(kind, bind): variant(kind, bind) for kind, bind in VARIANTS},
    }


if __name__ == "__main__":
    PATH.write_text(json.dumps(generate(), indent=1) + "\n")
