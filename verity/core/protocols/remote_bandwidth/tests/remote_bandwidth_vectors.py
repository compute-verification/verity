"""The remote bandwidth audit's pinned vectors (`vectors.json`): `python remote_bandwidth_vectors.py` rewrites them, and
`test_remote_bandwidth_vectors.py` holds the code to them.

Payload blocks are generated, not stored: payload `j` is 64 words, word w the first 64 bytes of
`SHA512("verity/remote-bandwidth/test-payload/v0\\0" ‖ u64be j ‖ u32be w)` with its top byte cleared (below p), and
payload 0 is all zeros (the weights may be).  Every coin and salt comes from `entropy(label)`, SHA-512 in counter mode:
call c returns the first n bytes of `SHA512(label ‖ u64be c) ‖ SHA512(label ‖ u64be c + 1) ‖ …`, the counter running on
across calls (`Entropy`).  The challenger draws coin k and then its salt, k = 0, 1 (window 0's challenge, then its
openings coin); the payload salts are drawn in block order and the receipt leaves' salts pass by pass, leaf by leaf.  The
three passes are one window's chain: pass 0 follows the registration's digest and each later pass the root before it.
A GPU receipter fills its store with the same payloads, encodes them under the pinned codec seed, and checks its blocks,
receipts and roots against these.  The middle pass reads a partial plan out of store order, so its receipt for position
p is block plan[p]'s and its tree has 10 positions.
"""

from __future__ import annotations

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
    words = (hashlib.sha512(PAYLOAD_DOMAIN + j.to_bytes(8, "big") + w.to_bytes(4, "big")).digest()[:63] + b"\0"
             for w in range(codec.BLOCK_WORDS))
    return b"".join(words)


def payload_salts() -> list[bytes]:
    e = Entropy(PAYLOAD_SALT_LABEL)
    return [e(receipts.SALT_BYTES) for _ in range(BLOCKS)]


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def generate() -> dict:
    payloads, salts = [payload(j) for j in range(BLOCKS)], payload_salts()
    tree = audit.payload_tree(NODE, payloads, salts)
    reg = audit.register(SOURCE, NODE, BLOCKS, tree.commitment.root, nonce=NONCE)
    key = reg.key
    blocks = [codec.encode(m, j, key) for j, m in enumerate(payloads)]
    tiles = audit.NoiseTiles(96)
    challenger = coins.Challenger(reg.digest, 0, PARAMS.coins, entropy=Entropy(COIN_LABEL))
    leaf_salts = Entropy(LEAF_SALT_LABEL)

    challenge = challenger.open(0)
    passes, previous = [], reg.digest
    for i, plan in enumerate(PLANS):
        s = receipts.seed(challenge.coin, i, previous)
        receipter = receipts.Receipter(s)
        ys = [receipter(j, blocks[j], tiles.output(s, j)) for j in plan]
        ss = [leaf_salts(receipts.SALT_BYTES) for _ in range(receipts.leaves(plan.length))]
        leaf0 = b"".join(ys[:receipts.PER_LEAF])
        root = receipts.tree(s, plan, ys, ss).commitment.root
        passes.append({"i": i, "previous": previous.hex(), "seed": s.hex(), "prefix": receipts.prefix(s).hex(),
                       "plan": [list(r) for r in plan.runs], "plan_encoding": plan.encode().hex(),
                       "plan_digest": plan.digest.hex(), "binding": receipts.binding(s, plan).hex(),
                       "z_1": tiles.output(s, 1).hex(), "y": {str(p): ys[p].hex() for p in (0, 1, plan.length - 1)},
                       "receipts_sha512": _sha(b"".join(ys)),
                       "leaf_0": {"x": sha512_row_v2_digest(leaf0, 8 * len(leaf0)).hex(), "salt": ss[0].hex(),
                                  "value": receipts.hidden(leaf0, ss[0]).hex()},
                       "root": root.hex()})
        previous = root

    opening = challenger.open(PARAMS.windows)
    draw_key = derive(opening.coin, audit.OPENINGS_DOMAIN, {"registration": reg.digest, "window": 0})
    first = coins.Challenger(reg.digest, 0, PARAMS.coins, entropy=Entropy(COIN_LABEL)).open(0)
    z_short, z_long = bytes(range(100)), bytes((7 * u) % 251 for u in range(codec.BLOCK_BYTES + 1000))
    return {
        "codec": {"domain": codec.DOMAIN.decode(), "seed": reg.codec_seed.hex(), "rounds": reg.rounds,
                  "c": [format(c, "x") for c in key.c], "tau": format(key.tau, "x"),
                  "words": [{"j": j, "w": w, "payload": payloads[j][64 * w:64 * (w + 1)].hex(),
                             "word": blocks[j][64 * w:64 * (w + 1)].hex()} for j, w in ((0, 0), (0, 63), (1, 0), (20, 5))],
                  "blocks_sha512": _sha(b"".join(blocks))},
        "payload": {"schema": audit.PAYLOAD_SCHEMA, "binding": audit.payload_domain(NODE, BLOCKS).binding.hex(),
                    "leaf_0": {"x": sha512_row_v2_digest(payloads[0], 8 * codec.BLOCK_BYTES).hex(), "salt": salts[0].hex(),
                               "value": receipts.hidden(payloads[0], salts[0]).hex()},
                    "root": tree.commitment.root.hex()},
        "registration": {"source": SOURCE.hex(), "node": NODE.hex(), "blocks": BLOCKS, "nonce": NONCE.hex(),
                         "payload_root": reg.payload_root.hex(), "codec_seed": reg.codec_seed.hex(),
                         "digest": reg.digest.hex()},
        "coins": {"scheme": coins.SCHEME, "session": 0, "count": PARAMS.coins,
                  "binding": coins.domain(reg.digest, 0, PARAMS.coins).binding.hex(),
                  "coin_0": {"coin": first.coin.hex(), "salt": first.salt.hex(),
                             "value": coins.leaf_value(first.coin, first.salt).hex(), "path": [x.hex() for x in first.path]},
                  "root": challenger.commitment.root.hex()},
        "spread": {"short": {"z": z_short.hex(), "sha512": _sha(receipts.spread(z_short))},
                   "long": {"z_len": len(z_long), "sha512": _sha(receipts.spread(z_long))}},
        "passes": {"window": 0, "challenge": challenge.coin.hex(), "z_bytes": tiles.z_bytes, "passes": passes},
        "openings": {"coin": opening.coin.hex(), "index": opening.index, "units": sum(p.length for p in PLANS),
                     "draws": list(draw_key.subset(PARAMS.samples, sum(p.length for p in PLANS), "openings"))},
    }


if __name__ == "__main__":
    PATH.write_text(json.dumps(generate(), indent=1) + "\n")
