"""The remote bandwidth audit's pinned vectors (`vectors.json`): `python remote_bandwidth_vectors.py` rewrites them, and
`test_remote_bandwidth_vectors.py` holds the code to them.

Payload blocks are generated, not stored: payload `j` is 64 words, word w the first 64 bytes of
`SHA512("verity/remote-bandwidth/test-payload/v0\\0" ‖ u64be j ‖ u32be w)` with its top byte cleared (below p), and
payload 0 is all zeros (the weights may be).  A GPU receipter fills its store with the same payloads, encodes them
under the pinned codec seed, and checks its blocks, receipts and roots against these.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.core.protocols.remote_bandwidth import audit, codec, receipts
from verity.ml.randomness import derive

PATH = Path(__file__).with_name("vectors.json")
PAYLOAD_DOMAIN = b"verity/remote-bandwidth/test-payload/v0\0"
SOURCE = b"verity remote bandwidth vector source"
NODE = b"verity remote bandwidth vector node"
NONCE = bytes(range(32))
CHALLENGE = bytes(range(32, 64))
BLOCKS = 21


def payload(j: int) -> bytes:
    if j == 0:
        return bytes(codec.BLOCK_BYTES)
    words = (hashlib.sha512(PAYLOAD_DOMAIN + j.to_bytes(8, "big") + w.to_bytes(4, "big")).digest()[:63] + b"\0"
             for w in range(codec.BLOCK_WORDS))
    return b"".join(words)


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def generate() -> dict:
    payloads = [payload(j) for j in range(BLOCKS)]
    tree = audit.payload_tree(NODE, payloads)
    reg = audit.register(SOURCE, NODE, BLOCKS, tree.commitment.root, nonce=NONCE)
    key = reg.key
    blocks = [codec.encode(m, j, key) for j, m in enumerate(payloads)]
    tiles = audit.NoiseTiles(96)

    chain, previous = [], reg.digest
    for i in range(3):
        s = receipts.seed(CHALLENGE, i, previous)
        receipter = receipts.Receipter(s)
        ys = [receipter(j, e, tiles.output(s, j)) for j, e in enumerate(blocks)]
        root = receipts.tree(s, ys).commitment.root
        chain.append({"i": i, "previous": previous.hex(), "seed": s.hex(), "prefix": receipts.prefix(s).hex(),
                      "z_1": tiles.output(s, 1).hex(), "y_0": ys[0].hex(), "y_1": ys[1].hex(), "y_20": ys[20].hex(),
                      "receipts_sha512": _sha(b"".join(ys)), "root": root.hex()})
        previous = root

    z_short, z_long = bytes(range(100)), bytes((7 * u) % 251 for u in range(codec.BLOCK_BYTES + 1000))
    window = derive(SOURCE, audit.WINDOW_DOMAIN, {"registration": reg.digest, "window": 0, "nonce": NONCE})
    return {
        "codec": {"domain": codec.DOMAIN.decode(), "seed": reg.codec_seed.hex(), "rounds": reg.rounds,
                  "c": [format(c, "x") for c in key.c], "tau": format(key.tau, "x"),
                  "words": [{"j": j, "w": w, "payload": payloads[j][64 * w:64 * (w + 1)].hex(),
                             "word": blocks[j][64 * w:64 * (w + 1)].hex()} for j, w in ((0, 0), (0, 63), (1, 0), (20, 5))],
                  "blocks_sha512": _sha(b"".join(blocks))},
        "registration": {"source": SOURCE.hex(), "node": NODE.hex(), "blocks": BLOCKS, "nonce": NONCE.hex(),
                         "payload_root": reg.payload_root.hex(), "codec_seed": reg.codec_seed.hex(),
                         "digest": reg.digest.hex()},
        "window": {"nonce": NONCE.hex(), "challenge": window.stream("challenge", receipts.CHALLENGE_BYTES).hex(),
                   "draws": list(window.subset(7, 3 * BLOCKS, "openings"))},
        "spread": {"short": {"z": z_short.hex(), "sha512": _sha(receipts.spread(z_short))},
                   "long": {"z_len": len(z_long), "sha512": _sha(receipts.spread(z_long))}},
        "chain": {"challenge": CHALLENGE.hex(), "z_bytes": tiles.z_bytes, "passes": chain},
    }


if __name__ == "__main__":
    PATH.write_text(json.dumps(generate(), indent=1) + "\n")
