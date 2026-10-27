"""Conformance vectors of every registered scheme (`verity.core.protocols.pous.schemes`), one file per name under
`fixtures/schemes/`: the certificate at the registered point, and a small instance of the same scheme (its parameters,
the digest of each encoded segment, and `vk`'s root) through `verity.core.protocols.pous.protocol`.  `point_vector` names the
whole-segment vector at the registered point in `fixtures/vectors.json`, which `VERITY_POUS_SLOW=1` checks through the
interface.

A scheme whose encode is slow in Python (`SLOW_ENCODE`: P2's two roots per label, about 2.4 s at 16,448 bits) also pins
its small instance's labels; the default tests decode those, and `VERITY_POUS_SLOW=1` re-encodes.

`python tests/pous_scheme_vectors.py` rewrites them (about 5 s).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pous_vectors import _bytes  # noqa: E402

from verity.core.protocols.pous import band, dense, protocol, schemes  # noqa: E402
from verity.core.protocols.pous.params import Params  # noqa: E402
from verity.core.protocols.pous.schemes.p2 import CHACHA8, P2Params  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "schemes"
SMALL = {"band-chain/d12/v1": band.BandParams(16, 64, 12, 2), "dense-chain/v1": dense.DenseParams(8, 64, 2),
         "p3/v1": Params(8, 64, 3, 2), "p2-16448/v4": P2Params(16448, 21065, 2, 1),
         "p2-16448/v2": P2Params(16448, 21065, 2, 1, CHACHA8)}
POINT_VECTOR = {"band-chain/d12/v1": "band/64kb/segment", "dense-chain/v1": None, "p3/v1": "segment_64kb",
                "p2-16448/v4": None, "p2-16448/v2": None}
SLOW_ENCODE = {"p2-16448/v4", "p2-16448/v2"}


def path(name: str) -> Path:
    return FIXTURES / (name.replace("/", "-") + ".json")


def small_inputs(name: str) -> tuple[bytes, bytes]:
    """The small instance's `W` and salt."""
    return _bytes(f"scheme/{name}/w", SMALL[name].w_bytes), _bytes(f"scheme/{name}/salt", 24)


def vector(name: str, blocks: list[list[bytes]] | None = None) -> dict:
    """The scheme's vector; with `blocks` (a pinned encoding), everything but the encode is recomputed from them."""
    cert = schemes.get(name).certificate
    small = schemes.get(name, params=SMALL[name])
    w, salt = small_inputs(name)
    c = protocol.encode(small, w, salt) if blocks is None else blocks
    if protocol.decode(small, salt, c) != w:
        raise ValueError(f"{name}: the small instance does not decode to its W")
    vk = protocol.verifier_key(small, protocol.commit(small, w), salt, mode=protocol.PUBLIC, encoded=c)
    out = {"name": name, "certificate": None if cert is None else cert.to_json(),
           "small": {"params": small.params.manifest(), "salt": salt.hex(),
                     "segments": [hashlib.sha256(b"".join(seg)).hexdigest() for seg in c], "root": vk.root.hex()},
           "point_vector": POINT_VECTOR[name]}
    if name in SLOW_ENCODE:
        out["small"]["blocks"] = [[b.hex() for b in seg] for seg in c]
    return out


def pinned_blocks(pinned: dict) -> list[list[bytes]] | None:
    """A vector's pinned labels, when it has them."""
    blocks = pinned["small"].get("blocks")
    return None if blocks is None else [[bytes.fromhex(b) for b in seg] for seg in blocks]


if __name__ == "__main__":
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name in sorted(schemes.REGISTRY):
        path(name).write_text(json.dumps(vector(name), indent=1, sort_keys=True) + "\n")
        print(f"wrote {path(name)}")
