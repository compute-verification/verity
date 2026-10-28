"""A synthetic PoUS run case of about SIZE bytes, for timing `pous-run-protocol --test serve` at a served case's size.

    big_case.py SIZE OUT [--workers N]

The case is an honest period at the smallest width, `(w, c0) = (64, 189)`, so its bytes are mostly the memory
challenger's table and `C*` (about 304 bytes a block, both hex twice over): setup of a zero store encoded in `N`
processes (`period.setup_store`), then the memory challenger's session over it under a test signer
(`tests/pous_test_signer.py`, whose runs only `--test` accepts). The case line, `period.case`'s canonical JSON, is
written to OUT a piece at a time, so the generator holds `C*` and its table but never the line. A test signer's case
is synthetic, so nothing in it is evidence of a real run. Prints the line's size and the times taken as JSON.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

from verity.ml.commitments.identity import canonical_json_bytes
from verity.ml.randomness import derive
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period
from verity.core.protocols.pous import run as R

ROOT = Path(__file__).resolve().parents[2]
W, C0 = 64, 189
PER_BLOCK = 4 * (4 + W // 8) + 4 * 64
"""A block's bytes in the line: its framed `C*` entry and its 64-byte digest, each hex in the evidence, hex again."""
CHUNK = 1 << 24


def _signer_module():
    spec = importlib.util.spec_from_file_location(
        "pous_test_signer", ROOT / "verity/core/protocols/pous/tests/pous_test_signer.py")
    m = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _hex_twice(out, data: bytes | str) -> None:
    """`data`'s hex, as `period.case` writes the evidence's bytes (the evidence is hex of JSON holding hex)."""
    data = data.encode() if isinstance(data, str) else data
    for i in range(0, len(data), CHUNK):
        out.write(data[i:i + CHUNK].hex().encode())


def write_case(out_path: Path, size: int, workers: int) -> dict:
    ts = _signer_module()
    times: dict[str, float] = {}
    t = time.perf_counter()
    point = period.point_for((size // PER_BLOCK) * ((W - 16) // 8), W, C0)
    q = point.params
    coins = b"big-case coins"
    held = period.setup_store(b"", coins, point, workers=workers)
    times["setup_s"] = time.perf_counter() - t

    t = time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        signer = ts.TestSigner(b"big case", height=1, directory=d)
        sid = derive(coins, period.SESSION_DOMAIN, {"node": "big-case", "setup": held.published.root}).stream("session", 32)
        tree = mc.CoinTree.draw(coins, sid, q.blocks, point.k)
        c = mc.MemoryChallenger(tree, signer, log=ts.session_log(d), limit_ns=point.limit_ns, block_bytes=q.block_bytes,
                                binding=held.published.root)
        session = R.ChallengePhase(c.run(period.honest(held.blocks)), c.commitment, c.openings())
        key = signer.public_key
    times["session_s"] = time.perf_counter() - t

    t = time.perf_counter()
    params = {"point": period.point_json(point), "session_id": sid.hex(), "trusted": [key.hex()]}
    setup_of = held.setup_case
    with open(out_path, "wb") as out:
        out.write(b'{"commitment":"' + canonical_json_bytes(held.published.to_json()).hex().encode() + b'","evidence":"')
        _hex_twice(out, b'{"digests":"')
        for i in range(0, len(held.digests), CHUNK // 2):
            _hex_twice(out, held.digests[i:i + CHUNK // 2].hex())
        _hex_twice(out, b'","session":' + canonical_json_bytes(session.to_json()) + b',"setup":{"commitment":"')
        _hex_twice(out, setup_of["commitment"])
        _hex_twice(out, b'","evidence":"')
        _hex_twice(out, setup_of["evidence"])
        _hex_twice(out, b'"}}')
        out.write(b'","params":' + canonical_json_bytes(params) + b"}\n")
    times["write_s"] = time.perf_counter() - t
    return {"bytes": out_path.stat().st_size, "blocks": q.blocks, "segments": q.segments, **times}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("size", type=int, help="the case line's approximate size in bytes")
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=1, help="processes encoding C*")
    a = ap.parse_args(argv)
    print(json.dumps(write_case(a.out, a.size, a.workers), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
