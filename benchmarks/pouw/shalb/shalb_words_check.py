"""The words committer's outputs (sha512_words_gpu.cu) against the core's reference, bit for bit.

    python3 shalb_words_check.py domain DOMAIN_ID_FILE     # writes the domain id the GPU program reads
    python3 shalb_words_check.py check DOMAIN_ID_FILE OUT  # checks OUT: each chunk's digest (hashlib), leaf and node

The domain is a range-u64 SHA-512 domain of 2^20 words chunks; a chunk at position pos has rank pos. A chunk's digest
is SHA-512(P(ℓ) ‖ row) (sha512/row/v2), its leaf `CommitmentDomain.leaf(pos, pos, "sha512/row/v2/l1-w", digest)` and
the level-0 node over leaves 2i and 2i + 1 `CommitmentDomain.node(0, pos/2, left, right)`, all from
verity.ml.commitments. Run from the source tree's root.
"""
from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from verity.ml.commitments.indexed import RangeU64Domain  # noqa: E402
from verity.ml.commitments.merkle import CommitmentDomain  # noqa: E402

SCHEMA = "sha512/row/v2/l1-w"
DOMAIN = CommitmentDomain(binding=bytes(range(64)), owner=0, positions=RangeU64Domain(0, 1 << 20), hash="sha512")


def words(raw: bytes) -> bytes:
    """u64 words as the GPU program wrote them (host order) to the big-endian bytes they stand for."""
    return b"".join(w.to_bytes(8, "big") for w in struct.unpack(f"<{len(raw) // 8}Q", raw))


def main() -> int:
    mode, did_path = sys.argv[1], Path(sys.argv[2])
    if mode == "domain":
        did_path.write_bytes(DOMAIN.domain_id)
        return 0
    data = Path(sys.argv[3]).read_bytes()
    n, pos0, row_bytes, _ = struct.unpack_from("<4I", data)
    off = 16
    rows = [words(data[off + i * row_bytes:off + (i + 1) * row_bytes]) for i in range(n)]
    off += n * row_bytes
    prefix = (b"verity/sha512-row/v2\0" + (8 * row_bytes).to_bytes(8, "big")).ljust(128, b"\0")
    digests = [hashlib.sha512(prefix + r).digest() for r in rows]
    leaves = [DOMAIN.leaf(pos0 + i, pos0 + i, SCHEMA, d) for i, d in enumerate(digests)]
    nodes = [DOMAIN.node(0, (pos0 + 2 * i) // 2, leaves[2 * i], leaves[2 * i + 1]) for i in range(n // 2)]
    bad = 0
    for kind in ("minimal", "served"):
        got = {}
        for name, count in (("digest", n), ("leaf", n), ("node", n // 2)):
            got[name] = [words(data[off + 64 * i:off + 64 * (i + 1)]) for i in range(count)]
            off += 64 * count
        wrong = {name: sum(a != b for a, b in zip(got[name], ref)) for name, ref in
                 (("digest", digests), ("leaf", leaves), ("node", nodes))}
        bad += sum(wrong.values())
        print(f"{kind}: {n} chunk digests, {n} leaves and {n // 2} nodes against verity.ml.commitments and hashlib; "
              f"differ: {wrong}")
    print(f"domain id {DOMAIN.domain_id.hex()[:16]}…, positions {pos0}..{pos0 + n - 1}, {row_bytes} B chunks; "
          f"{'all equal' if not bad else f'{bad} differ'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
