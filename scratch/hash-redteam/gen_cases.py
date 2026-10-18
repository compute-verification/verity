"""Served-Python reference cases for RowProbe.lean (hash-redteam scratch, never for a PR).

hm96-sha512 row trees from `audit.commit_rows`, at counts that are and aren't powers of two, rows of byte lengths around
the 1024-byte segment boundary (0 included), every rank opened with `MerkleTree.open` and checked by `verify_opening`.
"""

import hashlib
import json
import sys

from verity.core.protocols.pouw.audit import HM96, HM96_KEY, ROW_SCHEMA, commit_rows, rows_domain
from verity.ml.commitments import VerificationLimits, verify_opening

LENS = [0, 1, 127, 128, 1023, 1024, 1025, 2048, 2049, 3073]
COUNTS = [0, 1, 2, 3, 5, 6, 7, 8, 9]


def rows_for(count: int, t: int) -> list[bytes]:
    return [hashlib.shake_256(b"rt-row|%d|%d" % (t, i)).digest(LENS[(t + i) % len(LENS)]) for i in range(count)]


def main() -> None:
    trees = []
    for t, count in enumerate(COUNTS):
        binding = hashlib.sha512(b"rt-binding|%d" % t).digest()
        owner = t
        key = hashlib.sha256(b"rt-salt-key|%d" % t).digest()
        rows = rows_for(count, t)
        tree = commit_rows(binding, owner, rows, HM96, key, nonce=t)
        dom = rows_domain(binding, owner, count, HM96)
        openings = []
        for rank in range(count):
            op = tree.open(rank)
            assert verify_opening(dom, tree.commitment, op, ROW_SCHEMA, VerificationLimits())
            openings.append({"rank": rank, "row": op.value[:-192].hex(), "salt": op.value[-192:].hex(),
                             "path": [p.hex() for p in op.path]})
        trees.append({"count": count, "binding": binding.hex(), "owner": owner, "salt_key": key.hex(), "nonce": t,
                      "domain_id": dom.domain_id.hex(), "root": tree.commitment.root.hex(),
                      "empty_root": dom.empty_root().hex(), "rows": [r.hex() for r in rows], "openings": openings})
    json.dump({"key": HM96_KEY.hex(), "trees": trees}, open(sys.argv[1], "w"))


if __name__ == "__main__":
    main()
