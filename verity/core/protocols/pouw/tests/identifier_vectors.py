"""Pinned vectors for the proof's identifier v1 (`verity.core.protocols.pouw.identifier`): ``uv run python
verity/core/protocols/pouw/tests/identifier_vectors.py`` rewrites ``tests/vectors/identifier_v1.json``.  A change to that file is a change
to the identifier, which C-Flock's verifier compares byte for byte, and needs a reason in the PR.

Inputs are derived, then pinned with each case: weight w's root in case c is SHA-256("verity/pouw/identifier-vectors/v1", c,
"weight", w), and the salt is `Epoch.unverified`'s from the beacon seed SHA-256(…, c, "beacon").  Every case's matmuls are in
its scheme's domain.  A case pins its inputs, the workload's JSON, the canonical bytes of the workload and of the identifier
as hex, and the SHA-512 of each."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.core.protocols.pouw import identifier as I
from verity.core.protocols.pouw.audit import Epoch, Workload
from verity.core.protocols.pouw.protocol import Matmul, Shape
from verity.ml.protocols.pouw.schemes import get

OUT = Path(__file__).with_name("vectors") / "identifier_v1.json"
U64_MAX = (1 << 64) - 1
#: (case, scheme, matmuls in declaration order as (index, (m, k, n), weight))
CASES = (("ncp-v1", "ncp-v1", ((0, (16, 1088, 1600), 0),)),
         ("ncp-v1-shift24-one-weight", "ncp-v1-shift24", ((0, (16, 2048, 2048), 1), (1, (32, 2048, 2048), 1))),
         ("pearl-fp8-v4", "pearl-fp8-v4", ((3, (16, 1024, 16), 9),)),
         ("pearl-c-h100-v0", "pearl-c-h100-v0", ((7, (64, 1024, 64), 2),)),
         ("pearl-c-h100-v1-layer", "pearl-c-h100-v1", ((0, (64, 4096, 4096), 0), (1, (64, 4096, 4096), 1),
                                                       (2, (64, 4096, 11008), 2), (3, (64, 11008, 4096), 3))),
         ("pearl-c-h100-v1-h3-glm53", "pearl-c-h100-v1-h3", ((0, (32, 2048, 6144), 0), (1, (32, 6144, 3072), 1))),
         ("pearl-c-sm120-v1", "pearl-c-sm120-v1", ((0, (32, 8192, 8192), 0),)),
         ("pearl-c-sm120-v1-h3-decode", "pearl-c-sm120-v1-h3", ((7, (32, 4096, 4096), 2),)),
         ("pearl-c-sm120-v1-declared-order", "pearl-c-sm120-v1", ((5, (64, 4096, 4096), 1), (2, (16, 4096, 4096), 1),
                                                                  (9, (64, 4096, 11008), 0))),
         ("pearl-c-nvfp4-v0", "pearl-c-nvfp4-v0", ((0, (64, 1024, 128), 0),)),
         ("pearl-c-nvfp4-v0-u64", "pearl-c-nvfp4-v0", ((U64_MAX, (64, 4096, 4096), (1 << 53) + 1),
                                                       (1 << 53, (64, 4096, 4096), (1 << 53) + 1),
                                                       (0, (64, 8192, 1024), U64_MAX - 1))))


def label(case: str, *parts: object) -> bytes:
    return hashlib.sha256("\0".join(("verity/pouw/identifier-vectors/v1", case, *map(str, parts))).encode()).digest()


def epoch_of(case: str, matmuls) -> Epoch:
    work = Workload(tuple(Matmul(i, Shape(*shape), w) for i, shape, w in matmuls))
    return Epoch.unverified(work, {w: label(case, "weight", w) for w in work.weights()}, label(case, "beacon"))


def case_vector(case: str, scheme: str, matmuls) -> dict:
    epoch, s = epoch_of(case, matmuls), get(scheme)
    for u in epoch.workload.matmuls:
        s.check(u)
    wb, ib = I.workload_bytes(epoch), I.identifier_bytes(scheme, epoch)
    return {
        "case": case, "scheme": scheme,
        "matmuls": [[u.index, u.shape.m, u.shape.k, u.shape.n, u.weight] for u in epoch.workload.matmuls],
        "weight_roots": [[w, epoch.weight_roots[w].hex()] for w in sorted(epoch.weight_roots)],
        "salt": epoch.salt.hex(),
        "workload": I.workload_json(epoch),
        "workload_bytes": wb.hex(), "workload_sha512": hashlib.sha512(wb).hexdigest(),
        "identifier": I.identifier_v1(scheme, epoch),
        "identifier_bytes": ib.hex(), "identifier_sha512": hashlib.sha512(ib).hexdigest(),
    }


def vectors() -> dict:
    return {"schema": "verity-pouw/identifier-vectors/v1", "format": I.FORMAT, "cases": [case_vector(*c) for c in CASES]}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
