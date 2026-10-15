"""The keys chain.cu reads: `audit.hm96_kernel_constants` (hm.bin), the prover's hm96 salt key (salt.bin), and for each m the
domain ids of X_0 .. X_S's row trees (dom_m<m>.bin, 64 bytes each), ``audit.rows_domain(binding(j), j, m, HM96)``.

    python gen.py DIR S M [M ...]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from verity.core.protocols.pouw import audit
from verity.core.protocols.pouw.protocol import tagged

SALT_KEY = hashlib.sha512(b"verity/chained-tensor-work/salt-key/v1").digest()[:32]


def binding(j: int) -> bytes:
    """X_j's row-tree binding."""
    return tagged("verity/chained-tensor-work/x/v1", j)


def domain(j: int, m: int):
    return audit.rows_domain(binding(j), j, m, audit.HM96)


def main(argv: list[str]) -> None:
    out, steps, ms = Path(argv[0]), int(argv[1]), [int(a) for a in argv[2:]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "hm.bin").write_bytes(audit.hm96_kernel_constants())
    (out / "salt.bin").write_bytes(SALT_KEY)
    for m in ms:
        (out / f"dom_m{m}.bin").write_bytes(b"".join(domain(j, m).domain_id for j in range(steps + 1)))


if __name__ == "__main__":
    main(sys.argv[1:])
