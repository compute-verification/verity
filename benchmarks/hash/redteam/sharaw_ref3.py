"""sharaw's forms against verity.ml.commitments at leaf counts whose positions and node indices need 3 bytes (n > 65,536).

sharaw's check.py stops at n = 8,192, and its bench holds the 13.49 GB step only to commit_gpu's levels on the device, whose
device functions sharaw shares. Run from the root of sharaw's tree (PYTHONPATH=.):

    python3 sharaw_ref3.py prepare WORK              the reference's levels and b || c (CPU only)
    python3 sharaw_ref3.py check BIN WORK FORM...     BIN's `check` with SHARAW_FORM=FORM on every case (exit 1 on a difference)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join("benchmarks", "pouw", "commit_gpu"))
import reference as R  # noqa: E402

NS = (65537, 70001, 131075)


def _rows_for(mode: str, n: int, n_bits: int, stride: int, rng) -> bytes:
    """reference.py's draw in chunks: one randbytes call overflows past 2^28 bytes (n > 65,535 rows of 4 KiB)."""
    assert mode == "rows" and n_bits == 8 * stride
    return b"".join(rng.randbytes(stride * min(8192, n - i)) for i in range(0, n, 8192))


R._rows_for = _rows_for


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "prepare":
        for i, n in enumerate(NS):
            print(json.dumps(R.prepare_case(os.path.join(argv[2], f"c{i:03d}"), "rows", n, 32768, 4096, "prf", R.KEY,
                                            seed=9100 + i)), flush=True)
        return 0
    if len(argv) >= 5 and argv[1] == "check":
        binary, work, forms = argv[2], argv[3], argv[4:]
        failed = 0
        for form in forms:
            os.environ["SHARAW_FORM"] = form
            for d in sorted(x for x in os.listdir(work) if x.startswith("c")):
                r = R.check_case(binary, os.path.join(work, d), f"form{form}")
                r["form"] = form
                failed += not r["ok"]
                print(json.dumps(r), flush=True)
        print(json.dumps({"summary": {"forms": forms, "cases": len(NS), "failed": failed}}), flush=True)
        return 1 if failed else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
