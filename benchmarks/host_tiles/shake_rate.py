"""SHAKE256 output rate on one core, as NCP derives F₁: one XOF call per weight column, k bytes out (``ncp.noise_row``).

    shake_rate.py SECONDS   -> one JSON line: bytes_per_s (best of the column pass and a single long digest)
"""

import hashlib
import json
import sys
import time

K, COLUMNS = 4096, 4096


def columns() -> float:
    t0 = time.perf_counter()
    for j in range(COLUMNS):
        hashlib.shake_256(b"verity/pouw/ncp/F1/v1\0" + j.to_bytes(8, "little")).digest(K)
    return K * COLUMNS / (time.perf_counter() - t0)


def long() -> float:
    t0 = time.perf_counter()
    hashlib.shake_256(b"seed").digest(K * COLUMNS)
    return K * COLUMNS / (time.perf_counter() - t0)


def main() -> None:
    budget = float(sys.argv[1])
    start, col, lng = time.perf_counter(), 0.0, 0.0
    while time.perf_counter() - start < budget:
        col, lng = max(col, columns()), max(lng, long())
    print(json.dumps({"kernel": "shake256", "columns_bytes_per_s": col, "long_bytes_per_s": lng,
                      "bytes_per_s": max(col, lng)}))


if __name__ == "__main__":
    main()
