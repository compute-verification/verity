"""Compare sha512_min_check's digests with hashlib's: stdlib only.

    ./sha512_min_check | python3 sha512_check.py

Each input line is "TAG LEN HEX", the SHA-512 of the first LEN bytes of the check's message (byte i = 151 i + 7 mod
256). Exits nonzero on the first mismatch or if no line was read.
"""
from __future__ import annotations

import hashlib
import sys

MSG = bytes((i * 151 + 7) % 256 for i in range(300))


def main() -> None:
    n = 0
    for line in sys.stdin:
        tag, length, digest = line.split()
        want = hashlib.sha512(MSG[: int(length)]).hexdigest()
        if digest != want:
            raise SystemExit(f"{tag} {length}: got {digest}, hashlib {want}")
        n += 1
    if not n:
        raise SystemExit("no digests read")
    print(f"sha512_min: {n} digests agree with hashlib")


if __name__ == "__main__":
    main()
