"""Compare sha512_min_check's and sha512_l1_check's digests with hashlib's: stdlib only.

    ./sha512_min_check | python3 sha512_check.py
    ./sha512_l1_check | python3 sha512_check.py

Each input line is "TAG LEN HEX", the SHA-512 of the first LEN bytes of the check's message (byte i = 151 i + 7 mod
256), or "l1 CASE HEX", the frame-v3-sha512 leaf of `l1_case(CASE)`. Exits nonzero on the first mismatch or if no line
was read.
"""
from __future__ import annotations

import hashlib
import sys

MSG = bytes((i * 151 + 7) % 256 for i in range(2000))
FRAME = b"veritor/protocol/merkle/frame/v3\0"


def l1_case(c: int) -> bytes:
    """sha512_l1_check's case c: the leaf message (frame, tag, then each part after its u64be length)."""
    dom = bytes((37 * i + 11 + 5 * c) % 256 for i in range(64))
    value = bytes((151 * j + 7 + 13 * c) % 256 for j in range(1024))
    parts = [dom, bytes([29 * c % 256]), bytes([(53 * c + 1) % 256]), b"u64be/1024/v1", value]
    return FRAME + (4).to_bytes(4, "big") + b"leaf" + b"".join(len(p).to_bytes(8, "big") + p for p in parts)


def main() -> None:
    n = 0
    for line in sys.stdin:
        tag, length, digest = line.split()
        msg = l1_case(int(length)) if tag == "l1" else MSG[: int(length)]
        want = hashlib.sha512(msg).hexdigest()
        if digest != want:
            raise SystemExit(f"{tag} {length}: got {digest}, hashlib {want}")
        n += 1
    if not n:
        raise SystemExit("no digests read")
    print(f"sha512_check: {n} digests agree with hashlib")


if __name__ == "__main__":
    main()
