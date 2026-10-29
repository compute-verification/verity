"""Compare sha512_min_check's, sha512_l1_check's and sha512_rowseg_check's digests with hashlib's: stdlib only.

    ./sha512_min_check | python3 sha512_check.py
    ./sha512_l1_check | python3 sha512_check.py
    ./sha512_rowseg_check | python3 sha512_check.py

Each input line is "TAG LEN HEX", the SHA-512 of the first LEN bytes of the check's message (byte i = 151 i + 7 mod
256), "l1 CASE HEX", the frame-v3-sha512 leaf of `l1_case(CASE)`, or "seg CASE HEX", the bound row-seg segment of
`seg_case(CASE)`. Exits nonzero on the first mismatch or if no line was read.
"""
from __future__ import annotations

import hashlib
import sys

MSG = bytes((i * 151 + 7) % 256 for i in range(2000))
FRAME = b"veritor/protocol/merkle/frame/v3\0"
SEGMENT_TAG = b"verity/sha512-row-seg/v1/segment\0"


def l1_case(c: int) -> bytes:
    """sha512_l1_check's case c: the leaf message (frame, tag, then each part after its u64be length)."""
    dom = bytes((37 * i + 11 + 5 * c) % 256 for i in range(64))
    value = bytes((151 * j + 7 + 13 * c) % 256 for j in range(1024))
    parts = [dom, bytes([29 * c % 256]), bytes([(53 * c + 1) % 256]), b"u64be/1024/v1", value]
    return FRAME + (4).to_bytes(4, "big") + b"leaf" + b"".join(len(p).to_bytes(8, "big") + p for p in parts)


def seg_case(c: int) -> bytes:
    """sha512_rowseg_check's case c: segment 3c of 1,024, 8,192 bits, its first block S(j, s, m) ‖ salt_w ‖ u64be f ‖
    u64be j_call ‖ u64be row zero padded to 128 bytes, then the segment."""
    salt = bytes((41 * i + 3 + 7 * c) % 256 for i in range(32))
    head = SEGMENT_TAG + b"".join(x.to_bytes(8, "big") for x in (3 * c, 1024, 8192)) + salt
    head += b"".join(x.to_bytes(8, "big") for x in (1000 + c, 5 + c, 77 * c + 1))
    return head.ljust(128, b"\0") + bytes((151 * k + 7 + 13 * c) % 256 for k in range(1024))


def main() -> None:
    n = 0
    cases = {"l1": l1_case, "seg": seg_case}
    for line in sys.stdin:
        tag, length, digest = line.split()
        msg = cases[tag](int(length)) if tag in cases else MSG[: int(length)]
        want = hashlib.sha512(msg).hexdigest()
        if digest != want:
            raise SystemExit(f"{tag} {length}: got {digest}, hashlib {want}")
        n += 1
    if not n:
        raise SystemExit("no digests read")
    print(f"sha512_check: {n} digests agree with hashlib")


if __name__ == "__main__":
    main()
