"""fuse.cu's ChaCha20 salts against RFC 8439's block function: OUT/chacha.bin is leaves 0 .. L-1's y (192 bytes each, blocks
3 pos .. 3 pos + 2 of the keystream under OUT/chacha.json's key and nonce, counter from 0).

    python3 check_chacha.py OUT
"""
from __future__ import annotations

import json
import struct
import sys


def _rotl(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def block(key: bytes, counter: int, nonce: bytes) -> bytes:
    s = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574, *struct.unpack("<8I", key), counter, *struct.unpack("<3I", nonce)]
    x = list(s)

    def qr(a: int, b: int, c: int, d: int) -> None:
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF; x[d] = _rotl(x[d] ^ x[a], 16)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF; x[b] = _rotl(x[b] ^ x[c], 12)
        x[a] = (x[a] + x[b]) & 0xFFFFFFFF; x[d] = _rotl(x[d] ^ x[a], 8)
        x[c] = (x[c] + x[d]) & 0xFFFFFFFF; x[b] = _rotl(x[b] ^ x[c], 7)

    for _ in range(10):
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
    return struct.pack("<16I", *((a + b) & 0xFFFFFFFF for a, b in zip(x, s)))


def main(out: str) -> int:
    key = bytes(range(32))
    # RFC 8439 2.3.2: key 00..1f, nonce 000000090000004a00000000, counter 1
    want = bytes.fromhex("10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
                         "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")
    rfc = block(key, 1, bytes.fromhex("000000090000004a00000000")) == want
    meta = json.load(open(f"{out}/chacha.json"))
    got = open(f"{out}/chacha.bin", "rb").read()
    k, nonce = bytes.fromhex(meta["key"]), bytes.fromhex(meta["nonce"])
    ref = b"".join(block(k, meta["counter0"] + i, nonce) for i in range(3 * meta["leaves"]))
    ok = rfc and got == ref
    print(json.dumps({"chacha_check": ok, "rfc8439_vector": rfc, "leaves": meta["leaves"], "bytes": len(got)}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
