"""TurboSHAKE128 (RFC 9861) in plain Python: the ticket hash of ``pearl-c`` (``verity/core/protocols/pouw``).

Standard-library only and deliberately slow.  TurboSHAKE128 is the sponge of Keccak-p[1600, 12], the last 12 of
Keccak-f[1600]'s 24 rounds, at rate 168 bytes: the message is followed by the domain byte ``D`` (0x01 … 0x7F), zeros to
the end of the block, and 0x80 XORed into the block's last byte; the output is squeezed 168 bytes per permutation.
``keccak_p(state, rounds=24)`` is Keccak-f[1600], which the tests check against ``hashlib``'s SHAKE128.
"""

from __future__ import annotations

__all__ = ["RATE", "keccak_p", "turboshake128"]

RATE = 168
_M = (1 << 64) - 1
_RC = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
    0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
#: ρ offsets by lane index x + 5y
_ROT = (0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14)


def _rotl(x: int, n: int) -> int:
    return ((x << n) | (x >> (64 - n))) & _M if n else x


def keccak_p(state: list[int], rounds: int = 12) -> None:
    """Keccak-p[1600, rounds] in place on 25 little-endian lanes (index x + 5y): rounds 24 − ``rounds`` … 23."""
    for rc in _RC[24 - rounds:]:
        c = [state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rotl(c[(x + 1) % 5], 1) for x in range(5)]
        a = [state[i] ^ d[i % 5] for i in range(25)]
        b = [0] * 25
        for x in range(5):
            for y in range(5):
                b[y + 5 * ((2 * x + 3 * y) % 5)] = _rotl(a[x + 5 * y], _ROT[x + 5 * y])
        for y in range(0, 25, 5):
            row = b[y:y + 5]
            for x in range(5):
                state[y + x] = row[x] ^ (~row[(x + 1) % 5] & row[(x + 2) % 5])
        state[0] ^= rc


def turboshake128(message: bytes, domain: int = 0x1F, length: int = 32) -> bytes:
    """TurboSHAKE128(M = ``message``, D = ``domain``, L = ``length``)."""
    if not 0x01 <= domain <= 0x7F:
        raise ValueError("TurboSHAKE's domain byte is in 0x01 … 0x7F")
    padded = bytearray(message) + bytes([domain])
    padded += bytes(-len(padded) % RATE)
    padded[-1] ^= 0x80
    state = [0] * 25
    for off in range(0, len(padded), RATE):
        for i in range(RATE // 8):
            state[i] ^= int.from_bytes(padded[off + 8 * i:off + 8 * i + 8], "little")
        keccak_p(state)
    out = bytearray()
    while True:
        out += b"".join(lane.to_bytes(8, "little") for lane in state[:RATE // 8])
        if len(out) >= length:
            return bytes(out[:length])
        keccak_p(state)
