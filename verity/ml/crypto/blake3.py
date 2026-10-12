"""BLAKE3 (hash and keyed hash, 32-byte output) in plain Python: the reference the row-leaf verifiers use.

Standard-library only and deliberately slow.  It follows the BLAKE3 specification's reference implementation:
1024-byte chunks of 64-byte blocks, ``CHUNK_START`` / ``CHUNK_END`` on a chunk's first / last block, the chunk counter
in every compression of the chunk, a left-heavy binary tree of chunk chaining values under ``PARENT``, ``ROOT`` on the
final compression.  Keyed mode replaces the IV with the key words and sets ``KEYED_HASH`` on every compression.
"""

from __future__ import annotations

__all__ = ["BLOCK_LEN", "CHUNK_LEN", "IV", "KEY_LEN", "blake3", "compress", "keyed"]

IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
MSG_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
CHUNK_START, CHUNK_END, PARENT, ROOT, KEYED_HASH = 1, 2, 4, 8, 16
BLOCK_LEN, CHUNK_LEN, KEY_LEN = 64, 1024, 32
_M = 0xFFFFFFFF


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _M


def _g(v: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    v[a] = (v[a] + v[b] + mx) & _M
    v[d] = _rotr(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d]) & _M
    v[b] = _rotr(v[b] ^ v[c], 12)
    v[a] = (v[a] + v[b] + my) & _M
    v[d] = _rotr(v[d] ^ v[a], 8)
    v[c] = (v[c] + v[d]) & _M
    v[b] = _rotr(v[b] ^ v[c], 7)


def compress(cv: tuple[int, ...], block: bytes, counter: int, block_len: int, flags: int) -> list[int]:
    """The 16-word output of one compression (the new chaining value is its first 8 words)."""

    m = [int.from_bytes(block[4 * i:4 * i + 4], "little") for i in range(16)]
    v = [*cv, *IV[:4], counter & _M, (counter >> 32) & _M, block_len, flags]
    for r in range(7):
        _g(v, 0, 4, 8, 12, m[0], m[1])
        _g(v, 1, 5, 9, 13, m[2], m[3])
        _g(v, 2, 6, 10, 14, m[4], m[5])
        _g(v, 3, 7, 11, 15, m[6], m[7])
        _g(v, 0, 5, 10, 15, m[8], m[9])
        _g(v, 1, 6, 11, 12, m[10], m[11])
        _g(v, 2, 7, 8, 13, m[12], m[13])
        _g(v, 3, 4, 9, 14, m[14], m[15])
        if r < 6:
            m = [m[i] for i in MSG_PERMUTATION]
    return [v[i] ^ v[i + 8] for i in range(8)] + [v[i + 8] ^ cv[i] for i in range(8)]


def _words(key: bytes) -> tuple[int, ...]:
    return tuple(int.from_bytes(key[4 * i:4 * i + 4], "little") for i in range(8))


def _hash(data: bytes, key_words: tuple[int, ...], flags: int) -> bytes:
    chunks = [data[i:i + CHUNK_LEN] for i in range(0, len(data), CHUNK_LEN)] or [b""]
    outputs = []  # per chunk: (cv, block, counter, block_len, flags) of its last compression, not yet finalised
    for counter, chunk in enumerate(chunks):
        blocks = [chunk[i:i + BLOCK_LEN] for i in range(0, len(chunk), BLOCK_LEN)] or [b""]
        cv = key_words
        for k, block in enumerate(blocks[:-1]):
            cv = tuple(compress(cv, block, counter, BLOCK_LEN, flags | (CHUNK_START if k == 0 else 0))[:8])
        last = blocks[-1]
        f = flags | CHUNK_END | (CHUNK_START if len(blocks) == 1 else 0)
        outputs.append((cv, last.ljust(BLOCK_LEN, b"\0"), counter, len(last), f))

    def cv_of(out) -> tuple[int, ...]:
        return tuple(compress(*out)[:8])

    def parent(left, right):
        block = b"".join(w.to_bytes(4, "little") for w in (*cv_of(left), *cv_of(right)))
        return (key_words, block, 0, BLOCK_LEN, flags | PARENT)

    def tree(outs):
        if len(outs) == 1:
            return outs[0]
        split = 1 << ((len(outs) - 1).bit_length() - 1)  # the largest power of two strictly below len: left-heavy
        return parent(tree(outs[:split]), tree(outs[split:]))

    cv, block, counter, block_len, f = tree(outputs)
    return b"".join(w.to_bytes(4, "little") for w in compress(cv, block, counter, block_len, f | ROOT)[:8])


def blake3(data: bytes) -> bytes:
    """``blake3(data)``, 32 bytes."""

    return _hash(bytes(data), IV, 0)


def keyed(key: bytes, data: bytes) -> bytes:
    """``blake3(data, key=key)``, 32 bytes; ``key`` is exactly 32 bytes."""

    if len(key) != KEY_LEN:
        raise ValueError("a BLAKE3 key is 32 bytes")
    return _hash(bytes(data), _words(key), KEYED_HASH)
