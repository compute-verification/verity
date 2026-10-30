"""Pinned vectors for an H1 table's leaves past K = 4096 (`h1zy.chunk_leaves`, `h1zy.chunk_words`), against which the GPU
trees that commit ``a/<K>@<s>`` and ``b/<K>`` are gated leaf by leaf: ``uv run python
verity/core/protocols/pouw/tests/splitk_rows_vectors.py`` rewrites ``tests/vectors/splitk_rows.json``, and with
``--words DIR`` also writes each case's words to ``DIR/<case>.u16`` (its leaves in order, each its words, u16
little-endian), which the 256 KiB cap keeps out of the tree. A change to the JSON is a change to the chunk-row layout and
needs a reason in the PR.

Inputs are derived, not stored: case c's rows are the first ``rows (K + 4)`` bytes of SHAKE256(`LABEL`, NUL, c), row r
the r-th K + 4 of them, its K int8 codes and then its f32 scale little-endian, whose bits are set into [2^-7, 2^-6)
(``0x3C000000 | bits & 0x007FFFFF``) so it is finite. The program digest (a stand-in for the window Program
descriptor's SHA-512), the window's open root and the 32-byte salt key are SHAKE256(`LABEL`, NUL, ``program`` /
``window`` / ``key``), 64, 64 and 32 bytes. Leaf j's salt is `audit.hm96_salt(key, nonce, j)` at the case's nonce:
the served A table of call 0's (vllm's ``pouw_h1_served.nonce(0, "a")``, `A_NONCE`), which binds the window, and the
weights' ``b/<K>`` at `weights.B_NONCE` (K the window's only K), which binds none. Each case pins every leaf's SHA-512 of
its words' bytes, its ``sha512/row/v2`` digest and its frame-v3 leaf, and the value's domain id and root, all as
`registered.Registered.commit` gives them under ``hm96-sha512/row/v2``."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

from verity.core.protocols.pouw import audit
from verity.core.protocols.pouw import weights as Wt
from verity.ml.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.sampled_proofs.one_stage import registered as G

OUT = Path(__file__).with_name("vectors") / "splitk_rows.json"
LABEL = b"verity/pouw/splitk-rows-vectors/v1"
#: the served A table of call 0's salt nonce: vllm's ``REQUESTS_NONCE + 2^42 + 2 * 0 + TABLES.index("a")``
A_NONCE = (1 << 61) + (1 << 42)
#: (case, value, K, rows, nonce, binds the window)
CASES = (("a-14336-s0", "a/14336@0", 14336, 64, A_NONCE, True),
         ("b-6144", "b/6144", 6144, 64, Wt.B_NONCE, False))


def _shake(*parts: bytes, n: int) -> bytes:
    return hashlib.shake_256(b"\0".join((LABEL, *parts))).digest(n)


def rows(case: str, k: int, count: int) -> list[bytes]:
    """Case ``case``'s ``count`` rows at contraction ``k``, each `h1zy.ROW`(k)'s bytes."""
    data, out = _shake(case.encode(), n=count * (k + 4)), []
    for r in range(count):
        row = data[r * (k + 4):(r + 1) * (k + 4)]
        bits = 0x3C000000 | struct.unpack("<I", row[k:])[0] & 0x007FFFFF
        out.append(row[:k] + struct.pack("<I", bits))
    return out


def key() -> bytes:
    return _shake(b"key", n=32)


def program() -> str:
    return _shake(b"program", n=64).hex()


def window() -> str:
    return _shake(b"window", n=64).hex()


def commit(case: tuple) -> tuple[list[list[int]], G.Registered]:
    """A case's leaves' words (`h1zy.chunk_words`) and their registration."""
    name, value, k, count, nonce, windowed = case
    words = Z.chunk_words(rows(name, k, count), k)
    salts = [audit.hm96_salt(key(), nonce, j) for j in range(len(words))]
    return words, G.Registered.commit(program(), value, words, salts, schema=G.SCHEMA_V2,
                                      window=window() if windowed else None)


def _bytes(words: list[int]) -> bytes:
    return struct.pack(f"<{len(words)}H", *words)


def vectors() -> dict:
    out = {"label": LABEL.decode(), "schema": G.SCHEMA_V2, "program": program(), "window": window(), "cases": []}
    for case in CASES:
        name, value, k, count, nonce, windowed = case
        words, reg = commit(case)
        n = Z.chunks(k)
        out["cases"].append({
            "case": name, "value": value, "k": k, "n": n, "k_c": k // n, "rows": count, "nonce": nonce,
            "window": windowed, "leaves": reg.leaves, "words": reg.words, "domain": reg.domain.domain_id.hex(),
            "root": reg.root.hex(),
            "leaf": [{"words_sha512": hashlib.sha512(_bytes(w)).hexdigest(),
                      "row_digest": G.row_digest(w, G.WORD_BITS, G.SCHEMA_V2).hex(), "frame_leaf": reg.levels[0][j].hex()}
                     for j, w in enumerate(words)]})
    return out


def write_words(directory: Path) -> list[Path]:
    """Each case's words to ``directory/<case>.u16``: its leaves in order, each its words, u16 little-endian."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for case in CASES:
        words, _reg = commit(case)
        path = directory / f"{case[0]}.u16"
        path.write_bytes(b"".join(_bytes(w) for w in words))
        paths.append(path)
    return paths


if __name__ == "__main__":
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    if sys.argv[1:2] == ["--words"]:
        for p in write_words(Path(sys.argv[2])):
            print(p)
