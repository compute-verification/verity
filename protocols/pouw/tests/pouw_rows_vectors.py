"""Pinned vectors for the tile checks' committed rows (`verity_pouw.circuit.leaves`), against which C-Flock's ports are
mapped: ``uv run python
protocols/pouw/tests/pouw_rows_vectors.py`` rewrites ``tests/vectors/pouw_rows.json``.  A change to that file is a change to
a row schema and needs a reason in the PR.

Inputs are derived, not stored: part ``p`` of case ``c`` is the first bytes of SHAKE256("verity/pouw/rows-vectors/v1", c, p),
read as the schema's elements (16-bit and 32-bit words little-endian, codes one per byte masked to their width), and the salt
is its 192 bytes of part "salt".  A tile case's C̃ and U are TM x TN 32-bit words, hashed by `pearl_c.leaf`.  Each case's
leaf is the row C-Flock commits from its words and salt (`leaves.ROW_SCHEMA`); a case's name says which tree (x or w) the
row sits in, which its registration domain binds and its leaf does not."""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

from verity.commitments.rowleaf import row_bytes
from verity_pouw.circuit import leaves as L
from verity_pouw.schemes import pearl_c as C

OUT = Path(__file__).with_name("vectors") / "pouw_rows.json"
#: (case, row, k or (TM, TN))
CASES = (("bf16-x-64", "bf16", 64), ("bf16-w-4096", "bf16", 4096),
         ("fp32-x-64", "fp32", 64), ("fp32-w-4096", "fp32", 4096),
         ("nvfp4-p-x-64", "nvfp4-p", 64), ("nvfp4-p-w-4096", "nvfp4-p", 4096),
         ("fp8-p-x-64", "fp8-p", 64), ("fp8-p-w-4096", "fp8-p", 4096),
         ("tile-2x33", "tile", (2, 33)), ("tile-64x64", "tile", (64, 64)),
         ("ok-x", "ok", 64), ("ok-w", "ok", 4096))


def shake(case: str, part: str, n: int) -> bytes:
    return hashlib.shake_256(f"verity/pouw/rows-vectors/v1\0{case}\0{part}".encode()).digest(n)


def u16(case: str, part: str, n: int) -> list[int]:
    raw = shake(case, part, 2 * n)
    return [int.from_bytes(raw[2 * i:2 * i + 2], "little") for i in range(n)]


def u32(case: str, part: str, n: int) -> list[int]:
    raw = shake(case, part, 4 * n)
    return [int.from_bytes(raw[4 * i:4 * i + 4], "little") for i in range(n)]


def codes(case: str, part: str, n: int, bits: int) -> list[int]:
    return [b & ((1 << bits) - 1) for b in shake(case, part, n)]


def inputs(case: str, row: str, k) -> dict:
    """The case's elements, as the gates read them."""
    if row == "bf16":
        return {"x": u16(case, "x", k)}
    if row == "fp32":
        return {"x": u32(case, "x", k)}
    if row == "nvfp4-p":
        return {"codes": codes(case, "codes", k, 4), "scales": codes(case, "scales", k // 16, 8), "p": u16(case, "p", L.P_WORDS)}
    if row == "fp8-p":
        return {"codes": codes(case, "codes", k, 8), "p": u16(case, "p", L.P_WORDS)}
    if row == "ok":
        return {"ok": codes(case, "ok", 1, 1)[0]}
    tm, tn = k
    return {"c": [u32(case, f"c{i}", tn) for i in range(tm)], "u": [u32(case, f"u{i}", tn) for i in range(tm)]}


def words(row: str, x: dict) -> list[int]:
    if row == "bf16":
        return L.words_bf16(x["x"])
    if row == "fp32":
        return L.words_fp32(x["x"])
    if row == "nvfp4-p":
        return L.words_nvfp4_p(x["codes"], x["scales"], x["p"])
    if row == "fp8-p":
        return L.words_fp8_p(x["codes"], x["p"])
    if row == "ok":
        return L.words_ok(x["ok"])
    return L.words_tile(C.leaf(types.SimpleNamespace(c=x["c"], u=x["u"])))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def case_vector(case: str, row: str, k) -> dict:
    w, salt = words(row, inputs(case, row, k)), shake(case, "salt", L.INST.salt_bytes)
    bc = L.commit(row, w, salt)
    out = {"case": case, "row": row, "schema": L.SCHEMAS[row], "row_schema": L.ROW_SCHEMA[row],
           "k": list(k) if row == "tile" else k, "n_words": len(w), "row_bytes_sha256": sha(row_bytes(w, 16)),
           "x": L.row_digest(row, w).hex(), "salt_sha256": sha(salt), "bc": bc.hex(), "tree_leaf": L.tree_leaf(bc).hex()}
    if row == "tile":
        out["tile_digest"] = bytes(v for x in w for v in x.to_bytes(2, "little")).hex()
    else:
        assert len(w) == L.ROW_WORDS[row](k)
    return out


def vectors() -> dict:
    return {"schema": "verity-pouw/rows-vectors/v1", "word_bits": 16, "key_sha256": sha(L.KEY),
            "cases": [case_vector(*c) for c in CASES]}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
