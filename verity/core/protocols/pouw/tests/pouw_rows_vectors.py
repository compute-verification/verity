"""Pinned vectors for the tile checks' committed rows (`verity.core.protocols.pouw.circuit.leaves`), against which Flock's ports are
mapped: ``uv run python
verity/core/protocols/pouw/tests/pouw_rows_vectors.py`` rewrites ``tests/vectors/pouw_rows.json``.  A change to that file is a change to
a row schema and needs a reason in the PR.

Inputs are derived, not stored: part ``p`` of case ``c`` is the first bytes of SHAKE256("verity/pouw/rows-vectors/v1", c, p),
read as the schema's elements (16-, 32- and 64-bit words little-endian, codes one per byte masked to their width), and the salt
is its 192 bytes of part "salt".  A tile case's C̃ and U are TM x TN 32-bit words, hashed by `pearl_c.leaf`.  Each case's
leaf is the row Flock commits from its words and salt under the schema of its row and its tree's role (x or w,
`leaves.ROW_SCHEMA`), which its registration domain binds and its leaf does not; a `sha512/row-seg/v1` row's case also
gives each segment's length in bits and digest d_j."""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

from verity.ml.commitments.rowleaf import (ROLE_W, ROLE_X, ROW_SEG_BYTES, SCHEMA_HM96_SHA512_ROW_SEG, row_bytes,
                                        sha512_row_seg_segment_prefix, sha512_row_seg_segments)
from verity.core.protocols.pouw.circuit import leaves as L
from verity.core.protocols.pouw.schemes import pearl_c as C

OUT = Path(__file__).with_name("vectors") / "pouw_rows.json"
ROLES = {"x": ROLE_X, "w": ROLE_W}
#: (case, row, role, k or (TM, TN)): the four element rows in both trees at k = 64 and 4,096 (in role X a row-seg row of
#: one short segment and of several), the tile digests, the ok rows, the cap's rows: the salt-dead counts (row-seg, one
#: short segment and two), a word's C̃ ‖ U, its flags row with padding (k = 256) and without (8,192), a tile row's chunk
#: digests at TN = 32 (one, then zeros) and 64 (two; the case's k is TN), a tile row's sums, and a strip's row; and
#: the row_k cut's six rows in both trees, and the audit window's side rows (fp8-pi) in both trees, in role X one short
#: segment and several
CASES = tuple((f"{row}-{role}-{k}", row, role, k) for row in ("bf16", "fp32", "nvfp4-p", "fp8-p")
              for role in ("x", "w") for k in (64, 4096)) + (
    ("tile-2x33", "tile", "x", (2, 33)), ("tile-64x64", "tile", "x", (64, 64)), ("ok-x", "ok", "x", 64), ("ok-w", "ok", "w", 4096),
    ("sd-x-64", "sd", "x", 64), ("sd-x-16384", "sd", "x", 16384), ("cap-word-x", "cap-word", "x", 64),
    ("cap-flags-x-256", "cap-flags", "x", 256), ("cap-flags-x-8192", "cap-flags", "x", 8192),
    ("cap-chunks-x-32", "cap-chunks", "x", 32), ("cap-chunks-x-64", "cap-chunks", "x", 64), ("cap-sums-x", "cap-sums", "x", 64),
    ("cap-rows-x", "cap-rows", "x", 64),
    ("pc8-head-x", "pc8-head", "x", 2048), ("pc8-head-w", "pc8-head", "w", 2048),
    ("pc8-carry-x", "pc8-carry", "x", 2048), ("pc8-carry-w", "pc8-carry", "w", 2048),
    ("fp8-codes-x-64", "fp8-codes", "x", 64), ("fp8-codes-w-2048", "fp8-codes", "w", 2048),
    ("pc4-head-x", "pc4-head", "x", 2048), ("pc4-head-w", "pc4-head", "w", 2048),
    ("pc4-carry-x", "pc4-carry", "x", 2048), ("pc4-carry-w", "pc4-carry", "w", 2048),
    ("nvfp4-codes-x-64", "nvfp4-codes", "x", 64), ("nvfp4-codes-w-2048", "nvfp4-codes", "w", 2048),
    ("fp8-pi-x-64", "fp8-pi", "x", 64), ("fp8-pi-x-4096", "fp8-pi", "x", 4096), ("fp8-pi-w-4096", "fp8-pi", "w", 4096))


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
    if row == "fp8-pi":
        return {"codes": codes(case, "codes", k, 8), "p": u16(case, "p", L.P_WORDS), "index": u32(case, "index", 1)[0]}
    if row == "ok":
        return {"ok": codes(case, "ok", 1, 1)[0]}
    if row == "sd":
        return {"counts": u32(case, "counts", k // 32)}
    if row == "cap-word":
        return {"c": u32(case, "c", 1)[0], "u": u32(case, "u", 1)[0]}
    if row == "cap-flags":
        return {"flags": codes(case, "flags", k // 32, 1), "atoms": u32(case, "atoms", 1)[0],
                "zeros": u32(case, "zeros", 1)[0], "good": codes(case, "good", 1, 1)[0]}
    if row == "cap-chunks":
        return {"digests": [shake(case, f"d{h}", 32) for h in range(-(-k // C.CHUNK))]}
    if row == "cap-sums":
        return {"atoms": u32(case, "atoms", 1)[0], "zeros": u32(case, "zeros", 1)[0], "elems": u32(case, "elems", 1)[0],
                "good": codes(case, "good", 1, 1)[0]}
    if row == "cap-rows":
        return {"oks": codes(case, "oks", k, 1), "bound": u32(case, "bound", 1)[0]}
    if row == "pc8-head":
        return {"alpha": u32(case, "alpha", 1)[0], "ep": codes(case, "ep", C.R, 8),
                "th": int.from_bytes(shake(case, "th", 8), "little"), "finite": codes(case, "finite", 1, 1)[0],
                "floor": codes(case, "floor", 1, 1)[0]}
    if row == "pc8-carry":
        return {"dead": u32(case, "dead", 1)[0], "tot": u32(case, "tot", C.R)}
    if row == "fp8-codes":
        return {"codes": codes(case, "codes", k, 8)}
    if row == "pc4-head":
        return {"alpha": u32(case, "alpha", 1)[0], "ae": codes(case, "ae", 1, 8)[0]}
    if row == "pc4-carry":
        return {"tot": u32(case, "tot", C.R)}
    if row == "nvfp4-codes":
        return {"codes": codes(case, "codes", k, 4), "scales": codes(case, "scales", k // 16, 8)}
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
    if row == "fp8-pi":
        return L.words_fp8_pi(x["codes"], x["p"], x["index"])
    if row == "ok":
        return L.words_ok(x["ok"])
    if row == "sd":
        return L.words_sd(x["counts"])
    if row == "cap-word":
        return L.words_cap_word(x["c"], x["u"])
    if row == "cap-flags":
        return L.words_cap_flags(x["flags"], x["atoms"], x["zeros"], x["good"])
    if row == "cap-chunks":
        return L.words_cap_chunks(x["digests"])
    if row == "cap-sums":
        return L.words_cap_sums(x["atoms"], x["zeros"], x["elems"], x["good"])
    if row == "cap-rows":
        return L.words_cap_rows(x["oks"], x["bound"])
    if row == "pc8-head":
        return L.words_pc8_head(x["alpha"], x["ep"], x["th"], x["finite"], x["floor"])
    if row == "pc8-carry":
        return L.words_pc8_carry(x["dead"], x["tot"])
    if row == "fp8-codes":
        return L.words_fp8_codes(x["codes"])
    if row == "pc4-head":
        return L.words_pc4_head(x["alpha"], x["ae"])
    if row == "pc4-carry":
        return L.words_pc4_carry(x["tot"])
    if row == "nvfp4-codes":
        return L.words_nvfp4_codes(x["codes"], x["scales"])
    return L.words_tile(C.leaf(types.SimpleNamespace(c=x["c"], u=x["u"])))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def segments(w: list[int]) -> list[dict]:
    """A row-seg row's segments: each one's length in bits and d_j = SHA-512(S(j, s, m_j) ‖ segment j)."""
    raw, ms = row_bytes(w, 16), sha512_row_seg_segments(16 * len(w))
    return [{"n_bits": m, "digest": hashlib.sha512(sha512_row_seg_segment_prefix(j, len(ms), m)
                                                   + raw[j * ROW_SEG_BYTES:j * ROW_SEG_BYTES + m // 8]).hexdigest()}
            for j, m in enumerate(ms)]


def case_vector(case: str, row: str, role: str, k) -> dict:
    w, salt = words(row, inputs(case, row, k)), shake(case, "salt", L.INST.salt_bytes)
    bc = L.commit(row, ROLES[role], w, salt)
    out = {"case": case, "row": row, "role": role, "schema": L.SCHEMAS[row], "row_schema": L.ROW_SCHEMA[row, ROLES[role]],
           "k": list(k) if row == "tile" else k, "n_words": len(w), "row_bytes_sha256": sha(row_bytes(w, 16)),
           "x": L.row_digest(row, ROLES[role], w).hex(), "salt_sha256": sha(salt), "bc": bc.hex(),
           "tree_leaf": L.tree_leaf(bc).hex()}
    if out["row_schema"] == SCHEMA_HM96_SHA512_ROW_SEG:
        out["segments"] = segments(w)
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
