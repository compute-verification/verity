"""Generate the hm96 conformance vectors from the reference: ``vectors.json`` (``hm96-sha256/v1``) and ``vectors_sha512.json``
(``hm96-sha512/v1``) beside this file.

    python -m verity.ml.commitments.hm96.vectors          # rewrite both files
    python -m verity.ml.commitments.hm96.vectors --check  # exit 1 if either is stale

Salts and values are deterministic expansions of labels (a vector is not a commitment: a real salt comes from the OS),
and every byte is spelled out so a Rust, CUDA or circuit implementation needs nothing but these files.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ..rowleaf import ROLE_X, sha256_row_layout, sha512_row_layout
from ..scheme import LeafLayout
from . import SHA256, SHA512, HidingLayout, Instance

VECTORS = {SHA256: Path(__file__).with_name("vectors.json"), SHA512: Path(__file__).with_name("vectors_sha512.json")}
_ROW = {"sha256": ("sha256/row/v1", sha256_row_layout), "sha512": ("sha512/row/v1", sha512_row_layout)}
#: The position leaf's prefix before ``u64be(len(value))``: ``H("verity/pos-leaf/v0" || u64be(n) || value)``.
POS_LEAF_TAG = b"verity/pos-leaf/v0"


def expand(label: str, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(label.encode() + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(out[:n])


def _label(inst: Instance, label: str) -> str:
    return label if inst is SHA256 else f"{inst.hash}/{label}"


def _other_key(inst: Instance) -> bytes:
    raw = bytearray(expand("hm96/other-key" if inst is SHA256 else "hm96-sha512/other-key", inst.key_bytes))
    raw[-1] &= 0x7F
    return inst.check_key(bytes(raw))


def _mask_vectors(inst: Instance) -> list[dict[str, Any]]:
    out = []
    n = inst.salt_bytes
    for key_name, key in (("default", inst.default_key), ("other", _other_key(inst))):
        for name, salt in (("zero", bytes(n)), ("e0", bytes([1]) + bytes(n - 1)), (f"e{inst.cols - 1}", bytes(n - 1) + bytes([0x80])),
                           ("ones", b"\xff" * n), ("expand", expand(_label(inst, f"mask/{key_name}"), n))):
            out.append({"key": key_name, "salt_case": name, "salt": salt.hex(), "mask": inst.mask(key, salt).hex(),
                        "salt_digest": inst.salt_digest(salt).hex()})
    return out


def _leaf_vectors(inst: Instance) -> list[dict[str, Any]]:
    out = []
    row_name, row_layout = _ROW[inst.hash]
    cases = [("pos-leaf", n, LeafLayout(inst.hash, POS_LEAF_TAG + n.to_bytes(8, "big"), n)) for n in (0, 1, 4, 26, 256, 4096)]
    cases += [(row_name, n, row_layout(ROLE_X, 16, n // 2)) for n in (2, 3072)]
    key = inst.default_key
    for inner_name, n, inner in cases:
        value = expand(_label(inst, f"leaf/{inner_name}/{n}/value"), n)
        salt = expand(_label(inst, f"leaf/{inner_name}/{n}/salt"), inst.salt_bytes)
        layout = HidingLayout(inner, key, inst)
        x = inner.digest(value)
        proven = layout.commit(value, salt)
        out.append({"inner": inner_name, "inner_prefix": inner.prefix.hex(), "inner_suffix": inner.suffix.hex(), "value": value.hex(),
                    "inner_digest": x.hex(), "salt": salt.hex(), "b": proven[:inst.digest_bytes].hex(), "c": proven[inst.digest_bytes:].hex(),
                    "leaf": inst.tree_leaf(key, proven).hex()})
        assert proven == inst.commit_string(key, x, salt)
    return out


def build(inst: Instance = SHA256) -> dict[str, Any]:
    h = "SHA-256" if inst is SHA256 else "SHA-512"
    key = inst.default_key
    return {
        "scheme": inst.name,
        "generator": "python -m verity.ml.commitments.hm96.vectors",
        "encoding": (f"lowercase hex; bit t of a byte string is bit t & 7 of byte t >> 3; M[i][j] = key bit i + j ({inst.rows} x {inst.cols}); "
                     f"mask = M * salt over GF(2); b = inner_digest XOR mask; c = {h}(salt_prefix || salt); "
                     f"leaf = {h}(leaf_prefix || b || c)"),
        "key_label": inst.key_label.hex(),
        "default_key": key.hex(),
        "default_key_xor_gates": inst.xor_count(key),
        "default_key_row_weights": [r.bit_count() for r in inst.rows_of(key)],
        "other_key": _other_key(inst).hex(),
        "salt_prefix": inst.salt_prefix.hex(),
        "leaf_prefix_default_key": inst.leaf_prefix(key).hex(),
        "scheme_digest_default_key": inst.scheme_digest(key).hex(),
        "masks": _mask_vectors(inst),
        "leaves": _leaf_vectors(inst),
    }


def render(inst: Instance = SHA256) -> str:
    return json.dumps(build(inst), indent=1, sort_keys=True) + "\n"


def load(inst: Instance = SHA256) -> dict[str, Any]:
    return json.loads(VECTORS[inst].read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    stale = 0
    for inst, path in VECTORS.items():
        text = render(inst)
        if "--check" in args:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                print(f"{path} is stale; rerun without --check", file=sys.stderr)
                stale = 1
        else:
            path.write_text(text, encoding="utf-8")
    return stale


if __name__ == "__main__":
    raise SystemExit(main())
