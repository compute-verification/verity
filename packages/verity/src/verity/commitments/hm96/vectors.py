"""Generate the ``hm96-sha256/v1`` conformance vectors (``vectors.json`` beside this file) from the reference.

    python -m verity.commitments.hm96.vectors          # rewrite vectors.json
    python -m verity.commitments.hm96.vectors --check  # exit 1 if vectors.json is stale

Salts and values are deterministic expansions of labels (a vector is not a commitment: a real salt comes from the OS),
and every byte is spelled out so a Rust, CUDA or circuit implementation needs nothing but this file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .. import vllm_v1
from ..rowleaf import ROLE_X, sha256_row_layout
from . import (
    DEFAULT_KEY,
    KEY_LABEL,
    NAME,
    SALT_PREFIX,
    HidingLayout,
    Hm96Sha256,
    commit_string,
    leaf_prefix,
    mask,
    rows,
    salt_digest,
    scheme_digest,
    tree_leaf,
    xor_count,
)

VECTORS = Path(__file__).with_name("vectors.json")


def expand(label: str, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(label.encode() + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(out[:n])


def _other_key() -> bytes:
    return key_bytes(expand("hm96/other-key", 160))


def key_bytes(raw: bytes) -> bytes:
    out = bytearray(raw)
    out[-1] &= 0x7F
    return bytes(out)


def _mask_vectors() -> list[dict[str, Any]]:
    out = []
    for key_name, key in (("default", DEFAULT_KEY), ("other", _other_key())):
        for name, salt in (("zero", bytes(128)), ("e0", bytes([1]) + bytes(127)), ("e1023", bytes(127) + bytes([0x80])),
                           ("ones", b"\xff" * 128), ("expand", expand(f"mask/{key_name}", 128))):
            out.append({"key": key_name, "salt_case": name, "salt": salt.hex(), "mask": mask(key, salt).hex(),
                        "salt_digest": salt_digest(salt).hex()})
    return out


def _leaf_vectors() -> list[dict[str, Any]]:
    out = []
    cases = [("pos-leaf", n, vllm_v1.pos_leaf_layout(n)) for n in (0, 1, 4, 26, 256, 4096)]
    cases += [("sha256/row/v1", n, sha256_row_layout(ROLE_X, 16, n // 2)) for n in (2, 3072)]
    for inner_name, n, inner in cases:
        value = expand(f"leaf/{inner_name}/{n}/value", n)
        salt = expand(f"leaf/{inner_name}/{n}/salt", 128)
        layout = HidingLayout(inner, DEFAULT_KEY)
        x = inner.digest(value)
        proven = layout.commit(value, salt)
        out.append({"inner": inner_name, "inner_prefix": inner.prefix.hex(), "inner_suffix": inner.suffix.hex(), "value": value.hex(),
                    "inner_digest": x.hex(), "salt": salt.hex(), "b": proven[:32].hex(), "c": proven[32:].hex(),
                    "leaf": tree_leaf(DEFAULT_KEY, proven).hex()})
        assert proven == commit_string(DEFAULT_KEY, x, salt)
    return out


def _tree_vectors() -> list[dict[str, Any]]:
    out = []
    digest = hashlib.sha256(b"hm96/vectors/step").digest()
    for n in (1, 2, 5):
        base = vllm_v1.VllmV1({"v": vllm_v1.StepDomain(digest, digest, digest, digest, n)}, value_nbytes={"v": lambda _i: 40})
        scheme = Hm96Sha256(base)
        values = [expand(f"tree/{n}/{i}/value", 40) for i in range(n)]
        salts = [expand(f"tree/{n}/{i}/salt", 128) for i in range(n)]
        proven = [scheme.leaf_layout("v", i).commit(v, s) for i, (v, s) in enumerate(zip(values, salts))]
        leaves = [scheme.tree_leaf("v", i, p) for i, p in enumerate(proven)]
        out.append({"domain": {"kind": "step", "program": digest.hex(), "ctx": digest.hex(), "geo": digest.hex(), "layout": digest.hex(),
                               "leaf_count": n}, "values": [v.hex() for v in values], "salts": [s.hex() for s in salts],
                    "proven": [p.hex() for p in proven], "leaves": [x.hex() for x in leaves], "root": scheme.root("v", leaves).hex(),
                    "paths": [[None if p is None else p.hex() for p in scheme.path("v", leaves, i)] for i in range(n)]})
    return out


def build() -> dict[str, Any]:
    return {
        "scheme": NAME,
        "generator": "python -m verity.commitments.hm96.vectors",
        "encoding": ("lowercase hex; bit t of a byte string is bit t & 7 of byte t >> 3; M[i][j] = key bit i + j (256 x 1024); "
                     "mask = M * salt over GF(2); b = inner_digest XOR mask; c = SHA-256(salt_prefix || salt); "
                     "leaf = SHA-256(leaf_prefix || b || c)"),
        "key_label": KEY_LABEL.hex(),
        "default_key": DEFAULT_KEY.hex(),
        "default_key_xor_gates": xor_count(DEFAULT_KEY),
        "default_key_row_weights": [r.bit_count() for r in rows(DEFAULT_KEY)],
        "other_key": _other_key().hex(),
        "salt_prefix": SALT_PREFIX.hex(),
        "leaf_prefix_default_key": leaf_prefix(DEFAULT_KEY).hex(),
        "scheme_digest_default_key": scheme_digest(DEFAULT_KEY).hex(),
        "masks": _mask_vectors(),
        "leaves": _leaf_vectors(),
        "trees": _tree_vectors(),
    }


def render() -> str:
    return json.dumps(build(), indent=1, sort_keys=True) + "\n"


def load() -> dict[str, Any]:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    text = render()
    if "--check" in args:
        if VECTORS.read_text(encoding="utf-8") != text:
            print(f"{VECTORS} is stale; rerun without --check", file=sys.stderr)
            return 1
        return 0
    VECTORS.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
