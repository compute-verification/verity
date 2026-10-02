"""The committers' weights root under hiding leaves (``registered-hm96-sha512/v1``): every parameter and buffer a registered value
in the format of ``verity_sampled_proofs.one_stage.registered`` (``cursor/registered-values-95d4``, agreed with proofs), and the
weights root over their roots.

A value's row is its last dimension (an ``[out, in]`` GEMM weight is ``out`` rows of ``in`` elements; a 0-dim or 1-D tensor is one
row), as ``ceil(row bytes / 2)`` u16 little-endian words (an odd row's last word has a zero high byte).  Row ``r`` commits as an input
row of a circuit statement does: ``x = SHA-512(sha512_row_prefix(ROLE_X, 16, words) || row)``, a 192-byte salt ``y``,
``b || c = hm96.SHA512.commit_string(K, x, y)`` and the frame-v3-sha512 leaf ``row_leaf(D, r, hm96.SHA512.tree_leaf(K, b || c),
"hm96-sha512/row/v1")``; the value's root is the frame-v3-sha512 root over its rows under ``D = CommitmentDomain(identity_digest(
"verity/one-stage/registered-domain/v0", {program, value, schema, words, leaves}), -2, RangeIndexedDomain(0, rows), "sha512")``.
The weights root is ``vllm_v1.weights_root(geo, scheme.names_digest(names), value roots, hash="sha512")``: the vllm-v1 rule's
SHA-256 geometry and names digests, the value roots folded and the root taken under SHA-512.

Salts: row ``r`` of value ``i`` (its index in ``names``) takes the 192 bytes at offset ``192 r`` of the RFC 8439 ChaCha20 stream under
the committer's 32-byte registration key with nonce ``u32le(0) || u64le(i)``, so a re-registration of the same weights gives the
same root and no two rows share a salt.  The key is prover material, like the weights it hides: it stays with the committer and in a
replay bundle's ``weights/registration.key`` (beside the bytes the bundle holds), never in a record.
"""
from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from verity.commitments import hm96, vllm_v1
from verity.commitments.identity import identity_digest
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain, tree_levels
from verity.commitments.rowleaf import ROLE_X, SCHEMA_HM96_SHA512_ROW, row_leaf, sha512_row_prefix
from verity_vllm.commit import hiding, scheme

NAME = "registered-hm96-sha512/v1"
REGISTERED_DOMAIN = "verity/one-stage/registered-domain/v0"
SCHEMA = SCHEMA_HM96_SHA512_ROW
WORD_BITS = 16
OWNER = -2
KEY_FILE = "registration.key"
_INST = hm96.SHA512
_LEAVES: list[hiding.HidingLeaves] = []


def _leaves() -> hiding.HidingLeaves:
    if not _LEAVES:
        _LEAVES.append(hiding.HidingLeaves(instance=_INST))
    return _LEAVES[0]


def rows_of(shape: Sequence[int], nbytes: int) -> tuple[int, int]:
    """(rows, bytes per row) of a registered tensor of `shape` and `nbytes` bytes: its last dimension is the row."""
    n = int(math.prod(shape)) if len(shape) else 1
    if n <= 0 or nbytes % n:
        raise ValueError(f"a tensor of shape {list(shape)} does not hold {nbytes} bytes in whole elements")
    width = (nbytes // n) * (int(shape[-1]) if len(shape) else 1)
    return nbytes // width, width


def domain(program: str, value: str, words: int, leaves: int) -> CommitmentDomain:
    binding = identity_digest(REGISTERED_DOMAIN, {"program": program, "value": value, "schema": SCHEMA, "words": int(words),
                                                  "leaves": int(leaves)})
    return CommitmentDomain(bytes.fromhex(binding), OWNER, RangeIndexedDomain(0, int(leaves)), hash="sha512")


def salts(key: bytes, index: int, rows: int) -> bytes:
    """The salts of rows [0, rows) of value `index`, concatenated."""
    return hiding.chacha20(key, (0).to_bytes(4, "little") + int(index).to_bytes(8, "little"), rows * _INST.salt_bytes)


def value_root(program: str, name: str, index: int, raw: Any, shape: Sequence[int], key: bytes) -> tuple[bytes, dict[str, Any]]:
    """The registered root of tensor `name` (value `index`, its bytes `raw`) and its registration entry {value, root, leaves, words,
    domain} (proofs' ``Registered.entry``)."""
    mv = memoryview(raw).cast("B")
    rows, width = rows_of(shape, len(mv))
    words = -(-width // 2)
    prefix = hashlib.sha512(sha512_row_prefix(ROLE_X, WORD_BITS, words))
    pad = b"\0" * (2 * words - width)
    xs = []
    for r in range(rows):
        h = prefix.copy()
        h.update(mv[r * width:(r + 1) * width])
        if pad:
            h.update(pad)
        xs.append(h.digest())
    dom = domain(program, name, words, rows)
    leaves = _leaves().leaves_with(xs, salts(key, index, rows))
    root = tree_levels(dom, [row_leaf(dom, r, t, SCHEMA) for r, t in enumerate(leaves)])[-1][0]
    return root, {"value": name, "root": root.hex(), "leaves": rows, "words": words, "domain": dom.domain_id.hex()}


def weights_root(geo_digest: bytes, names: list, roots: Sequence[bytes]) -> bytes:
    return vllm_v1.weights_root(bytes(geo_digest), scheme.names_digest(names), [bytes(r) for r in roots], hash="sha512")


def register(com: Any, items: list[tuple[str, Any]], t0: float, mode: str, dtype_name: Callable[[Any], str]) -> dict[str, Any]:
    """`NativeHostCommitter.register_weights` under hiding leaves: `items` the (name, tensor) to register, in order, `mode` the
    weights hash's (`VERITY_WEIGHTS_HASH`).  Keeps the tree under the root on `com.weights_tree` (off the record) and the registration
    key on the committer, drawn once."""
    import time

    import torch

    if mode != "host":
        raise ValueError(f"VERITY_WEIGHTS_HASH={mode}: the {NAME} weights root is hashed on the host only")
    key = getattr(com, "_registration_key", None) or os.urandom(32)
    com._registration_key = key
    program = com.program_digest.hex()
    names, roots, entries, n_bytes = [], [], [], 0
    for i, (name, p) in enumerate(items):
        flat = p.detach().contiguous().reshape(-1)
        host = (flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat).cpu().numpy()
        r, e = value_root(program, name, len(names), host, list(p.shape), key)
        names.append([name, dtype_name(p), list(p.shape), int(host.size)])
        roots.append(r)
        entries.append(e)
        n_bytes += int(host.size)
    root = weights_root(com._geo_digest, names, roots)
    com.weights_tree = {"scheme": NAME, "program": program, "names": names, "tensor_roots": [r.hex() for r in roots], "root": root.hex(),
                        "geo_digest": com._geo_digest.hex(), "registered": entries}
    com.weights = {"root": root.hex(), "scheme": NAME, "tensors": len(roots), "bytes": n_bytes, "wall_s": time.perf_counter() - t0,
                   "label": "measured; once per process, excluded from steady state", "hash": "host"}
    return com.weights


def tensor_root(tree: dict[str, Any], i: int, raw: np.ndarray, key: bytes | None) -> bytes:
    """The root tensor `i` of the weights tree `tree` takes over its bytes `raw`, under the tree's scheme (vllm-v1's pos-leaf fold for a
    tree without one); `key` the registration key a registered tree needs."""
    if tree.get("scheme") != NAME:
        mv, chunk = memoryview(raw).cast("B"), int(tree["chunk"])
        return scheme.fold([scheme.pos_leaf(mv[o:o + chunk]) for o in range(0, len(mv), chunk)])
    if key is None:
        raise ValueError(f"a {NAME} weights tree opens with its registration key only")
    name, _dt, shape, _nb = tree["names"][i]
    return value_root(tree["program"], name, i, raw, shape, key)[0]


def tree_root(tree: dict[str, Any]) -> str:
    """The weights root the tree's tensor roots fold to, under its scheme."""
    roots = [bytes.fromhex(r) for r in tree["tensor_roots"]]
    if tree.get("scheme") == NAME:
        return weights_root(bytes.fromhex(tree["geo_digest"]), tree["names"], roots).hex()
    return scheme.weights_root(bytes.fromhex(tree["geo_digest"]), scheme.names_digest(tree["names"]), roots).hex()
