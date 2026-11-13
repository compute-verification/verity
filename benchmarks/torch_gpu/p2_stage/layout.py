"""Where a layer's matrices' plaintext comes from in the P2-encoded weights store, and P2's decode of one block on the CPU
(the check's reference). No torch.

The store's layout is p2_v1's fixture (`benchmarks/pous/p2_v1/store.py`, schema `verity-pous/p2-store-layout/v1`), as
`p2_v1.load` holds it: `StoreLayout` over its `Tensor`s, a layer's `MATRICES`, the codec's `PARAMS` and the sizes of a
block's row and payload (`ROW`, `PAYLOAD`, `LEAD`). They are defined there and read here by this directory's modules.
`store_layout` reads the fixture's JSON, the real store's (`store.py layout`, pinned by its `LAYOUT_SHA512`) or the toy's
(`store-toy/toy.json`'s `layout`), with the key prefix: K and t from counter-mode SHA-512 after `schemes.p2.expand_prefix`
of the protocol's 24-byte salt (`registered_prefix`).

`Weights` is where the plaintext comes from: the seeded weights stream (`benchmarks.torch_gpu.p2_stage.served_weights manifest`), every tensor
at its stream offset, QKV being wq, wk and wv one after another, each with its f32 scales, and the head the stream's
`head` (`Tensor.stream_names`). `payloads` lays it out as the store's W.

`check_blocks` holds rows to W's payloads by memory-accounting's reference decode of one block (`p2_v1/store.py`'s
`decode_block`: `p2_v1.reference.decode` under `reference.keys_after`), which refuses a row not below p, or a decode
wider than the payload.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from verity.protocols.pous.verifier.schemes import p2 as S

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "pous"))
from p2_v1 import reference as R  # noqa: E402
from p2_v1.load import (COLUMNS, LEAD, MATRICES, PARAMS, PARTS, PAYLOAD, ROW, SPEC, StoreLayout, Tensor,  # noqa: E402, F401
                        blocks_of, matrix_name)

SCHEMA = "verity-pous/p2-store-layout/v1"
SCHEME = "p2-16448/v4"

Read = Callable[[int, int], bytes]


def registered_prefix(salt: bytes) -> bytes:
    """The registered scheme's key prefix for the protocol's 24-byte `salt`: whole 128-byte blocks, one midstate."""
    return S.expand_prefix(PARAMS, salt)


def store_layout(doc: dict, prefix: bytes) -> StoreLayout:
    """The layout of the fixture's JSON `doc` (the layout itself, or the toy's `toy.json`, which holds it as `layout`)."""
    doc = doc.get("layout", doc)
    if doc.get("schema") != SCHEMA or doc.get("scheme") != SCHEME or tuple(doc.get("tensor_columns", ())) != COLUMNS:
        raise ValueError(f"not a {SCHEMA} layout of {SCHEME} with columns {COLUMNS}")
    q = doc["params"]
    params = S.P2Params(q["w"], q["c0"], q["blocks_per_segment"], q["segments"])
    if params.manifest() != q or (doc["block_bytes"], doc["payload_bytes"], doc["blocks"]) != (
            params.block_bytes, params.payload_bytes, params.blocks):
        raise ValueError(f"the layout's params {q} are not the registered codec's")
    return StoreLayout(params, tuple(Tensor(*row) for row in doc["tensors"]), prefix)


@dataclass(frozen=True)
class Weights:
    """The weights stream's tensors (`benchmarks.torch_gpu.p2_stage.served_weights manifest`: rows with ``name``, ``offset``, ``nbytes`` and
    ``shape``); layer l's tensor t is ``l{l}.{t}``, its f32 scales ``l{l}.{t}.scale``."""

    rows: dict[str, dict]

    @classmethod
    def of(cls, manifest: dict) -> Weights:
        return cls({t["name"]: t for t in manifest["tensors"]})

    def parts(self, layer: int, matrix: str, scales: bool = False) -> list[tuple[int, int]]:
        """`(offset, nbytes)` in the stream of `matrix`'s tensors (or their scales), in the order its rows are stacked."""
        if matrix not in PARTS:
            raise KeyError(f"a matrix is one of {sorted(PARTS)}, not {matrix!r}")
        rows = [self.rows[f"l{layer}.{p}{'.scale' if scales else ''}"] for p in PARTS[matrix]]
        return [(r["offset"], r["nbytes"]) for r in rows]

    def shape(self, layer: int, matrix: str) -> tuple[int, int]:
        """`(N, K)`: its tensors' rows stacked, their common row length."""
        ss = [self.rows[f"l{layer}.{p}"]["shape"] for p in PARTS[matrix]]
        if len({s[1] for s in ss}) != 1:
            raise ValueError(f"{matrix}'s tensors {PARTS[matrix]} have different row lengths")
        return sum(s[0] for s in ss), ss[0][1]

    def tensor_parts(self, t: Tensor) -> list[tuple[int, int]]:
        """`(offset, nbytes)` in the stream of store tensor `t`'s stream tensors (`Tensor.stream_names`), stacked; refused
        unless they stack to `t`'s [N, K]."""
        rows = [self.rows[n] for n in t.stream_names]
        if len({r["shape"][1] for r in rows}) != 1 or (sum(r["shape"][0] for r in rows), rows[0]["shape"][1]) != (t.n, t.k):
            raise ValueError(f"{t.name} is [{t.n}, {t.k}] in the store, {[r['shape'] for r in rows]} in the stream")
        return [(r["offset"], r["nbytes"]) for r in rows]

    def span(self, layer: int) -> tuple[int, int]:
        """The stream bytes [lo, hi) holding the layer's matrices and their scales."""
        ps = [p for m in PARTS for s in (False, True) for p in self.parts(layer, m, s)]
        return min(o for o, _ in ps), max(o + n for o, n in ps)

    @staticmethod
    def read(parts: list[tuple[int, int]], lo: int, hi: int, read: Read) -> bytes:
        """Bytes [lo, hi) of `parts` one after another, each read from the stream by `read(offset, size)`."""
        out, at = bytearray(), 0
        for off, n in parts:
            a, b = max(lo, at), min(hi, at + n)
            if a < b:
                out += read(off + a - at, b - a)
            at += n
        if len(out) != hi - lo:
            raise ValueError(f"bytes [{lo}, {hi}) run past the {at} the parts hold")
        return bytes(out)


def reader(path: Path) -> Read:
    """The stream's bytes from a file holding them at their stream offsets (`weights write`)."""
    def read(offset: int, size: int) -> bytes:
        with open(path, "rb") as f:
            data = os.pread(f.fileno(), size, offset)
        if len(data) != size:
            raise ValueError(f"{path} ends before byte {offset + size}")
        return data
    return read


def seed_bytes(text: str) -> bytes:
    """The generator's ``--seed`` form: ``text:...`` or ``hex:...``."""
    kind, _, v = text.partition(":")
    if kind == "text":
        return v.encode()
    if kind == "hex":
        return bytes.fromhex(v)
    raise ValueError(f"a seed is text:... or hex:..., not {text!r}")


def generator_reader(module: str, shape: str, seed: str) -> Read:
    """The stream regenerated by the seeded generator (`benchmarks.torch_gpu.p2_stage.served_weights.stream`), when no stream file is at
    hand."""
    w = importlib.import_module(module)
    sh, sd = w.ALIASES.get(shape) or w.SHAPES[shape], seed_bytes(seed)

    def read(offset: int, size: int) -> bytes:
        data = w.stream(sh, sd, offset, offset + size)
        if len(data) != size:
            raise ValueError(f"the stream ends before byte {offset + size}")
        return data
    return read


def payloads(lay: StoreLayout, src: Weights, read: Read, j0: int, n: int) -> bytes:
    """The payloads of blocks `j0 .. j0 + n - 1` (2,054 bytes each): the store's W, each tensor's bytes from the stream,
    zero past every tensor's end."""
    lo, hi = j0 * PAYLOAD, (j0 + n) * PAYLOAD
    out = bytearray(hi - lo)
    for t in lay.tensors:
        a, b = max(lo, t.offset), min(hi, t.offset + t.size)
        if a >= b:
            continue
        out[a - lo:b - lo] = src.read(src.tensor_parts(t), a - t.offset, b - t.offset, read)
    return bytes(out)


def store_salt(setup: dict, layout_doc: dict, layout_sha512: str, source: bytes, commitment: bytes) -> bytes:
    """The salt an encoded store is under (`p2_v1/store.py encode`'s directory, whose `setup.json` is `setup`):
    `setup.draw_salt(source, commitment)`, the auditor's coin `source` flipped after the store's commitment. Refused unless
    the store commits to `commitment` in this layout (`layout_sha512`, its JSON's SHA-512) under its parameters, and
    under this coin and salt where `setup.json` records them."""
    from verity.protocols.pous.verifier import setup as P

    salt = P.draw_salt(source, commitment)
    want = {"commitment": commitment.hex(), "layout_sha512": layout_sha512, "manifest": layout_doc.get("params"),
            "source": source.hex(), "salt": salt.hex()}
    bad = sorted(k for k, v in want.items() if (k in setup or k not in ("source", "salt")) and setup.get(k) != v)
    if bad:
        raise ValueError(f"the store's setup.json is not this layout's, coin's and commitment's: {bad} differ")
    return salt


def weights_name(manifest: dict) -> str | None:
    """A weights manifest's name of its weights: its `model` (the generator's `model_id`), or, in a manifest with no
    `model`, its `program`, the key the sessions' pinned manifest (`full.sh`'s and `session.sh`'s `MANIFEST_SHA256`)
    holds that name under. None where it gives neither, as the generator's manifest with no seed does."""
    name = manifest["model"] if "model" in manifest else manifest.get("program")
    return name if isinstance(name, str) and name else None


def open_store(layout_path: Path, manifest: dict | None, encoded: Path | None, source: str | None,
               commitment: str | None, salt: str | None) -> tuple[StoreLayout, dict | None]:
    """The layout at `layout_path` under its store's prefix, and the store's `setup.json`: an encoded store's
    (`encoded`, under `store_salt` of the hex `source` and `commitment`), or a development store's under the public test
    salt `salt` (hex), with no setup. An encoded store is held to the weights `manifest` its caller checks the load
    against: refused unless its `setup.json`'s `program` is the manifest's name of its weights (`weights_name`), and so
    is a manifest that gives no name. `manifest` is None only for a caller that holds nothing to a weights manifest."""
    doc = json.loads(layout_path.read_text())
    setup = None
    if encoded is not None:
        if not (source and commitment):
            raise ValueError("an encoded store needs its coin and commitment")
        setup = json.loads((encoded / "setup.json").read_text())
        if manifest is not None:
            name = weights_name(manifest)
            if name is None:
                raise ValueError("the weights manifest names no weights (no `model`: the generator writes it under "
                                 "--seed), so nothing holds the store to it")
            if setup.get("program") != name:
                raise ValueError(f"the store holds {setup.get('program')}, the weights manifest is {name}")
        key = store_salt(setup, doc, hashlib.sha512(layout_path.read_bytes()).hexdigest(), bytes.fromhex(source),
                         bytes.fromhex(commitment))
    elif salt:
        key = bytes.fromhex(salt)
    else:
        raise ValueError("a development store needs its test salt")
    return store_layout(doc, registered_prefix(key)), setup


def segment_path(root: Path, s: int, ext: str) -> Path:
    """Segment `s`'s `c` (its blocks' rows in order) or `json` (written last: its digests) in an encoded store."""
    return root / "seg" / f"{s:05d}.{ext}"


def segment_rows(root: Path, spec: R.Params, j0: int, n: int) -> bytes:
    """The rows of blocks `j0 .. j0 + n - 1` (2,056 bytes each) from an encoded store's segment files."""
    out, bps = bytearray(), spec.blocks_per_segment
    j, stop = j0, j0 + n
    while j < stop:
        s, v = divmod(j, bps)
        k = min(stop - j, bps - v)
        with open(segment_path(root, s, "c"), "rb") as f:
            data = os.pread(f.fileno(), k * ROW, v * ROW)
        if len(data) != k * ROW:
            raise ValueError(f"segment {s} ends before block {j + k}")
        out += data
        j += k
    return bytes(out)


def check_blocks(params: S.P2Params, prefix: bytes, first: int, rows: bytes | memoryview,
                 payloads: bytes | memoryview) -> list[tuple[int, str]]:
    """The blocks `first ..` of `rows` (2,056 bytes each) whose reference decode (`p2_v1/store.py`'s `decode_block`) is
    refused or differs from their 2,054 bytes of `payloads`, with why."""
    from p2_v1.store import decode_block

    n = len(rows) // ROW
    if len(rows) != n * ROW or len(payloads) != n * PAYLOAD:
        raise ValueError(f"{len(rows)} bytes of rows and {len(payloads)} of payload are not the same whole blocks")
    bad = []
    for i in range(n):
        try:
            got = decode_block(params, prefix, first + i, bytes(rows[i * ROW:(i + 1) * ROW]))
        except ValueError as e:
            bad.append((first + i, str(e)))
            continue
        if got != bytes(payloads[i * PAYLOAD:(i + 1) * PAYLOAD]):
            bad.append((first + i, "decodes to other bytes than W's"))
    return bad
