"""The weights' store and its decode call: the stand-in for memory-accounting's layout and decode call, on
the conventions of P2's staging decode (`benchmarks/torch_gpu/p2_stage`).

**Layout.** W is the seeded weights stream (`weights.Weights.stream`: every tensor's little-endian bytes, s8 codes, f32
scales and bf16 gains and embeddings, at `Tensor.offset`). Block j of the store carries W bytes
[2054 j, 2054 j + 2054) as bytes 2 .. 2055 of its big-endian 2,056-byte row, P2 v4 (`p2-16448/v4`,
`verity.core.protocols.pous.schemes.p2`) at the registered point, keys from `sha512_keys` under `expand_prefix(salt)`; the
last block is zero-padded. Block j is (segment, block) = divmod(j, 2^16).

**The decode call.** `Store.stage(name, dst)` writes linear ``name``'s codes, (N, K) s8, into the staging buffer ``dst`` (or a
new one) from the blocks it overlaps, and `Store.load(name)` returns any tensor (scales, gains, embeddings) the same way: on the GPU by the staging decode
(`p2_stage.stage.EncodedRows.decode_into`, memory-accounting's `libp2v1.so`) when its library's directory is given, else by
the P2 reference on the CPU. Every stage is checked against the reference's refusals (a c >= p, or a decode wider than the
payload): a refused block raises, never stages.

Stores:
* `P2Store`: the encoded rows C, in host memory or on a CUDA device; built by `encode` (the CPU reference, seconds a
  block: tiny shapes only) or loaded from a file of raw rows (`load`), the form memory-accounting's store ships in.
* `GeneratorStore`: no store; ``stage`` regenerates the tensor from the seed. The FIXTURE for running the frozen shape's
  compute before C* is on the card; the record says so.
* `FileStore`: W itself, the file network's load sink writes, read in place. Before it serves, every tensor's
  bytes are checked against the manifest's sha256 (``verity/seeded-weights/manifest/v1``, `SW.manifest`), and it refuses
  (`Mismatch`) a file whose size, model, tensors or any tensor's sha256 isn't the manifest's.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.pous.schemes import p2 as S

from .weights import SW, Weights

ROOT = Path(__file__).resolve().parents[3]
P2_STAGE = ROOT / "benchmarks" / "torch_gpu" / "p2_stage"
sys.path.insert(0, str(ROOT / "benchmarks" / "pous"))
from p2_v1 import reference as R  # noqa: E402

PARAMS = S.P2Params(*S.POINT)
SPEC = R.DEPLOYMENT
ROW, PAYLOAD = PARAMS.block_bytes, PARAMS.payload_bytes     # 2,056 and 2,054
SALT_BYTES = 24


class Refused(ValueError):
    """A block the registered reference refuses to decode."""


def blocks_of(base: int, length: int) -> range:
    return range(base // PAYLOAD, -(-(base + length) // PAYLOAD)) if length else range(0)


def nblocks(w_bytes: int) -> int:
    return -(-w_bytes // PAYLOAD)


def keys(prefix: bytes, j: int) -> tuple[int, int]:
    s, b = divmod(j, PARAMS.blocks_per_segment)
    return S.sha512_keys(PARAMS, prefix, s, b)


def encode_block(prefix: bytes, j: int, payload: bytes) -> bytes:
    k, t = keys(prefix, j)
    return R.encode(SPEC, int.from_bytes(payload.ljust(PAYLOAD, b"\0"), "big"), k, t).to_bytes(ROW, "big")


def decode_block(prefix: bytes, j: int, row: bytes) -> bytes:
    """Block j's 2,054 payload bytes; `Refused` where the registered reference refuses it."""
    c = int.from_bytes(row, "big")
    if c >= SPEC.p:
        raise Refused(f"block {j}: c >= p")
    k, t = keys(prefix, j)
    m = R.decode(SPEC, c, k, t)
    if m >> (8 * PAYLOAD):
        raise Refused(f"block {j}: decodes wider than the payload")
    return m.to_bytes(PAYLOAD, "big")


def p2_stage() -> Any:
    """The staging decode (`benchmarks/torch_gpu/p2_stage/stage.py`), loaded as the package ``p2_stage`` so its modules
    import one another as they do in their own tree."""
    if "p2_stage" not in sys.modules:
        spec = importlib.machinery.ModuleSpec("p2_stage", None, is_package=True)
        spec.submodule_search_locations = [str(P2_STAGE)]
        sys.modules["p2_stage"] = importlib.util.module_from_spec(spec)
    return importlib.import_module("p2_stage.stage")


def _encode_range(args: tuple[bytes, int, bytes]) -> bytes:
    prefix, first, w = args
    return b"".join(encode_block(prefix, first + i, w[i * PAYLOAD:(i + 1) * PAYLOAD]) for i in range(-(-len(w) // PAYLOAD)))


class Store:
    kind = "abstract"
    weights: Weights

    def stage(self, name: str, dst: Any | None = None) -> Any:
        raise NotImplementedError

    def load(self, name: str) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": self.kind}

    def close(self) -> None:
        pass


def _generate_rows(args: tuple) -> None:
    shape, seed, name, lo, hi, shm_name = args
    shm = SharedMemory(shm_name)
    try:
        rows = SW.generate(shape, seed, name, range(lo, hi))
        view = np.ndarray(rows.shape, rows.dtype, shm.buf, lo * rows[0].nbytes)
        view[...] = rows
        del view
    finally:
        shm.close()


class GeneratorStore(Store):
    """No encoded store: ``stage`` regenerates the codes (a FIXTURE for the decode call), over ``workers`` processes that
    write disjoint rows of one shared buffer (the generator is about 110 MB/s a core: the frozen shape's 130 GiB a step)."""
    kind = "generator-fixture"

    def __init__(self, weights: Weights, workers: int = 1) -> None:
        self.weights, self.workers = weights, workers
        self._pool: ProcessPoolExecutor | None = None
        self._shm: SharedMemory | None = None

    def stage(self, name: str, dst: Any | None = None) -> Any:
        rows, cols = self.weights.dims[name]
        if self.workers <= 1 or rows < 2 * self.workers:
            return _into(self.weights.codes(name), dst)
        if self._pool is None:
            self._pool = ProcessPoolExecutor(self.workers, mp_context=get_context("forkserver"))
            self._shm = SharedMemory(create=True, size=max(r * c for r, c in self.weights.dims.values()))
        step = -(-rows // (4 * self.workers))
        jobs = [(self.weights.shape, self.weights.seed, name, lo, min(lo + step, rows), self._shm.name)
                for lo in range(0, rows, step)]
        list(self._pool.map(_generate_rows, jobs))
        codes = np.ndarray((rows, cols), np.int8, self._shm.buf)
        try:
            return _into(codes, dst) if dst is not None else codes.copy()
        finally:
            del codes

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._shm.close()
            self._shm.unlink()
            self._pool = self._shm = None

    def describe(self) -> dict:
        return {"kind": self.kind, "workers": self.workers}

    def load(self, name: str) -> np.ndarray:
        return self.weights.array(name)


class P2Store(Store):
    """C's rows, (blocks, 2,056) u8, host (numpy) or device (torch); the deployment salt's key prefix; W's layout."""
    kind = "p2-v4"

    def __init__(self, weights: Weights, salt: bytes, rows: Any, stage_lib: str | None = None) -> None:
        if len(salt) != SALT_BYTES:
            raise ValueError(f"the deployment salt is {SALT_BYTES} bytes")
        n = nblocks(weights.nbytes)
        if tuple(rows.shape) != (n, ROW):
            raise ValueError(f"W of {weights.nbytes} bytes is {n} blocks of {ROW}, not {tuple(rows.shape)}")
        self.weights, self.salt, self.rows = weights, salt, rows
        self.prefix = S.expand_prefix(PARAMS, salt)
        self._stage = None
        if stage_lib:
            st = p2_stage()
            self._stage, self._refused = st.EncodedRows(st.L.StoreLayout(PARAMS, (), self.prefix), rows, 0, stage_lib), st.StoreError

    @classmethod
    def encode(cls, weights: Weights, salt: bytes, workers: int = 1, chunk_blocks: int = 16) -> P2Store:
        prefix = S.expand_prefix(PARAMS, salt)
        n, wb = nblocks(weights.nbytes), weights.nbytes
        jobs = []
        for first in range(0, n, chunk_blocks):
            lo, hi = first * PAYLOAD, min(wb, (first + chunk_blocks) * PAYLOAD)
            jobs.append((prefix, first, weights.stream(lo, hi)))
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                parts = list(ex.map(_encode_range, jobs))
        else:
            parts = [_encode_range(j) for j in jobs]
        return cls(weights, salt, np.frombuffer(b"".join(parts), dtype=np.uint8).reshape(n, ROW).copy())

    @classmethod
    def load(cls, weights: Weights, salt: bytes, path: str | Path, device: Any = None, stage_lib: str | None = None) -> P2Store:
        rows = np.fromfile(path, dtype=np.uint8).reshape(-1, ROW)
        if device is not None:
            import torch
            rows = torch.from_numpy(rows).to(device)
        return cls(weights, salt, rows, stage_lib)

    def save(self, path: str | Path) -> None:
        np.asarray(self.rows).tofile(path)

    def digest(self) -> str:
        """SHA-512 over C's rows (a FIXTURE identity of the store, not PoUS's verification root)."""
        rows = self.rows if isinstance(self.rows, np.ndarray) else self.rows.cpu().numpy()
        return hashlib.sha512(rows.tobytes()).hexdigest()

    def describe(self) -> dict:
        return {"kind": self.kind, "blocks": int(self.rows.shape[0]), "salt": self.salt.hex(),
                "decode": "p2_stage staging decode (libp2v1.so)" if self._stage else "p2-reference-cpu"}

    def block(self, j: int) -> bytes:
        r = self.rows[j]
        return bytes(r if isinstance(r, np.ndarray) else r.cpu().numpy())

    def _bytes(self, base: int, length: int) -> bytes:
        js = blocks_of(base, length)
        buf = bytearray()
        for j in js:
            buf += decode_block(self.prefix, j, self.block(j))
        lo = base - js.start * PAYLOAD
        return bytes(buf[lo:lo + length])

    def load(self, name: str) -> np.ndarray:
        t = self.weights.t[name]
        a = np.frombuffer(self._bytes(t.offset, t.nbytes), {"s8": np.int8, "f32": "<f4", "bf16": "<u2"}[t.dtype])
        return a.astype(a.dtype.newbyteorder("=")).reshape(t.shape)

    def stage(self, name: str, dst: Any | None = None) -> Any:
        n, k = self.weights.dims[name]
        base, length = self.weights.base(name), n * k
        if self._stage is not None:
            import torch
            out = (torch.empty(length, dtype=torch.uint8, device=self.rows.device) if dst is None
                   else dst.view(-1).view(torch.uint8))
            try:
                self._stage.decode_into(base, length, out)
                self._stage.check()
            except self._refused as e:
                raise Refused(f"{name}: {e}") from e
            return out.view(torch.int8).view(n, k)
        return _into(np.frombuffer(self._bytes(base, length), dtype=np.int8).reshape(n, k), dst)


class Mismatch(ValueError):
    """A weights file that isn't the manifest's W."""


#: bytes a hash update reads at once (hashlib drops the GIL on each)
HASH_CHUNK = 1 << 26


def _sha256(buf: np.ndarray) -> str:
    h = hashlib.sha256()
    for lo in range(0, buf.size, HASH_CHUNK):
        h.update(buf[lo:lo + HASH_CHUNK])
    return h.hexdigest()


class FileStore(Store):
    """W in a file (`SW.stream`'s bytes), checked against its manifest before it serves, read through a read-only map."""
    kind = "w-file"

    def __init__(self, weights: Weights, path: str | Path, manifest: str | Path | dict, workers: int = 1) -> None:
        self.weights, self.path = weights, Path(path)
        doc = manifest if isinstance(manifest, dict) else json.loads(Path(manifest).read_text())
        model = SW.model_id(weights.shape, weights.seed)
        if doc.get("format") != SW.MANIFEST or doc.get("model") != model:
            raise Mismatch(f"the manifest is {doc.get('format')!r} of {doc.get('model')!r}, not {SW.MANIFEST} of "
                           f"the served weights' {model}: serve the seed the file was written from")
        size = self.path.stat().st_size
        if not doc.get("stream_bytes") == size == weights.nbytes:
            raise Mismatch(f"{self.path} is {size} bytes, the manifest's W {doc.get('stream_bytes')}, the shape's "
                           f"{weights.nbytes}")
        want = {t["name"]: t for t in doc.get("tensors", [])}
        if list(want) != list(weights.t) or any((want[n]["offset"], want[n]["nbytes"]) != (t.offset, t.nbytes)
                                                for n, t in weights.t.items()):
            raise Mismatch(f"the manifest's tensors aren't {weights.shape.name}'s, in its stream order and offsets")
        self.w = np.memmap(self.path, np.uint8, "r")
        names = list(weights.t)
        with ThreadPoolExecutor(max(1, workers)) as ex:
            got = dict(zip(names, ex.map(lambda n: _sha256(self._raw(n)), names)))
        bad = [n for n in names if got[n] != want[n]["sha256"]]
        if bad:
            raise Mismatch(f"{len(bad)} of {len(names)} tensors of {self.path} aren't the manifest's sha256: "
                           f"{', '.join(bad[:8])}{' ...' if len(bad) > 8 else ''}")
        self.checked = len(names)

    def _raw(self, name: str) -> np.ndarray:
        t = self.weights.t[name]
        return self.w[t.offset:t.offset + t.nbytes]

    def load(self, name: str) -> np.ndarray:
        t = self.weights.t[name]
        a = self._raw(name).view({"s8": np.int8, "f32": "<f4", "bf16": "<u2"}[t.dtype])
        a = a.astype(a.dtype.newbyteorder("="))
        return a.reshape(-1) if t.rows == 1 else a.reshape(t.rows, t.cols)

    def stage(self, name: str, dst: Any | None = None) -> Any:
        return _into(self._raw(name).view(np.int8).reshape(self.weights.dims[name]), dst)

    def describe(self) -> dict:
        return {"kind": self.kind, "path": str(self.path), "tensors_checked": self.checked,
                "check": "every tensor's sha256 the manifest's"}


def _into(codes: np.ndarray, dst: Any | None) -> Any:
    """``codes`` written into the staging buffer ``dst`` (a numpy array or a torch tensor on any device), or returned."""
    if dst is None:
        return codes
    if isinstance(dst, np.ndarray):
        dst[...] = codes
    else:
        import torch
        dst.copy_(torch.from_numpy(np.require(codes, requirements=["C", "W"])))
    return dst
