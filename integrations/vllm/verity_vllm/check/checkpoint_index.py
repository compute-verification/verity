"""The safetensors shards of a checkpoint of record: every tensor's shard, dtype, shape and byte range, read by positioned reads (the
whole tensor, or a row range of it), and the shard pin -- each shard's sha256 against the one `manifests/checkpoints.json` records.

The pin is a function of the files and the manifest alone, so a TP driver takes it once (`shard_pin`) and hands it to every rank's
`CheckpointIndex(pin=...)`: the ranks record the pin they would have hashed themselves, without reading the checkpoint eight times over.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

#: safetensors dtype -> (numpy view dtype, element bytes)
_ST_DTYPE = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4), "F64": (np.float64, 8),
             "I64": (np.int64, 8), "I32": (np.int32, 4), "I16": (np.int16, 2), "I8": (np.int8, 1), "U8": (np.uint8, 1),
             "BOOL": (np.uint8, 1), "F8_E4M3": (np.uint8, 1), "F8_E5M2": (np.uint8, 1)}

#: shards hashed at once by `shard_pin` (hashlib releases the GIL, so threads hash in parallel; the bound is the file system)
PIN_WORKERS = 16


def read_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def shard_names(ckpt_dir: str) -> list[str]:
    return sorted(fn for fn in os.listdir(ckpt_dir) if fn.endswith(".safetensors"))


def manifest_entry(manifest: str | None, ckpt_dir: str, model: str | None, revision: str | None) -> dict[str, Any] | None:
    """The checkpoints.json entry of `ckpt_dir`: the one whose local_path is the dir, else the one of `model` at `revision`."""
    if not (manifest and os.path.exists(manifest)):
        return None
    for c in json.load(open(manifest)).get("checkpoints", []):
        lp = str(c.get("local_path") or "")
        if (lp and os.path.realpath(lp) == os.path.realpath(ckpt_dir)) or \
                (model and c.get("repo") == model and (not revision or str(c.get("revision", "")).startswith(revision[:10]))):
            return c
    return None


def shard_pin(ckpt_dir: str, *, checkpoints_manifest: str | None, model: str | None, revision: str | None, workers: int | None = None,
              shards: list[str] | None = None) -> dict[str, Any]:
    """{manifest, entry, shards: {shard: {sha256, pinned_sha256, pinned}}} for every shard of `ckpt_dir`, in shard order: `pinned` is
    True when the sha256 is the entry's, False when it differs, None when the manifest has no entry or no sha256 for the shard."""
    shards = shard_names(ckpt_dir) if shards is None else list(shards)
    entry = manifest_entry(checkpoints_manifest, ckpt_dir, model, revision)
    n = max(1, min(workers or PIN_WORKERS, len(shards) or 1))
    with ThreadPoolExecutor(n) as ex:
        shas = list(ex.map(lambda fn: sha256_file(os.path.join(ckpt_dir, fn)), shards))
    pin: dict[str, Any] = {"manifest": checkpoints_manifest, "entry": None if entry is None else {"repo": entry.get("repo"), "revision": entry.get("revision")},
                           "shards": {}}
    for fn, sha in zip(shards, shas):
        pinned = ((entry or {}).get("files") or {}).get(fn, {}).get("sha256") if entry else None
        pin["shards"][fn] = {"sha256": sha, "pinned_sha256": pinned, "pinned": (sha == pinned) if pinned else None}
    return pin


class CheckpointIndex:
    """Every tensor of every safetensors shard under `ckpt_dir`: name -> (shard, dtype, shape, byte range).  With `checkpoints_manifest`
    (manifests/checkpoints.json) the shard sha256s are verified against the pinned entry by name; `pin` is that verification taken by
    another process on the same files and manifest (`shard_pin`), adopted instead of hashing again."""

    def __init__(self, ckpt_dir: str, *, checkpoints_manifest: str | None = None, model: str | None = None, revision: str | None = None,
                 verify_shards: bool = True, pin: dict[str, Any] | None = None, workers: int | None = None):
        self.dir = ckpt_dir
        self.tensors: dict[str, tuple[str, str, list[int], int, int]] = {}
        self.shards: list[str] = []
        self.config = {}
        cfg = os.path.join(ckpt_dir, "config.json")
        if os.path.exists(cfg):
            self.config = json.load(open(cfg))
        for fn in shard_names(ckpt_dir):
            hdr, base = read_safetensors_header(os.path.join(ckpt_dir, fn))
            self.shards.append(fn)
            for name, meta in hdr.items():
                if name == "__metadata__":
                    continue
                lo, hi = meta["data_offsets"]
                self.tensors[name] = (fn, meta["dtype"], [int(x) for x in meta["shape"]], base + int(lo), base + int(hi))
        if not self.shards:
            raise FileNotFoundError(f"no *.safetensors shard under {ckpt_dir}")
        self.pin: dict[str, Any] = {"manifest": checkpoints_manifest, "entry": None, "shards": {}}
        if pin is not None:
            if sorted(pin.get("shards") or {}) != self.shards:
                raise ValueError(f"{ckpt_dir}: the pin covers shards {sorted(pin.get('shards') or {})[:4]}, the checkpoint has {self.shards[:4]} "
                                 f"({len(pin.get('shards') or {})} vs {len(self.shards)})")
            if pin.get("manifest") != checkpoints_manifest:
                raise ValueError(f"{ckpt_dir}: the pin was taken against manifest {pin.get('manifest')}, this index reads {checkpoints_manifest}")
            self.pin = {"manifest": checkpoints_manifest, "entry": pin.get("entry"), "shards": {fn: dict(pin["shards"][fn]) for fn in self.shards}}
        elif verify_shards:
            self.pin = shard_pin(ckpt_dir, checkpoints_manifest=checkpoints_manifest, model=model, revision=revision, workers=workers, shards=self.shards)

    def has(self, name: str) -> bool:
        return name in self.tensors

    def array(self, name: str, rows: tuple[int, int] | None = None) -> np.ndarray:
        """The tensor (raw words), or its rows [lo, hi) along dim 0: a contiguous sub-range of its bytes, read by itself.  A row read
        is refused with the whole read's words when the whole tensor is not on disk, so what a rank records never depends on its rows."""
        fn, dt, shape, lo, hi = self.tensors[name]
        npdt, isz = _ST_DTYPE[dt]
        n = 1
        for s in shape:
            n *= s
        with open(os.path.join(self.dir, fn), "rb") as f:     # a positioned read: a whole-shard memmap keeps every page it touched resident
            if rows is None:
                raw = f.read(hi - lo) if f.seek(lo) == lo else b""
            else:
                r0, r1 = (int(x) for x in rows)
                if not shape or not 0 <= r0 <= r1 <= shape[0]:
                    raise ValueError(f"{name}: rows [{r0}, {r1}) outside dim 0 of shape {shape}")
                size = os.fstat(f.fileno()).st_size
                if size < hi or hi - lo != n * isz:
                    raise ValueError(f"{name}: {max(0, min(size, hi) - lo)} bytes for shape {shape} dtype {dt}")
                row = (n // shape[0]) * isz if shape[0] else 0
                f.seek(lo + r0 * row)
                raw = f.read((r1 - r0) * row)
                shape, n = [r1 - r0] + list(shape[1:]), (r1 - r0) * (n // shape[0] if shape[0] else 0)
        if len(raw) != n * isz:
            raise ValueError(f"{name}: {len(raw)} bytes for shape {shape} dtype {dt}")
        return np.frombuffer(raw, dtype=npdt).reshape(shape) if shape else np.frombuffer(raw, dtype=npdt).reshape(())

    def dtype(self, name: str) -> str:
        return self.tensors[name][1]
