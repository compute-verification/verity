"""Readers of a sampler launch's snapshotted arguments, for the value check of a stochastic row
(`check.replay.stoch_recompute`): a small argument's words from the snapshot store, as fp32 or as integers."""
from __future__ import annotations

import numpy as np


def _as_f32(words: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if dtype == "float16":
        return words.view(np.float16).astype(np.float32)
    if dtype == "float32":
        return words.view(np.float32)
    if dtype == "float64":
        return words.view(np.float64).astype(np.float32)
    raise ValueError(dtype)


def _small_arg(store, launch, name: str, dtype):
    from verity_vllm.observe.events import ViewDesc
    v = launch.args.get(name)
    if not isinstance(v, ViewDesc):
        return None
    snap = store.find(v, at=launch.id) or store.find(v, before=launch.id + 1)
    if snap is None:
        return None
    w = store.extract(snap, v)
    if dtype is np.float32:
        return _as_f32(w, v.dtype).reshape(-1)
    return w.reshape(-1)


def _small_int_arg(store, launch, name: str):
    from verity_vllm.observe.events import ViewDesc
    v = launch.args.get(name)
    w = _small_arg(store, launch, name, None)
    if w is None or not isinstance(v, ViewDesc):
        return None
    np_dt = {"int64": np.int64, "int32": np.int32, "uint32": np.uint32, "int16": np.int16, "uint8": np.uint8}.get(v.dtype)
    return w.view(np_dt).astype(np.int64) if np_dt is not None else None
