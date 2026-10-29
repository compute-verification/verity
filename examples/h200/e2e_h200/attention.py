"""MQA attention at one position, on the CPU through the Definitions' own numpy kernels: the stand-in for kernels' attention
kernel until it is handed over (a BF16 tensor-core product computed by torch is another function).

    scores   GemmCoordinate_v2{K=d, GEMM_HOPPER} (`HopperBF16WgmmaDot16`'s 16-wide steps in ascending k from +0, then
             `F2fpBf16`): query head h of slot b against the KV head's key at each position s <= t
    softmax  SoftmaxRow_v1{N = t + 1} (circuits' plan, section 1): m the first greatest score,
             e_j = MufuEx2Ftz(RN(RN(x_j - m) * RN32(log2 e))), s their sum from +0 in ascending j, r = RN(1 / s),
             p_j = bf16(RN(e_j * r))
    pv       GemmCoordinatePad_v1{T = t + 1, K = 32}: p and the value column padded with +0 to K = 32, then
             GemmCoordinate_v2{K=32, GEMM_HOPPER}

Words are numpy u16 (bf16) and u32 (f32); everything is evaluated by `evaluation.evaluate_batch` or numpy float32 ufuncs, each
one IEEE round to nearest.

`Attention` runs a micro-batch of slots, each at its own position (``lens`` = position + 1 keys), on the tensors' device:
on CUDA through attn_bf16's reference kernels (`attn.qk` and `attn.pv` with ``path="ref"``, the kind ``attn_ref``: QK^T
and PV, each word ``GemmCoordinate_v2{K, GEMM_HOPPER}``, PV's K the keys padded to a multiple of 16, which equals the
K = 32 padding: both pad with +0 codes) and `softmax_t` (SoftmaxRow_v1 in
torch integer and f32 ops); elsewhere through the numpy kernels above, one call per distinct length. Scores and
probabilities come out (B, H, C), zero past each slot's length. The step commits a head's probabilities, padded with +0 to 32
keys, as one ``QkSoftmaxRowPad_v1{T=32, D}`` unit and its PV row as one ``PvRow_v1{T=32}`` (`plan`): the same words.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import mufu as MU

PAD_K = 32
LOG2E = np.float32(1.4426950408889634)


def qk_template(d: int):
    return bind(G.GemmCoordinate, K=d, **G.GEMM_HOPPER)


def pv_template():
    return bind(G.GemmCoordinate, K=PAD_K, **G.GEMM_HOPPER)


def f32_of_bf16(w: np.ndarray) -> np.ndarray:
    return (np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_of_f32(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def scores(q: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """q (B, H, d) u16, keys (B, T, d) u16 -> (B, H, T) u16."""
    b, h, d = q.shape
    t = keys.shape[1]
    x = np.broadcast_to(q[:, :, None, :], (b, h, t, d)).reshape(-1, d)
    w = np.broadcast_to(keys[:, None, :, :], (b, h, t, d)).reshape(-1, d)
    return np.asarray(evaluate_batch(qk_template(d), x, w), dtype=np.uint16).reshape(b, h, t)


@dataclass
class Softmax:
    probs: np.ndarray    # (B, H, T) u16
    m: np.ndarray        # (B, H) u16: the first greatest score's word
    s: np.ndarray        # (B, H) u32: the sum of the exponentials
    r: np.ndarray        # (B, H) u32: RN(1 / s)


def softmax(x: np.ndarray) -> Softmax:
    """x (B, H, T) u16 scores -> the probabilities and the reductions' words."""
    xf = f32_of_bf16(x)
    first = np.argmax(xf, axis=-1)
    m = np.take_along_axis(np.asarray(x, dtype=np.uint16), first[..., None], -1)[..., 0]
    y = ((xf - f32_of_bf16(m)[..., None]) * LOG2E).astype(np.float32)
    e = MU.mufu_ex2_words(y.view(np.uint32), MU.table("ex2")).view(np.float32)
    s = np.zeros(x.shape[:-1], dtype=np.float32)
    for j in range(x.shape[-1]):
        s = (s + e[..., j]).astype(np.float32)
    r = (np.float32(1) / s).astype(np.float32)
    p = bf16_of_f32((e * r[..., None]).astype(np.float32))
    return Softmax(p, m, s.view(np.uint32), r.view(np.uint32))


def pad(a: np.ndarray, k: int = PAD_K) -> np.ndarray:
    out = np.zeros((*a.shape[:-1], k), dtype=np.uint16)
    out[..., :a.shape[-1]] = a
    return out


def pv(p: np.ndarray, values: np.ndarray) -> np.ndarray:
    """p (B, H, T) u16, values (B, T, d) u16 -> (B, H, d) u16."""
    b, h, t = p.shape
    d = values.shape[-1]
    x = np.broadcast_to(pad(p)[:, :, None, :], (b, h, d, PAD_K)).reshape(-1, PAD_K)
    w = np.broadcast_to(pad(np.swapaxes(values, 1, 2))[:, None, :, :], (b, h, d, PAD_K)).reshape(-1, PAD_K)
    return np.asarray(evaluate_batch(pv_template(), x, w), dtype=np.uint16).reshape(b, h, d)


# --- per-slot lengths, on the tensors' device ----------------------------------------------------------------------------

ATTN = Path(__file__).resolve().parents[3] / "benchmarks" / "torch_gpu" / "attn_bf16" / "attn.py"
_ONE, _INF, _NAN = 0x3F800000, 0x7F800000, 0x7FFFFFFF


@lru_cache(maxsize=None)
def ex2_table(device: Any) -> torch.Tensor:
    return torch.from_numpy(MU.table("ex2").astype(np.int64)).to(device)


def ex2_t(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """`MU.mufu_ex2_words` in torch: x int64 f32 words (0 .. 2^32 - 1) -> int64 words."""
    ax = x & 0x7FFFFFFF
    e = ax >> 23
    neg = (x >> 31) != 0
    mant = (ax & 0x7FFFFF) | 0x800000
    unb = e - 127
    k = torch.where(unb >= 0, mant << unb.clamp(0, 29), mant >> (-unb).clamp(0, 24))
    fixed = torch.where(~neg, k, torch.where((k & 0x7FFFFF) == 0, -k, -k - 1))
    t = table[fixed & 0x7FFFFF]
    re = ((t >> 23) & 0xFF) + (fixed >> 23)
    out = torch.where(re >= 0xFF, _INF, torch.where(re <= 0, 0, (re.clamp(0, 0xFF) << 23) | (t & 0x7FFFFF)))
    out = torch.where(e == 0, _ONE, out)
    out = torch.where(unb >= 30, torch.where(neg, 0, _INF), out)
    return torch.where((e == 0xFF) & ((ax & 0x7FFFFF) != 0), _NAN, out)


def _f32(words: torch.Tensor) -> torch.Tensor:
    """int64 words 0 .. 2^32 - 1 as f32."""
    return torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32).view(torch.float32)


def _words(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def softmax_t(scores: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """`softmax` with each row's first ``lens`` scores: scores (B, H, C) bf16, lens (B,) -> probabilities (B, H, C) bf16,
    +0 past the length."""
    c = scores.shape[-1]
    xf = scores.float()
    valid = torch.arange(c, device=scores.device)[None, None, :] < lens.to(scores.device)[:, None, None]
    first = torch.argmax(torch.where(valid, xf, float("-inf")), dim=-1, keepdim=True)
    m = torch.gather(xf, -1, first)
    y = (xf - m) * torch.tensor(LOG2E, dtype=torch.float32, device=xf.device)
    e = torch.where(valid, _f32(ex2_t(_words(y), ex2_table(scores.device))), 0.0)
    s = torch.zeros(e.shape[:-1], dtype=torch.float32, device=e.device)
    for j in range(c):
        s = s + e[..., j]
    r = (1.0 / s.double()).float()
    return torch.where(valid, (e * r[..., None]).to(torch.bfloat16), torch.zeros((), dtype=torch.bfloat16, device=e.device))


@lru_cache(maxsize=1)
def attn_bf16() -> Any:
    """attn_bf16's launches (`benchmarks/torch_gpu/attn_bf16/attn.py`), or None when its file or sm_90a cubin isn't here."""
    if not ATTN.exists():
        return None
    spec = importlib.util.spec_from_file_location("attn", ATTN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["attn"] = mod
    spec.loader.exec_module(mod)
    return mod if Path(mod.cubin_path()).exists() else None


def _host_words(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _dev(w: np.ndarray, device: Any) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(w).view(np.int16)).to(device).view(torch.bfloat16)


class Attention:
    """One micro-batch's attention on ``device``: ``kind`` is ``attn_ref`` (CUDA, attn_bf16's reference kernels from its
    cubin) or ``numpy``."""

    def __init__(self, device: Any, kind: str | None = None) -> None:
        self.device = torch.device(device)
        if kind is None:
            kind = "attn_ref" if self.device.type == "cuda" and attn_bf16() is not None else "numpy"
        if kind == "attn_ref" and (self.device.type != "cuda" or attn_bf16() is None):
            raise ValueError(f"attn_ref runs on CUDA with attn_bf16's cubin ({ATTN.parent}/build.sh, $ATTN_BF16_CUBIN)")
        self.kind = kind

    def __call__(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, lens: torch.Tensor):
        """q (B, H, d), keys and values (B, C, d) bf16, lens (B,) -> scores, probabilities (B, H, C) and PV (B, H, d)."""
        if self.kind == "attn_ref":
            ab, ln = attn_bf16(), lens.to(device=q.device, dtype=torch.int32)
            s = ab.qk(q.contiguous(), keys.contiguous(), ln, fill=0, check_lens=False, path="ref")
            p = softmax_t(s, ln)
            return s, p, ab.pv(p, values.contiguous(), ln, check_lens=False, path="ref")
        b, h, d = q.shape
        c = keys.shape[1]
        qw, kw, vw, ln = _host_words(q), _host_words(keys), _host_words(values), lens.cpu().numpy()
        sc, pr, o = (np.zeros((b, h, c), np.uint16), np.zeros((b, h, c), np.uint16), np.zeros((b, h, d), np.uint16))
        for n in np.unique(ln):
            i = np.flatnonzero(ln == n)
            s = scores(qw[i], kw[i, :n])
            p = softmax(s).probs
            sc[i, :, :n], pr[i, :, :n], o[i] = s, p, pv(p, vw[i, :n])
        return _dev(sc, q.device), _dev(pr, q.device), _dev(o, q.device)
