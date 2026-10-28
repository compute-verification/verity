"""Independent CPU model of vLLM's FlashAttention-2 forward (Ada/Ampere, BF16, head dim 128), W11.

The arithmetic is in ``cpp/fa2_model.cpp`` (which reuses W4b's ``tc_model.cpp`` tensor-core model);
this module builds/loads it, reconstructs the two MUFU tables from the shipped delta files and exposes

* :func:`attention_rows` -- the model for a batch of (query row, head) pairs against canonical KV rows;
* :func:`kernel_geometry` -- the kernel's per-row block range / visible-column limit for the causal
  (prefill) and the ``seqlen_q == 1`` (decode, ``is_causal`` forced false, ngroups-swapped) launches;
* :func:`scale_log2` -- ``params.scale_softmax_log2`` exactly as ``flash_api.cpp`` derives it.

Domain (what was measured): kernel
``flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, bf16>, Is_causal,
Is_local=false, Has_alibi=false, Is_even_MN=false, Is_even_K=true, Is_softcap=false, Split=false,
Append_KV=false>`` (vllm-flash-attn 506341a, ``--use_fast_math``), num_splits=1, window (-1,-1),
softcap 0, no alibi, BF16 in/out, MUFU tables measured on the RTX 4090 (sm_89).
"""

from __future__ import annotations

import ctypes
import hashlib
import lzma
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verity.ml.definitions.numerics import _jit

HERE = Path(__file__).resolve().parent
CPP_SRC = HERE / "cpp" / "fa2_model.cpp"
CPP_DEP = HERE / "cpp" / "tc_model.cpp"
LIB = _jit.lib_path("libfa2_model.so")                 # build product (the _jit cache, outside the checkout), compiled on first use

M_LOG2E = 1.4426950408889634  # <math.h> M_LOG2E as a double
KBLOCK_M = 64
KBLOCK_N = 64
HEAD_DIM = 128

KERNEL = {
    "source": "vllm-flash-attn (vllm-project/flash-attention) commit 506341a143fcabd4bb79052a7605ada727d6b3f5, csrc/flash_attn (FA2)",
    "build": "--use_fast_math (ftz, MUFU.EX2 for exp2f, MUFU.RCP for 1.f/x); sm_80 cubin run on sm_89",
    "kernel": "flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>, "
              "Is_causal, Is_local=false, Has_alibi=false, Is_even_MN=false, Is_even_K=true, Is_softcap=false, Split=false, Append_KV=false>",
    "mma": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (veritor.core.silicon AMPERE_BF16_M16N8K16)",
    "block": {"kBlockM": KBLOCK_M, "kBlockN": KBLOCK_N, "kNWarps": 4, "head_dim": HEAD_DIM},
}

_lib = None


def build(force: bool = False) -> Path:
    """Compile cpp/fa2_model.cpp (g++ or $CXX; OpenMP when available, serial otherwise).  ``-ffp-contract=off``
    is essential: the model spells out which multiply-adds the GPU fused; the host compiler must not fuse others.
    ``_jit``: atomic publish + builder lock, so a concurrent first user never dlopens a half-written .so."""
    return _jit.atomic_build(LIB, [CPP_SRC, CPP_DEP], lambda out: _jit.cxx_compile(out, CPP_SRC, cwd=CPP_SRC.parent), force=force)


def _load() -> ctypes.CDLL:
    try:
        return ctypes.CDLL(str(build()))
    except OSError:  # a stale or foreign-platform .so (e.g. copied from another machine): rebuild here
        return ctypes.CDLL(str(build(force=True)))


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load()
        _lib.fa2_rows.restype = None
        _lib.fa2_rows.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 3
            + [ctypes.c_int, ctypes.c_int, ctypes.c_float] + [ctypes.c_void_p] * 9 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        )
        for name in ("mufu_ex2_words", "mufu_rcp_words"):
            fn = getattr(_lib, name)
            fn.restype = None
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
    return _lib


# ---- MUFU tables --------------------------------------------------------------------------------

TABLE_N = 1 << 23
EX2_T0 = 0x3F800000  # ex2(0) = 1.0
RCP_R0 = 0x3F800000  # rcp(1.0) = 1.0


@dataclass(frozen=True)
class MufuTables:
    ex2: np.ndarray  # (2^23,) u32: hw ex2.approx.ftz of j / 2^23
    rcp: np.ndarray  # (2^23,) u32: hw rcp.approx.ftz of 1 + m / 2^23
    source: dict

    @staticmethod
    def load(directory: str | Path) -> "MufuTables":
        """From ``mufu_ex2_delta_int8.xz`` / ``mufu_rcp_delta_int8.xz``: int8 first differences of the
        u32 tables (platform-independent reconstruction by cumulative sum)."""
        d = Path(directory)
        src = {}
        tabs = []
        for name, t0 in (("ex2", EX2_T0), ("rcp", RCP_R0)):
            p = d / f"mufu_{name}_delta_int8.xz"
            raw = p.read_bytes()
            delta = np.frombuffer(lzma.decompress(raw), dtype=np.int8)
            if delta.shape[0] != TABLE_N - 1:
                raise ValueError(f"{p}: expected {TABLE_N - 1} deltas, got {delta.shape[0]}")
            t = np.empty(TABLE_N, dtype=np.int64)
            t[0] = t0
            np.cumsum(delta.astype(np.int64), out=t[1:])
            t[1:] += t0
            tabs.append(t.astype(np.uint32))
            src[name] = {"file": p.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return MufuTables(tabs[0], tabs[1], src)


def mufu_ex2(x: np.ndarray, tables: MufuTables) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    y = np.empty_like(x)
    lib().mufu_ex2_words(x.ctypes.data, y.ctypes.data, x.size, tables.ex2.ctypes.data)
    return y.view(np.float32)


def mufu_rcp(x: np.ndarray, tables: MufuTables) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    y = np.empty_like(x)
    lib().mufu_rcp_words(x.ctypes.data, y.ctypes.data, x.size, tables.rcp.ctypes.data)
    return y.view(np.float32)


# ---- kernel parameters --------------------------------------------------------------------------


def scale_log2(softmax_scale: float) -> np.float32:
    """``set_params_fprop(float softmax_scale)``: ``params.scale_softmax_log2 = softmax_scale * M_LOG2E``
    -- the Python double is narrowed to float at the C++ boundary, multiplied by the double constant,
    and the double product is stored in a float."""
    return np.float32(np.float64(np.float32(softmax_scale)) * np.float64(M_LOG2E))


def kernel_geometry(seqlen_q: int, seqlen_k: int, rows: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per query row ``r`` (0-based in the sequence): (visible columns, n_block_max) as the kernel computes
    them.  ``seqlen_q == 1``: ``flash_api.cpp`` forces ``is_causal = false`` (all ``seqlen_k`` keys visible);
    otherwise causal with the mask aligned bottom-right (``col_idx_limit = min(seqlen_k, row + 1 +
    seqlen_k - seqlen_q)``) and ``n_block_max = min(ceil(seqlen_k/kBlockN), ceil(((m_block+1)*kBlockM +
    seqlen_k - seqlen_q) / kBlockN))``."""
    rows = np.asarray(rows, dtype=np.int64)
    nb_all = -(-seqlen_k // KBLOCK_N)
    if seqlen_q == 1:
        return np.full(rows.shape, seqlen_k, np.int32), np.full(rows.shape, nb_all, np.int32)
    off = seqlen_k - seqlen_q
    col_limit = np.minimum(seqlen_k, rows + 1 + off)
    m_block = rows // KBLOCK_M
    nb = np.minimum(nb_all, -(-((m_block + 1) * KBLOCK_M + off) // KBLOCK_N))
    return col_limit.astype(np.int32), nb.astype(np.int32)


def _nudge(x: np.float32, ulps: int) -> np.float32:
    if not ulps:
        return x
    return (np.array([x], np.float32).view(np.int32) + np.int32(ulps)).view(np.float32)[0]


@dataclass
class RowsResult:
    out: np.ndarray        # (n, D) u16 BF16 words
    out_f32: np.ndarray    # (n, D) u32 words before the BF16 cast
    dbg: dict | None = None
    nonfinite: np.ndarray | None = None  # (n,) bool: a non-finite intermediate occurred -> the row is `unresolved`


def attention_rows(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, kvh: np.ndarray, col_limit: np.ndarray, n_block_max: np.ndarray,
    softmax_scale: float, tables: MufuTables, *, debug: bool = False, variant: int = 0, scale_log2_nudge: int = 0,
) -> RowsResult:
    """``q`` (n, D) u16 words (one head's query per row); ``k``/``v`` (Lk, KVH, D) u16 canonical KV rows;
    ``kvh`` (n,) KV head per row; ``col_limit``/``n_block_max`` from :func:`kernel_geometry`."""
    q = np.ascontiguousarray(q, dtype=np.uint16)
    k = np.ascontiguousarray(k, dtype=np.uint16)
    v = np.ascontiguousarray(v, dtype=np.uint16)
    n, D = q.shape
    Lk, KVH, D2 = k.shape
    if D != HEAD_DIM or D2 != D or v.shape != k.shape:
        raise ValueError(f"shapes: q {q.shape} k {k.shape} v {v.shape}")
    kvh = np.ascontiguousarray(kvh, dtype=np.int32)
    col_limit = np.ascontiguousarray(col_limit, dtype=np.int32)
    n_block_max = np.ascontiguousarray(n_block_max, dtype=np.int32)
    if col_limit.max(initial=0) > Lk or (n_block_max * KBLOCK_N).max(initial=0) < col_limit.max(initial=0):
        raise ValueError("col_limit exceeds the KV rows / block range")
    out = np.zeros((n, D), np.uint16)
    out_f32 = np.zeros((n, D), np.uint32)
    flags = np.zeros(n, np.uint8)
    max_blocks = int(n_block_max.max(initial=1))
    dbg = None
    ptrs = [None] * 5
    if debug:
        dbg = {
            "s": np.zeros((n, max_blocks * KBLOCK_N), np.uint32),
            "p": np.zeros((n, max_blocks * KBLOCK_N), np.uint32),
            "m": np.zeros((n, max_blocks), np.uint32),
            "l": np.zeros((n, 4), np.uint32),
            "o_pre": np.zeros((n, D), np.uint32),
        }
        ptrs = [dbg[x].ctypes.data for x in ("s", "p", "m", "l", "o_pre")]
    lib().fa2_rows(
        q.ctypes.data, k.ctypes.data, v.ctypes.data, KVH, D, kvh.ctypes.data, col_limit.ctypes.data,
        n_block_max.ctypes.data, n, KBLOCK_N, ctypes.c_float(float(_nudge(scale_log2(softmax_scale), scale_log2_nudge))),
        tables.ex2.ctypes.data, tables.rcp.ctypes.data, out.ctypes.data, out_f32.ctypes.data, *ptrs, max_blocks, int(variant),
        flags.ctypes.data,
    )
    return RowsResult(out, out_f32, dbg, (flags & 1).astype(bool))


def attention_call(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, rows: np.ndarray, seqlen_q: int, seqlen_k: int, softmax_scale: float,
    tables: MufuTables, *, debug: bool = False, variant: int = 0, scale_log2_nudge: int = 0,
) -> RowsResult:
    """One kernel call for one sequence: ``q`` (len(rows), H, D) words for the query rows ``rows``
    (0-based in the sequence), ``k``/``v`` (>= seqlen_k, KVH, D).  Returns (len(rows), H, D) words."""
    nrow, H, D = q.shape
    KVH = k.shape[1]
    ngroups = H // KVH
    rows = np.asarray(rows, dtype=np.int64)
    col_limit, nb = kernel_geometry(seqlen_q, seqlen_k, rows)
    qf = q.reshape(nrow * H, D)
    kvh = np.repeat(np.arange(H, dtype=np.int32) // ngroups, 1)[None, :].repeat(nrow, 0).reshape(-1)
    res = attention_rows(qf, k[:seqlen_k], v[:seqlen_k], kvh, np.repeat(col_limit, H), np.repeat(nb, H), softmax_scale, tables,
                         debug=debug, variant=variant, scale_log2_nudge=scale_log2_nudge)
    res.out = res.out.reshape(nrow, H, D)
    res.out_f32 = res.out_f32.reshape(nrow, H, D)
    res.nonfinite = res.nonfinite.reshape(nrow, H)
    return res


MEASURED = {
    # what the model was measured bit-exact against (natural capture + §11 stress, this lane, RTX 4090):
    "arch": ("sm_89",),                       # the sm_80 cubin executed on an RTX 4090; MUFU tables measured there
    "head_dim": (128,),
    "heads_kv_heads": ((12, 2),),             # Qwen2.5-1.5B (GQA 6:1); the KV-head index map is h // (H // KVH)
    "block_size": (16,),                      # the paged layout does not enter the arithmetic, but only 16 was run
    "seqlen_q": (1, 4096),
    "seqlen_k": (1, 4096),
    "fa_version": 2, "num_splits": 1, "window": ([-1, -1], [-1, 0]), "softcap": 0.0, "alibi": False,
    "dtype": "bfloat16", "finite_operands": True,
}


def arch_of(consts: dict, arch: str | None = None) -> str | None:
    """Declared arch: explicit argument, ``consts['arch']`` or a GPU name in ``consts['gpu'|'gpu_name']``.
    None when nothing is declared."""
    a = arch or consts.get("arch")
    if a:
        return str(a)
    name = str(consts.get("gpu") or consts.get("gpu_name") or "")
    if any(t in name for t in ("4090", "4080", "4070", "4060", "RTX 40", "L40", "L4", "6000 Ada")):
        return "sm_89"
    return None


def coverage(seqlen_q: int, seqlen_k: int, H: int, KVH: int, D: int, consts: dict, *, arch: str | None = None) -> str | None:
    """None when the call is inside the measured family, else the reason it is ``unresolved``.

    The gate is *what was measured* (:data:`MEASURED`), not what the arithmetic plausibly generalises
    to: another head_dim selects other kernel traits; another arch was not run; more than one split
    adds a combine step; a sliding window / softcap / alibi changes the softmax; other (H, KVH) pairs
    were not exercised; non-finite operands (checked by the caller on the data) exercise inf/NaN
    semantics the shared MMA model does not represent (production writes 0x7FFF for the head).
    """
    a = arch_of(consts, arch)
    if a is None:
        return "arch not declared (consts.arch / consts.gpu); measured only on sm_89"
    if a not in MEASURED["arch"]:
        return f"arch {a} not measured (measured: {MEASURED['arch']})"
    if D != HEAD_DIM:
        return f"head_dim {D} != {HEAD_DIM} (kernel traits differ)"
    if int(consts.get("fa_version", 2)) != 2:
        return "fa_version != 2"
    if int(consts.get("num_splits", 1)) != 1:
        return "num_splits != 1 (split-KV combine not modelled)"
    ws = consts.get("window_size") or consts.get("sliding_window") or [-1, -1]
    if list(ws) not in MEASURED["window"]:
        return "sliding window not modelled"
    if float(consts.get("softcap", consts.get("logits_soft_cap", 0)) or 0) != 0:
        return "softcap not modelled"
    if consts.get("alibi") or consts.get("alibi_slopes"):
        return "alibi not modelled"
    if str(consts.get("dtype", "bfloat16")).replace("torch.", "") != "bfloat16":
        return "dtype != bfloat16"
    if int(consts.get("block_size", 16)) not in MEASURED["block_size"]:
        return f"block_size {consts.get('block_size')} not measured (measured: {MEASURED['block_size']})"
    if str(consts.get("kv_cache_dtype", "auto")).replace("torch.", "") not in ("auto", "bfloat16"):
        return "kv_cache_dtype != auto/bfloat16 (quantised KV not modelled)"
    if (H, KVH) not in MEASURED["heads_kv_heads"]:
        return f"(num_heads, num_kv_heads) = ({H}, {KVH}) not measured (measured: {MEASURED['heads_kv_heads']})"
    if not (MEASURED["seqlen_k"][0] <= seqlen_k <= MEASURED["seqlen_k"][1]):
        return f"seqlen_k {seqlen_k} outside the measured range {MEASURED['seqlen_k']}"
    if not (MEASURED["seqlen_q"][0] <= seqlen_q <= MEASURED["seqlen_q"][1]) or seqlen_q > seqlen_k:
        return f"seqlen_q {seqlen_q} outside the measured range {MEASURED['seqlen_q']} (and <= seqlen_k)"
    return None


def nonfinite_words(*arrays: np.ndarray) -> int:
    """Number of BF16 words that are inf or NaN across the given u16 arrays."""
    n = 0
    for a in arrays:
        w = np.asarray(a, np.uint16)
        n += int(((w & 0x7F80) == 0x7F80).sum())
    return n
