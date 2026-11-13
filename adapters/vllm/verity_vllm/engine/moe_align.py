"""vLLM's `moe_align_block_size` on the Marlin MoE path, made a function of its inputs: each expert's pairs in ascending pair
index.

`fused_marlin_moe` (`CALLER`), in vLLM's module `SITE` (DeepSeek-V4-Pro's MXFP4 experts on Hopper), calls its module's
`NAME`: vLLM's wrapper (fused_moe/moe_align_block_size.py), which allocates three int32 buffers and has the op
``torch.ops._moe_C.moe_align_block_size`` (`OP`) fill them.  `install` replaces that name with `wrapper`, which allocates the
same buffers (`sorted_len`) and has `Align` fill them; it refuses a module whose `CALLER` does not load `NAME` as the
module's global, since the experts would then bypass it.  Every other MoE path keeps vLLM's op.

The op takes the router's ``topk_ids`` (``[tokens, top_k]``, any integer dtype, contiguous), ``num_experts`` E, the moe
block ``BM``, the three outputs and an optional ``expert_map``. A pair is p = t * top_k + k, a slot of ``topk_ids`` in
memory order. An id outside [0, E), or one the map sends to -1, is no expert's. Expert e's c_e pairs are written from
``base[e] = sum over e' < e of round_up(c_e', BM)``:

- ``sorted_token_ids[base[e] + r]`` is e's pair of rank r, the ranks in ascending pair index; every other slot of the
  buffer is ``numel`` (the pair count P);
- ``expert_ids[b]`` is the expert whose padded range holds slot ``b * BM`` for each block below
  ``num_tokens_post_pad / BM``, and -1 from there to ``cdiv(len(sorted_token_ids), BM)``;
- ``num_tokens_post_pad[0]`` is ``sum over e of round_up(c_e, BM)``.

vLLM's kernel at E > 64 (count_and_sort_expert_tokens_kernel) writes the same buffers except the order within an
expert, which is the order its atomicAdd on the expert's offset hands out slots, so it varies run to run. That order
decides which pairs share a Marlin block, and so where the GEMM's reductions are cut. `reference` is the definition in
Python, and the Triton kernels (`moe_align_kernels`) compute it with no atomics. A call of at most `SMALL_MAX` pairs is
one launch (`align_small`) of tiles of `TILE` pairs; a longer one is three, over chunks of `CHUNK` pairs and tiles of
`LONG_TILE`. Each program places a tile, ranking its pairs by comparing them with each other, `COLS` at a time.

It is always on, as `engine.topp_pin` is.  Importing this module `arm`\\ s it: it is installed when vLLM's Marlin module is
imported, and nothing is imported for it.  The driver (`engine.build.build_engine`) and each tensor-parallel rank
(`pipeline.tp.rank_worker`) import it before vLLM builds the model, and `register`, the distribution's vLLM general plugin
(``verity_moe_align`` in ``vllm.general_plugins``), arms it in every process of a vLLM that has the integration installed,
vLLM's own server included.  It reads no environment variable.  `installed` is the `Align` this process's Marlin experts
call; its `Align.record` counts their calls on each path.
"""
from __future__ import annotations

import dis
import importlib
import inspect
import os
import sys
import types
from collections.abc import Callable, Sequence
from typing import Any

from verity_vllm.engine import hooks

#: vLLM's Marlin experts module, the function in it that runs the experts, and the module global that function calls
SITE = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
CALLER = "fused_marlin_moe"
NAME = "moe_align_block_size"
#: the op vLLM's wrapper calls, under ``torch.ops._moe_C``
OP = "moe_align_block_size"
#: the most pairs `align_small` takes in one launch; a longer call is chunked
SMALL_MAX = 2048
#: pairs a chunk of the long path; pairs a tile (a program) of each path; the pairs each step of a tile's rank compare reads
CHUNK = 512
TILE = 32
LONG_TILE = 64
COLS = 32
#: the chunk rows and experts `scan_chunks` reads at a time, the slots a tail-store loop writes at a time, and the
#: pad slots of an expert a pad store writes at a time
SCAN_ROWS = 32
SCAN_EXPERTS = 64
STORE = 1024
PAD_STORE = 16
#: vLLM's op refuses ``padded_num_experts >= 1024`` (moe_align_sum_kernels.cu): E rounded up to 32 below 1024
MAX_EXPERTS = 992


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def next_pow2(x: int) -> int:
    return 1 << max(int(x) - 1, 0).bit_length()


def int32(v: int) -> int:
    """An id as vLLM's kernel reads it, ``int expert_id = topk_ids[idx]``: its low 32 bits, signed."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def sorted_len(pairs: int, num_experts: int, block: int, pad_sorted_ids: bool = False) -> int:
    """``len(sorted_token_ids)`` as vLLM's wrapper allocates it (fused_moe/moe_align_block_size.py)."""
    n = pairs + num_experts * (block - 1)
    if pad_sorted_ids:
        n = cdiv(n, block) * block
    return min(pairs * block, n) if pairs < num_experts else n


def reference(ids: Sequence[int], num_experts: int, block: int, expert_map: Sequence[int] | None = None,
              length: int | None = None) -> tuple[list[int], list[int], int]:
    """(``sorted_token_ids``, ``expert_ids``, ``num_tokens_post_pad``) for the flat ``ids``, each expert's pairs in ascending
    pair index; ``length`` is ``len(sorted_token_ids)`` (default vLLM's, `sorted_len`)."""
    P = len(ids)
    L = sorted_len(P, num_experts, block) if length is None else length
    pairs: list[list[int]] = [[] for _ in range(num_experts)]
    for p, raw in enumerate(ids):
        e = int32(int(raw))
        if 0 <= e < num_experts and expert_map is not None:
            e = int32(int(expert_map[e]))
        if 0 <= e < num_experts:
            pairs[e].append(p)
    sorted_ids, expert_ids, at = [P] * L, [-1] * cdiv(L, block), 0
    for e, mine in enumerate(pairs):
        sorted_ids[at:at + len(mine)] = mine
        blocks = cdiv(len(mine), block)
        expert_ids[at // block:at // block + blocks] = [e] * blocks
        at += blocks * block
    return sorted_ids, expert_ids, at


def canonical(sorted_ids: Sequence[int], expert_ids: Sequence[int], total: int, block: int, pairs: int) -> list[int]:
    """``sorted_ids`` with the pairs of each expert's padded range put in ascending order: the canonical output of any
    output vLLM's kernel can give. The run of blocks of one expert is its padded range."""
    out = list(sorted_ids)
    b, nb = 0, total // block
    while b < nb:
        end = b
        while end < nb and expert_ids[end] == expert_ids[b]:
            end += 1
        lo, hi = b * block, end * block
        mine = sorted(p for p in out[lo:hi] if p < pairs)
        out[lo:hi] = mine + [pairs] * (hi - lo - len(mine))
        b = end
    return out


class _Launch:
    """A Triton kernel, launched through its compiled handle once Triton has compiled it for a call's specialization
    (the constexprs, ``num_warps``, the device, and each tensor's dtype and 16-byte alignment). Triton's own launch
    rederives the specialization from the arguments on every call. The tensors come first in each kernel's
    signature, then the integers, then the constexprs."""

    def __init__(self, fn: Any, fast: bool) -> None:
        self.fn, self.fast = fn, fast
        self.compiled: dict[tuple, Any] = {}

    def __call__(self, grid: tuple[int, ...], tensors: tuple, ints: tuple, consts: dict, num_warps: int) -> None:
        key = (tuple(consts.values()), num_warps, tensors[0].device.index,
               tuple((t.dtype, t.data_ptr() % 16 == 0) for t in tensors))
        k = self.compiled.get(key) if self.fast else None
        if k is None:
            k = self.fn[grid](*tensors, *ints, **consts, num_warps=num_warps)
            if self.fast and hasattr(k, "__getitem__"):
                self.compiled[key] = k
        else:
            k[(grid + (1, 1))[:3]](*tensors, *ints, *consts.values())


_KERNELS: dict[str, Any] = {}


def kernels() -> dict[str, Any]:
    """The Triton kernels (`moe_align_kernels`), imported on first use: importing this module needs no Triton."""
    if not _KERNELS:
        from verity_vllm.engine import moe_align_kernels as mod

        _KERNELS.update({k: getattr(mod, k) for k in ("align_small", "count_chunks", "scan_chunks", "place_chunks")})
    return _KERNELS


def warps(pairs: int) -> int:
    return 4 if pairs <= 512 else 8 if pairs <= 2048 else 16


class Align:
    """The op: writes the canonical outputs into vLLM's buffers, on the inputs' device and stream, with no host sync.
    ``fast_launch`` launches a compiled kernel through its handle (`_Launch`)."""

    def __init__(self, small_max: int = SMALL_MAX, chunk: int = CHUNK, tile: int = TILE, long_tile: int = LONG_TILE,
                 cols: int = COLS, fast_launch: bool = True) -> None:
        for name, v in (("chunk", chunk), ("tile", tile), ("long_tile", long_tile), ("cols", cols)):
            if v < 16 or v & (v - 1):
                raise ValueError(f"{name}: a power of two of at least 16, not {v}")
        if not (cols <= tile and cols <= long_tile <= chunk):
            raise ValueError(f"cols {cols} must not exceed a tile ({tile}, {long_tile}), nor long_tile chunk {chunk}")
        self.small_max, self.chunk, self.tile, self.long_tile, self.cols = small_max, chunk, tile, long_tile, cols
        self.fast_launch = fast_launch
        self._launch: dict[str, _Launch] = {}
        self.calls = {"one": 0, "chunked": 0}

    def __call__(self, topk_ids: Any, num_experts: int, block_size: int, sorted_token_ids: Any, experts_ids: Any,
                 num_tokens_post_pad: Any, expert_map: Any = None) -> None:
        import torch

        P, L, E, BM = topk_ids.numel(), sorted_token_ids.numel(), int(num_experts), int(block_size)
        NB = cdiv(L, BM)
        if not 0 < E <= MAX_EXPERTS or BM < 1:
            raise ValueError(f"{OP}: num_experts in [1, {MAX_EXPERTS}] and block_size >= 1, not {E} and {BM}")
        if topk_ids.dtype.is_floating_point or topk_ids.dtype.is_complex or topk_ids.dtype == torch.bool:
            raise TypeError(f"{OP}: topk_ids must be an integer tensor, not {topk_ids.dtype}")
        outs = (sorted_token_ids, experts_ids, num_tokens_post_pad) + (() if expert_map is None else (expert_map,))
        if any(t.dtype != torch.int32 for t in outs):
            raise TypeError(f"{OP}: sorted_token_ids, experts_ids, num_tokens_post_pad and expert_map must be int32")
        if not all(t.is_contiguous() for t in (topk_ids, *outs)):
            raise ValueError(f"{OP}: topk_ids and the outputs must be contiguous (a pair is its slot in memory order)")
        if experts_ids.numel() < NB or num_tokens_post_pad.numel() < 1 or max(P, L) >= 1 << 30:
            raise ValueError(f"{OP}: experts_ids holds {experts_ids.numel()} blocks, {NB} needed; or {P} pairs or {L} "
                             "slots overflow the kernels' int32 offsets")
        if not self._launch:
            self._launch.update({k: _Launch(fn, self.fast_launch) for k, fn in kernels().items()})
        K, EP, has_map = self._launch, next_pow2(E + 1), expert_map is not None
        mp = expert_map if has_map else topk_ids
        bufs = (sorted_token_ids, experts_ids, num_tokens_post_pad)
        if self.small_max >= P:
            self.calls["one"] += 1
            PB = max(next_pow2(P), 16)
            T = min(self.tile, PB)
            # program 0 writes num_tokens_post_pad and the tail, so it runs even at 0 pairs
            K["align_small"]((max(cdiv(P, T), 1),), (topk_ids, mp, *bufs), (P, L, NB, E, BM),
                             {"HAS_MAP": has_map, "PB": PB, "T": T, "JB": min(self.cols, T), "EP": EP, "TB": STORE,
                              "PW": PAD_STORE}, warps(PB))
        else:
            self.calls["chunked"] += 1
            CH, T = self.chunk, self.long_tile
            C, ES = cdiv(P, CH), min(SCAN_EXPERTS, EP)
            if (2 * C + 1) * EP >= 1 << 31:
                raise ValueError(f"{OP}: {C} chunks of {CH} pairs overflow the workspace's int32 offsets")
            ws = torch.empty((2 * C + 1) * EP, dtype=torch.int32, device=topk_ids.device)
            K["count_chunks"]((C,), (topk_ids, mp, ws), (P, E), {"HAS_MAP": has_map, "CHUNK": CH, "EP": EP}, warps(CH))
            K["scan_chunks"]((EP // ES,), (ws,), (C,), {"EP": EP, "ES": ES, "CB": SCAN_ROWS}, 4)
            K["place_chunks"]((C, CH // T), (topk_ids, mp, ws, *bufs), (P, L, NB, E, BM, C),
                              {"HAS_MAP": has_map, "CHUNK": CH, "T": T, "JB": self.cols, "EP": EP, "TB": STORE,
                               "PW": PAD_STORE}, warps(CH))

    def record(self) -> dict[str, Any]:
        """This process's calls on each path, and the parameters they ran at."""
        return {"pid": os.getpid(), "calls": dict(self.calls), "small_max": self.small_max, "chunk": self.chunk,
                "tile": self.tile, "long_tile": self.long_tile, "cols": self.cols, "fast_launch": self.fast_launch}


def wrapper(align: Align) -> Callable[..., tuple[Any, Any, Any]]:
    """vLLM's wrapper (fused_moe/moe_align_block_size.py) with ``align`` for its op: the buffers it allocates, at the lengths it
    allocates them (`sorted_len`), and the expert map passed to the op when invalid experts are ignored, else applied to
    ``expert_ids`` after it."""
    import torch

    i32 = torch.int32

    def moe_align_block_size(topk_ids: Any, block_size: int, num_experts: int, expert_map: Any = None,
                             pad_sorted_ids: bool = False, ignore_invalid_experts: bool = False) -> tuple[Any, Any, Any]:
        dev = topk_ids.device
        L = sorted_len(topk_ids.numel(), num_experts, block_size, pad_sorted_ids)
        sorted_ids = torch.empty((L,), dtype=i32, device=dev)
        expert_ids = torch.empty((cdiv(L, block_size),), dtype=i32, device=dev)
        total = torch.empty((1,), dtype=i32, device=dev)
        align(topk_ids, num_experts, block_size, sorted_ids, expert_ids, total, expert_map if ignore_invalid_experts else None)
        if expert_map is not None and not ignore_invalid_experts:
            expert_ids = expert_map[expert_ids]
        return sorted_ids, expert_ids, total

    return moe_align_block_size


def loads_global(fn: Any, name: str) -> bool:
    """Whether ``fn`` (unwrapped), or a function nested in it, loads ``name`` as its module's global: what replacing the
    module's ``name`` reaches.  An attribute of the same name (``ops.name``) is not one."""
    def walk(code: types.CodeType) -> bool:
        return any(i.opname == "LOAD_GLOBAL" and i.argval == name for i in dis.get_instructions(code)) or any(
            walk(c) for c in code.co_consts if isinstance(c, types.CodeType))

    code = getattr(inspect.unwrap(fn), "__code__", None)
    return isinstance(code, types.CodeType) and walk(code)


_INSTALLED: tuple[hooks.Patch, Align] | None = None


def install(module: Any = None, align: Align | None = None) -> Align:
    """Replace `NAME` on ``module`` (`SITE`, imported when None) with `wrapper` of ``align`` (a new `Align` when None), once per
    process; returns the `Align` installed.  Refused when the module's `CALLER` does not load `NAME` as its global
    (`loads_global`): the Marlin experts would bypass the canonical op.  Refused while another patch of that name is on: a
    capture's wrap of it (`acquire.sources.moe_align_source`) goes on top of this one, never under it."""
    global _INSTALLED
    if _INSTALLED is not None and _INSTALLED[0].installed:
        return _INSTALLED[1]
    if module is None:
        module = importlib.import_module(SITE)
    if not loads_global(getattr(module, CALLER, None), NAME):
        raise RuntimeError(f"refused to install the canonical {NAME}: {module.__name__}.{CALLER} does not call the module's {NAME}, "
                           "so the Marlin experts would bypass it; install it where this vLLM's experts call their aligner")
    if on := [p for p in hooks.live() if isinstance(p, hooks.Patch) and p.owner is module and p.name == NAME]:
        raise RuntimeError(f"refused to install the canonical {NAME} under {on[-1].value!r}, which is already on: arm it before "
                           "vLLM builds the model (engine.moe_align.arm)")
    align = align or Align()
    _INSTALLED = (hooks.patch(module, NAME, wrapper(align)), align)
    return align


class _OnImport(hooks.OnImport):
    """Runs `install` on ``site`` once its own code has run."""

    def __init__(self, site: str) -> None:
        super().__init__(site, install)


def arm(site: str = SITE) -> None:
    """`install` now if ``site`` is imported, else when it is: nothing is imported here.  Once per process."""
    if site in sys.modules:
        install(sys.modules[site])
    elif not any(isinstance(f, _OnImport) and f.site == site for f in sys.meta_path):
        sys.meta_path.insert(0, _OnImport(site))


def uninstall() -> None:
    """Put vLLM's wrapper back and disarm the import hook."""
    global _INSTALLED
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _OnImport)]
    if _INSTALLED is not None:
        _INSTALLED[0].uninstall()
    _INSTALLED = None


def installed() -> Align | None:
    """The `Align` this process's Marlin experts call, or None while vLLM's wrapper is in place."""
    return _INSTALLED[1] if _INSTALLED is not None and _INSTALLED[0].installed else None


def register() -> None:
    """The distribution's vLLM general plugin (``verity_moe_align = verity_vllm.engine.moe_align:register``, group
    ``vllm.general_plugins``), which vLLM calls in each process it loads its plugins in: `arm`."""
    arm()


arm()

__all__ = ["CALLER", "MAX_EXPERTS", "NAME", "OP", "SITE", "Align", "arm", "canonical", "cdiv", "install", "installed", "loads_global",
           "reference", "register", "sorted_len", "uninstall", "wrapper"]
