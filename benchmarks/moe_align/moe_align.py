"""vLLM's `moe_align_block_size` at the integration's pin, made a function of its inputs: a vLLM general plugin (`register`,
entry point group ``vllm.general_plugins``, run in every process whose path holds the site `site` wrote) that, with
``VERITY_MOE_ALIGN=canonical``, replaces the op ``torch.ops._moe_C.moe_align_block_size`` (`OP`) with `Align`, which
writes the same three outputs in the same buffers, with each expert's pairs in ascending pair index.

The op takes the router's ``topk_ids`` (``[tokens, top_k]``, any integer dtype, contiguous), ``num_experts`` E, the moe
block ``BM``, the three int32 outputs vLLM's wrapper (fused_moe/moe_align_block_size.py) allocates, and an optional
``expert_map``. A pair is p = t * top_k + k, a slot of ``topk_ids`` in memory order. An id outside [0, E), or one the map
sends to -1, is no expert's. Expert e's c_e pairs are written from ``base[e] = sum over e' < e of round_up(c_e', BM)``:

- ``sorted_token_ids[base[e] + r]`` is e's pair of rank r, the ranks in ascending pair index; every other slot of the
  buffer is ``numel`` (the pair count P);
- ``expert_ids[b]`` is the expert whose padded range holds slot ``b * BM`` for each block below
  ``num_tokens_post_pad / BM``, and -1 from there to ``cdiv(len(sorted_token_ids), BM)``;
- ``num_tokens_post_pad[0]`` is ``sum over e of round_up(c_e, BM)``.

vLLM's kernel at E > 64 (count_and_sort_expert_tokens_kernel) writes the same buffers except the order within an
expert, which is the order its atomicAdd on the expert's offset hands out slots, so it varies run to run. `reference`
is the definition in Python, and the Triton kernels (`moe_align_kernels`) compute it with no atomics. A call of at most
`SMALL_MAX` pairs is one launch (`align_small`); a longer one is three, over chunks of `CHUNK` pairs. Each program places
a tile of `TILE` pairs, ranking them by comparing them with each other, `COLS` at a time.

``VERITY_MOE_ALIGN_LOG=DIR`` writes, per process, ``moe_align.PID.json``: the calls on each path.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ENV = "VERITY_MOE_ALIGN"
LOG_ENV = "VERITY_MOE_ALIGN_LOG"
#: the op this replaces, under ``torch.ops._moe_C``
OP = "moe_align_block_size"
#: the most pairs `align_small` takes in one launch; a longer call is chunked
SMALL_MAX = 4096
#: pairs a chunk of the long path, pairs a tile (a program), and the pairs each step of a tile's rank compare reads
CHUNK = 1024
TILE = 64
COLS = 32
#: the chunk rows and experts `scan_chunks` reads at a time, the slots a tail-store loop writes at a time, and the
#: pad slots of an expert a pad store writes at a time
SCAN_ROWS = 32
SCAN_EXPERTS = 64
STORE = 1024
PAD_STORE = 16
#: vLLM's op refuses ``padded_num_experts >= 1024`` (moe_align_sum_kernels.cu): E rounded up to 32 below 1024
MAX_EXPERTS = 992
#: `log_to` rewrites its record every this many calls outside a CUDA graph's capture
FLUSH_EVERY = 4096


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
        self.fn, self.fast, self.compiled = fn, fast, {}

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
        import importlib

        mod = importlib.import_module("moe_align_kernels")
        _KERNELS.update({k: getattr(mod, k) for k in ("align_small", "count_chunks", "scan_chunks", "place_chunks")})
    return _KERNELS


def warps(pairs: int) -> int:
    return 4 if pairs <= 512 else 8 if pairs <= 2048 else 16


class Align:
    """The op: writes the canonical outputs into vLLM's buffers, on the inputs' device and stream, with no host sync.
    ``fast_launch`` launches a compiled kernel through its handle (`_Launch`)."""

    def __init__(self, small_max: int = SMALL_MAX, chunk: int = CHUNK, tile: int = TILE, cols: int = COLS,
                 fast_launch: bool = True) -> None:
        for name, v in (("chunk", chunk), ("tile", tile), ("cols", cols)):
            if v < 16 or v & (v - 1):
                raise ValueError(f"{name}: a power of two of at least 16, not {v}")
        if not cols <= tile <= chunk:
            raise ValueError(f"cols {cols}, tile {tile} and chunk {chunk} must not decrease")
        self.small_max, self.chunk, self.tile, self.cols, self.fast_launch = small_max, chunk, tile, cols, fast_launch
        self._launch: dict[str, _Launch] = {}
        self.calls = {"one": 0, "chunked": 0}
        #: called every `FLUSH_EVERY` calls outside a CUDA graph's capture (`log_to`)
        self.on_flush: Callable[[], None] | None = None

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
        if P <= self.small_max:
            self.calls["one"] += 1
            PB = max(next_pow2(P), 16)
            T = min(self.tile, PB)
            K["align_small"]((cdiv(P, T),), (topk_ids, mp, *bufs), (P, L, NB, E, BM),
                             {"HAS_MAP": has_map, "PB": PB, "T": T, "JB": min(self.cols, T), "EP": EP, "TB": STORE,
                              "PW": PAD_STORE}, warps(PB))
        else:
            self.calls["chunked"] += 1
            CH, T = self.chunk, self.tile
            C, ES = cdiv(P, CH), min(SCAN_EXPERTS, EP)
            if (2 * C + 1) * EP >= 1 << 31:
                raise ValueError(f"{OP}: {C} chunks of {CH} pairs overflow the workspace's int32 offsets")
            ws = torch.empty((2 * C + 1) * EP, dtype=torch.int32, device=topk_ids.device)
            K["count_chunks"]((C,), (topk_ids, mp, ws), (P, E), {"HAS_MAP": has_map, "CHUNK": CH, "EP": EP}, warps(CH))
            K["scan_chunks"]((EP // ES,), (ws,), (C,), {"EP": EP, "ES": ES, "CB": SCAN_ROWS}, 4)
            K["place_chunks"]((C, CH // T), (topk_ids, mp, ws, *bufs), (P, L, NB, E, BM, C),
                              {"HAS_MAP": has_map, "CHUNK": CH, "T": T, "JB": self.cols, "EP": EP, "TB": STORE,
                               "PW": PAD_STORE}, warps(CH))
        if self.on_flush is not None and sum(self.calls.values()) % FLUSH_EVERY == 0 and not _capturing():
            self.on_flush()

    def record(self) -> dict[str, Any]:
        return {"pid": os.getpid(), "calls": dict(self.calls), "small_max": self.small_max, "chunk": self.chunk,
                "tile": self.tile, "cols": self.cols, "fast_launch": self.fast_launch}


def _capturing() -> bool:
    import torch

    return bool(torch.cuda.is_current_stream_capturing())


def install(ns: Any = None, align: Align | None = None) -> Align:
    """Replace `OP` on ``ns`` (default ``torch.ops._moe_C``) with `Align`; installing twice replaces once. Returns the
    installed `Align`."""
    if ns is None:
        import torch

        import vllm._custom_ops  # noqa: F401  (loads the _moe_C ops)

        ns = torch.ops._moe_C
    real = getattr(ns, OP)
    if isinstance(real, types.FunctionType) and "moe_align" in real.__dict__:
        return real.__dict__["moe_align"]
    align = align or Align()

    def moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad,
                             maybe_expert_map=None):
        align(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad, maybe_expert_map)

    moe_align_block_size.moe_align = align  # type: ignore[attr-defined]
    moe_align_block_size.real = real  # type: ignore[attr-defined]
    setattr(ns, OP, moe_align_block_size)
    return align


def log_to(align: Align, d: str) -> None:
    """Keep ``align``'s record in ``d/moe_align.PID.json``: every `FLUSH_EVERY` calls and at exit."""
    path = Path(d) / f"moe_align.{os.getpid()}.json"

    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(align.record(), indent=1) + "\n")
        tmp.replace(path)

    align.on_flush = write
    atexit.register(write)


def settings() -> tuple[str, str]:
    """The plugin's two variables, read here only: ``VERITY_MOE_ALIGN`` and ``VERITY_MOE_ALIGN_LOG`` ("" when unset)."""
    return os.environ.get(ENV, ""), os.environ.get(LOG_ENV, "")


def register() -> None:
    """The plugin: with ``VERITY_MOE_ALIGN=canonical``, `install` on ``torch.ops._moe_C`` in this process."""
    mode, log = settings()
    if mode in ("", "0"):
        return
    if mode != "canonical":
        raise ValueError(f"{ENV}: 'canonical' or unset, not {mode!r}")
    align = install()
    if log:
        log_to(align, log)


def site(d: Path) -> Path:
    """Add this module to the site directory ``d`` as a vLLM general plugin: a server or replay with ``d`` on its
    ``PYTHONPATH`` (and ``VLLM_PLUGINS`` unset or naming ``verity_moe_align``) loads it in every process."""
    d = Path(d)
    info = d / "verity_moe_align-0.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-moe-align\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_moe_align = moe_align:register\n")
    for name in ("moe_align.py", "moe_align_kernels.py"):
        shutil.copy(Path(__file__).with_name(name), d / name)
    return d


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3 or sys.argv[1] != "site":
        raise SystemExit("usage: python moe_align.py site DIR")
    print(site(Path(sys.argv[2])))
