"""The DSA indexer's top-k at the integration's vLLM pin, made a function of its inputs: a vLLM general plugin (`register`,
entry point group ``vllm.general_plugins``, run in every process whose path holds the site `site` wrote) that, with
``VERITY_DSA_TOPK=canonical``, wraps vLLM's four indexer top-k ops (`OPS`) so that after each call's fast kernel the rows
past ``k`` keys are rewritten in their canonical form (`reference`), on the device, before attention reads them.

vLLM's kernels write a row of ``n <= k`` keys as ``0, 1, ..., n - 1`` and then ``-1``, the same every run, and this
leaves those rows untouched (no kernel runs when no row of the call can be longer than ``k``). A longer row's kernels
place the selected indices with shared-memory atomics, so their order varies run to run, and they admit the keys tied at
the k-th score in atomic order, so the set can vary too; the sparse attention then sums over the keys in that order.
The canonical row is upstream vLLM's deterministic selector's (its batch-invariant op library's
``top_k_per_row_prefill_index_order``):

- each score's 32 bits are made order-preserving as a signed integer (`ordered`), with both zeros one key and every NaN
  the key of ``-inf``;
- the selection is the ``k`` largest (key, index) pairs, so a tie at the k-th key goes to the larger indices;
- the output is the selection in ascending index order, ``k`` indices, each relative to its row's first key.

The device pass reads the fast kernel's selection only for a candidate k-th key ``tau``, which it checks against the
whole row's counts (``#{key > tau} < k <= #{key >= tau}``); a row whose candidate fails (a NaN the fast kernel ranks
first, say) is selected again by bisection over the keys. Either way the output is the row's canonical one. Rows of
``n > k`` keys are a prefill's (``top_k_per_row_prefill``: keys ``[start, end)`` of the gathered row) or a decode's (the
three decode selectors: keys ``[0, length)``).

``VERITY_DSA_TOPK_LOG=DIR`` writes, per process, ``dsa_topk.PID.json``: each op's calls and canonicalizing launches, and
the rows rewritten, those with a tie across the k-th key, and those bisected.
"""
from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import struct
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ENV = "VERITY_DSA_TOPK"
LOG_ENV = "VERITY_DSA_TOPK_LOG"
#: the indexer's top-k ops at the pin, under ``torch.ops._C``
OPS = ("top_k_per_row_prefill", "top_k_per_row_decode", "persistent_topk", "cooperative_topk")
#: the device counters: rows rewritten, rows with a tie across the k-th key, rows bisected
STATS = ("rows_long", "rows_tied", "rows_bisected")
#: a segment of a row (a call of few rows splits each over programs) is at least this many keys
SEG_MIN_KEYS = 8192
#: keys a program reads at a time (below 2**15: dsa_topk_kernels packs two block counts in an int32), and its warps
BLOCK = 4096
NUM_WARPS = 8
#: programs a multiprocessor should get before a row is split
FILL = 2
#: `log_to` rewrites its record every this many calls (a device sync), since vLLM's workers may never reach atexit
FLUSH_EVERY = 4096

INT32_MIN = -(1 << 31)
NEG_INF_BITS = -8388608  # 0xFF800000, the bits of -inf as an int32


def ordered(bits: int) -> int:
    """The order-preserving signed key of a float32's bits (an int32 value): ``-0`` is ``+0``, a NaN is ``-inf``."""
    mag = bits & 0x7FFFFFFF
    if mag > 0x7F800000:
        bits = NEG_INF_BITS
    elif mag == 0:
        bits = 0
    return bits ^ 0x7FFFFFFF if bits < 0 else bits


def float_key(v: float) -> int:
    return ordered(struct.unpack("<i", struct.pack("<f", v))[0])


def reference(values: Sequence[float], k: int) -> list[int]:
    """The canonical top-k row of ``values`` (one row's keys, as float32): ``0..n-1`` then ``-1`` up to ``k`` for
    ``n <= k``, else the ``k`` largest (key, index) pairs' indices in ascending order."""
    n = len(values)
    if n <= k:
        return list(range(n)) + [-1] * (k - n)
    keys = [(float_key(v), i) for i, v in enumerate(values)]
    return sorted(i for _, i in sorted(keys, reverse=True)[:k])


def keys(logits: Any) -> Any:
    """`ordered` over a float32 tensor, in torch: an int64 tensor of the keys."""
    import torch

    bits = logits.contiguous().view(torch.int32).to(torch.int64)
    mag = bits & 0x7FFFFFFF
    bits = torch.where(mag > 0x7F800000, NEG_INF_BITS, torch.where(mag == 0, 0, bits))
    return torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


def reference_rows(logits: Any, lo: Any, n: Any, k: int) -> Any:
    """`reference` for each row ``r`` of a float32 tensor (rows x columns) over its keys ``[lo[r], lo[r] + n[r])``
    (``n[r] <= 0``: none), in torch on the tensor's device: an int32 tensor of shape (rows, k)."""
    import torch

    dev = logits.device
    lo = torch.as_tensor(lo, dtype=torch.int64, device=dev).reshape(-1, 1)
    n = torch.as_tensor(n, dtype=torch.int64, device=dev).reshape(-1, 1).clamp(min=0)
    col = torch.arange(logits.shape[1], dtype=torch.int64, device=dev).reshape(1, -1)
    rel = col - lo
    inside = (rel >= 0) & (rel < n)
    comp = torch.where(inside, keys(logits) * (1 << 32) + rel, torch.iinfo(torch.int64).min)
    top = torch.topk(comp, min(k, logits.shape[1]), dim=1).values
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32, device=dev)
    out[:, :top.shape[1]] = (top & 0xFFFFFFFF).sort(dim=1).values.to(torch.int32)
    short = torch.arange(k, dtype=torch.int64, device=dev).reshape(1, -1)
    return torch.where(n <= k, torch.where(short < n, short, -1).to(torch.int32), out)


_KERNELS: dict[str, Any] = {}


def kernels() -> dict[str, Any]:
    """The Triton kernels (`dsa_topk_kernels`), imported on first use: importing this module needs no Triton."""
    if not _KERNELS:
        import importlib

        mod = importlib.import_module("dsa_topk_kernels")
        _KERNELS.update(canon_one=mod.canon_one, count_seg=mod.count_seg, emit_seg=mod.emit_seg)
    return _KERNELS


class Canon:
    """The device pass, and its counters: `run` rewrites the rows of ``out`` (int32, rows x k) longer than ``k`` keys of
    ``logits`` (float32, rows x columns) in place."""

    def __init__(self, seg_min_keys: int = SEG_MIN_KEYS, block: int = BLOCK, num_warps: int = NUM_WARPS,
                 fill: int = FILL) -> None:
        if not 0 < block < 1 << 15 or block & (block - 1):
            raise ValueError(f"block: a power of two below 2**15, not {block}")
        self.seg_min_keys, self.block, self.num_warps, self.fill = seg_min_keys, block, num_warps, fill
        self.calls = {op: 0 for op in OPS}
        self.launches: dict[str, int] = {op: 0 for op in OPS}
        self._stats: dict[Any, Any] = {}
        self._sms: dict[Any, int] = {}
        self._ws: dict[Any, Any] = {}
        #: called every `FLUSH_EVERY` calls outside a CUDA graph's capture (`log_to`)
        self.on_flush: Callable[[], None] | None = None

    def note(self, op: str) -> None:
        self.calls[op] += 1
        if self.on_flush is not None and sum(self.calls.values()) % FLUSH_EVERY == 0 and not _capturing():
            self.on_flush()

    def stats(self, device: Any) -> Any:
        import torch

        if device not in self._stats:
            self._stats[device] = torch.zeros(len(STATS), dtype=torch.int32, device=device)
        return self._stats[device]

    def segments(self, device: Any, rows: int, ncols: int) -> int:
        """Programs per row: one when the rows alone give each multiprocessor `fill` programs, else enough to, each
        segment at least `seg_min_keys` keys."""
        import torch

        if device not in self._sms:
            self._sms[device] = torch.cuda.get_device_properties(device).multi_processor_count
        want = math.ceil(self.fill * self._sms[device] / max(rows, 1))
        return max(1, min(want, math.ceil(ncols / self.seg_min_keys)))

    def run(self, op: str, logits: Any, out: Any, k: int, mode: int, a: Any, b: Any = None, *, rows: int,
            ncols: int, ls0: int, ls1: int, next_n: int = 1) -> None:

        self.launches[op] = self.launches.get(op, 0) + 1
        K = kernels()
        dev = logits.device
        if out.stride(1) != 1:
            raise ValueError(f"{op}: the top-k output's rows must be contiguous, stride {tuple(out.stride())}")
        a = a.reshape(-1)
        b = a if b is None else b
        stats = self.stats(dev)
        seg = self.segments(dev, rows, ncols)
        args = (logits, ls0, ls1, out, out.stride(0), a, b, ncols, next_n)
        const = dict(K=k, KP2=next_pow2(k), BLOCK=self.block, MODE=mode, num_warps=self.num_warps)
        if seg == 1:
            K["canon_one"][(rows,)](*args, stats, **const)
            return
        ws = self.workspace(dev, rows * (2 * seg + 1))
        K["count_seg"][(rows, seg)](*args, ws, seg, **const)
        const.pop("KP2")
        K["emit_seg"][(rows, seg)](*args, ws, stats, seg, SEGP2=next_pow2(seg), **const)

    def workspace(self, device: Any, words: int) -> Any:
        """An int32 scratch of at least ``words`` for the current stream, kept between calls (a call's two kernels
        are ordered on its stream, and each stream has its own)."""
        import torch

        key = (device, torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else 0)
        ws = self._ws.get(key)
        if ws is None or ws.numel() < words:
            ws = self._ws[key] = torch.empty(max(words, 1 << 16), dtype=torch.int32, device=device)
        return ws

    def record(self) -> dict[str, Any]:
        """The counters so far (synchronizes each device it read)."""
        dev = {str(d): dict(zip(STATS, s.tolist())) for d, s in self._stats.items()}
        return {"pid": os.getpid(), "calls": dict(self.calls), "launches": dict(self.launches), "rows": dev}


def next_pow2(x: int) -> int:
    return 1 << max(int(x) - 1, 0).bit_length()


def _capturing() -> bool:
    import torch

    return bool(torch.cuda.is_current_stream_capturing())


def wrappers(canon: Canon) -> dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """Each op's wrapper around its real op: the op, then `Canon.run` on its output when a row of the call can be
    longer than its ``k``. The ops' signatures are the pin's (csrc/libtorch_stable/torch_bindings.cpp)."""

    def prefill(real: Callable[..., Any]) -> Callable[..., Any]:
        def top_k_per_row_prefill(logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k):
            real(logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k)
            canon.note("top_k_per_row_prefill")
            if num_rows > 0 and logits.shape[1] > top_k:
                canon.run("top_k_per_row_prefill", logits, indices, top_k, 0, row_starts, row_ends, rows=num_rows,
                          ncols=logits.shape[1], ls0=stride0, ls1=stride1)
        return top_k_per_row_prefill

    def decode(real: Callable[..., Any]) -> Callable[..., Any]:
        def top_k_per_row_decode(logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k):
            real(logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k)
            canon.note("top_k_per_row_decode")
            if num_rows > 0 and logits.shape[1] > top_k:
                canon.run("top_k_per_row_decode", logits, indices, top_k, 1 if seq_lens.dim() == 2 else 2, seq_lens,
                          rows=num_rows, ncols=logits.shape[1], ls0=stride0, ls1=stride1, next_n=next_n)
        return top_k_per_row_decode

    def lengths(name: str, bound: Callable[[Any, int], int], hint: bool) -> Callable[..., Any]:
        # hint: the op's max_seq_len bounds every row's length (vLLM passes the batch's longest sequence), so a call
        # whose longest row is at most k keys launches nothing; never trusted while a CUDA graph captures
        def wrap(real: Callable[..., Any]) -> Callable[..., Any]:
            def op(logits, lengths, output, workspace, k, max_seq_len):
                real(logits, lengths, output, workspace, k, max_seq_len)
                canon.note(name)
                ncols = bound(logits, max_seq_len)
                if logits.shape[0] == 0 or ncols <= k or (hint and max_seq_len <= k and not _capturing()):
                    return
                canon.run(name, logits, output, k, 1, lengths, rows=logits.shape[0], ncols=ncols,
                          ls0=logits.stride(0), ls1=logits.stride(1))
            op.__name__ = name
            return op
        return wrap

    return {
        "top_k_per_row_prefill": prefill,
        "top_k_per_row_decode": decode,
        # persistent_topk clamps each length to min(row stride, max_seq_len); vLLM passes the logits' width
        "persistent_topk": lengths("persistent_topk",
                                   lambda x, msl: min(x.shape[1], x.stride(0), max(int(msl), 0)), hint=False),
        "cooperative_topk": lengths("cooperative_topk", lambda x, msl: x.shape[1], hint=True),
    }


def install(ns: Any = None, canon: Canon | None = None) -> Canon:
    """Wrap each of `OPS` on ``ns`` (default ``torch.ops._C``) with its `wrappers` entry; an op already wrapped stays
    as it is, so installing twice wraps once. Returns the wrappers' `Canon`."""
    if ns is None:
        import torch

        import vllm._custom_ops  # noqa: F401  (loads the _C ops)

        ns = torch.ops._C
    present = [name for name in OPS if hasattr(ns, name)]
    for name in present:
        fn = getattr(ns, name)
        if isinstance(fn, types.FunctionType) and "dsa_topk_canon" in fn.__dict__:
            return fn.__dict__["dsa_topk_canon"]
    canon = canon or Canon()
    wraps = wrappers(canon)
    for name in present:
        real = getattr(ns, name)
        fn = wraps[name](real)
        fn.dsa_topk_canon = canon  # type: ignore[attr-defined]
        fn.real = real  # type: ignore[attr-defined]
        setattr(ns, name, fn)
    return canon


def log_to(canon: Canon, d: str) -> None:
    """Keep ``canon``'s record in ``d/dsa_topk.PID.json``: every `FLUSH_EVERY` calls and at exit."""
    path = Path(d) / f"dsa_topk.{os.getpid()}.json"

    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(canon.record(), indent=1) + "\n")
        tmp.replace(path)

    canon.on_flush = write
    atexit.register(write)


def register() -> None:
    """The plugin: with ``VERITY_DSA_TOPK=canonical``, `install` on ``torch.ops._C`` in this process."""
    mode = os.environ.get(ENV, "")
    if mode in ("", "0"):
        return
    if mode != "canonical":
        raise ValueError(f"{ENV}: 'canonical' or unset, not {mode!r}")
    canon = install()
    if os.environ.get(LOG_ENV):
        log_to(canon, os.environ[LOG_ENV])


def site(d: Path) -> Path:
    """Add this module to the site directory ``d`` as a vLLM general plugin: a server or replay with ``d`` on its
    ``PYTHONPATH`` (and ``VLLM_PLUGINS`` unset or naming ``verity_dsa_topk``) loads it in every process."""
    d = Path(d)
    info = d / "verity_dsa_topk-0.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-dsa-topk\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_dsa_topk = dsa_topk:register\n")
    for name in ("dsa_topk.py", "dsa_topk_kernels.py"):
        shutil.copy(Path(__file__).with_name(name), d / name)
    return d


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3 or sys.argv[1] != "site":
        raise SystemExit("usage: python dsa_topk.py site DIR")
    print(site(Path(sys.argv[2])))
