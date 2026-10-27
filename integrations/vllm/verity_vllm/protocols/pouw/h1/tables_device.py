"""`pouw_h1.H1Run`'s call tables hashed on the GPU: a call's A table and z table, each
`pouw_h1_requests.hiding_rows` of its leaves, computed by the Commit's row-table kernels (`protocol_options/pouw_cuda/row_table.cu`,
`commit.row_tables_device`) as a step of one table.

The kernels hash rows of K 8-bit codes then a 4-byte word, so a table's leaf value is that row followed by its 192-byte salt,
and its root is `hiding_rows`' when the table's layout says so (`layout`): the domain id is ``audit.rows_domain``'s (the
host's), the step's nonce is ``bytes(4) || u64le nonce`` and the table starts at the step's row 0 in rank order, so row i's salt
is ChaCha20 blocks 3i .. 3i + 2 at that nonce, ``hiding_rows``' keystream. An A table is the call's T slots of K s8 codes and
f32 scale; a z table's leaf is a ``ZY_BYTES`` record, its first ``ZY_BYTES - 4`` bytes as the codes and its last 4 as the word
(`record_rows`). The step's own root, over its one table, is not read.

`device_hash` launches a call's tables on one of the device's side streams (`STREAMS`, in turn) behind everything the current
stream has enqueued, nothing waiting on the host, and hands back a `Pending` whose `roots` waits for them: `H1Run` reads a
step's roots at its round's close, so a call's tables hash beside the forward rather than in it. `device_roots` waits at once.
`host_roots` runs the same functions on the CPU (`row_table_host.cpp`), the twin the CPU tests hold to `hiding_rows` on
machines without a GPU. Builds share the Commit's row-table builds (`row_tables_device._build`); nvcc is `pouw_device`'s.

`commit_tables` is `H1Run`'s commit of a call's tables, by `hiding_rows` (the reference) or by these kernels where `path`
says, and `settle` records the roots of the calls the GPU has finished hashing, in call order.
"""
from __future__ import annotations

import ctypes
import math
import warnings
from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy as np
from verity.core.protocols.pouw.audit import HM96, rows_domain
from verity.core.protocols.pouw.protocol import tagged

from verity_vllm.commit import row_tables_device as RTD
from verity_vllm.protocols.pouw.h1 import prover as PV
from verity_vllm.protocols.pouw.h1 import records as ZR
from verity_vllm.protocols.pouw.h1 import requests as RQ
from verity_vllm.protocols.pouw.h1.requests import TABLE_NONCES, TABLE_TAGS, table_leaves

__all__ = ["MODE", "Pending", "STREAMS", "available", "commit_tables", "device_hash", "device_lib", "device_roots",
           "host_roots", "layout", "path", "record_rows", "row_words", "settle"]

#: where `pouw_h1.H1Run` hashes a call's tables: ``"auto"`` (the GPU when the call's tensors are CUDA tensors and the kernels
#: build for their device, else ``hiding_rows``), ``"gpu"``, ``"host"`` (the kernels' functions on the CPU, the twin) or
#: ``"python"`` (``hiding_rows``, the reference)
MODE = "auto"
MODES = ("auto", "gpu", "host", "python")
#: the side streams a device's calls hash on, one call's tables each in turn
STREAMS = 4
_M32 = 0xFFFFFFFF


def layout(domain_id: bytes, key: bytes, nonce: int, n: int, k: int) -> tuple[np.ndarray, RTD.Args]:
    """The kernels' step of one table of ``n`` rows of ``k`` codes and a 4-byte word under ``domain_id``, salted by ChaCha20
    under the 32-byte ``key`` at the u64 ``nonce``: its ``Tab`` (14 u64 words, the codes' and words' addresses zero until
    bound) and its ``Args``."""
    if n < 1 or k < 2 or k % 2 or 3 * n > 1 << 32 or len(domain_id) != 64 or len(key) != 32 or not 0 <= nonce < 1 << 64:
        raise ValueError(f"{n} rows of {k} codes under a {len(domain_id)}-byte domain id, a {len(key)}-byte key and nonce "
                         f"{nonce}: not 1 to 2^32 / 3 rows of an even K >= 2, a 64-byte id, a 32-byte key and a u64 nonce")
    s = -(-(k + 4) // RTD.SEG_BYTES)
    tab = np.concatenate([np.asarray([0, 0, k | n << 32, s << 32, 0, RTD._width(n)], np.uint64), RTD._words(domain_id)])
    args = RTD.Args()
    args.key[:] = np.frombuffer(key, "<u4").tolist()
    args.nonce[:] = [0, nonce & _M32, nonce >> 32]
    args.T = 1
    return tab, args


def _root(words: np.ndarray) -> bytes:
    return np.asarray(words).view(np.uint64).astype(">u8").tobytes()


def record_rows(records: Any) -> tuple[Any, Any]:
    """A z table's rows as the kernels take them, from its records ``(units, ZY_BYTES)`` uint8 (a tensor or an array): the codes
    ``(units, ZY_BYTES - 4)`` and the last 4 bytes of each as one 32-bit word."""
    k = records.shape[1] - 4
    if hasattr(records, "contiguous"):
        import torch

        return records[:, :k].contiguous(), records[:, k:].contiguous().view(torch.int32).reshape(-1)
    return np.ascontiguousarray(records[:, :k]), np.ascontiguousarray(records[:, k:]).view("<u4").reshape(-1)


def _host(x: Any) -> np.ndarray:
    return np.ascontiguousarray(x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x))


def host_roots(tables: Sequence[tuple[bytes, int, Any, Any]], key: bytes) -> list[bytes]:
    """Each table's root, ``(domain_id, nonce, codes, words)`` with codes ``(n, K)`` of 8-bit and words ``(n,)`` of 32-bit
    values, by the kernels' functions on the CPU: the device path's twin."""
    out = []
    for did, nonce, c, w in tables:
        codes, words = _host(c), _host(w)
        if codes.ndim != 2 or codes.itemsize != 1 or words.itemsize != 4 or words.shape != (codes.shape[0],):
            raise ValueError(f"codes {codes.shape} of {codes.dtype} and words {words.shape} of {words.dtype} are not (n, K) "
                             "8-bit and (n,) 32-bit")
        n, k = codes.shape
        tab, args = layout(did, key, nonce, n, k)
        tab[0], tab[1] = codes.ctypes.data, words.ctypes.data
        roots = np.zeros(16, np.uint64)
        rc = RTD.host_lib().rt_step_host(tab.reshape(1, -1), np.arange(n, dtype=np.uint32), n * (int(tab[3]) >> 32), n,
                                         RTD._hm(), RTD._prefixes(), ctypes.byref(args), roots)
        if rc:
            raise ValueError(f"rt_step_host refused the table's layout ({rc})")
        out.append(_root(roots[8:]))
    return out


def nvcc() -> str | None:
    """`pouw_device`'s nvcc: its configured `TOOLCHAIN`'s, or nvcc on the path."""
    from verity_vllm.protocols.pouw import device as pouw_device

    return pouw_device.nvcc()


_vp = ctypes.c_void_p


@cache
def device_lib(arch: str) -> ctypes.CDLL:
    """`row_table.cu` built with `nvcc` for ``sm_<arch>``: the Commit's build when the nvcc is the same."""
    tool = nvcc()
    if tool is None:
        raise RuntimeError("no nvcc on the path or configured")
    dll = RTD._build(tool, RTD.DEVICE_SOURCES, (*RTD.NVCC_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}"),
                     "librow_table_cuda.so")
    dll.rt_midstates.argtypes = [_vp, _vp, _vp]
    dll.rt_step.argtypes = [_vp, _vp, _vp, ctypes.c_uint64, ctypes.c_uint32, _vp, _vp, ctypes.POINTER(RTD.Args)] + [_vp] * 5
    dll.rt_midstates.restype = dll.rt_step.restype = ctypes.c_int
    return dll


class _Device:
    """One CUDA device's build, its copies of the byte tables and midstates, the identity rank order tables share, and the
    side streams tables hash on."""

    def __init__(self, torch: Any, device: Any) -> None:
        major, minor = torch.cuda.get_device_capability(device)
        self.lib = device_lib(f"{major}{minor}")
        self.hm = torch.from_numpy(RTD._hm().view(np.int64)).to(device)
        pre = torch.from_numpy(RTD._prefixes().view(np.int64)).to(device)
        self.ms = torch.empty(16, dtype=torch.int64, device=device)
        rc = self.lib.rt_midstates(torch.cuda.current_stream(device).cuda_stream, pre.data_ptr(), self.ms.data_ptr())
        if rc:
            raise RuntimeError(f"rt_midstates: CUDA error {rc}")
        torch.cuda.current_stream(device).synchronize()
        self.perm = torch.arange(0, dtype=torch.int32, device=device)
        self.streams, self.turn = [torch.cuda.Stream(device) for _ in range(STREAMS)], 0

    def ranks(self, torch: Any, n: int) -> Any:
        if len(self.perm) < n:
            self.perm = torch.arange(max(n, 2 * len(self.perm)), dtype=torch.int32, device=self.perm.device)
        return self.perm

    def stream(self) -> Any:
        self.turn = (self.turn + 1) % len(self.streams)
        return self.streams[self.turn]


_DEVICES: dict[int, _Device] = {}


def _device(torch: Any, device: Any) -> _Device:
    if device.index not in _DEVICES:
        _DEVICES[device.index] = _Device(torch, device)
    return _DEVICES[device.index]


@cache
def available() -> bool:
    """Whether this process can hash on the GPU: torch with CUDA, and the kernels built for its current device (a warning
    says why not when CUDA is there)."""
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        _device(torch, torch.device("cuda", torch.cuda.current_device()))
    except Exception as e:  # noqa: BLE001 - nvcc missing or a failed build leaves hiding_rows, said once
        warnings.warn(f"H1's call tables hash with hiding_rows: the GPU path doesn't build here ({type(e).__name__}: {e})",
                      RuntimeWarning, stacklevel=2)
        return False
    return True


def path(*tensors: Any) -> str:
    """Where a call's tables hash, `MODE`'s choice: ``"gpu"``, ``"host"``, ``"python"``, or for ``"auto"`` the GPU when every
    tensor given is a CUDA tensor and `available`, else ``"python"``."""
    if MODE not in MODES:
        raise ValueError(f"MODE={MODE!r}: one of {', '.join(MODES)}")
    if MODE != "auto":
        return MODE
    return "gpu" if tensors and all(getattr(t, "is_cuda", False) for t in tensors) and available() else "python"


class Pending:
    """Tables launched on a side stream (`device_hash`): ``ready()`` without blocking, ``roots()`` their roots in order. It
    holds their inputs and scratch until the roots are read; each is also recorded on the side stream, so dropping it
    early gives the allocator nothing back that the side stream still uses."""

    def __init__(self, done: Any, host: Any, held: tuple[Any, ...]) -> None:
        self.done, self.host, self.held = done, host, held

    def ready(self) -> bool:
        return bool(self.done.query())

    def roots(self) -> list[bytes]:
        self.done.synchronize()
        self.held = ()
        host = self.host.numpy()
        return [_root(host[i, 8:]) for i in range(len(host))]


def device_hash(tables: Sequence[tuple[bytes, int, Any, Any]], key: bytes) -> Pending:
    """`host_roots` hashed on the next side stream of the tables' device, after everything its current stream has enqueued
    so far, every codes and words a contiguous CUDA tensor there, nothing waiting on the host."""
    import torch

    dev = tables[0][2].device
    if dev.type != "cuda":
        raise ValueError(f"device_hash hashes CUDA tensors, not tensors on {dev}")
    for _did, _nonce, c, w in tables:
        if c.device != dev or w.device != dev or c.dim() != 2 or c.element_size() != 1 or w.element_size() != 4 \
                or tuple(w.shape) != (c.shape[0],) or not c.is_contiguous() or not w.is_contiguous():
            raise ValueError(f"codes {tuple(c.shape)} and words {tuple(w.shape)} are not contiguous (n, K) 8-bit and (n,) "
                             f"32-bit tensors on {dev}")
    d = _device(torch, dev)
    laid = [layout(did, key, nonce, int(c.shape[0]), int(c.shape[1])) for did, nonce, c, _w in tables]
    tabs = np.stack([tab for tab, _args in laid])
    for i, (_did, _nonce, c, w) in enumerate(tables):
        tabs[i, 0], tabs[i, 1] = c.data_ptr(), w.data_ptr()
    # each table's scratch, in 64-byte units: its segment digests, leaves, tree and step tree; then every table's output
    units = [(n * s, n, RTD._width(n), 2) for n, s in ((int(t[2]) >> 32, int(t[3]) >> 32) for t, _args in laid)]
    staged = torch.from_numpy(tabs.view(np.int64)).pin_memory()
    meta = staged.to(dev, non_blocking=True)
    perm = d.ranks(torch, max(int(c.shape[0]) for _did, _nonce, c, _w in tables))
    written = torch.cuda.Event()
    written.record(torch.cuda.current_stream(dev))
    side = d.stream()
    side.wait_event(written)
    inputs = (meta, perm, *(x for _did, _nonce, c, w in tables for x in (c, w)))
    for x in inputs:
        x.record_stream(side)
    with torch.cuda.stream(side):
        scratch = torch.empty(8 * (sum(map(sum, units)) + 2 * len(laid)), dtype=torch.int64, device=dev)
        at, outs = scratch.data_ptr(), scratch[-16 * len(laid):].view(len(laid), 16)
        for i, ((_tab, args), (segs, n, w, st)) in enumerate(zip(laid, units)):
            dig, leaves, tree, step = at, at + 64 * segs, at + 64 * (segs + n), at + 64 * (segs + n + w)
            at += 64 * (segs + n + w + st)
            rc = d.lib.rt_step(side.cuda_stream, meta.data_ptr() + 8 * RTD.TAB_WORDS * i, perm.data_ptr(), segs, n,
                               d.hm.data_ptr(), d.ms.data_ptr(), ctypes.byref(args), dig, leaves, tree, step,
                               outs[i].data_ptr())
            if rc:
                raise RuntimeError(f"rt_step: CUDA error {rc}")
        host = torch.empty((len(laid), 16), dtype=torch.int64, pin_memory=True)
        host.copy_(outs, non_blocking=True)
        done = torch.cuda.Event()
        done.record(side)
    return Pending(done, host, (staged, scratch, *inputs))


def device_roots(tables: Sequence[tuple[bytes, int, Any, Any]], key: bytes) -> list[bytes]:
    """`device_hash`'s roots, waited for."""
    return device_hash(tables, key).roots()


def row_words(t: Any, m: int, dtype: str) -> np.ndarray:
    """`t`'s m rows as little-endian `dtype` words, a bf16 tensor by its bits."""
    if hasattr(t, "detach") and str(t.dtype) == "torch.bfloat16":
        import torch
        t = t.detach().contiguous().view(torch.int16)
    t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return np.ascontiguousarray(t.reshape(m, -1), dtype=dtype)


def settle(run: Any, wait: bool) -> None:
    """Record the roots of ``run``'s calls whose tables the GPU has hashed, in call order: all of them with ``wait``, else
    the leading ones already done."""
    while run._hashing and (wait or run._hashing[0][1].ready()):
        index, pending = run._hashing.popleft()
        run._roots[index].update(zip(("a", "z"), (r.hex() for r in pending.roots())))


def _binding(run: Any, call: Any, name: str) -> tuple[bytes, int]:
    """The call's `name` table's domain binding and salt nonce."""
    return tagged(TABLE_TAGS[name], run.salt, call.index), RQ.TABLES_NONCE + 3 * call.index + TABLE_NONCES[name]


def _table(run: Any, call: Any, name: str, leaves: list[bytes]) -> str:
    binding, nonce = _binding(run, call, name)
    return RQ.hiding_rows(binding, call.index, leaves, run._key, nonce).commitment.root.hex()


def _kernel_tables(run: Any, call: Any, a: Any, s_a: Any, zy: Any, where: str) -> None:
    """`commit_tables`' roots by the row-table kernels, on ``a``'s device's side streams (``"gpu"``, recorded once hashed:
    ``run.tables`` waits for them) or the CPU (``"host"``): the A table's T rows of K codes and f32 scale, the z table's
    records split as the kernels take them."""
    import torch

    m, t, k = call.m, run.rows, call.k
    dev = a.device if where == "gpu" else torch.device("cpu")
    codes, scales = torch.zeros((t, k), dtype=torch.int8, device=dev), torch.zeros(t, dtype=torch.float32, device=dev)
    if m:
        codes[:m], scales[:m] = torch.as_tensor(a).reshape(m, k), torch.as_tensor(s_a).reshape(m)
    tables = [("a", codes, scales)]
    if not m:
        tables.append(("z", torch.zeros((call.units, ZR.ZY_BYTES - 4), dtype=torch.uint8, device=dev),
                       torch.zeros(call.units, dtype=torch.int32, device=dev)))
    elif zy is not None:
        tables.append(("z", *record_rows(torch.as_tensor(zy).to(dev))))
    laid = []
    for name, c, w in tables:
        binding, nonce = _binding(run, call, name)
        laid.append((rows_domain(binding, call.index, len(c), HM96).domain_id, nonce, c, w))
    run._roots[call.index] = {"a": None, "z": None}
    if where == "gpu":
        run._hashing.append((call.index, device_hash(laid, run._key)))
        settle(run, wait=False)
    else:
        run._roots[call.index].update(zip(("a", "z"), (r.hex() for r in host_roots(laid, run._key))))


def commit_tables(run: Any, call: Any, a: Any, s_a: Any, records: np.ndarray | None, zy: Any = None) -> None:
    """``run``'s (a `pouw_h1.H1Run`'s) commit of the call's A table (`table_leaves`) and its z table, leaf u proof block u's z/y
    record (`pouw_h1_records`; an empty call's all zero), none for a served call whose arm handed over no records. Hashed
    where `path` says (by default on the GPU when the call's tensors, ``zy`` its records, are CUDA tensors), each root
    `hiding_rows`'."""
    m, t = call.m, run.rows
    given = [x for x in (a, s_a, zy) if x is not None]
    kernels = (a is not None and call.k % 2 == 0 and call.units > 0 and (records is None or zy is not None)
               and (not m or math.prod(s_a.shape) == m))
    where = path(*given) if kernels else "python"
    if where != "python":
        _kernel_tables(run, call, a, s_a, zy if m and records is not None else None, where)
        if run.prover is not None and m:
            PV.keep_call(run.prover, call.index, row_words(a, m, "i1"), row_words(s_a, m, "<f4"), records)
        return
    if a is not None and m:
        av, sv = row_words(a, m, "i1"), row_words(s_a, m, "<f4")
    else:
        av = sv = np.zeros((0, 1), "i1")
    out: dict[str, str | None] = {"a": _table(run, call, "a", table_leaves(t, call.k + 4, m, av, sv))}
    leaves = [bytes(ZR.ZY_BYTES)] * call.units if not m else None if records is None else [r.tobytes() for r in records]
    out["z"] = None if leaves is None else _table(run, call, "z", leaves)
    run._roots[call.index] = out
    if run.prover is not None and m:
        PV.keep_call(run.prover, call.index, av, sv, records)
