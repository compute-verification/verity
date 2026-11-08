"""A two-stage audit's interior commit on the GPU (a prototype): `verity.ml.windows.rows.commit_ports`'s roots and every row's
``b || c`` for row ports of u16 words under ``hm96-sha512/row/v1``, each port's salts ChaCha20's under the window's salt key at
the port's index (`rows.salts`), computed by `interior_rows.cu` from rows already on the device. `interior_rows_host.cpp` is
its CPU twin, which the tests hold to `commit_ports` byte for byte; the device is held to the host the same way where a GPU is.

`Interior` takes a span's rows as a replay produces them: `Interior.add` a port's rows j0 .. j0 + m - 1 (one step's units of
it), `Interior.flush` enqueues every added piece's leaves in one launch, and `Interior.finish` adds the pads and one launch a
tree level over every port, then reads out the roots and the strings. Nothing but the strings (128 bytes a row, the public
body) and the roots leaves the device; the salts are expanded from the key where they are used, so the same key reproduces
every leaf on a second replay.

    python interior_gpu.py bench --ops OPS.jsonl [--tokens T] [--steps N] [--out FILE]

times the commit of a recorded step's values (`capture.py`'s ``ops.jsonl``): every distinct non-parameter tensor whose leading
dimension counts the step's tokens, ``T`` of its rows a step (the sampled request's), over ``N`` steps of one span.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CUDA_DIR = ROOT / "integrations" / "vllm" / "verity_vllm" / "protocols" / "pouw" / "pouw_cuda"
KERNELS = ROOT / "verity" / "ml" / "kernels"
#: every header a build includes, by which it is keyed
HEADERS = (HERE / "interior_rows.cuh", CUDA_DIR / "registered_rows.cuh", CUDA_DIR / "row_table.cuh", CUDA_DIR / "ncp2.cuh",
           KERNELS / "row_commit.cuh", KERNELS / "sha512.cuh")
BUILD_DIR = Path.home() / ".cache" / "verity" / "interior_rows"
NVCC_FLAGS = ("-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared")
CFLAGS = ("-O2", "-std=c++17", "-fPIC", "-shared")
M32 = 0xFFFFFFFF
#: the most rows a token's value has in a recorded step (GLM-5.3's routed experts a token)
TOPK = 8
_vp, _u32, _u64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64


class Table(ctypes.Structure):
    """`registered_rows.cuh`'s ``Table``."""
    _fields_ = [("rows", _u64), ("salts", _u64), ("n", _u32), ("row_bytes", _u32), ("W", _u32), ("depth", _u32),
                ("key", _u32 * 8), ("nonce", _u32 * 4), ("did", _u64 * 8)]


class Seg(ctypes.Structure):
    """`interior_rows.cuh`'s ``Seg``."""
    _fields_ = [("t", Table), ("bcs", _u64), ("levels", _u64), ("first", _u64), ("j0", _u32), ("count", _u32)]


assert ctypes.sizeof(Table) == 144 and ctypes.sizeof(Seg) == 176


def _built(tool: str, source: Path, flags: tuple[str, ...], name: str) -> Path:
    """``source`` compiled by ``tool`` into ``name``, keyed by it, `HEADERS`, the flags and the compiler's version."""
    h = hashlib.sha256()
    for p in (source, *HEADERS):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout
    h.update(" ".join(flags).encode() + b"\0" + version.encode())
    lib = BUILD_DIR / h.hexdigest()[:12] / name
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run([tool, *flags, f"-I{HERE}", f"-I{CUDA_DIR}", f"-I{KERNELS}", "-o", str(tmp), str(source)], check=True)
        tmp.replace(lib)
    return lib


def _sign(dll: ctypes.CDLL, suffix: str, stream: bool) -> ctypes.CDLL:
    head = (_vp,) if stream else ()
    getattr(dll, f"ir_midstates{suffix}").argtypes = (*head, _vp, _vp)
    getattr(dll, f"ir_leaves{suffix}").argtypes = (*head, _vp, _u32, _u64, _vp, _vp)
    getattr(dll, f"ir_nodes{suffix}").argtypes = (*head, _vp, _u32, _u64)
    return dll


@cache
def host_lib() -> ctypes.CDLL:
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        raise RuntimeError("no C++ compiler on the path for interior_rows_host.cpp")
    return _sign(ctypes.CDLL(str(_built(cxx, HERE / "interior_rows_host.cpp", CFLAGS, "libinterior_rows_host.so"))), "_host", False)


@cache
def device_lib(arch: str, nvcc: str | None = None) -> ctypes.CDLL:
    tool = nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    flags = (*NVCC_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}")
    return _sign(ctypes.CDLL(str(_built(tool, HERE / "interior_rows.cu", flags, f"libinterior_rows_{arch}.so"))), "", True)


@cache
def _tables() -> tuple[np.ndarray, np.ndarray]:
    """hm96-sha512's byte tables under its default key (192 x 256 entries of 8 words) and the salt_prefix and leaf_prefix
    blocks (32 words), SHA-512's big-endian words as u64."""
    from verity.ml.commitments import hm96
    inst = hm96.SHA512
    words = lambda b: np.frombuffer(b, dtype=">u8").astype(np.uint64)  # noqa: E731
    hm = words(b"".join(t.to_bytes(64, "little") for table in inst.byte_tables(inst.default_key) for t in table))
    return hm, words(inst.salt_prefix + inst.leaf_prefix(inst.default_key))


def _be(words: np.ndarray) -> bytes:
    return np.ascontiguousarray(words).view(np.uint64).astype(">u8").tobytes()


class Interior:
    """One interior's row ports, committed as `commit_ports` commits them: ``ports`` maps each port to its domain (its
    ``count`` the port's rows) and its words a row, ``index`` to its salt nonce (`rows.salts`'s port index), ``key`` the
    window's 32-byte salt key. ``salts``: by port, its rows' given salts, ``count`` x 192 bytes (`rows.commit`'s
    ``salts_of``), in place of the key's. ``device``: hash on the current CUDA device (rows and salts given as CUDA tensors),
    else on the CPU twin (as NumPy arrays or bytes)."""

    def __init__(self, ports: Mapping[str, tuple[Any, int]], key: bytes | None, index: Mapping[str, int], *,
                 salts: Mapping[str, Any] | None = None, device: bool = True, nvcc: str | None = None) -> None:
        from verity.ml.commitments.merkle import merkle_depth
        salts = salts or {}
        if key is None and set(salts) != set(ports):
            raise ValueError(f"no salt key, and no given salts for {sorted(set(ports) - set(salts))}")
        if key is not None and len(key) != 32:
            raise ValueError("a window's salt key is 32 bytes")
        self.device = device
        hm, prefixes = _tables()
        if device:
            import torch
            self.torch = torch
            cap = torch.cuda.get_device_capability()
            self.lib = device_lib(f"{cap[0]}{cap[1]}", nvcc)
            self.hm = torch.from_numpy(hm.view(np.int64)).cuda()
            pre = torch.from_numpy(prefixes.view(np.int64)).cuda()
            self.ms = torch.empty(16, dtype=torch.int64, device="cuda")
            self._check(self.lib.ir_midstates(self._stream(), pre.data_ptr(), self.ms.data_ptr()), "ir_midstates")
        else:
            self.lib = host_lib()
            self.hm = hm
            self.ms = np.zeros(16, dtype=np.uint64)
            self._check(self.lib.ir_midstates_host(prefixes.ctypes.data, self.ms.ctypes.data), "ir_midstates_host")
        kw = np.frombuffer(key or bytes(32), "<u4").tolist()
        self.ports: dict[str, tuple[Table, Any, Any]] = {}
        self.salts: list[Any] = []
        for p, (dom, words) in ports.items():
            n, depth = int(dom.count), merkle_depth(int(dom.count))
            if not 0 < n < 1 << 31 or words < 1:
                raise ValueError(f"{p}: {n} rows of {words} words")
            t = Table(n=n, row_bytes=2 * int(words), W=1 << depth, depth=depth)
            t.key[:] = kw
            if p in salts:
                s = self._bytes(salts[p])
                if self._size(s) != 192 * n:
                    raise ValueError(f"{p}: given salts are not 192 bytes for each of its {n} rows")
                self.salts.append(s)
                t.salts = self._ptr(s)
            i = int(index[p])
            t.nonce[:] = [i & M32, (i >> 32) & M32, (i >> 64) & M32, 0]
            t.did[:] = np.frombuffer(dom.domain_id, ">u8").tolist()
            self.ports[p] = (t, self._alloc(16 * n), self._alloc(8 * (2 * t.W - 1)))
        self.pending: list[tuple[str, int, Any]] = []
        self.held: list[Any] = []

    def _alloc(self, words: int) -> Any:
        return self.torch.empty(words, dtype=self.torch.int64, device="cuda") if self.device else np.zeros(words, np.uint64)

    def _ptr(self, x: Any) -> int:
        return x.data_ptr() if self.device else x.ctypes.data

    def _bytes(self, x: Any) -> Any:
        if self.device:
            return x.detach().contiguous().view(self.torch.uint8).reshape(-1)
        return np.frombuffer(x, np.uint8) if isinstance(x, bytes | bytearray) else np.ascontiguousarray(x).view(np.uint8).reshape(-1)

    def _size(self, x: Any) -> int:
        return x.numel() if self.device else x.size

    def _stream(self) -> int:
        return self.torch.cuda.current_stream().cuda_stream

    @staticmethod
    def _check(rc: int, what: str) -> None:
        if rc:
            raise RuntimeError(f"{what} failed ({rc})")

    def add(self, port: str, j0: int, rows: Any) -> None:
        """Rows ``j0`` .. ``j0 + m - 1`` of ``port``: ``rows`` (m, 2 words) bytes, or any contiguous tensor or array whose
        rows are the units' words little-endian."""
        t = self.ports[port][0]
        r = self._bytes(rows)
        m = self._size(r) // t.row_bytes
        if m < 1 or m * t.row_bytes != self._size(r) or j0 < 0 or j0 + m > t.n:
            raise ValueError(f"{port}: {m} rows of {t.row_bytes} bytes at {j0} of {t.n}")
        self.pending.append((port, j0, r))

    def _launch(self, segs: list[Seg], total: int, leaves: bool) -> None:
        arr = (Seg * len(segs))(*segs)
        if self.device:
            buf = self.torch.frombuffer(bytearray(bytes(arr)), dtype=self.torch.uint8).cuda()
            self.held.append(buf)
            rc = (self.lib.ir_leaves(self._stream(), buf.data_ptr(), len(segs), total, self.hm.data_ptr(), self.ms.data_ptr())
                  if leaves else self.lib.ir_nodes(self._stream(), buf.data_ptr(), len(segs), total))
        else:
            rc = (self.lib.ir_leaves_host(ctypes.addressof(arr), len(segs), total, self.hm.ctypes.data, self.ms.ctypes.data)
                  if leaves else self.lib.ir_nodes_host(ctypes.addressof(arr), len(segs), total))
        self._check(rc, "ir_leaves" if leaves else "ir_nodes")

    def flush(self) -> None:
        """Every added piece's leaves and strings, in one launch."""
        if not self.pending:
            return
        segs, total = [], 0
        for p, j0, r in self.pending:
            t, bcs, levels = self.ports[p]
            tt = Table.from_buffer_copy(t)
            tt.rows = self._ptr(r)
            m = self._size(r) // t.row_bytes
            segs.append(Seg(tt, self._ptr(bcs), self._ptr(levels), total, j0, m))
            total += m
        self._launch(segs, total, True)
        self.held += [r for _, _, r in self.pending]
        self.pending = []

    def finish(self) -> tuple[dict[str, bytes], dict[str, bytes]]:
        """Each port's root and its rows' ``b || c``, once every row is added: the pads, then one launch a level."""
        self.flush()
        top = max((t.depth for t, _, _ in self.ports.values()), default=0)
        for level in range(top + 1):
            segs, total = [], 0
            for t, bcs, levels in self.ports.values():
                count = t.W - t.n if level == 0 else (t.W >> level if level <= t.depth else 0)
                if count:
                    segs.append(Seg(t, self._ptr(bcs), self._ptr(levels), total, level, count))
                    total += count
            if segs:
                self._launch(segs, total, False)
        if self.device:
            self.torch.cuda.current_stream().synchronize()
        self.held = []
        roots, strings = {}, {}
        for p, (t, bcs, levels) in self.ports.items():
            last = levels[8 * (2 * t.W - 2):]
            roots[p] = _be(last.cpu().numpy() if self.device else last)
            strings[p] = _be(bcs.cpu().numpy() if self.device else bcs)
        return roots, strings


def commit_ports(domains: Mapping[str, Any], rows: Mapping[str, Any], key: bytes | None, index: Mapping[str, int], *,
                 salts: Mapping[str, Any] | None = None, device: bool = True) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """`rows.commit_ports(domains, rows, {}, ys)` by `Interior`, ``ys[p]`` ``salts[p]`` if given, else
    ``rows.salts(key, index[p], count)``: ``rows[p]`` (count, words) u16 (CUDA tensors under ``device``)."""
    it = Interior({p: (domains[p], int(r.shape[1])) for p, r in rows.items()}, key, index, salts=salts, device=device)
    for p, r in rows.items():
        it.add(p, 0, r)
    return it.finish()


# --- timing a recorded step ---------------------------------------------------------------------------------------------

def step_ports(ops: Path, tokens: int | None) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
    """A recorded step's values as interior ports: every distinct tensor an op reads or writes (by its name), not a parameter
    nor a whole cache, of at least two dimensions whose leading one is k = 1 .. `TOPK` times the step's tokens (the most common
    leading dimension of its ops' tensors), as k ``tokens`` rows a step (``None``: all the step's) of its bytes per leading
    index, rounded up to whole u16 words. Returns them, ``{name: (rows a step, words)}``, and what was left out (bytes)."""
    refs: list[dict] = []

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            if "shape" in v and "dtype" in v and ("sha512" in v or "chunks_sha512" in v or "big" in v):
                refs.append(v)
                return
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    for line in ops.read_text().splitlines():
        e = json.loads(line)
        walk(e.get("in"))
        walk(e.get("out"))
    lead = [int(r["shape"][0]) for r in refs if r["shape"]]
    batch = max(set(lead), key=lead.count) if lead else 1
    per = batch if tokens is None else tokens
    size = {"float32": 4, "int32": 4, "float16": 2, "bfloat16": 2, "int16": 2, "uint8": 1, "int8": 1, "bool": 1,
            "int64": 8, "float64": 8, "float8_e4m3fn": 1, "uint16": 2, "uint32": 4}
    ports: dict[str, tuple[int, int]] = {}
    left: dict[str, int] = {}
    for r in refs:
        shape, el = [int(x) for x in r["shape"]], size.get(str(r["dtype"]).removeprefix("torch."), 0)
        nb = int(np.prod(shape)) * el if shape else el
        name = r.get("sha512") or r.get("chunks_sha512") or f"big{len(left)}"
        if "chunks_sha512" in r:
            left["whole caches"] = left.get("whole caches", 0) + nb
        elif ("big" in r or len(shape) < 2 or not el or not shape[0] or shape[0] % batch or shape[0] > TOPK * batch
              or name in ports):
            left["not token-major, or seen"] = left.get("not token-major, or seen", 0) + nb
        else:
            k = shape[0] // batch
            ports[name] = (k * per, -(-(nb // shape[0]) // 2))
    return ports, left


def bench(ops: Path, tokens: int | None, steps: int, nvcc: str | None = None) -> dict[str, Any]:
    """The interior commit of ``steps`` steps of one span whose every step's values are the recorded step's (`step_ports`):
    each step's leaves timed (one flush, synchronised), then the span's trees, all on the current CUDA device."""
    import torch
    from verity.ml.commitments.indexed import RangeIndexedDomain
    from verity.ml.commitments.merkle import CommitmentDomain
    per, left = step_ports(ops, tokens)
    ports = {f"p{i}": (CommitmentDomain(hashlib.sha512(f"p{i}".encode()).digest(), -1, RangeIndexedDomain(0, steps * r)), w)
             for i, (r, w) in enumerate(per.values()) if r}
    shapes = {f"p{i}": (r, w) for i, (r, w) in enumerate(per.values()) if r}
    gen = torch.Generator(device="cuda").manual_seed(1)
    data = {p: torch.randint(0, 1 << 16, (r, w), dtype=torch.int32, device="cuda", generator=gen).to(torch.int16)
            for p, (r, w) in shapes.items()}
    t0 = time.time()
    it = Interior(ports, os.urandom(32), {p: i for i, p in enumerate(ports)}, nvcc=nvcc)
    setup = time.time() - t0
    times = []
    for s in range(steps):
        torch.cuda.synchronize()
        t = time.time()
        for p, (r, _) in shapes.items():
            it.add(p, s * r, data[p])
        it.flush()
        torch.cuda.synchronize()
        times.append(time.time() - t)
    t = time.time()
    roots, _ = it.finish()
    trees = time.time() - t
    rows = sum(r for r, _ in shapes.values())
    nbytes = sum(2 * r * w for r, w in shapes.values())
    return {"ops": str(ops), "tokens": tokens, "steps": steps, "ports": len(shapes), "rows_per_step": rows,
            "bytes_per_step": nbytes, "left_out_bytes": left, "setup_s": round(setup, 3),
            "step_s": {"first": round(times[0], 4), "median": round(float(np.median(times)), 4), "max": round(max(times), 4)},
            "gb_per_s": round(nbytes / float(np.median(times)) / 1e9, 2), "trees_s": round(trees, 4), "roots": len(roots),
            "device": torch.cuda.get_device_name()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="time the interior commit of a recorded step's values")
    b.add_argument("--ops", type=Path, required=True, nargs="+", help="recorded steps' ops.jsonl (capture.py)")
    b.add_argument("--tokens", type=int, default=None, help="rows a step of each token-major value (default: all the step's)")
    b.add_argument("--steps", type=int, default=16, help="steps in the span")
    b.add_argument("--nvcc", default=None)
    b.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = [bench(p, a.tokens, a.steps, a.nvcc) for p in a.ops]
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
