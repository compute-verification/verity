"""PoUB's receipts beside the staging decode: each pass's sketch of every unit of `C*` as the decode
reads it, the units' receipts and the pass's root, bit for bit `remote_bandwidth.p2`'s (the interface is `PROTOCOL.md`
§11 of `verity/core/protocols/remote_bandwidth`; the fixture is `benchmarks/pous/p2_v1/serve.py fixture`). ::

    units = plan_units(layout.tensors)         # the units the store's tensors lie in: the pass's Plan.whole
    tree = Tree(units, device)                 # the receipt tree's row and leaf tables over one y buffer, built once
    P = Pass(seed, units, device)              # pass i's signs S_i on the device (16 rows over a unit's 8,224 bytes)
    store.sketch = P                           # stage.Store.decode_rows sketches each chunk's units before its decode
    ...                                        # the step's decode
    root = P.finish(tree, salts)               # y in unit order (P.y), the root; P.start_ns, P.end_ns

- **σ** (`Pass.read`): the units that blocks j0 .. j1 - 1 lie in, each unit's 8,224 stored bytes as int8 against S_i in
  one int8 GEMM (`torch._int_mm`, int32 accumulators, exact: |σ[c]| ≤ 8,224 · 128 < 2^21), read from the store's rows
  on the device before the decode reads them. The GEMM sums the unit's four block partials π in its accumulators.
  σ[g] is assigned, never added to, so a block read twice (one two tensors share, a unit two chunks share) changes
  nothing; the store's rows hold whole units, so a chunk's units at its ends are read whole. `Pass.partials` gives each
  block's π alone (a GEMM against S_i's four block columns, the one of j mod 4 kept), which the fixture is held to
  first.
- **y** (`receipts.cu`): a thread a unit, one SHA-512 compression from `sketch_prefix(seed)`'s midstate over
  u64be g ‖ σ_g, with sha512_rows' compression; `receipts_host.cpp` is its host twin.
- **root** (`Tree`): sha512_rows' engine over the y buffer: each leaf's 16 receipts (1,024 bytes) a sha512/row/v2 row,
  its hm96-sha512 leaf under the leaf's 192-byte salt, its frame-v3-sha512 leaf in the pass's domain
  (`receipts.domain(seed, Plan.whole(units))`), and the root.

A pass covers the units the store's tensors lie in (`plan_units`), not the store's: the encoded padding past the head
is held, not streamed, so it is in no receipt. `finish` refuses a pass that didn't read every unit. On the CPU every
step runs on host memory (the engine's and `receipts_host.cpp`'s host twins), which is what the tests hold to the
fixture.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import time
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity.core.protocols.remote_bandwidth import p2
from verity.core.protocols.remote_bandwidth import receipts as R
from verity_config.settings import of

settings = of(__file__)
HERE = Path(__file__).resolve().parent
ROWS_DIR = HERE.parent / "sha512_rows"
sys.path.insert(0, str(ROWS_DIR))
import engine as SE  # noqa: E402

BLOCK, UNIT, PER = p2.BLOCK_BYTES, p2.UNIT_BYTES, p2.UNIT_BLOCKS
DEVICE_SOURCES = (ROWS_DIR / "sha512x.cuh", HERE / "receipts.cuh", HERE / "receipts.cu")
HOST_SOURCES = (ROWS_DIR / "sha512x.cuh", HERE / "receipts.cuh", HERE / "receipts_host.cpp")
BUILD_DIR = Path(settings.P2_RECEIPTS_BUILD or Path.home() / ".cache" / "verity" / "p2_receipts")
vp, u64 = ctypes.c_void_p, ctypes.c_uint64


def _build(tool: str, sources: tuple[Path, ...], flags: tuple[str, ...], name: str) -> ctypes.CDLL:
    h = hashlib.sha256()
    for s in sources:
        h.update(s.name.encode() + b"\0" + s.read_bytes() + b"\0")
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout
    h.update(" ".join(flags).encode() + b"\0" + version.encode())
    lib = BUILD_DIR / h.hexdigest()[:12] / name
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        r = subprocess.run([tool, *flags, "-o", str(tmp), str(sources[-1])], capture_output=True, text=True, check=False)
        lib.with_suffix(".log").write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError(f"{tool} failed:\n{r.stderr[-4000:]}")
        tmp.replace(lib)
    dll = ctypes.CDLL(str(lib))
    dll.p2r_midstate.argtypes, dll.p2r_midstate.restype = [vp, vp], None
    dll.p2r_receipts.argtypes, dll.p2r_receipts.restype = [vp, vp, vp, u64, u64, vp], ctypes.c_int
    return dll


@cache
def receipts_lib(arch: str | None) -> ctypes.CDLL:
    """`receipts.cu` for `sm_<arch>` with nvcc, or `receipts_host.cpp` with the system C++ compiler (arch None)."""
    if arch is None:
        cxx = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
        if cxx is None:
            raise RuntimeError("no C++ compiler on the path")
        return _build(cxx, HOST_SOURCES, ("-O2", "-std=c++17", "-fPIC", "-shared"), "libp2r_host.so")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("no nvcc on the path")
    flags = (*SE.NVCC_FLAGS, *SE.variant_flags(SE.DEFAULT_VARIANT), f"-gencode=arch=compute_{arch},code=sm_{arch}")
    return _build(nvcc, DEVICE_SOURCES, flags, "libp2r_cuda.so")


def _arch(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}{minor}" + ("a" if major in (9, 10, 12) else "")


def plan_units(tensors: Any) -> int:
    """The units a pass covers, 0 .. ⌈e / 4⌉ − 1 with e the end of the last tensor's blocks (`Plan.whole` of them):
    every unit a tensor's block lies in, the head's last unit whole (with any padding block it shares), and none of the
    padding past it. Refused if a unit below them holds no tensor's block, which `Plan.whole` would read as padding."""
    at = 0
    for t in sorted(tensors, key=lambda t: t.first_block):
        if t.first_block // PER > at:
            raise ValueError(f"units {at} .. {t.first_block // PER - 1} hold no tensor's block")
        at = max(at, -(-t.block_range.stop // PER))
    if not at:
        raise ValueError("a pass covers at least one tensor's units")
    return at


def signs(seed: bytes, rows: int = R.SKETCH_ROWS) -> np.ndarray:
    """S_i as ±1 int8, [rows, 8224]: bit x of sign row c (byte x // 8, LSB first) 1 is −1, 0 is +1 (`p2.Sketcher.signs`)."""
    bits = np.frombuffer(b"".join(p2.Sketcher(seed, rows).signs()), dtype=np.uint8).reshape(rows, UNIT // 8)
    return (1 - 2 * np.unpackbits(bits, axis=1, bitorder="little").astype(np.int8)).astype(np.int8)


def int_mm(x: Any, w: Any) -> Any:
    """x [n, K] int8 against w [r, K] int8: [n, r] int32, exact. `torch._int_mm` takes more than 16 rows; fewer go
    elementwise."""
    if x.shape[0] > 16:
        return torch._int_mm(x, w.t())
    return (x.to(torch.int32)[:, None, :] * w.to(torch.int32)[None]).sum(-1, dtype=torch.int32)


def midstate(seed: bytes, lib: ctypes.CDLL) -> np.ndarray:
    """sketch_prefix(seed)'s chaining value, 8 words."""
    block = R.sketch_prefix(seed)
    if len(block) != 128:
        raise ValueError("a sketch prefix is one SHA-512 block")
    out = np.zeros(8, dtype=np.uint64)
    buf = ctypes.create_string_buffer(block, 128)
    lib.p2r_midstate(ctypes.addressof(buf), out.ctypes.data)
    return out


def _ptr(t: Any) -> int:
    return t.ctypes.data if isinstance(t, np.ndarray) else t.data_ptr()


def _stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else 0


def receipts(seed: bytes, sigma: Any, g0: int = 0, out: Any = None) -> Any:
    """y of units g0 .. g0 + n − 1 from their σ ([n, 16] int32, contiguous): [n, 64] uint8 on σ's device."""
    if sigma.dtype != torch.int32 or sigma.dim() != 2 or sigma.shape[1] != R.SKETCH_ROWS or not sigma.is_contiguous():
        raise ValueError("σ is a contiguous [n, 16] int32 tensor")
    n = sigma.shape[0]
    out = torch.empty(n, R.DIGEST_BYTES, dtype=torch.uint8, device=sigma.device) if out is None else out
    if out.dtype != torch.uint8 or tuple(out.shape) != (n, R.DIGEST_BYTES) or not out.is_contiguous():
        raise ValueError(f"y is a contiguous [{n}, 64] uint8 tensor")
    lib = receipts_lib(_arch(sigma.device))
    mid = midstate(seed, lib)
    rc = lib.p2r_receipts(_stream(sigma.device), mid.ctypes.data, sigma.data_ptr(), g0, n, out.data_ptr())
    if rc:
        raise RuntimeError(f"receipts: CUDA error {rc}")
    return out


class Tree:
    """The receipt tree of passes of `units` receipts by sha512_rows' engine: the y buffer (`y`, [units, 64] uint8 on
    the device) and its row table, built once, since every pass writes the same buffer."""

    def __init__(self, units: int, device: Any) -> None:
        self.device = torch.device(device)
        self.host = self.device.type != "cuda"
        self.E = SE.Engine("host" if self.host else self.device)
        self.units, self.G = units, R.leaves(units)
        self.plan = R.Plan.whole(units)
        self.y = torch.zeros(units, R.DIGEST_BYTES, dtype=torch.uint8, device=self.device)
        self.table = self._rows()
        rows = np.zeros(self.G, dtype=SE.LEAF_ROW)
        rows["rank"] = rows["position"] = np.arange(self.G, dtype=np.uint64)
        rows["salt_row"] = np.arange(self.G, dtype=np.uint32)
        self.leaf_rows = self.E._up(rows)
        self.schema = self.E._schema(R.SCHEMA)

    def _arr(self, t: Any) -> Any:
        return t.numpy() if self.host else t

    def _rows(self) -> SE.RowTable:
        """The engine's row table, built vectorized (`Engine.rows` takes a Python loop a row): leaf c's row is bytes
        1,024 c .. of the y buffer, 16 receipts (the last leaf's fewer)."""
        y = self._arr(self.y.view(1, -1))
        width = R.DIGEST_BYTES * self.units
        off = np.arange(self.G, dtype=np.uint64) * (R.PER_LEAF * R.DIGEST_BYTES)
        n = np.minimum(R.PER_LEAF * R.DIGEST_BYTES, width - off.astype(np.int64)).astype(np.uint32)
        spec = np.zeros(self.G, dtype=SE.ROW_SPEC)
        spec["a"] = np.uint64(_ptr(y)) + off
        spec["a_stride"], spec["a_len"], spec["nbytes"] = width, n, n
        spec["a16"] = ((spec["a"] % 16 == 0) & (width % 16 == 0)).astype(np.uint32)
        E = self.E
        d = E._up(spec)
        E._ok(E.lib.sxr_row_setup(E._stream(), E._ptr(d), self.G), "row setup")
        nbytes = n.astype(np.int64)
        order = np.argsort(-((nbytes + 17 + 127) // 128), kind="stable").astype(np.uint32)
        return SE.RowTable(d, self.G, 1, tuple(nbytes.tolist()), (y,), E._up(order))

    def salts(self, raw: bytes | Any) -> Any:
        """Leaf salts (192 bytes each, leaf order) as the engine takes them: (G, 1, 192) uint8 on the device."""
        if isinstance(raw, (bytes, bytearray)):
            raw = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        t = raw.reshape(self.G, 1, R.SALT_BYTES).to(self.device)
        return self._arr(t)

    def leaf_table(self, did: bytes) -> SE.LeafTable:
        """Leaf c's frame leaf: rank and position c in the domain `did`, under its salt c. The rows are uploaded once;
        the engine's leaf setup reads only their domain, rank and position, and writes each pass's prefixes over the
        last pass's."""
        E = self.E
        d_did = E._up(SE._did_words([did]))
        d_schema, n_schema = self.schema
        rc = E.lib.sxr_leaf_setup(E._stream(), E._ptr(self.leaf_rows), self.G, E._ptr(d_did), E._ptr(d_schema), n_schema)
        E._ok(rc, "leaf setup")
        return SE.LeafTable(self.leaf_rows, (d_did, d_schema), self.G)

    def root(self, seed: bytes, salts: Any) -> bytes:
        """The root of the pass with seed `seed` over the receipts in `y`, leaf c hidden under salts[c]."""
        E = self.E
        did = R.domain(seed, self.plan).domain_id
        x = E.row_digests(self.table)
        leaves = E.leaves(self.leaf_table(did), x, salts)
        return SE.digests(E.root(leaves.reshape(1, self.G, 8), [did]))[0]


class Pass:
    """One pass of the audit over `units` units (blocks 0 .. 4 units − 1, `plan_units`; blocks past them are in no
    receipt), on `device`: `read` as the decode reads, `finish` at the step's end."""

    def __init__(self, seed: bytes, units: int, device: Any, rows: int = R.SKETCH_ROWS) -> None:
        self.start_ns = time.time_ns()
        self.seed, self.units, self.device = seed, units, torch.device(device)
        self.S = torch.from_numpy(signs(seed, rows)).to(self.device)
        self.B = self.S.view(rows, PER, BLOCK).permute(1, 0, 2).reshape(PER * rows, BLOCK).contiguous()
        self.sigma = torch.zeros(units, rows, dtype=torch.int32, device=self.device)
        self.seen = torch.zeros(units, dtype=torch.bool, device=self.device)
        self.y: Any = None
        self.root: bytes | None = None
        self.end_ns: int | None = None

    def units_of(self, j0: int, j1: int) -> range:
        """The units blocks j0 .. j1 − 1 lie in."""
        return range(min(self.units, j0 // PER), min(self.units, -(-j1 // PER)))

    def read(self, rows: Any, first: int, j0: int, j1: int) -> None:
        """σ of the units blocks j0 .. j1 − 1 lie in, from `rows` ([n, 2056] uint8 on the device, blocks first ..),
        which hold those units whole."""
        u = self.units_of(j0, j1)
        if not u:
            return
        a, b = PER * u.start - first, PER * u.stop - first
        if a < 0 or b > rows.shape[0]:
            raise ValueError(f"blocks [{first}, {first + rows.shape[0]}) don't hold units {u.start} .. {u.stop - 1} whole")
        x = rows[a:b].view(torch.int8).reshape(len(u), UNIT)
        self.sigma[u.start:u.stop] = int_mm(x, self.S)
        self.seen[u.start:u.stop] = True

    def partials(self, j0: int, rows: Any) -> Any:
        """π of blocks j0 .. j0 + n − 1 ([n, 16] int32) from their rows ([n, 2056] uint8)."""
        n = rows.shape[0]
        p = int_mm(rows.view(torch.int8), self.B).view(n, PER, -1)
        q = (torch.arange(n, device=rows.device) + j0) % PER
        return p[torch.arange(n, device=rows.device), q]

    def finish(self, tree: Tree, salts: Any) -> bytes:
        """The receipts (`y`, the tree's buffer) and the root; refused if a unit wasn't read."""
        if tree.units != self.units or tree.y.device != self.sigma.device:
            raise ValueError("the tree is of another store or device")
        missing = int((~self.seen).sum())
        if missing:
            raise ValueError(f"{missing} of {self.units} units weren't read this pass")
        self.y = receipts(self.seed, self.sigma, 0, tree.y)
        self.root = tree.root(self.seed, salts)
        self.end_ns = time.time_ns()
        return self.root
