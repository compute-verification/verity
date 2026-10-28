"""s8_wgmma.cubin (build.sh's) from Python: the exact W8A8 linear on sm_90a, on torch tensors.

`scaled_mm(xq, s_a, w, s_w, y, cfg)` launches `s8mm_<TM>x<BN>` on torch's current stream: xq [M, K] int8, s_a [M] float32,
w [N, K] int8 (row-major), s_w [N] float32 -> y [M, N] bf16, in place. It allocates nothing and a CUDA graph captures it (the
tensor maps are kernel parameters, encoded on the host and cached by address and shape); `linear(x_s8, sx_f32, w_s8, sw_f32)`
is the same with y allocated, `int_mm.linear`'s signature. A config (`Cfg`, text
"128x256:c0:g8") is the tile (TM x BN), the CTAs (c0: the fewest that keep the tiles' rounds over every SM, `Lib.grid`; cN: at
most N, so the GEMM holds N SMs and leaves the rest) and the raster group (tile rows a group). The cubin is $S8_WGMMA_CUBIN,
else build/s8_wgmma.cubin here; loading it checks its `s8mm_layout` (Params' layout, each tile's shared memory and stages)
and `s8mr_layout` against this file's.

`reduce_mm(xq, s_a, w, s_w, y, rk, z, cfg, flush)` is the same linear with the reduction of its s32 sums O in the epilogue
(`s8mr_*`, `s8mrb_*`): z [M, r] int32 = O viewed as int8 [M, 4N] (each cell's four little-endian bytes) times R^T, R [r, 4N]
int8 public, for r in RS, on FUSED_TILES at K unsplit; y as `scaled_mm` writes it. rk is R in the kernel's order
(`kernel_order(R)`, once per R); z is zeroed by the call, in the same stream, then summed into across the tiles of each row
("red": red.global.add per s32; "bulk": one bulk reduce-add a warpgroup through shared memory).
"""
from __future__ import annotations

import ctypes as C
import functools
import hashlib
import json
from pathlib import Path

import torch
from verity_config.settings import of

settings = of(__file__)
HERE = Path(__file__).resolve().parent
TILES = ((128, 256), (128, 128), (128, 64), (64, 256), (64, 128), (64, 64))
BK, SMEM_BUDGET, MAX_STAGES = 128, 220 * 1024, 12
#: CUtensorMapDataType UINT8, CUtensorMapSwizzle 128B, CUtensorMapL2promotion 128B; CUfunction_attribute
#: MAX_DYNAMIC_SHARED_SIZE_BYTES; CUdevice_attribute MULTIPROCESSOR_COUNT
TMAP_UINT8, TMAP_SWIZZLE_128B, TMAP_L2_128B, MAX_DYN_SMEM, SM_COUNT = 0, 3, 2, 8, 16
#: the configs run.py gates beside the picked one
EXTRA_CFGS = ("64x128:c0:g8", "64x256:c0:g8", "128x128:c0:g8", "128x256:c0:g8", "128x256:c0:g1", "128x64:c0:g8",
              "64x64:c0:g8", "128x256:c32:g8", "128x256:c0:g8:m2", "128x128:c0:g8:m2", "128x64:c0:g1:m2",
              "64x256:c0:g8:m2", "64x64:c0:g8:m2", "128x256:c32:g8:m2", "64x64:c0:g8:k0", "128x64:c0:g8:k0",
              "64x128:c0:g8:k3", "64x256:c0:g1:k0")

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER


class LaunchConfig(C.Structure):
    """CUlaunchConfig, for cuOccupancyMaxActiveClusters (the cluster kernels' dimensions are compiled in: no attributes)."""
    _fields_ = [("grid", u * 3), ("block", u * 3), ("shared", u), ("stream", vp), ("attrs", vp), ("nattrs", u)]


CUDA = {
    "cuDeviceGet": [P(i32), i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuDeviceGetAttribute": [P(i32), i32, i32],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuGetErrorString": [i32, P(C.c_char_p)],
    "cuTensorMapEncodeTiled": [vp, i32, u, vp, P(u64), P(u64), P(u), P(u), i32, i32, i32, i32],
    "cuOccupancyMaxActiveClusters": [P(i32), vp, P(LaunchConfig)],
}


class Params(C.Structure):
    """s8_wgmma.cu's Params: the two tensor maps, s_a, s_w, y, M, N, K, the raster group, split K's workspace, the clock
    record, padded to its 128-byte alignment."""
    _fields_ = [("a", C.c_ubyte * 128), ("b", C.c_ubyte * 128), ("sa", u64), ("sw", u64), ("y", u64), ("M", C.c_int32),
                ("N", C.c_int32), ("K", C.c_int32), ("group", C.c_int32), ("splits", C.c_int32), ("pad0", C.c_int32),
                ("ws", u64), ("cnt", u64), ("clk", u64), ("pad", C.c_ubyte * 56)]


assert C.sizeof(Params) == 384 and Params.sa.offset == 256 and Params.M.offset == 280 and Params.ws.offset == 304
assert Params.clk.offset == 320

#: the reduction's widths and tiles (s8_wgmma.cu's S8MR), and its flushes by kernel prefix
RS = (8, 16, 32, 64, 128)
FUSED_TILES = ((128, 256), (128, 128))
FLUSHES = {"red": "s8mr", "bulk": "s8mrb"}
#: (tile rows, tile columns, r, flush) with no kernel: ptxas would serialize their wgmma pipeline (s8_wgmma.cu's S8MR)
UNBUILT = frozenset({(128, 256, 8, "red"), (128, 256, 16, "red")})


def has_reduction(cfg: "Cfg", r: int, flush: str) -> bool:
    """Whether a reduction kernel runs cfg's tile at width r with this flush."""
    return ((cfg[0], cfg[1]) in FUSED_TILES and cfg[4] == 1 and r in RS and flush in FLUSHES
            and (cfg[0], cfg[1], r, flush) not in UNBUILT)
#: within each 32 bytes of R's depth, the kernel's byte kk is R's byte KERNEL_ORDER[kk]: the accumulator words of columns
#: 2 t and 2 t + 1 of an 8-column group are the A fragment's bytes 4 t .. 4 t + 3 and 16 + 4 t .. 16 + 4 t + 3
KERNEL_ORDER = tuple(4 * (2 * ((kk % 16) // 4) + kk // 16) + kk % 4 for kk in range(32))


class RParams(C.Structure):
    """s8_wgmma.cu's RParams: the GEMM's Params, R's tensor map and Z, padded to its 128-byte alignment."""
    _fields_ = [("p", Params), ("r", C.c_ubyte * 128), ("z", u64), ("pad", C.c_ubyte * 120)]


assert C.sizeof(RParams) == 640 and RParams.r.offset == 384 and RParams.z.offset == 512


def red_geometry(tm: int, bn: int, r: int) -> tuple[int, int]:
    """(chunks a stage, stages a tile) of R in the ring: a tile's 4 BN bytes of depth in 128-byte chunks of r rows."""
    ch, stage = bn // 32, (tm + bn) * BK
    q = min(stage // (r * 128), ch)
    return q, cdiv(ch, q)


def kernel_order(rmat: torch.Tensor) -> torch.Tensor:
    """R [r, 4N] int8 (byte b of O's column j at 4 j + b) in the order s8mr reads it."""
    r, k4 = rmat.shape
    assert rmat.dtype == torch.int8 and k4 % 32 == 0, (rmat.dtype, rmat.shape)
    idx = torch.tensor(KERNEL_ORDER, dtype=torch.long, device=rmat.device)
    return rmat.reshape(r, k4 // 32, 32)[:, :, idx].contiguous().reshape(r, k4)


def stages(tm: int, bn: int) -> int:
    return min(SMEM_BUDGET // ((tm + bn) * BK), MAX_STAGES)


def smem(tm: int, bn: int) -> int:
    return 1024 + stages(tm, bn) * (tm + bn) * BK + 16 * stages(tm, bn)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class Cfg(tuple):
    """(TM, BN, CTAs, group, cluster, splits); `Cfg.parse("128x256:c0:g8")`, `"128x256:c0:g8:m2"`, `"64x64:c0:g8:k0"`.
    CTAs 0: as many as keep the rounds (`Lib.grid`). Cluster 2: s8mm2_*, CTA pairs on M-adjacent tiles sharing W's box by
    multicast (the group counts pairs of tile rows, the CTAs stay a cap: half as many pairs). Splits of K (cluster 1
    only, and only where the tiles leave SMs idle, `splits_for`): their s32 sums meet in a workspace, exactly
    (`Lib.workspace`); 0: as many as fill the SMs (`auto_splits`)."""

    def __new__(cls, tm: int, bn: int, ctas: int = 0, group: int = 8, cluster: int = 1, splits: int = 1):
        if ((tm, bn) not in TILES or ctas < 0 or group < 1 or cluster not in (1, 2) or ctas == 1 and cluster == 2
                or splits < 0 or splits != 1 and cluster != 1):
            raise ValueError(f"no such config: {tm}x{bn}:c{ctas}:g{group}:m{cluster}:k{splits}")
        return super().__new__(cls, (tm, bn, ctas, group, cluster, splits))

    @classmethod
    def parse(cls, text: str) -> Cfg:
        dims, *rest = text.split(":")
        kw = {"c": 0, "g": 8, "m": 1, "k": 1}
        for r in rest:
            kw[r[0]] = int(r[1:])
        return cls(*map(int, dims.split("x")), kw["c"], kw["g"], kw["m"], kw["k"])

    def __str__(self) -> str:
        tm, bn, ctas, group, cluster, splits = self
        return (f"{tm}x{bn}:c{ctas}:g{group}" + (f":m{cluster}" if cluster != 1 else "")
                + (f":k{splits}" if splits != 1 else ""))


def auto_splits(units: int, kb: int, sms: int = 132) -> int:
    """Splits of K that put a unit on every SM when the tiles alone don't, each at least two 128-deep segments."""
    return max(1, min(sms // units, kb // 2))


def splits_for(units: int, kb: int, splits: int, sms: int = 132) -> int:
    """The splits a launch runs: none where the tiles alone give every SM one (so the workspace stays under a round of
    tiles, never [M, N] at prefill), else `splits` (0: `auto_splits`), at most one a 128-deep segment."""
    return 1 if units >= sms else min(splits or auto_splits(units, kb, sms), kb)


def default_cfg(m: int, n: int, k: int, sms: int = 132) -> Cfg:
    """64-row tiles up to M 64; 256 columns where that still gives every SM a tile, else 128 (64 for N under 128)."""
    tm = 64 if m <= 64 else 128
    bn = 256 if cdiv(m, tm) * cdiv(n, 256) >= sms else 128 if n >= 128 else 64
    return Cfg(tm, bn)


class Lib:
    """The cubin in torch's context on the current device: its kernels, their shared memory, the tensor maps."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cubin = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.cubin).hexdigest()
        self.cu = C.CDLL(settings.S8_LIBCUDA)
        for f, a in CUDA.items():
            getattr(self.cu, f).argtypes, getattr(self.cu, f).restype = a, i32
        torch.cuda.init()
        self.device = torch.cuda.current_device()
        dev, ctx = i32(), vp()
        self.ck("cuDeviceGet", C.byref(dev), self.device)
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), self.cubin)
        sms = i32()
        self.ck("cuDeviceGetAttribute", C.byref(sms), SM_COUNT, dev)
        self.sms = sms.value
        self.fns, self.clusters = {}, {}
        for tm, bn in TILES:
            for cl, name in ((1, f"s8mm_{tm}x{bn}"), (2, f"s8mm2_{tm}x{bn}")):
                f = self.fn(name)
                self.ck("cuFuncSetAttribute", f, MAX_DYN_SMEM, smem(tm, bn))
                self.fns[(tm, bn, cl)] = f
            self.clusters[(tm, bn)] = self.max_clusters(self.fns[(tm, bn, 2)], 128 * (tm // 64 + 1), smem(tm, bn))
        self.rfns = {}
        for tm, bn in FUSED_TILES:
            for r in RS:
                for flush, prefix in FLUSHES.items():
                    if (tm, bn, r, flush) in UNBUILT:
                        continue
                    f = self.fn(f"{prefix}_{tm}x{bn}_r{r}")
                    self.ck("cuFuncSetAttribute", f, MAX_DYN_SMEM, smem(tm, bn))
                    self.rfns[(tm, bn, r, flush)] = f
        self.maps: dict[tuple[int, int, int, int], bytes] = {}
        self.works: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.layout = self._layout()

    def ck(self, f: str, *a) -> None:
        r = getattr(self.cu, f)(*a)
        if r:
            s = C.c_char_p()
            self.cu.cuGetErrorString(r, C.byref(s))
            raise RuntimeError(f"{f} returned CUresult {r} ({s.value.decode() if s.value else 'unknown'})")

    def fn(self, name: str) -> vp:
        f = vp()
        self.ck("cuModuleGetFunction", C.byref(f), self.mod, name.encode())
        return f

    def max_clusters(self, f: vp, threads: int, shared: int) -> int:
        """How many of a pair kernel's clusters the card holds at once (a cluster's CTAs share a GPC)."""
        cfg = LaunchConfig((u * 3)(2, 1, 1), (u * 3)(threads, 1, 1), shared, None, None, 0)
        n = i32()
        self.ck("cuOccupancyMaxActiveClusters", C.byref(n), f, C.byref(cfg))
        if n.value < 1:
            raise RuntimeError(f"{self.path}: no cluster of two fits ({threads} threads, {shared} bytes of shared memory)")
        return n.value

    def _launch(self, f: vp, grid: int, threads: int, shared: int, param) -> None:
        params = (vp * 1)(C.addressof(param))
        self.ck("cuLaunchKernel", f, grid, 1, 1, threads, 1, 1, shared, vp(torch.cuda.current_stream().cuda_stream), params, None)

    def _layout(self) -> dict:
        out = torch.zeros(16, dtype=torch.int32, device="cuda")
        self._launch(self.fn("s8mm_layout"), 1, 1, 0, u64(out.data_ptr()))
        got = out.tolist()
        want = [C.sizeof(Params), Params.sa.offset, Params.M.offset] + [v for tm, bn in TILES for v in (smem(tm, bn), stages(tm, bn))]
        if got[:len(want)] != want:
            raise RuntimeError(f"{self.path}'s s8mm_layout {got[:len(want)]} is not wgmma.py's {want}: rebuild or update")
        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        self._launch(self.fn("s8mr_layout"), 1, 1, 0, u64(out.data_ptr()))
        rgot = out.tolist()
        rwant = [C.sizeof(RParams), RParams.r.offset, RParams.z.offset] + [
            v for tm, bn in FUSED_TILES for r in RS for v in red_geometry(tm, bn, r)]
        if rgot[:len(rwant)] != rwant:
            raise RuntimeError(f"{self.path}'s s8mr_layout {rgot[:len(rwant)]} is not wgmma.py's {rwant}: rebuild or update")
        return {"params_bytes": got[0], "tiles": {f"{tm}x{bn}": {"smem": smem(tm, bn), "stages": stages(tm, bn)} for tm, bn in TILES},
                "rparams_bytes": rgot[0], "reduction": {f"{tm}x{bn}:r{r}": dict(zip(("chunks_a_stage", "stages_a_tile"),
                                                                                       red_geometry(tm, bn, r)))
                                                         for tm, bn in FUSED_TILES for r in RS}}

    def tmap(self, ptr: int, rows: int, depth: int, box_rows: int) -> bytes:
        """A CUtensorMap over a row-major (rows x depth) byte matrix as s8_wgmma.cu loads it: UINT8, rank 2, boxes of
        box_rows x 128 bytes, SWIZZLE_128B, L2 promotion 128B, zeros past the matrix."""
        key = (ptr, rows, depth, box_rows)
        if key not in self.maps:
            raw = C.create_string_buffer(128 + 64)
            at = (C.addressof(raw) + 63) & ~63
            dims, strides = (u64 * 2)(depth, rows), (u64 * 1)(depth)
            box, estr = (u * 2)(128, box_rows), (u * 2)(1, 1)
            self.ck("cuTensorMapEncodeTiled", vp(at), TMAP_UINT8, 2, vp(ptr), dims, strides, box, estr, 0, TMAP_SWIZZLE_128B,
                    TMAP_L2_128B, 0)
            self.maps[key] = C.string_at(at, 128)
        return self.maps[key]

    def grid(self, units: int, ctas: int, most: int | None = None) -> int:
        """At most `ctas` (0: every SM) CTAs, and of those the fewest that keep the same rounds of units: tiles, or with
        `most` (the clusters the card holds) pairs of tiles, two CTAs each."""
        if most is None:
            cap = min(ctas or self.sms, self.sms)
            return cdiv(units, cdiv(units, cap))
        cap = min(ctas // 2 or most, most)
        return 2 * cdiv(units, cdiv(units, cap))

    def workspace(self, m: int, n: int, tm: int, units: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Split K's s32 partial sums [M, N] and a counter a consumer warp of each tile, allocated zero once a shape and
        left zero by every launch (the last split of each warp's rows clears them), so a CUDA graph replays it as is."""
        key = (torch.cuda.current_device(), m, n, tm, units)
        if key not in self.works:
            dev = torch.device("cuda", key[0])
            self.works[key] = (torch.zeros(m * n, dtype=torch.int32, device=dev),
                               torch.zeros(units * (tm // 64) * 4, dtype=torch.int32, device=dev))
        return self.works[key]

    def _params(self, xq, s_a, w, s_w, y, cfg: Cfg, clk=None) -> tuple[Params, int, int]:
        """The launch's Params, its units and its grid."""
        m, k = xq.shape
        n = w.shape[0]
        tm, bn, ctas, group, cl, splits = cfg
        p = Params()
        C.memmove(C.addressof(p) + Params.a.offset, self.tmap(xq.data_ptr(), m, k, tm), 128)
        C.memmove(C.addressof(p) + Params.b.offset, self.tmap(w.data_ptr(), n, k, bn // cl), 128)
        p.sa, p.sw, p.y = s_a.data_ptr(), s_w.data_ptr(), y.data_ptr()
        p.M, p.N, p.K, p.group = m, n, k, group
        units, kb = cdiv(cdiv(m, tm), cl) * cdiv(n, bn), cdiv(k, BK)
        p.splits = splits_for(units, kb, splits, self.sms)
        if p.splits > 1:
            ws, cnt = self.workspace(m, n, tm, units)
            p.ws, p.cnt = ws.data_ptr(), cnt.data_ptr()
        grid = self.grid(units * p.splits, ctas, self.clusters[(tm, bn)] if cl == 2 else None)
        if clk is not None:
            if clk.dtype != torch.int64 or not clk.is_contiguous() or clk.numel() < 4 * grid:
                raise ValueError(f"clk takes 4 int64 words a CTA: {grid} CTAs, {clk.numel()} {clk.dtype} words")
            p.clk = clk.data_ptr()
        return p, units, grid

    def scaled_mm(self, xq, s_a, w, s_w, y, cfg: Cfg, clk=None) -> dict:
        tm, bn, _, _, cl, _ = cfg
        p, units, grid = self._params(xq, s_a, w, s_w, y, cfg, clk)
        self._launch(self.fns[(tm, bn, cl)], grid, 128 * (tm // 64 + 1), smem(tm, bn), p)
        return {"grid": grid, "units": units, "cluster": cl, "splits": p.splits}

    def reduce_mm(self, xq, s_a, w, s_w, y, rk, z, cfg: Cfg, flush: str, clk=None) -> dict:
        tm, bn, _, _, cl, _ = cfg
        r = rk.shape[0]
        if not has_reduction(cfg, r, flush):
            raise ValueError(f"no reduction kernel for {cfg} at r {r}, flush {flush}: tiles {FUSED_TILES}, one CTA, r in {RS}, "
                             f"none of {sorted(UNBUILT)}")
        q = RParams()
        q.p, units, grid = self._params(xq, s_a, w, s_w, y, cfg, clk)
        if q.p.splits != 1:
            raise ValueError(f"{cfg} splits K {q.p.splits} ways at this shape; the reduction reads whole sums: no :k")
        C.memmove(C.addressof(q) + RParams.r.offset, self.tmap(rk.data_ptr(), r, rk.shape[1], r), 128)
        q.z = z.data_ptr()
        z.zero_()
        self._launch(self.rfns[(tm, bn, r, flush)], grid, 128 * (tm // 64 + 1), smem(tm, bn), q)
        return {"grid": grid, "units": units, "r": r, "flush": flush}

    def identity(self) -> dict:
        """The cubin's hash and build.sh's SASS gate on it: ok only when the sass.json beside it has no problem and the
        MANIFEST.sha256 beside both names this cubin and that sass.json."""
        doc = {"cubin": str(self.path), "sha256": self.sha256, "sms": self.sms}
        gate, man = self.path.with_name("sass.json"), self.path.with_name("MANIFEST.sha256")
        if gate.exists():
            g = json.loads(gate.read_text())
            doc["sass_problems"] = g.get("problems")
            doc["forms"] = sorted({f for r in g.get("kernels", {}).values() for f in r.get("forms", [])})
        if gate.exists() and man.exists():
            sums = {p.removeprefix("./"): h for h, p in (line.split(None, 1) for line in man.read_text().splitlines() if line.strip())}
            doc["manifest"] = (sums.get(self.path.name) == self.sha256
                               and sums.get(gate.name) == hashlib.sha256(gate.read_bytes()).hexdigest())
        doc["ok"] = doc.get("sass_problems") == [] and doc.get("manifest") is True
        return doc


@functools.cache
def lib() -> Lib:
    return Lib(Path(settings.S8_WGMMA_CUBIN or HERE / "build" / "s8_wgmma.cubin"))


def scaled_mm(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor,
              cfg: Cfg | None = None, clk: torch.Tensor | None = None) -> dict:
    """xq [M, K] int8, s_a [M] f32, w [N, K] int8 (row-major), s_w [N] f32 -> y [M, N] bf16, in place. clk: None, or int64
    [>= 4 x CTAs] that each CTA's clock record goes into (`sm_mhz`)."""
    m, k = xq.shape
    n = w.shape[0]
    assert xq.dtype == torch.int8 and w.dtype == torch.int8 and w.shape[1] == k and xq.is_contiguous() and w.is_contiguous()
    assert s_a.dtype == torch.float32 and s_a.numel() == m and s_a.is_contiguous()
    assert s_w.dtype == torch.float32 and s_w.numel() == n and s_w.is_contiguous()
    assert y.dtype == torch.bfloat16 and y.shape == (m, n) and y.is_contiguous()
    assert m >= 1 and k % 16 == 0 and n % 2 == 0, (m, n, k)
    assert xq.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0 and s_w.data_ptr() % 8 == 0 and y.data_ptr() % 4 == 0
    lb = lib()
    return lb.scaled_mm(xq, s_a, w, s_w, y, cfg or default_cfg(m, n, k, lb.sms), clk)


def sm_mhz(clk: torch.Tensor) -> float | None:
    """The median over CTAs of a launch's clock record (clk as `scaled_mm` filled it): each CTA's SM cycles over its own
    nanoseconds, in MHz; None where no CTA wrote one (clk zero)."""
    c = clk.view(-1, 4).cpu()
    c = c[(c[:, 3] > c[:, 1]) & (c[:, 2] > c[:, 0])]
    if not len(c):
        return None
    return float(((c[:, 2] - c[:, 0]).double() / (c[:, 3] - c[:, 1]).double()).median()) * 1e3


def reduce_mm(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor, rk: torch.Tensor,
              z: torch.Tensor, cfg: Cfg, flush: str = "red", clk: torch.Tensor | None = None) -> dict:
    """`scaled_mm`, and z [M, r] int32 = O's bytes [M, 4N] times R^T, rk = `kernel_order(R)` [r, 4N] int8, in place."""
    m, k = xq.shape
    n = w.shape[0]
    r = rk.shape[0]
    assert xq.dtype == torch.int8 and w.dtype == torch.int8 and w.shape[1] == k and xq.is_contiguous() and w.is_contiguous()
    assert s_a.dtype == torch.float32 and s_a.numel() == m and s_a.is_contiguous()
    assert s_w.dtype == torch.float32 and s_w.numel() == n and s_w.is_contiguous()
    assert y.dtype == torch.bfloat16 and y.shape == (m, n) and y.is_contiguous()
    assert rk.dtype == torch.int8 and rk.shape == (r, 4 * n) and rk.is_contiguous() and rk.data_ptr() % 16 == 0
    assert z.dtype == torch.int32 and z.shape == (m, r) and z.is_contiguous() and z.data_ptr() % 16 == 0
    assert m >= 1 and k % 16 == 0 and n % 8 == 0, (m, n, k)
    assert xq.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0 and s_w.data_ptr() % 8 == 0 and y.data_ptr() % 4 == 0
    return lib().reduce_mm(xq, s_a, w, s_w, y, rk, z, cfg, flush, clk)


def linear(x_s8: torch.Tensor, sx_f32: torch.Tensor, w_s8: torch.Tensor, sw_f32: torch.Tensor) -> torch.Tensor:
    """`int_mm.linear`'s signature: y [M, N] bf16 of x_s8 [M, K], sx_f32 [M], w_s8 [N, K] and sw_f32 [N]."""
    y = torch.empty((x_s8.shape[0], w_s8.shape[0]), dtype=torch.bfloat16, device=x_s8.device)
    scaled_mm(x_s8, sx_f32.reshape(-1), w_s8, sw_f32.reshape(-1), y)
    return y
