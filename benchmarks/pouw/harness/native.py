"""The device, through ctypes: `libpouw_harness.so` (native/build.sh) and NVML. `Device` is what the harness, its baselines
and every arm see (`arm.py`); `Nvml` reads one GPU's sensors, addressed by its UUID."""

from __future__ import annotations

import ctypes as C
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from arm import Buffer
from identity import uuid_string
from verity_config.settings import of

HERE = Path(__file__).parent
settings = of(__file__)
u64, vp, sz, i32 = C.c_uint64, C.c_void_p, C.c_size_t, C.c_int


class NativeError(RuntimeError):
    pass


class Info(C.Structure):
    _fields_ = [("name", C.c_char * 256), ("uuid", C.c_ubyte * 16), ("pci_bus_id", C.c_char * 32),
                ("major", i32), ("minor", i32), ("sms", i32), ("l2_bytes", i32), ("smem_optin", i32),
                ("persisting_l2_max", i32), ("clock_khz_unused", i32), ("mem_bus_width", i32),
                ("total_mem", C.c_ulonglong), ("runtime_version", i32), ("driver_version", i32),
                ("cublaslt_version", C.c_ulonglong)]


class LtCandidate(C.Structure):
    _fields_ = [("algo", C.c_uint64 * 8), ("workspace", sz), ("waves", C.c_float), ("algo_id", i32), ("tile", i32),
                ("stages", i32), ("splitk", i32), ("reduction", i32), ("swizzle", i32), ("custom", i32),
                ("inner_shape", i32), ("cluster_shape", i32)]

    def config(self) -> dict:
        return {k: getattr(self, k) for k in ("algo_id", "tile", "stages", "splitk", "reduction", "swizzle", "custom",
                                              "inner_shape", "cluster_shape")} | {"workspace": self.workspace,
                                                                                    "waves": round(self.waves, 3)}


def lib_path() -> Path:
    return Path(settings.POUW_HARNESS_LIB or HERE / "native" / "build" / "libpouw_harness.so")


def lt2_path() -> Path | None:
    """The second cuBLASLt's library (build.sh with LT2_HOME): POUW_HARNESS_LT2, else libpouw_lt2.so beside the harness's
    library if the build made one; POUW_HARNESS_LT2=none leaves it out. A named library that can't load is an error."""
    v = settings.POUW_HARNESS_LT2
    if v == "none":
        return None
    if v:
        return Path(v)
    p = lib_path().parent / "libpouw_lt2.so"
    return p if p.exists() else None


def bind_lt(L) -> None:
    L.ph_last_error.restype = C.c_char_p
    L.ph_lt_run.restype = i32
    L.ph_lt_run.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp]
    L.ph_lt_plan_create.argtypes = [i32, i32, i32, i32, sz, i32, C.POINTER(vp)]
    L.ph_lt_plan_create_ld.argtypes = [i32, i32, i32, i32, i32, sz, i32, C.POINTER(vp)]
    L.ph_lt_plan_destroy.argtypes = [vp]
    L.ph_lt_candidates.argtypes = [vp, vp, vp, i32, C.POINTER(LtCandidate), C.POINTER(i32)]
    if hasattr(L, "ph_lt_enumerate"):
        L.ph_lt_enumerate.argtypes = [vp, vp, vp, C.POINTER(i32), i32, i32, C.POINTER(LtCandidate), C.POINTER(i32),
                                      C.POINTER(i32)]
    if hasattr(L, "ph_lt_version"):
        L.ph_lt_version.restype = C.c_ulonglong
    if C.sizeof(LtCandidate) != L.ph_lt_candidate_size():
        raise NativeError("a cuBLASLt library's ph_lt_candidate doesn't match native.py's")


@dataclass
class LtLib:
    """One cuBLASLt the harness calls: the one libpouw_harness.so links (candidate names `lt_…`), or a second one
    (libpouw_lt2.so, names `lt<major>_…`)."""

    lib: object
    prefix: str
    version: int
    path: str

    @property
    def can_enumerate(self) -> bool:
        return hasattr(self.lib, "ph_lt_enumerate")

    def ck(self, r: int, what: str = "") -> None:
        if r:
            raise NativeError(f"{what} {r}: {self.lib.ph_last_error().decode()}".strip())


@dataclass
class Graph:
    exec: int
    nodes: int


class Device:
    """One CUDA device (device 0 of the process: CUDA_VISIBLE_DEVICES names it), its timing stream and its L2 flush."""

    def __init__(self, path: Path | None = None):
        self.lib = C.CDLL(str(path or lib_path()))
        L = self.lib
        bind_lt(L)
        L.ph_mma_flops.restype = C.c_longlong
        L.ph_mma_flops.argtypes = [i32]
        for f in ("ph_fill", "ph_read", "ph_copy", "ph_xor_byte", "ph_cutlass_launch", "ph_graph_launch",
                  "ph_event_record", "ph_capture_begin", "ph_count_eq", "ph_d2h", "ph_h2d", "ph_memset", "ph_d2h_2d"):
            getattr(L, f).restype = i32
        L.ph_fill.argtypes = [i32, vp, sz, u64, vp]
        L.ph_read.argtypes = [vp, sz, i32, vp, i32, vp]
        L.ph_copy.argtypes = [vp, vp, sz, i32, vp]
        L.ph_xor_byte.argtypes = [vp, sz, i32, vp]
        L.ph_count_eq.argtypes = [vp, sz, i32, C.c_uint32, C.POINTER(C.c_ulonglong)]
        L.ph_h2d.argtypes = [vp, vp, sz]
        L.ph_d2h.argtypes = [vp, vp, sz]
        L.ph_d2h_2d.argtypes = [vp, sz, vp, sz, sz, sz]
        L.ph_memset.argtypes = [vp, i32, sz]
        L.ph_cutlass_prepare.argtypes = [i32, i32, i32, i32, vp, vp, vp, vp, vp, vp, sz, vp, C.POINTER(vp)]
        L.ph_cutlass_launch.argtypes = [vp, vp]
        L.ph_cutlass_release.argtypes = [vp]
        L.ph_cutlass_query.argtypes = [i32, i32, i32, i32, C.POINTER(sz), C.POINTER(sz), C.POINTER(sz)]
        L.ph_cutlass_describe.argtypes = [i32, C.POINTER(C.c_char_p), C.POINTER(i32), C.POINTER(i32)]
        #: a library built before the tile orders (cutlass_api.h's swizzle and raster) has only CUTLASS's default order
        self.has_cutlass_sched = hasattr(L, "ph_cutlass_query_sched")
        if self.has_cutlass_sched:
            L.ph_cutlass_query_sched.argtypes = [i32] * 6 + [C.POINTER(sz)] * 3
            L.ph_cutlass_prepare_sched.argtypes = [i32] * 6 + [vp, vp, vp, vp, vp, vp, sz, vp, C.POINTER(vp)]
        L.ph_mma_peak.argtypes = [i32, i32, i32, i32, vp, C.POINTER(C.c_float), vp]
        L.ph_capture_begin.argtypes = [vp]
        L.ph_capture_end.argtypes = [vp, C.POINTER(vp), C.POINTER(sz)]
        L.ph_graph_launch.argtypes = [vp, vp]
        L.ph_graph_destroy.argtypes = [vp]
        L.ph_event_record.argtypes = [vp, vp]
        L.ph_event_sync.argtypes = [vp]
        L.ph_event_ms.argtypes = [C.POINTER(C.c_float), vp, vp]
        L.ph_stream_sync.argtypes = [vp]
        L.ph_free.argtypes = [vp]
        L.ph_event_destroy.argtypes = [vp]
        #: a library built before the dependent chain has no derive: it still times independent bursts
        self.has_chain = hasattr(L, "ph_derive") and hasattr(L, "ph_stream_wait_event")
        if self.has_chain:
            L.ph_derive.restype = L.ph_stream_wait_event.restype = i32
            L.ph_derive.argtypes = [i32, vp, i32, i32, i32, vp, vp, i32, vp]
            L.ph_stream_wait_event.argtypes = [vp, vp]
        self._side = None
        if C.sizeof(Info) != L.ph_info_size():
            raise NativeError("libpouw_harness.so's structs don't match native.py's")
        count = i32()
        self.ck(L.ph_init(C.byref(count)))
        self.device_count = count.value
        info = Info()
        self.ck(L.ph_device_info(C.byref(info)))
        self.info = {"name": info.name.decode(), "uuid": uuid_string(bytes(info.uuid)), "pci_bus_id": info.pci_bus_id.decode(),
                     "capability": f"{info.major}.{info.minor}", "sms": info.sms, "l2_bytes": info.l2_bytes,
                     "smem_optin": info.smem_optin, "mem_bus_width": info.mem_bus_width, "total_mem": info.total_mem,
                     "cuda_runtime": info.runtime_version, "cuda_driver_api": info.driver_version,
                     "cublaslt": info.cublaslt_version}
        self.lts = [LtLib(L, "lt", info.cublaslt_version, str(path or lib_path()))]
        if (p2 := lt2_path()) is not None:
            L2 = C.CDLL(str(p2))
            bind_lt(L2)
            v = L2.ph_lt_version()
            self.lts.append(LtLib(L2, f"lt{v // 10000}", v, str(p2)))
        s = vp()
        self.ck(L.ph_stream_create(C.byref(s)))
        self.stream = s.value
        self.l2 = info.l2_bytes
        self._flush = self.alloc(4 * self.l2)
        self.memset(self._flush.ptr, 0, self._flush.nbytes)
        self._sink = self.alloc(256)
        self.sync()

    def ck(self, r: int, what: str = "") -> None:
        if r:
            raise NativeError(f"{what} {r}: {self.lib.ph_last_error().decode()}".strip())

    # memory
    def alloc(self, nbytes: int) -> Buffer:
        p = vp()
        self.ck(self.lib.ph_malloc(C.byref(p), sz(nbytes)), "alloc")
        return Buffer(p.value, nbytes)

    def free(self, buf: Buffer | None) -> None:
        if buf is not None:
            self.ck(self.lib.ph_free(vp(buf.ptr)), "free")

    def memset(self, ptr: int, value: int, n: int) -> None:
        self.ck(self.lib.ph_memset(vp(ptr), value, n), "memset")

    def write(self, ptr: int, data: bytes) -> None:
        self.ck(self.lib.ph_h2d(vp(ptr), C.c_char_p(data), len(data)), "h2d")

    def read(self, ptr: int, n: int) -> bytes:
        b = C.create_string_buffer(n)
        self.ck(self.lib.ph_d2h(b, vp(ptr), n), "d2h")
        return b.raw

    def read_rows(self, ptr: int, pitch: int, width: int, rows: list[int]) -> bytes:
        """`width` bytes of each of `rows` (row r at ptr + r·pitch), concatenated."""
        return b"".join(self.read(ptr + r * pitch, width) for r in rows)

    def fill(self, fmt: int, buf: Buffer, units: int, seed: int, stream: int | None = None) -> None:
        self.ck(self.lib.ph_fill(fmt, vp(buf.ptr), units, u64(seed & (2**64 - 1)), vp(stream or self.stream)), "fill")

    def sync(self) -> None:
        self.ck(self.lib.ph_sync(), "sync")

    def mem_info(self) -> tuple[int, int]:
        f, t = sz(), sz()
        self.ck(self.lib.ph_mem_info(C.byref(f), C.byref(t)))
        return f.value, t.value

    # timing
    def flush(self, stream: int | None = None) -> None:
        """The read-only L2 flush: one pass over 4x the L2 of a buffer zeroed once."""
        self.ck(self.lib.ph_read(vp(self._flush.ptr), self._flush.nbytes, 1, vp(self._sink.ptr), 4, vp(stream or self.stream)))

    def write_flush(self, stream: int | None = None) -> None:
        """The write flush (the W5 control only): zeroing 4x the L2 leaves dirty lines."""
        self.ck(self.lib.ph_fill(8, vp(self._flush.ptr), self._flush.nbytes, u64(0), vp(stream or self.stream)))

    def event(self) -> int:
        e = vp()
        self.ck(self.lib.ph_event_create(C.byref(e)))
        return e.value

    def record(self, e: int, stream: int | None = None) -> None:
        self.ck(self.lib.ph_event_record(vp(e), vp(stream or self.stream)))

    def event_destroy(self, e: int) -> None:
        self.ck(self.lib.ph_event_destroy(vp(e)))

    @property
    def side_stream(self) -> int:
        """A second stream (non-blocking), for work a chain forks off its timing stream and joins at the chain's end."""
        if self._side is None:
            s = vp()
            self.ck(self.lib.ph_stream_create(C.byref(s)))
            self._side = s.value
        return self._side

    def wait(self, stream: int, e: int) -> None:
        self.ck(self.lib.ph_stream_wait_event(vp(stream), vp(e)), "stream wait")

    def derive(self, src: int, y: int, m: int, n: int, kind: int, a: int, sf: int | None, k: int,
               stream: int | None = None) -> None:
        """The chain's f (derive.py): A (m x k, and its block scales) from y (m x n), in stream order."""
        self.ck(self.lib.ph_derive(src, vp(y), m, n, kind, vp(a), vp(sf) if sf else None, k, vp(stream or self.stream)),
                "derive")

    def elapsed_ms(self, a: int, b: int) -> float:
        self.ck(self.lib.ph_event_sync(vp(b)))
        ms = C.c_float()
        self.ck(self.lib.ph_event_ms(C.byref(ms), vp(a), vp(b)))
        return ms.value

    def capture(self, body) -> Graph:
        """A CUDA graph of whatever `body(stream)` enqueues on the timing stream (thread-local capture mode: a sync,
        allocation or copy inside it fails the capture)."""
        self.ck(self.lib.ph_capture_begin(vp(self.stream)), "capture begin")
        try:
            body(self.stream)
        finally:
            x, n = vp(), sz()
            r = self.lib.ph_capture_end(vp(self.stream), C.byref(x), C.byref(n))
        self.ck(r, "capture end")
        return Graph(x.value, n.value)

    def graph_launch(self, g: Graph, stream: int | None = None) -> None:
        self.ck(self.lib.ph_graph_launch(vp(g.exec), vp(stream or self.stream)))

    def graph_destroy(self, g: Graph) -> None:
        self.ck(self.lib.ph_graph_destroy(vp(g.exec)))

    # gates
    def xor_byte(self, ptr: int, off: int, mask: int = 0x01, stream: int | None = None) -> None:
        self.ck(self.lib.ph_xor_byte(vp(ptr), off, mask, vp(stream or self.stream)))

    def count_eq(self, buf: Buffer, word_bytes: int, pattern: int) -> int:
        c = C.c_ulonglong()
        self.ck(self.lib.ph_count_eq(vp(buf.ptr), buf.nbytes // word_bytes, word_bytes, pattern, C.byref(c)))
        return c.value

    # characterization
    def read_bw(self, buf: Buffer, passes: int, stream: int | None = None) -> None:
        self.ck(self.lib.ph_read(vp(buf.ptr), buf.nbytes, passes, vp(self._sink.ptr), 4, vp(stream or self.stream)))

    def copy(self, dst: Buffer, src: Buffer, stream: int | None = None) -> None:
        self.ck(self.lib.ph_copy(vp(dst.ptr), vp(src.ptr), min(dst.nbytes, src.nbytes), 4, vp(stream or self.stream)))

    def mma_peak(self, kind: int, blocks: int, threads: int, iters: int, scratch: Buffer) -> float:
        ms = C.c_float()
        self.ck(self.lib.ph_mma_peak(kind, blocks, threads, iters, vp(scratch.ptr), C.byref(ms), vp(self.stream)))
        return ms.value

    def mma_flops(self, kind: int) -> int:
        return self.lib.ph_mma_flops(kind)


# ---- NVML ------------------------------------------------------------------------------------------------------------------

#: throttle (clock event) reasons that mean the clock was pulled down by power, heat or the board
BAD_THROTTLE = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}
CLOCK_GRAPHICS, CLOCK_SM, CLOCK_MEM = 0, 1, 2


class Nvml:
    """One GPU's NVML handle, by UUID. `sample()` is the per-rep record; `identity()` the card's."""

    def __init__(self, uuid: str):
        self.lib = C.CDLL(settings.POUW_LIBNVML)
        self.uuid = uuid
        self._ck(self.lib.nvmlInit_v2())
        self.h = vp()
        self._ck(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.h)))
        self.reasons = getattr(self.lib, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            self.lib.nvmlDeviceGetCurrentClocksThrottleReasons

    def _ck(self, r: int) -> None:
        if r:
            self.lib.nvmlErrorString.restype = C.c_char_p
            raise NativeError(f"NVML {r}: {self.lib.nvmlErrorString(r).decode()}")

    def _u(self, f, *args) -> int | None:
        v = C.c_uint()
        return v.value if getattr(self.lib, f)(self.h, *args, C.byref(v)) == 0 else None

    def _s(self, f, n: int = 96) -> str | None:
        b = C.create_string_buffer(n)
        return b.value.decode() if getattr(self.lib, f)(self.h, b, n) == 0 else None

    def sample(self) -> dict:
        thr, energy = C.c_ulonglong(), C.c_ulonglong()
        self.reasons(self.h, C.byref(thr))
        self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h, C.byref(energy))
        return {"sm_mhz": self._u("nvmlDeviceGetClockInfo", CLOCK_SM), "mem_mhz": self._u("nvmlDeviceGetClockInfo", CLOCK_MEM),
                "temp_c": self._u("nvmlDeviceGetTemperature", 0), "power_w": (self._u("nvmlDeviceGetPowerUsage") or 0) / 1e3,
                "energy_mj": energy.value, "throttle": hex(thr.value)}

    def supported_clocks(self) -> dict[int, list[int]]:
        """Supported graphics clocks per memory clock."""
        out: dict[int, list[int]] = {}
        n = C.c_uint(64)
        mems = (C.c_uint * 64)()
        if self.lib.nvmlDeviceGetSupportedMemoryClocks(self.h, C.byref(n), mems) != 0:
            return out
        for mem in list(mems)[:n.value]:
            g, gc = C.c_uint(512), (C.c_uint * 512)()
            if self.lib.nvmlDeviceGetSupportedGraphicsClocks(self.h, C.c_uint(mem), C.byref(g), gc) == 0:
                out[int(mem)] = sorted(int(x) for x in list(gc)[:g.value])
        return out

    def identity(self) -> dict:
        drv = C.create_string_buffer(96)
        self.lib.nvmlSystemGetDriverVersion(drv, 96)
        lo, hi = C.c_uint(), C.c_uint()
        self.lib.nvmlDeviceGetPowerManagementLimitConstraints(self.h, C.byref(lo), C.byref(hi))
        return {"name": self._s("nvmlDeviceGetName"), "uuid": self.uuid, "driver": drv.value.decode(),
                "vbios": self._s("nvmlDeviceGetVbiosVersion"), "board_part_number": self._s("nvmlDeviceGetBoardPartNumber"),
                "serial": self._s("nvmlDeviceGetSerial"),
                "power_limit_w": (self._u("nvmlDeviceGetPowerManagementLimit") or 0) / 1e3,
                "power_limit_default_w": (self._u("nvmlDeviceGetPowerManagementDefaultLimit") or 0) / 1e3,
                "power_limit_enforced_w": (self._u("nvmlDeviceGetEnforcedPowerLimit") or 0) / 1e3,
                "power_limit_range_w": [lo.value / 1e3, hi.value / 1e3],
                "clocks_max_mhz": {"graphics": self._u("nvmlDeviceGetMaxClockInfo", CLOCK_GRAPHICS),
                                   "sm": self._u("nvmlDeviceGetMaxClockInfo", CLOCK_SM),
                                   "mem": self._u("nvmlDeviceGetMaxClockInfo", CLOCK_MEM)},
                "clocks_application_mhz": {"graphics": self._u("nvmlDeviceGetApplicationsClock", CLOCK_GRAPHICS),
                                           "mem": self._u("nvmlDeviceGetApplicationsClock", CLOCK_MEM)},
                "clocks_default_application_mhz": {"graphics": self._u("nvmlDeviceGetDefaultApplicationsClock", CLOCK_GRAPHICS),
                                                   "mem": self._u("nvmlDeviceGetDefaultApplicationsClock", CLOCK_MEM)},
                "persistence_mode": self._u("nvmlDeviceGetPersistenceMode"),
                "memory_bus_width": self._u("nvmlDeviceGetMemoryBusWidth"), "cores": self._u("nvmlDeviceGetNumGpuCores"),
                "pcie_max_gen": self._u("nvmlDeviceGetMaxPcieLinkGeneration"),
                "pcie_max_width": self._u("nvmlDeviceGetMaxPcieLinkWidth"), "sample": self.sample()}


def smi(uuid: str, *args: str) -> dict:
    """nvidia-smi for this GPU only (by UUID): its exit code and output."""
    if not shutil.which("nvidia-smi"):
        return {"returncode": None, "out": "no nvidia-smi"}
    r = subprocess.run(["nvidia-smi", "-i", uuid, *args], capture_output=True, text=True)
    return {"returncode": r.returncode, "out": (r.stdout + r.stderr).strip()[-4000:]}


def versions(dev: Device) -> dict:
    """Driver, CUDA runtime and driver API, cuBLASLt (from the process), nvcc, numpy, Python, and the build record (its SASS
    MMA counts), which must describe the loaded library."""
    import hashlib
    import json
    import platform

    import numpy as np

    lib = lib_path()
    build_json = lib.parent / "build.json"
    build = json.loads(build_json.read_text()) if build_json.exists() else None
    lib_sha = hashlib.sha256(lib.read_bytes()).hexdigest()
    if build is not None and build.get("lib_sha256") != lib_sha:
        raise RuntimeError(f"{build_json} describes another library (sha256 {build.get('lib_sha256')}, loaded {lib_sha})")
    libs = [{"prefix": t.prefix, "version": t.version, "path": t.path,
             "sha256": hashlib.sha256(Path(t.path).read_bytes()).hexdigest()} for t in dev.lts]
    lt2 = (build or {}).get("lt2")
    for t in libs[1:]:
        if build is not None and Path(t["path"]).parent == lib.parent and (lt2 or {}).get("lib_sha256") != t["sha256"]:
            raise RuntimeError(f"{build_json} describes another libpouw_lt2.so ({(lt2 or {}).get('lib_sha256')}, loaded "
                               f"{t['sha256']})")
    nvcc = shutil.which("nvcc") or ((settings.CUDA_HOME or "/usr/local/cuda") + "/bin/nvcc")
    maps = Path("/proc/self/maps")
    loaded = sorted({line.split()[-1] for line in maps.read_text().splitlines() if "libcublasLt" in line}) \
        if maps.exists() else None
    return {"cuda_runtime": dev.info["cuda_runtime"], "cuda_driver_api": dev.info["cuda_driver_api"],
            "cublaslt": dev.info["cublaslt"], "cublaslt_libs": libs, "cublaslt_loaded": loaded, "numpy": np.__version__, "python": platform.python_version(),
            "nvcc": subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
            if os.path.exists(nvcc) else None,
            "lib": str(lib), "lib_sha256": lib_sha, "build": build}
