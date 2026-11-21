"""What both Pearl-C runners (pearl_c_sm90/run.py, pearl_c_sm120/run.py) share of their driver-API plumbing, hashing helpers
and check constants: standard library only, loaded by each run.py by path (a ship's copy beside it first) and named by its
bytes' SHA-256, which `kernel.source_digest` covers as ``driver.py``."""

from __future__ import annotations

import ctypes as C
import json
import os

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER


NVML = {
    "nvmlInit_v2": [], "nvmlDeviceGetHandleByUUID": [C.c_char_p, P(vp)], "nvmlSystemGetDriverVersion": [C.c_char_p, u],
    "nvmlDeviceGetClockInfo": [vp, i32, P(u)], "nvmlDeviceGetCurrentClocksThrottleReasons": [vp, P(C.c_ulonglong)],
    "nvmlDeviceGetPowerUsage": [vp, P(u)], "nvmlDeviceGetTemperature": [vp, i32, P(u)],
}


class Args(C.Structure):
    _fields_ = [("p", C.c_uint64 * 16), ("i", C.c_int32 * 16)]


def bind(lib, sigs):
    for f, a in sigs.items():
        getattr(lib, f).argtypes, getattr(lib, f).restype = a, i32
    return lib


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def uuid_str(raw: bytes) -> str:
    h = raw[:16].hex()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


class Nvml:
    """NVML on the device of one UUID; reads nothing (and says so, and why) if NVML or the UUID isn't there."""

    def __init__(self, uuid: str):
        self.lib, self.h, self.driver, self.uuid = None, vp(), "", uuid
        self.why = f"NVML did not start, or has no device of UUID {uuid}"
        try:
            lib = bind(C.CDLL(os.environ.get("PEARLC_LIBNVML", "libnvidia-ml.so.1")), NVML)
        except (OSError, AttributeError) as e:
            self.why = f"NVML did not load ({type(e).__name__}: {e})"
            return
        if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.h)) == 0:
            self.lib = lib
            b = C.create_string_buffer(96)
            if lib.nvmlSystemGetDriverVersion(b, 96) == 0:
                self.driver = b.value.decode()

    def read(self) -> dict:
        if not self.lib:
            return {"nvml": "unavailable", "why": self.why}
        sm, mem, pw, t, thr = u(), u(), u(), u(), C.c_ulonglong()
        self.lib.nvmlDeviceGetClockInfo(self.h, 1, C.byref(sm))
        self.lib.nvmlDeviceGetClockInfo(self.h, 2, C.byref(mem))
        self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(self.h, C.byref(thr))
        self.lib.nvmlDeviceGetPowerUsage(self.h, C.byref(pw))
        self.lib.nvmlDeviceGetTemperature(self.h, 0, C.byref(t))
        return {"sm_mhz": sm.value, "mem_mhz": mem.value, "throttle": hex(thr.value), "power_w": round(pw.value / 1000, 1),
                "temp_c": t.value}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


TILE = 64                                        # the leaf's tile


#: a tree node's bytes: SHA-512's
NB = 64


#: hash_fast.cuh's blocks: the segment kernels' (RS_FAST_NT, as built) and s512_folds' (RS_FOLD_NT)
RS_FAST_NT, RS_FOLD_NT = int(os.environ.get("PEARLC_RS_FAST_NT", "128")), 32


#: their scratch, the segment digests of A's rows, B's and the tiles'
SEGDIG = ("segdig_a", "segdig_b", "segdig_t")


#: an hm96 leaf's salt in bytes (`hm96.SHA512.salt_bytes`, hash_h3.cuh's HM_SALT) and the widest level s512_tree_small (64
#: threads) takes
HM_SALT, S512_SMALL = 192, 128


#: the trees with hm96 leaves, each with its salts in its own buffer ``salts_<tree>``: A's rows, B's rows and the tiles
SALTED = ("a", "b", "t")


def rs_fold_counts(name: str, i) -> str | None:
    """Why s512_folds' segment counts are not counts, None if they are.  An int slot is an int32: a negative count
    reaches the kernel as its low 32 bits, so it is refused always.  A zero is refused where a unit would fold it:
    i1 for the units below i2, i3 for the rest of i0 (a caller passes 0 for a count no unit has: the tile leaves' last
    row block when every block is full)."""
    units, s, first, rest = i[:4]
    if s < 0 or rest < 0:
        return f"{name} takes counts of 1 KiB segments, not i1 = {s}, i3 = {rest} (an int32 slot keeps a negative's low bits)"
    if not s and min(units, first) > 0 or not rest and units > first:
        return (f"{name} takes at least one 1 KiB segment a unit, not {s} for the first {min(units, first)} units and "
                f"{rest} for the other {max(units - first, 0)}")
    return None


def pow2(n: int) -> int:
    """A tree's level-0 width over n leaves."""
    return 1 << max(n - 1, 0).bit_length()


def depth(n: int) -> int:
    return pow2(n).bit_length() - 1


def salts_at(at: int, name: str, count: int, given: int | None) -> int:
    """The address `at` of a buffer of hm96 salts as a leaf kernel may read it, else a refusal naming the buffer `name`: an
    address says nothing of the bytes behind it, so the buffer's owner states its length `given`, which is `count` leaves'
    `HM_SALT` bytes each, no fewer and no more (none stated is refused, not assumed); and the kernels read a salt by
    16-byte loads (hash_h3.cuh's `load_salt`), so the address is a multiple of 16."""
    if given is None:
        raise ValueError(f"{name}: its owner stated no length, and the launch reads {HM_SALT * count} bytes ({count} leaves, "
                         f"{HM_SALT} a leaf): a caller's salts are (address, bytes)")
    if given != HM_SALT * count:
        raise ValueError(f"{name}: {given} bytes for {count} leaves, which take {HM_SALT * count} ({HM_SALT} a leaf)")
    if at % 16:
        raise ValueError(f"{name} at {at:#x}: the leaf kernels read a salt by 16-byte loads, so the buffer's address is a "
                         "multiple of 16 bytes (a slice of a buffer from an odd offset is not)")
    return at


def stated_salts(bufs: dict) -> tuple[dict[str, int], dict[str, int]]:
    """A caller's buffers `bufs` as (each buffer's address, each tree's salts' stated length): a ``salts_<tree>`` buffer is
    given as (address, bytes), the length its owner states.  One given as a bare address has none, and no launch reads it
    (`Pipeline.salts`)."""
    b, said = dict(bufs), {}
    for name, given in bufs.items():
        if name.startswith("salts_") and isinstance(given, tuple):
            b[name], said[name[len("salts_"):]] = (int(x) for x in given)
    return b, said


PLAIN_BUFFERS = {"qa", "qb", "sa", "sb", "y_plain", "y_fast", "qa_s5", "qb_s5", "y_adopt"}


GEMM_RERUNS = (("gemm_pearlc", ("c", "u", "y")), ("gemm_ct", ("c2",)), ("gemm_plain", ("y_plain",)), ("gemm_fast", ("y_fast",)),
               ("af", ("af",)), ("bf", ("bf",)), ("gemm_adopt", ("y_adopt",)), ("gemm_pearlc_s", ("y",)))


ALIASES = {"c2": "c"}


#: what pd's launches write (pd_commit: A's commitment and E_A's lines; pd_tiles: the tile leaves)
PD_OUTPUTS = ("rowdig_a", "tree_a", "seed_a", "line_a", "seeds_a", "lkeys_a", "ea_codes", "leaves")


POISON = 0xA5


CALL_PHASES = ("commit", "forming", "gemm", "cleanup", "hashing")


TIMING_KEYS = {"line_b": 32, "dom_a": 64, "dom_b": 64, "dom_t": 64, "unit": 104, "hm": 128 + 192 * 256 * 64}


def parse_shape(s: str) -> tuple[int, int, int]:
    """SHAPES' entries: "8192" (m = n = k) or "MxNxK"."""
    parts = [int(x) for x in s.lower().split("x")]
    return (parts[0],) * 3 if len(parts) == 1 else tuple(parts)
