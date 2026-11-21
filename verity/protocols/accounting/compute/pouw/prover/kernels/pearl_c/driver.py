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


class DevBase:
    """What both runners' devices share: the driver API's functions, memory, copies and events, on the subclass's `cu`, `mod`,
    `ck`, `fns`, `live` and `stream`."""

    def get(self, f: str, *a) -> int:
        v = i32()
        self.ck(f, C.byref(v), *a)
        return v.value

    def fn(self, name: str) -> vp:
        if name not in self.fns:
            f = vp()
            self.ck("cuModuleGetFunction", C.byref(f), self.mod, name.encode())
            self.fns[name] = f
        return self.fns[name]

    def alloc(self, n: int) -> int:
        p = u64()
        self.ck("cuMemAlloc_v2", C.byref(p), max(n, 16))
        self.ck("cuMemsetD8_v2", p, 0, max(n, 16))
        self.live.add(p.value)
        return p.value

    def free(self, p: int) -> None:
        if p in self.live:
            self.live.discard(p)
            self.ck("cuMemFree_v2", u64(p))

    def put(self, data: bytes) -> int:
        p = self.alloc(len(data))
        self.ck("cuMemcpyHtoD_v2", u64(p), C.c_char_p(data), len(data))
        return p

    def read(self, p: int, n: int) -> bytes:
        buf = C.create_string_buffer(n)
        self.ck("cuMemcpyDtoH_v2", buf, u64(p), n)
        return buf.raw

    def copy(self, dst: int, src: int, n: int) -> None:
        self.ck("cuMemcpyDtoD_v2", u64(dst), u64(src), n)

    def copy_async(self, dst: int, src: int, n: int, stream=None) -> None:
        self.ck("cuMemcpyDtoDAsync_v2", u64(dst), u64(src), n, self.stream if stream is None else stream)

    def sync(self) -> None:
        self.ck("cuCtxSynchronize")

    def timed(self, body, reps: int) -> list[float]:
        body()
        self.sync()
        out = []
        for _ in range(reps):
            a, b, ms = vp(), vp(), C.c_float()
            self.ck("cuEventCreate", C.byref(a), 0)
            self.ck("cuEventCreate", C.byref(b), 0)
            self.ck("cuEventRecord", a, None)
            body()
            self.ck("cuEventRecord", b, None)
            self.ck("cuEventSynchronize", b)
            self.ck("cuEventElapsedTime", C.byref(ms), a, b)
            self.cu.cuEventDestroy_v2(a)
            self.cu.cuEventDestroy_v2(b)
            out.append(ms.value)
        return out

    def free_mem(self) -> tuple[int, int]:
        f, t = sz(), sz()
        self.ck("cuMemGetInfo_v2", C.byref(f), C.byref(t))
        return f.value, t.value


class PipelineBase:
    """What both runners' pipelines share: the hiding commitments' salts and trees, the forming alternatives' check, and the steps' run, on the subclass's buffers, steps and kernels."""

    def salts(self, tag: str, count: int) -> int:
        """The device address of tree `tag`'s hm96 salts, read at each launch: ``salts_<tag>``, `HM_SALT` bytes a leaf in leaf
        order, which the caller has filled.  A tree whose salts nobody gave is refused: a caller's buffers without
        ``salts_<tag>``, or the pipeline's own before `put_salts`.  So is a buffer whose stated length (the pipeline's own
        size, or what a caller's ``bufs`` says) is not the `count` leaves' this launch reads, and a caller's with none
        stated (`salts_at`)."""
        at = self.b.get(f"salts_{tag}")
        if not at or tag in getattr(self, "unsalted", ()):
            raise NotImplementedError(
                f"tree {tag!r}'s hm96 leaves read their salts from the buffer salts_{tag} ({HM_SALT} bytes a leaf, in leaf "
                "order), and this pipeline's caller gave none.  The kernels derive no salt: the caller hands in fresh "
                "operating-system bytes for a call's trees and a registered row's salt from its registration's salt file "
                "(compute-accounting's goal: Pearl+ hands the leaf kernels their salts)")
        return salts_at(at, f"salts_{tag}", count, self.salt_bytes.get(tag))

    def lent(self) -> dict:
        """This pipeline's buffers as another pipeline's ``bufs``: each address, a salts buffer with its stated length."""
        return {**self.b, **{x: (self.b[x], n) for t, n in self.salt_bytes.items() if (x := f"salts_{t}") in self.b}}

    def put_salts(self, tag: str, salts: bytes) -> None:
        """`salts` from the host into tree `tag`'s ``salts_<tag>``, whole: a check's or a bench's bytes (a serving caller
        writes its own buffer on the device)."""
        if len(salts) != self.size[f"salts_{tag}"]:
            raise ValueError(f"tree {tag!r} takes {self.size[f'salts_{tag}']} bytes of salts ({HM_SALT} a leaf), not {len(salts)}")
        self.dev.ck("cuMemcpyHtoD_v2", u64(self.b[f"salts_{tag}"]), C.c_char_p(salts), len(salts))
        self.unsalted.discard(tag)

    def root(self, tag: str) -> int:
        """The device address of a tree's root, its last node."""
        return self.b[f"tree_{tag}"] + self.size[f"tree_{tag}"] - NB

    def levels(self, tag: str, count: int, dom: str | None = None) -> None:
        """A tree's levels above level 0 (leaves and pads written): s512_nodes a level at a time down to S512_SMALL nodes,
        then s512_tree_small for the rest in one launch, under the domain id `dom` (default dom_<tag>)."""
        d, base, di = self.dev, self.b[f"tree_{tag}"], self.keys[dom or f"dom_{tag}"]
        width, off, level = pow2(count), 0, 0
        while width > S512_SMALL:
            d.launch("s512_nodes", cdiv(width // 2, 64), 64, p=(base + NB * off, base + NB * (off + width), di),
                     i=(level, width // 2))
            off, width, level = off + width, width // 2, level + 1
        if width > 1:
            d.launch("s512_tree_small", 1, 64, p=(base + NB * off, di), i=(width, level))

    def tree_setup(self, tag: str, count: int, dom: str) -> list[tuple[str, object]]:
        """A tree's pads (s512_pads): its domain's alone, so once per unit."""
        d, b, width = self.dev, self.b, pow2(count)
        if width == count:
            return []
        return [(f"pad_{tag}", lambda: d.launch("s512_pads", cdiv(width - count, 64), 64, p=(b[f"tree_{tag}"], self.keys[dom]),
                                                i=(count, width)))]

    def tile_leaves(self) -> None:
        """The tile leaves over C~ and U into `leaves`: s512_tile_leaves, or with `segdig_t` s512_tile_segs then s512_folds
        (the full row blocks' tiles at 32 segments, the last block's at its rows / 2)."""
        d, b, mr, n = self.dev, self.b, self.m_real, self.n
        if self.sha == "ref" or not b.get("segdig_t"):
            d.launch("s512_tile_leaves", self.tiles, 32, p=(b["c"], b["u"], b["leaves"]), i=(mr, n))
            return
        full, last = mr // TILE * (n // TILE), mr % TILE // 2
        d.launch("s512_tile_segs", cdiv(32 * full + n // TILE * last, RS_FAST_NT), RS_FAST_NT, p=(b["c"], b["u"], b["segdig_t"]),
                 i=(mr, n))
        d.launch("s512_folds", cdiv(self.tiles, RS_FOLD_NT), RS_FOLD_NT, p=(b["segdig_t"], b["leaves"]), i=(self.tiles, 32, full, last))

    def form_alt_launch(self, tag: str, kernel: str) -> None:
        """`kernel` (one of FORM_ALTS) over one side's rows into at2."""
        b, rows = self.b, self.m if tag == "a" else self.n
        self.dev.launch(kernel, (self.k // 128, rows // 64), 128,
                        p=(b[f"x{tag}"], b[f"alpha_{tag}"], b[f"ep{tag}_codes"], b[f"f{tag}_codes"], b["at2"]), i=(rows, self.k))

    def form_alt(self, tag: str, kernel: str) -> int:
        """`kernel` over one side's rows into at2, and the words where form_s5's codes (at or bt) differ from it."""
        self.form_alt_launch(tag, kernel)
        self.dev.sync()
        return self.diff(f"{tag}t", "at2")

    def free(self) -> None:
        for name, p in self.b.items():
            if self.owned and name not in ("xa", "xb") or name in self.allocated:
                self.dev.free(p)

    def run_all(self) -> None:
        for _, _, f in self.steps():
            f()
        self.dev.sync()


def compare(dev: DevBase, pipe: PipelineBase, name: str, want: bytes, alias: str | None = None) -> dict:
    got = dev.read(pipe.b[alias or name], len(want))
    if got == want:
        return {"ok": True, "bytes": len(want)}
    bad = [i for i in range(0, len(want), 4) if got[i:i + 4] != want[i:i + 4]]
    return {"ok": False, "bytes": len(want), "words_differing": len(bad),
            "first": [{"offset": i, "got": got[i:i + 4].hex(), "want": want[i:i + 4].hex()} for i in bad[:6]]}
