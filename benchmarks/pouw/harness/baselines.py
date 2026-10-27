"""The plain GEMM baselines: per family (precision), every cuBLASLt heuristic candidate (up to 32) from each cuBLASLt loaded
(`native.LtLib`), with `enumerate` every configuration cuBLASLt's capabilities admit (`lt_enumerated`), and every CUTLASS
sm_120 kernel that implements the shape (with `sched` the 3.x ones in each tile order too), each gated bit-exactly, then
autotuned; the fastest per family is what the arms divide by.

GEMM alone on pre-quantized operands, TN (A m x k and B n x k, both K-major), D m x n row-major in BF16 (int8: int32),
alpha 1, beta 0. Scale factors are all 1 (their values don't change a kernel's time; their bytes are read as usual).

**The gate** runs each candidate twice at the timed shape on a gate operand set of ternary values ({-1, 0, 1}, scales 1),
whose sums are exact under any accumulator the card could have, and checks the second call (so that whatever the first
leaves behind, a stream-K workspace say, can't break the calls after it): D is filled with a sentinel before each, then no
sentinel word may remain, and 32 x 32 sampled entries (the corners among them) must equal numpy's exact product, bit for bit.
**The known-bad variants** (the harness's negative control): the same candidate with one output byte flipped at a sampled
entry, and cuBLASLt over half of k; the gate must reject both, or the run stops.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import random
from dataclasses import dataclass, field

import fill as F
import numpy as np
from arm import Buffer, Shape

LT_WORKSPACE = 32 << 20
#: a verity entry's workspace at least: its two 128-byte CUtensorMaps (cutlass_api.h)
PLAIN_WORKSPACE = 256
LT_CANDIDATES = 32
LT_ENUMERATED_MAX = 8192
#: a cuBLASLt configuration's identity (with the library and the fp8-blk128 swap)
LT_CONFIG = ("algo_id", "tile", "stages", "splitk", "reduction", "swizzle", "custom", "inner_shape", "cluster_shape")
SAMPLE_ROWS = SAMPLE_COLS = 32
SENTINEL = {"bf16": (2, 0xFFFF, 0xFF), "int32": (4, 0x7F7F7F7F, 0x7F)}  # (word bytes, word, memset byte): a BF16 NaN; an
#                                                                        int32 no ternary sum reaches


@dataclass(frozen=True)
class Family:
    name: str
    lt_kind: int | None
    cutlass_kind: int | None
    fmt: int           # the operands' fill format (timing); its ternary twin for the gate
    scale: str | None  # "tensor" (one f32), "outer" (f32 per row), "ue8m0/32", "ue4m3/16", "f32/128", or None
    out: str = "bf16"

    def elem_bytes(self) -> float:
        return {F.BF16: 2, F.E4M3: 1, F.E2M1: 0.5, F.INT8: 1}[self.fmt]


FAMILIES = {f.name: f for f in (
    Family("bf16", 0, 0, F.BF16, None),
    Family("fp8-e4m3", 1, 1, F.E4M3, "tensor"),
    Family("fp8-e4m3-rowwise", 2, None, F.E4M3, "outer"),
    Family("mxfp8", 3, 3, F.E4M3, "ue8m0/32"),
    Family("fp8-blk128", 4, 4, F.E4M3, "f32/128"),
    Family("nvfp4", 5, 5, F.E2M1, "ue4m3/16"),
    Family("mxfp4", 6, 6, F.E2M1, "ue8m0/32"),
    Family("int8", 7, 7, F.INT8, None, "int32"),
)}


def _pad(x: int, a: int) -> int:
    return -(-x // a) * a


def scale_bytes(scale: str | None, rows: int, k: int) -> int:
    """cuBLASLt's scale tensor for an operand of `rows` x k (the block formats in its 128 x 4 tiled layout)."""
    if scale is None:
        return 0
    if scale == "tensor":
        return 4
    if scale == "outer":
        return 4 * rows
    if scale == "ue8m0/32":
        return _pad(rows, 128) * _pad(-(-k // 32), 4)
    if scale == "ue4m3/16":
        return _pad(rows, 128) * _pad(-(-k // 16), 4)
    if scale == "f32/128":
        return 4 * _pad(rows, 4) * _pad(-(-k // 128), 4)
    raise ValueError(scale)


def scale_fill(scale: str) -> int:
    return {"ue8m0/32": F.UE8M0_ONE, "ue4m3/16": F.UE4M3_ONE}.get(scale, F.F32_ONE)


def operand_bytes(fmt: int, rows: int, k: int) -> int:
    return rows * k // 2 if fmt == F.E2M1 else rows * k * F.unit_bytes(fmt)


def fill_units(fmt: int, nbytes: int) -> int:
    return nbytes // F.unit_bytes(fmt)


# ---- candidates --------------------------------------------------------------------------------------------------------------


@dataclass
class Candidate:
    """One kernel of one family at one shape: its source (cublaslt; cutlass, or verity for nvf4_plain.cu's entries in the
    same registry), id and config."""

    family: str
    source: str
    name: str
    config: dict
    lt: object = None       # (plan, LtCandidate) for cuBLASLt
    cutlass: int | None = None
    sf_bytes: tuple[int, int] = (0, 0)
    ws: int = 0
    ltlib: object = None    # the native.LtLib that owns the plan


@dataclass
class Operands:
    """One operand set of one family: A, B, their scales and D."""

    a: Buffer
    b: Buffer
    sa: Buffer | None
    sb: Buffer | None
    d: Buffer


class BaselineCall:
    """A candidate prepared on one operand set, with the arm interface's `Call` members. A verity entry keeps its tensor maps
    in its workspace from prepare to every launch, so it gets a workspace of its own; the others share `lt_ws`."""

    def __init__(self, dev, cand: Candidate, shape: Shape, ops: Operands, lt_ws: Buffer):
        self.dev, self.cand, self.shape, self.ops = dev, cand, shape, ops
        self.outputs = {"d": ops.d}
        self._h, self._own_ws = None, None
        if cand.cutlass is not None:
            ws = lt_ws
            if cand.source == "verity":
                ws = self._own_ws = dev.alloc(max(cand.ws, PLAIN_WORKSPACE))
            h = C.c_void_p()
            sw, r = cand.config.get("swizzle", 0), cand.config.get("raster", 0)
            dims = (cand.cutlass, shape.m, shape.n, shape.k) + ((sw, r) if sw or r else ())
            prepare = dev.lib.ph_cutlass_prepare_sched if sw or r else dev.lib.ph_cutlass_prepare
            try:
                dev.ck(prepare(*dims, ops.a.ptr, ops.b.ptr, ops.sa.ptr if ops.sa else None, ops.sb.ptr if ops.sb else None,
                               ops.d.ptr, ws.ptr, ws.nbytes, dev.stream, C.byref(h)), f"{cand.source} prepare {cand.name}")
            except Exception:
                self.free()
                raise
            self._h = h.value
        self._ws = lt_ws

    def launch(self, stream: int) -> None:
        d, o = self.dev, self.ops
        if self.cand.cutlass is not None:
            d.ck(d.lib.ph_cutlass_launch(C.c_void_p(self._h), C.c_void_p(stream)), self.cand.name)
        else:
            plan, c = self.cand.lt
            lt = self.cand.ltlib
            lt.ck(lt.lib.ph_lt_run(plan, C.byref(c), o.a.ptr, o.b.ptr, o.d.ptr, o.sa.ptr if o.sa else None,
                                   o.sb.ptr if o.sb else None, self._ws.ptr, stream), self.cand.name)

    def weight_side(self, stream: int) -> None:
        pass

    def free(self) -> None:
        if self._h:
            self.dev.lib.ph_cutlass_release(C.c_void_p(self._h))
            self._h = None
        if self._own_ws is not None:
            self.dev.free(self._own_ws)
            self._own_ws = None


def _lt_plans(dev, fam: Family, shape: Shape, lt, ld: int | None):
    """(swap, plan) per plan the family has: fp8-blk128 has two (the 128 x 128 blocks on either operand)."""
    for swap in ((0, 1) if fam.name == "fp8-blk128" else (0,)):
        plan = C.c_void_p()
        lt.ck(lt.lib.ph_lt_plan_create_ld(fam.lt_kind, shape.m, shape.n, shape.k, ld or shape.k, LT_WORKSPACE, swap,
                                          C.byref(plan)))
        yield swap, plan


def lt_candidates(dev, fam: Family, shape: Shape, ops: Operands, want: int = LT_CANDIDATES,
                  ld: int | None = None, refused: list[str] | None = None, lt=None) -> list[Candidate]:
    """cuBLASLt's top `want` heuristic algorithms for the shape (A and B with leading dimension `ld`, default k), from `lt`
    (a `native.LtLib`; the harness library's own by default). Each candidate holds its plan; `release_plans` frees them.
    Where cuBLASLt refuses the family outright (on sm_120, cuBLASLt 12.9's heuristic query returns INVALID_VALUE for MXFP4),
    that is no candidate, and the reason goes to `refused`."""
    from native import LtCandidate, NativeError

    if fam.lt_kind is None:
        return []
    lt = lt or dev.lts[0]
    out: list[Candidate] = []
    for swap, plan in _lt_plans(dev, fam, shape, lt, ld):
        arr, got = (LtCandidate * want)(), C.c_int()
        try:
            lt.ck(lt.lib.ph_lt_candidates(plan, ops.sa.ptr if ops.sa else None, ops.sb.ptr if ops.sb else None, want,
                                          arr, C.byref(got)), f"{lt.prefix} heuristics {fam.name}")
        except NativeError as ex:
            lt.lib.ph_lt_plan_destroy(plan)
            if refused is None:
                raise
            refused.append(str(ex))
            continue
        if got.value == 0:
            lt.lib.ph_lt_plan_destroy(plan)
        for i in range(got.value):
            c = LtCandidate()
            C.memmove(C.byref(c), C.byref(arr[i]), C.sizeof(c))
            cfg = c.config() | {"heuristic_rank": i, "blk128_swap": swap, "library": lt.prefix, "cublaslt": lt.version}
            out.append(Candidate(fam.name, "cublaslt", f"{lt.prefix}_{fam.name}_{swap}_{i}_algo{c.algo_id}_tile{c.tile}",
                                 cfg, lt=(plan, c), ws=c.workspace, ltlib=lt))
    return out


def lt_key(c: Candidate) -> tuple:
    return (c.config["library"], c.config["blk128_swap"], *(c.config[k] for k in LT_CONFIG))


def lt_splitks(shape: Shape, sms: int) -> list[int]:
    """The split-k counts the enumeration tries: 1, and up to 16 where 128 x 128 output tiles would fill fewer than two
    waves of the SMs (decode's skinny m)."""
    tiles = -(-shape.m // 128) * -(-shape.n // 128)
    return [1] if tiles >= 2 * sms else [1, 2, 3, 4, 6, 8, 12, 16]


@dataclass(frozen=True)
class LtPick:
    """Which of an enumeration's configurations a run takes: index (the enumeration's order, per library and swap) in
    [start, stop), and, if `names` is given, named in it. The enumeration keeps the first max(limit, stop) and counts every
    one it accepts, so a slice past the default limit still reaches its configurations and the run reports the total."""
    start: int = 0
    stop: int | None = None
    names: frozenset[str] | None = None
    limit: int = LT_ENUMERATED_MAX

    @property
    def want(self) -> int:
        return max(self.limit, self.stop or 0)

    def takes(self, index: int, name: str) -> bool:
        return self.start <= index and (self.stop is None or index < self.stop) and (self.names is None or name in self.names)

    def record(self) -> dict:
        return {"start": self.start, "stop": self.stop, "limit": self.limit,
                "names": None if self.names is None else len(self.names)}


#: the default pick: every configuration, up to the default limit
EVERY = LtPick()


def lt_enumerated(dev, fam: Family, shape: Shape, ops: Operands, lt, splitks: list[int], seen: set,
                  refused: list[str] | None = None, pick: LtPick = EVERY) -> tuple[list[Candidate], dict]:
    """Every configuration of every algorithm `lt` has for the family's types that cublasLtMatmulAlgoCheck accepts at the
    shape (lt.cu's ph_lt_enumerate) and `pick` takes, less those in `seen` (the heuristic's, by `lt_key`). Returns them
    and what was tried: configurations checked, accepted, whether the limit cut the list short, the sha256 of the
    taken configurations' names in order and, if `pick` names configurations, the names it took (`named`), a heuristic's
    duplicate included."""
    from native import LtCandidate, NativeError

    want = pick.want
    info = {"checked": 0, "accepted": 0, "new": 0, "truncated": False, "splitks": splitks, "pick": pick.record(),
            "taken": 0}
    if pick.names is not None:
        info["named"] = []
    taken = hashlib.sha256()
    if fam.lt_kind is None or not lt.can_enumerate:
        return [], info | {"skipped": "no cuBLASLt path" if fam.lt_kind is None else "library predates ph_lt_enumerate"}
    out: list[Candidate] = []
    ks = (C.c_int * len(splitks))(*splitks)
    for swap, plan in _lt_plans(dev, fam, shape, lt, None):
        arr, got, checked = (LtCandidate * want)(), C.c_int(), C.c_int()
        try:
            lt.ck(lt.lib.ph_lt_enumerate(plan, ops.sa.ptr if ops.sa else None, ops.sb.ptr if ops.sb else None, ks,
                                         len(splitks), want, arr, C.byref(got), C.byref(checked)),
                  f"{lt.prefix} enumerate {fam.name}")
        except NativeError as ex:
            lt.lib.ph_lt_plan_destroy(plan)
            if refused is None:
                raise
            refused.append(str(ex))
            continue
        info["checked"] += checked.value
        info["accepted"] += got.value
        info["truncated"] |= got.value > want and pick.stop is None
        mine = []
        for i in range(min(got.value, want)):
            c = LtCandidate()
            C.memmove(C.byref(c), C.byref(arr[i]), C.sizeof(c))
            cfg = c.config() | {"enumerated": i, "blk128_swap": swap, "library": lt.prefix, "cublaslt": lt.version}
            cand = Candidate(fam.name, "cublaslt", _enum_name(lt.prefix, fam.name, swap, cfg), cfg, lt=(plan, c),
                             ws=c.workspace, ltlib=lt)
            if not pick.takes(i, cand.name):
                continue
            info["taken"] += 1
            taken.update(cand.name.encode() + b"\n")
            if pick.names is not None:
                info["named"].append(cand.name)
            if lt_key(cand) not in seen:
                seen.add(lt_key(cand))
                mine.append(cand)
        if not mine:
            lt.lib.ph_lt_plan_destroy(plan)
        out += mine
    info["new"] = len(out)
    info["taken_sha256"] = taken.hexdigest()
    return out, info


def _enum_name(prefix: str, fam: str, swap: int, c: dict) -> str:
    extra = "".join(f"_{tag}{c[k]}" for k, tag in (("splitk", "sk"), ("reduction", "r"), ("swizzle", "sw"), ("custom", "c"),
                                                    ("inner_shape", "i"), ("cluster_shape", "cl")) if c[k] and
                    not (k == "splitk" and c[k] == 1))
    return f"{prefix}_{fam}_{swap}_algo{c['algo_id']}_tile{c['tile']}_st{c['stages']}{extra}"


def release_plans(dev, cands: list[Candidate]) -> None:
    for plan, lt in {c.lt[0].value: c.ltlib for c in cands if c.lt is not None}.items():
        lt.lib.ph_lt_plan_destroy(C.c_void_p(plan))
    for c in cands:
        c.lt = None


def cutlass_kernels(dev) -> list[dict]:
    out = []
    for i in range(dev.lib.ph_cutlass_count()):
        name, kind, swap = C.c_char_p(), C.c_int(), C.c_int()
        if dev.lib.ph_cutlass_describe(i, C.byref(name), C.byref(kind), C.byref(swap)) == 0:
            out.append({"index": i, "name": name.value.decode(), "kind": kind.value, "swap": swap.value})
    return out


#: the 3.x tile scheduler's max_swizzle_size values past its default (0, which orders as 1)
CUTLASS_SWIZZLES = (2, 4, 8)
#: raster 1 along our m, 2 along our n (cutlass_api.h)
CUTLASS_RASTERS = (1, 2)


def cutlass_tile(name: str) -> tuple[int, int] | None:
    """A 3.x kernel's CTA tile (M, N) from its name (`cutlass3x_<family>_<M>x<N>x<K>_...`); None for a 2.x kernel, whose
    tile order is fixed at compile time."""
    parts = name.split("_")
    if parts[0] != "cutlass3x":
        return None
    for dims in (p.split("x") for p in parts[1:]):
        if len(dims) == 3 and all(d.isdigit() for d in dims):
            return int(dims[0]), int(dims[1])
    return None


def cutlass_swizzles(tile: tuple[int, int], shape: Shape, swap: int) -> list[int]:
    """The swizzles that order this shape's tiles differently from each other and from the default: CUTLASS lowers
    max_swizzle_size to 4 unless the grid is at least 6 CTAs on its short side, to 2 below 3, and to 1 below 2
    (`get_log_swizzle_size`, tile_scheduler_params.h), so a small grid's larger swizzles repeat a smaller one's order."""
    rows, cols = (shape.n, shape.m) if swap else (shape.m, shape.n)
    short = min(-(-rows // tile[0]), -(-cols // tile[1]))
    return [s for s, need in zip(CUTLASS_SWIZZLES, (2, 3, 6)) if short >= need]


def registry_source(name: str) -> str:
    """A registry kernel's source: ``verity`` for nvf4_plain.cu's (NVFP4 and FP8), else ``cutlass``."""
    return "verity" if name.startswith("verity_") else "cutlass"


def cutlass_sched_name(name: str, swizzle: int, raster: int) -> str:
    return name + (f"_raster{'MN'[raster - 1]}" if raster else "") + (f"_sw{swizzle}" if swizzle else "")


def cutlass_candidates(dev, fam: Family, shape: Shape, kernels: list[dict],
                       sched: bool = False) -> tuple[list[Candidate], list[dict]]:
    """The CUTLASS kernels of this family that implement the shape, and the ones that don't (with why). With `sched`, also
    each 3.x kernel in every tile order its scheduler takes (`CUTLASS_RASTERS` x swizzles, as `cutlass_swizzles` leaves
    them, less the default), each a candidate of its own with config `raster` and `swizzle`."""
    ok, no = [], []
    for kr in kernels:
        if fam.cutlass_kind is None or kr["kind"] != fam.cutlass_kind:
            continue
        tile = cutlass_tile(kr["name"])
        orders = [(0, 0)]
        if sched and tile is not None:
            sws = [0, *cutlass_swizzles(tile, shape, kr["swap"])]
            orders += [(sw, r) for r in (0, *CUTLASS_RASTERS) for sw in sws if (sw, r) != (0, 0)]
        for sw, r in orders:
            name = cutlass_sched_name(kr["name"], sw, r)
            ws, sfa, sfb = C.c_size_t(), C.c_size_t(), C.c_size_t()
            args = (kr["index"], shape.m, shape.n, shape.k)
            st = (dev.lib.ph_cutlass_query_sched(*args, sw, r, C.byref(ws), C.byref(sfa), C.byref(sfb)) if (sw, r) != (0, 0)
                  else dev.lib.ph_cutlass_query(*args, C.byref(ws), C.byref(sfa), C.byref(sfb)))
            if st:
                no.append({"name": name, "status": st})
                continue
            cfg = {"swap": kr["swap"], "workspace": ws.value} | ({"swizzle": sw, "raster": r} if (sw, r) != (0, 0) else {})
            ok.append(Candidate(fam.name, registry_source(kr["name"]), name, cfg, cutlass=kr["index"],
                                sf_bytes=(sfa.value, sfb.value), ws=ws.value))
    return ok, no


# ---- operands --------------------------------------------------------------------------------------------------------------


@dataclass
class ShapeOperands:
    """Every family's operand sets at one shape: the timing sets (full-entropy, seeded) and the gate set (ternary). A and B
    are shared between families of one element format; each family has its own scales; D is shared between families of one
    output type."""

    shape: Shape
    sets: int
    seed: int
    data: dict = field(default_factory=dict)     # (fmt, set) -> (A, B); set -1 is the gate set
    scales: dict = field(default_factory=dict)   # (family, set) -> (sa, sb)
    outs: dict = field(default_factory=dict)     # (out, set) -> D
    allocated: list = field(default_factory=list)

    def ops(self, fam: Family, s: int) -> Operands:
        a, b = self.data[(fam.fmt, s)]
        sa, sb = self.scales.get((fam.name, s), (None, None))
        return Operands(a, b, sa, sb, self.outs[(fam.out, s)])

    def free(self, dev) -> None:
        for buf in self.allocated:
            dev.free(buf)
        self.allocated.clear()


def set_seed(seed: int, shape: Shape, role: str, s: int) -> int:
    return hash_seed(f"{seed}/{shape.label}/{role}/{s}")


def hash_seed(text: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def allocate(dev, shape: Shape, fams: list[Family], sets: int, seed: int, sf_bytes: dict[str, tuple[int, int]],
             extra_fmts: tuple[int, ...] = ()) -> ShapeOperands:
    """Allocates and fills every family's operand sets (and the gate set, index -1) on the device, and A and B in each of
    `extra_fmts` (an arm's inputs that no family has)."""
    so = ShapeOperands(shape, sets, seed)

    def new(nbytes: int) -> Buffer:
        b = dev.alloc(nbytes)
        so.allocated.append(b)
        return b

    for fmt in sorted({f.fmt for f in fams} | set(extra_fmts)):
        for s in [-1, *range(sets)]:
            a, b = new(operand_bytes(fmt, shape.m, shape.k)), new(operand_bytes(fmt, shape.n, shape.k))
            f = F.TERNARY[fmt] if s < 0 else fmt
            dev.fill(f, a, fill_units(f, a.nbytes) if fmt != F.E2M1 else a.nbytes, set_seed(seed, shape, f"a{fmt}", s))
            dev.fill(f, b, fill_units(f, b.nbytes) if fmt != F.E2M1 else b.nbytes, set_seed(seed, shape, f"b{fmt}", s))
            so.data[(fmt, s)] = (a, b)
    for fam in fams:
        if fam.scale is None:
            continue
        need_a = max(scale_bytes(fam.scale, shape.m, shape.k), sf_bytes.get(fam.name, (0, 0))[0])
        need_b = max(scale_bytes(fam.scale, shape.n, shape.k), sf_bytes.get(fam.name, (0, 0))[1])
        for s in [-1, *range(sets)]:
            sa, sb = new(need_a), new(need_b)
            fmt = scale_fill(fam.scale)
            dev.fill(fmt, sa, fill_units(fmt, sa.nbytes), 0)
            dev.fill(fmt, sb, fill_units(fmt, sb.nbytes), 0)
            so.scales[(fam.name, s)] = (sa, sb)
    for out in sorted({f.out for f in fams}):
        wb = SENTINEL[out][0]
        for s in [-1, *range(sets)]:
            so.outs[(out, s)] = new(shape.m * shape.n * wb)
    dev.sync()
    return so


def weight_bytes(fam: Family, shape: Shape) -> int:
    """B and its scales: one operand set's weight."""
    return operand_bytes(fam.fmt, shape.n, shape.k) + scale_bytes(fam.scale, shape.n, shape.k)


def memory_bytes(shape: Shape, fams: list[Family], sets: int, extra_fmts: tuple[int, ...] = ()) -> int:
    fmts = {f.fmt for f in fams} | set(extra_fmts)
    outs = {f.out for f in fams}
    per = sum(operand_bytes(f, shape.m, shape.k) + operand_bytes(f, shape.n, shape.k) for f in fmts)
    per += sum(scale_bytes(f.scale, shape.m, shape.k) + scale_bytes(f.scale, shape.n, shape.k) for f in fams)
    per += sum(shape.m * shape.n * SENTINEL[o][0] for o in outs)
    return per * (sets + 1)


# ---- the gate ------------------------------------------------------------------------------------------------------------------


def sample_indices(n: int, count: int, rng: random.Random) -> list[int]:
    picks = {0, n - 1}
    while len(picks) < min(count, n):
        picks.add(rng.randrange(n))
    return sorted(picks)


def expected(fam: Family, a_rows: np.ndarray, b_rows: np.ndarray) -> np.ndarray:
    """The exact product of sampled rows (float64 dots of exact values), in D's words."""
    v = a_rows @ b_rows.T
    if fam.out == "int32":
        return v.astype(np.int64).astype(np.int32).view(np.uint32)
    return F.bf16_rne(v)


def gate(dev, fam: Family, shape: Shape, call: BaselineCall, rng_seed: int, corrupt: bool = False) -> dict:
    """Runs `call` twice on the gate set and checks the second call's D (see the module docstring). `corrupt` flips a byte
    of D at a sampled entry after the launches: the negative control."""
    ops = call.ops
    wb, sentinel, byte = SENTINEL[fam.out]
    for _ in range(2):
        dev.memset(ops.d.ptr, byte, ops.d.nbytes)
        dev.sync()
        call.launch(dev.stream)
        dev.sync()
    rng = random.Random(rng_seed)
    rows, cols = sample_indices(shape.m, SAMPLE_ROWS, rng), sample_indices(shape.n, SAMPLE_COLS, rng)
    if corrupt:
        dev.xor_byte(ops.d.ptr, (rows[len(rows) // 2] * shape.n + cols[len(cols) // 2]) * wb, 0x10)
    dev.sync()
    left = dev.count_eq(ops.d, wb, sentinel)
    pitch_in = operand_bytes(fam.fmt, 1, shape.k)
    tfmt = F.TERNARY[fam.fmt]
    a = F.decode(tfmt, dev.read_rows(ops.a.ptr, pitch_in, pitch_in, rows)).reshape(len(rows), shape.k)
    b = F.decode(tfmt, dev.read_rows(ops.b.ptr, pitch_in, pitch_in, cols)).reshape(len(cols), shape.k)
    want = expected(fam, a, b)
    got_rows = np.frombuffer(dev.read_rows(ops.d.ptr, shape.n * wb, shape.n * wb, rows), np.uint16 if wb == 2 else np.uint32)
    got = got_rows.reshape(len(rows), shape.n)[:, cols]
    bad = np.argwhere(got != want)
    return {"ok": left == 0 and len(bad) == 0, "unwritten_words": int(left), "sampled": [len(rows), len(cols)],
            "mismatches": int(len(bad)),
            "first": [{"row": rows[i], "col": cols[j], "got": int(got[i, j]), "want": int(want[i, j])} for i, j in bad[:4]]}


def half_k_call(dev, fam: Family, shape: Shape, ops: Operands, lt_ws: Buffer):
    """cuBLASLt over half of k (lda stays k): a GEMM that skips half its MACs, which the gate must reject; None where the
    family has no cuBLASLt path or no algorithm for it."""
    half = Shape(shape.label + "/half-k", shape.m, shape.k // 2, shape.n)
    cands = lt_candidates(dev, fam, half, ops, want=1, ld=shape.k, refused=[])
    if not cands:
        return None
    return BaselineCall(dev, cands[0], shape, ops, lt_ws)
