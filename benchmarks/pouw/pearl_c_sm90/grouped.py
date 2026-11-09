"""Pearl-C's grouped call on sm_90a: a fused-MoE layer's gathered rows, every routed expert's, through pearl_c_sm90.cu's
grouped kernels in one launch a step, where `run.Pipeline` takes a call per expert.

The rows come gathered by expert (expert position e's ``counts[e]`` rows, in order), and the experts' weight side stacked
(`STACKED`: each kind's buffer holds every expert's, at its position).  `units` cuts each expert's rows into blocks of at most
pd_commit's PD_MAX_ROWS, and each block is a unit: one matmul of m₁₆ rows (its rows rounded up to 16, the filler zero) under
its expert's weight, with its own index, keys, A tree and tile leaves, as `run.Pipeline` commits those rows called alone.  The
units lie one after another in one A, each in a segment of its m₁₆ rounded up to SEG = 64 rows (the grouped GEMMs' tile), and
each 64-row block of A names its expert (``ex``).  The steps, one launch each (`Grouped.steps`):

* ``commit_a``: pd_commit_gm, unit u's pd_commit (A's commitment and E_A's lines) on the CTAs `rows_order` gives its rows,
  from its record (`Grouped.meta`); a unit's filler rows (m to m₁₆, zero) take a zero row's digest (`zero_state`, a
  function of k alone) rather than hash their rows, the same bytes.  Each CTA has an SM to itself (`run.pd_setup`'s
  PD_WHOLE) while the units' real rows fit the SMs (`commit_rows`: a filler row's CTA is done at once, so it holds no SM
  from a real row), and past that takes its least shared memory (PD_PAIR_SMEM), so the rows past the SMs share them rather than
  wait a whole chain for one (the launch's alone: the code and bytes are the same).  A cubin without pd_commit_gm runs
  pd_commit_g, unit u's rows on CTA row u, at an SM to a CTA only while every unit's m₁₆ rows fit;
* ``stats_a``, ``eprime_a``, ``pa``: stats_s5, eprime and pa over all of A's rows (a row at a time, no keys);
* ``form_a``: form_s5_g, each 64-row block's F_A its expert's;
* ``af``, ``gemm_pearlc``: gemm_words_g_d and gemm_pearlc_g_d over A, each 64-row block's B side its expert's;
* ``hash_leaf``: pd_tiles_gw (or pd_tiles_g), unit u's tile leaves on CTA row u, into the pass's leaves at its first tile,
  `tiles_g` tiles of a row block a CTA (`TILES_PD`).

With ``split`` (`SPLIT_STEPS`, as `run.Pipeline`'s ``split_a`` without ``fillers``), A's commitment is two launches of the same
bytes: ``rows_a``, pd_commit_gm in mode 1 (each unit's pads, row digests, leaves, seeds, line keys and E_A lines), then
``tree_seed_a``, pd_tree_g, unit u's pd_tree on CTA u (A's levels, seed_A and line_a), which writes nothing the forming reads:
a unit's padding rows (m₁₆ to its segment's end, outside its matmul) keep the E_A lines the buffer holds.

pd_commit_gm, pd_tree_g and pd_tiles_g run hash_decode.cuh's bodies on each unit's Args, and the rest compute each row, and
each output, as the ungrouped kernels do, so a unit's bytes are its own call's."""

from __future__ import annotations

import ctypes as C
import os
from dataclasses import dataclass

import numpy as np

#: hash_decode.cuh's variant the grouped launches take (pearl_c_sm90.cu's PD_GROUPED): a CTA a tile
PD = 851
#: the tile leaves' variant where the cubin has it (PD_GROUPED_TILES): PD_GROUP's, PD_G tiles of a row block a CTA, their folds
#: side by side on one warp, about a third of PD's issue at decode's partial row blocks, which a call's tile leaves share the
#: SMs with the next call's pd_commit_gm on
TILES_PD = 979
#: the tiles of a row block a CTA of pd_tiles_gw (pearl_c_sm90.cu's PD_GW), where the cubin has it and PD_G's CTAs are past
#: the SMs (`tiles_g`): a quarter of their CTAs, in one wave
TILES_GW = 16
#: PROBE: the older forms, "+"-joined, for an A/B in one session: tiles851 (the tile leaves at PD's), tiles4 (pd_tiles_g's
#: PD_G tiles a CTA rather than pd_tiles_gw's TILES_GW), treewhole (pd_tree_g at
#: a whole SM's shared memory), hashzero (pd_commit_gm hashes the filler rows: no zero row digest in the records), rows2d
#: (pd_commit_g's grid, a unit's rows on its CTA row, rather than pd_commit_gm's), pairsmem (pd_commit_gm at an SM to a CTA
#: only while every unit's m₁₆ rows fit the SMs, not its real rows: `commit_rows`)
PLACE = frozenset(os.environ.get("PEARLC_GROUPED_PLACE", "").split("+")) - {"", "base"}
#: a unit's segment rows divide by it: the grouped GEMMs' tile rows (run.BM_D) and the tile leaves' (run.TILE)
SEG = 64
STEPS = ("commit_a", "stats_a", "eprime_a", "form_a", "af", "pa", "gemm_pearlc", "hash_leaf")
SPLIT_STEPS = ("rows_a", "tree_seed_a", *STEPS[1:])
#: the steps that hash, which a call without hashing skips
HASH_STEPS = frozenset({"commit_a", "rows_a", "tree_seed_a", "hash_leaf"})
#: the weight side's buffers the grouped kernels read stacked, each expert's at its position
STACKED = ("bt", "pb_t", "ia_b", "fbT32", "fa_codes")
#: pearl_c_sm90.cu's Args
ARGS = np.dtype([("p", "<u8", 16), ("i", "<i4", 16)])
#: a unit's record in `meta`: its pd_commit Args, its pd_tiles Args, its keys (dom_a ‖ unit, 168 bytes), then its pd_tree Args
REC, TILES_AT, KEYS_AT, TREE_AT, KEY_BYTES = 768, 192, 384, 576, 168
NB = 64


@dataclass(frozen=True)
class Unit:
    pos: int      # its expert's position in the call
    src: int      # its first row among the gathered rows
    m: int        # its rows
    m16: int      # the matmul's rows: m rounded up to 16
    row: int      # its segment's first row of A
    seg: int      # the segment's rows: m16 rounded up to SEG
    tile: int     # its first tile, counted from the call's
    tiles: int


def units(counts, n: int, rows: int) -> list[Unit]:
    """A call's units: each expert position's rows (``counts``) in blocks of at most `rows`, in order."""
    out, src, row, tile = [], 0, 0, 0
    for pos, count in enumerate(counts):
        for lo in range(0, count, rows):
            m = min(rows, count - lo)
            m16 = -(-m // 16) * 16
            seg, tiles = -(-m16 // SEG) * SEG, -(-m16 // 64) * (n // 64)
            out.append(Unit(pos, src + lo, m, m16, row, seg, tile, tiles))
            row, tile = row + seg, tile + tiles
        src += count
    return out


def tree_bytes(R) -> int:
    """A unit's A-tree slot: a tree of PD_MAX_ROWS leaves."""
    return NB * (2 * R.PD_MAX_ROWS - 1)


def rows_order(us: list[Unit], sms: int) -> list[tuple[int, int]]:
    """pd_commit_gm's CTAs in launch order, each (unit, row): the units' rows on CTAs of their own below `sms`, where a whole
    row's chain (about 87 µs at decode) runs alone on its SM, and their zero filler rows (a short chain, `zero_state`) on the
    rest.  Past `sms`, the CTAs x and x + sms share an SM as the launch places them on an idle GPU, so the first CTAs past
    it and the CTAs that share their SMs are filler rows while there are enough."""
    rows = [(j, r) for j, u in enumerate(us) for r in range(u.m)]
    fill = [(j, r) for j, u in enumerate(us) for r in range(u.m, u.m16)]
    head = min(max(len(rows) + len(fill) - sms, 0), len(fill))
    return fill[:head] + rows + fill[head:]


def root_at(R, j: int, u: Unit) -> int:
    """Unit j's A root's byte offset in ``tree_a``: its slot's node 2 pow2(m₁₆) - 2."""
    return tree_bytes(R) * j + NB * (2 * R.pow2(u.m16) - 2)


def sizes(R, rows: int, count: int, n: int, k: int) -> dict[str, int]:
    """The grouped call's own buffers at A of at most `rows` rows in at most `count` units (the experts' stacked side aside):
    ``meta`` holds a call's unit records, its expert map (``ex``, int32 a 64-row block), its gathered rows' rows of A and its
    units' root nodes (int64), and pd_commit_gm's map (u32 a row)."""
    return {"xa": 4 * rows * k, "alpha_a": 4 * rows, "beta_a": 4 * rows, "ia_a": 4 * rows, "aux_a": 8 * rows,
            "epa_codes": 32 * rows, "ea_codes": 32 * rows, "at": rows * k, "af": 4 * 32 * rows, "pa_t": 128 * rows,
            "c": 4 * rows * n, "u": 4 * rows * n, "y": 2 * rows * n, "rowdig_a": 64 * rows, "seeds_a": 32 * rows,
            "lkeys_a": 32 * rows, "tree_a": tree_bytes(R) * count, "seed_a": 32 * count, "line_a": 32 * count,
            "pd_count": 16 * count, "meta": REC * count + 4 * (rows // SEG) + 8 + 8 * rows + 8 * count + 4 * rows}


def pd_smem(R, dev, name: str) -> int:
    """Kernel `name`'s dynamic shared memory as `run.pd_setup` gives pd_commit's and pd_tree's, set as its maximum."""
    fn = dev.fn(name)
    optin = dev.get("cuDeviceGetAttribute", R.MAX_SHARED_OPTIN, dev.dev)
    smem = optin - dev.get("cuFuncGetAttribute", R.SHARED_SIZE, fn) if PD & R.PD_WHOLE else R.PD_PAIR_SMEM
    if smem < R.PD_PAIR_SMEM:
        raise RuntimeError(f"{name} gets {smem} bytes of dynamic shared memory, not its {R.PD_PAIR_SMEM}")
    dev.ck("cuFuncSetAttribute", fn, R.MAX_DYN, smem)
    return smem


def setup(R, dev) -> None:
    """`run.pd_setup`, pd_commit_g's and pd_commit_gm's dynamic shared memory (`pd_smem`), and pd_tiles_gw's maximum at its
    PD_PAIR_SMEM / 2, which with its static digests (33 KiB) is past the 48 KiB a launch gets without it."""
    R.pd_setup(dev)
    if getattr(dev, "pd_smem_g", None) is None:
        names = [f"pd_commit_g_{PD}", *([f"pd_commit_gm_{PD}"] if has(dev, f"pd_commit_gm_{PD}") else [])]
        dev.pd_smem_g = min(pd_smem(R, dev, name) for name in names)
        if has(dev, f"pd_tiles_gw_{TILES_PD}"):
            dev.ck("cuFuncSetAttribute", dev.fn(f"pd_tiles_gw_{TILES_PD}"), R.MAX_DYN, R.PD_PAIR_SMEM // 2)


def has(dev, name: str) -> bool:
    """Whether the cubin has kernel `name` (a ship built before pd_tree_g or pd_tiles_g_979 lacks it), looked up once."""
    got = getattr(dev, "grouped_has", None)
    if got is None:
        got = dev.grouped_has = {}
    if name not in got:
        try:
            dev.fn(name)
            got[name] = True
        except RuntimeError:
            got[name] = False
    return got[name]


def has_tree(dev) -> bool:
    """Whether the cubin has pd_tree_g (one built before it runs only unsplit calls)."""
    return has(dev, f"pd_tree_g_{PD}")


def has_rows(dev) -> bool:
    """Whether the call's commitment runs as pd_commit_gm (`rows_order`'s CTAs): where the cubin has it."""
    return "rows2d" not in PLACE and has(dev, f"pd_commit_gm_{PD}")


def commit_rows(dev, us: list[Unit]) -> int:
    """The rows of a call's commitment that each want an SM to themselves: pd_commit_gm's real rows, since its filler rows
    take `zero_state` without a chain; pd_commit_g's (or hashing the filler rows) every unit's m₁₆."""
    if has_rows(dev) and not PLACE & {"hashzero", "pairsmem"}:
        return sum(u.m for u in us)
    return sum(u.m16 for u in us)


def tiles_pd(dev) -> int:
    """pd_tiles_g's variant: TILES_PD where the cubin has it, else PD."""
    return PD if "tiles851" in PLACE or not has(dev, f"pd_tiles_g_{TILES_PD}") else TILES_PD


def tiles_g(R, dev, us: list[Unit], n: int) -> int:
    """TILES_PD's tiles a CTA for units `us` at `n` columns: TILES_GW where the cubin has pd_tiles_gw and PD_G's CTAs are past
    the SMs (decode at 6144 columns, 192; prefill), so none waits a whole chain for one; else PD_G (pd_tiles_g's), whose
    CTAs (128 at 4096 columns) each run fewer chains at once."""
    if "tiles4" in PLACE or not has(dev, f"pd_tiles_gw_{TILES_PD}"):
        return R.PD_G
    ctas = sum(R.cdiv(u.m16, 64) * R.cdiv(n // 64, R.PD_G) for u in us)
    return TILES_GW if ctas > dev.sms else R.PD_G


def tree_setup(R, dev) -> None:
    """pd_tree_g's dynamic shared memory, at a split call's first use: PD_PAIR_SMEM, not PD_WHOLE's whole SM, since its CTAs
    run beside gemm_pearlc_g_d, whose CTA (129 KiB) fits beside one at that and would otherwise wait its chain out."""
    if getattr(dev, "pd_tree_smem_g", None) is None:
        name = f"pd_tree_g_{PD}"
        if "treewhole" in PLACE:
            dev.pd_tree_smem_g = pd_smem(R, dev, name)
        else:
            dev.ck("cuFuncSetAttribute", dev.fn(name), R.MAX_DYN, R.PD_PAIR_SMEM)
            dev.pd_tree_smem_g = R.PD_PAIR_SMEM


def zero_state(k: int) -> bytes:
    """The row digest of k zero FP32 words (`pearl_c.row_digest`) as pd_commit_gm reads it for a unit's filler rows: its eight
    state words, each from eight of the digest's bytes big-endian."""
    from verity.ml.protocols.pouw.schemes.pearl_c import row_digest
    return np.frombuffer(row_digest(bytes(4 * k)), ">u8").astype("<u8").tobytes()


def gemm_smem(R, dev, name: str) -> int:
    """A GEMM's dynamic shared memory, from its _smem probe once, set as its maximum (`run.Pipeline.gemm`'s)."""
    cache = R.Pipeline.GEMM_SMEM
    if name not in cache:
        probe = dev.alloc(16)
        dev.launch(name + "_smem", 1, 1, p=(probe,))
        dev.sync()
        cache[name] = int.from_bytes(dev.read(probe, 4), "little")
        dev.free(probe)
        dev.ck("cuFuncSetAttribute", dev.fn(name), R.MAX_DYN, cache[name])
    return cache[name]


class Grouped:
    """The grouped calls of one (n, k) and expert count over buffers the caller owns (`b`: device pointers, `sizes`' at `rows`
    and `count`, and the experts' `STACKED`), with the constant keys `keys` (``salt_key``, ``hm``).  Its tensor maps, two sets
    in its own ``tma`` (the chain's: A, the experts' B̃, P_A, their P_B; A'F_B's: A and their F_B transposed), are encoded once,
    and a zero row's digest (`zero_state`) goes up once into its own ``zero``."""

    def __init__(self, R, dev, n: int, k: int, experts: int, rows: int, count: int, b: dict[str, int], keys: dict[str, int]):
        if n % 64 or k % 256 or k > R.PD_MAX_K or rows % SEG:
            raise ValueError(f"the grouped kernels take n a multiple of 64, k of 256 up to {R.PD_MAX_K} and rows of {SEG}, "
                             f"not {(n, k, rows)}")
        self.R, self.dev, self.n, self.k, self.experts, self.rows, self.count = R, dev, n, k, experts, rows, count
        self.b, self.keys = dict(b), keys
        setup(R, dev)
        self.tma = dev.alloc(2 * R.MAP_SET)
        d, e = dev, experts
        sets = (d.tmap(b["at"], k, rows, SEG) + d.tmap(b["bt"], k, e * n, 64) + d.tmap(b["pa_t"], R.PEEL_ROW, rows, SEG)
                + d.tmap(b["pb_t"], R.PEEL_ROW, e * n, 64),
                (d.tmap(b["at"], k, rows, SEG) + d.tmap(b["fbT32"], k, e * R.WORDS_N, R.WORDS_N)) * 2)
        for slot, maps in enumerate(sets):
            d.ck("cuMemcpyHtoD_v2", R.u64(self.tma + R.MAP_SET * slot), C.c_char_p(maps), R.MAP_SET)
        self.zero = dev.alloc(NB)
        d.ck("cuMemcpyHtoD_v2", R.u64(self.zero), C.c_char_p(zero_state(k)), NB)

    def free(self) -> None:
        self.dev.free(self.tma)
        self.dev.free(self.zero)

    def meta(self, us: list[Unit], keys: bytes, leaves: int, split: bool = False) -> bytes:
        """A call's ``meta``: each unit's record (its pd_commit Args, with ``zero`` and its rows past which A's rows are its
        zero filler rows at p14 and i4, its pd_tiles Args, its 168 key bytes from `keys`, the units' in order, and with `split`
        its pd_tree Args, pd_commit's then in mode 1), the expert map, each gathered row's row of A and each unit's A root as a
        node of ``tree_a`` (int64s, for the caller's gathers), then pd_commit_gm's map (`rows_order`, unit << 16 | row a CTA,
        at `map_at`); `leaves` the address of the call's first tile leaf."""
        if len(us) > self.count or sum(u.seg for u in us) > self.rows or len(keys) != KEY_BYTES * len(us):
            raise ValueError(f"{len(us)} units of {sum(u.seg for u in us)} rows, {len(keys)} key bytes: past the call's "
                             f"{self.count} units and {self.rows} rows, or not {KEY_BYTES} key bytes a unit")
        b, k, n, kk, R = self.b, self.k, self.n, self.keys, self.R
        rec = np.zeros((len(us), REC), np.uint8)
        commit, tiles, tree = np.zeros(len(us), ARGS), np.zeros(len(us), ARGS), np.zeros(len(us), ARGS)
        for j, u in enumerate(us):
            key = b["meta"] + REC * j + KEYS_AT
            commit["p"][j, :13] = (b["xa"] + 4 * k * u.row, b["rowdig_a"] + 64 * u.row, b["tree_a"] + tree_bytes(R) * j,
                                   b["seeds_a"] + 32 * u.row, b["lkeys_a"] + 32 * u.row, key + 64, key, kk["salt_key"], kk["hm"],
                                   b["pd_count"] + 16 * j, b["seed_a"] + 32 * j, b["line_a"] + 32 * j, b["ea_codes"] + 32 * u.row)
            commit["p"][j, 14] = 0 if "hashzero" in PLACE else self.zero
            commit["i"][j, :5] = (u.m16, k, int(split), u.seg, u.m)
            tiles["p"][j, :3] = (b["c"] + 4 * n * u.row, b["u"] + 4 * n * u.row, leaves + NB * u.tile)
            tiles["i"][j, :2] = (u.m16, n)
            if split:
                tree["p"][j, :12] = (0, 0, b["tree_a"] + tree_bytes(R) * j, 0, 0, key + 64, key, 0, 0, 0, b["seed_a"] + 32 * j,
                                     b["line_a"] + 32 * j)
                tree["i"][j, :4] = (u.m16, 0, 0, u.seg)
        rec[:, :TILES_AT] = commit.view(np.uint8).reshape(len(us), -1)
        rec[:, TILES_AT:KEYS_AT] = tiles.view(np.uint8).reshape(len(us), -1)
        rec[:, KEYS_AT:KEYS_AT + KEY_BYTES] = np.frombuffer(keys, np.uint8).reshape(len(us), KEY_BYTES)
        rec[:, TREE_AT:] = tree.view(np.uint8).reshape(len(us), -1)
        ex = np.repeat(np.array([u.pos for u in us], "<i4"), [u.seg // SEG for u in us])
        rows = np.concatenate([np.arange(u.row, u.row + u.m, dtype="<i8") for u in us])
        nodes = np.array([root_at(R, j, u) // NB for j, u in enumerate(us)], "<i8")
        ctas = np.array([j << 16 | r for j, r in rows_order(us, self.dev.sms)], "<u4")
        return rec.tobytes() + ex.tobytes() + bytes(-len(ex) * 4 % 8) + rows.tobytes() + nodes.tobytes() + ctas.tobytes()

    @staticmethod
    def rows_at(us: list[Unit]) -> int:
        """The byte offset in ``meta`` of the gathered rows' rows of A; the units' root nodes follow them."""
        ex = sum(u.seg for u in us) // SEG
        return REC * len(us) + 4 * ex + (-ex * 4 % 8)

    @staticmethod
    def map_at(us: list[Unit]) -> int:
        """The byte offset in ``meta`` of pd_commit_gm's map, after the units' root nodes."""
        return Grouped.rows_at(us) + 8 * (sum(u.m for u in us) + len(us))

    def steps(self, us: list[Unit], split: bool = False) -> list[tuple[str, object]]:
        """A call's launches (`STEPS`, or with `split` `SPLIT_STEPS`), its `meta` (`meta`'s, at the same `split`) uploaded."""
        d, b, k, n, R = self.dev, self.b, self.k, self.n, self.R
        rows, count, meta = sum(u.seg for u in us), len(us), b["meta"]
        ex, s = meta + REC * count, k // 256
        smem = d.pd_smem_g if commit_rows(d, us) <= d.sms else R.PD_PAIR_SMEM

        def commit():
            block = 32 * (R.cdiv(s, 32) * 2 + 2)
            if has_rows(d):
                d.launch(f"pd_commit_gm_{PD}", sum(u.m16 for u in us), block, smem, p=(meta, meta + self.map_at(us)), i=(REC,))
            else:
                d.launch(f"pd_commit_g_{PD}", (max(u.m16 for u in us), count), block, smem, p=(meta,), i=(REC,))

        def tree():
            d.launch(f"pd_tree_g_{PD}", (1, count), 128, d.pd_tree_smem_g, p=(meta + TREE_AT,), i=(REC,))

        def tiles():
            if tv & R.PD_GROUP:
                tg = tiles_g(R, d, us, n)
                grid = max(R.cdiv(u.m16, 64) * R.cdiv(n // 64, tg) for u in us)
                d.launch(f"pd_tiles_g{'w' if tg == TILES_GW else ''}_{tv}", (grid, count), 32 * tg, R.PD_PAIR_SMEM // 2,
                         p=(meta + TILES_AT,), i=(REC,))
            else:
                d.launch(f"pd_tiles_g_{tv}", (max(u.tiles for u in us), count), 32, p=(meta + TILES_AT,), i=(REC,))

        tv = tiles_pd(d)
        if split:
            tree_setup(R, d)
        return list(zip(SPLIT_STEPS if split else STEPS, (
            commit, *((tree,) if split else ()),
            lambda: d.launch("stats_s5", R.cdiv(rows, 8), 256, p=(b["xa"], b["alpha_a"], b["beta_a"], b["ia_a"], b["aux_a"]),
                             i=(rows, k)),
            lambda: d.launch("eprime", R.cdiv(rows, 128), 128, p=(b["ea_codes"], b["beta_a"], b["epa_codes"]), i=(rows,)),
            lambda: d.launch("form_s5_g", (k // 128, rows // SEG), 128,
                             p=(b["xa"], b["alpha_a"], b["epa_codes"], b["fa_codes"], b["at"], ex), i=(rows, k)),
            lambda: self.gemm("gemm_words_g_d", 1, rows, R.WORDS_N, ex, (b["af"],)),
            lambda: d.launch("pa", R.cdiv(rows * 64, 256), 256, p=(b["epa_codes"], 0, b["af"], b["pa_t"]), i=(rows,)),
            lambda: self.gemm("gemm_pearlc_g_d", 0, rows, n, ex, (b["c"], b["u"], b["y"]), b["pa_t"], b["pb_t"], b["ia_a"], b["ia_b"]),
            tiles)))

    def gemm(self, name: str, slot: int, M: int, N: int, ex: int, out: tuple, pa: int = 0, pb: int = 0, rs: int = 0,
             cs: int = 0) -> None:
        """A grouped GEMM over A's M rows (the chain at map set 0, A'F_B at 1), its expert map at p12."""
        R, d = self.R, self.dev
        smem = gemm_smem(R, d, name)
        tm, bn = R.gemm_tile(name)
        p = [self.b["at"], self.b["bt" if slot == 0 else "fbT32"], 0, 0, 0, pa, pb, rs, cs, 0, self.tma + R.MAP_SET * slot, 0, ex]
        p[2:2 + len(out)] = out
        d.launch(name, min((M // tm) * (N // bn), d.sms), R.gemm_threads(name), smem, p=p, i=(M, N, self.k, R.RASTER))
