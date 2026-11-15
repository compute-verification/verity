"""The device arena's layout over one session of ``flock_cuda_prove_circuit``, exactly: a replica of ``DeviceArena`` (first fit
from the bottom for blocks of 64 MiB and more, the top of the highest free block that holds them for smaller ones, 256-byte
alignment, neighbours merged on release) and every ``ffi_malloc``/``ffi_free`` and Ligerito arena call the prover makes, in its
order and at its sizes, for a shape (m, k_log, M1 or M0, the statement's slot types, level 0's padding, the ring switch's
extra claims). A session is two reps: with keep, rep 0 leaves level 0's commitment in the open arena and rep 1 takes it;
without, each rep begins and finishes the arena. ``need`` is the smallest arena (in MiB) a session fits in.

Ligerito's ladder is upstream's: ``LIGERITO[m]`` is ``prover_config_for(m - 7, embedded_initial_k_or_default(m), Fast100)``
where Flock b684b12 registers one (m 22 to 35, every shape the prover accepts) and its ``default_config(m - 7, 6, 1)``
elsewhere."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace
from typing import Iterator

MIB = 1 << 20
SMALL = 64 * MIB
#: m -> (initial_k, log_inv_rates, recursive_ks, queries)
LIGERITO = {
    21: (6, (1, 3), (4,), (241, 121)),
    22: (6, (1, 2, 4), (4, 4), (218, 106, 53)),
    23: (6, (1, 2, 3), (4, 4), (218, 106, 71)),
    24: (6, (1, 2, 3), (4, 4), (218, 106, 71)),
    25: (6, (1, 2, 3, 4), (4, 4, 4), (218, 106, 71, 53)),
    26: (6, (1, 2, 3, 4), (4, 4, 4), (218, 106, 71, 53)),
    27: (6, (1, 2, 3, 4), (4, 4, 4), (218, 106, 71, 53)),
    28: (4, (1, 2, 3, 4, 5), (4, 4, 4, 4), (218, 106, 71, 53, 43)),
    29: (5, (1, 2, 3, 4, 5), (4, 4, 4, 4), (218, 106, 71, 53, 43)),
    30: (6, (1, 2, 3, 4, 5), (4, 4, 4, 4), (218, 106, 71, 53, 43)),
    31: (6, (1, 2, 3, 4, 5, 6), (4, 4, 4, 4, 4), (218, 106, 71, 53, 43, 36)),
    32: (6, (1, 2, 3, 4, 5, 6), (4, 4, 4, 4, 4), (218, 106, 71, 53, 43, 36)),
    33: (6, (1, 2, 3, 4, 5, 6), (4, 4, 4, 4, 4), (218, 106, 71, 53, 43, 36)),
    34: (6, (1, 2, 3, 4, 5, 6, 7), (4, 4, 4, 4, 4, 4), (218, 106, 71, 53, 43, 36, 32)),
    35: (6, (1, 2, 3, 4, 5, 6, 7), (4, 4, 4, 4, 4, 4), (218, 106, 71, 53, 43, 36, 32)),
    36: (6, (1, 2, 3, 4, 5, 6, 7), (4, 4, 4, 4, 4, 4), (241, 148, 121, 110, 105, 103, 102)),
    37: (6, (1, 2, 3, 4, 5, 6, 7, 8), (4, 4, 4, 4, 4, 4, 4), (241, 148, 121, 110, 105, 103, 102, 101)),
}
F128, F256, HB = 16, 32, 64
SALT = 192  # an hm96-sha512 leaf's salt
BLOCKS = 2048  # ZT_MAX_BLOCKS, LC_MAX_BLOCKS, SMC_MAX_BLOCKS, LF_MAX_BLOCKS


@dataclass(frozen=True)
class Shape:
    m: int
    k_log: int
    zk: bool
    #: the statement's slot types' log2 columns (fc_fold holds one 2^log-word fold per type)
    type_logs: tuple[int, ...] = ()
    sub_log: int = 14
    #: M1: level 0's padding words per lane, twice its query count (zk_veil::level0_t_pad)
    t_pad: int = 512
    #: the ring switch's extra claims' free coordinates (each gathers 2^n_free words)
    extra_free: tuple[int, ...] = (0,)
    #: the zerocheck's packed path (zf_packed): None follows the prover's rule
    zc_packed: bool | None = None
    #: before the lane groups, level 0's M1 commit held every witness lane's codeword at once
    lane_groups: bool = True
    #: level 0's M1 commit transforms the witness lanes in place in its leaves (launch_ntt_rows; not under FC_NTT_REGS=0)
    l0_rows: bool = True
    #: before keep-L0, a keeping rep 0 held a and b through Ligerito and a copy of z from its witness on
    keep_ab: bool = False
    ligerito: tuple = field(default=None, compare=False)

    def lig(self) -> tuple:
        return self.ligerito or LIGERITO[self.m]

    @property
    def types(self) -> tuple[int, ...]:
        return self.type_logs or (self.k_log,)

    @property
    def queries(self) -> tuple[int, ...]:
        """Each level's query count: under M1 level 0's is half its padding."""
        q = tuple(self.lig()[3])
        return (self.t_pad // 2,) + q[1:] if self.zk else q


def zf_packed(m: int) -> bool:
    """zerocheck_fused.cuh's rule (ZT_FINISH_OP = 2^10)."""
    return m - 7 >= 3 and (1 << (m - 6)) // 16 > 1 << 10


def rep_events(s: Shape, mode: str) -> Iterator[tuple]:
    """One call's arena events: ("b",), ("a", name, bytes), ("r", name), ("e",). mode: "plain" (begin, prove, finish), "keep"
    (rep 0 keeping level 0: begin, no finish), "take" (rep 1 from rep 0's level 0: no begin, finish)."""
    m, k_log = s.m, s.k_log
    log_n = m - 7
    ln = 1 << log_n
    W = ln * F128
    ik, rates, ks, queries = s.lig()
    w = 1 << ik
    take, keep = mode == "take", mode == "keep"
    a = lambda name, n: ("a", name, n)  # noqa: E731
    r = lambda name: ("r", name)  # noqa: E731
    if not take:
        yield ("b",)
    yield a("df", W)
    yield a("d_a", W)
    yield a("d_b", W)
    if keep and s.keep_ab:
        yield a("z_copy", W)
    k_code = log_n - ik + rates[0]
    n = 1 << k_code
    l0 = ["d_cw0", "d_tree0", "d_zk_pads", "d_zk_extra"] if s.zk else ["d_cw0", "d_tree0"]
    if take:
        pass
    elif s.zk:
        l = 1 << (log_n - ik)
        yield a("d_zk_pads", (w + 4) * s.t_pad * F128)
        yield a("d_zk_extra", 4 * l * F128)
        yield a("d_cw0", n * (w + 4) * F128)
        yield a("d_tree0", (2 * n - 1) * HB)
        if s.l0_rows:
            yield a("cw_x", n * 4 * F128)
            yield r("cw_x")
        else:
            wg = w >> min(ik, 2) if s.lane_groups else w
            yield a("cw_w", n * wg * F128)
            yield a("cw_x", n * 4 * F128)
            yield r("cw_w")
            yield r("cw_x")
    else:
        yield a("d_cw0", n * w * F128)
        yield a("d_tree0", (2 * n - 1) * HB)
    # zerocheck
    packed = zf_packed(m) if s.zc_packed is None else s.zc_packed
    zc_out = 1 << (m - 6)
    zc_first, zc_next = (zc_out // 8, zc_out // 4) if packed else (zc_out, zc_out // 2)
    zt_dfull = m - 7
    zt_lobits = zt_dfull - 7 if zt_dfull > 7 else 0
    zc = [("d_eq", (1 << (m - 13)) * F128), ("d_r1ab", 64 * F128), ("d_r1c", 64 * F128), ("d_ft", 8 * 256 * F128),
          ("d_am", zc_first * F128), ("d_bm", zc_first * F128), ("d_amn", zc_next * F128), ("d_bmn", zc_next * F128),
          ("d_p1", BLOCKS * F128), ("d_pinf", BLOCKS * F128), ("d_m1d", F128), ("d_mid", F128),
          ("d_eqlo", (1 << zt_lobits) * F128), ("d_eqhi", (1 << (zt_dfull - zt_lobits)) * F128),
          ("d_part8", 8 * BLOCKS * F128), ("d_out8", 8 * F128)]
    for name, b in zc:
        yield a(name, b)
    for name, _ in zc:
        yield r(name)
    if not (keep and s.keep_ab):
        yield r("d_a")
        yield r("d_b")
    yield a("d_zlin", W)
    # lincheck
    K, KS = 1 << k_log, 1 << s.sub_log
    irl = k_log - 6
    lc = [("d_eq_inner", K * F128), ("d_comb", K * F128), ("d_comb_sub", KS * F128), ("d_ratio", 256 * F128),
          ("d_zvec", K * F128), ("d_nC", K * F128), ("d_nZ", K * F128), ("d_eq_outer", (1 << (m - k_log)) * F128),
          ("d_p1", BLOCKS * F128), ("d_pinf", BLOCKS * F128), ("d_e1", F128), ("d_einf", F128)]
    for name, b in lc:
        yield a(name, b)
    yield a("d_rest", (1 << irl) * F128)
    yield r("d_rest")
    for t, tl in enumerate(s.types):
        yield a(f"d_base{t}", (1 << tl) * F128)
    for t in range(len(s.types)):
        yield r(f"d_base{t}")
    yield a("d_zvec_initial", K * F128)
    for name, _ in lc:
        yield r(name)
    yield r("d_zlin")
    # ring switch
    yield a("d_eq_shat_ab", (1 << (irl - 1)) * F128)
    yield a("d_shat", 128 * F128)
    yield r("d_eq_shat_ab")
    yield r("d_zvec_initial")
    sfx_d = m - 7
    sfx_k = sfx_d // 2
    sfx = [("d_sab_lo", (1 << sfx_k) * F128), ("d_sab_hi", (1 << (sfx_d - sfx_k)) * F128),
           ("d_sc_lo", (1 << sfx_k) * F128), ("d_sc_hi", (1 << (sfx_d - sfx_k)) * F128)]
    for name, b in sfx:
        yield a(name, b)
    yield a("d_shat_partials", 128 * 1024 * F128)
    max_n = max([1] + [1 << f for f in s.extra_free])
    yield a("d_e", max_n * F128)
    yield a("d_wx", max_n * F128)
    yield a("d_w", 256 * F128)
    yield a("dcb", W)
    for name, _ in sfx:
        yield r(name)
    for name in ("d_e", "d_wx", "d_w", "d_shat", "d_shat_partials"):
        yield r(name)
    sm = [("p0", BLOCKS * F128), ("p2", BLOCKS * F128), ("du0", F128), ("du2", F128)]
    for name, b in sm:
        yield a(name, b)
    for name, _ in sm:
        yield r(name)
    yield from ligerito_events(s, n, w + 4 if s.zk else w)
    if not keep:
        for name in l0:
            yield r(name)
    if keep and s.keep_ab:
        yield r("d_a")
        yield r("d_b")
        yield r("z_copy")
    yield r("df")
    yield r("dcb")
    if not keep:
        yield ("e",)


def ligerito_events(s: Shape, l0_bl: int, l0_lanes: int) -> Iterator[tuple]:
    """run_ligerito_f256's arena calls (ligerito_f256.cuh, with ligerito_zk.cuh's lane phase under M1)."""
    log_n = s.m - 7
    ln = 1 << log_n
    ik, rates, ks, _ = s.lig()
    queries = s.queries
    steps = len(ks)
    a = lambda name, n: ("a", name, n)  # noqa: E731
    r = lambda name: ("r", name)  # noqa: E731
    scratch = [("d_p256_0", BLOCKS * F256), ("d_p256_2", BLOCKS * F256), ("d_out256", 2 * F256), ("d_p0", BLOCKS * F128),
               ("d_p2", BLOCKS * F128), ("d_podd", BLOCKS * F128), ("d_u0", F128), ("d_u2", F128), ("d_hnew", F128)]
    for name, b in scratch:
        yield a(name, b)
    eq_k = log_n // 2
    yield a("d_eq_lo", (1 << eq_k) * F128)
    yield a("d_eq_hi", (1 << (log_n - eq_k)) * F128)
    yield a("region2", ln * F128)
    yield a("region3", ln * F128)
    n1 = log_n - ik
    levels, d = [], n1 + 1
    for lvl in range(1, steps + 1):
        log_cols = d - ks[lvl - 1]
        bl = 1 << (log_cols + rates[lvl])
        levels.append((bl, 1 << ks[lvl - 1]))
        d = log_cols + 1
    # the first fold's copies of the levels' hm96 salts (cuda_sha512_patch.py's copy_salts_ahead)
    for lvl, (bl, _) in enumerate(levels, 1):
        yield a(f"salts{lvl}", bl * SALT)
    # the first fold adds level 0's last OOD glue over the eq tables (ligerito_folds.cuh), then releases them
    yield r("d_eq_hi")
    yield r("d_eq_lo")
    if s.zk:
        yield a("d_r", max(ik, 1) * F256)
        yield a("d_ybar", max(s.t_pad, 1) * F256)
        yield r("d_r")
        yield r("d_ybar")
    max_block = max([l0_bl] + [bl for bl, _ in levels])
    max_q = max(queries)
    max_al = max(1 << (q - 1).bit_length() for q in queries)
    yield a("d_ind_basis", max_block * F128)
    yield a("d_ind_alpha", max_al * F128)
    yield a("d_ind_q", max_q * 8)
    yield a("d_eq2", (1 << (n1 + 1)) * F128)

    def commit(lvl: int) -> Iterator[tuple]:
        bl, lanes = levels[lvl - 1]
        yield a(f"cw{lvl}", bl * lanes * F128)
        yield a(f"tree{lvl}", (2 * bl - 1) * HB)
        yield r(f"salts{lvl}")

    def query(lvl: int) -> Iterator[tuple]:
        lanes = l0_lanes if lvl == 0 else levels[lvl - 1][1]
        yield a(f"rows{lvl}", queries[lvl] * lanes * F128)
        yield r(f"rows{lvl}")

    yield from commit(1)
    yield from query(0)
    for i in range(steps):
        if i + 1 == steps:
            yield from query(i + 1)
            yield r(f"cw{i + 1}")
            yield r(f"tree{i + 1}")
            break
        yield from commit(i + 2)
        yield from query(i + 1)
        yield r(f"cw{i + 1}")
        yield r(f"tree{i + 1}")
    for name in ("d_eq2", "d_ind_basis", "d_ind_alpha", "d_ind_q", "region2", "region3"):
        yield r(name)
    for name, _ in scratch:
        yield r(name)


def session_events(s: Shape, keep: bool) -> list[tuple]:
    if keep:
        return list(rep_events(s, "keep")) + list(rep_events(s, "take"))
    return list(rep_events(s, "plain")) * 2


class Arena:
    """DeviceArena (prove_ffi.cu, with cuda_chunk_patch.py's placement)."""

    def __init__(self, capacity: int, small_top: bool = True):
        self.capacity, self.small_top = capacity, small_top
        self.offsets: list[int] = []
        self.sizes: dict[int, int] = {}
        self.live: dict[str, tuple[int, int]] = {}

    def begin(self) -> None:
        assert not self.live, f"begin with {sorted(self.live)} live"
        self.offsets, self.sizes = [0], {0: self.capacity}

    def allocate(self, name: str, nbytes: int) -> int | None:
        assert nbytes > 0 and name not in self.live, name
        aligned = (nbytes + 255) & ~255
        top = self.small_top and aligned < SMALL
        chosen = None
        for off in self.offsets:
            if self.sizes[off] < aligned:
                continue
            chosen = off
            if not top:
                break
        if chosen is None:
            return None
        size = self.sizes.pop(chosen)
        self.offsets.remove(chosen)
        remaining = size - aligned
        offset = chosen + remaining if top else chosen
        if remaining:
            rest = chosen if top else chosen + aligned
            bisect.insort(self.offsets, rest)
            self.sizes[rest] = remaining
        self.live[name] = (offset, aligned)
        return offset

    def release(self, name: str) -> None:
        offset, nbytes = self.live.pop(name)
        i = bisect.bisect_left(self.offsets, offset)
        if i > 0 and self.offsets[i - 1] + self.sizes[self.offsets[i - 1]] == offset:
            prev = self.offsets.pop(i - 1)
            offset, nbytes = prev, nbytes + self.sizes.pop(prev)
            i -= 1
        if i < len(self.offsets) and offset + nbytes == self.offsets[i]:
            nxt = self.offsets.pop(i)
            nbytes += self.sizes.pop(nxt)
        bisect.insort(self.offsets, offset)
        self.sizes[offset] = nbytes


@dataclass
class Replay:
    fits: bool
    #: the highest end of any block, and the most bytes live at once
    high: int = 0
    live_peak: int = 0
    failed: str = ""
    layout: list = field(default_factory=list)


def replay(events: list[tuple], capacity: int, record: bool = False, small_top: bool = True) -> Replay:
    ar = Arena(capacity, small_top)
    out = Replay(True)
    live = 0
    for ev in events:
        if ev[0] == "b":
            ar.begin()
        elif ev[0] == "a":
            off = ar.allocate(ev[1], ev[2])
            if off is None:
                return Replay(False, out.high, out.live_peak, ev[1], out.layout)
            live += ar.live[ev[1]][1]
            out.high = max(out.high, off + ar.live[ev[1]][1])
            out.live_peak = max(out.live_peak, live)
            if record:
                out.layout.append((ev[1], off, ar.live[ev[1]][1]))
        elif ev[0] == "r":
            live -= ar.live[ev[1]][1]
            ar.release(ev[1])
    return out


def need(events: list[tuple], hi: int) -> int:
    """The smallest arena, in whole MiB, the events fit in (hi MiB must fit)."""
    assert replay(events, hi * MIB).fits, f"{hi} MiB doesn't fit"
    lo = max(1, replay(events, hi * MIB).live_peak // MIB)
    if replay(events, lo * MIB).fits:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if replay(events, mid * MIB).fits:
            hi = mid
        else:
            lo = mid
    return hi


def shapes(ms=range(21, 38), k_logs=range(20, 28)) -> Iterator[Shape]:
    """Every shape the prover accepts in the grid (fc_bad_shape: 20 <= k_log <= 27; the lincheck's eight-row stripes: m - k_log
    >= 3), under M1 and M0."""
    for m in ms:
        for k_log in k_logs:
            if m - k_log < 3:
                continue
            for zk in (True, False):
                yield Shape(m, k_log, zk)


def worst(s: Shape) -> list[Shape]:
    """s and its statement-dependent extremes: twelve slot types of 2^(k_log - 4) columns and one of K (fc_fold's folds), an
    extra claim of the whole witness (the ring switch's gather), level 0's padding at its cap (t_pad = L, so L/2 queries)."""
    log_l = s.m - 7 - s.lig()[0]
    return [s, replace(s, type_logs=(s.k_log - 4,) * 11 + (s.k_log,)), replace(s, extra_free=(s.m - 7,)),
            replace(s, t_pad=1 << log_l)]
