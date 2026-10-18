"""KV block lifetimes in vLLM v1's block pool, and what they mean for crediting live KV under a lifetime cap.

`Recorder` wraps one engine's `BlockPool` (instance attributes, so nothing else changes) and logs each event as
(time, kind, block id):

  ALLOC  `get_new_blocks` hands the block to a request: a new content epoch (cached content it held is evicted)
  HIT    `touch` of a block no request holds: a prefix-cache hit on resident content
  SHARE  `touch` of a block a request holds
  FREE   the last holder releases a block without a hash: its content is dead
  CACHE  the last holder releases a hashed block: its content stays resident for later hits until evicted
  RESET  `reset_prefix_cache` (block -1): every cached block's content is dead

A block's content epoch runs from its ALLOC to its eviction (the next ALLOC of the same block, a RESET, or its FREE when
unhashed); its held segments are the spans some request holds it (ALLOC or HIT to the FREE or CACHE that drops its
count to zero). Under memory-map/v1 a pass's plan is the blocks its step reads, so only held blocks are ever in a
plan: a cached block no request holds is resident but credited nothing.

`analyze` turns the events into the distributions the memory-accounting frontier needs, for a block of `block_bytes`:

- lifetimes: held life (ALLOC to the last release) and resident life (ALLOC to eviction), weighted by bytes (every
  block is `block_bytes`; a request's partly filled last block counts whole);
- stock ages: the share of held byte-seconds older than x (content age, from ALLOC);
- occupancy: held and cached blocks on a grid;
- for each window T and lifetime cap L (an appended block joins its window and no plan L or more windows later may
  hold it, PROTOCOL.md §4) and each policy, the share of held byte-seconds credited and the encode bytes it costs:
    rekey    every resident block is re-keyed (re-encoded under a fresh coin) at each cap it outlives, cached ones
             included, so every held byte-second is credited
    held     a block is re-keyed at a cap only if a request holds it then, and released otherwise (later hits
             uncredited)
    expire   no re-key: a block is credited until its first cap and uncredited after
  each averaged over `phases` window phases. The append of every written block (encode-on-write) is reported
  beside them, and `delayed` gives the encode bytes and credit lost when a block is appended only once it is `tau`
  seconds old.
- for a carrier stock C (bytes), the per-step credited share max(0, held_s - C) / held_s, time-weighted over steps.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np

ALLOC, HIT, SHARE, FREE, CACHE, RESET = range(6)
KINDS = ("alloc", "hit", "share", "free", "cache", "reset")
STEP_FIELDS = ("t0", "t1", "T", "X", "Q", "K", "n", "n0", "running", "waiting", "free_blocks")


class Recorder:
    """Events of one BlockPool, on `clock` (seconds)."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self.t: list[float] = []
        self.k: list[int] = []
        self.b: list[int] = []

    def log(self, kind: int, blocks: Iterable[int]) -> None:
        now = self.clock()
        for bid in blocks:
            self.t.append(now)
            self.k.append(kind)
            self.b.append(bid)

    def install(self, pool) -> Callable[[], None]:
        """Wrap `pool`'s get_new_blocks, touch, free_blocks and reset_prefix_cache; returns the uninstaller."""
        get0, touch0, free0, reset0 = pool.get_new_blocks, pool.touch, pool.free_blocks, pool.reset_prefix_cache

        def get_new_blocks(num_blocks):
            out = get0(num_blocks)
            self.log(ALLOC, (blk.block_id for blk in out))
            return out

        def touch(blocks):
            blocks = list(blocks)
            hits = [blk.block_id for blk in blocks if not blk.is_null and blk.ref_cnt == 0]
            shares = [blk.block_id for blk in blocks if not blk.is_null and blk.ref_cnt > 0]
            out = touch0(blocks)
            self.log(HIT, hits)
            self.log(SHARE, shares)
            return out

        def free_blocks(ordered_blocks):
            blocks = list(ordered_blocks)
            out = free0(blocks)
            done = [blk for blk in blocks if not blk.is_null and blk.ref_cnt == 0]
            self.log(FREE, (blk.block_id for blk in done if blk.block_hash is None))
            self.log(CACHE, (blk.block_id for blk in done if blk.block_hash is not None))
            return out

        def reset_prefix_cache(*a, **kw):
            out = reset0(*a, **kw)
            if out:
                self.log(RESET, (-1,))
            return out

        pool.get_new_blocks, pool.touch, pool.free_blocks = get_new_blocks, touch, free_blocks
        pool.reset_prefix_cache = reset_prefix_cache

        def uninstall() -> None:
            for name in ("get_new_blocks", "touch", "free_blocks", "reset_prefix_cache"):
                vars(pool).pop(name, None)

        return uninstall

    def arrays(self) -> dict[str, np.ndarray]:
        return {"t": np.asarray(self.t, dtype=np.float64), "kind": np.asarray(self.k, dtype=np.int8),
                "block": np.asarray(self.b, dtype=np.int32)}


def parse(t: np.ndarray, kind: np.ndarray, block: np.ndarray, t_end: float) -> dict[str, np.ndarray]:
    """Content epochs, held segments and cached spans from time-ordered events. Spans open at `t_end` close there and
    are flagged censored."""
    order = np.argsort(block, kind="stable")
    ep_a, ep_res, ep_held, ep_cens, ep_hits = [], [], [], [], []
    sg_s, sg_e, sg_ep = [], [], []
    ca_s, ca_e = [], []
    resets = t[kind == RESET]
    # per block: state 0 empty, 1 held, 2 cached; the open epoch's index, the open segment's start, the cached start
    tb, kb, bb = t[order], kind[order], block[order]
    i, n = 0, len(tb)
    while i < n:
        bid = bb[i]
        j = i
        while j < n and bb[j] == bid:
            j += 1
        if bid < 0:
            i = j
            continue
        state, ep, seg0, c0 = 0, -1, 0.0, 0.0
        r = 0  # next reset to check against a cached span
        for x in range(i, j):
            tx, kx = tb[x], kb[x]
            if state == 2:
                while r < len(resets) and resets[r] < c0:
                    r += 1
                if r < len(resets) and resets[r] <= tx:
                    ca_s.append(c0), ca_e.append(resets[r])
                    ep_res[ep] = resets[r]
                    state = 0
            if kx == ALLOC:
                if state == 1:
                    raise ValueError(f"block {bid} allocated at {tx} while held")
                if state == 2:
                    ca_s.append(c0), ca_e.append(tx)
                    ep_res[ep] = tx
                ep = len(ep_a)
                ep_a.append(tx), ep_res.append(t_end), ep_held.append(t_end), ep_cens.append(True), ep_hits.append(0)
                state, seg0 = 1, tx
            elif kx == HIT:
                if state != 2:
                    raise ValueError(f"block {bid} hit at {tx} while not cached")
                ca_s.append(c0), ca_e.append(tx)
                ep_hits[ep] += 1
                state, seg0 = 1, tx
            elif kx in (FREE, CACHE):
                if state != 1:
                    raise ValueError(f"block {bid} released at {tx} while not held")
                sg_s.append(seg0), sg_e.append(tx), sg_ep.append(ep)
                ep_held[ep] = tx
                ep_cens[ep] = False
                if kx == FREE:
                    ep_res[ep] = tx
                    state = 0
                else:
                    state, c0 = 2, tx
        if state == 1:
            sg_s.append(seg0), sg_e.append(t_end), sg_ep.append(ep)
        elif state == 2:
            while r < len(resets) and resets[r] < c0:
                r += 1
            end = resets[r] if r < len(resets) else t_end
            ca_s.append(c0), ca_e.append(end)
            ep_res[ep] = end
        i = j
    f8, i8 = np.float64, np.int64
    return {"a": np.asarray(ep_a, f8), "res_end": np.asarray(ep_res, f8), "held_end": np.asarray(ep_held, f8),
            "censored": np.asarray(ep_cens, bool), "hits": np.asarray(ep_hits, i8),
            "seg_start": np.asarray(sg_s, f8), "seg_end": np.asarray(sg_e, f8), "seg_epoch": np.asarray(sg_ep, i8),
            "cached_start": np.asarray(ca_s, f8), "cached_end": np.asarray(ca_e, f8)}


def count_at(starts: np.ndarray, ends: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """How many of the spans [start, end) contain each grid point."""
    return (np.searchsorted(np.sort(starts), grid, side="right")
            - np.searchsorted(np.sort(ends), grid, side="right")).astype(np.int64)


def weighted_quantiles(x: np.ndarray, w: np.ndarray, qs: Sequence[float]) -> list[float]:
    if len(x) == 0:
        return [float("nan")] * len(qs)
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return [float(x[o][min(len(x) - 1, int(np.searchsorted(c, q * c[-1])))]) for q in qs]


def age_tail(seg_start: np.ndarray, seg_end: np.ndarray, origin: np.ndarray, x: float) -> float:
    """The share of the segments' byte-seconds older than `x` since their origin."""
    tot = float(np.sum(seg_end - seg_start))
    if tot <= 0:
        return 0.0
    return float(np.sum(np.maximum(0.0, seg_end - np.maximum(seg_start, origin + x)))) / tot


def caps(a: np.ndarray, T: float, L: int, phase: float) -> np.ndarray:
    """The first cap of a block appended at `a`: the start of window w0 + L, w0 the window holding `a`, windows
    starting at phase + jT."""
    w0 = np.floor((a - phase) / T)
    return phase + (w0 + L) * T


def policy(ep: dict[str, np.ndarray], T: float, L: int, phase: float, name: str) -> dict[str, np.ndarray]:
    """For one phase: each epoch's re-keys and each held segment's credited span [start, cut) under `name`."""
    c1 = caps(ep["a"], T, L, phase)
    life = L * T
    s, e, k = ep["seg_start"], ep["seg_end"], ep["seg_epoch"]
    if name == "expire":
        return {"rekeys": np.zeros(len(ep["a"]), np.int64), "cut": np.minimum(e, np.maximum(s, c1[k]))}
    if name == "rekey":
        n = np.where(ep["res_end"] > c1, np.floor((ep["res_end"] - c1) / life) + 1, 0).astype(np.int64)
        return {"rekeys": n, "cut": e.copy()}
    if name != "held":
        raise ValueError(name)
    # re-keyed at a cap a request holds it through; released at the first cap it is idle (cached), its later
    # segments uncredited. Segments keyed (epoch, start in ms) to find the one a cap falls in.
    order = np.lexsort((s, k))
    key = (k[order] << 32) + np.round(s[order] * 1e3).astype(np.int64)
    kk, ke = k[order], e[order]
    rekeys = np.zeros(len(ep["a"]), np.int64)
    release = ep["res_end"].copy()
    cap = c1.copy()
    active = cap < ep["res_end"]
    while active.any():
        idx = np.flatnonzero(active)
        q = (idx.astype(np.int64) << 32) + np.round(cap[idx] * 1e3).astype(np.int64)
        pos = np.clip(np.searchsorted(key, q, side="right") - 1, 0, None)
        inside = (kk[pos] == idx) & (ke[pos] > cap[idx]) if len(kk) else np.zeros(len(idx), bool)
        rekeys[idx[inside]] += 1
        release[idx[~inside]] = cap[idx[~inside]]
        active[idx[~inside]] = False
        cap[idx[inside]] += life
        active[idx[inside]] = cap[idx[inside]] < ep["res_end"][idx[inside]]
    return {"rekeys": rekeys, "cut": np.minimum(e, np.maximum(s, release[k]))}


def step_credit(steps: dict[str, np.ndarray], seg_start: np.ndarray, cut: np.ndarray, held_start: np.ndarray,
                held_end: np.ndarray, block_bytes: float, stocks: Sequence[float]) -> dict:
    """Each step's held bytes and credited bytes (the credited spans at the step's midpoint), and for each carrier
    stock C the time-weighted credited share max(0, credited_s - C) of held_s."""
    mid = (steps["t0"] + steps["t1"]) / 2
    dur = np.maximum(0.0, steps["t1"] - steps["t0"])
    held = count_at(held_start, held_end, mid) * block_bytes
    cred = count_at(seg_start, cut, mid) * block_bytes
    tot = float(np.sum(dur * held))
    out = {"held_bytes_mean": tot / max(1e-12, float(np.sum(dur))),
           "credited_before_stock": float(np.sum(dur * cred)) / max(1e-12, tot), "by_stock": {}}
    for C in stocks:
        out["by_stock"][repr(float(C))] = float(np.sum(dur * np.maximum(0.0, cred - C))) / max(1e-12, tot)
    return out


def analyze(ev: dict[str, np.ndarray], steps: dict[str, np.ndarray], t_end: float, block_bytes: float, *,
            t_start: float = 0.0, windows: Sequence[float] = (20.0, 60.0), caps_L: Sequence[int] = (1, 2, 3, 5, 10, 60),
            phases: int = 8, taus: Sequence[float] = (0.0, 1.0, 5.0, 10.0, 30.0, 60.0),
            stocks: Sequence[float] = (), grid_s: float = 1.0) -> dict:
    """The report for events and steps on one clock, over [t_start, t_end): epochs that start before t_start are
    dropped from the lifetime distributions, and credited shares are over steps inside the interval."""
    ep = parse(ev["t"], ev["kind"], ev["block"], t_end)
    keep = ep["a"] >= t_start
    a, he, re, cens = ep["a"][keep], ep["held_end"][keep], ep["res_end"][keep], ep["censored"][keep]
    held_life, res_life = he - a, re - a
    w = np.full(len(a), float(block_bytes))
    span = max(1e-12, t_end - t_start)
    points = (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180, 300, 600, 1200)
    qs = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    origin = ep["a"][ep["seg_epoch"]]
    insteps = (steps["t0"] >= t_start) & (steps["t1"] <= t_end)
    st = {k: v[insteps] for k, v in steps.items()}
    grid = np.arange(t_start, t_end, grid_s)
    out = {
        "block_bytes": block_bytes, "interval_s": [t_start, t_end], "epochs": int(len(a)),
        "censored_share": float(np.mean(cens)) if len(a) else 0.0, "hits": int(ep["hits"][keep].sum()),
        "written_bytes_per_s": len(a) * block_bytes / span,
        "held_life_s": {"quantiles_byte_weighted": dict(zip(map(str, qs), weighted_quantiles(held_life, w, qs))),
                        "share_longer_than": {str(p): float(np.mean(held_life > p)) if len(a) else 0.0
                                              for p in points},
                        "mean": float(np.mean(held_life)) if len(a) else 0.0},
        "resident_life_s": {"quantiles_byte_weighted": dict(zip(map(str, qs), weighted_quantiles(res_life, w, qs))),
                            "share_longer_than": {str(p): float(np.mean(res_life > p)) if len(a) else 0.0
                                                  for p in points}},
        "stock_age_tail": {str(p): age_tail(ep["seg_start"], ep["seg_end"], origin, p) for p in points},
        "occupancy": {"grid_s": grid_s, "t0": t_start,
                      "held_blocks": count_at(ep["seg_start"], ep["seg_end"], grid).tolist(),
                      "cached_blocks": count_at(ep["cached_start"], ep["cached_end"], grid).tolist()},
        "frontier": {}, "delayed": {},
    }
    hb = out["occupancy"]["held_blocks"]
    out["occupancy"]["held_bytes_mean"] = float(np.mean(hb)) * block_bytes if hb else 0.0
    out["occupancy"]["held_bytes_p10_p50_p90"] = [float(np.percentile(hb, q)) * block_bytes for q in (10, 50, 90)] if hb else []
    for T in windows:
        for L in caps_L:
            row = {}
            for name in ("rekey", "held", "expire"):
                rk, cred = [], []
                for ph in range(phases):
                    p = policy(ep, T, L, T * ph / phases, name)
                    rk.append(float(p["rekeys"][keep].sum()))
                    cred.append(step_credit(st, ep["seg_start"], p["cut"], ep["seg_start"], ep["seg_end"],
                                            block_bytes, stocks))
                row[name] = {"rekey_bytes_per_s": float(np.mean(rk)) * block_bytes / span,
                             "credited_before_stock": float(np.mean([c["credited_before_stock"] for c in cred])),
                             "by_stock": {C: float(np.mean([c["by_stock"][C] for c in cred]))
                                          for C in cred[0]["by_stock"]},
                             "held_bytes_mean": cred[0]["held_bytes_mean"]}
            out["frontier"][f"T={T:g},L={L}"] = row
    seg_ep = ep["seg_epoch"]
    for tau in taus:
        appended = (he - a) > tau
        cut = np.maximum(ep["seg_start"], ep["a"][seg_ep] + tau)
        lost = float(np.sum(np.minimum(ep["seg_end"], cut) - ep["seg_start"])) / max(
            1e-12, float(np.sum(ep["seg_end"] - ep["seg_start"])))
        out["delayed"][str(tau)] = {"append_bytes_per_s": float(appended.sum()) * block_bytes / span,
                                    "credit_lost": lost}
    return out


def steps_arrays(rows: Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, len(STEP_FIELDS))
    return {k: arr[:, i] for i, k in enumerate(STEP_FIELDS)}
