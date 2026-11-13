"""How often a request meets `halves_split`: the case `sampled_accept.sampled_accept` reports when lane 0 of
`_rejection_kernel`'s warp accepts a draft token that is not the last and lanes 16-31 refuse it.  CPU only.

For a request's draft position i the reference keeps both halves' log-sum-exp words (`row_lse`: lane 0's and lane 16's).
With x the logit at the draft token, ``lp = x - lse`` and ``lp_hi = x - lse_hi`` in fp32, and the position splits when
``lp > log(u)`` and not ``lp_hi > log(u)``, u the uniform of the row's Philox word (`accept_uniform`).

The table.  `accept_uniform` reads its 32-bit word w only through ``mag = w`` for w < 2^31 and ``mag = 2^32 - 1 - w``
otherwise, so every mag in [0, 2^31) is the image of exactly two words.  `build_table` evaluates u for every one of the 2^31
mags (numpy: int64 to f32 is `cvt.rn.f32.s32`, an f32 product is `mul.f32`, then the floor), collects the distinct u words
with how many mags give each, takes libdevice's log of each distinct u (`sampling_rows.nv_logf`, which is `nv_logf_gpu(..,
ftz=False)` on these words: the two differ only below FLT_MIN and on NaN), and sorts the distinct log words by value with
their word counts.  ``count_below(lp)`` is then the number of the 2^32 words whose log(u) is below lp, counted and not
derived from any ordering of log(u) in the word; whether log(u) is non-decreasing in u is swept and reported besides.

A position's numbers, exact given its row:

    p      = count_below(lp) / 2^32                       lane 0 accepts
    q      = max(count_below(lp) - count_below(lp_hi), 0) / 2^32      lane 0 accepts and lane 16 refuses (the split)
    q_rev  = max(count_below(lp_hi) - count_below(lp), 0) / 2^32      lane 0 refuses and lane 16 accepts

A request's numbers, its rows fixed and its K uniforms' words independent and uniform on 2^32 (an assumption on Philox):

    w_i    = prod_{j<i} p_j                               position i is reached under lane 0's verdicts
    P      = sum_{i<K-1} (prod_{j<i} min(p_j, p_hi_j)) q_i            the request reports `halves_split`
    E      = sum_{i<K-1} w_i q_i                          the expected number of its positions that split

and, under the request's own seed (the Philox words the reference computes): which positions are reached under lane 0's
verdicts, whether each splits, and what `sampled_accept` reports.  `halves_split` is taken from the reference itself, run
with its resample stubbed out (the resampled token is not read here); the first requests of each kind are also run with
the resample and compared.

Two kinds of rows, each request a function of (seed, its number) alone:

    random   `tests.program.sampled_accept_vectors.random_rows` as the tests have it.
    peaked   `peaked_rows` below: the draft token carries most of its row's mass.

    python scratch/cir264/halves_rate.py --out DIR [--minutes M] [--workers N] [--block B] [--seed S]
    python scratch/cir264/halves_rate.py --report DIR/halves_rate.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "adapters" / "vllm", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SV = importlib.import_module("tests.program.sampled_accept_vectors")
A = importlib.import_module("verity.definitions.registry.sampled_accept")
SA = importlib.import_module("verity.definitions.registry.sampling")
SR = importlib.import_module("verity.definitions.registry.sampling_rows")

F = np.float32
V, K, TILE = 154880, 5, A.STATS_TILE
WORDS = 1 << 32
#: `peaked_rows`: the nucleus sizes, the standard deviations of the other tokens' logits, the range of 1 - p (p the draft
#: token's probability under its row; log-uniform) and the range of the draft token's logit (uniform)
PEAKED_NUCLEI = (2, 3, 5, 10, 30, 100, 300, 1000, 3000)
PEAKED_SPREADS = (1.0, 3.0)
PEAKED_MISS = (1e-3, 0.5)
PEAKED_LEVEL = (5.0, 25.0)
#: the classes a position's probability of lane 0 accepting falls in
P_BINS = (0.5, 0.9, 0.99)
P_NAMES = ("p < 0.5", "0.5 <= p < 0.9", "0.9 <= p < 0.99", "p >= 0.99")

TABLE: dict = {}


def peaked_rows(Kn: int, Vn: int, count: int, seed: int, tile: int = TILE, first: int = 0) -> list[dict]:
    """Requests ``first`` to ``first + count - 1`` of ``seed``'s, in `sampled_accept_vectors.vectors`' form, each a function
    of (seed, its number) alone.  A request has one nucleus size n of `PEAKED_NUCLEI` and one spread of `PEAKED_SPREADS`.
    Each of its K + 1 rows is -inf but on n tokens chosen uniformly without replacement from the vocabulary; the first of
    them is the draft token, at a logit uniform in `PEAKED_LEVEL`; the other n - 1 are normal at the spread, shifted so that
    together they hold 1 - p of the row's mass, 1 - p log-uniform in `PEAKED_MISS` (computed in float64, stored as f32)."""
    out = []
    for c in range(first, first + count):
        rng = np.random.default_rng((seed, c))
        n = int(rng.choice(PEAKED_NUCLEI))
        spread = float(rng.choice(PEAKED_SPREADS))
        logits = np.full((Kn + 1, Vn), -np.inf, F)
        draft = []
        for i in range(Kn + 1):
            idx = rng.choice(Vn, n, replace=False)
            tail = rng.normal(0, spread, n - 1)
            miss = float(np.exp(rng.uniform(np.log(PEAKED_MISS[0]), np.log(PEAKED_MISS[1]))))
            level = float(rng.uniform(*PEAKED_LEVEL))
            top = tail.max()
            tail_lse = top + np.log(np.exp(tail - top).sum())
            logits[i, idx[0]] = level
            logits[i, idx[1:]] = (level + np.log(miss / (1 - miss)) + tail - tail_lse).astype(F)
            if i < Kn:
                draft.append(int(idx[0]))
        s, pos = int(rng.integers(1 << 62)), [int(p) for p in 1000 + 10 * c + np.arange(Kn + 1)]
        out.append({"name": f"peaked-{seed}-{c}", "logits": logits, "draft": draft, "seed": s, "pos": pos, "temp": SV.TEMP,
                    "ref": A.sampled_accept(logits, draft, s, pos, SV.TEMP, tile)})
    return out


KINDS = {"random": lambda seed, c: SV.random_rows(K, V, 1, seed, TILE, c)[0],
         "peaked": lambda seed, c: peaked_rows(K, V, 1, seed, TILE, c)[0]}


# ---------------------------------------------------------------------------------------------------------
# the table of log(u) over the 2^32 words
# ---------------------------------------------------------------------------------------------------------


def uniform_words(mag: np.ndarray) -> np.ndarray:
    """`accept_uniform`'s fp32 word for each magnitude (int64, in [0, 2^31))."""
    scale, floor = np.uint32(SA.TL_RAND_SCALE_BITS).view(F), np.uint32(SA.TL_RAND_MIN_BITS).view(F)
    u = (mag.astype(F) * scale).astype(F)
    return np.where(u > floor, u, floor).astype(F).view(np.uint32)


def mag_of_word(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, np.int64)
    return np.where(w < (1 << 31), w, WORDS - 1 - w)


def _chunk(a: int) -> tuple[np.ndarray, np.ndarray]:
    u = uniform_words(np.arange(a, a + (1 << 24), dtype=np.int64))
    return np.unique(u, return_counts=True)


def _log_chunk(u: np.ndarray) -> np.ndarray:
    return SR.nv_logf(u)


def _merge(vals: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal values of a sorted array merged, their counts added."""
    starts = np.concatenate(([0], np.flatnonzero(vals[1:] != vals[:-1]) + 1))
    return vals[starts], np.add.reduceat(counts, starts)


def finish_table(u: np.ndarray, mags: np.ndarray, logs: np.ndarray) -> dict:
    """The table from the distinct u words in ascending order, the magnitudes that give each and each one's log word."""
    lf = logs.view(F)
    assert not np.isnan(lf).any() and np.isfinite(lf).all() and (lf < 0).all()
    descents = np.flatnonzero(lf[1:] < lf[:-1])
    order = np.argsort(lf, kind="stable")
    lsorted, lmags = _merge(lf[order], mags[order])
    cum = np.concatenate(([0], np.cumsum(2 * lmags))).astype(np.int64)
    assert int(cum[-1]) == 2 * int(mags.sum())
    return {"u": u, "log_of_u": logs, "log_sorted": lsorted, "cum": cum,
            "facts": {"words": int(cum[-1]), "magnitudes": int(mags.sum()), "distinct_u": int(u.size),
                      "distinct_log_u": int(lsorted.size), "u_min": int(u[0]), "u_max": int(u[-1]),
                      "log_u_min": int(lsorted[:1].view(np.uint32)[0]), "log_u_max": int(lsorted[-1:].view(np.uint32)[0]),
                      "log_u_descents_in_u": int(descents.size),
                      "first_descents": [[int(u[i]), int(logs[i]), int(u[i + 1]), int(logs[i + 1])] for i in descents[:8]]}}


def build_table(workers: int) -> dict:
    with mp.Pool(workers) as pool:
        parts = pool.map(_chunk, range(0, 1 << 31, 1 << 24))
        vals = np.concatenate([p[0] for p in parts])
        counts = np.concatenate([p[1] for p in parts]).astype(np.int64)
        del parts
        order = np.argsort(vals, kind="stable")
        u, mags = _merge(vals[order], counts[order])
        del vals, counts, order
        assert int(mags.sum()) == 1 << 31, int(mags.sum())
        step = 1 << 20
        logs = np.concatenate(pool.map(_log_chunk, [u[i:i + step] for i in range(0, u.size, step)]))
    return finish_table(u, mags, logs)


def count_below(lp) -> int:
    """How many of the 2^32 words give log(u) < lp (an f32; NaN compares false)."""
    lp = F(lp)
    if lp != lp:
        return 0
    return int(TABLE["cum"][np.searchsorted(TABLE["log_sorted"], lp, side="left")])


def table_log(word: int) -> tuple[int, int]:
    """(u, log u) words of a Philox word, read from the table."""
    u = uniform_words(mag_of_word(np.array([word])))[0]
    j = int(np.searchsorted(TABLE["u"], u))
    if TABLE.get("partial") and (j == TABLE["u"].size or TABLE["u"][j] != u):
        return int(u), SA.nv_logf_gpu(int(u), ftz=False)
    assert TABLE["u"][j] == u
    return int(u), int(TABLE["log_of_u"][j])


def check_table(samples: int, seed: int) -> dict:
    """The table against the scalar reference (`accept_uniform`, `nv_logf_gpu(.., ftz=False)`) on edge words and random ones,
    and `count_below` against a count over the same random words' logs."""
    rng = np.random.default_rng((seed, 0xC0FFEE))
    edges = [0, 1, 2, 3, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 31) - 65, (1 << 31) - 64, (1 << 31) - 1, 1 << 31,
             (1 << 31) + 1, WORDS - 3, WORDS - 2, WORDS - 1]
    words = edges + [int(w) for w in rng.integers(0, WORDS, samples)]
    logs = []
    for w in words:
        u = A.accept_uniform(w)
        log_u = SA.nv_logf_gpu(u, ftz=False)
        assert table_log(w) == (u, log_u), (w, u, log_u, table_log(w))
        logs.append(log_u)
    lf = np.array(logs[len(edges):], np.uint32).view(F)
    worst = 0.0
    for lp in (-21.0, -10.0, -3.0, -1.0, -0.5, -0.1, -0.01, -1e-4):
        exact, seen = count_below(lp) / WORDS, float((lf < F(lp)).mean())
        sigma = max((exact * (1 - exact) / samples) ** 0.5, 1e-12)
        worst = max(worst, abs(seen - exact) / sigma)
    assert TABLE.get("partial") or worst < 6, worst
    return {"words_checked": len(words), "count_below_worst_sigma": round(worst, 2)}


# ---------------------------------------------------------------------------------------------------------
# a request
# ---------------------------------------------------------------------------------------------------------


def _w(x) -> int:
    return int(np.asarray(x, F).reshape(1).view(np.uint32)[0])


def request_numbers(kind: str, c: int, req: dict) -> tuple[dict, dict, list]:
    """One request's sums by class of position, its own sums, and a record of each draft row whose halves' sums differ."""
    ref, logits, draft = req["ref"], req["logits"], req["draft"]
    tiles = len(ref["max"][0])
    agg: dict = {}
    detail = []
    w = r = 1.0
    live = True
    P = E = own_q = 0.0
    own_first = None
    n0 = None
    any_differ = False
    for i, d in enumerate(draft):
        x = logits[i, d] if 0 <= d < V else F(-np.inf)
        lo, hi = ref["lse"][i], ref["lse_hi"][i]
        with np.errstate(invalid="ignore"):
            lp, lp_hi = (F(x) - np.array([lo, hi], np.uint32).view(F)).astype(F)
        assert _w(lp) == ref["logprob"][i], (kind, c, i)
        u, log_u = table_log(SA.stream_key(req["seed"], int(req["pos"][i])))
        assert (u, log_u) == (ref["u"][i], ref["log_u"][i]), (kind, c, i, u, log_u)
        lu = np.uint32(log_u).view(F)
        test, test_hi = bool(lp > lu), bool(lp_hi > lu)
        assert int(test) == ref["test"][i], (kind, c, i)
        c_lo, c_hi = count_below(lp), count_below(lp_hi)
        assert TABLE.get("partial") or test == (count_below(np.nextafter(lu, F(np.inf))) <= c_lo), (kind, c, i)
        p, q, q_rev, both = c_lo / WORDS, max(c_lo - c_hi, 0) / WORDS, max(c_hi - c_lo, 0) / WORDS, min(c_lo, c_hi) / WORDS
        differ = lo != hi
        any_differ = any_differ or differ
        m = np.asarray(ref["max"][i], F)
        full = m != -np.inf
        pairs = [t for t in range(tiles - 16) if full[t] and full[t + 16]]
        td = d // TILE if 0 <= d < V else -1
        in_pair = int(td in pairs or td - 16 in pairs)
        n = int(np.isfinite(logits[i]).sum())
        n0 = n if n0 is None else n0
        key = (n, len(pairs), in_pair, int(np.searchsorted(P_BINS, p, side="right")), int(m.max() >= 15.0))
        a = agg.setdefault(key, dict.fromkeys(FIELDS, 0.0))
        a["rows"] += 1
        a["rows_differ"] += differ
        elig = i < len(draft) - 1
        split, rev = test and not test_hi, test_hi and not test
        if elig:
            a["pos"] += 1
            a["pos_differ"] += differ
            a["w"] += w
            a["w_differ"] += w * differ
            a["wq"] += w * q
            a["wq_rev"] += w * q_rev
            a["rq"] += r * q
            P += r * q
            E += w * q
            if live:
                a["own_pos"] += 1
                a["own_differ"] += differ
                a["own_q"] += q
                a["own_q_rev"] += q_rev
                a["own_split"] += split
                a["own_rev"] += rev
                own_q += q
                if split and own_first is None:
                    own_first = i
        if differ:
            detail.append({"kind": kind, "request": c, "i": i, "n": n, "pairs": pairs, "draft_tile": td, "in_pair": in_pair,
                           "top": _w(m.max()), "lse": lo, "lse_hi": hi, "lp": _w(lp), "lp_hi": _w(lp_hi), "below_lp": c_lo,
                           "below_lp_hi": c_hi, "w": w, "r": r, "live": live, "test": test, "test_hi": test_hi})
        w, r, live = w * p, r * both, live and test
    assert ref["halves_split"] == own_first, (kind, c, ref["halves_split"], own_first)
    own = {"requests": 1, "P": P, "P2": P * P, "E": E, "E2": E * E, "own_q": own_q, "own_q2": own_q * own_q,
           "halves_split": int(ref["halves_split"] is not None), "any_differ": int(any_differ), "accepted": ref["accepted"]}
    return agg, {n0: own}, detail


FIELDS = ("rows", "rows_differ", "pos", "pos_differ", "w", "w_differ", "wq", "wq_rev", "rq", "own_pos", "own_differ", "own_q",
          "own_q_rev", "own_split", "own_rev")


def _add(into: dict, more: dict) -> None:
    for key, fields in more.items():
        t = into.setdefault(key, {})
        for name, v in fields.items():
            t[name] = t.get(name, 0) + v


def run_block(kind: str, seed: int, first: int, count: int) -> dict:
    agg: dict = {}
    reqs: dict = {}
    detail: list = []
    for c in range(first, first + count):
        a, o, d = request_numbers(kind, c, KINDS[kind](seed, c))
        _add(agg, a)
        _add(reqs, o)
        detail += d
    return {"agg": agg, "requests": reqs, "detail": detail}


def _worker(kind, wid, seed, block, counter, lock, deadline, out) -> None:
    try:
        SR.gumbel_select_row = lambda *a, **k: (-1, None)
        while True:
            with lock:
                if time.monotonic() >= deadline:
                    break
                b = counter.value
                counter.value += 1
            out.put(("block", kind, b, run_block(kind, seed, b * block, block)))
    except BaseException:
        out.put(("error", kind, wid, traceback.format_exc()))
    finally:
        out.put(("done", kind, wid, None))


def check_stub(kind: str, seed: int, count: int) -> int:
    """The first ``count`` requests with the reference's resample as it is and with it stubbed out: every other word equal."""
    whole = [KINDS[kind](seed, c) for c in range(count)]
    keep = SR.gumbel_select_row
    SR.gumbel_select_row = lambda *a, **k: (-1, None)
    try:
        for c, full in enumerate(whole):
            got = KINDS[kind](seed, c)
            assert np.array_equal(got["logits"].view(np.uint32), full["logits"].view(np.uint32)) and got["draft"] == full["draft"]
            assert (got["seed"], got["pos"]) == (full["seed"], full["pos"])
            for name, value in full["ref"].items():
                if name in ("token", "appended"):
                    continue
                same = (all(np.array_equal(x, y) for x, y in zip(value, got["ref"][name])) if name in ("max", "sumexp")
                        else value == got["ref"][name])
                assert same, (kind, c, name)
    finally:
        SR.gumbel_select_row = keep
    return count


# ---------------------------------------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------------------------------------


def _rate(num: float, den: float) -> str:
    return "-" if not den else f"{num / den:.3e}"


def _slice(agg: dict, at: int | None) -> dict:
    out: dict = {}
    for key, fields in agg.items():
        _add(out, {("all",) if at is None else (key[at],): fields})
    return out


def _lines(title: str, sliced: dict, names=None) -> list[str]:
    out = [f"  {title}",
           f"    {'class':<18}{'positions':>10}{'reached':>10}{'E reached':>12}{'differ':>9}{'share':>10}"
           f"{'rate/pos own':>14}{'rate/pos':>12}{'reverse':>12}{'split own':>10}"]
    for key in sorted(sliced):
        a = sliced[key]
        name = str(key[0]) if names is None or key[0] == "all" else names[key[0]]
        out.append(f"    {name:<18}{int(a['pos']):>10}{int(a['own_pos']):>10}{a['w']:>12.1f}{int(a['own_differ']):>9}"
                   f"{_rate(a['own_differ'], a['own_pos']):>10}{_rate(a['own_q'], a['own_pos']):>14}{_rate(a['wq'], a['w']):>12}"
                   f"{_rate(a['wq_rev'], a['w']):>12}{int(a['own_split']):>10}")
    return out


def report(record: dict) -> str:
    out = [f"table: {json.dumps(record['table'])}", f"checks: {json.dumps(record['checks'])}"]
    for kind, k in record["kinds"].items():
        agg = {tuple(json.loads(key)): v for key, v in k["agg"].items()}
        reqs = {int(key): v for key, v in k["requests"].items()}
        tot: dict = {}
        _add(tot, {"all": {name: sum(v[name] for v in reqs.values()) for name in next(iter(reqs.values()))}})
        t, N = tot["all"], tot["all"]["requests"]
        se = lambda s, s2: ((s2 / N - (s / N) ** 2) / N) ** 0.5 if N > 1 else float("nan")  # noqa: E731
        out += ["", f"{kind}: requests {k['first']} to {k['first'] + N - 1} of seed {k['seed']} ({N} requests, {k['blocks']} blocks of "
                    f"{k['block']})",
                f"  rate a request, exact (P): {t['P'] / N:.3e} +- {se(t['P'], t['P2']):.1e} (standard error over requests)",
                f"  expected split positions a request (E): {t['E'] / N:.3e} +- {se(t['E'], t['E2']):.1e}",
                f"  rate a request, the request's own reached positions: {t['own_q'] / N:.3e} +- {se(t['own_q'], t['own_q2']):.1e}",
                f"  direct count: {int(t['halves_split'])} of {N} requests report halves_split"
                + (f" (95% bound {3 / N:.2e} a request)" if not t["halves_split"] else "")
                + f"; {int(t['any_differ'])} requests have a draft row whose halves' sums differ"]
        allp = _slice(agg, None)[("all",)]
        out += [f"  draft rows {int(allp['rows'])}, of them {int(allp['rows_differ'])} with differing sums; positions i < K-1 "
                f"{int(allp['pos'])}, of them {int(allp['pos_differ'])} differing; reached under the request's own seed "
                f"{int(allp['own_pos'])}, of them {int(allp['own_differ'])} differing; own splits {int(allp['own_split'])}, own "
                f"reverse {int(allp['own_rev'])}"]
        out += _lines("all positions i < K-1", _slice(agg, None))
        out += _lines("by nucleus size (finite logits in the row)", _slice(agg, 0))
        out += _lines("by tile pairs (t, t+16) that both hold a finite logit", _slice(agg, 1))
        out += _lines("by whether the draft token's tile is in such a pair", _slice(agg, 2), ("no", "yes"))
        out += _lines("by lane 0's probability of accepting", _slice(agg, 3), P_NAMES)
        out += _lines("by the row's top logit", _slice(agg, 4), ("below 15", "15 or above"))
        out += ["  rate a request by the request's nucleus size (its first row's):",
                f"    {'n':<10}{'requests':>10}{'P':>12}{'own':>12}{'halves_split':>14}"]
        out += [f"    {n:<10}{int(v['requests']):>10}{v['P'] / v['requests']:>12.3e}{v['own_q'] / v['requests']:>12.3e}"
                f"{int(v['halves_split']):>14}" for n, v in sorted(reqs.items())]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="the directory halves_rate.json and halves_rate_positions.jsonl go to")
    ap.add_argument("--report", help="print the report of a kept halves_rate.json")
    ap.add_argument("--minutes", type=float, default=26.0, help="how long the workers take new blocks")
    ap.add_argument("--workers", type=int, default=16, help="processes, shared equally by the kinds")
    ap.add_argument("--block", type=int, default=100, help="requests a block")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--check", type=int, default=24, help="requests of each kind compared with the unstubbed reference")
    ap.add_argument("--kinds", default="random,peaked")
    ap.add_argument("--smoke", action="store_true",
                    help="a table of 2^18 random magnitudes in place of all 2^31: runs every step, and its numbers mean nothing")
    a = ap.parse_args()
    if a.report:
        print(report(json.loads(Path(a.report).read_text())))
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = a.kinds.split(",")
    t0 = time.monotonic()
    if a.smoke:
        mags = np.unique(np.random.default_rng(a.seed).integers(0, 1 << 31, 1 << 18))
        u, counts = np.unique(uniform_words(mags), return_counts=True)
        TABLE.update(finish_table(u, counts.astype(np.int64), SR.nv_logf(u)), partial=True)
    else:
        TABLE.update(build_table(a.workers))
    facts = TABLE.pop("facts")
    print(f"table in {time.monotonic() - t0:.0f} s: {json.dumps(facts)}", flush=True)
    checks = check_table(20000, a.seed)
    hand = {v["name"]: v for v in SV.vectors(K, V, TILE)}["halves-split"]
    _, own, detail = request_numbers("hand", 0, hand)
    assert hand["ref"]["halves_split"] == 0 and detail and detail[0]["i"] == 0
    checks["hand_vector_halves_split"] = {"P": next(iter(own.values()))["P"], "position_0": detail[0]}
    checks["stub"] = {kind: check_stub(kind, a.seed, a.check) for kind in kinds}
    print(f"checks in {time.monotonic() - t0:.0f} s: {json.dumps(checks)}", flush=True)

    ctx = mp.get_context("fork")
    results = ctx.Queue()
    deadline = time.monotonic() + 60 * a.minutes
    per = max(a.workers // len(kinds), 1)
    counters = {kind: (ctx.Value("q", 0), ctx.Lock()) for kind in kinds}
    procs = [ctx.Process(target=_worker, args=(kind, w, a.seed, a.block, *counters[kind], deadline, results))
             for kind in kinds for w in range(per)]
    for p in procs:
        p.start()
    merged = {kind: {"agg": {}, "requests": {}, "blocks": set()} for kind in kinds}
    errors, left = [], len(procs)
    with (out / "halves_rate_positions.jsonl").open("w") as positions:
        while left:
            what, kind, b, res = results.get()
            if what == "done":
                left -= 1
            elif what == "error":
                errors.append(res)
                print(res, flush=True)
            else:
                _add(merged[kind]["agg"], res["agg"])
                _add(merged[kind]["requests"], res["requests"])
                merged[kind]["blocks"].add(b)
                for d in res["detail"]:
                    positions.write(json.dumps(d) + "\n")
    for p in procs:
        p.join()
    record = {"schema": "scratch/cir264/halves-rate/v1", "statics": {"K": K, "V": V, "TILE": TILE}, "table": facts,
              "checks": checks, "smoke": a.smoke, "minutes": a.minutes, "workers": a.workers, "errors": errors, "kinds": {}}
    for kind in kinds:
        m = merged[kind]
        blocks = sorted(m["blocks"])
        assert blocks == list(range(len(blocks))) and len(blocks) == counters[kind][0].value or errors, (kind, blocks[-5:])
        record["kinds"][kind] = {"seed": a.seed, "first": 0, "block": a.block, "blocks": len(blocks),
                                 "agg": {json.dumps(list(key)): v for key, v in sorted(m["agg"].items())},
                                 "requests": {str(n): v for n, v in sorted(m["requests"].items())}}
    (out / "halves_rate.json").write_text(json.dumps(record, indent=1) + "\n")
    print(report(record), flush=True)
    print(f"done in {time.monotonic() - t0:.0f} s; errors {len(errors)}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
