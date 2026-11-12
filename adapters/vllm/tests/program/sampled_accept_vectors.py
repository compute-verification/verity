"""Hand vectors of `SampledAccept_v1` (import as `tests.program.sampled_accept_vectors`): one request's K + 1 rows of logits
after top-p, its K draft tokens, seed, positions and temperature, each built to land on one edge of the sampled accept.

    accepted-everywhere       every draft token holds its row's mass to the last place: every position accepted, the bonus row
                              resampled
    refused-everywhere        every draft token is 30 below its row's maximum, under the smallest log(u): every position refused
    draft-outside-top-p       position AT's draft token is -inf in its row (outside the nucleus): refused, and the mask moves
                              nothing
    draft-placeholder         position AT's draft token is -1: refused with no test, the row resampled whole
    u-at-floor-accepted       position AT's uniform is `tl.rand`'s floor and the draft token's log-probability one place above
                              log(u): accepted at the smallest u there is
    logprob-equals-log-u      the same with the log-probability equal to log(u): refused, the comparison strict
    logprob-above-log-u       an ordinary uniform, the log-probability one place above log(u): accepted
    logprob-at-log-u          the same with the log-probability equal to log(u): refused
    refused-draft-masked      position AT's draft token is refused at half its row's mass, and the Gumbel select of the whole
                              row would pick it again: the resample masks it and picks the other
    halves-split              (K >= 2) position 0's log-sum-exp is one place apart in the warp's two halves and log(u) falls
                              between the two log-probabilities: lane 0 accepts, lanes 16-31 refuse.  Outside the Definition's
                              domain: the kernel leaves the next position undefined, and the reference is lane 0's arithmetic

`AT` is the last draft position, so the positions before it are accepted and its own test decides the count.  `FLOOR_SEED`
is a seed whose stream word at position `FLOOR_POS` is one of the four `tl.rand` maps to its floor (0, 1, 2^32 - 2 and
2^32 - 1), found by scanning seeds.  A vector's expected outputs are the reference's (`sampled_accept.sampled_accept`);
`expect` names what its construction promises of them, which `vectors` asserts.

`random_rows` makes random requests instead: nuclei of 1 lane to the whole row at three spreads and one of exponents far below the
maximum's (where `ex2.approx.f32` halves its operand), each draft token its row's best lane, another lane of the nucleus or
any lane of the row, so positions are accepted and refused in every mix.
"""
from __future__ import annotations

import numpy as np

from verity.definitions.registry import sampled_accept as A, sampling as SA
from verity.definitions.registry.sampling_rows import gumbel_select_row

F = np.float32
#: `sampling.stream_key(FLOOR_SEED, FLOOR_POS)` is 0: the uniform `includes_zero=False` lifts to the floor
FLOOR_SEED, FLOOR_POS = 1579546896, 1002
#: the ordinary vectors' seed and first position, the position whose uniform under that seed the two ordinary comparisons
#: are built at (about 0.038), and the temperature of every vector (0.8)
SEED, POS, ORDINARY_POS = 20261112, 500, 505
#: halves-split's row at `ORDINARY_POS`: the logit words of lanes 0-3 of tile 0 and of tile 16, and the draft lane's, in tile 5
SPLIT_LOW = (1092177707, 1093132478, 1091618192, 1092462643)
SPLIT_HIGH = (1091526101, 1093346733, 1092720289, 1092446874)
SPLIT_DRAFT = 1091379375
TEMP = int(F(0.8).view(np.uint32))


def _f(word: int) -> np.float32:
    return np.uint32(word).view(F)


def _w(x) -> int:
    return int(F(x).view(np.uint32))


def lanes(V: int, tile: int, count: int, salt: int) -> list[int]:
    """`count` lanes of [0, V), each in a tile of its own: the last tile first, then tiles of both halves of the warp."""
    tiles = -(-V // tile)
    last = V - (tiles - 1) * tile
    spots = [(tiles - 1, (4 * salt) % last), (0, 3), (7, 129), (1, 126), (12, tile - 1), (9, 77), (15 if tiles == 17 else 16, 5)]
    assert count <= len(spots)
    return [t * tile + (off if t == tiles - 1 else (off + 4 * salt) % tile) for t, off in spots[:count]]


def _logit_for(logprob: int, V: int, tile: int, a: int, b: int) -> int:
    """The logit word of lane b at which its log-probability is the word `logprob`, in a row whose other finite lane is a
    at 0.0 (another tile's): x near logprob - log(1 - e^logprob), found among its neighbours with `row_lse`."""
    tiles = -(-V // tile)
    one = A.tl_exp(F(0.0))
    lp = float(_f(logprob))
    x0 = _w(lp - np.log1p(-np.exp(lp)))
    assert x0 >> 31, "the search walks negative logits"
    for d in sorted(range(-256, 257), key=abs):
        x = x0 + d
        m, s = np.full(tiles, -np.inf, F), np.zeros(tiles, F)
        m[a // tile], s[a // tile] = 0.0, one
        m[b // tile], s[b // tile] = _f(x), one
        if _w(_f(x) - _f(A.row_lse(m, s)[0])) == logprob:
            return x
    raise AssertionError(f"no logit within 256 places gives the log-probability {logprob:#010x}")


def vectors(K: int, V: int, tile: int = A.STATS_TILE) -> list[dict]:
    at = K - 1
    out: list[dict] = []

    def base(seed: int, first_pos: int):
        logits = np.full((K + 1, V), -np.inf, F)
        draft = []
        for i in range(K):
            row = lanes(V, tile, 4, i)
            logits[i, row[0]] = 3.0 + i
            logits[i, row[1:]] = 3.0 + i - 30.0
            draft.append(row[0])
        logits[K, lanes(V, tile, 5, K)] = np.array([1.5, 1.25, 1.0, 0.75, 0.5], F)
        return logits, draft, seed, [first_pos + i for i in range(K + 1)]

    def add(name: str, logits, draft, seed, pos, **expect) -> dict:
        ref = A.sampled_accept(logits, draft, seed, pos, TEMP, tile)
        for key, want in expect.items():
            assert ref[key] == want, (name, key, ref[key], want)
        out.append({"name": name, "logits": logits, "draft": draft, "seed": seed, "pos": pos, "temp": TEMP, "ref": ref})
        return ref

    logits, draft, seed, pos = base(SEED, POS)
    add("accepted-everywhere", logits, draft, seed, pos, verdict=[1] * K, accepted=K, test=[1] * K)

    logits, draft, seed, pos = base(SEED, POS)
    draft = [lanes(V, tile, 4, i)[1] for i in range(K)]
    add("refused-everywhere", logits, draft, seed, pos, verdict=[0] * K, accepted=0, test=[0] * K)

    logits, draft, seed, pos = base(SEED, POS)
    draft[at] += 1
    assert logits[at, draft[at]] == -np.inf
    add("draft-outside-top-p", logits, draft, seed, pos, verdict=[1] * at + [0], accepted=at)

    logits, draft, seed, pos = base(SEED, POS)
    draft[at] = -1
    add("draft-placeholder", logits, draft, seed, pos, verdict=[1] * at + [0], accepted=at)

    # position AT holds two finite lanes in two tiles, lane a at 0.0 and the draft lane b
    for seed, first_pos, equal, above in ((FLOOR_SEED, FLOOR_POS - at, "logprob-equals-log-u", "u-at-floor-accepted"),
                                          (SEED, ORDINARY_POS - at, "logprob-at-log-u", "logprob-above-log-u")):
        u = A.accept_uniform(SA.stream_key(seed, first_pos + at))
        assert (u == SA.TL_RAND_MIN_BITS) == (seed == FLOOR_SEED), hex(u)
        log_u = SA.nv_logf_gpu(u, ftz=False)
        for name, logprob in ((equal, log_u), (above, _w(np.nextafter(_f(log_u), F(0.0))))):
            logits, draft, _, pos = base(seed, first_pos)
            a, b = lanes(V, tile, 2, at)
            logits[at] = -np.inf
            logits[at, a], logits[at, b] = 0.0, _f(_logit_for(logprob, V, tile, a, b))
            draft[at] = b
            ref = add(name, logits, draft, seed, pos, verdict=[1] * at + [int(name == above)], accepted=at + (name == above))
            assert ref["u"][at] == u and ref["log_u"][at] == log_u and ref["logprob"][at] == logprob, name

    # two lanes of equal logits; the first positions at which the draft lane is refused and the whole row's select picks it
    for first_pos in range(POS, POS + 64):
        logits, draft, seed, pos = base(SEED, first_pos)
        a, b = lanes(V, tile, 2, at)
        logits[at] = -np.inf
        logits[at, [a, b]] = 2.0
        draft[at] = b
        whole, _ = gumbel_select_row(np.ascontiguousarray(logits[at]).view(np.uint32), seed, pos[at], TEMP)
        if whole == b and not A.sampled_accept(logits, draft, seed, pos, TEMP, tile)["test"][at]:
            ref = add("refused-draft-masked", logits, draft, seed, pos, verdict=[1] * at + [0], accepted=at, token=a)
            break
    else:
        raise AssertionError("no position in 64 refuses the draft lane and selects it from the whole row")

    if K >= 2:
        logits, draft, seed, pos = base(SEED, ORDINARY_POS)
        logits[0] = -np.inf
        logits[0, 0:4] = np.array(SPLIT_LOW, np.uint32).view(F)
        logits[0, 16 * tile:16 * tile + 4] = np.array(SPLIT_HIGH, np.uint32).view(F)
        draft[0] = 5 * tile + 3
        logits[0, draft[0]] = _f(SPLIT_DRAFT)
        ref = add("halves-split", logits, draft, seed, pos, verdict=[1] * K, accepted=K, halves_split=0)
        assert ref["lse_hi"][0] == ref["lse"][0] + 1
    return out


#: `random_rows`' nucleus sizes (0: the whole row) and the standard deviations of its logits around 4.0; a spread of 0 spreads
#: the logits uniformly over 200 below 4.0 instead, most of their exponentials under `ex2.approx.f32`'s -126
NUCLEI = (1, 2, 5, 40, 400, 5000, 0)
SPREADS = (0.3, 2.5, 8.0, 0.0)


def random_rows(K: int, V: int, count: int, seed: int, tile: int = A.STATS_TILE, first: int = 0) -> list[dict]:
    """Requests ``first`` to ``first + count - 1`` of ``seed``'s, in `vectors`' form, each a function of (seed, its number) alone:
    each of the K + 1 rows a nucleus of one of `NUCLEI` sizes at one of `SPREADS`, -inf outside it; a draft token the row's
    best lane (6 in 10), another lane of the nucleus (3 in 10) or any lane of the row."""
    out = []
    for c in range(first, first + count):
        rng = np.random.default_rng((seed, c))
        keep = int(rng.choice(NUCLEI)) or V
        spread = float(rng.choice(SPREADS))
        logits = np.full((K + 1, V), -np.inf, F)
        draft = []
        for i in range(K + 1):
            idx = rng.choice(V, min(keep, V), replace=False)
            logits[i, idx] = (rng.normal(0, spread, idx.size) if spread else -200 * rng.random(idx.size)).astype(F) + F(4)
            if i < K:
                kind = rng.random()
                draft.append(int(idx[np.argmax(logits[i, idx])]) if kind < 0.6 else int(rng.choice(idx)) if kind < 0.9
                             else int(rng.integers(V)))
        s, pos = int(rng.integers(1 << 62)), [int(p) for p in 1000 + 10 * c + np.arange(K + 1)]
        out.append({"name": f"random-{seed}-{c}", "logits": logits, "draft": draft, "seed": s, "pos": pos, "temp": TEMP,
                    "ref": A.sampled_accept(logits, draft, s, pos, TEMP, tile)})
    return out
