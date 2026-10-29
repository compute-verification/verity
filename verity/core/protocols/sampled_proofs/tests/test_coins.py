"""The verifier's coins: exact samplers over its own randomness, and the one-stage draws they make, byte for byte the Lean
verifier's `Flock.Draw`."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from verity.core.protocols.sampled_proofs.coins import OwnRandomness, closure_of, draw_first
from verity.core.protocols.sampled_proofs.one_stage import draw as D, verifier as V


def _seeded(seed: int) -> OwnRandomness:
    return OwnRandomness(random.Random(seed).randbytes)


def _stream(b: bytes):
    at = [0]

    def read(n: int) -> bytes:
        out = b[at[0]:at[0] + n]
        at[0] += n
        return out
    return read


def _stratified(k: int = 4) -> dict:
    """A stratified sampling design over 10 proof blocks: stratum A the proof blocks [0, 4) and [7, 10), stratum B [4, 7)."""
    strata = [("A", [[0, 4], [7, 10]]), ("B", [[4, 7]])]
    return {"law": "stratified", "k": k,
            "strata": [{"template": t, "units": rs, "k": D.stratum_k(k, sum(h - l for l, h in rs), 10)} for t, rs in strata]}


def _work(k: int = 4) -> dict:
    """A work sampling design over the same strata: A's proof blocks do work 3, B's none (sampled at their floor)."""
    strata = [("A", [[0, 4], [7, 10]], 3), ("B", [[4, 7]], 0)]
    total = sum(w * sum(h - l for l, h in rs) for _, rs, w in strata)
    return {"law": "work", "k": k, "strata": [{"template": t, "units": rs, "work": w, "floor": 1,
                                               "k": D.work_k(k, w, sum(h - l for l, h in rs), total, 1)}
                                              for t, rs, w in strata]}


def _uniform_on(n, src, pos):
    """The Lean verifier's `Flock.Draw.uniform` on a raw stream."""
    w = (n.bit_length() + 64 + 7) // 8
    limit = (1 << 8 * w) - (1 << 8 * w) % n
    while pos + w <= len(src):
        x = int.from_bytes(src[pos:pos + w], "big")
        pos += w
        if x < limit:
            return x % n, pos
    return None


def _subset_on(k, n, src):
    swapped, pos = {}, 0
    for i in range(k):
        r = _uniform_on(n - i, src, pos)
        if r is None:
            return None
        j, pos = i + r[0], r[1]
        swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
    return sorted(swapped.get(i, i) for i in range(k))


def _draw_on(law, n, src):
    """The Lean verifier's `Flock.Draw.drawOn`: subset and bernoulli on the one stream; stratified and work on consecutive
    segments, stratum s's the next 2 k_s width(n_s) + 64 bytes (`verity/ml/flock/tests/stratified_agree.py`)."""
    if law["law"] == "subset":
        return _subset_on(law["k"], n, src)
    if law["law"] == "bernoulli":
        num, den = map(int, law["p"].split("/"))
        if num == 0:
            return []
        if num == den:
            return list(range(n))
        out, pos = [], 0
        for u in range(n):
            r = _uniform_on(den, src, pos)
            if r is None:
                return None
            out += [u] if r[0] < num else []
            pos = r[1]
        return out
    out, at = [], 0
    for s in law["strata"]:
        n_s = sum(hi - lo for lo, hi in s["units"])
        need = 2 * s["k"] * ((n_s.bit_length() + 64 + 7) // 8) + 64
        local = _subset_on(s["k"], n_s, src[at:at + need])
        if local is None:
            return None
        flat = [u for lo, hi in s["units"] for u in range(lo, hi)]
        out += [flat[j] for j in local]
        at += need
    return sorted(out)


def test_own_randomness_is_the_lean_verifiers_samplers_on_raw_bytes():
    rng = random.Random(7)
    for case in range(200):
        n = rng.randint(1, 5000)
        k = rng.randint(0, min(n, 20))
        stream = bytearray(rng.randbytes(2000))
        if case % 4 == 0:
            stream[:16] = b"\xff" * 16
        assert OwnRandomness(_stream(bytes(stream))).subset(k, n) == tuple(_subset_on(k, n, bytes(stream)))
        p = Fraction(rng.randint(1, 9), 10)
        got = OwnRandomness(_stream(bytes(stream))).bernoulli(p)
        assert got == (_uniform_on(p.denominator, bytes(stream), 0)[0] < p.numerator)
    assert OwnRandomness(_stream(b"")).bernoulli(Fraction(1)) is True
    with pytest.raises(ValueError):
        OwnRandomness(_stream(b"\x00" * 3)).uniform(10)


def test_draw_first_is_the_lean_verifiers_draw_on_the_same_bytes():
    """`draw_first` reads the verifier's bytes as `Flock.Draw.drawWith` does: on a stream that doesn't run out, its proof blocks
    are `drawOn`'s on the same bytes, for every sampling design (a bernoulli sampling design at its NUM/DEN as written, unreduced), and
    `one_stage.draw.check_draw` accepts the sample against the sampling design and the population."""
    rng = random.Random(8)
    laws = [{"law": "subset", "k": 0}, {"law": "subset", "k": 4}, {"law": "subset", "k": 10}, {"law": "bernoulli", "p": "2/6"},
            {"law": "bernoulli", "p": "0/3"}, {"law": "bernoulli", "p": "3/3"}, _stratified(1), _stratified(4), _stratified(10),
            _work(1), _work(4), _work(10)]
    for case in range(120):
        law = laws[case % len(laws)]
        src = rng.randbytes(4096)
        d = draw_first(OwnRandomness(_stream(src)), law, 10)
        assert d["units"] == _draw_on(law, 10, src), (case, law)
        assert d == {**law, "population": 10, "units": d["units"]} and D.check_draw(d, {"population": 10}, law).ok
    big = {"law": "subset", "k": 300}
    src = rng.randbytes(8192)
    assert draw_first(OwnRandomness(_stream(src)), big, 1_000_000)["units"] == _draw_on(big, 1_000_000, src)


def test_a_stream_that_runs_out_is_extended_as_lean_does():
    """Lean's `onWith`: when the first `need` bytes can't finish the sample (here every word is rejected), `need` more are
    read and the sampler reruns from the start of the longer stream; a source that ends is refused."""
    need = 2 * 3 * 9 + 64
    src = b"\xff" * need + random.Random(9).randbytes(need)
    assert draw_first(OwnRandomness(_stream(src)), {"law": "subset", "k": 3}, 10)["units"] == _subset_on(3, 10, src)
    with pytest.raises(ValueError, match="ran out"):
        draw_first(OwnRandomness(_stream(b"\xff" * need)), {"law": "subset", "k": 3}, 10)


def test_a_work_draw_carries_its_closure():
    """PoUW's shape: a work sampling design over tiles with the verifier's closure map (a tile and what its credit reads).  The sample
    carries `closure_of` (Lean `closureOf`: what sampled proof blocks' closures hold beyond the sampled proof blocks, ascending, omitted
    when empty), and `one_stage.draw.proved` is sampled and closure."""
    closure = {0: [1, 4], 2: [3], 4: [5, 0], 7: [8, 9], 9: [7]}
    assert closure_of(closure, [0, 4]) == [1, 5] and closure_of(closure, [7, 9]) == [8] and closure_of(closure, [6]) == []
    law = _work()
    for seed in range(20):
        d = draw_first(_seeded(seed), law, 10, closure)
        want = closure_of(closure, d["units"])
        assert d.get("closure", []) == want and ("closure" in d) == bool(want)
        assert D.check_draw(d, {"population": 10}, law).ok and D.proved(d) == sorted(d["units"] + want)
    for other in ({"law": "subset", "k": 3}, _stratified()):
        with pytest.raises(ValueError, match="work draw only"):
            draw_first(_seeded(0), other, 10, closure)


def test_draw_first_refuses_what_lean_refuses():
    for law, n in (({"law": "subset", "k": 4}, 3), (_stratified(), 9), ({**_stratified(), "k": 5}, 10),
                   ({"law": "bernoulli", "p": "1/0"}, 10), ({"law": "uniform", "k": 1}, 10)):
        with pytest.raises(ValueError):
            draw_first(_seeded(0), law, n)


def test_the_draws_come_from_the_lean_verifiers_coins():
    d = draw_first(OwnRandomness(V.coins), _work(), 10)
    assert D.check_draw(d, {"population": 10}, _work()).ok
