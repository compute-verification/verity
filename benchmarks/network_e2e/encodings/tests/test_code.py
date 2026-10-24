"""The advice's codes (`code.py`): round trips, prefix-freeness and Kraft (what the charge rests on), and pinned bit counts."""

from __future__ import annotations

import importlib.util
import itertools
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str):
    """By path: the directory's name shadows the stdlib's `encodings`, and `code.py` the stdlib's `code`."""
    if f"enc_{name}" in sys.modules:
        return sys.modules[f"enc_{name}"]
    spec = importlib.util.spec_from_file_location(f"enc_{name}", HERE.parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("code")
S = _load("synth")
M = _load("measure")


# ---------------------------------------------------------------------------------------------- universal codes


def test_gamma_round_trip_and_length():
    w = C.BitWriter()
    for n in range(1, 2000):
        k = len(w)
        C.put_gamma(w, n)
        assert len(w) - k == C.gamma_len(n) == 2 * (n.bit_length() - 1) + 1
    r = C.BitReader(w.bits)
    assert [C.get_gamma(r) for _ in range(1, 2000)] == list(range(1, 2000))
    with pytest.raises(ValueError, match="Elias-gamma codes n ≥ 1"):
        C.put_gamma(C.BitWriter(), 0)


def test_zigzag_and_width():
    assert [C.zigzag(v) for v in (0, -1, 1, -2, 2)] == [0, 1, 2, 3, 4]
    assert all(C.unzigzag(C.zigzag(v)) == v for v in range(-500, 500))
    assert [C.width(n) for n in (1, 2, 3, 4, 5, 1024, 1025)] == [0, 1, 2, 2, 3, 10, 11]


def test_bytes_round_trip():
    w = C.BitWriter()
    w.uint(0b1011001, 7)
    w.uint(5, 3)
    assert w.to_bytes() == bytes([0b10110011, 0b01000000])
    assert C.BitReader.from_bytes(w.to_bytes(), len(w)).bits == w.bits


# --------------------------------------------------------------------------------------------- arithmetic coder


def _random_table(rng: random.Random, lo: int, n: int) -> object:
    freqs = [rng.randint(1, 400) for _ in range(n)]
    return C.Table(lo, tuple(freqs), rng.randint(1, 50))


def test_arithmetic_round_trip_with_any_continuation():
    """Each segment decodes to its symbols whatever follows it, and the decoder knows where it ends without reading it."""
    rng = random.Random(1)
    for trial in range(40):
        tables = [_random_table(rng, rng.randint(-5, 5), rng.randint(1, 30)) for _ in range(3)]
        msgs = []
        out = C.BitWriter()
        lens = []
        for seg in range(3):
            enc = C.ArithEncoder(out)
            msg = []
            for _ in range(rng.randint(0, 60)):
                t = rng.choice(range(3))
                flo = rng.randint(tables[t].lo - 20, tables[t].lo + 3)
                fhi = rng.randint(flo, tables[t].hi + 20)
                v = rng.randint(flo, fhi)
                tables[t].put(enc, v, flo, fhi)
                msg.append((t, flo, fhi, v))
            k = len(out)
            lens.append(enc.finish())
            msgs.append(msg)
        garbage = bytearray(rng.getrandbits(1) for _ in range(rng.randint(0, 80)))
        r = C.BitReader(out.bits + garbage)
        for seg, msg in enumerate(msgs):
            dec = C.ArithDecoder(r)
            got = [(t, flo, fhi, tables[t].get(dec, flo, fhi)) for t, flo, fhi, _ in msg]
            assert got == msg
            assert dec.finish() == lens[seg]
        assert r.pos == len(out)


def test_arithmetic_segment_with_nothing_to_code_is_empty():
    out = C.BitWriter()
    enc = C.ArithEncoder(out)
    C.Table(3, (5,), 1).put(enc, 3, 3, 3)  # the only feasible value: no symbol coded
    assert enc.finish() == 0 and len(out) == 0
    dec = C.ArithDecoder(C.BitReader(out.bits))
    assert C.Table(3, (5,), 1).get(dec, 3, 3) == 3 and dec.finish() == 0


def test_arithmetic_length_is_within_two_bits_of_the_model():
    rng = random.Random(2)
    t = C.Table(0, (60000, 4000, 1000, 100), 10)
    out = C.BitWriter()
    enc = C.ArithEncoder(out)
    ideal = 0.0
    for _ in range(5000):
        v = rng.choices(range(4), weights=(60000, 4000, 1000, 100))[0]
        ideal += t.cost(v, 0, 3)
        t.put(enc, v, 0, 3)
    n = enc.finish()
    assert ideal <= n <= ideal + 5000 * C.OVERHEAD_PER_SYMBOL + 2


def _all_codewords(table, flo: int, fhi: int, k: int) -> dict:
    out = {}
    for msg in itertools.product(range(flo, fhi + 1), repeat=k):
        w = C.BitWriter()
        enc = C.ArithEncoder(w)
        for v in msg:
            table.put(enc, v, flo, fhi)
        enc.finish()
        out[msg] = bytes(w.bits)
    return out


def _prefix_free(words) -> bool:
    ws = sorted(words)
    return all(not b.startswith(a) for a, b in zip(ws, ws[1:]))


def test_arithmetic_codewords_are_prefix_free_and_satisfy_kraft():
    """What the charge rests on: a prefix-free code has at most 2^L codewords of at most L bits (Kraft)."""
    t = C.Table(0, (50000, 9000, 900, 90, 9), 7)  # with an escape over [5, 9]
    words = _all_codewords(t, 0, 9, 3)
    assert len(set(words.values())) == len(words) == 1000
    assert _prefix_free(words.values())
    assert sum(2.0 ** -len(w) for w in words.values()) <= 1.0
    for L in range(1, 40):
        assert sum(len(w) <= L for w in words.values()) <= 2 ** L


def test_fixed_and_gamma_streams_are_prefix_free():
    for cls in (C.FixedStream, C.GammaStream):
        words = []
        for a, b in itertools.product(range(-3, 4), range(0, 6)):
            w = C.BitWriter()
            st = cls(w=w)
            st.sym("x", a, -3, 3)
            st.sym("y", b, 0, 5)
            words.append(bytes(w.bits))
        assert _prefix_free(words) and sum(2.0 ** -len(w) for w in words) <= 1.0


# ------------------------------------------------------------------------------------------------------ tables


def test_table_escape_and_truncation_round_trip():
    rng = random.Random(3)
    t = C.Table(-4, tuple(rng.randint(1, 900) for _ in range(9)), 3)
    b = C.Binned(C.Table(0, tuple(rng.randint(1, 900) for _ in range(12)), 2))
    cases = [(t, v, lo, hi) for lo, hi in ((-100, 100), (-4, 4), (2, 70), (-30, -5), (0, 0)) for v in range(lo, hi + 1)]
    cases += [(b, v, lo, hi) for lo, hi in ((0, 3000), (5, 9), (1000, 1023)) for v in range(lo, hi + 1, 7)]
    out = C.BitWriter()
    enc = C.ArithEncoder(out)
    ideal = 0.0
    for m, v, lo, hi in cases:
        ideal += m.cost(v, lo, hi)
        assert m.cost(v, lo, hi) <= m.worst(lo, hi) + 1e-9
        m.put(enc, v, lo, hi)
    enc.finish()
    dec = C.ArithDecoder(C.BitReader(out.bits))
    assert [m.get(dec, lo, hi) for m, _, lo, hi in cases] == [v for _, v, _, _ in cases]
    with pytest.raises(ValueError, match="outside its feasible range"):
        t.put(C.ArithEncoder(C.BitWriter()), 5, -4, 4)


def test_fit_table_floor_and_total():
    t = C.fit_table([0] * 1000 + [1] * 10 + [99] * 3, 0, 4, escape_share=2 ** -8)
    assert min(t.freqs) >= 1 and sum(t.freqs) + t.escape <= C.MAX_TOTAL
    assert t.escape >= C.MAX_TOTAL // 256
    assert t.cost(0, 0, 10) < 0.05 < t.cost(3, 0, 10)
    assert t.worst(0, 10) <= 8 + 0.02 + C.uniform_bits(6)


# ------------------------------------------------------------------------------------------------- the codes


@pytest.fixture(scope="module")
def corpus():
    """Two small synthetic traces: the codebook is fit on one, the codes are measured on the other."""
    p = dict(rate=1.5, windows=2, T=600, prompt_max=1024, max_tokens=256, max_batched_tokens=1024)
    train = M.synth_corpus_params([S.Params(seed=11, **p)])
    test = M.synth_corpus_params([S.Params(seed=12, **p)])
    caps = C.Caps(prompt_max=1024, output_max=256, seqs_max=256, tokens_max=1024, steps_per_bucket=32)
    cb = M.fit(train, caps, 600, 100_000)
    return cb, test[0]


def test_every_code_round_trips(corpus):
    cb, t = corpus
    adv, pub, admit = t["advice"], t["public"], t["admit"]
    rep = C.TraceReplay(adv)
    for fam in C.FAMILIES:
        bits, lens = C.encode_v0(adv, pub, cb, fam)
        assert sum(lens) == len(bits)
        back = C.decode_v0(C.BitReader(bits.bits), pub, cb, fam)
        assert back.steps == adv.steps and back.release == adv.release
        assert [(q.prompt_len, q.output_len, q.finish) for q in back.requests] == \
            [(q.prompt_len, q.output_len, q.finish) for q in adv.requests]
        bits, lens = C.encode_admit(admit, pub, cb, fam, rep)
        assert sum(lens) == len(bits)
        assert C.decode_admit(C.BitReader(bits.bits), pub, cb, fam, rep) == admit


#: Bits per window on the test trace (seed 12) under the codebook fit on seed 11. A change here changes what the
#: table reports: rerun `measure.py` and restate the table.
PINNED = {
    "v0/fixed": [980172, 762122, 23884],
    "v0/gamma": [165816, 129240, 3751],
    "v0/static": [32252, 25187, 465],
    "v0/cond": [2484, 2195, 47],
    "admit/fixed": [2450, 2075, 0],
    "admit/gamma": [992, 947, 0],
    "admit/static": [724, 661, 0],
    "admit/cond": [302, 259, 0],
    "admit/cond+adapt": [316, 258, 0],
}


def test_bit_counts_are_pinned(corpus):
    cb, t = corpus
    rep = C.TraceReplay(t["advice"])
    got = {}
    for key in PINNED:
        subj, fam = key.split("/")
        if subj == "v0":
            got[key] = C.encode_v0(t["advice"], t["public"], cb, fam)[1]
        else:
            got[key] = C.encode_admit(t["admit"], t["public"], cb, fam, rep)[1]
    assert got == PINNED


def test_each_window_is_within_its_worst_case(corpus):
    cb, t = corpus
    rep = C.TraceReplay(t["advice"])
    for fam in C.FAMILIES:
        lens = C.encode_admit(t["admit"], t["public"], cb, fam, rep)[1]
        for w, syms in enumerate(C.symbols_admit(t["admit"], t["public"], cb, fam, rep)):
            assert lens[w] <= C.worst_bits(syms, cb, fam) + 1e-9
            assert lens[w] <= len(syms) * M.worst_admit_arrival(cb, t["public"].T, fam) + 2
        lens = C.encode_v0(t["advice"], t["public"], cb, fam)[1]
        for w, syms in enumerate(C.symbols_v0(t["advice"], t["public"], cb, fam)):
            assert lens[w] <= C.worst_bits(syms, cb, C._v0_family(fam) if fam != "fixed" else fam) + 1e-9


def test_the_conditional_codes_shrink_the_advice(corpus):
    cb, t = corpus
    rep = C.TraceReplay(t["advice"])
    v0 = {f: sum(C.encode_v0(t["advice"], t["public"], cb, f)[1]) for f in ("fixed", "gamma", "static", "cond")}
    ad = {f: sum(C.encode_admit(t["admit"], t["public"], cb, f, rep)[1]) for f in ("fixed", "gamma", "static", "cond")}
    assert v0["fixed"] > v0["gamma"] > v0["static"] > v0["cond"]
    assert ad["fixed"] > ad["gamma"] > ad["static"] > ad["cond"]
    assert v0["fixed"] > 100 * v0["cond"]


def test_codebook_json_round_trip(corpus):
    cb, _ = corpus
    back = C.Codebook.from_json(cb.to_json())
    assert back.digest() == cb.digest() and len(cb.digest()) == 128
    assert set(back.tables) == set(C.schema(cb.caps, 600))


# --------------------------------------------------------------------------------------------------- negatives


def test_refuses_a_field_past_its_cap(corpus):
    cb, t = corpus
    adv = t["advice"]
    q = adv.requests[0]
    bad = C.Advice((C.RequestAdvice(q.id, cb.caps.prompt_max + 1, q.output_len, q.finish),) + adv.requests[1:],
                   adv.steps, adv.release)
    for fam in C.FAMILIES:
        with pytest.raises(ValueError, match="prompt_len 1025 outside its feasible range"):
            C.encode_v0(bad, t["public"], cb, fam)


def test_refuses_truncated_advice(corpus):
    cb, t = corpus
    for fam in ("fixed", "gamma"):
        bits, _ = C.encode_v0(t["advice"], t["public"], cb, fam)
        with pytest.raises(ValueError, match="advice ends inside a codeword"):
            C.decode_v0(C.BitReader(bits.bits[:len(bits) // 2]), t["public"], cb, fam)


def test_refuses_admission_out_of_order(corpus):
    cb, t = corpus
    admit = list(t["admit"])
    admit[2], admit[3] = admit[3] + 1, admit[2]
    with pytest.raises(ValueError, match="non-decreasing"):
        C.encode_admit(admit, t["public"], cb, "cond", C.TraceReplay(t["advice"]))


def test_refuses_admission_past_the_replayed_backlog(corpus):
    """An admission after the engine went idle with nothing to run is no honest step: the conditional code has no
    codeword for it."""
    cb, t = corpus
    rep = C.TraceReplay(t["advice"])
    admit = list(t["admit"])
    i = next(i for i in range(1, len(admit)) if admit[i] == rep.end(i) + 1)
    admit = admit[:i] + [a + 1 for a in admit[i:]]
    with pytest.raises(ValueError, match="admit_res_.* outside its feasible range"):
        C.encode_admit(admit, t["public"], cb, "cond", rep)


def test_refuses_a_missing_table(corpus):
    cb, t = corpus
    thin = C.Codebook(cb.caps, cb.tau_us, cb.steptime, {k: v for k, v in cb.tables.items() if k != "s:admit_gap"})
    with pytest.raises(KeyError, match="admit_gap"):
        C.encode_admit(t["admit"], t["public"], thin, "static")
