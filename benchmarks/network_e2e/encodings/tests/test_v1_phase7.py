"""REDTEAM phase 7, target 2: advice v1 (#3584) attacked past its own tests.

The lead's five probes, each an executable case:

1. prefix-freeness beyond the 18-bit exhaustive test (`test_v1.test_the_strings_are_prefix_free` only reaches 18 bits on
   a 1-2 window record): long, many-window strings, and every internal window boundary;
2. totality on adversarial frames, not just random bytes and one-bit flips: a `decode` that raises anything but `V1Error`
   breaks the "any byte string is one I1 v0 or a named refusal" contract the circuit's `AdviceCheck_v1` rests on;
3. the KV pool size field: who sets it (the prover, via the header, unless pinned), whether it can lie (yes, anywhere in
   `[least, max_gpu_blocks]`), the bits it carries, and the derived-hit undercharge a bigger claimed pool buys;
4. the "contexts read no release time" claim under design v3: the coder's contexts are a registered (public) field and
   read only the record's integer buckets at `bucket_s = 0`, so they neither leak a release time nor give a per-run
   channel;
5. the two named gaps (PROTOCOL.md "Advice v1" > "Open"), with the bits each could carry per window measured on the
   fixtures: async's lag row is not in the sync `ReplayStep_v1`, and past the no-eviction bound the cache hit is witness
   the circuit doesn't bound by the real pool.

Measured numbers are asserted as bounds and printed (run with `-s` to see them)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


V, R, C, D = _load("v1"), _load("replay"), _load("code"), _load("derived")
from verity.core.protocols.certifier import advice as I1  # noqa: E402

EOS = 2
V1_CASES = sorted(p.stem for p in FIXTURES.glob("*.jsonl"))


def step_trace(name: str):
    tok = FIXTURES / f"{name}.tokens.json"
    t = R.read_step_trace(FIXTURES / f"{name}.jsonl", tokens=json.loads(tok.read_text()) if tok.exists() else None)
    t.requests = [r if r.tokens is not None else dataclasses.replace(
        r, tokens=(1000 + i,) * r.prompt_len + (3,) * (r.output_len - r.eos) + (EOS,) * r.eos)
        for i, r in enumerate(t.requests)]
    return t


def case(name: str, **kw):
    t = step_trace(name)
    rec = V.record_of(t, (EOS,))
    reg = V.Registered(t.engine, (EOS,), row_order="scheduling", **kw)
    return t, rec, reg, V.advice_of(t, rec)


def exact_bits(data: bytes, rec, reg) -> bytearray:
    return C.BitReader.from_bytes(data, sum(V.bits_by_window(data, rec, reg).values())).bits


def _crush(name: str):
    """A larger ENC-DERIVED schedule (sync/async `Scheduler` driven to crush the KV pool): its config, I1 advice, holdings
    and I1->certified order. Exercises many past-bound hits and resumes the small v1 fixtures don't reach."""
    tr = D.load_trace(FIXTURES / f"{name}.json.gz")
    adv, order = D.advice_of(tr)
    return D.config_of(tr), adv, D.holdings_of(tr), order


def _wiring(name: str, T: int = 20, **kw):
    """ENC-DERIVED's residual of a v1 case, wired as `v1._advice` wires it (engine order, I1 numbering by first row): the
    config, the Holdings in engine order, the I1->engine map, and each I1 request's certifier window (its record
    request's `end` bucket). `rebuild(res) == adv` checks the wiring is right."""
    t, rec, reg, adv = case(name, T=T, **kw)
    engine = reg.engine
    order = V.engine_order(rec)
    reqs = [rec.requests[k] for k in order]
    index = {rec.requests[k].id: e for e, k in enumerate(order)}
    first_of = tuple(index[adv.requests[i].id] for i in range(len(adv.requests)))
    held = D.Holdings(tuple(D.CertifiedRequest(q.id, q.prompt, q.max_tokens) for q in reqs),
                      tuple(q.output for q in reqs))
    cfg = reg.config(engine)
    res = D.residual(adv, cfg, held, order=first_of)
    assert D.rebuild(res, cfg, held, ()) == adv
    win = [reqs[first_of[i]].end // T for i in range(len(adv.requests))]
    return dict(t=t, rec=rec, reg=reg, adv=adv, cfg=cfg, held=held, first_of=first_of, res=res, win=win)


# ---------------------------------------------------------------------- 1. prefix-freeness beyond 18 bits


def _long_accepted():
    """Genuinely long, many-window accepted strings: every fixture encoded under a small T, both plain and egress."""
    out = []
    for name in V1_CASES:
        for contexts in ("plain", "egress"):
            t, rec, reg, adv = case(name, T=4, contexts=contexts)
            data = V.encode(adv, rec, reg)
            out.append((exact_bits(data, rec, reg), rec, reg, adv))
    return out


def test_prefix_free_on_long_many_window_strings():
    """Beyond the 18-bit exhaustion: the strings stay prefix-free and self-delimiting at real length (hundreds of bits,
    many windows). No accepted string is a proper prefix of another; every internal window boundary and every single-bit
    truncation is refused; any one-byte extension is refused; each decodes and re-encodes to itself."""
    corpus = _long_accepted()
    lengths = sorted(len(b) for b, *_ in corpus)
    assert lengths[-1] > 100  # far beyond the 18-bit exhaustion
    assert sum(V.windows(rec, reg.T) for _, rec, reg, _ in corpus) > 2 * len(corpus)  # many windows each

    # (a) no accepted string is an accepted proper prefix of another (across every record and registration).
    for a, (ba, _, _, _) in enumerate(corpus):
        for b, (bb, rb, gb, _) in enumerate(corpus):
            if a != b and len(ba) < len(bb) and bb[: len(ba)] == ba:
                try:
                    V.decode_bits(ba, rb, gb)
                except V.V1Error:
                    continue
                raise AssertionError(f"string {a} is an accepted proper prefix of string {b}")

    # (b) every internal window boundary of a long string is refused (the self-delimiting frames need all W windows), as
    #     is every single-bit truncation; any one-byte extension is refused.
    for bits, rec, reg, adv in corpus:
        assert V.decode_bits(bits, rec, reg) == adv  # the whole string is accepted
        charge = V.bits_by_window(_bytes(bits), rec, reg)
        offset = 0
        for w in sorted(charge)[:-1]:  # the boundary after every window but the last
            offset += charge[w]
            with pytest.raises(V.V1Error):
                V.decode_bits(bits[:offset], rec, reg)
        for n in (1, len(bits) - 1):
            with pytest.raises(V.V1Error):
                V.decode_bits(bits[:n], rec, reg)
        data = V.encode(adv, rec, reg)
        for ext in (b"\x00", b"\x80"):
            with pytest.raises(V.V1Error):
                V.decode(data + ext, rec, reg)

    # (c) Kraft over the accepted strings of one record stays within 1 (prefix-free codeword lengths).
    t, rec, reg, _ = case(V1_CASES[0], T=4)
    acc_lens = []
    for bits, r, g, _ in corpus:
        if r == rec:
            acc_lens.append(len(bits))
    assert sum(2.0 ** -n for n in acc_lens) <= 1
    print(f"[prefix-free] {len(corpus)} long strings, max {lengths[-1]} bits, all self-delimiting and prefix-free")


def _bytes(bits) -> bytes:
    w = C.BitWriter()
    for b in bits:
        w.bit(b)
    return w.to_bytes()


# ---------------------------------------------------------------------- 2. totality on adversarial frames


def _adversarial_frames(reg, W: int) -> list[bytes]:
    """Structured byte strings aimed at the frame parser and the per-window segment decoder, not random noise."""
    out = []

    def pack(header, seglens, fill_bits=0, extra=b""):
        w = C.BitWriter()
        if header is not None:
            C.put_gamma(w, header)
        for n in seglens:
            C.put_gamma(w, n + 1)
            for _ in range(n):
                w.bit(1)
        for _ in range(fill_bits):
            w.bit(1)
        return w.to_bytes() + extra

    pool = reg.engine.num_gpu_blocks
    out.append(pack(1 << 34, [0] * W))                         # an absurd pool, empty segments
    out.append(pack(1, [0] * W))                               # a pool below `least`
    out.append(pack(pool, [10_000] + [0] * (W - 1)))           # a segment longer than the string (truncated)
    out.append(pack(pool, [0] * W, fill_bits=3))               # trailing nonzero pad
    out.append(pack(pool, [0] * W, extra=b"\xff"))             # a whole extra byte past the last window
    out.append(pack(pool, [0] * (W - 1)))                      # too few windows
    w = C.BitWriter()                                          # a maximal gamma prefix (refused past 64 zero bits)
    for _ in range(80):
        w.bit(0)
    out.append(w.to_bytes())
    for seglen in (1, 2, 4, 8, 16):                            # near-canonical junk the segment check rejects
        out.append(pack(pool, [seglen] + [0] * (W - 1)))
    return out


def test_decode_total_on_adversarial_frames():
    """Every structured adversarial frame, plus a fuzz of byte and bit mutations of a real encoding, decodes to a valid
    I1 v0 that re-encodes, or is refused by `V1Error` — never any other exception. `decode`'s `_total` wraps only
    ValueError/RuntimeError, so an AssertionError, KeyError or IndexError escaping it would be a real totality break."""
    t, rec, reg, adv = case("async-prefix-cache", T=20)
    W = V.windows(rec, reg.T)
    data = V.encode(adv, rec, reg)
    rng = random.Random(7)
    tries = list(_adversarial_frames(reg, W))
    for _ in range(300):  # multi-bit splices and truncations of a real encoding
        b = bytearray(data)
        for _ in range(rng.randrange(1, 6)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        tries.append(bytes(b[: rng.randrange(1, len(b) + 1)]))
    tries += [bytes(rng.randrange(256) for _ in range(rng.randrange(1, 300))) for _ in range(200)]
    tries += [data * 2, data[: len(data) // 2]]
    decoded = 0
    for x in tries:
        try:
            got = V.decode(x, rec, reg)
        except V.V1Error:
            continue
        except Exception as e:  # noqa: BLE001 — the point of the test: nothing but V1Error may escape `decode`
            raise AssertionError(f"decode raised {type(e).__name__} (not V1Error): {e}") from e
        assert type(got) is I1.Advice
        assert V.encode(got, rec, V.registered_of(x, rec, reg)) == x
        decoded += 1
    print(f"[totality] {len(tries)} adversarial frames: {decoded} decoded to valid advice, the rest refused by V1Error")
    assert decoded


# ---------------------------------------------------------------------- 3. the KV pool size field


def test_kv_pool_is_prover_set_and_lies_within_the_bound():
    """`num_gpu_blocks` is in the string's header unless pinned (so the prover sets it), free to be any value in
    `[least, max_gpu_blocks]`, and a bigger claimed pool moves cache hits from the charged region (past the no-eviction
    bound) into the free derived region. Pinning removes the header and the choice."""
    w = _wiring("async-prefix-cache")
    rec, reg, adv, engine = w["rec"], w["reg"], w["adv"], w["reg"].engine
    B, L = engine.block_size, engine.max_model_len
    least = max(R.cdiv(min(len(q.prompt) + len(q.output) + 2, L), B) for q in rec.requests) + 1
    field_bits = math.log2(reg.max_gpu_blocks - least + 1)
    real = [C.gamma_len(n) for n in (1 << 14, 1 << 17, 1 << 20)]  # the header over realistic pool sizes
    print(f"[pool] least={least} max={reg.max_gpu_blocks} -> field carries {field_bits:.1f} bits; "
          f"gamma header {min(real)}..{max(real)} bits over 2^14..2^20 blocks")
    assert 15 < field_bits < 40 and min(real) >= 19 and C.gamma_len(1 << 20) > C.gamma_len(1 << 14)

    # (a) prover-set: `registered_of` reads the header's pool; a different in-range pool is a different, valid string.
    data = V.encode(adv, rec, reg)
    assert V.registered_of(data, rec, reg).engine.num_gpu_blocks == engine.num_gpu_blocks
    big = dataclasses.replace(reg, engine=dataclasses.replace(engine, num_gpu_blocks=engine.num_gpu_blocks + 64))
    data_big = V.encode(adv, rec, big)
    assert V.decode(data_big, rec, big) == adv
    assert V.registered_of(data_big, rec, reg).engine.num_gpu_blocks == engine.num_gpu_blocks + 64
    assert data_big != data

    # (b) a bigger claimed pool keeps the no-eviction bound longer, so charged hits become free (derived). On a sync
    #     prefix-caching schedule, raising the declared pool monotonically lowers the charged (sent) hit count — the
    #     undercharge direction. The lie is bounded both ways: below `least` and above the pool_bound preemption ceiling
    #     are refused.
    cfg, adv_c, held_c, order_c = _crush("sync-crush")
    counts = {}
    for pool in (cfg.num_gpu_blocks // 2, cfg.num_gpu_blocks):
        counts[pool] = len(D.residual(adv_c, dataclasses.replace(cfg, num_gpu_blocks=pool), held_c, order=order_c).hits)
    print(f"[pool] sync-crush charged hits by declared pool: {counts}")
    assert counts[cfg.num_gpu_blocks] < counts[cfg.num_gpu_blocks // 2]  # a bigger claimed pool charges fewer
    with pytest.raises((ValueError, RuntimeError)):  # above the preemption ceiling, the lie is refused
        D.residual(adv_c, dataclasses.replace(cfg, num_gpu_blocks=cfg.num_gpu_blocks * 8), held_c, order=order_c)

    # (c) pinned: no header, the pool fixed, the prover can't choose.
    pinned = dataclasses.replace(reg, pinned=True)
    pdata = V.encode(adv, rec, pinned)
    assert V.registered_of(pdata, rec, pinned) is pinned  # _with_pool(reg, None) is reg
    assert V.bits_by_window(pdata, rec, pinned)[0] <= V.bits_by_window(data, rec, reg)[0]


# ---------------------------------------------------------------------- 4. contexts read no release time (v3)


def test_contexts_are_public_and_read_only_certified_buckets():
    """Under v3 the coder's contexts are a registered field (public, not a per-run prover choice) and read only the
    record's integer release buckets at `bucket_s = 0`, never a release time. So `decode` is invariant to the context (it
    only changes the charge), and the context is a function of the record's integer buckets."""
    t = step_trace("async-prefix-cache")
    rec = V.record_of(t, (EOS,))
    adv = V.advice_of(t, rec)
    charges = {}
    for contexts in R.CONTEXTS:
        reg = V.Registered(t.engine, (EOS,), row_order="scheduling", T=20, contexts=contexts)
        data = V.encode(adv, rec, reg)
        assert V.decode(data, rec, reg) == adv  # the context never changes which advice decodes
        charges[contexts] = sum(V.bits_by_window(data, rec, reg).values())
    print(f"[contexts] decode invariant across {R.CONTEXTS}; charges = {charges}")
    assert len(set(charges.values())) >= 2  # the context changes only the length

    # the Certified v1 builds holds integer buckets at bucket_s = 0 — no release time can enter the coder.
    reg = V.Registered(t.engine, (EOS,), row_order="scheduling", T=20, contexts="egress")
    order = V.engine_order(rec)
    reqs = [rec.requests[k] for k in order]
    ios = V._ios(rec, reg.engine, order, reg.config(reg.engine).stops)
    cert = R.Certified(reg.engine, ios, 0.0, [q.end for q in reqs], [list(q.release) for q in reqs])
    assert cert.bucket_s == 0.0
    assert all(isinstance(b, int) for b in cert.ingress)
    assert all(isinstance(b, int) for r in cert.release for b in r)

    # a per-run covert channel would need the prover to pick the context; it is in `Registered`, fixed before the run.
    assert "contexts" in {f.name for f in dataclasses.fields(V.Registered)}


# ---------------------------------------------------------------------- 5a. async lag not in the sync circuit


def test_async_lag_row_is_absent_from_the_sync_circuit():
    """The lag row (async scheduling runs one decode past a stop) is in v1's replay and walk but, per PROTOCOL.md's
    "Open", not in the sync `ReplayStep_v1`. Measured: one lag-eligible request can carry one bit a window, and the sync
    walk cannot reproduce an async advice — so the circuit would refuse the honest session or read `output_len` off by
    one for each lag."""
    for name in ("async-kv-pressure", "async-prefix-cache"):
        w = _wiring(name)
        cfg, res, win, rec, reg = w["cfg"], w["res"], w["win"], w["rec"], w["reg"]
        assert cfg.scheduler == "async" and res.lag is not None
        per_window: dict[int, int] = {}
        for i in range(len(res.lag)):
            per_window[win[i]] = per_window.get(win[i], 0) + 1  # a lag-eligible row ≤ 1 bit
        bits = sum(per_window.values())
        print(f"[async-lag] {name}: lag-eligible per window = {dict(sorted(per_window.items()))}, "
              f"≤ {bits} bits over {V.windows(rec, reg.T)} windows ({sum(res.lag)} truly ran past the stop)")
        assert bits >= 1

        # the sync circuit can't take an async advice: force sync and the walk diverges or refuses.
        sync = dataclasses.replace(cfg, scheduler="sync")
        try:
            back = D.rebuild(D.residual(w["adv"], sync, w["held"], order=w["first_of"]), sync, w["held"], ())
            assert back != w["adv"], "the sync walk reproduced an async advice; the lag row was invisible"
        except (ValueError, RuntimeError):
            pass  # refused: the sync walk has no lag row for this schedule


# ---------------------------------------------------------------------- 5b. cache hit witness past the no-eviction bound


def _sent_hit_bits(w) -> tuple[dict[int, float], int]:
    """Per-window upper bound on the past-bound cache-hit channel: at each admitting row the walk sends (cached is None),
    the witness may be any whole `unit` prefix below the row's known tokens K, i.e. ⌈K/unit⌉ values. Returns the bits a
    window and the count of sent hits. One honest walk collects the sent (step, request) pairs so it round-trips."""
    adv, cfg, held, first_of, win = w["adv"], w["cfg"], w["held"], w["first_of"], w["win"]
    rows = tuple(tuple(r.request for r in st.rows) for st in adv.steps)
    prompt_len = tuple(q.prompt_len for q in adv.requests)
    pool = D.pool_bound([[(r.request, r.start, r.n) for r in st.rows] for st in adv.steps], cfg)
    row_of = [{r.request: r for r in st.rows} for st in adv.steps]
    before, counts = [], [0] * len(adv.requests)  # samples of request i before step s (for K), from the advice itself
    for st in adv.steps:
        before.append(dict(enumerate(counts)))
        for r in st.rows:
            if r.start + r.n >= prompt_len[r.request] + counts[r.request]:
                counts[r.request] += 1
    sent: list[tuple[int, int]] = []
    number = {t: i for i, t in enumerate(first_of)}

    def hit_of(s, t, cached):
        i = number[t]
        if cached is None:
            sent.append((s, i))
        return row_of[s][i].cache_hit

    D._walk(rows, prompt_len, cfg, held=held, pool=pool, hit_of=hit_of,
            resumed_of=lambda s, k: adv.steps[s].rows[k].preempted,
            n_of=lambda s, k: adv.steps[s].rows[k].n, lag_of=lambda i: False,
            order_of=lambda s, numbers: tuple(first_of[i] for i in numbers))
    unit = cfg.unit
    per_window: dict[int, float] = {}
    for s, i in sent:
        K = prompt_len[i] + before[s].get(i, 0)
        choices = (K - 1) // unit + 1
        per_window[win[i]] = per_window.get(win[i], 0.0) + (math.log2(choices) if choices > 0 else 0.0)
    return per_window, len(sent)


def test_cache_hit_witness_is_bounded_by_donors_not_the_real_pool():
    """Past the no-eviction bound the hit is sent as witness; PROTOCOL.md's "Open" calls it a channel v1 doesn't charge
    because the circuit doesn't replay the pool. Measured and refined: the canonical-sources rule DOES bound the hit —
    a fabricated hit with no donor prefix is refused — so it is not "unconstrained"; it is bounded by the longest DONOR
    prefix, not by what LRU eviction actually left. The prover may claim any prefix a donor ever computed, evicted or
    not, so the uncaptured channel is "cached-but-evicted" recompute."""
    w = _wiring("async-prefix-cache")
    cfg, held, res, adv = w["cfg"], w["held"], w["res"], w["adv"]
    assert len(res.hits) > 0
    per_window, n = _sent_hit_bits(w)
    total = sum(per_window.values())
    print(f"[cache-hit] async-prefix-cache: {n} past-bound hits, upper-bound channel per window = "
          f"{ {wn: round(b, 1) for wn, b in sorted(per_window.items())} }, total ≤ {total:.1f} bits")
    assert n >= 1 and total > 0

    # the witness is NOT unconstrained: an upward tamper to a prefix no donor computed is refused by the sources rule.
    j = next(k for k, v in enumerate(res.hits) if v > 0)
    up = list(res.hits)
    up[j] += 1
    with pytest.raises((ValueError, RuntimeError)):
        D.rebuild(dataclasses.replace(res, hits=tuple(up)), cfg, held, ())

    # but a downward tamper (claim less or no cache) is accepted and yields a different advice: the real pool's shorter
    # hit is never enforced, so the prover may claim any donor prefix up to the longest — eviction is uncaptured.
    down = list(res.hits)
    down[j] = 0
    other = D.rebuild(dataclasses.replace(res, hits=tuple(down)), cfg, held, ())
    assert other != adv
    print("[cache-hit] a no-donor hit is refused (sources checked); a shorter donor hit is accepted (eviction uncaptured)")
