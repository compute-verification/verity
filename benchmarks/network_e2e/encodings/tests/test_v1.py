"""Advice v1: `decode(encode(a)) == a` on the replay's step traces and a trace-corpus trace, under the uniform codebook
and a fitted one; the engine's arrival order is sent as its deviations from the warden's release order and recovered
from the advice's rows; the hits past the pool bound are sent and held to the replay's; the KV pool is the
registration's, never the string's; each value is charged to A5's window, and the certifier's state chain carries
the decisions charged before their step's window; the strings are prefix-free (so the charge
meets LEAK-LEAN's `BitsCount`); and `decode` is total, refusing by name everything but a canonical string. REDTEAM's
phase 7 probes of v1 (prefix-freeness on long strings, totality on adversarial frames, the pool field, what the book
changes, a shortened shortfall) are among them."""

from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import json
import random
import shutil
import sys
from itertools import accumulate, product
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


V, R, C, CV = _load("v1"), _load("replay"), _load("code"), _load("code_v1")
from verity.core.protocols.certifier import advice as I1  # noqa: E402
from verity.core.protocols.certifier import replay_chain as RC  # noqa: E402

CASES = sorted(p.stem for p in FIXTURES.glob("*.jsonl"))
EOS = 2
STOPS = (128009, 128001, 128008)


def step_trace(name: str):
    """A replay fixture (`test_replay`'s), each request given tokens where the trace has none: its own prompt id, and
    outputs that end on the stop id exactly when it stopped there."""
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


def _fit_cases():
    """A book fit on every case's residual (a test's book; a registration fits on other runs)."""
    return CV.fit_book([V.residual(a, rec, reg) for _, rec, reg, a in (case(n, T=20) for n in CASES)], V.CAPS)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("book", ("uniform", "fit"))
def test_decode_inverts_encode(name, book):
    t, rec, reg, adv = case(name, T=20, book=_fit_cases() if book == "fit" else None)
    data = V.encode(adv, rec, reg)
    assert V.decode(data, rec, reg) == adv == I1.decode(I1.encode(adv))
    charge = V.bits_by_window(data, rec, reg)
    assert len(charge) == V.windows(rec, reg.T) > 1
    assert 0 <= 8 * len(data) - sum(charge.values()) < 8
    assert sum(charge.values()) < I1.total_bits(adv) // 100
    assert V.decode_bits(exact_bits(data, rec, reg), rec, reg) == adv


def test_the_case_set_covers_hits_resumes_and_async():
    advs = {name: case(name)[3] for name in CASES}
    rows = [x for a in advs.values() for st in a.steps for x in st.rows]
    assert any(x.sources for x in rows) and any(x.preempted for x in rows)
    assert {case(n)[0].engine.async_scheduling for n in CASES} == {False, True}


def _corpus(tmp_path: Path) -> Path:
    """The derived rules' trace-corpus fixture (4 s at 12/s on 100 KV blocks, prefix caching, preemptions) with each
    batch in Model Runner V2's order (the scheduler's, sorted stably by n) and each stop-id finish marked `eos`."""
    src = FIXTURES / "corpus-smallkv"
    for f in ("meta.json", "server.log"):
        shutil.copy(src / f, tmp_path / f)
    out = []
    for line in gzip.decompress((src / "steps.jsonl.gz").read_bytes()).decode().splitlines():
        s = json.loads(line)
        if s.get("batch"):
            sched = ([r for r in s["cached"] if not r["resumed"]] + [r for r in s["cached"] + s["new"] if r["resumed"]]
                     + [r for r in s["new"] if not r["resumed"]])
            s["batch"] = [r["id"] for r in sorted((r for r in sched if r["id"] in set(s["batch"])),
                                                  key=lambda r: r["num_scheduled_tokens"])]
        out.append(json.dumps(s) + "\n")
    (tmp_path / "steps.jsonl").write_text("".join(out))
    reqs = [json.loads(line) for line in gzip.decompress((src / "requests.jsonl.gz").read_bytes()).decode().splitlines()]
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps({**q, "eos": q["finish_reason"] == "stop"}) + "\n"
                                                     for q in reqs))
    return tmp_path


@pytest.fixture(scope="module")
def smallkv(tmp_path_factory):
    """The trace-corpus fixture's trace, record, advice and registration: sync scheduling on 100 blocks, where the pool
    bound fails and the replay evicts."""
    t = R.read_corpus_trace(_corpus(tmp_path_factory.mktemp("smallkv")), ingress="warden")
    rec = V.record_of(t, STOPS)
    return t, rec, V.advice_of(t, rec), V.Registered(t.engine, STOPS)


def test_a_trace_corpus_trace_round_trips(smallkv):
    t, rec, adv, reg = smallkv
    rows = [x for st in adv.steps for x in st.rows]
    assert any(x.preempted for x in rows) and any(x.sources for x in rows)
    for book in (None, CV.fit_book([V.residual(adv, rec, reg)], V.CAPS)):
        r = dataclasses.replace(reg, book=book)
        data = V.encode(adv, rec, r)
        assert V.decode(data, rec, r) == adv
        assert 0 <= 8 * len(data) - sum(V.bits_by_window(data, rec, r).values()) < 8
    with pytest.raises(V.V1Error, match="outside v1|isn't the replay's"):
        V.encode(adv, rec, dataclasses.replace(reg, row_order="scheduling"))


def _string(wins, reg) -> bytes:
    return CV.pack(V._string(wins, reg)[0])[0]


def _ordinals(t, rec) -> list[int]:
    """The trace's engine order, as the record's ordinals."""
    ordinal = {q.id: k for k, q in enumerate(rec.requests)}
    return [ordinal[i] for i in t.ids]


def test_the_order_is_sent_as_its_deviations_from_release_order():
    """Two requests released in one bucket that the engine took the other way round: the record's release order puts
    the later arrival first, so v1 sends one order entry, (its position, rank 1), and decodes back the advice, the
    engine's arrival order and each I1 request's warden ordinal (`streams`, the I/O tie). Another rank decodes to
    another advice or is refused, and a rank or position past the requests left is refused."""
    t = step_trace("async-light")
    c = t.certified(0.1)
    k = next(k for k in range(len(t.requests) - 1) if c.ingress[k] == c.ingress[k + 1])
    ts = list(t.ingress_t)
    ts[k], ts[k + 1] = ts[k + 1], ts[k]
    t = dataclasses.replace(t, ingress_t=ts)
    rec = V.record_of(t, (EOS,))
    reg = V.Registered(t.engine, (EOS,), row_order="scheduling", T=20)
    adv = V.advice_of(t, rec)
    assert V.arrival_order(adv, rec, reg.row_order) == _ordinals(t, rec) != V.release_order(rec)
    wins = V.residual(adv, rec, reg)
    assert [e for w in wins for e in w.order] == [(k, (1,))]
    data = V.encode(adv, rec, reg)
    got = V.decode_full(data, rec, reg)
    assert got == (adv, V.streams(adv, rec), tuple(_ordinals(t, rec)))
    assert [rec.requests[o].id for o in got.streams] == [q.id for q in adv.requests]
    w = next(j for j, x in enumerate(wins) if x.order)
    for p, r, refused in ((k, 2, False), (k, len(rec.requests) - k, True), (len(rec.requests), 1, True)):
        bad = _string(wins[:w] + [dataclasses.replace(wins[w], order=((p, (r,)),))] + wins[w + 1:], reg)
        try:
            other = V.decode(bad, rec, reg)
        except V.V1Error as e:
            assert not refused or "rank" in str(e) or "names no arrival" in str(e)
            continue
        assert not refused and other != adv and V.encode(other, rec, reg) == bad


def test_the_batch_order_is_recovered_from_later_steps(smallkv):
    """Under the batch order (a stable sort by n) a step's new requests of different n don't show their arrival order;
    the later steps where they run with equal n do. With the release order inside such a step reversed, so that release
    rank would pick the wrong order, `arrival_order` still recovers the engine's, and v1 sends and decodes it."""
    t, _, _, reg = smallkv
    _, rec0, adv0, _ = smallkv
    seen, group = set(), None
    for st in adv0.steps:
        new = [x for x in st.rows if x.request not in seen]
        seen.update(x.request for x in new)
        if len({x.n for x in new}) > 2:
            group = [t.ids.index(adv0.requests[x.request].id) for x in new]
            break
    assert group is not None
    ts = list(t.ingress_t)
    for a, b in zip(sorted(group), sorted(group, reverse=True)):
        ts[a] = t.ingress_t[b]
    t = dataclasses.replace(t, ingress_t=ts)
    rec = V.record_of(t, STOPS)
    adv = V.advice_of(t, rec)
    assert adv == adv0 and V.arrival_order(adv, rec, "batch") == _ordinals(t, rec)
    wins = V.residual(adv, rec, reg)
    assert sum(len(w.order) for w in wins) >= len(group) // 2
    got = V.decode_full(V.encode(adv, rec, reg), rec, reg)
    assert got.advice == adv and list(got.arrival) == _ordinals(t, rec)


def test_hits_past_the_bound_are_sent_and_held_to_the_replay(smallkv):
    """Past the pool bound every lookup's shortfall is sent and charged, nonzero where the replay evicted; any other
    shortfall (one more, one fewer, a shortened one), or one hit more or fewer, is refused."""
    t, rec, adv, reg = smallkv
    wins = V.residual(adv, rec, reg)
    hits = [h for w in wins for h in w.hits]
    assert len(hits) > sum(len(st.new) for st in t.steps) and min(hits) == 0 < max(hits)
    assert _string(wins, reg) == V.encode(adv, rec, reg)
    w = next(k for k, x in enumerate(wins) if x.hits)
    s = next((k, j) for k, x in enumerate(wins) for j, h in enumerate(x.hits) if h)
    short = list(wins[s[0]].hits)
    short[s[1]] -= 1
    for at, changed in ((w, (wins[w].hits[0] + 1,) + wins[w].hits[1:]), (w, wins[w].hits[1:]),
                        (w, wins[w].hits + (0,)), (s[0], tuple(short))):
        bad = wins[:at] + [dataclasses.replace(wins[at], hits=changed)] + wins[at + 1:]
        with pytest.raises(V.V1Error, match="replay's residual"):
            V.decode(_string(bad, reg), rec, reg)


def test_the_pool_is_registered_never_sent(smallkv):
    """The KV pool's size was a header field the prover set, about 20 bits, and a bigger declared
    pool bought derived hits. Now the registration holds it (no `pinned`, no header): the string is code_v1's encoding
    of the residual alone, nothing in it moves the pool, and the same string under another registered pool decodes to
    another advice or is refused."""
    t, rec, adv, reg = smallkv
    assert not {"pinned", "max_gpu_blocks"} & {f.name for f in dataclasses.fields(V.Registered)}
    assert not hasattr(V, "registered_of")
    data = V.encode(adv, rec, reg)
    assert data == CV.pack(CV.encode(V.residual(adv, rec, reg), CV.Book.uniform(V.CAPS)).bitlist())[0]
    for blocks in (t.engine.num_gpu_blocks + 60, t.engine.num_gpu_blocks - 20):
        other = dataclasses.replace(reg, engine=dataclasses.replace(t.engine, num_gpu_blocks=blocks))
        try:
            assert V.decode(data, rec, other) != adv
        except V.V1Error:
            pass


def test_the_book_changes_only_the_length():
    """On the book that replaced the contexts: the registered book is public and fixed before the
    run, and changes only a string's length, never which advice it decodes to."""
    t, rec, reg, adv = case("async-prefix-cache", T=20)
    charges = set()
    for book in (None, _fit_cases()):
        r = dataclasses.replace(reg, book=book)
        data = V.encode(adv, rec, r)
        assert V.decode(data, rec, r) == adv
        charges.add(sum(V.bits_by_window(data, rec, r).values()))
    assert len(charges) == 2


def test_no_hit_is_sent_while_the_bound_holds():
    """Sync scheduling with every request's blocks inside the pool: each hit is derived, at 0 bits."""
    rec, reg = _tiny(2)
    reg = dataclasses.replace(reg, engine=dataclasses.replace(reg.engine, enable_prefix_caching=True))
    order = V.release_order(rec)
    rp, _, _ = V._replay(rec, reg, reg.engine, order, lambda i, k: True)
    adv = V._advice(rec, reg, V._check(rec, reg, reg.engine), order, rp)
    assert V.decode(V.encode(adv, rec, reg), rec, reg) == adv
    assert not any(w.hits for w in V.residual(adv, rec, reg))
    t, rec, reg, adv = case("async-prefix-cache", T=20)
    assert t.engine.async_scheduling and sum(len(w.hits) for w in V.residual(adv, rec, reg)) > len(rec.requests)


@pytest.mark.parametrize("name", CASES)
def test_each_value_goes_to_a5s_window(name):
    """A decision about arrival i at step k is charged no later than the ingress window of i or of any later arrival,
    nor than the first window that releases an output of step k or later (the window whose rows it changes first); the
    windows are nondecreasing in the replay's order."""
    t, rec, reg, adv = case(name, T=20)
    wins = V.residual(adv, rec, reg)
    order = V.arrival_order(adv, rec, reg.row_order)
    first = [min(s for s, st in enumerate(adv.steps) if any(adv.requests[x.request].id == rec.requests[k].id
                                                              for x in st.rows)) for k in order]
    _, decisions, _ = V._replay(rec, reg, reg.engine, order, lambda i, k: k >= first[i])
    released, at, kept = [None] * len(adv.steps), 0, [0] * len(adv.requests)
    for s, st in enumerate(adv.steps):
        for x in st.rows:
            q = adv.requests[x.request]
            if x.start + x.n == q.prompt_len + kept[x.request] and kept[x.request] < q.output_len:
                w = adv.release[at] // reg.T
                released[s] = w if released[s] is None else min(released[s], w)
                kept[x.request] += 1
                at += 1
    charged = [w for w, x in enumerate(wins) for _ in x.arrived]
    assert len(charged) == len(decisions) and charged == sorted(charged)
    for (i, k, _), w in zip(decisions, charged):
        assert w <= min(rec.requests[j].end for j in order[i:]) // reg.T
        assert all(r is None or w <= r for r in released[k:])


@pytest.mark.parametrize("name, n, early", [("sync-kv-pressure", 15, 5), ("async-kv-pressure", 17, 5)])
def test_the_replay_chain_carries_what_a5_charges_early(name, n, early):
    """At T = 20, A5 charges `early` of the trace's n decisions to a window before their step's (window 0, read in
    windows 1 and 2). The certifier's state chain (`replay_chain`) loads each window's charged decisions after the ones
    the last window left unread, each step reads its own at the window row's head, and `ReplayClose` accepts the
    session's end; each part's circuit accepts what its specification does. A window ends with every decision it
    holds that a later window's step reads still unread, which the load before the carry refused."""
    t, rec, reg, adv = case(name, T=20)
    wins = V.residual(adv, rec, reg)
    order = V.arrival_order(adv, rec, reg.row_order)
    first = [min(s for s, st in enumerate(adv.steps) if any(adv.requests[x.request].id == rec.requests[k].id
                                                              for x in st.rows)) for k in order]
    rp, decisions, _ = V._replay(rec, reg, reg.engine, order, lambda i, k: k >= first[i])
    W = V.windows(rec, reg.T)
    window, _ = V._charged(rec, reg, order, rp, W)
    charged = [w for w, x in enumerate(wins) for _ in x.arrived]
    assert len(decisions) == n and all(k < len(window) for _i, k, _b in decisions)
    read = [window[k] for _i, k, _b in decisions]
    assert sum(c < r for c, r in zip(charged, read)) == early
    steps = [sum(1 for w in window if w == v) for v in range(W)]
    S = RC.Shape(R=7, W=2, A=3, L=1024, NB=96, IW=4, Q=14, D=n, P=max(steps), K=14)
    cur, last, ends = RC.empty(S), RC.Window(), []
    loads, advances, s = [], [], 0
    for v in range(W):
        loaded = RC.load(S, cur, last, (), [int(b) for b in wins[v].arrived], steps[v])
        assert loaded is not None
        loads.append(RC.load_arguments(S, cur, last, (), [int(b) for b in wins[v].arrived], len(wins[v].arrived),
                                       steps[v], RC.encode(loaded[0], S), RC.encode_window(loaded[1], S)))
        cur, win = loaded
        for _u in range(S.P):
            bits = [int(b) for _i, k, b in decisions if k == s] if cur.steps else []
            arr, front = RC.front(S, win, cur)
            assert tuple(bits) == front[:len(bits)]
            stepped = RC.Row(free=cur.free)
            nxt = RC.advance(S, win, cur, True, stepped, arr, front, 0, len(bits))
            assert nxt is not None
            advances.append(RC.advance_arguments(S, win, cur, True, stepped, arr, front, 0, len(bits), RC.encode(nxt, S)))
            s += bool(cur.steps)
            cur = nxt
        ends.append((cur, win))
        last = win
    assert s == len(rp.steps) == len(window)
    unread = [len(w.decisions) - r.dh for r, w in ends]
    assert unread == [sum(c <= v < r for c, r in zip(charged, read)) for v in range(W)] and unread[0] == early
    assert [RC.close(S, r, w) for r, w in ends] == [not u for u in unread] and RC.close(S, *ends[-1])
    assert RC.evaluate_load(S, loads) == [1] * W
    assert all(o[0] == 1 for o in RC.evaluate_advance(S, advances))
    assert RC.evaluate_close(S, [RC.close_arguments(S, r, w) for r, w in ends]) == [int(not u) for u in unread]


def test_the_corpus_table(tmp_path, capsys):
    out = tmp_path / "table.json"
    (tmp_path / "c").mkdir()
    assert V.main([f"smallkv={_corpus(tmp_path / 'c')}", "--stops", "128009,128001,128008", "--jobs", "1",
                   "--out", str(out)]) == 0
    (row,) = json.loads(out.read_text())["rows"]
    assert row["held_out"]["round_trip"] and row["uniform"]["round_trip"]
    assert row["held_out"]["bits"] == sum(row["held_out"]["by_window"].values()) < row["v0"] // 10
    assert row["hits_sent"] > row["hits_short"] > 0
    assert row["order_entries"] == row["inversions"] == 0 <= row["held_out"]["order_bits"] < row["windows"] * 8
    assert "| smallkv | " in capsys.readouterr().out


def _tiny(windows: int):
    """Three requests into an engine of two sequences and an 8-token budget, so the replay asks about each arrival; the
    book fit on windows of a few decisions each, as a registration fits it on history."""
    e = R.Engine(max_num_seqs=2, max_num_batched_tokens=8, max_model_len=64, num_gpu_blocks=32, block_size=4)
    reqs = tuple(V.Request(f"u{k}", (10 + k,) * 6, 4, (5, 5, EOS), end, (end, end, end + 1))
                 for k, end in enumerate((0, 0, 3 * (windows - 1))))
    history = [[CV.Advice(arrived=bits)] * windows for n in range(4) for bits in product((False, True), repeat=n)]
    caps = CV.Caps(arrived=7, hits=1, hit_units=1, resumes=1, order=1, order_len=1, gap=1, index=1, lag=1)
    return V.CertifiedRequests(reqs), V.Registered(e, (EOS,), T=3, book=CV.fit_book(history, caps))


@pytest.mark.parametrize("w", (1, 2))
def test_the_strings_are_prefix_free(w):
    """Every bit string of up to 16 bits through `decode_bits`: no accepted one is a proper prefix of another, so
    #{accepted strings of at most c bits} ≤ 2^c (Kraft), LEAK-LEAN's `BitsCount` for the charge `bits_by_window`; and
    each accepted one is the encoding of the advice it decodes to, a well-formed I1 v0 (`advice.decode`'s refusals, which
    ADV-SPEC's validity circuit makes)."""
    rec, reg = _tiny(w)
    accepted = {}
    for n in range(17):
        for bits in product((0, 1), repeat=n):
            try:
                accepted[bits] = V.decode_bits(bits, rec, reg)
            except V.V1Error:
                pass
    assert len(set(accepted.values())) == len(accepted) >= 3
    strings = sorted(accepted)
    assert not any(b[:len(a)] == a for a, b in zip(strings, strings[1:]))
    assert sum(2.0 ** -len(s) for s in strings) <= 1
    for c in range(17):
        assert sum(len(s) <= c for s in strings) <= 2 ** c
    for s, adv in accepted.items():
        data = V.encode(adv, rec, reg)
        assert exact_bits(data, rec, reg) == bytearray(s) and sum(V.bits_by_window(data, rec, reg).values()) == len(s)
        assert I1.decode(I1.encode(adv)) == adv


def test_truncated_and_extended_strings_are_refused():
    t, rec, reg, adv = case("async-prefix-cache", T=20)
    bits = exact_bits(V.encode(adv, rec, reg), rec, reg)
    for n in range(len(bits)):
        with pytest.raises(V.V1Error):
            V.decode_bits(bits[:n], rec, reg)
    for b in (0, 1):
        with pytest.raises(V.V1Error, match="after the last window"):
            V.decode_bits(bits + bytearray([b]), rec, reg)
    data = V.encode(adv, rec, reg)
    bad = [data[:-1], data + b"\0"]
    if 8 * len(data) > len(bits):
        bad.append(data[:-1] + bytes([data[-1] | 1]))
    for x in bad:
        with pytest.raises(V.V1Error):
            V.decode(x, rec, reg)


def test_decode_is_total():
    """Random strings, every one-bit change of an encoding, adversarial frames (empty, all zeros,
    all ones, truncations, splices and extensions of an encoding), and the codes of edited residuals (a decision
    flipped, an order entry added): `V1Error` and nothing else, or an advice whose encoding they are."""
    rng = random.Random(5)
    caps = CV.Caps(arrived=64, hits=64, hit_units=64, resumes=1, order=4, order_len=1, gap=64, index=64, lag=1)
    t, rec, reg, adv = case("sync-kv-pressure", T=20, book=CV.Book.uniform(caps))
    data = V.encode(adv, rec, reg)
    wins = V.residual(adv, rec, reg)
    edits = [(w, dataclasses.replace(x, arrived=x.arrived[:j] + (not x.arrived[j],) + x.arrived[j + 1:]))
             for w, x in enumerate(wins) for j in range(len(x.arrived))]
    edits += [(0, dataclasses.replace(wins[0], order=((p, (1,)),))) for p in range(len(rec.requests) - 1)]
    tries = [bytes(rng.randrange(256) for _ in range(rng.randrange(40))) for _ in range(200)]
    tries += [bytes(b ^ (1 << j if k == i else 0) for k, b in enumerate(data)) for i in range(len(data)) for j in range(8)]
    tries += [b"", b"\0" * len(data), b"\xff" * len(data), b"\0" * 64, data * 2, data + b"\0", data + b"\xff",
              data[:len(data) // 2]]
    for _ in range(200):
        b = bytearray(data)
        for _ in range(rng.randrange(1, 6)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        tries.append(bytes(b[:rng.randrange(1, len(b) + 1)]))
    tries += [_string(wins[:w] + [x] + wins[w + 1:], reg) for w, x in edits]
    decoded = set()
    for x in tries:
        try:
            got = V.decode(x, rec, reg)
        except V.V1Error:
            continue
        except Exception as e:  # noqa: BLE001 - nothing but V1Error may leave decode
            raise AssertionError(f"decode raised {type(e).__name__}: {e}") from e
        decoded.add(x)
        assert type(got) is I1.Advice and V.encode(got, rec, reg) == x
        assert I1.decode(I1.encode(got)) == got
    assert len(decoded - {data}) >= 2


@pytest.mark.parametrize("book", ("uniform", "fit"))
def test_long_strings_are_prefix_free_and_self_delimiting(book):
    """Past the exhaustive 16 bits: each case's string under T = 4 (hundreds of bits, many
    windows) decodes to its advice, and is refused cut at any window's end but the last, cut by one bit, or extended
    by a byte; and no case's string is a proper prefix of another's that decodes under the other's record."""
    fit = _fit_cases() if book == "fit" else None
    corpus = []
    for name in CASES:
        t, rec, reg, adv = case(name, T=4, book=fit)
        data = V.encode(adv, rec, reg)
        corpus.append((exact_bits(data, rec, reg), data, rec, reg, adv))
    assert max(len(b) for b, *_ in corpus) > 100 and min(V.windows(rec, reg.T) for _, _, rec, reg, _ in corpus) > 3
    for bits, data, rec, reg, adv in corpus:
        assert V.decode_bits(bits, rec, reg) == adv
        ends = list(accumulate(V.bits_by_window(data, rec, reg)[w] for w in range(V.windows(rec, reg.T))))
        for n in sorted(set(ends[:-1]) | {1, len(bits) - 1}):
            if n < len(bits):
                with pytest.raises(V.V1Error):
                    V.decode_bits(bits[:n], rec, reg)
        for ext in (b"\0", b"\x80"):
            with pytest.raises(V.V1Error):
                V.decode(data + ext, rec, reg)
    for a, *_ in corpus:
        for b, _, rec, reg, _ in corpus:
            if len(a) < len(b) and b[:len(a)] == a:
                with pytest.raises(V.V1Error):
                    V.decode_bits(a, rec, reg)


def test_refuses_over_the_cap_and_off_the_record():
    t, rec, reg, adv = case("async-light", T=20)
    data = V.encode(adv, rec, reg)
    with pytest.raises(V.V1Error, match="cap"):
        V.decode(data, rec, dataclasses.replace(reg, cap_bits=8 * len(data) - 1))
    with pytest.raises(V.V1Error, match="pool"):
        V.encode(adv, rec, dataclasses.replace(reg, engine=dataclasses.replace(t.engine, num_gpu_blocks=2)))
    with pytest.raises(V.V1Error, match="bytes"):
        V.decode(list(data), rec, reg)
    moved = I1.Advice(adv.requests, adv.steps, adv.release[:-1] + (adv.release[-1] + 1,))
    with pytest.raises(V.V1Error, match="another advice"):
        V.encode(moved, rec, reg)
    s = next(k for k, st in enumerate(adv.steps) if len(st.rows) > 1 and k > 0
             and all(any(x.request == y.request for z in adv.steps[:k] for y in z.rows) for x in st.rows))
    swapped = I1.Advice(adv.requests, adv.steps[:s] + (I1.StepAdvice(adv.steps[s].rows[::-1]),) + adv.steps[s + 1:],
                        adv.release)
    with pytest.raises(V.V1Error, match="outside v1"):
        V.encode(swapped, rec, reg)
