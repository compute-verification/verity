"""The negatives harness (`negatives.py`): each tampering is a pure function on a run's artifacts, and the check it
names refuses it by its code.

At the certifier, on certifier-serve runs on an injected clock against a stub endpoint (`test_certifier_serve`'s): the
shim's `late`, `extra`, `drop` and `token`, certified by the certifier, and the prover's tampered head under its original
certificate. N2's verdict of record refuses late by the record's status, extra and drop by the counts (the served counts on
a served link) and the head by its signed digest; it accepts token, whose counts are honest's and whose changed frames
only the I/O tie (IO-TIE, pending) reads. The advice's tamperings run on I1 v0 (ADV-SPEC's classes when they're
importable, a stand-in of their fields until then): each changes exactly what its check reads and leaves the advice
valid under v0's schedule rules (`_check`; the real classes refuse invalid advice on construction). The Commit's runs on
a stand-in of I3 as drafted."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from verity.core.protocols.certifier.certificate import Direction, Issuer

HERE = Path(__file__).resolve().parents[1]
TOKENS, REQUESTS = 4, 8
SESSION = bytes(range(32))


def _load(name: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


NG = _load("negatives")
cs, ar = NG.cs, NG.ar


async def _endpoint(reader, writer):
    """A streaming completions endpoint that answers at once: `max_tokens` chunk events and `[DONE]`, chunked."""
    head = await reader.readuntil(b"\r\n\r\n")
    body = json.loads(await reader.readexactly(int(re.search(rb"content-length: (\d+)", head, re.I).group(1))))
    events = [b'data: {"choices":[{"text":" t%d"}]}\n\n' % k for k in range(body["max_tokens"])] + [b"data: [DONE]\n\n"]
    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
                 + b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in events) + b"0\r\n\r\n")
    await writer.drain()
    writer.close()


def _run(run: Path, plan: dict, report: dict, url: str, upstream: tuple, certifier, challenger, strategy: str) -> None:
    """One served run of `strategy` on the injected clock: the shim serves `negatives.shim(strategy)`'s plan."""
    setup = ar.Setup(report)
    run.mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "calibration.json").write_text(json.dumps(report))
    served = plan
    if strategy != "honest":
        NG.make(strategy, run, run.parent / f"{strategy}.shim")
        served = json.loads((run.parent / f"{strategy}.shim" / "shim.json").read_text())

    async def after(proxies, clock):
        record = cs.deployment(proxies[0], setup, plan["windows"], session=SESSION, n2_link=cs.N2_LINK,
                               link=("127.0.0.1", 0), reveal=("127.0.0.1", 0), origin_unix=0.0, url=url)
        (run / "deployment.json").write_text(json.dumps(record))

    publish = cs.CertifyingPublisher(run, setup, Issuer(certifier, SESSION), cs.N2_LINK, threaded=False)
    ar.run_virtual(served, report, run, upstream, salt=True, publish=publish, after=after,
                   proxy_args={"challenger_key": challenger.public_key(), "signer": certifier,
                               "served": cs.served_profile(plan)})


def _runs(tmp: Path, profile: str, strategies) -> dict[str, Path]:
    challenger = cs.signer(tmp / "keys" / "challenger", 6)

    async def go():
        endpoint = await asyncio.start_server(_endpoint, "127.0.0.1", 0)
        upstream = endpoint.sockets[0].getsockname()[:2]
        url = "http://%s:%d" % upstream
        probed = await asyncio.to_thread(cs.probe, url, cs.request_body("stub", TOKENS), 2, clock=lambda: 0.0)
        report, plan = cs.served_plan(probed, REQUESTS, profile=profile)
        for strategy in strategies:
            certifier = cs.signer(tmp / "keys" / strategy, 6)
            try:
                await asyncio.to_thread(_run, tmp / strategy, plan, report, url, upstream, certifier, challenger,
                                        strategy)
            finally:
                certifier.close()
        endpoint.close()

    try:
        asyncio.run(go())
    finally:
        challenger.close()
    return {s: tmp / s for s in strategies}


STRATEGIES = ("honest", "late", "extra", "drop", "token")
PROFILES = ("planned", "served")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Per N2 profile: each strategy's run, the tampered head of honest's, and N2's verdict on each."""
    out = {}
    for profile in PROFILES:
        tmp = tmp_path_factory.mktemp(profile)
        rs = _runs(tmp, profile, STRATEGIES)
        rs["head"] = NG.make("head", rs["honest"], tmp / "head")
        out[profile] = rs, {s: NG.n2(r, clock=lambda: 0.0) for s, r in rs.items()}
    return out


def _egress(run: Path, window: int = 0):
    return cs.read_certified(run / "certified.jsonl")[0][(Direction.EGRESS, window)].head


@pytest.mark.parametrize("profile", PROFILES)
def test_honest_is_accepted_on_every_window_with_no_code(runs, profile):
    rs, verdicts = runs[profile]
    v = verdicts["honest"]
    assert v["accepted"] and v["codes"] == [] and all(r["accepted"] for r in v["windows"])
    assert len(v["windows"]) == json.loads((rs["honest"] / "plan.json").read_text())["windows"] and v["profile"] == profile


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("name", ["late", "extra", "drop", "head"])
def test_each_certifier_side_negative_is_refused_by_the_code_it_names(runs, profile, name):
    """Window 0's, where the tampering is, and no other window's: the served counts on a served link for extra and
    drop, the counts against the profile on a planned one."""
    _, verdicts = runs[profile]
    got = NG.judge(name, verdicts[name])
    assert got["refused"] and got["by_code"], got
    assert got["code"] == (NG.SERVED if profile == "served" and name in ("extra", "drop") else NG.NEGATIVES[name].code)
    assert [r["window"] for r in verdicts[name]["windows"] if r["codes"]] == [0]
    today = {c for c in NG.NEGATIVES[name].also if c.split(".")[0] in NG.CHECKS_TODAY}
    assert today <= set(verdicts[name]["codes"])


def test_drop_and_extra_certify_one_frame_fewer_and_one_more_than_honest(runs):
    rs, _ = runs["planned"]
    honest = sum(_egress(rs["honest"]).counts)
    assert sum(_egress(rs["drop"]).counts) == honest - 1
    assert sum(_egress(rs["extra"]).counts) == honest + 1


def test_a_tampered_head_fails_its_signed_digest_and_its_counts_under_the_original_certificate(runs):
    rs, verdicts = runs["planned"]
    honest, head = (cs.read_certified(rs[s] / "certified.jsonl")[0] for s in ("honest", "head"))
    assert [k for k in honest if honest[k] != head[k]] == [(Direction.EGRESS, 0)]
    assert head[(Direction.EGRESS, 0)].certificate == honest[(Direction.EGRESS, 0)].certificate
    assert verdicts["head"]["windows"][0]["codes"] == [f"{NG.HEAD}.counts", f"{NG.HEAD}.signed", "audit.signature"]


@pytest.mark.parametrize("profile", PROFILES)
def test_certified_tokens_that_arent_the_endpoints_pass_n2_and_wait_for_the_tie(runs, profile):
    """token's changed frames, every token of the responses to window 0's requests, are certified under honest's
    counts: N2 accepts every window, and only the I/O tie (IO-TIE, pending) can refuse them, which the registry says."""
    rs, verdicts = runs[profile]
    honest, token = _egress(rs["honest"]), _egress(rs["token"])
    assert token.counts == honest.counts and token.root != honest.root
    assert verdicts["token"]["accepted"] and verdicts["token"]["codes"] == []
    plan = json.loads((rs["honest"] / "plan.json").read_text())
    T = ar.Setup(json.loads((rs["honest"] / "calibration.json").read_text())).egress_params.buckets
    window0 = {q["session"] for q in plan["requests"] if q["enqueue"] // T == 0}
    assert set(json.loads((rs["token"].parent / "token.shim" / "shim.json").read_text())["token"]) == window0 != set()
    events = [ar.read_events(rs[s] / "server.events.gz") for s in ("honest", "token")]
    differ = {(a[1], a[2]) for a, b in zip(*events, strict=True) if a[4] != b[4]}
    tokens = {(e[1], e[2]) for e in events[0] if e[1] in window0 and ar.event_data(ar.event_body(e[4]))}
    assert differ == tokens and len(tokens) == TOKENS * len(window0)
    assert not NG.judge("token", verdicts["token"])["refused"]
    n = NG.NEGATIVES["token"]
    assert n.code == NG.TIE and "IO-TIE" in n.waits


def test_the_suite_reports_each_negative_for_the_one_command(runs, tmp_path):
    """From a rehearsal's runs: each certifier-side negative refused by its code, token passing N2 (its check pending),
    the head made from honest's, and the advice and Commit negatives not run, saying what they need."""
    rs, _ = runs["planned"]
    rows = NG.suite({s: rs[s] for s in STRATEGIES}, tmp_path, clock=lambda: 0.0)
    assert list(rows) == list(NG.NEGATIVES)
    assert {k for k, r in rows.items() if r.get("ok")} == {"late", "extra", "drop", "head", "token"}
    assert rows["head"]["run"] == str(tmp_path / "head") and rows["head"]["refused_windows"] == [0]
    assert rows["token"]["n2_accepted"] and rows["token"]["waits"] == ["IO-TIE", "CERT-VARLEN"]
    assert all(not r["ran"] and ("--advice" in r["why"] or "--commit" in r["why"])
               for r in rows.values() if r["tampers"] in ("advice", "commit"))
    assert not NG.suite({"honest": rs["honest"]}, tmp_path / "again", clock=lambda: 0.0)["late"]["ran"]


def test_every_negative_names_a_check_and_a_code_and_the_table_lists_each():
    rows = NG.table().splitlines()
    assert len(rows) == 2 + len(NG.NEGATIVES)
    for name, n in NG.NEGATIVES.items():
        assert n.family in ("io", "advice", "token", "circuit") and n.tampers in ("plan", "certified", "frames",
                                                                                   "advice", "commit")
        assert re.fullmatch(r"[A-Za-z0-9_]+(\.[a-z_-]+)?", n.code), n.code
        assert any(row.startswith(f"| `{name}` |") for row in rows)
    assert {n.family for n in NG.NEGATIVES.values()} == {"io", "advice", "token", "circuit"}
    assert [n for n in NG.NEGATIVES.values() if not n.waits and n.tampers in ("advice", "commit")] == []


def test_a_frame_with_one_byte_changed_no_longer_opens_under_the_records_frames_root(tmp_path):
    """On the certificate's own tree (`certificate.frames_root`, `open_frame`, `verify_frame`): every honest frame opens
    under the root, the changed one doesn't, and no other frame changes. The run keeps no frames, so `make` says what
    the negative waits for."""
    from verity.core.protocols.certifier.certificate import frames_root, open_frame, verify_frame
    from verity.core.protocols.certifier.grid import GridParams

    params = GridParams(buckets=4, allocation=2, frame_bytes=64, bucket_ms=100, queue_capacity=8)
    frames = [bytes([k]) * 64 for k in range(5)]
    root = frames_root(frames, params)
    bad = NG.frame_byte(frames, 2)
    assert [k for k, (a, b) in enumerate(zip(frames, bad)) if a != b] == [2] and bad[2][-1] == 3
    assert all(verify_frame(root, params, k, f, open_frame(frames, params, k)) for k, f in enumerate(frames))
    assert not verify_frame(root, params, 2, bad[2], open_frame(frames, params, 2))
    assert not verify_frame(root, params, 2, bad[2], open_frame(bad, params, 2))
    with pytest.raises(NG.NotApplicable, match="WARDEN"):
        NG.make("frame", tmp_path, tmp_path / "out")


# --- the advice: I1 v0, or a stand-in of its fields until ADV-SPEC lands -----------------------------------------------

@dataclasses.dataclass(frozen=True)
class RequestAdvice:
    id: str
    prompt_len: int
    output_len: int
    finish: str


@dataclasses.dataclass(frozen=True)
class Source:
    request: int
    n: int


@dataclasses.dataclass(frozen=True)
class Row:
    request: int
    kind: str
    start: int
    n: int
    cache_hit: int = 0
    preempted: bool = False
    sources: tuple = ()


@dataclasses.dataclass(frozen=True)
class StepAdvice:
    rows: tuple


@dataclasses.dataclass(frozen=True)
class Advice:
    requests: tuple
    steps: tuple
    release: tuple


V = NG.advice_module() or sys.modules[__name__]


def _advice(rows=None, release=(3, 3, 3, 3, 5)):
    """r0: a prompt of 5 in two prefill chunks (3, then 2 beside r1's whole prefill), 3 tokens out, the last at EOS;
    r1: a prompt of 2, 2 tokens out to its cap. Released: the first four outputs in bucket 3, r0's last in 5."""
    rows = rows or [[(0, "prefill", 0, 3)], [(0, "prefill", 3, 2), (1, "prefill", 0, 2)],
                    [(0, "decode", 5, 1), (1, "decode", 2, 1)], [(0, "decode", 6, 1)]]
    return V.Advice((V.RequestAdvice("r0", 5, 3, "eos"), V.RequestAdvice("r1", 2, 2, "length")),
                    tuple(V.StepAdvice(tuple(V.Row(*r) for r in step)) for step in rows), tuple(release))


def _check(a) -> list[tuple[int, str, int]]:
    """The advice's kept outputs (step, request id, index), once it's held to I1 v0's schedule rules (`advice._replay`,
    which the real module also runs on construction): requests in order of first row, a request's first row at its
    cache hit, each later row where the last left off (a resume at its hit), positions within the known tokens, `kind`
    decode exactly when n = 1 past the prompt, a cache hit's sources KV their requests computed by its step, every
    request keeping its `output_len`, and a release bucket a kept output, in order within each request."""
    R = len(a.requests)
    computed, known, kept, first, out = [0] * R, [q.prompt_len for q in a.requests], [0] * R, [], []
    made: list[set[int]] = [set() for _ in range(R)]
    for s, st in enumerate(a.steps):
        assert st.rows and len({r.request for r in st.rows}) == len(st.rows)
        for row in st.rows:
            r, q = row.request, a.requests[row.request]
            assert sum(x.n for x in row.sources) == row.cache_hit
            if r == len(first):
                first.append(s)
                assert not row.preempted and row.start == row.cache_hit
            else:
                assert r < len(first) and row.start == (row.cache_hit if row.preempted else computed[r])
                assert row.preempted or not row.cache_hit
            end = row.start + row.n
            assert row.n >= 1 and end <= known[r]
            assert row.kind == ("decode" if row.n == 1 and row.start >= q.prompt_len else "prefill")
            computed[r] = end
            made[r] |= set(range(row.start, end))
            if end == known[r] and kept[r] < q.output_len:
                out.append((s, q.id, kept[r]))
                kept[r], known[r] = kept[r] + 1, known[r] + 1
        for row in st.rows:
            at = 0
            for x in row.sources:
                assert x.request < len(first) and set(range(at, at + x.n)) <= made[x.request]
                assert x.request != row.request or row.preempted
                at += x.n
    assert len(first) == R and kept == [q.output_len for q in a.requests] and len(a.release) == len(out)
    for rid in {q.id for q in a.requests}:
        mine = [b for (_, i, _), b in zip(out, a.release) if i == rid]
        assert mine == sorted(mine)
    return out


def _served(a) -> dict[str, list[int]]:
    """Each request's positions, in the order its rows schedule them."""
    out: dict[str, list[int]] = {q.id: [] for q in a.requests}
    for st in a.steps:
        for r in st.rows:
            out[a.requests[r.request].id] += list(range(r.start, r.start + r.n))
    return out


def _tokens(a) -> dict[str, int]:
    """Each request's kept outputs by its rows (`_check`)."""
    out = {q.id: 0 for q in a.requests}
    for _, q, _ in _check(a):
        out[q] += 1
    return out


def _shapes(a) -> list[list[tuple]]:
    """What a step's compiled Program reads of the advice: each row's kind, start, n and cache hit, by request id."""
    return [sorted((a.requests[r.request].id, r.kind, r.start, r.n, r.cache_hit) for r in st.rows) for st in a.steps]


def test_the_advice_is_whole():
    a = _advice()
    assert _check(a) == [(1, "r0", 0), (1, "r1", 0), (2, "r0", 1), (2, "r1", 1), (3, "r0", 2)]
    assert _served(a) == {"r0": list(range(7)), "r1": list(range(3))}


def test_every_advice_negative_is_valid_advice_that_round_trips_through_the_codec():
    a = _advice()
    for name, n in NG.NEGATIVES.items():
        if n.tampers == "advice":
            b = n.make.edit(a)
            assert b != a and _check(b), name
            if V is not sys.modules[__name__]:
                assert V.decode(V.encode(b)) == b, name
    assert a == _advice()


def test_advice_length_keeps_one_output_more_or_fewer_and_stays_whole():
    """One more: a decode row past r0's last token, in a step of its own (r0 keeps its last in the last step), released
    with the output before it; r1's, a decode row beside r0's last. One fewer: r1's last kept sample discarded, its
    row kept. A row past an EOS keeps the sample that one more needs, and goes with the one fewer."""
    a = _advice()
    b = NG.output_length(a)
    assert [q.output_len for q in b.requests] == [4, 2] and b.steps[:4] == a.steps
    assert b.steps[4].rows == (V.Row(0, "decode", 7, 1),) and b.release == (3, 3, 3, 3, 5, 5)
    assert _tokens(b) == {"r0": 4, "r1": 2}
    c = NG.output_length(a, 1, -1)
    assert c.requests[1].output_len == 1 and c.steps == a.steps and c.release == (3, 3, 3, 5)
    assert _tokens(c) == {"r0": 3, "r1": 1}
    d = NG.output_length(a, 1)
    assert d.steps[3].rows == (V.Row(0, "decode", 6, 1), V.Row(1, "decode", 3, 1)) and _tokens(d) == {"r0": 3, "r1": 3}
    assert d.release == (3, 3, 3, 3, 5, 5)
    past = _advice([[(0, "prefill", 0, 3)], [(0, "prefill", 3, 2), (1, "prefill", 0, 2)],
                    [(0, "decode", 5, 1), (1, "decode", 2, 1)], [(0, "decode", 6, 1), (1, "decode", 3, 1)]])
    assert _check(past) == _check(a)
    e = NG.output_length(past, 1)
    assert e.steps == past.steps and _tokens(e) == {"r0": 3, "r1": 3} and e.release == (3, 3, 3, 3, 5, 5)
    f = NG.output_length(past, 1, -1)
    assert f.steps == c.steps and f.release == c.release
    with pytest.raises(NG.NotApplicable):
        NG.output_length(c, 1, -1)
    assert a == _advice()


def test_advice_missing_leaves_out_one_request_and_its_rows_and_keeps_the_others_whole():
    a = _advice()
    for i, gone, release in ((-1, "r1", (3, 3, 5)), (0, "r0", (3, 3))):
        b = NG.missing_request(a, i)
        assert [q.id for q in b.requests] == [q.id for q in a.requests if q.id != gone] and b.release == release
        assert {k: v for k, v in _served(a).items() if k != gone} == _served(b)
        assert _tokens(b) == {q.id: q.output_len for q in b.requests}
    with pytest.raises(NG.NotApplicable, match="reads request 0's KV"):
        NG.missing_request(NG.start_shifted(a), 0)


def test_advice_extra_serves_one_request_more_beside_the_original_and_is_whole():
    a = _advice()
    b = NG.extra_request(a)
    assert [q.id for q in b.requests] == ["r0", "r0~extra", "r1"]
    assert _served(b)["r0~extra"] == _served(a)["r0"] and _tokens(b) == {q.id: q.output_len for q in b.requests}
    assert b.release == (3, 3, 3, 3, 3, 3, 5, 5)
    assert [q.id for q in NG.extra_request(a, 1).requests] == ["r0", "r1", "r1~extra"]
    shifted = NG.start_shifted(a)
    c = NG.extra_request(shifted, 1)
    assert c.steps[1].rows[2].sources == shifted.steps[1].rows[1].sources == (V.Source(0, 1),) and _check(c)


def test_advice_release_moves_one_output_one_bucket_and_keeps_the_order():
    """By default the first kept output (r0's, at step 1), the outputs after it moved up with it."""
    a = _advice()
    b = NG.release_moved(a)
    assert b.release == (4, 4, 4, 4, 5) and b.steps == a.steps and b.requests == a.requests
    assert NG.release_moved(a, -1).release == (3, 3, 3, 3, 6)
    assert NG.release_moved(a, 2, by=3).release == (3, 3, 6, 6, 6)


def test_advice_start_claims_a_cache_hit_with_its_frames_kept_and_changes_one_steps_shape():
    """r0's first row has no KV of another request to read at step 0, so r1's moves: its first position read from r0's
    KV. N2 reads a row's end only, so it accepts the shifted advice (the same outputs); the step's Program changes. One
    position fewer undoes it."""
    a = _advice()
    b = NG.start_shifted(a)
    assert [k for k, (x, y) in enumerate(zip(_shapes(a), _shapes(b))) if x != y] == [1]
    assert b.steps[1].rows[1] == V.Row(1, "prefill", 1, 1, 1, False, (V.Source(0, 1),))
    assert b.requests == a.requests and b.release == a.release and _check(b) == _check(a)
    assert NG.start_shifted(b, 1, by=-1) == a
    with pytest.raises(NG.NotApplicable):
        NG.start_shifted(a, 0)


def test_advice_chunk_serves_the_same_positions_in_other_step_shapes():
    """The circuit negative: every request is served the same positions with the same outputs in the same buckets, so
    the I/O agrees; only the chunks' steps compile to other Programs. An unchunked prefill is split in two."""
    a = _advice()
    b = NG.chunk_split(a)
    assert _served(b) == _served(a) and _check(b) == _check(a) and b.release == a.release
    assert [k for k, (x, y) in enumerate(zip(_shapes(a), _shapes(b))) if x != y] == [0, 1]
    assert (b.steps[0].rows[0].n, b.steps[1].rows[0].start, b.steps[1].rows[0].n) == (2, 2, 3)
    whole = _advice([[(0, "prefill", 0, 5)], [(1, "prefill", 0, 2)], [(0, "decode", 5, 1), (1, "decode", 2, 1)],
                     [(0, "decode", 6, 1)]], (2, 3, 3, 3, 5))
    c = NG.chunk_split(whole)
    assert [st.rows for st in c.steps[:2]] == [(V.Row(0, "prefill", 0, 4),), (V.Row(0, "prefill", 4, 1),)]
    assert c.steps[2:] == whole.steps[1:] and c.release == whole.release and _served(c) == _served(whole)
    one = V.Advice((V.RequestAdvice("x", 1, 1, "eos"),), (V.StepAdvice((V.Row(0, "prefill", 0, 1),)),), (0,))
    with pytest.raises(NG.NotApplicable, match="no prefill row of two tokens"):
        NG.chunk_split(one)


FRAMING = (1, 1, 2, 2, 2)  # the link's request frames by prompt length: r0's prompt of 5 two, r1's of 2 one


def _n2(b, received=(3,)):
    """`b` as N2-ADVICE reads it (`n2_advice`), under `FRAMING`."""
    return NG.n2_advice(b, FRAMING, list(received))


def test_the_declared_received_is_the_advices_requests_frames_under_the_links_framing():
    """`received` as a prover declares it beside the advice: the honest run's, less or more the frames of a request the
    advice leaves out or adds (a copy's are its original's), in the first window that receives any."""
    a = _advice()
    assert NG.link_received(a, FRAMING, [3]) == [3] and NG.link_received(a, FRAMING, [0, 3]) == [0, 3]
    assert NG.link_received(NG.extra_request(a), FRAMING, [3]) == [5]
    assert NG.link_received(NG.missing_request(a), FRAMING, [3]) == [2]


def test_n2s_advised_counts_refuse_the_advice_negatives_and_accept_the_step_units():
    """Against N2-ADVICE's `advised` (skipped until it lands), on heads an honest run of the advice certifies: the
    honest advice is accepted; length, missing, extra and release are refused by the advised counts, each well formed;
    start and chunk are accepted, their frames honest's, so only the step unit can refuse them; a `received` that
    isn't the requests' frames, and an extra request whose frames pass the egress allocation R in a bucket, are
    refused by the view."""
    AD = pytest.importorskip(NG.ADVISED_MODULE)
    from verity.core.protocols.certifier.schedule import ClockSync, SyncSet

    T, R, RIN, OWN, syncs, sigma = 8, 8, 8, 0, [ClockSync(1, Fraction(1))], SyncSet((1,), (Fraction(1),))
    a = _advice()
    heads = NG.honest_heads(_n2(a), T, R, syncs, sigma, FRAMING)
    assert [e for _, e in heads] == [[0, 0, 0, 4, 0, 1, 0, 0]]

    def codes(link) -> list[str]:
        return sorted({c for w, (i, e) in enumerate(heads)
                       for c in NG.advised_codes(link, w, syncs, sigma, FRAMING, T, R, RIN, OWN, i, e)})

    assert codes(_n2(a)) == []
    for name, n in NG.NEGATIVES.items():
        if n.tampers == "advice":
            assert codes(_n2(n.make.edit(a))) == ([] if n.code.startswith("StepUnit") else [n.code]), name
    assert codes(AD.LinkAdvice(a, (4,))) == [f"{NG.ADVISED}.view"]
    assert NG.advised_codes(_n2(NG.extra_request(a)), 0, syncs, sigma, FRAMING, T, 4, RIN, OWN,
                            *heads[0]) == [f"{NG.ADVISED}.view"]


def test_the_advice_suite_makes_each_advice_negative_and_names_n2s_codes_on_it():
    """`negatives.py advice` on a run's advice, here the stand-in's through I1's own codec (skipped until both I1 and
    N2-ADVICE are importable): honest accepted, each advice negative valid advice that round-trips, refused by the code
    it names or, a step-unit negative, accepted by N2."""
    pytest.importorskip(NG.ADVISED_MODULE)
    if V is sys.modules[__name__]:
        pytest.skip(f"{NG.ADVICE_MODULE} (I1) isn't importable")
    from verity.core.protocols.certifier.schedule import ClockSync

    rows = NG.advice_suite(_advice(), T=8, R=8, syncs=[ClockSync(1, Fraction(1))], request_frames=FRAMING,
                           received=[3])
    assert rows["honest"] == {"bits": V.total_bits(_advice()), "codes": [], "ok": True}
    made = {k: r for k, r in rows.items() if k != "honest"}
    assert set(made) == {k for k, n in NG.NEGATIVES.items() if n.tampers == "advice"}
    assert all(r["ran"] and r["valid"] and r["ok"] for r in made.values()), made
    assert {k for k, r in made.items() if r["codes"]} == {"advice-length", "advice-missing", "advice-extra",
                                                           "advice-release"}


def test_an_advice_that_releases_in_its_requests_own_bucket_runs_no_negative_until_shifted(tmp_path):
    """N2-ADVICE's floor wants every output a bucket after its request's: an honest advice that releases in bucket 0
    is refused by the view, which its row says, and no negative runs; `advice_run`'s shift moves every release later,
    and then honest is accepted and the negatives run."""
    pytest.importorskip(NG.ADVISED_MODULE)
    if V is sys.modules[__name__]:
        pytest.skip(f"{NG.ADVICE_MODULE} (I1) isn't importable")
    from verity.core.protocols.certifier.schedule import ClockSync

    early = _advice(release=(0, 0, 0, 0, 5))
    rows = NG.advice_suite(early, T=8, R=8, syncs=[ClockSync(1, Fraction(1))], request_frames=FRAMING, received=[3])
    assert rows["honest"]["codes"] == [f"{NG.ADVISED}.view"] and "before its request" in rows["honest"]["why"]
    assert all(not r["ran"] for k, r in rows.items() if k != "honest")
    (tmp_path / "advice.bin").write_bytes(V.encode(early))
    assert not NG.advice_run(tmp_path / "advice.bin", T=8, R=8)["negatives"]["honest"]["ok"]
    doc = NG.advice_run(tmp_path / "advice.bin", T=8, R=8, shift=1)
    assert doc["shift"] == 1 and doc["negatives"]["honest"]["ok"]
    assert all(r["ran"] and r["ok"] for k, r in doc["negatives"].items() if k != "honest"), doc


def test_an_advice_negative_waits_for_its_codec(tmp_path, monkeypatch):
    monkeypatch.setattr(NG, "advice_module", lambda: None)
    with pytest.raises(NG.NotApplicable, match="ADV-SPEC"):
        NG.make("advice-length", tmp_path, tmp_path / "out", advice=tmp_path / "advice.bin")
    with pytest.raises(NG.NotApplicable, match="--advice"):
        NG.make("advice-length", tmp_path, tmp_path / "out")


class _Codec:
    """A codec of I1's shape (`encode`, `decode`) for the stand-in: JSON."""

    @staticmethod
    def encode(a) -> bytes:
        return json.dumps(dataclasses.asdict(a)).encode()

    @staticmethod
    def decode(b: bytes):
        d = json.loads(b)
        rows = [[Row(**(r | {"sources": tuple(Source(**x) for x in r["sources"])})) for r in st["rows"]]
                for st in d["steps"]]
        return Advice(tuple(RequestAdvice(**q) for q in d["requests"]), tuple(StepAdvice(tuple(r)) for r in rows),
                      tuple(d["release"]))

    @staticmethod
    def total_bits(a) -> int:
        return 8 * len(_Codec.encode(a))


def test_an_advice_negative_writes_the_edited_advice_beside_a_copy_of_the_run(tmp_path, monkeypatch):
    """Through I1's codec (the stand-in's, until it lands): the tampered copy is the run and the edited advice."""
    codec = V if V is not sys.modules[__name__] else _Codec
    monkeypatch.setattr(NG, "advice_module", lambda: codec)
    run = tmp_path / "run"
    run.mkdir()
    (run / "certified.jsonl").write_text("{}\n")
    (tmp_path / "advice.bin").write_bytes(codec.encode(_advice()))
    out = NG.make("advice-start", run, tmp_path / "out", advice=tmp_path / "advice.bin")
    assert (out / "certified.jsonl").read_text() == "{}\n"
    assert codec.decode((out / "advice.bin").read_bytes()) == NG.start_shifted(_advice())


def test_the_suites_advice_rows_are_n2_advices_codes_on_the_given_advice(runs, tmp_path, monkeypatch):
    """With --advice, each advice negative's row in the one command's report is `advice_run`'s on it (N2-ADVICE's codes,
    not N2's on the unchanged certified records), its tampered advice beside a copy of honest's run; until N2-ADVICE
    lands, not run, saying so."""
    codec = V if V is not sys.modules[__name__] else _Codec
    monkeypatch.setattr(NG, "advice_module", lambda: codec)
    (tmp_path / "advice.bin").write_bytes(codec.encode(_advice()))
    rs, _ = runs["planned"]
    rows = NG.suite({"honest": rs["honest"]}, tmp_path / "work", advice=tmp_path / "advice.bin", clock=lambda: 0.0)
    advice = {k: r for k, r in rows.items() if r["tampers"] == "advice"}
    assert advice and all(r["copy"] == str(tmp_path / "work" / k) for k, r in advice.items())
    assert all("run" not in r for r in advice.values())
    try:
        importlib.import_module(NG.ADVISED_MODULE)
    except ModuleNotFoundError:
        assert all(not r["ran"] and "N2-ADVICE" in r["why"] and "ok" not in r for r in advice.values())
        return
    assert all(r["ran"] and r["valid"] and r["ok"] and r["honest_codes"] == [] for r in advice.values()), advice
    assert {k for k, r in advice.items() if r["codes"]} == {"advice-length", "advice-missing", "advice-extra",
                                                           "advice-release"}


# --- the Commit record: a stand-in of I3 as drafted --------------------------------------------------------------------

def test_wrong_token_changes_one_committed_sampled_id_and_keeps_the_registered_root(tmp_path):
    commit = {"root": "ab" * 64, "steps": [{"rows": [{"user": "r0", "positions": [5]}, {"user": "r1", "positions": [2]}],
                                            "c00.input/ids": [11, 21], NG.SAMPLED: [12, 22]}]}
    out = NG.wrong_token(commit, step=0, row=1)
    assert out["steps"][0][NG.SAMPLED] == [12, 23] and out["root"] == commit["root"]
    assert commit["steps"][0][NG.SAMPLED] == [12, 22]
    assert NG.wrong_token(commit, row=0, vocab=13)["steps"][0][NG.SAMPLED] == [0, 22]
    (tmp_path / "commit.json").write_text(json.dumps(commit))
    (tmp_path / "run").mkdir()
    written = NG.make("wrong-token", tmp_path / "run", tmp_path / "out", commit=tmp_path / "commit.json")
    assert json.loads((written / "commit.json").read_text()) == NG.wrong_token(commit)
