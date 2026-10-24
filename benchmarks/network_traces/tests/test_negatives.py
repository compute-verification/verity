"""The negatives harness (`negatives.py`): each tampering is a pure function on a run's artifacts, and the check it
names refuses it by its code.

At the certifier, on certifier-serve runs on an injected clock against a stub endpoint (`test_certifier_serve`'s): the
shim's `late`, `extra`, `drop` and `token`, certified by the certifier, and the prover's tampered head under its original
certificate. N2's verdict of record refuses late by the record's status, extra and drop by the counts (the served counts on
a served link) and the head by its signed digest; it accepts token, whose counts are honest's and whose one changed frame
only the I/O tie (IO-TIE, pending) reads. The advice's and the Commit's tamperings run on stand-ins of I1 and I3 as
drafted (ADV-SPEC and SERVE-COMMIT haven't landed): each changes exactly what its check reads and leaves the advice whole."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import re
import sys
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
def test_a_certified_token_that_isnt_the_endpoints_passes_n2_and_waits_for_the_tie(runs, profile):
    """token's one changed frame is certified, under honest's counts: N2 accepts every window, and only the I/O tie
    (IO-TIE, pending) can refuse it, which the registry says."""
    rs, verdicts = runs[profile]
    honest, token = _egress(rs["honest"]), _egress(rs["token"])
    assert token.counts == honest.counts and token.root != honest.root
    assert verdicts["token"]["accepted"] and verdicts["token"]["codes"] == []
    events = [ar.read_events(rs[s] / "server.events.gz") for s in ("honest", "token")]
    differ = [(a[1], a[2]) for a, b in zip(*events, strict=True) if a[4] != b[4]]
    assert differ == [(json.loads((rs["honest"] / "plan.json").read_text())["sessions"][0][0], 0)]
    assert not NG.judge("token", verdicts["token"])["refused"]
    n = NG.NEGATIVES["token"]
    assert n.code == "IoTie.token" and "IO-TIE" in n.waits


def test_every_negative_names_a_check_and_a_code_and_the_table_lists_each():
    rows = NG.table().splitlines()
    assert len(rows) == 2 + len(NG.NEGATIVES)
    for name, n in NG.NEGATIVES.items():
        assert n.family in ("io", "advice", "token", "circuit") and n.tampers in ("plan", "certified", "advice",
                                                                                   "commit")
        assert re.fullmatch(r"[A-Za-z0-9_]+(\.[a-z_-]+)?", n.code), n.code
        assert any(row.startswith(f"| `{name}` |") for row in rows)
    assert {n.family for n in NG.NEGATIVES.values()} == {"io", "advice", "token", "circuit"}
    assert [n for n in NG.NEGATIVES.values() if not n.waits and n.tampers in ("advice", "commit")] == []


# --- the advice: a stand-in of I1 as drafted ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RequestAdvice:
    id: str
    prompt_len: int
    output_len: int
    finish: str


@dataclasses.dataclass(frozen=True)
class Row:
    request: int
    kind: str
    start: int
    n: int
    cache_hit: int = 0
    preempted: bool = False


@dataclasses.dataclass(frozen=True)
class StepAdvice:
    rows: list


@dataclasses.dataclass(frozen=True)
class Advice:
    requests: list
    steps: list
    release: list


def _advice() -> Advice:
    """r0: a prompt of 5 in two prefill chunks (3, then 2 beside r1's whole prefill), 3 tokens out, the last at EOS;
    r1: a prompt of 2, 2 tokens out to its cap."""
    return Advice(
        requests=[RequestAdvice("r0", 5, 3, "eos"), RequestAdvice("r1", 2, 2, "length")],
        steps=[StepAdvice([Row(0, "prefill", 0, 3)]),
               StepAdvice([Row(0, "prefill", 3, 2), Row(1, "prefill", 0, 2)]),
               StepAdvice([Row(0, "decode", 5, 1), Row(1, "decode", 2, 1)]),
               StepAdvice([Row(0, "decode", 6, 1)])],
        release=[2, 3, 3, 5])


def _served(a: Advice) -> dict[str, list[int]]:
    """Each request's positions, in the order its rows schedule them."""
    out: dict[str, list[int]] = {q.id: [] for q in a.requests}
    for st in a.steps:
        for r in st.rows:
            out[a.requests[r.request].id] += list(range(r.start, r.start + r.n))
    return out


def _tokens(a: Advice) -> dict[str, int]:
    """Each request's tokens out by its rows: one at its prefill's last chunk, one at each decode."""
    out = {q.id: 0 for q in a.requests}
    for st in a.steps:
        for r in st.rows:
            q = a.requests[r.request]
            out[q.id] += r.kind == "decode" or r.start + r.n == q.prompt_len
    return out


def _shapes(a: Advice) -> list[list[tuple]]:
    """What a step's compiled Program reads of the advice: each row's kind, start, n and cache hit, by request id."""
    return [sorted((a.requests[r.request].id, r.kind, r.start, r.n, r.cache_hit) for r in st.rows) for st in a.steps]


def test_the_stand_in_advice_is_whole():
    a = _advice()
    assert _tokens(a) == {q.id: q.output_len for q in a.requests}
    assert _served(a) == {"r0": list(range(7)), "r1": list(range(3))}


def test_advice_length_changes_one_output_length_and_nothing_else():
    a = _advice()
    b = NG.output_length(a)
    assert [q.output_len for q in b.requests] == [4, 2] and b.steps == a.steps and b.release == a.release
    assert _tokens(b) != {q.id: q.output_len for q in b.requests}
    assert NG.output_length(a, 1, -1).requests[1].output_len == 1 and a == _advice()


def test_advice_missing_leaves_out_one_request_and_its_rows_and_keeps_the_others_whole():
    a = _advice()
    for i, gone in ((-1, "r1"), (0, "r0")):
        b = NG.missing_request(a, i)
        assert [q.id for q in b.requests] == [q.id for q in a.requests if q.id != gone]
        assert {k: v for k, v in _served(a).items() if k != gone} == _served(b)
        assert _tokens(b) == {q.id: q.output_len for q in b.requests}


def test_advice_extra_serves_one_request_more_beside_the_original_and_is_whole():
    a = _advice()
    b = NG.extra_request(a)
    assert [q.id for q in b.requests] == ["r0", "r1", "r0~extra"]
    assert _served(b)["r0~extra"] == _served(a)["r0"] and _tokens(b) == {q.id: q.output_len for q in b.requests}
    assert b.release == a.release


def test_advice_release_moves_one_step_one_bucket_and_keeps_the_order():
    a = _advice()
    b = NG.release_moved(a)
    assert [y - x for x, y in zip(a.release, b.release)] == [1, 0, 0, 0]
    assert b.release == sorted(b.release) and b.steps == a.steps and b.requests == a.requests
    assert NG.release_moved(a, step=3).release == [2, 3, 3, 6]


def test_advice_start_shifts_one_decode_row_and_changes_one_steps_shape():
    a = _advice()
    b = NG.start_shifted(a)
    assert [k for k, (x, y) in enumerate(zip(_shapes(a), _shapes(b))) if x != y] == [3]
    assert b.steps[3].rows[0].start == 7 and b.requests == a.requests and b.release == a.release
    with pytest.raises(NG.NotApplicable):
        NG.start_shifted(a, step=0)


def test_advice_chunk_serves_the_same_positions_in_two_other_step_shapes():
    """The circuit negative: every request is served the same positions with the same tokens out, so the I/O agrees;
    only the two chunks' steps compile to other Programs."""
    a = _advice()
    b = NG.chunk_split(a)
    assert _served(b) == _served(a) and _tokens(b) == _tokens(a)
    assert [k for k, (x, y) in enumerate(zip(_shapes(a), _shapes(b))) if x != y] == [0, 1]
    assert (b.steps[0].rows[0].n, b.steps[1].rows[0].start, b.steps[1].rows[0].n) == (2, 2, 3)
    unchunked = dataclasses.replace(a, steps=[StepAdvice([Row(0, "prefill", 0, 5)])] + a.steps[1:])
    with pytest.raises(NG.NotApplicable, match="no request's prefill is chunked"):
        NG.chunk_split(dataclasses.replace(unchunked, steps=[unchunked.steps[0], StepAdvice([Row(1, "prefill", 0, 2)])]
                                           + a.steps[2:]))


def test_an_advice_negative_waits_for_its_codec(tmp_path, monkeypatch):
    monkeypatch.setattr(NG, "advice_module", lambda: None)
    with pytest.raises(NG.NotApplicable, match="ADV-SPEC"):
        NG.make("advice-length", tmp_path, tmp_path / "out", advice=tmp_path / "advice.bin")
    with pytest.raises(NG.NotApplicable, match="--advice"):
        NG.make("advice-length", tmp_path, tmp_path / "out")


def test_an_advice_negative_writes_the_edited_advice_beside_a_copy_of_the_run(tmp_path, monkeypatch):
    """Through a codec of I1's shape (`encode`, `decode`): the tampered copy is the run and the edited advice."""
    class Codec:
        @staticmethod
        def encode(a: Advice) -> bytes:
            return json.dumps(dataclasses.asdict(a)).encode()

        @staticmethod
        def decode(b: bytes) -> Advice:
            d = json.loads(b)
            return Advice([RequestAdvice(**q) for q in d["requests"]],
                          [StepAdvice([Row(**r) for r in st["rows"]]) for st in d["steps"]], d["release"])

    monkeypatch.setattr(NG, "advice_module", lambda: Codec)
    run = tmp_path / "run"
    run.mkdir()
    (run / "certified.jsonl").write_text("{}\n")
    (tmp_path / "advice.bin").write_bytes(Codec.encode(_advice()))
    out = NG.make("advice-release", run, tmp_path / "out", advice=tmp_path / "advice.bin")
    assert (out / "certified.jsonl").read_text() == "{}\n"
    assert Codec.decode((out / "advice.bin").read_bytes()) == NG.release_moved(_advice())


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
