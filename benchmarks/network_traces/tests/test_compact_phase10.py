"""Adversarial cases against the compact framing: the 8-byte record's finish field and ordinal, the proxy's "record
names another request" fault, and the client-side renderer (`compact_render`). On an honest compact run (B = 8,
`_Varied`) its certified records are rewritten and read back by `certifier_serve.certified_io`; the renderer is run on
the fixture's real Llama-3.2 vocab. Its findings are kept as regressions of their fixes:

- the finish rule lost "ending at its first EOS" under compact, so an unmarked record
  carrying an EOS id read as a token that more follow and was accepted, where the id-event rule refuses "a token after
  EOS"; `certified_io` now refuses an EOS id in any record but its response's stop, which
  carries the EOS id the model sampled (`test_compact_eos_phase12`);
- the proxy holds no finish state, so that rule is all there is: it certifies a record after a finish and an unmarked
  EOS id, and faults only a record of another request or one that isn't a record;
- two token streams render byte-identically (lone UTF-8 continuation bytes before the same token), so the text can't
  stand in for the ids; the render is a function of the cells and the client's own prompt;
- the renderer raised KeyError on an id outside the vocab, `done` closed a stream with
  no finish, and the default `request=None` took another request's record; each is now refused with ValueError, and
  `request` is required.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# test_certifier_serve puts the benchmark on the path
from test_certifier_serve import VARIED_TOKENS, _no_sleep, _Varied, _varied_runs, ar, cs
import compact_render as cr  # noqa: E402
from test_compact_render import FIXTURE, VOCAB  # noqa: E402
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.grid import Reason

B = 8
EOS_ID = "an EOS id outside a stop record"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    endpoint = _Varied(merges=lambda r: r % 2 == 1)
    args = argparse.Namespace(model=None, api_key_env=None, traffic="varied", requests=16, rate=2.0,
                              prompt_range="16:1024", tokens=VARIED_TOKENS, seed=3, probes=4, eos_ids=None, compact=B)
    tmp, _probed, report, plan, runs = _varied_runs(tmp_path_factory.mktemp("compact-p10"), endpoint,
                                                    lambda url: cs.plan_of(args, url, lambda: 0.0, _no_sleep),
                                                    ("honest",))
    setup = ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    ins = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(tmp / "honest" / "requests.records")]
    outs = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(tmp / "honest" / "responses.records")]
    return SimpleNamespace(plan=plan, setup=setup, ins=ins, outs=outs, check=runs["honest"]["check"])


def _rewrite(outs, stream: int, index: int, record: bytes):
    """`outs` with output `index` of `stream`'s response replaced by `record` (its cell, B bytes)."""
    seen, out = 0, []
    for rec in outs:
        grid = []
        for row in rec.grid:
            new = []
            for slot in row:
                got = None if slot is None else active.compact_record_data(slot)
                if got is not None and got[0] == stream:
                    if seen == index:
                        slot = active.compact_cell(record, B)
                    seen += 1
                new.append(slot)
            grid.append(new)
        out.append(SimpleNamespace(window=rec.window, grid=grid))
    assert seen > index
    return out


def _as_id_events(outs):
    """The same records as id events (`active.id_event`, 256-byte frames), each record's id as it is."""
    def event(slot):
        if slot is None:
            return None
        request, token, _ = active.compact_record_data(slot)
        return active.encode_frame(active.DATA, active.CANONICAL_STREAM, active.id_event(request, token),
                                   cs.FRAME_BYTES)
    return [SimpleNamespace(window=r.window, grid=[[event(s) for s in row] for row in r.grid]) for r in outs]


def test_an_eos_id_in_a_compact_response_is_refused_as_id_events_refuse_a_token_after_it(run):
    """Finding 3. The honest shim (`active_replay.compact_records`) makes every EOS id the stop record, so a cell that
    carries one and isn't marked a stop is never honest. A response whose first output is the model's EOS id, sent
    unmarked, then continues to its marked stop (or to its `COMPACT_LAST` at `max_tokens`), was accepted with finish
    `eos` (or `length`): it is refused, as the same records as id events are ("a token after EOS"). So is a response
    whose last output at `max_tokens` is the EOS id marked `COMPACT_LAST` alone, which as id events is an honest
    `eos`."""
    plan = run.plan
    eos_id = plan["varied"]["eos_token_ids"][0]
    honest = cs.certified_io(plan, run.setup, run.ins, run.outs)
    assert honest["accepted"] and run.check["verdict"] == "accepted"
    for finish in ("eos", "length"):
        r = next(r for r in honest["records"] if r["finish"] == finish and len(r["output"]) >= 2)
        forged = _rewrite(run.outs, r["stream"], 0, active.compact_record(r["stream"], eos_id))
        got = cs.certified_io(plan, run.setup, run.ins, forged)
        mine = next(x for x in got["records"] if x["stream"] == r["stream"])
        assert not got["accepted"] and got["violations"] == [[r["stream"], EOS_ID]] and mine["finish"] == EOS_ID
        assert mine["output"][0] == eos_id and len(mine["output"]) == len(r["output"])

        ids = {**plan, "compact": None}
        events = cs.certified_io(ids, run.setup, run.ins, _as_id_events(forged))
        assert not events["accepted"] and events["violations"] == [[r["stream"], "a token after EOS"]]
        assert cs.certified_io(ids, run.setup, run.ins, _as_id_events(run.outs))["accepted"]
    r = next(r for r in honest["records"] if r["finish"] == "length")
    last = _rewrite(run.outs, r["stream"], len(r["output"]) - 1, active.compact_record(r["stream"], eos_id, "length"))
    got = cs.certified_io(plan, run.setup, run.ins, last)
    assert not got["accepted"] and got["violations"] == [[r["stream"], EOS_ID]]
    assert cs.certified_io({**plan, "compact": None}, run.setup, run.ins, _as_id_events(last))["accepted"]


def test_the_proxy_holds_no_finish_so_the_finish_rule_is_all_there_is():
    """The ordinal is the connection's and the field is canonical, or the window is `malformed` (the PR's own case).
    Past that the proxy takes any record of its request: one after a `COMPACT_LAST` or a stop, an unmarked EOS id, and
    a stop on an id that isn't EOS, are each certified with no fault, so they rest on `certified_io` alone, which
    refuses each ("a token after its finish", the test above, and `test_compact_eos_phase12`)."""
    sys.path.insert(0, str(Path(active.__file__).parent / "tests"))
    from test_network_certifier_active import EGRESS, INGRESS, SYNC_SET, SYNCS

    egress = dataclasses.replace(EGRESS, frame_bytes=B)
    eos_id = FIXTURE["eos_token_ids"][0]
    for records in ([active.compact_record(1, 5, "length"), active.compact_record(1, 6)],
                    [active.compact_record(1, eos_id, "stop"), active.compact_record(1, 6)],
                    [active.compact_record(1, 5, "stop"), active.compact_record(1, 6)],
                    [active.compact_record(1, eos_id), active.compact_record(1, 6)]):
        proxy = active.Proxy(("127.0.0.1", 1), egress, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                             ordinals=True, compact=True)
        st = active._Stream(anchor=0)
        for k, record in enumerate(records):
            proxy._egress_record(1, st, k, record, 1)
        assert proxy.shaper.certifier._fault is None and len(proxy.frames) == 2
    proxy = active.Proxy(("127.0.0.1", 1), egress, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                         ordinals=True, compact=True)
    proxy._egress_record(1, active._Stream(anchor=0), 0, active.compact_record(2, 5), 1)
    assert proxy.shaper.certifier._fault == Reason.MALFORMED and not proxy.frames


def test_two_token_streams_render_byte_identically():
    """Yes, a server can make two token streams render identically, on the real vocab: ids 222, 231, 236 and 248 are
    lone UTF-8 continuation bytes, so each, then the same token, renders as a chunk of no text and then U+FFFD and the
    token's text. Four streams, one rendering: the client's text can't stand in for the ids, so only the ids' tie to
    the committed sampled ones (the token shim is accepted by `check`) refuses a substitution. An unmarked EOS id
    renders as a chunk of no text with no finish, like a held-back byte's, which is why `certified_io` refuses one."""
    stream = FIXTURE["streams"][0]
    then = stream["chunks"][0][1][0]
    fragments = [i for i in (222, 231, 236, 248) if VOCAB.bytes[i] in (b"\x80", b"\x89", b"\x8e", b"\x9a")]
    assert len(fragments) == 4

    def render(ids):
        r = cr.Renderer(VOCAB, stream["prompt"], FIXTURE["model"], request=1, id="cmpl-x", created=1)
        return b"".join(r.chunk(active.compact_cell(active.compact_record(1, t), B)) for t in ids)

    lines = {render([f, then]) for f in fragments}
    assert len(lines) == 1
    line = lines.pop()
    assert line.count(b"data: ") == 2 and b'"text":""' in line and "\ufffd".encode() in line
    r = cr.Renderer(VOCAB, stream["prompt"], FIXTURE["model"], request=1)
    assert r.text(active.compact_record(1, FIXTURE["eos_token_ids"][0])) == ("", None)
    assert r.text(active.compact_record(1, then))[1] is None


def test_the_renderer_refuses_an_unknown_id_an_unfinished_end_and_another_requests_record():
    """Finding 8. An id outside the vocab raised KeyError: it is ValueError, whatever the record's finish, with
    nothing rendered, so the stream's next record renders as on a fresh renderer. `done` closed a stream that had no
    finish: it refuses one, and gives `[DONE]` after the stop. The default `request=None` let a record of another
    request render into this one: `request` is required, and another's record is refused."""
    stream = FIXTURE["streams"][0]
    then = stream["chunks"][0][1][0]
    r = cr.Renderer(VOCAB, stream["prompt"], FIXTURE["model"], request=1)
    for finish in (None, "length", "stop"):
        with pytest.raises(ValueError):
            r.text(active.compact_record(1, 999_999, finish))
        assert r.finished is None
    with pytest.raises(ValueError):
        r.done()
    fresh = cr.Renderer(VOCAB, stream["prompt"], FIXTURE["model"], request=1)
    assert r.text(active.compact_record(1, then)) == fresh.text(active.compact_record(1, then))
    with pytest.raises(ValueError):
        r.done()
    assert r.text(active.compact_record(1, FIXTURE["eos_token_ids"][0], "stop")) == ("", "stop") and r.done() == cr.DONE
    with pytest.raises(TypeError):
        cr.Renderer(VOCAB, stream["prompt"], FIXTURE["model"])
    with pytest.raises(ValueError):
        cr.Renderer(VOCAB, stream["prompt"], FIXTURE["model"], request=1).text(active.compact_record(7, then))
