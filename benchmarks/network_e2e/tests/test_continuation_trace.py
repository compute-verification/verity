"""The continuation rule (`verity.protocols.certifier.verifier.continuation`) on a recorded varied-plan run: the token
run's ingress and egress records (`tests/data/token`: 16 execution phases of 60 buckets, 40 id requests of a varied
plan through the certifying proxy), read through the rehearsal's certified I/O (`certifier_serve.certified_io`), each
request's stream its ordinal and its place among the released streams its release number. The run is the token
negative's: its shim changed ids on their way to the proxy, and the rule reads what the records hold as released.

The proxy's records hold no end cell: a response there ends at its `max_tokens`-th id or its first end-of-sequence
id, which only the reference reads. So at each close every request released before it is stamped, its e the id frames
of its stream in the egress records before the close, counted from the frames themselves, and the reference's split
does all the work: 40 stamps at the last close, no continuation. Nine closes have a continuation, thirteen in all:
four of responses cut with ids already released (16, 3, 18 and 3 of them), nine of requests released before the close
whose first id comes after it (e = 0). Each is its certified request with those ids after its prompt and `max_tokens`
− e left; the records before the close alone give the same stamps and continuations; and a stamp dropped (of a
continued response or a finished one), doubled, moved out of release order, added for an unreleased request, or of
another N or e is refused by its name. No response of this run continues at two closes:
`test_network_certifier_continuation.py` holds that case, and the end cells, on the network certifier's own records."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.protocols.certifier.verifier import continuation as C

HERE = Path(__file__).resolve().parents[1]
TOKEN = HERE / "tests" / "data" / "token"
T = 60
#: each close with a continuation, and its continuations as (q, N, e)
CONTINUED = {60: ((2, 1, 16),), 120: ((3, 2, 0),), 360: ((13, 12, 0), (14, 13, 0)),
             480: ((19, 18, 3), (20, 19, 0), (21, 20, 0)), 540: ((22, 21, 18),), 600: ((24, 23, 0),),
             660: ((27, 26, 0), (28, 27, 0)), 720: ((34, 33, 0),), 840: ((40, 39, 3),)}


def _load():
    spec = importlib.util.spec_from_file_location("network_e2e", HERE / "e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chain = _load().chain
ar, cs = chain.ar, chain.NG.cs


def _responses(io: dict) -> list[C.Response]:
    """The certified I/O's released requests as the rule's responses: N a stream's place among the released streams,
    each id frame at its egress bucket, and no end cell, which the proxy's records don't have."""
    rows = sorted((r for r in io["records"] if r["release"] is not None), key=lambda r: r["stream"])
    return [C.Response(r["stream"], n, r["release"],
                       C.Request(**{k: r[k] for k in C.Request._fields})._replace(prompt=tuple(r["prompt"])),
                       tuple((t, x) for (_, t), x in zip(r["frames"], r["output"], strict=True)))
            for n, r in enumerate(rows)]


@pytest.fixture(scope="module")
def run() -> SimpleNamespace:
    plan, report = (json.loads((TOKEN / f).read_text()) for f in ("plan.json", "calibration.json"))
    setup = ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    ins = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(TOKEN / "requests.records")]
    outs = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(TOKEN / "responses.records")]
    io = cs.certified_io(plan, setup, ins, outs)
    assert io["accepted"] and plan["name"] == "varied" and (e.buckets, i.buckets, len(ins), len(outs)) == (T, T, 16, 16)
    assert all(rec.status.complete for rec in ins + outs)
    return SimpleNamespace(plan=plan, setup=setup, ins=ins, outs=outs, io=io, responses=_responses(io),
                           eos=tuple(plan["varied"]["eos_token_ids"]))


def test_the_recorded_runs_stamps_at_each_close_are_its_records_released_counts(run):
    assert len(run.responses) == 40 and [r.n for r in run.responses] == [r.q - 1 for r in run.responses]
    assert all(r.ended is None for r in run.responses)
    release = {r["stream"]: r["release"] for r in run.io["records"]}
    frames: Counter = Counter()
    for w in range(17):
        b = w * T
        want = tuple((q, q - 1, frames[q]) for q in sorted(release) if release[q] < b)
        assert C.stamps(run.responses, b) == want
        C.check(want, run.responses, b)
        assert tuple((c.q, c.n, c.e) for c in C.continuations(run.responses, b, run.eos)) == CONTINUED.get(b, ())
        assert set(CONTINUED.get(b, ())) <= set(want)
        if w < 16:
            rec = next(r for r in run.outs if r.window == w)
            frames.update(q for q in (chain._output_stream(s, False) for row in rec.grid for s in row if s is not None)
                          if q is not None)
    assert len(C.stamps(run.responses, 16 * T)) == 40 and sum(frames.values()) == run.io["tokens"] == 597
    assert sum(len(s) for s in CONTINUED.values()) == 13
    assert sorted(e for s in CONTINUED.values() for _, _, e in s if e) == [3, 3, 16, 18]
    by_q = {r.q: r for r in run.responses}
    for b, want in CONTINUED.items():
        for q, _, e in want:
            r = by_q[q]
            assert r.released < b and e < r.request.max_tokens and len(r.cells) > e
            assert r.cells[e][0] >= b and (e == 0 or r.cells[e - 1][0] < b)
    for r in run.responses:
        for b in range(0, 17 * T, T):
            if C.is_open(r, b) and (r.q, r.n, len(C.released(r, b))) not in CONTINUED.get(b, ()):
                assert C.finished(r, b, run.eos), r.q


def test_the_recorded_runs_continuations_are_its_certified_requests_and_released_ids(run):
    rows = {r["stream"]: r for r in run.io["records"]}
    for b, want in CONTINUED.items():
        conts = C.continuations(run.responses, b, run.eos)
        assert [(c.q, c.n, c.e) for c in conts] == list(want)
        for c in conts:
            was = rows[c.q]
            assert c.request.prompt == tuple(was["prompt"]) + tuple(was["output"][:c.e])
            assert 0 < c.request.max_tokens == was["max_tokens"] - c.e <= run.plan["varied"]["max_tokens"]
            assert (c.request.user, c.request.model, c.request.seed, c.request.temperature, c.request.top_p) == \
                (was["user"], was["model"], was["seed"], was["temperature"], was["top_p"])
    (two,) = C.continuations(run.responses, 60, run.eos)
    assert (len(two.request.prompt), two.request.max_tokens, two.request.user) == (83 + 16, 4, "varied.00001")
    assert [c.q for c in C.continuations(run.responses, 60, run.eos)] == [2] and \
        [s.q for s in C.stamps(run.responses, 60)] == [1, 2]


def test_the_records_before_a_close_alone_give_its_stamps(run):
    for w in range(17):
        b = w * T
        io = cs.certified_io(run.plan, run.setup, run.ins[:w], run.outs[:w]) if w else {"records": []}
        alone = _responses(io)
        assert C.stamps(alone, b) == C.stamps(run.responses, b)
        assert C.continuations(alone, b, run.eos) == C.continuations(run.responses, b, run.eos)


def test_each_disagreement_with_the_recorded_stamps_is_refused_by_name(run):
    """At bucket 480 streams 1 … 21 are released: 19, 20 and 21 continue, the rest are finished; 22 is released
    after it."""
    want = list(C.stamps(run.responses, 480))
    assert [s.q for s in want] == list(range(1, 22)) and want[-3:] == list(CONTINUED[480])
    for name, written in (("missing", want[:-1]), ("missing", want[:18] + want[19:]), ("missing", want[1:]),
                          ("missing", want[:9] + want[10:]), ("duplicate", want + want[:1]),
                          ("order", want[::-1]), ("order", [want[1], want[0], *want[2:]]),
                          ("extra", want + [(22, 21, 0)]), ("extra", want + [(41, 40, 0)]),
                          ("release", [*want[:18], (19, 17, 3), *want[19:]]),
                          ("count", [*want[:18], (19, 18, 4), *want[19:]]),
                          ("count", [*want[:18], (19, 18, 12), *want[19:]]),
                          ("count", [(1, 0, 2), *want[1:]]), ("count", [*want[:20], (21, 20, 1)])):
        with pytest.raises(C.Refused, match=f"^{name}: ") as e:
            C.check(written, run.responses, 480)
        assert e.value.name == name
    with pytest.raises(C.Refused, match="^missing: response 1 "):
        C.check(CONTINUED[480], run.responses, 480)
    for b in range(0, 17 * T, T):
        if C.stamps(run.responses, b) != tuple(want):
            with pytest.raises(C.Refused):
                C.check(C.stamps(run.responses, b), run.responses, 480)
