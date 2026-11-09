"""The continuation rule (`verity.core.protocols.certifier.continuation`) on a recorded varied-plan run: the token
run's ingress and egress records (`tests/data/token`: 16 execution phases of 60 buckets, 40 id requests of a varied
plan through the certifying proxy), read through the rehearsal's certified I/O (`certifier_serve.certified_io`), each
request's stream its ordinal and its place among the released streams its release number. The run is the token
negative's: its shim changed ids on their way to the proxy, and the rule reads what the records hold as released.

Nine of its closes have a response open: four cut with ids already released (16, 3, 18 and 3 of them), and nine
released before the close whose first id comes after it, stamped e = 0. At every close the stamps are the ones
written out here, each e the id frames of its stream in the egress records before the close, counted from the frames
themselves; each continuation is its certified request with those ids after its prompt and `max_tokens` − e left; the
records before the close alone give the same stamps; and a stamp dropped, doubled, moved out of release order, added
for a finished response, or of another N or e is refused by its name. No response of this run is open at two closes:
`test_network_certifier_continuation.py` holds that case on the network certifier's own records."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import continuation as C

HERE = Path(__file__).resolve().parents[1]
TOKEN = HERE / "tests" / "data" / "token"
T = 60
#: each close with a response open, and its stamps (q, N, e)
STAMPS = {60: ((2, 1, 16),), 120: ((3, 2, 0),), 360: ((13, 12, 0), (14, 13, 0)),
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
    each id frame at its egress bucket."""
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
    frames: Counter = Counter()
    for w in range(17):
        b = w * T
        want = STAMPS.get(b, ())
        assert C.stamps(run.responses, b, run.eos) == want
        C.check(want, run.responses, b, run.eos)
        assert all(e == frames[q] for q, _, e in want)
        if w < 16:
            rec = next(r for r in run.outs if r.window == w)
            frames.update(q for q in (chain._output_stream(s, False) for row in rec.grid for s in row if s is not None)
                          if q is not None)
    assert sum(len(s) for s in STAMPS.values()) == 13 and sorted(e for s in STAMPS.values() for _, _, e in s if e) == \
        [3, 3, 16, 18]
    by_q = {r.q: r for r in run.responses}
    for b, want in STAMPS.items():
        for q, _, e in want:
            r = by_q[q]
            assert r.released < b and e < r.request.max_tokens and len(r.cells) > e
            assert r.cells[e][0] >= b and (e == 0 or r.cells[e - 1][0] < b)


def test_the_recorded_runs_continuations_are_its_certified_requests_and_released_ids(run):
    rows = {r["stream"]: r for r in run.io["records"]}
    for b, want in STAMPS.items():
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


def test_the_records_before_a_close_alone_give_its_stamps(run):
    for b, want in STAMPS.items():
        w = b // T
        io = cs.certified_io(run.plan, run.setup, run.ins[:w], run.outs[:w])
        assert C.stamps(_responses(io), b, run.eos) == want


def test_each_disagreement_with_the_recorded_stamps_is_refused_by_name(run):
    want = list(STAMPS[480])
    for name, written in (("missing", want[1:]), ("missing", want[:2]), ("duplicate", want + want[:1]),
                          ("order", want[::-1]), ("order", [want[1], want[0], want[2]]),
                          ("extra", want + [(18, 17, 12)]), ("extra", want + [(22, 21, 0)]),
                          ("release", [(19, 17, 3), *want[1:]]), ("count", [(19, 18, 4), *want[1:]]),
                          ("count", [(19, 18, 12), *want[1:]]), ("count", [*want[:2], (21, 20, 1)])):
        with pytest.raises(C.Refused, match=f"^{name}: ") as e:
            C.check(written, run.responses, 480, run.eos)
        assert e.value.name == name
    for b in STAMPS:
        if b != 480:
            with pytest.raises(C.Refused):
                C.check(STAMPS[b], run.responses, 480, run.eos)
            with pytest.raises(C.Refused, match="^missing: "):
                C.check((), run.responses, b, run.eos)
