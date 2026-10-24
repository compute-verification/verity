"""vLLM's served run r20261024-141912-8eb1 (`fixtures/vllm-8eb1`, its `meta.json`): the replay is the served scheduler on
every step, every window opens on a drained engine, ReplayStep_v1 is the scheduler on window 212 under the served async
engine, and `replay_vectors --slice` builds the compile slice's w212 fixture from the run's own records."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
REPO = HERE.parents[3]
RUN = HERE / "fixtures" / "vllm-8eb1"
SLICE = REPO / "verity" / "core" / "protocols" / "certifier" / "tests" / "fixtures" / "compile_slice_w212.json"


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RV = _load("enc_replay_vectors", HERE.parent / "replay_vectors.py")
RP = RV.RP


@pytest.fixture(scope="module")
def recs():
    return RP.records(RUN / "advice_trace.jsonl.gz")


def _sync(tr):
    return dataclasses.replace(tr, engine=dataclasses.replace(tr.engine, async_scheduling=False))


def test_the_replay_is_the_served_scheduler(recs):
    """The async replay (`replay.decisions`, the advice's decisions from each request's first scheduled step) reproduces all
    6,169 steps of the run's 480 requests; under sync scheduling it diverges, so the engine's async mode matters."""
    tr = RP.load_step_trace(recs)
    assert tr.engine.async_scheduling
    ds, _rp = RP.decisions(tr)
    assert (len(tr.steps), len(tr.requests), len(ds), sum(d.arrived for d in ds)) == (6169, 480, 6624, 466)
    with pytest.raises(ValueError, match="diverges"):
        RP.decisions(_sync(tr))


@pytest.mark.slow
def test_every_window_opens_drained(recs):
    """All 339 windows open on an empty engine, so each replays from it, and the cut windows are the whole run. Under sync
    scheduling 314 of them are the served schedule, w212 among them."""
    windows = sorted({r["window"] for r in recs if r.get("kind") == "window"})
    assert windows == list(range(339))
    steps, sync = 0, []
    for w in windows:
        sub, _base = RV.served_window(recs, w)
        tr = RP.load_step_trace(sub)
        steps += len(tr.steps)
        try:
            RP.decisions(_sync(tr))
            sync.append(w)
        except ValueError:
            pass
    assert steps == 6169 and len(sync) == 314 and 212 in sync


def test_a_window_on_a_busy_engine_is_refused(recs):
    """A window opened while a request still runs (window 213's open moved before w212's last step) is refused by name."""
    moved = [{**r, "before_step": 3968} if r.get("kind") == "window" and (r["window"], r["event"]) == (213, "open") else r
             for r in recs]
    with pytest.raises(ValueError, match="1 requests unfinished, cmpl-ca0396cc2f3e49b18f0f2d65890a3a51-0"):
        RV.served_window(moved, 213)


def test_replay_step_is_w212s_scheduler(recs):
    """Under the served engine (AsyncScheduler, AS = 1) each of w212's 22 units (20 steps, one lag schedule after the last
    finish and the update after it) is ReplayStep_v1's, by its reference and by its circuit."""
    sub, _base = RV.served_window(recs, 212)
    st, vs = RV.vectors(RP.load_step_trace(sub), R=2, W=1, A=2)
    assert st["AS"] == 1 and len(vs) == 22 and all(RV.fits(st, *v) for v in vs)
    assert RV.check(st, vs, circuit=True) == {"checked": 22, "reference_mismatches": 0, "reference_first": [],
                                              "circuit_mismatches": 0, "circuit_first": []}


def test_the_slice_fixture_is_w212s(recs, tmp_path):
    """`replay_vectors --slice 212` writes the certifier's `compile_slice_w212.json` from the run's records: requests 308
    to 310 of the session, their 14 decisions (each a 0 until the request is admitted), the 20 steps' rows, and the seal's
    served rows for the window with the runner's ids."""
    out = tmp_path / "w212.json"
    RV.main([str(RUN / "advice_trace.jsonl.gz"), "--slice", "212", "--served",
             str(RUN / "seal-w212.jsonl"), "--R", "2", "--W", "1", "--A", "2", "--SEG", "64", "--out", str(out)])
    assert json.loads(out.read_text()) == json.loads(SLICE.read_text())
    fx = json.loads(SLICE.read_text())
    assert [a[0] for a in fx["arrivals"]] == [308, 309, 310] and fx["IW"] == 9
    assert fx["decisions"] == [0] * 6 + [1] + [0] * 6 + [1] and fx["P"] == fx["steps"] == len(fx["rows"]) == 20
    sub, base = RV.served_window(recs, 212)
    assert base == 308
    assert [[[base + i, s, n] for i, s, n in st.rows] for st in RP.load_step_trace(sub).steps] == \
        [[[r[1], r[3], r[4]] for r in step] for step in fx["rows"]]
