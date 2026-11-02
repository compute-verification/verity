"""vLLM's served run in `fixtures/vllm-8eb1` (with its `meta.json`): it serves without prefix caching,
the replay is the served scheduler on every step, every window opens on a drained engine, ReplayStep_v1 is the
scheduler on windows 212 and 1 under the served async engine, `replay_vectors --slice` builds the compile slice's w212
fixtures (at the engine class's AsyncScheduler, and at the sync class) and its async w1 fixtures from the run's own
records, and the certifier's charge chain is `code_v1`'s charge of the decisions alone on every window."""
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
CLASS = SLICE.with_name("compile_slice_w212_class.json")
ARITH = SLICE.with_name("arith_encode_vectors.json")


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
CV = _load("enc_code_v1", HERE.parent / "code_v1.py")


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


def test_the_run_serves_without_prefix_caching(recs):
    """8eb1's trace header says it serves without prefix caching, so every window's hits are 0, and the charge chain,
    which charges the decisions alone, leaves no hit uncharged. Its order isn't 0: the run wasn't
    served in release order, and the order's charge is the I/O tie's."""
    header = recs[0]
    assert header["kind"] == "header" and header["config"]["enable_prefix_caching"] is False
    assert RP.load_step_trace(recs).engine.enable_prefix_caching is False


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


@pytest.mark.slow
def test_the_charge_chain_is_code_v1s_through_forced_admissions(recs):
    """Each window opens on a drained engine, so its first arrival is admitted unasked (`Replay.forced`), which is no
    decision. The certifier's charge chain (`charge_chain.session` at the engine class's D = 32, under the async-light
    book) over the 339 windows' decisions is `code_v1`'s charge of them alone (`Window(arrived=…)`, every other count
    0) on every window, 5,500 bits: its context ``run`` counts
    decisions only, so a forced admission doesn't reset it as it resets the replay's ``pc``. Charged as a decision (a 1,
    which resets ``run``), each forced admission would change 330 windows' charges."""
    from verity.core.protocols.certifier import charge_chain as CC

    book = next(s for s in json.loads(ARITH.read_text())["streams"]
                if s["source"].startswith("async-light"))["decisions"]["book"]
    per, as_ones, forced = [], [], []
    for w in range(339):
        ds, rp = RP.decisions(RP.load_step_trace(RV.served_window(recs, w)[0]))
        per.append([int(d.arrived) for d in ds])
        forced.append(sum(rp.forced))
        ones = [(rp.admitted[i], i, 1) for i, f in enumerate(rp.forced) if f]
        as_ones.append([b for *_at, b in sorted([(d.step, d.request, int(d.arrived)) for d in ds] + ones)])
    D = 32
    ses = CC.session(D, book, [([1] * len(d) + [0] * (D - len(d)), d + [0] * (D - len(d))) for d in per])
    lens = list(CV.encode([CV.Advice(arrived=d) for d in per], CV.Book.from_json(book)).lens)
    other = CV.encode([CV.Advice(arrived=d) for d in as_ones], CV.Book.from_json(book)).lens
    assert forced == [1] * 339
    assert [s[0] for s in ses] == lens and sum(lens) == 5500
    assert (sum(a != b for a, b in zip(other, lens)), sum(other)) == (330, 5968)


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
    """`replay_vectors --slice 212 --AS 0` writes the certifier's sync-class `compile_slice_w212.json` from the run's
    records: requests 308 to 310 of the session, their 14 decisions (each a 0 until the request is admitted), the 20
    steps' rows, and the seal's served rows for the window with the runner's ids."""
    out = tmp_path / "w212.json"
    RV.main([str(RUN / "advice_trace.jsonl.gz"), "--slice", "212", "--served",
             str(RUN / "seal-w212.jsonl"), "--R", "2", "--W", "1", "--A", "2", "--AS", "0", "--out", str(out)])
    assert json.loads(out.read_text()) == json.loads(SLICE.read_text())
    fx = json.loads(SLICE.read_text())
    assert [a[0] for a in fx["arrivals"]] == [308, 309, 310] and fx["IW"] == 9
    assert fx["decisions"] == [0] * 6 + [1] + [0] * 6 + [1] and fx["P"] == fx["steps"] == len(fx["rows"]) == 20
    sub, base = RV.served_window(recs, 212)
    assert base == 308
    assert [[[base + i, s, n] for i, s, n in st.rows] for st in RP.load_step_trace(sub).steps] == \
        [[[r[1], r[3], r[4]] for r in step] for step in fx["rows"]]


def test_the_class_fixture_is_w212_padded(tmp_path):
    """At the engine's public sizes (8, 1, 9; Q 4, D 32, P 54, K 4) under the sync class's scheduler,
    `--slice 212 --AS 0` writes `compile_slice_w212_class.json`: w212's own arrivals, decisions, step count and rows at
    the class's statics, which the slice pads."""
    out = tmp_path / "w212_class.json"
    RV.main([str(RUN / "advice_trace.jsonl.gz"), "--slice", "212", "--served", str(RUN / "seal-w212.jsonl"),
             "--R", "8", "--W", "1", "--A", "9", "--Q", "4", "--D", "32", "--P", "54", "--K", "4", "--AS", "0",
             "--out", str(out)])
    got, own = json.loads(out.read_text()), json.loads(SLICE.read_text())
    assert got == json.loads(CLASS.read_text())
    assert {k: got[k] for k in ("R", "W", "A", "Q", "D", "P", "K")} == {"R": 8, "W": 1, "A": 9, "Q": 4, "D": 32, "P": 54,
                                                                       "K": 4}
    assert {k: got[k] for k in ("arrivals", "decisions", "steps", "served", "ids", "IW")} == \
        {k: own[k] for k in ("arrivals", "decisions", "steps", "served", "ids", "IW")}


def test_the_async_class_fixture_is_w212s(tmp_path):
    """At 8eb1's engine class `--slice 212` writes `compile_slice_w212_class_async.json` with no `--AS`: the scheduler
    is the class's, the trace's AsyncScheduler (AS = 1). So the engine keeps `async_scheduling`, each arrival carries
    its cap, the window runs 21 units (the last schedule, on the drained engine, dropped) with the seal at the units
    that ran rows, and its 14 decisions are the sync class's."""
    out = tmp_path / "w212_class_async.json"
    RV.main([str(RUN / "advice_trace.jsonl.gz"), "--slice", "212", "--served", str(RUN / "seal-w212.jsonl"),
             "--R", "8", "--W", "1", "--A", "9", "--Q", "4", "--D", "32", "--P", "54", "--K", "4", "--out", str(out)])
    got, sync = json.loads(out.read_text()), json.loads(CLASS.read_text())
    assert got == json.loads(CLASS.with_name("compile_slice_w212_class_async.json").read_text())
    assert got["engine"]["async_scheduling"] is True and got["source"].endswith("K 4, AS 1)")
    assert got["arrivals"] == [a + [1] for a in sync["arrivals"]] and got["decisions"] == sync["decisions"]
    assert got["steps"] == len(got["rows"]) == len(got["served"]) == 21
    assert [len(s) for s in got["served"]] == [len(r) for r in got["rows"]]


def test_a_class_smaller_than_the_window_is_refused(recs):
    sub, base = RV.served_window(recs, 212)
    with pytest.raises(ValueError, match="smaller than the window: D = 13 < 14, P = 19 < 21"):
        RV.slice_inputs(RP.load_step_trace(sub), R=2, W=1, A=2, base=base, D=13, P=19)


def test_w1_is_async_only(recs):
    """Window 1 (requests 4 to 7, 3 decisions) is the smallest of the 25 windows the sync slice refuses with more than one
    request: its sync replay diverges at step 2, and under the served async engine each of its 39 units is ReplayStep_v1's
    at the engine class (AS = 1), by its reference and by its circuit."""
    sub, base = RV.served_window(recs, 1)
    tr = RP.load_step_trace(sub)
    with pytest.raises(ValueError, match="diverges from the trace at step 2"):
        RV.slice_inputs(tr, R=8, W=1, A=9, base=base, AS=0)
    st, vs = RV.vectors(tr, R=8, W=1, A=9)
    assert st["AS"] == 1 and len(vs) == 39 and all(RV.fits(st, *v) for v in vs)
    assert RV.check(st, vs, circuit=True) == {"checked": 39, "reference_mismatches": 0, "reference_first": [],
                                              "circuit_mismatches": 0, "circuit_first": []}


def test_the_async_class_fixture_is_w1s(recs, tmp_path):
    """`--slice 1 --AS 1` at the engine class writes `compile_slice_w1_class.json`: the engine with `async_scheduling`,
    requests 4 to 7 with their cap (request 4 stops at its max_tokens), the 3 decisions, the 38 units the slice runs (the
    last schedule, on the drained engine, dropped) and their rows, and the seal's 38 steps, one per unit with rows."""
    out = tmp_path / "w1_class.json"
    RV.main([str(RUN / "advice_trace.jsonl.gz"), "--slice", "1", "--served", str(RUN / "seal-w1.jsonl"), "--R", "8",
             "--W", "1", "--A", "9", "--Q", "4", "--D", "32", "--P", "54", "--K", "4", "--AS", "1", "--out", str(out)])
    got = json.loads(out.read_text())
    assert got == json.loads(SLICE.with_name("compile_slice_w1_class.json").read_text())
    assert got["engine"]["async_scheduling"] is True and got["IW"] == 9
    assert got["arrivals"] == [[4, 78, 5, 1], [5, 190, 4, 0], [6, 157, 37, 0], [7, 196, 2, 0]]
    assert got["decisions"] == [1, 1, 1] and got["steps"] == len(got["rows"]) == len(got["served"]) == 38
    assert [len(s) for s in got["served"]] == [len(r) for r in got["rows"]]


def test_the_async_own_fixture_is_w1s(tmp_path):
    """At w1's own statics (4, 1, 5) `--slice 1 --AS 1` writes `compile_slice_w1.json`: Q 4, D 3, P 38, K 4, and the
    class fixture's engine, arrivals, decisions, units, rows and seal."""
    out = tmp_path / "w1.json"
    RV.main([str(RUN / "advice_trace.jsonl.gz"), "--slice", "1", "--served", str(RUN / "seal-w1.jsonl"), "--R", "4",
             "--W", "1", "--A", "5", "--AS", "1", "--out", str(out)])
    got, cls = json.loads(out.read_text()), json.loads(SLICE.with_name("compile_slice_w1_class.json").read_text())
    assert got == json.loads(SLICE.with_name("compile_slice_w1.json").read_text())
    assert {k: got[k] for k in ("R", "W", "A", "Q", "D", "P", "K")} == {"R": 4, "W": 1, "A": 5, "Q": 4, "D": 3, "P": 38,
                                                                       "K": 4}
    same = ("engine", "IW", "arrivals", "decisions", "steps", "rows", "served", "ids")
    assert {k: got[k] for k in same} == {k: cls[k] for k in same}


def test_the_seal_is_placed_by_unit():
    """`by_unit` puts the sealed steps at the units that ran rows, and refuses a count or a step that doesn't match."""
    assert RV.by_unit([[1], [], [2, 3]], [["a"], ["b", "c"]]) == [["a"], [], ["b", "c"]]
    with pytest.raises(ValueError, match="1 sealed steps for 2 units with rows"):
        RV.by_unit([[1], [], [2, 3]], [["a"]])
    with pytest.raises(ValueError, match="unit 2: 2 rows, sealed 1"):
        RV.by_unit([[1], [], [2, 3]], [["a"], ["b"]])
