"""`schedule.py`'s schedule check held to its Lean (`RemoteAudit.Schedule`, served by `remote-audit-difftest`'s
`schedule`), refusals as outputs so that both sides' reasons are compared word for word: drawn schedules (registered
frees, appended chunks never resident, never freed and freed, passes at increasing steps), each session's passes and
appends the schedule's or with one fault (a plan steered off the schedule's, a step, a pass or an append missing or
moved, a schedule that is not one, a schedule over the cap committed before the coins), with malformed inputs
rejected by both.  The Lean is of record: `audit.ledger`
reads the Python's verdict (`AuditRun.scheduled`) and this holds it to the Lean, on which
`Properties.RemoteAudit.ScheduleChecked` is proved."""

from __future__ import annotations

import shutil

import pytest
from hypothesis import strategies as st

import ra_cases as M
import verity_difftest as D
from verity.core.protocols.remote_bandwidth import schedule

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def port(inputs=st.nothing(), reference=schedule.run_json) -> D.Port:
    return D.Port(M.EXE, "schedule", reference, inputs, package=M.PACKAGE, of_record=M.OF_RECORD,
                  rejects=M.REFUSALS)


server = D.server_fixture(port())

HONEST = {"blocks": 8, "cap": 256, "schedule": {"registered": [[2, 1, 1]],
                                    "appends": [[[0, None], [0, 1], [0, None], [0, None]], [[2, None], [2, None]]],
                                    "steps": [0, 1, 2, 3, 4]},
          "passes": [[0, [[0, 8]]], [1, [[0, 12]]], [2, [[0, 2], [3, 6], [10, 2]]], [3, [[0, 2], [3, 6], [10, 4]]],
                     [4, [[0, 2], [3, 6], [10, 4]]]],
          "appends": [[8, 4], [12, 2]]}
"""`tests/test_appends.py`'s honest session: its cap (`CAP`), its schedule, its passes' plans and its appends."""


@st.composite
def schedules(draw) -> tuple[int, schedule.Schedule]:
    blocks = draw(st.integers(1, 12))
    registered, cur = [], 0
    while cur < blocks and draw(st.integers(0, 2)):
        f = draw(st.integers(cur, blocks - 1))
        c = draw(st.integers(1, blocks - f))
        registered.append((f, c, draw(st.integers(0, 9))))
        cur = f + c
    block = st.one_of(st.none(), st.tuples(st.integers(0, 9), st.none()),
                      st.integers(0, 8).flatmap(lambda w: st.tuples(st.just(w), st.integers(w + 1, 10))))
    appends = draw(st.lists(st.lists(block, min_size=1, max_size=4).map(tuple), max_size=4))
    steps = sorted(draw(st.sets(st.integers(0, 11), max_size=6)))
    return blocks, schedule.Schedule(tuple(registered), tuple(appends), tuple(steps))


def as_json(blocks: int, cap: int, sc: schedule.Schedule, passes, appends) -> dict:
    return {"blocks": blocks, "cap": cap, "schedule": sc.to_json(),
            "passes": [[s, [list(r) for r in runs]] for s, runs in passes], "appends": [list(a) for a in appends]}


@st.composite
def inputs(draw) -> dict:
    """A drawn schedule with its own session's passes and appends and a cap at or above its bits, or one fault (most
    often none)."""
    blocks, sc = draw(schedules())
    passes = [(s, list(sc.plan(blocks, s))) for s in sc.steps]
    appends = [list(a) for a in sc.append_runs(blocks)]
    fault = draw(st.sampled_from([None] * 6 + ["plan", "step", "pass", "append", "extent", "free", "steps",
                                                "registered", "malformed", "plan", "malformed"]))
    if fault == "plan" and passes:
        k = draw(st.integers(0, len(passes) - 1))
        runs = passes[k][1]
        passes[k] = (passes[k][0], draw(st.sampled_from([runs[1:], runs[::-1], runs + [[blocks + 40, 1]],
                                                          [[f, c + 1] for f, c in runs[:1]] + runs[1:]])))
    elif fault == "step" and passes:
        k = draw(st.integers(0, len(passes) - 1))
        passes[k] = (passes[k][0] + draw(st.integers(1, 3)), passes[k][1])
    elif fault == "pass":
        passes = passes[:-1] if passes and draw(st.booleans()) else passes + [(99, [])]
    elif fault == "append":
        appends = appends[1:] if appends and draw(st.booleans()) else appends + [[0, 1]]
    elif fault == "extent" and appends:
        k = draw(st.integers(0, len(appends) - 1))
        appends[k] = [appends[k][0], appends[k][1] + 1]
    elif fault == "free" and sc.appends:
        a = list(sc.appends[0])
        a[0] = (5, draw(st.integers(0, 5)))
        sc = schedule.Schedule(sc.registered, (tuple(a),) + sc.appends[1:], sc.steps)
    elif fault == "steps":
        sc = schedule.Schedule(sc.registered, sc.appends, sc.steps + draw(st.sampled_from([(0,), (2 ** 64,)])))
    elif fault == "registered":
        sc = schedule.Schedule(sc.registered + draw(st.sampled_from([((0, 1, 0),), ((0, blocks + 1, 0),),
                                                                       ((blocks - 1, 1, 2 ** 64),)])),
                               sc.appends, sc.steps)
    try:
        bits = sc.bits
    except ValueError:  # a chunk freed before its write has no encoding; the check refuses it before reading the cap
        bits = 0
    cap = bits
    if fault is None and draw(st.sampled_from([False] * 5 + [True])):
        fault, cap = "cap", draw(st.integers(0, bits - 1))
    elif fault is None:
        cap += draw(st.sampled_from([0, 1, 8, 2 ** 70]))
    v = as_json(blocks, cap, sc, passes, appends)
    if fault == "malformed":
        bad = draw(st.sampled_from([-1, None, True, "1", [], {}, [1, 2, 3]]))
        k = draw(st.sampled_from(["blocks", "cap", "passes", "appends", "registered", "block", "step", "drop"]))
        if k in ("blocks", "cap", "passes", "appends"):
            v[k] = bad
        elif k == "registered":
            v["schedule"]["registered"] = [bad]
        elif k == "block":
            v["schedule"]["appends"] = [[bad if bad is not None else [1, 2, 3]]]
        elif k == "step":
            v["schedule"]["steps"] = [bad]
        else:
            v.pop(draw(st.sampled_from(["blocks", "cap", "schedule", "passes", "appends"])))
    return v


def test_schedule(server) -> None:
    told = {"bits": 0, "refused": 0, "plan": 0, "cap": 0}

    def reference(v):
        out = schedule.run_json(v)
        told["refused" if "refused" in out else "bits"] += 1
        told["plan"] += "plan is not the schedule's" in out.get("refused", "")
        told["cap"] += "committed before the coins" in out.get("refused", "")
        return out

    stats = D.run(port(inputs(), reference), examples=500, server=server)
    assert (stats.rejects > 10 and told["bits"] > 150 and told["refused"] > 100 and told["plan"] > 15
            and told["cap"] > 10), (stats, told)


def test_the_honest_session_s_verdict_is_the_leans(server) -> None:
    """At its cap and at its own bits the honest schedule passes, and one bit under them it is refused."""
    for cap, out in ((256, {"bits": 200}), (200, {"bits": 200}),
                     (199, {"refused": "the schedule is 200 bits, over the cap of 199 committed before the coins"})):
        v = {**HONEST, "cap": cap}
        assert schedule.run_json(v) == out
        assert D.replay(port(), v, server) == ("output", D.canonical(out))


@pytest.mark.parametrize("change, why", [
    (lambda v: v["schedule"].update(registered=[[2, 1, 1], [2, 1, 3]]),
     "the registered frees are runs of the registered store, in store order and disjoint, each at a step below 2^64"),
    (lambda v: v["schedule"]["appends"].append([]), "append 2 has no block"),
    (lambda v: v["schedule"]["appends"][1].__setitem__(0, [2 ** 64, None]),
     "append 1's block 0 is written at a step past 2^64"),
    (lambda v: v["schedule"]["appends"][1].__setitem__(1, [2, 2 ** 64]), "append 1's block 1 is freed at a step past 2^64"),
    (lambda v: v["schedule"]["appends"][0].__setitem__(1, [3, 3]),
     "append 0's block 1 is freed at step 3, not after its write at step 3"),
    (lambda v: v["schedule"].update(steps=[0, 1, 1, 3, 4]), "the passes' steps strictly increase below 2^64"),
    (lambda v: v["appends"].pop(), "the session made 1 appends, the schedule 2"),
    (lambda v: v["appends"].__setitem__(1, [12, 3]), "append 1 is (12, 3), the schedule's (12, 2)"),
    (lambda v: v["passes"].pop(), "the session has 4 passes, the schedule 5"),
    (lambda v: v["passes"][4].__setitem__(0, 5), "pass 4 is at step 5, the schedule's at 4"),
    (lambda v: v["passes"][3].__setitem__(1, [[0, 2], [3, 6], [10, 2], [13, 1]]),
     "pass 3's plan is not the schedule's at step 3"),
    (lambda v: v["passes"][1].__setitem__(1, [[0, 8], [8, 4]]), "pass 1's plan is not the schedule's at step 1"),
    (lambda v: v["schedule"]["registered"].append([7, 1, 2 ** 40]),
     "the schedule is 264 bits, over the cap of 256 committed before the coins"),
], ids=["registered", "empty-append", "write-past", "free-past", "free-before-write", "steps", "appends",
        "append-extent", "passes", "step", "steered-plan", "unjoined-runs", "over-cap"])
def test_each_refusal_is_the_leans_word_for_word(server, change, why) -> None:
    v = {"blocks": HONEST["blocks"], "cap": HONEST["cap"],
         "schedule": {k: [list(x) if isinstance(x, list) else x for x in vs]
                                                  for k, vs in HONEST["schedule"].items()},
         "passes": [[s, [list(r) for r in runs]] for s, runs in HONEST["passes"]],
         "appends": [list(a) for a in HONEST["appends"]]}
    v["schedule"]["appends"] = [[None if b is None else list(b) for b in a] for a in HONEST["schedule"]["appends"]]
    change(v)
    assert schedule.run_json(v) == {"refused": why}
    assert D.replay(port(), v, server) == ("output", D.canonical({"refused": why}))
