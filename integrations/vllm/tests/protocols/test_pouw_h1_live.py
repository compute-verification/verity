"""The live proof block's tables on the served path (`pouw_h1_live`). With the Commit a window registers round 0's request
headers ``requests@0`` and each round's slot table ``slots@<s>`` as values of the live Program, under its round roots, and
each round's ``live@<s>`` (`PouwLiveSlots_v2{T}`) accepts its slot table at m_s, the step's scheduled rows, and refuses any
other m. Under CUDA graphs m_s is the scheduler's rows, not the graph's padded size."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verity.ml.protocols.pouw.circuit import live as L  # noqa: E402
from verity.core.protocols.pouw import live_reads as LR  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402
from verity_vllm.protocols import pouw  # noqa: E402
from verity_vllm.protocols.pouw import h1 as PH  # noqa: E402
from verity_vllm.protocols.pouw.h1 import live as LV  # noqa: E402
from verity_vllm.protocols.pouw.h1 import requests as RQ  # noqa: E402
from verity_vllm.protocols.pouw.h1 import served as SV  # noqa: E402
from verity_vllm.protocols.pouw.h1.static import graph_pad  # noqa: E402

from tests.protocols.test_credit_gate import (  # noqa: E402,F401
    FILES, LAYOUT, PHC, ROWS, ZyArm, _credit, _descriptor, _linear_is_the_stand_in, _period_weights, _record, _serve,
    _served, periodic)
from tests.protocols.test_pouw_h1_served import _unit  # noqa: E402

GRAPHS = (1, 2, 4, 8)


def _window(**kw):
    """`_serve`'s one-call window (m = 3) under in-process coins with the Commit in its engine."""
    _y, run = _serve(ZyArm(), n=64, served=SV.Served(PH.TILE, _unit(), weights=_period_weights(), window=bytes(64)), **kw)
    assert not run.served.refused
    return run


def test_a_served_window_registers_round_0s_request_headers_and_each_rounds_slot_table():
    """``requests@0`` is round 0's request table (its root the ``request_table`` inside round 0's request root), bound to
    the window's challenge phase; ``slots@0`` is round 0's slot table, bound to the open root and listed whole among round 0's
    segments. Both are among the window's entries, open in the live proof only, and core's ROOTS1 steps tie the window."""
    from tests.protocols import pouw_h1_gpu as GPU
    run = _window()
    served, row = run.served, run.round_log[0]
    req, slots = served.live[LV.value(served, LV.REQUESTS)], served.live[LV.value(served, LV.SLOTS)]
    assert sorted(served.live) == ["h1/requests@0", "h1/slots@0"]
    assert slots.root.hex() == row["slots"] and slots.value in row["a"]
    assert req.root.hex() == row["request_table"] == run.commits["request_table"]
    assert req.leaves == slots.leaves == ROWS
    for reg, bound in ((req, {"session": SV.LOCAL_SESSION}), (slots, {"window": served.root})):
        e = reg.entry()
        assert e["domain"] == G.domain(served.program_digest, e["value"], e["words"], e["leaves"], **bound).domain_id.hex()
        assert e in served.entries()
    assert [e["words"] for e in (req.entry(), slots.entry())] == [LR.HEADER_WORDS, LR.SLOT_WORDS]
    for reg, schema, other in ((req, G.SCHEMA_SEG, G.SCHEMA_V2), (slots, G.SCHEMA_V2, G.SCHEMA_SEG)):
        assert reg.schema == schema
        assert G.verify_read(reg.domain, reg.root, 1, reg.bcs[1], reg.path(1), schema)
        assert not G.verify_read(reg.domain, reg.root, 1, reg.bcs[1], reg.path(1), other)
    ports = LV.statement(run, 0)[1]["ports"]
    assert ports["req_0"][G.SCHEMA_KEY] == G.SCHEMA_SEG and ports["slot_0"][G.SCHEMA_KEY] == G.SCHEMA_V2
    assert LV.verdicts(run) == [{"round": 0, "m": 3, "admitted": 3, "ok": True}]
    with pytest.raises(ValueError, match="live table"):
        run.open_request(0)
    with pytest.raises(ValueError, match="live table"):
        run.open_slot(0, 0)
    v = GPU.core_verdict({"commitment": dict(run.commits), "calls": run.manifest(), "descriptor": _descriptor().hex()})
    assert v["a_link"] is None and v["tied"], v


def test_a_per_round_window_states_no_live_unit(tmp_path):
    """``live@<s>`` reads round 0's request table every round, so a window that admits per round is refused by name
    (`admission_gap`): its statement, evaluation and dump raise it, and every round's verdict carries it, not ok."""
    run = _window()
    assert LV.admission_gap(run) is None
    run.admission = RQ.PER_ROUND
    gap = "per-round admission .* isn't served yet"
    for build in (lambda: LV.statement(run, 0), lambda: LV.evaluated(run, 0), lambda: LV.dump(run, tmp_path)):
        with pytest.raises(ValueError, match=gap):
            build()
    assert LV.verdicts(run) == [{"round": 0, "m": 3, "ok": False, "reason": LV.admission_gap(run)}]


def test_the_gpu_harness_holds_a_commit_windows_requests_to_its_live_verdicts():
    """`pouw_h1_gpu.requests_check` on a window with the Commit: its request and slot tables are live, so each round's
    ``live@<s>`` verdict stands in place of the leaf openings, one a round, all accepted."""
    from tests.protocols import pouw_h1_gpu as GPU
    run = _window()
    out = GPU.requests_check(run, 1)
    assert out["live"] == LV.verdicts(run) and out["live_accepted"] == "1/1", out
    assert out["entries_refused"] == out["slots_refused"] == [] and out["rounds"] is None


@pytest.mark.parametrize("claim", [0, 2, 4, ROWS])
def test_a_live_unit_claiming_other_than_the_served_rows_is_refused(claim):
    """``live@0`` on the window's slot table evaluates to 1 at m = 3, the manifest's served rows, and to 0 at any other
    m; its ``m`` port's root at a claimed m is not the one the served rows' public ``m@0`` row commits, while its slot and
    header ports are the same tables."""
    run = _window()
    assert run.manifest()[0]["served"] == LV.m_of(run, 0) == 3 and LV.evaluated(run, 0) == 1
    assert LV.evaluated(run, 0, claim) == 0
    (rows, honest), (other_rows, other) = LV.statement(run, 0), LV.statement(run, 0, claim)
    served = LR.m_table(run.served.program_digest, "h1/", run.served.root, 0, 3)[2].root.hex()
    assert honest["ports"]["m"]["root"] == served != other["ports"]["m"]["root"]
    assert all(honest["ports"][p] == other["ports"][p] for p in honest["ports"] if p != "m")
    assert all((rows[p] == other_rows[p]).all() for p in rows if p != "m")


def test_a_cuda_graph_padded_step_states_the_scheduled_rows():
    """A step of 3 scheduled rows that a CUDA graph ran at its capture size 4: the call's A table holds the 4 rows it
    computed, its ``pad`` is 1 and its served rows (the manifest's, m_s) are 3; the Commit's padded input id is not a
    slot, the window is tied and has no gap, and ``live@0`` accepts m = 3 and refuses 4."""
    run = _window(pad=1, graphs=GRAPHS)
    (call,) = run.calls
    assert (call.m, call.pad, call.served) == (4, 1, 3)
    assert run.manifest()[0]["served"] == 3 and run.step_rows() == [3]
    assert run.request_gaps() == [] and run.details["roots1"]["tied"], run.details["roots1"]
    c = run.credit()
    assert c["gate_passed"]["rows"] + c["refused"]["rows"] == 3 and c["padded"]["rows"] == ROWS - 3
    assert LV.verdicts(run) == [{"round": 0, "m": 3, "admitted": 3, "ok": True}] and LV.evaluated(run, 0, 4) == 0


@pytest.mark.parametrize("graphs", [(), (8,), (2, 3, 4)], ids=["eager", "past-the-size", "not-the-smallest"])
def test_rows_past_the_batch_no_capture_size_pads_to_are_a_gap_and_live_refuses_them(graphs):
    """The same 4 rows for 3 scheduled when the engine is eager, or 4 isn't the smallest capture size at least 3: the call
    serves 4, the step is a gap and the window credits nothing, and ``live@0`` at m = 4 refuses its 3 slots."""
    run = _window(pad=1, graphs=graphs)
    (call,) = run.calls
    assert (call.m, call.pad) == (4, 0) and run.step_rows() == [4]
    assert run.request_gaps() == ["step 0: 4 rows served and 3 in the step's batch, so the slot table can't name each "
                                  "served row's request"]
    assert run.credit()["requests"] == run.request_gaps() and {v["reason"] for v in run.gated().values()} == {RQ.UNENTERED}
    (v,) = LV.verdicts(run)
    assert not v["ok"] and v["reason"] == "live@0: live_ok is 0 at m = 4" and v["admitted"] == 3


def test_graph_pad_is_vllms_padding_to_the_smallest_capture_size():
    assert graph_pad(4, 3, GRAPHS) == 1 and graph_pad(8, 5, (8, 1, 4)) == 3
    assert graph_pad(4, 3, ()) == graph_pad(8, 3, GRAPHS) == graph_pad(3, 3, (4,)) == graph_pad(4, 5, (4, 8)) == 0


@pytest.mark.parametrize("bad", [(0,), (True,), "8", (2.0,)])
def test_the_graphs_setting_is_positive_capture_sizes(bad):
    with pytest.raises(ValueError, match="'graphs'"):
        pouw.Settings.of({"graphs": bad})
    assert pouw.Settings.of({"graphs": [1, 2]}).graphs == (1, 2) and pouw.Settings.of({}).graphs == ()


def test_core_credit_ties_the_slot_table_under_the_auditors_coins(periodic):
    """Under the auditor's challenger, core's credit (`credit.window`) passes the window's registration and a-link with
    its live tables among its entries, and refuses it only for want of its drawn calls' proof files."""
    auditor = periodic(PHC.draw_k(LAYOUT), files=FILES)
    _y, run = _serve(ZyArm(), remote=True, n=64, served=_served())
    assert auditor.ended() is None
    entries = _record(run)["commitment"]["served"]["entries"]
    assert {"h1/requests@0", "h1/slots@0"} <= {e["value"] for e in entries}
    assert LV.verdicts(run)[0]["ok"]
    out = _credit(auditor, _record(run))
    assert out["window"][0] == "missing" and out["credit"] is None, out


def test_the_provers_live_dump_opens_every_row_its_statement_reads(tmp_path):
    """`pouw_h1_live.dump`: window.json holds only what the verifier may read (T, S, the prefix, the window Program's
    digest, the open root, the challenge phase, the live roots, m_s and the public max_tokens), and each round's rows.npz and
    opening.json give every port's words, salt, value, position and path, each opening its registered root
    (`registered.verify_read`) under the row format its reads file names; ``m``, ``s``, ``cap_<i>`` and ``o0`` are the
    known ``m@<s>``, ``round@<s>``, ``caps`` and ``live-ok`` rows."""
    import json

    import numpy as np

    run = _window()
    got = LV.dump(run, tmp_path / "live")
    pub = json.loads((tmp_path / "live" / "window.json").read_text())
    served = run.served
    assert got == {"dir": str(tmp_path / "live"), "rounds": 1}
    assert pub == {"T": ROWS, "S": run.steps, "prefix": "h1/", "program": served.program_digest, "window": served.root,
                   "session": SV.LOCAL_SESSION, "roots": {v: r.root.hex() for v, r in served.live.items()}, "served": [3],
                   "max_tokens": LV.caps(run.admitted[0])}
    assert pub["max_tokens"] and all(pub["max_tokens"])
    z = dict(np.load(tmp_path / "live" / "0" / "rows.npz"))
    opening = json.loads((tmp_path / "live" / "0" / "opening.json").read_text())
    rows, doc = LV.statement(run, 0)
    rows[LR.OK_PORT] = np.asarray(LR.ok_table(pub["program"], "h1/", pub["window"], 1)[0], np.uint16)
    assert list(z["units"]) == [0] and sorted(opening) == sorted(doc["ports"])
    for p, e in doc["ports"].items():
        assert np.array_equal(z[f"w_{p}"], rows[p]), p
        op = opening[p]
        assert (op["value"], op["index"], str(z[f"v_{p}"][0]), int(z[f"r_{p}"][0]), str(z[f"p_{p}"][0])) == (
            e["value"], e["reads"][0], e["value"], e["reads"][0], op["path"])
        bind = {"session": pub["session"]} if e["value"] == "h1/requests@0" else {"window": pub["window"]}
        dom = G.domain(pub["program"], e["value"], e["words"], e["leaves"], **bind)
        assert dom.domain_id.hex() == e["domain"]
        assert G.verify_read(dom, bytes.fromhex(e["root"]), op["index"], bytes.fromhex(op["bc"]), bytes.fromhex(op["path"]),
                             e.get(G.SCHEMA_KEY, G.SCHEMA)), p
