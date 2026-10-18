"""The replay harness (`replay.py`) on records it can be sure of: transcripts of recorded sessions, and the
`verifier.json` that `benchmarks/remote_session/verifier.py` would write of them, built here from the session as it
builds it. Both verifiers must agree on each and give the recorded verdict; a record whose verdict the replay
contradicts (an accepted session with a window that drew another count) is caught."""

from __future__ import annotations

import json
import shutil

import pytest

import ra_cases as M
import replay as R
from verity.core.protocols.remote_bandwidth import audit

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def server():
    with R.server() as s:
        yield s


def verifier_json(r: M.Run) -> dict:
    """`r`'s session as `verifier.py` records it: arrivals of the roots its session took, each window's draws and the
    time they were sent, and an accepted session's record."""
    s, reg = r.s, r.store.registration
    arrivals: dict[int, list[int]] = {}
    for e in r.events:
        if e["kind"] == "receive" and 0 <= e["t"] < len(s.windows):
            arrivals.setdefault(e["t"], []).append(e["now"])
    drawn_at = {e["t"]: e["now"] for e in r.events if e["kind"] == "close"}
    v = s.failed
    session = {
        "k": 0, "accepted": s.outcome == "accepted",
        "failure": None if v is None else {"t": v.window, "failure": v.failure, "at": None if v.at is None else list(v.at)},
        "allowance_ns": r.params.deadline_ns, "samples": r.params.samples, "window_ns": r.params.window_ns,
        "windows": r.params.windows, "registration": reg.digest.hex(),
        "draws": {str(t): [list(d) for d in w.draws] for t, w in enumerate(s.windows) if w.draws},
        "windows_seen": [{"t": t, "released_ns": w.released_ns, "boundary_ns": w.boundary_ns,
                          "deadline_ns": s.deadline(t) if w.boundary_ns is not None else None,
                          "passes": len(w.roots), "arrivals_ns": arrivals.get(t, [])[:len(w.roots)],
                          "draws_sent_ns": drawn_at.get(t) if w.draws else None} for t, w in enumerate(s.windows)]}
    if s.outcome == "accepted":
        rec = s.record()
        session["record"] = {"public": rec.public(), "profile": rec.profile(), "end_ns": rec.end_ns}
    registration = {"node": reg.node.hex(), "blocks": reg.blocks, "payload_root": reg.payload_root.hex(),
                    "codec_seed": reg.codec_seed.hex(), "codec": reg.codec.name, "receipt": reg.receipt,
                    "bind": reg.bind, "layout": reg.layout, "rows": reg.rows, "digest": reg.digest.hex()}
    return {"schema": "verity/remote-session/verifier/v0", "registration": registration, "sessions": [session]}


def run(name: str, **kw) -> M.Run:
    world, base, build, _ = M.SCENARIOS[name]
    r = M.Run(world, **(base | kw))
    build(r)
    return r


def zero_passes(r: M.Run) -> None:
    """Window 1 sees no root: its close fails `few` at a minimum of one pass."""
    r.start()
    r.run()
    r.boundary()
    r.boundary()
    r.finish()


@pytest.mark.parametrize("name", ["honest", "drop-registered", "unchained-head", "late"])
def test_a_transcript_replays_to_the_definitions_verdict(server, tmp_path, name) -> None:
    case, outcome = M.recorded(name)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    row, = R.replay_transcript(path, server)
    assert row["agree"] and row["held"], row
    assert row["lean"]["outcome"] == M.SCENARIOS[name][3] == outcome


@pytest.mark.parametrize("name", ["honest", "honest-subset", "honest-by-root", "drop-registered", "stale-challenge"])
def test_a_verifier_json_replays_to_its_recorded_verdict(server, tmp_path, name) -> None:
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(verifier_json(run(name))))
    row, = R.replay_record(path, server)
    assert row["agree"] and row["held"] and row["digest"], row
    assert row["lean"]["rejected"] is None and row["lean"]["refused"] == 0


def test_too_few_passes_replays_to_the_same_rejection(server, tmp_path) -> None:
    r = M.Run(windows=2)
    zero_passes(r)
    assert r.s.outcome == "few"
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(verifier_json(r)))
    row, = R.replay_record(path, server)
    assert row["agree"] and row["held"], row
    assert row["lean"]["rejected"] == {"failure": "few", "window": 1, "at": None}


def test_a_record_its_replay_contradicts_is_caught(server, tmp_path) -> None:
    """An accepted session's record whose last pass arrived after its window's deadline: both verifiers reject it
    `late`, so its recorded verdict doesn't hold."""
    v = verifier_json(run("honest"))
    seen = v["sessions"][0]["windows_seen"][-1]
    seen["arrivals_ns"][-1] = seen["deadline_ns"] + 1
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["agree"] and not row["held"], row
    assert row["lean"]["rejected"]["failure"] == "late"


def test_a_registration_whose_digest_isnt_recorded_is_caught(server, tmp_path) -> None:
    v = verifier_json(run("honest"))
    v["registration"]["payload_root"] = "00" * 64
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["digest"] is False
    assert R.main(["record", str(path)]) == 1


def test_a_registration_from_before_layout_and_rows_replays_without_its_digest(server, tmp_path) -> None:
    v = verifier_json(run("honest"))
    for k in R.LATER:
        del v["registration"][k]
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["agree"] and row["held"] and row["digest"] is None, row
    assert "layout and rows" in row["made"]["registration"]


def test_a_session_from_before_the_stated_window_replays_under_its_shortest(server, tmp_path) -> None:
    """A window shorter than the record's `window_ns` is a boundary today's `Session` refuses, so the next window's
    challenge never leaves and its first root fails `order`; the case states the session's shortest window instead."""
    v = verifier_json(run("honest"))
    s = v["sessions"][0]
    shortest = min(w["boundary_ns"] - w["released_ns"] for w in s["windows_seen"])
    s["window_ns"] = shortest + 1
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["agree"] and row["held"], row
    assert row["made"]["window_ns"].startswith(f"{shortest},")


def test_a_refused_registration_has_no_verdict_to_replay(server, tmp_path) -> None:
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps({"schema": "verity/remote-session/verifier/v0", "refused": "node 00 has a live "
                                "registration", "registry": audit.Registry().to_json()}))
    row, = R.replay_record(path, server)
    assert row["agree"] is None and row["held"] is None and row["recorded"] == "registration refused"
    assert R.main(["record", str(path)]) == 0


def test_case_of_keeps_the_recorded_parameters_and_clock() -> None:
    v = verifier_json(run("honest"))
    s = v["sessions"][0]
    case, made = R.case_of(v["registration"], s)
    assert case["params"] == {
        "deadline_ns": s["allowance_ns"], "deferred": False, "lifetime": s["record"]["public"]["lifetime"],
        "min_passes": 1, "samples": s["samples"], "window_ns": s["window_ns"], "windows": s["windows"]}
    assert s["record"]["public"]["lifetime"] == audit.LIFETIME
    receives = [e for e in case["events"] if e["kind"] == "receive"]
    assert [e["now"] for e in receives] == [x for w in s["windows_seen"] for x in w["arrivals_ns"]]
    assert [e["step"] for e in receives] == [x for w in s["record"]["profile"]["windows"] for x in w["steps"]]
    assert made == {"plans": "recorded lengths", "steps": "recorded"}
