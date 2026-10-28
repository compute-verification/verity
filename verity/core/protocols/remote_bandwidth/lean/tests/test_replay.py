"""The replay harness (`replay.py`) on records it can be sure of: transcripts of recorded sessions, and the
`verifier.json` that `benchmarks/remote_session/verifier.py` would write of them, built here from the session as it
builds it. Both verifiers must agree on each and give the recorded verdict; a record whose verdict the replay
contradicts (an accepted session with a window that drew another count) is caught."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil

import pytest

import ra_cases as M
import replay as R
from verity.core.protocols.remote_bandwidth import audit, wire

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def server():
    with R.server() as s:
        yield s


def verifier_json(r: M.Run) -> dict:
    """`r`'s session as `verifier.py` records it: arrivals of the roots its session took, each window's draws and the
    time they were sent (its draw's, at its draw's last window's close, for every window of a draw that drew), its
    `groups_per_coin` above 1, and an accepted session's record."""
    s, reg, g = r.s, r.store.registration, r.params.groups_per_coin
    arrivals: dict[int, list[int]] = {}
    for e in r.events:
        if e["kind"] == "receive" and 0 <= e["t"] < len(s.windows):
            arrivals.setdefault(e["t"], []).append(e["now"])
    drawn_at = {e["t"]: e["now"] for e in r.events if e["kind"] == "close"}
    drew = {t // g for t, w in enumerate(s.windows) if w.draws}
    v = s.failed
    pace = r.params.pace
    session = ({"groups_per_coin": g} if g != 1 else {}) | {
        "k": 0, "accepted": s.outcome == "accepted",
        "failure": None if v is None else {"t": v.window, "failure": v.failure, "at": None if v.at is None else list(v.at)},
        "allowance_ns": r.params.deadline_ns, "samples": r.params.samples, "window_ns": r.params.window_ns,
        "windows": r.params.windows, "lifetime": r.params.lifetime, "registration": reg.digest.hex(),
        "pace": None if pace is None else {"k": pace.k, "t_ms": pace.window_ns / 1e6},
        "draws": {str(t): [list(d) for d in w.draws] for t, w in enumerate(s.windows) if w.draws},
        "windows_seen": [{"t": t, "released_ns": w.released_ns, "boundary_ns": w.boundary_ns,
                          "deadline_ns": s.deadline(t) if w.boundary_ns is not None else None,
                          "passes": len(w.roots), "arrivals_ns": arrivals.get(t, [])[:len(w.roots)],
                          "draws_sent_ns": drawn_at.get(R.last_of(t, g, len(s.windows))) if t // g in drew else None}
                         for t, w in enumerate(s.windows)]}
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


@pytest.mark.parametrize("name", ["honest", "drop-registered", "unchained-head", "late", "honest-paced",
                                  "order-past-pace"])
def test_a_transcript_replays_to_the_definitions_verdict(server, tmp_path, name) -> None:
    case, outcome = M.recorded(name)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    row, = R.replay_transcript(path, server)
    assert row["agree"] and row["held"], row
    assert row["lean"]["outcome"] == M.SCENARIOS[name][3] == outcome


def test_a_transcript_reads_a_windows_frames_as_the_live_verifiers_do(server, tmp_path) -> None:
    """A transcript's `verify` carries the window's frames as the wire did (`frames`), which both verifiers read as the
    live ones do (`Wire.openingsOf`, `wire.openings_of`): every frame's openings in turn, or none once a frame doesn't
    decode. So a window whose openings all came, beside a frame neither can read, is malformed on replay as it was
    live; a transcript that kept the frames that decoded replayed it accepted."""
    case, _ = M.recorded("honest")

    def framed(extra: list[bytes]) -> dict:
        c = copy.deepcopy(case)
        for e in c["events"]:
            if e["kind"] == "verify":
                openings = [M.opening_of(o) for o in e.pop("openings")]
                h = len(openings) // 2
                e["frames"] = [b.hex() for b in (wire.encode_openings(openings[:h]), wire.encode_openings(openings[h:]),
                                                  *extra)]
        return c

    path = tmp_path / "case.json"
    path.write_text(json.dumps(framed([])))
    row, = R.replay_transcript(path, server)
    assert row["agree"] and row["lean"]["outcome"] == "accepted", row
    first = next(e["t"] for e in case["events"] if e["kind"] == "verify")
    path.write_text(json.dumps(framed([b"\0"])))
    row, = R.replay_transcript(path, server)
    assert row["agree"] and row["lean"]["rejected"] == {"failure": "malformed", "window": first, "at": None}, row


@pytest.mark.parametrize("key", ["samples", "min_passes"])
def test_a_transcript_whose_windows_could_draw_nothing_is_refused_by_both(server, tmp_path, key) -> None:
    """Both verifiers reject a window with no draws `malformed` whatever its frames (`Session.verify`), and no session
    has one: with `samples` or `min_passes` at 0, the transcript, a frame neither can read beside each window's
    openings, is refused by both before its first event."""
    case = copy.deepcopy(M.recorded("honest")[0])
    case["params"][key] = 0
    for e in case["events"]:
        if e["kind"] == "verify":
            e["frames"] = [wire.encode_openings([M.opening_of(o) for o in e.pop("openings")]).hex(), "00"]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    row, = R.replay_transcript(path, server)
    assert row["agree"] and row["held"] is False and "reject" in row["lean"] and "reject" in row["python"], row


def test_a_transcript_that_names_no_pace_replays_only_with_the_pace_its_verifier_json_gives(server, tmp_path) -> None:
    """A transcript written before it named its session's pace: replayed unpaced, a paced session's windows would be
    held to no cap on their roots, so it replays only with the pace its `verifier.json` recorded, and its bytes the
    ones recorded there; and its outcome is held to the one recorded there."""
    case = copy.deepcopy(M.recorded("order-past-pace")[0])
    del case["params"]["pace"]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    row, = R.replay_transcript(path, server)
    assert row["agree"] is None and row["held"] is False and "pace" in row["why"], row
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    paced = {"pace": {"k": 2, "window_ns": 10}, "sha256": digest, "outcome": "order"}
    row, = R.replay_transcript(path, server, {path.name: paced})
    assert row["agree"] and row["held"] and row["lean"]["rejected"]["failure"] == "order", row
    row, = R.replay_transcript(path, server, {path.name: paced | {"outcome": "accepted"}})
    assert row["agree"] and row["held"] is False and row["why"] == "replayed order, the record accepted", row
    row, = R.replay_transcript(path, server, {path.name: paced | {"sha256": "00" * 32}})
    assert row["held"] is False and "sha256" in row["why"], row


@pytest.mark.parametrize("name", ["honest", "honest-subset", "honest-by-root", "drop-registered", "stale-challenge",
                                  "honest-paced"])
def test_a_verifier_json_replays_to_its_recorded_verdict(server, tmp_path, name) -> None:
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(verifier_json(run(name))))
    row, = R.replay_record(path, server)
    assert row["agree"] and row["held"] and row["digest"], row
    assert row["lean"]["rejected"] is None and row["lean"]["refused"] == 0


@pytest.mark.parametrize("name, outcome", [("honest-shared-4", "accepted"), ("honest-shared-paced-uneven", "accepted"),
                                           ("few-shared", "few")])
def test_a_verifier_json_drawn_more_than_one_window_a_coin_replays_to_its_recorded_verdict(server, tmp_path, name,
                                                                                            outcome) -> None:
    """Drawn g > 1 windows a coin (`groups_per_coin`) at two windows in flight, each draw closes at its last window
    under W + ⌈W/g⌉ coins and draws as many receipts as its windows' shares recorded: an honest session, even and
    uneven, replays every draw, and a draw with a window short of its passes is rejected `few` at that window."""
    r = run(name)
    assert r.s.outcome == outcome and r.params.groups_per_coin > 1
    v = verifier_json(r)
    case, _ = R.case_of(v["registration"], v["sessions"][0])
    W, g = r.params.windows, r.params.groups_per_coin
    assert case["params"]["groups_per_coin"] == g and len(case["coins"]) == W + -(-W // g)
    assert {e["t"] for e in case["events"] if e["kind"] == "close"} <= {R.last_of(t, g, W) for t in range(W)}
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["agree"] and row["held"] and row["lean"]["refused"] == 0, row
    if outcome == "accepted":
        assert len(row["lean"]["drawn"]) == -(-W // g) and row["why"] == "every draw drawn, as many as recorded"


def test_a_verifier_json_drawn_more_than_one_window_a_coin_at_one_in_flight_isnt_replayed(server, tmp_path) -> None:
    """At one window in flight no property covers a session drawn g > 1 windows a coin, and the driver refuses one:
    a record of one has no verdict to replay."""
    v = verifier_json(run("honest-shared-4"))
    v["sessions"][0]["pace"] = {"k": 2, "window_ns": 10, "in_flight": 1}
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["agree"] is None and row["held"] is None and "one window in flight" in row["why"], row


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


def refused() -> dict:
    """The record `verifier.py` writes of a second registration for a node with one live: the refusal and the
    registry after it."""
    registry = audit.Registry()
    reg = M.Run().store.registration
    hosts = M.site_of(M.site_json(reg.node))
    registry.register(M.SOURCE, reg.node, reg.blocks, reg.payload_root, reg.codec, nonce=bytes(32), site=hosts)
    with pytest.raises(audit.RegistryError) as e:
        registry.register(M.SOURCE, reg.node, reg.blocks, reg.payload_root, reg.codec, nonce=bytes(32), site=hosts)
    return {"schema": "verity/remote-session/verifier/v0", "refused": str(e.value), "registry": registry.to_json()}


def test_a_refused_registration_replays_on_the_registry_before_it(server, tmp_path) -> None:
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(refused()))
    row, = R.replay_record(path, server)
    assert row["recorded"] == "registration refused" and row["agree"] and row["held"] and row["registry"], row
    assert R.main(["record", str(path)]) == 0


def test_a_refusal_by_another_registration_than_the_recorded_one_is_caught(server, tmp_path) -> None:
    v = refused()
    v["registry"]["live"][M.NODE.hex()]["registration"] = "11" * 64
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["agree"] and row["held"] is False, row
    assert R.main(["record", str(path)]) == 1


def test_a_refusal_with_no_registry_recorded_has_no_verdict_to_replay(server, tmp_path) -> None:
    v = refused()
    del v["registry"]
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["agree"] is None and row["held"] is None
    assert R.main(["record", str(path)]) == 0


def test_a_recorded_registry_both_verifiers_cannot_read_back_is_caught(server, tmp_path) -> None:
    v = verifier_json(run("honest")) | {"registry": refused()["registry"]}
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    assert R.replay_record(path, server)[0]["registry"] is True
    v["registry"]["closed"] = [{"node": M.NODE.hex(), "registration": "00" * 63, "sessions": []}]
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["registry"] is False and row["agree"] and row["held"], row
    assert R.main(["record", str(path)]) == 1


def test_a_registration_with_no_session_is_one_row_with_its_digest(server, tmp_path) -> None:
    v = verifier_json(run("honest")) | {"sessions": []}
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["recorded"] == "no session" and row["agree"] is None and row["digest"] is True


def test_case_of_keeps_the_recorded_parameters_and_clock() -> None:
    v = verifier_json(run("honest"))
    s = v["sessions"][0]
    case, made = R.case_of(v["registration"], s)
    assert case["params"] == {
        "deadline_ns": s["allowance_ns"], "deferred": False, "lifetime": s["record"]["public"]["lifetime"],
        "min_passes": 1, "pace": None, "samples": s["samples"], "window_ns": s["window_ns"], "windows": s["windows"]}
    assert s["record"]["public"]["lifetime"] == s["lifetime"] == audit.lifetime_windows(s["window_ns"])
    receives = [e for e in case["events"] if e["kind"] == "receive"]
    assert [e["now"] for e in receives] == [x for w in s["windows_seen"] for x in w["arrivals_ns"]]
    assert [e["step"] for e in receives] == [x for w in s["record"]["profile"]["windows"] for x in w["steps"]]
    assert made == {"plans": "recorded lengths", "steps": "recorded"}


@pytest.mark.parametrize("name", ["honest-paced", "order-past-pace"])
def test_case_of_keeps_a_paced_sessions_pace(server, tmp_path, name) -> None:
    """A paced session's pace, from its record (accepted) or the pace it ran under (`pace_of`'s k and T), so the case
    holds each window to its k roots."""
    v = verifier_json(run(name))
    case, _ = R.case_of(v["registration"], v["sessions"][0])
    assert case["params"]["pace"] == {"k": 2, "window_ns": 10}
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["agree"] and row["held"], row


def test_case_of_reads_the_pace_the_pace_k_driver_wrote(server, tmp_path) -> None:
    """The `--pace K` driver wrote a session's pace as its `audit.Pace` (`{k, window_ns}`), not
    `pace_of`'s k and T: a rejected session, with no record to read it from, replays under it. Its `order` isn't in
    what the record keeps (the books keep no refused root), so the replay holds without reaching it."""
    v = verifier_json(run("order-past-pace"))
    v["sessions"][0]["pace"] = {"k": 2, "window_ns": 10}
    case, _ = R.case_of(v["registration"], v["sessions"][0])
    assert case["params"]["pace"] == {"k": 2, "window_ns": 10}
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(v))
    row, = R.replay_record(path, server)
    assert row["agree"] and row["held"], row
