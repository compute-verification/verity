"""What `two_stage.py second` writes of its own cost and takes of its cards, without a session: the replay's seconds
and cards are the replay's own record's (the command runs no replay), ``--devices`` reaches the audit as the cards'
indices or the command stops at its parser, and the rehearsal's check (`two_stage_small.costed`) holds a summary to
saying where each session proved and for how long. The interior's commitment by the caller's clock is
``test_two_stage_second``'s, and the sessions' own timing is the rehearsal's (``two_stage_small.sh``)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
TS = importlib.import_module("two_stage")
SM = importlib.import_module("two_stage_small")

SECOND = ["second", "--stage1", "s1.json", "--boundary", "class", "--capture", "capture", "--challenger", "local",
          "--main", "main.txt", "--m0-pythonpath", "path", "--bin", "flock-circuit", "--lean", "flock-verify", "--out", "out"]


def _replay(tmp_path: Path, **record) -> Path:
    f = tmp_path / "replay.json"
    f.write_text(json.dumps(record))
    return f


def test_the_replays_cost_is_its_own_records_seconds_and_cards(tmp_path):
    """``seconds`` is the replay's wall time over the window, its engine's load beside it and not in it; ``gpus`` the
    cards its engine held, which a record written before it named them gives as its tensor-parallel width."""
    cost = TS.replay_cost(_replay(tmp_path, replay_wall_s=41.5, engine_ready_s=300.25, gpus=2, tp=2, steps=7))
    assert cost == {"seconds": 41.5, "gpus": 2, "ready_s": 300.25}
    assert TS.replay_cost(_replay(tmp_path, replay_wall_s=3.0, engine_ready_s=1.0, tp=8))["gpus"] == 8


def test_a_replay_record_that_lacks_a_field_gives_none_for_it_never_a_guess(tmp_path):
    assert TS.replay_cost(_replay(tmp_path, steps=7)) == {"seconds": None, "gpus": None, "ready_s": None}


@pytest.fixture
def ran(monkeypatch) -> list:
    """The namespaces `main` hands the second stage, which a stand-in keeps and does not run."""
    got: list = []
    monkeypatch.setattr(TS, "_second", lambda a: got.append(a) or 0)
    return got


def test_second_takes_its_cards_from_its_command_line_as_their_indices(ran):
    assert TS.main([*SECOND, "--devices", "0,2,3,4,5,6,7", "--replay-record", "replay.json"]) == 0
    assert ran[-1].devices == [0, 2, 3, 4, 5, 6, 7] and ran[-1].replay_record == "replay.json"
    assert TS.main(SECOND) == 0
    assert ran[-1].devices is None and ran[-1].replay_record is None


@pytest.mark.parametrize("more, why", [
    (["--devices", "0,0"], "their indices, each once"),
    (["--devices", "GPU-1f0c"], "their indices, each once"),
    (["--devices", "0", "--firewall", "flock-firewall"], "drop --devices"),
])
def test_second_stops_at_its_parser_on_cards_it_cannot_prove_on(ran, capsys, more, why):
    """Cards named wrongly, or with sessions behind the ZK server (process pairs on the CPU), end the command before
    anything runs: it never proves on the CPU in silence where a card was asked for."""
    with pytest.raises(SystemExit) as e:
        TS.main([*SECOND, *more])
    assert e.value.code == 2 and why in capsys.readouterr().err and ran == []


def _summary(**timing) -> dict:
    return {"audit": {"timing": timing, "staging": {"before_draw_s": 1.5, "drawn_s": 0.5, "gpus": 0}},
            "interior_commit": {"seconds": 0.25, "gpus": 0}}


ON_CARD = {"session_s": 9.0, "busy_s": 4.0, "gpus": 1, "device": 2, "prove_seconds_source": "busy"}
ON_CPU = {"session_s": 9.0, "busy_s": 4.0, "gpus": 0, "device": None, "prove_seconds_source": "busy"}
PAIR = {"session_s": 9.0, "gpus": 0, "device": None, "prove_seconds_source": "session"}


def test_a_summary_is_costed_when_every_session_says_where_it_proved_and_for_how_long():
    assert SM.costed(_summary(m0=ON_CARD, m1=ON_CPU, m2=PAIR))


@pytest.mark.parametrize("broken", [
    {},
    {"m0": {k: v for k, v in ON_CARD.items() if k != "gpus"}},
    {"m0": {**ON_CARD, "device": None}},
    {"m0": {**ON_CPU, "device": 0}},
    {"m0": {**ON_CARD, "busy_s": None}},
    {"m0": {**PAIR, "prove_seconds_source": "busy"}},
    {"m0": {**ON_CARD, "prove_seconds_source": "wall"}},
])
def test_a_summary_missing_a_sessions_cards_or_its_named_seconds_is_not_costed(broken):
    assert not SM.costed(_summary(**broken))


def test_a_summary_without_the_stagings_cards_or_the_interiors_seconds_is_not_costed():
    good = _summary(m0=ON_CARD)
    assert not SM.costed({**good, "audit": {**good["audit"], "staging": {"before_draw_s": 1.5}}})
    assert not SM.costed({**good, "interior_commit": {"gpus": 0}})
    assert not SM.costed({k: v for k, v in good.items() if k != "interior_commit"})
