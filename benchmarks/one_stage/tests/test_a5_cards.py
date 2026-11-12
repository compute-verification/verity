"""An audit's cards: ``--devices`` is the one place it takes them from, SPOT's proof server proves on them (one CPU
prover without them, as before), a session the proof server doesn't prove refuses them by name, and the summary says of
each session where it proved and which of its seconds are its proving's. No card is needed here: the proof server is a
stand-in that keeps what it was given, and nvidia-smi's answer is handed in."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a0  # noqa: E402
import a5  # noqa: E402
from verity_flock.proof_server import Session  # noqa: E402

PERFORMANCE = """
==============NVSMI LOG==============

Driver Version                                         : 580.173.02

Attached GPUs                                          : 8
GPU 00000000:DB:00.0
    Performance State                                  : P0
    Clocks Event Reasons
        Idle                                           : Not Active
        HW Slowdown                                    : Not Active
            HW Thermal Slowdown                        : Not Active
        SW Thermal Slowdown                            : Not Active
    Clocks Event Reasons Counters
        SW Power Capping                               : 1038546 us
        Sync Boost                                     : 0 us
        SW Thermal Slowdown                            : 1038546 us
        HW Thermal Slowdown                            : 0 us
        HW Power Braking                               : 12 us
    Sparse Operation Mode                              : N/A
"""


def _args(**more) -> argparse.Namespace:
    return argparse.Namespace(**{"bin": "flock-circuit", "plain": False, "firewall": None, **more})


@pytest.fixture
def made(monkeypatch) -> list[dict]:
    """Every proof server `a5.proof_server` makes, as its arguments."""
    got: list[dict] = []
    monkeypatch.setattr(a5, "ProofServer", lambda bin_, out, **kw: got.append({"bin": bin_, "out": out, **kw}) or got[-1])
    return got


def test_devices_names_the_cards_by_index_each_once():
    assert a5.cards("0") == [0] and a5.cards("0,2,3,4,5,6,7") == [0, 2, 3, 4, 5, 6, 7] and a5.cards(" 3 , 1 ") == [3, 1]
    for text in ("", "0,", "0,0", "a", "GPU-1f0c", "-1", "0 1", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError, match="their indices, each once"):
            a5.cards(text)


def test_an_audit_without_devices_proves_on_one_cpu_prover_as_before(tmp_path, made):
    """No ``devices`` (an older caller's namespace has none), None and an empty list are all the CPU: the proof server
    gets no devices and the environment it had."""
    for a in (_args(), _args(devices=None), _args(devices=[])):
        assert a5.devices(a) is None and a5.devices_refusal(a) is None
        assert a5.proof_server(a, tmp_path) is made[-1]
        assert made[-1] == {"bin": "flock-circuit", "out": tmp_path / "proof-server", "devices": None, "env": a0.DIRECT,
                            "challenger": a0.own_challenger(zk=True)}


def test_an_audits_cards_go_to_the_proof_server_numbered_as_nvidia_smi_numbers_them(tmp_path, made):
    a = _args(devices=[0, 2, 3])
    a5.proof_server(a, tmp_path)
    assert made[-1]["devices"] == [0, 2, 3] == a5.devices(a)
    assert made[-1]["env"] == {**a0.DIRECT, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


@pytest.mark.parametrize("how", [{"plain": True}, {"firewall": "flock-firewall"}])
def test_a_session_the_proof_server_does_not_prove_takes_no_card_and_says_so(tmp_path, made, how):
    """A plain session and one behind the ZK server are process pairs on the CPU: without cards they are as before (no
    proof server), and with cards the audit is refused by name, never run on the CPU in silence."""
    assert a5.proof_server(_args(**how), tmp_path) is None and a5.devices_refusal(_args(**how)) is None
    a = _args(devices=[0], **how)
    assert "drop --devices" in a5.devices_refusal(a)
    with pytest.raises(ValueError, match="drop --devices"):
        a5.proof_server(a, tmp_path)
    assert made == []


def test_the_timing_says_where_a_session_proved_and_which_seconds_are_its_proving():
    """On the proof server a session's proving is its share of its prover's busy seconds (its ``session_s`` holds its
    wait behind the sessions before it); a process pair of its own has only its session's seconds. ``gpus`` is always
    written, 0 on the CPU, and ``device`` is the card a figure came from."""
    on_card, on_cpu = Session("m0", "pin", 2, None, "prover-gpu2", 0.0), Session("m1", "pin", None, None, "prover-cpu", 0.0)
    on_card.busy_seconds, on_cpu.busy_seconds = 4.25, 7.5
    assert a5.proving_of(on_card) == {"gpus": 1, "device": 2, "busy_s": 4.25, "prove_seconds_source": "busy"}
    assert a5.proving_of(on_cpu) == {"gpus": 0, "device": None, "busy_s": 7.5, "prove_seconds_source": "busy"}
    assert a5.proving_of(None) == {"gpus": 0, "device": None, "prove_seconds_source": "session"}


def test_a_cards_clock_counters_are_read_by_its_index_as_nvidia_smi_prints_them():
    asked: list[list[str]] = []

    def run(cmd: list[str]) -> str:
        asked.append(cmd)
        return PERFORMANCE

    counters = {"SW Power Capping": 1038546, "Sync Boost": 0, "SW Thermal Slowdown": 1038546, "HW Thermal Slowdown": 0,
                "HW Power Braking": 12}
    assert a5.card_clocks([0, 2], run) == {"0": counters, "2": counters}
    assert asked == [["nvidia-smi", "-q", "-d", "PERFORMANCE", "-i", card] for card in ("0", "2")]
    # a driver that prints no counters has none: the states above them are not counters
    assert a5.card_clocks([0], lambda cmd: PERFORMANCE.partition("    Clocks Event Reasons Counters")[0]) == {"0": {}}


def test_no_cards_or_no_nvidia_smi_is_no_clock_record():
    def gone(cmd: list[str]) -> str:
        raise FileNotFoundError(cmd[0])

    assert a5.card_clocks(None) is None and a5.card_clocks([]) is None and a5.card_clocks([0], gone) is None
