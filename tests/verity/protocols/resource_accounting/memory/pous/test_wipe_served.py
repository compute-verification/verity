"""The wipe's attach (`pous.prover.wipe_served`): its settings are the protocol's parameters of `period.json`'s
``wipe`` and its launcher's facts the rest, and it refuses by name before any session outside its piece's process,
with no side, private directory, pin, fact or challenger binding, and on a busy card."""

from __future__ import annotations

import json
import os
import re

import pytest

from verity import hooks as H
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.memory.pous.prover import wipe_run
from verity.protocols.resource_accounting.memory.pous.prover import wipe_served as WS
from verity.protocols.resource_accounting.memory.pous.scheme import WIPE
from verity.protocols.resource_accounting.memory.pous.verifier.wipe_period import LIMIT_NS

PIN = {"party": "memory-challenger", "protocol": "wipe", "scheme": "lms-sha512/n64-w1-h20/v0", "key-sha512": "ab" * 64}
BOUND = {"host": "auditor", "address": "auditor:7000", "command": ["ssh", "auditor"]}


def test_its_settings_and_facts_are_period_json_s_wipe_with_none_unset() -> None:
    """Every key of `period.json`'s ``wipe`` but the pin is a parameter or a fact, none both, and their defaults are
    CONFIG's but for the served store: a card's (``cuda``), streamed natively."""
    assert not set(WIPE.settings) & set(WS.FACTS) and WIPE.settings["limit_ns"] == LIMIT_NS
    got = WS.period_wipe(WIPE.settings | WS.FACTS)
    assert set(got) | {"trusted_sha512"} == set(wipe_run.CONFIG)
    assert got | {"trusted_sha512": None} == wipe_run.CONFIG | {"store": "cuda", "streamer": "native"}
    assert WS.period_wipe(WIPE.settings | WS.FACTS | {"foreign_past_bound": True, "simulated": True}) == got | {
        "foreign_bytes": "fM", "placement": "simulated"}


def test_an_application_states_the_protocols_parameters_alone() -> None:
    """What an application states, and its hash covers, changes what the wipe proves: which memory it fills and how
    much (``store``, ``tier``, ``segments``, ``reserve_bytes``), the deadline a read and the reads. Where and how a node
    runs it (its card, links, disks, challenger's address, streamer, the cheats and the rehearsals) is a launcher's."""
    assert set(WIPE.settings) == {"store", "tier", "segments", "reserve_bytes", "limit_ns", "reads"}
    assert {"host", "address", "command", "links", "disk", "dram", "device"} <= set(WS.FACTS)


def test_the_wipe_keeps_no_weight_copy() -> None:
    """The wipe holds labels derived from a key fresh for the wipe, and closes its store and the card's memory before
    its attach returns: it touches no weight, so storage may attach beside it (vllm's `weight_copy_refusals`)."""
    assert WIPE.keeps_weight_copy is False


@pytest.mark.parametrize("given, why", [
    ({"store": "tape"}, "store is python, host, cuda or disk"),
    ({"reads": "5500"}, "wipe.reads is a int, not '5500'"),
    ({"host": "auditor"}, "wipe has no setting ['host']")])
def test_settings_it_refuses_are_refused_by_name(given, why) -> None:
    with pytest.raises(ValueError, match=re.escape(why)):
        S.settings(WIPE, given)


@pytest.mark.parametrize("facts, why", [
    ({"host": "auditor"}, "host, address and command bind the challenger together, not ['host'] alone"),
    (BOUND | {"run_keys": True}, "run_keys and simulated run the challenger on this node, not on the bound host"),
    ({"foreign_past_bound": True, "foreign_bytes": 1}, "foreign_past_bound is the cheat one label past the bound"),
    ({"idle_mib": "1024"}, "idle_mib is a int, not '1024'")])
def test_facts_it_refuses_are_refused_by_name(tmp_path, facts, why) -> None:
    assert any(why in w for w in WS.refusals(_ctx(tmp_path, facts)))


def test_a_fact_its_launcher_doesnt_hand_is_refused_by_name(tmp_path) -> None:
    ctx = {k: v for k, v in _ctx(tmp_path, BOUND).items() if k not in ("device", "links")}
    assert WS.refusals(ctx) == ("no device, links handed to it: the launcher's facts, each in ctx by name",)


def _ctx(tmp_path, given=None, **launcher):
    given = given or {}
    return ({"settings": S.settings(WIPE, {k: v for k, v in given.items() if k in WIPE.settings}), "private": tmp_path,
             "process": H.PIECE, "side_of": lambda *a: None, "pins": [PIN]}
            | WS.FACTS | {k: v for k, v in given.items() if k in WS.FACTS} | launcher)


@pytest.mark.parametrize("settings, launcher, why", [
    (BOUND, {"process": H.FRONT}, "it runs in its piece's process, before any engine is on the card, not in 'front'"),
    (BOUND, {"process": H.ENGINE}, "it runs in its piece's process, before any engine is on the card, not in 'engine'"),
    (BOUND, {"side_of": None}, "no challenger's side was handed to it"),
    (BOUND, {"private": None}, "no private directory was handed to it"),
    (BOUND, {"pins": None}, "no pin was handed to it (ctx['pins'], the application's [[pin]] entries)"),
    (BOUND, {"pins": [PIN | {"protocol": "pous"}]}, "the pins handed to it give the memory-challenger no key for the wipe"),
    ({}, {}, "no challenger binding (host, address, command) and run_keys unset")])
def test_the_attach_refuses_by_name_before_any_session(tmp_path, monkeypatch, settings, launcher, why) -> None:
    monkeypatch.setattr(wipe_run, "subprotocol", lambda *a, **kw: pytest.fail("a session started"))
    with pytest.raises(ValueError, match=re.escape(why)):
        WIPE.attach(None, _ctx(tmp_path, settings, **launcher))


def test_run_keys_needs_no_pin_and_no_binding() -> None:
    assert WS.refusals({k: v for k, v in _ctx("p", {"run_keys": True}).items() if k != "pins"}) == ()


SMI = """#!/bin/sh
case "$*" in
  *compute-apps*) printf '%s' "$FAKE_SMI_APPS" ;;
  *memory.used*) echo "$FAKE_SMI_USED" ;;
esac
"""


@pytest.fixture
def smi(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text(SMI)
    (bin_ / "nvidia-smi").chmod(0o755)
    monkeypatch.setattr(WS, "SMI", str(bin_ / "nvidia-smi"))
    monkeypatch.setenv("FAKE_SMI_APPS", "")
    monkeypatch.setenv("FAKE_SMI_USED", "0")
    return monkeypatch


def test_a_card_with_another_process_or_its_memory_in_use_is_busy(smi) -> None:
    """Another process on the card, or more than ``idle_mib`` of it in use, is busy; this process's own context
    isn't another's."""
    assert WS.busy("0", 1024) is None
    smi.setenv("FAKE_SMI_APPS", f"{os.getpid()}\n")
    assert WS.busy("0", 1024) is None
    smi.setenv("FAKE_SMI_APPS", "4242\n")
    assert WS.busy("0", 1024) == "the card 0 isn't idle: process 4242 on it, and the wipe kills nothing"
    smi.setenv("FAKE_SMI_APPS", "")
    smi.setenv("FAKE_SMI_USED", "2048")
    assert WS.busy("0", 1024) == "the card 0 isn't idle: 2048 MiB of its memory in use, past idle_mib 1024"


def test_on_a_busy_card_the_attach_refuses_before_any_session(tmp_path, smi) -> None:
    smi.setenv("FAKE_SMI_APPS", "4242\n")
    smi.setattr(WS, "card_of", lambda ordinal: "GPU-1234")
    smi.setattr(wipe_run, "subprotocol", lambda *a, **kw: pytest.fail("a session started"))
    with pytest.raises(ValueError, match=re.escape("wipe: the card GPU-1234 isn't idle: process 4242 on it")):
        WIPE.attach(None, _ctx(tmp_path, BOUND))


@pytest.mark.parametrize("launcher, want", [({}, WS.built), ({"ready": print}, print)])
def test_the_judge_answers_once_ready_returns_built_unless_its_launcher_hands_its_own(tmp_path, monkeypatch, launcher,
                                                                                         want) -> None:
    """`wipe_run.subprotocol` calls ``ready`` before the judge answers; the attach hands it `built`, which waits for a
    judge `served_wipe.sh` still builds beside the wipe, or the launcher's ``ready``."""
    seen = []

    def subprotocol(period, node, out, *, side_of, ready):
        seen.append(ready)
        (out / "wipe.case.json").write_text("{}")
        (out / "wipe.json").write_text("{}")
        return {"verdict": "accept"}

    monkeypatch.setattr(wipe_run, "subprotocol", subprotocol)
    WIPE.attach(None, _ctx(tmp_path, BOUND | {"store": "python"}, **launcher))
    assert seen == [want]


@pytest.mark.parametrize("facts, development", [
    (BOUND, []), ({"run_keys": True}, ["run_keys"]), ({"run_keys": True, "simulated": True}, ["run_keys", "simulated"]),
    (BOUND | {"foreign_bytes": 1}, ["foreign_bytes"])])
def test_a_record_names_the_development_facts_it_ran_under(tmp_path, monkeypatch, facts, development) -> None:
    """A record made under ``run_keys``, ``simulated`` or a cheat says so in ``served.development``, in the file and
    the report alike, so a simulated wipe never reads as one; its placement's premise reads rehearsal besides
    (`wipe_party.placement`)."""
    def subprotocol(period, node, out, *, side_of, ready):
        (out / "wipe.case.json").write_text("{}")
        (out / "wipe.json").write_text("{}")
        return {"verdict": "accept"}

    monkeypatch.setattr(wipe_run, "subprotocol", subprotocol)
    got = WIPE.attach(None, _ctx(tmp_path, facts | {"store": "python"}))
    assert got.record()["served"]["development"] == development
    assert json.loads((tmp_path / "wipe.json").read_text()) == got.record()
