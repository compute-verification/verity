"""The wipe's attach (`pous.prover.wipe_served`): its settings are the protocol's parameters of `period.json`'s
``wipe`` and its launcher's facts the rest, and it refuses by name before any session outside its piece's process,
with no side, private directory, pin, fact or challenger binding, and on a busy card."""

from __future__ import annotations

import json
import os
import re

import pytest

from verity import hooks as H
from verity.protocols.accounting import scheme as S
from verity.protocols.accounting.memory.pous.prover import wipe_run
from verity.protocols.accounting.memory.pous.prover import wipe_served as WS
from verity.protocols.accounting.memory.pous.scheme import MEMORY_ATTESTER, WIPE
from verity.protocols.accounting.memory.pous.verifier.period import UNPINNED
from verity.protocols.accounting.memory.pous.verifier.wipe_period import LIMIT_NS

PIN = {"party": "memory-challenger", "protocol": "wipe", "scheme": "lms-sha512/n64-w1-h20/v0", "key-sha512": "ab" * 64}
#: the hash `run_key` stands in for here: making an LMS key at deployed parameters is the challenger's, not this suite's
RUN_KEY = "cd" * 64
BOUND = {"host": "auditor", "address": "auditor:7000", "command": ["ssh", "auditor"]}


def test_its_settings_and_facts_are_period_json_s_wipe_with_none_unset() -> None:
    """Every key of `period.json`'s ``wipe`` but the verifier's (the pin and the fill floor) is a parameter or a
    fact, none both, and their defaults are CONFIG's but for the served store: a card's (``cuda``), streamed natively.
    The declaration writes its own defaults, since loading it loads no prover file; this holds them to CONFIG and
    UNSET."""
    assert not set(WIPE.settings) & set(WS.FACTS) and WIPE.settings["limit_ns"] == LIMIT_NS
    got = WS.period_wipe(WIPE.settings | WS.FACTS)
    verifiers = {"trusted_sha512": None, "fill_bytes": None}
    assert set(got) | set(verifiers) == set(wipe_run.CONFIG)
    assert got | verifiers == wipe_run.CONFIG | {"store": "cuda", "streamer": "native"}
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
             "process": H.PIECE, "side_of": lambda *a: None, "pins": [PIN],
             "party": MEMORY_ATTESTER, "fill": 1 << 25}
            | WS.FACTS | {k: v for k, v in given.items() if k in WS.FACTS} | launcher)


@pytest.mark.parametrize("settings, launcher, why", [
    (BOUND, {"process": H.FRONT}, "it runs in its piece's process, before any engine is on the card, not in 'front'"),
    (BOUND, {"process": H.ENGINE}, "it runs in its piece's process, before any engine is on the card, not in 'engine'"),
    (BOUND, {"side_of": None}, "no challenger's side was handed to it"),
    (BOUND, {"private": None}, "no private directory was handed to it"),
    (BOUND, {"pins": None}, "no pin was handed to it (ctx['pins'], the application's [[pin]] entries)"),
    (BOUND, {"pins": [PIN | {"protocol": "pous"}]}, "the pins handed to it give the memory-challenger no key for the wipe"),
    (BOUND, {"wipe_stream": "/exec/wipe_stream"}, "wipe_stream is the streamer of a memory attester on this node; one on "
                                                  "its own host (host, address, command) builds its own"),
    (BOUND, {"fill": None}, "no fill floor for the card (ctx['fill'], the card file's store.wipe_fill)"),
    ({}, {}, "no challenger binding (host, address, command) and run_keys unset")])
def test_the_attach_refuses_by_name_before_any_session(tmp_path, monkeypatch, settings, launcher, why) -> None:
    monkeypatch.setattr(wipe_run, "subprotocol", lambda *a, **kw: pytest.fail("a session started"))
    with pytest.raises(ValueError, match=re.escape(why)):
        WIPE.attach(None, _ctx(tmp_path, settings, **launcher))


def test_a_context_without_the_party_is_refused_by_name(tmp_path) -> None:
    """The party comes from the scheme (its attach, or a launcher's check reading `MEMORY_ATTESTER`): a context without it
    is refused by name, never a KeyError."""
    ctx = {k: v for k, v in _ctx(tmp_path, BOUND).items() if k != "party"}
    assert "no party was handed to it (ctx['party'], the memory challenger's, which the scheme hands)" in WS.refusals(ctx)


def test_run_keys_needs_no_pin_and_no_binding() -> None:
    assert WS.refusals({k: v for k, v in _ctx("p", {"run_keys": True}).items() if k != "pins"}) == ()


def test_with_no_pin_for_the_party_the_refusal_states_the_pin_to_add(tmp_path) -> None:
    """The verifier trusts no key with no pin, so the attach refuses, and its message is the fix: the party's
    ``[[pin]]`` in the application's configuration file, each field by name."""
    ctx = _ctx(tmp_path, BOUND, pins=[PIN | {"protocol": "pous"}])
    [why] = [w for w in WS.refusals(ctx) if "no key for the wipe" in w]
    assert why == ("the pins handed to it give the memory-challenger no key for the wipe, so the verifier would trust "
                   "none: state the party's [[pin]] in the application's configuration file, with party = "
                   "\"memory-challenger\", protocol = \"wipe\", scheme = \"lms-sha512/n64-w1-h20/v0\" and key-sha512 = "
                   "the SHA-512 of its public key in 128 lowercase hex digits")


def test_a_run_keys_wipe_pins_the_key_its_challenger_signs_under(tmp_path, monkeypatch) -> None:
    """``run_keys`` states no pin from the configuration: the attach makes the challenger's key in the directory the
    challenger loads it from (`run_key`) and hands its SHA-512 as ``wipe.trusted_sha512``, never None."""
    seen, made = [], []

    def subprotocol(period, node, out, *, side_of, ready):
        seen.append(json.loads((period / "period.json").read_text())["wipe"]["trusted_sha512"])
        (out / "wipe.case.json").write_text("{}")
        (out / "wipe.json").write_text("{}")
        return {"verdict": "accept"}

    monkeypatch.setattr(wipe_run, "subprotocol", subprotocol)
    monkeypatch.setattr(WS, "run_key", lambda device: made.append(device.name) or RUN_KEY)
    WIPE.attach(None, _ctx(tmp_path, {"run_keys": True, "store": "python"}))
    assert seen == [RUN_KEY] and made == ["wipe-challenger"]


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


@pytest.mark.parametrize("launcher, tools", [
    ({}, {}), ({"judge": "/lean/wipe-protocol", "wipe_stream": "/exec/wipe_stream"},
               {"wipe-protocol": "/lean/wipe-protocol", "wipe-stream": "/exec/wipe_stream"})])
def test_the_period_names_the_judge_and_the_streamer_its_launcher_built(tmp_path, monkeypatch, launcher, tools) -> None:
    """The launcher's judge and streamer, built where programs run, are `period.json`'s ``tools``, which the memory
    attester on this node runs (`wipe_run.subprotocol`); without them it builds its own. Neither is a setting, so no
    application's hash reads them."""
    seen = []

    def subprotocol(period, node, out, *, side_of, ready):
        seen.append(json.loads((period / "period.json").read_text())["tools"])
        (out / "wipe.case.json").write_text("{}")
        (out / "wipe.json").write_text("{}")
        return {"verdict": "accept"}

    monkeypatch.setattr(wipe_run, "subprotocol", subprotocol)
    monkeypatch.setattr(WS, "run_key", lambda device: RUN_KEY)
    WIPE.attach(None, _ctx(tmp_path, {"run_keys": True, "store": "python"}, **launcher))
    assert seen == [tools] and not set(launcher) & set(WIPE.settings)


@pytest.mark.parametrize("facts, development", [
    (BOUND, None), ({"run_keys": True}, f"development run: ran with run_keys; {UNPINNED}"),
    ({"run_keys": True, "simulated": True}, f"development run: ran with run_keys; ran with simulated; {UNPINNED}"),
    (BOUND | {"foreign_bytes": 1}, "development run: ran with foreign_bytes")])
def test_a_record_names_the_development_facts_it_ran_under(tmp_path, monkeypatch, facts, development) -> None:
    """A record made under ``run_keys``, ``simulated``, a cheat or a key no pin gives says why in
    ``served.development``, in the file and the report alike, so a simulated wipe never reads as one, and its piece
    writes ``development.json`` in place of ``answer.json`` (`period.publish`); its placement's premise reads rehearsal
    besides (`wipe_party.placement`)."""
    def subprotocol(period, node, out, *, side_of, ready):
        (out / "wipe.case.json").write_text("{}")
        (out / "wipe.json").write_text("{}")
        return {"verdict": "accept"}

    monkeypatch.setattr(wipe_run, "subprotocol", subprotocol)
    monkeypatch.setattr(WS, "run_key", lambda device: RUN_KEY)
    got = WIPE.attach(None, _ctx(tmp_path, facts | {"store": "python"}))
    assert got.record()["served"]["development"] == development
    assert json.loads((tmp_path / "wipe.json").read_text()) == got.record()
