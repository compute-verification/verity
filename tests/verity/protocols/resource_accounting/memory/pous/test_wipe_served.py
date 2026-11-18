"""The wipe's attach (`pous.prover.wipe_served`): its settings are `period.json`'s ``wipe``, and it refuses by name
before any session outside the engine's process, with no side, private directory, pin or challenger binding, and on a
busy card."""

from __future__ import annotations

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


def test_its_settings_are_period_json_s_wipe_with_none_unset() -> None:
    """Every key of `period.json`'s ``wipe`` but the pin is a setting, and its defaults are CONFIG's but for the served
    store: a card's (``cuda``), streamed natively."""
    got = WS.period_wipe(WIPE.settings)
    assert set(got) | {"trusted_sha512"} == set(wipe_run.CONFIG) and WIPE.settings["limit_ns"] == LIMIT_NS
    assert got | {"trusted_sha512": None} == wipe_run.CONFIG | {"store": "cuda", "streamer": "native"}
    assert WS.period_wipe(WIPE.settings | {"foreign_past_bound": True, "simulated": True}) == got | {
        "foreign_bytes": "fM", "placement": "simulated"}


@pytest.mark.parametrize("given, why", [
    ({"host": "auditor"}, "host, address and command bind the challenger together, not ['host'] alone"),
    (BOUND | {"run_keys": True}, "run_keys and simulated run the challenger on this node, not on the bound host"),
    ({"foreign_past_bound": True, "foreign_bytes": 1}, "foreign_past_bound is the cheat one label past the bound"),
    ({"store": "tape"}, "store is python, host, cuda or disk"),
    ({"reads": "5500"}, "wipe.reads is a int, not '5500'")])
def test_settings_it_refuses_are_refused_by_name(given, why) -> None:
    with pytest.raises(ValueError, match=re.escape(why)):
        S.settings(WIPE, given)


def _ctx(tmp_path, settings=None, **launcher):
    return {"settings": S.settings(WIPE, settings or {}), "private": tmp_path, "process": H.ENGINE,
            "side_of": lambda *a: None, "pins": [PIN]} | launcher


@pytest.mark.parametrize("settings, launcher, why", [
    (BOUND, {"process": H.FRONT}, "it runs in the engine's process, before the model loads, not in 'front'"),
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
    assert WS.refusals({"settings": S.settings(WIPE, {"run_keys": True}), "process": H.ENGINE, "side_of": print,
                        "private": "p"}) == ()


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
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
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
    smi.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    smi.setattr(wipe_run, "subprotocol", lambda *a, **kw: pytest.fail("a session started"))
    with pytest.raises(ValueError, match=re.escape("wipe: the card GPU-1234 isn't idle: process 4242 on it")):
        WIPE.attach(None, _ctx(tmp_path, BOUND))
