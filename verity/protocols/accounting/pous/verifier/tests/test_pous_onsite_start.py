"""A deployment's start of PoUS's parties, the timed rounds' and the wipe's (`ProtocolDef.starts`): the memory
challenger's key refused unless the configuration pins it, the wipe's raw clock, and the goal each start that reads the
served store or the card raises under."""

from __future__ import annotations

import hashlib
import types

import pytest

from verity.channel import Refused
from verity.protocol import ProtocolDef
from verity.protocols.accounting.pous.device import challenger_party as CP
from verity.protocols.accounting.pous.device import challenger_session as CS
from verity.protocols.accounting.pous.device import onsite
from verity.protocols.accounting.pous.device import wipe_challenger as WC
from verity.protocols.accounting.pous.prover import challenger_prover as CPR
from verity.protocols.accounting.pous.prover import wipe_prover as WR
from verity.protocols.accounting.pous.verifier import streamed_wipe as sw
from verity.protocols.accounting.pous.verifier import wipe_parties as WPS
from verity.protocols.accounting.pous.verifier import wipe_party as WP

POUS_KEY, WIPE_KEY = b"pous public key", b"wipe public key"


def _sha(key: bytes) -> str:
    return hashlib.sha512(key).hexdigest()


CONFIG = {"pin": [{"party": CS.CHALLENGER, "protocol": CS.COMPONENT, "scheme": "lms", "key-sha512": _sha(POUS_KEY)},
                  {"party": WPS.CHALLENGER, "protocol": WPS.COMPONENT, "scheme": "lms", "key-sha512": _sha(WIPE_KEY)}]}
REQUEST = {"node": "the prover's node"}


@pytest.fixture
def keyed(monkeypatch):
    """`onsite.device_key` handing back a signer whose public key is set per test."""
    signer = types.SimpleNamespace(public_key=POUS_KEY)
    monkeypatch.setattr(onsite, "device_key", lambda state: signer)
    return signer


def test_every_party_of_both_protocols_has_a_start():
    assert set(CP.PROTOCOL.starts) == {CS.CHALLENGER}
    assert set(CPR.PROTOCOL.starts) == {CS.PROVER}
    assert set(WC.PROTOCOL.starts) == {WPS.CHALLENGER}
    assert set(WR.PROTOCOL.starts) == {WPS.PROVER}


@pytest.mark.parametrize("start", [CP.start, CPR.start, WC.start, WR.start])
def test_a_start_that_reads_the_served_store_or_the_card_raises_its_goal(start, tmp_path):
    with pytest.raises(NotImplementedError, match=CS.ONSITE_GOAL):
        start(CONFIG, {}, tmp_path)


def test_a_pin_is_the_configurations_for_its_party_and_protocol():
    assert onsite.pin(CONFIG, CS.CHALLENGER, CS.COMPONENT) == _sha(POUS_KEY)
    assert onsite.pin(CONFIG, WPS.CHALLENGER, WPS.COMPONENT) == _sha(WIPE_KEY)
    with pytest.raises(Refused) as e:
        onsite.pin(CONFIG, CS.CHALLENGER, "bandwidth")
    assert e.value.name == "pin"


def test_the_challengers_key_is_refused_unless_pinned(keyed, tmp_path):
    assert onsite.signer(tmp_path, _sha(POUS_KEY)) is keyed
    with pytest.raises(Refused) as e:
        onsite.signer(tmp_path, _sha(WIPE_KEY))
    assert e.value.name == "pin"
    with pytest.raises(Refused) as e:
        onsite.signer(None, _sha(POUS_KEY))
    assert e.value.name == "state"


def test_the_rounds_challenger_starts_with_its_pinned_key_and_its_logs(keyed, monkeypatch, tmp_path):
    monkeypatch.setattr(CP, "deployed_request", lambda configuration, bound: dict(REQUEST))
    s = CP.start(CONFIG, {}, tmp_path)
    assert s.params == REQUEST and s.clock is None
    assert s.capabilities["signer"] is keyed
    assert (tmp_path / "salts").exists() and (tmp_path / "sessions").exists()
    keyed.public_key = WIPE_KEY
    with pytest.raises(Refused, match="pins"):
        CP.start(CONFIG, {}, tmp_path)


def test_the_wipes_challenger_starts_on_the_raw_clock_with_its_own_pin(keyed, monkeypatch, tmp_path):
    monkeypatch.setattr(WC, "deployed_request", lambda configuration, bound: dict(REQUEST))
    keyed.public_key = WIPE_KEY
    s = WC.start(CONFIG, {}, tmp_path)
    assert s.params == REQUEST
    assert s.capabilities["signer"] is keyed and s.capabilities["reads"] == WP.READS
    assert (tmp_path / "sessions").exists()
    assert s.clock is sw.raw_clock_ns
    keyed.public_key = POUS_KEY
    with pytest.raises(Refused, match="pins"):
        WC.start(CONFIG, {}, tmp_path)


def test_a_start_needs_a_factory():
    with pytest.raises(ValueError, match="no factory"):
        ProtocolDef(WPS.MANIFEST, {}, {WPS.PROVER: WR.start})
