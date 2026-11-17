"""A deployment's start of PoUS's parties, the timed rounds' and the wipe's (`ProtocolDef.starts`): the memory
attester's key refused unless the configuration pins it, before its tree or any other input is read, and the goal each
start that reads the served store or the card raises under."""

from __future__ import annotations

import hashlib
import json
import types

import pytest

from verity.channel import Refused
from verity.protocol import ProtocolDef
from verity.protocols.resource_accounting.memory.pous.attester import challenger_party as CP
from verity.protocols.resource_accounting.memory.pous.attester import challenger_session as CS
from verity.protocols.resource_accounting.memory.pous.attester import onsite
from verity.protocols.resource_accounting.memory.pous.attester import wipe_challenger as WC
from verity.protocols.resource_accounting.memory.pous.prover import challenger_prover as CPR
from verity.protocols.resource_accounting.memory.pous.prover import wipe_prover as WR
from verity.commitments.crypto import lms
from verity.protocols.resource_accounting.memory.pous.verifier import wipe_parties as WPS
from verity.protocols.resource_accounting.memory.pous.verifier import wipe_party as WP

POUS_KEY, WIPE_KEY = b"pous public key", b"wipe public key"


def _sha(key: bytes) -> str:
    return hashlib.sha512(key).hexdigest()


CONFIG = {"pin": [{"party": CS.CHALLENGER, "protocol": CS.COMPONENT, "scheme": "lms", "key-sha512": _sha(POUS_KEY)},
                  {"party": WPS.CHALLENGER, "protocol": WPS.COMPONENT, "scheme": "lms", "key-sha512": _sha(WIPE_KEY)}]}
REQUEST = {"node": "the prover's node"}


def key_state(state, public: bytes) -> None:
    """What `onsite.device_key` leaves in a state directory that holds a key, but its tree: the seed and the leaf state
    with its public key."""
    state.mkdir(parents=True, exist_ok=True)
    (state / "seed").write_bytes(bytes(lms.DEPLOYED.N))
    lms.LeafState.create(state / onsite.KEY_STATE, public, lms.DEPLOYED.H)


@pytest.fixture
def keyed(monkeypatch):
    """`onsite.device_key` handing back a signer whose public key is the one its state holds."""
    signer = types.SimpleNamespace(public_key=POUS_KEY)

    def device_key(state):
        signer.public_key = bytes.fromhex(json.loads((state / onsite.KEY_STATE).read_text())["public"])
        return signer

    monkeypatch.setattr(onsite, "device_key", device_key)
    return signer


@pytest.fixture
def untouched(monkeypatch):
    """`onsite.device_key` and each request failing the test if read."""
    def read(*_):
        raise AssertionError("read before the key was refused")

    monkeypatch.setattr(onsite, "device_key", read)
    monkeypatch.setattr(CP, "deployed_request", read)
    monkeypatch.setattr(WC, "deployed_request", read)


def test_every_party_of_both_protocols_has_a_start():
    assert set(CP.PROTOCOL.starts) == {CS.CHALLENGER}
    assert set(CPR.PROTOCOL.starts) == {CS.PROVER}
    assert set(WC.PROTOCOL.starts) == {WPS.CHALLENGER}
    assert set(WR.PROTOCOL.starts) == {WPS.PROVER}


@pytest.mark.parametrize("start, key", [(CP.start, POUS_KEY), (CPR.start, None), (WC.start, WIPE_KEY),
                                        (WR.start, None)])
def test_a_start_that_reads_the_served_store_or_the_card_raises_its_goal(start, key, keyed, tmp_path):
    if key is not None:
        key_state(tmp_path, key)
    with pytest.raises(NotImplementedError, match=CS.ONSITE_GOAL):
        start(CONFIG, {}, tmp_path)


@pytest.mark.parametrize("start, other", [(CP.start, WIPE_KEY), (WC.start, POUS_KEY)])
def test_a_challenger_start_refuses_a_wrong_key_or_none_before_reading_anything_else(start, other, untouched,
                                                                                      tmp_path):
    with pytest.raises(Refused) as e:
        start(CONFIG, {}, tmp_path)
    assert e.value.name == "key"
    key_state(tmp_path, other)
    with pytest.raises(Refused) as e:
        start(CONFIG, {}, tmp_path)
    assert e.value.name == "pin" and _sha(other) in e.value.detail


def test_a_pin_is_the_configurations_for_its_party_and_protocol():
    assert onsite.pin(CONFIG, CS.CHALLENGER, CS.COMPONENT) == _sha(POUS_KEY)
    assert onsite.pin(CONFIG, WPS.CHALLENGER, WPS.COMPONENT) == _sha(WIPE_KEY)
    with pytest.raises(Refused) as e:
        onsite.pin(CONFIG, CS.CHALLENGER, "bandwidth")
    assert e.value.name == "pin"


def test_the_challengers_key_is_refused_unless_pinned(keyed, tmp_path):
    key_state(tmp_path, POUS_KEY)
    assert onsite.signer(tmp_path, _sha(POUS_KEY)) is keyed
    with pytest.raises(Refused) as e:
        onsite.signer(tmp_path, _sha(WIPE_KEY))
    assert e.value.name == "pin"
    with pytest.raises(Refused) as e:
        onsite.signer(None, _sha(POUS_KEY))
    assert e.value.name == "state"
    (tmp_path / "seed").unlink()
    with pytest.raises(Refused) as e:
        onsite.signer(tmp_path, _sha(POUS_KEY))
    assert e.value.name == "key"


def test_the_rounds_challenger_starts_with_its_pinned_key_and_its_logs(keyed, monkeypatch, tmp_path):
    monkeypatch.setattr(CP, "deployed_request", lambda configuration, bound: dict(REQUEST))
    key_state(tmp_path, POUS_KEY)
    s = CP.start(CONFIG, {}, tmp_path)
    assert s.params == REQUEST
    assert s.capabilities["signer"] is keyed
    assert (tmp_path / "salts").exists() and (tmp_path / "sessions").exists()


def test_the_wipes_challenger_starts_with_its_own_pinned_key_and_its_log(keyed, monkeypatch, tmp_path):
    monkeypatch.setattr(WC, "deployed_request", lambda configuration, bound: dict(REQUEST))
    key_state(tmp_path, WIPE_KEY)
    s = WC.start(CONFIG, {}, tmp_path)
    assert s.params == REQUEST
    assert s.capabilities["signer"] is keyed and s.capabilities["reads"] == WP.READS
    assert (tmp_path / "sessions").exists()


def test_a_start_needs_a_factory():
    with pytest.raises(ValueError, match="no factory"):
        ProtocolDef(WPS.MANIFEST, {}, {WPS.PROVER: WR.start})
