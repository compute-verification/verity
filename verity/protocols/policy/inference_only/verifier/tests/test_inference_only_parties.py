"""The inference-only policy's manifests (`parties`): each one party, the verifier, as SPOT declares it, with no channel
and no message; v0's under the id SPOT lists the policy by in both composites; a placement holds the verifier to the
auditor's machine; and each manifest's id, its check and its parameters' keys are the Lean's."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity.protocol import Role, place, site_of
from verity.channel import Refused
from verity.protocols.policy.inference_only.verifier import parties as IO
from verity.protocols.proofs.spot.verifier import parties as SP

LEAN = Path(__file__).resolve().parents[1] / "nci" / "lean"
SITE = site_of([{"name": "auditor", "owner": "auditor", "provides": ["component"]},
                {"name": "node", "owner": "developer", "provides": ["component"]}])


MANIFESTS = (IO.MANIFEST, IO.ROWS_MANIFEST)


@pytest.mark.parametrize("m", MANIFESTS, ids=lambda m: m.protocol)
def test_the_manifest_is_the_verifier_alone(m) -> None:
    assert m.parties == (Role("verifier", "auditor", per="one", needs=("component",)),)
    assert m.channels == () and m.messages == () and not m.registry.types and m.peers(IO.VERIFIER) == ()


def test_the_verifier_is_spots_party_and_the_id_the_one_spot_lists() -> None:
    mine, theirs = IO.MANIFEST.role(IO.VERIFIER), SP.MANIFEST.role(SP.VERIFIER)
    assert (mine.name, mine.principal, mine.per) == (theirs.name, theirs.principal, theirs.per)
    assert set(mine.needs) <= set(theirs.needs)
    assert (IO.MANIFEST.protocol, IO.MANIFEST.version) == (SP.IDS["inference-only"], 1)
    assert all(subs["inference-only"] == IO.ID for subs in SP.SUBPROTOCOLS.values())


@pytest.mark.parametrize("m", MANIFESTS, ids=lambda m: m.protocol)
def test_it_places_the_verifier_on_the_auditors_machine(m) -> None:
    assert place(m, SITE, {IO.VERIFIER: "auditor"}).mode == "placed"
    with pytest.raises(Refused) as e:
        place(m, SITE, {IO.VERIFIER: "node"})
    assert e.value.name == "principal"
    with pytest.raises(Refused) as e:
        place(m, SITE, {})
    assert e.value.name == "unbound"


@pytest.mark.parametrize(("m", "params"), [
    (IO.MANIFEST, ["alpha", "gamma", "log", "parameters", "program"]),
    (IO.ROWS_MANIFEST, ["alpha", "circuits", "gamma", "log", "pins", "program"]),
], ids=["v0", "rows-v1"])
def test_its_id_its_check_and_its_parameters_keys_are_the_leans(m, params) -> None:
    exe, command, argument = m.params.split()
    assert (command, argument) == ("check", "PARAMS")
    lake = (LEAN / "lakefile.toml").read_text(encoding="utf-8")
    root = re.search(rf'^name = "{re.escape(exe)}"\nroot = "([\w.]+)Main"$', lake, re.MULTILINE)
    assert root, f"{exe} is no executable of {LEAN.name}/lakefile.toml"
    text = (LEAN / (root[1].replace(".", "/") + ".lean")).read_text(encoding="utf-8")
    assert f'def protocolId : String := "{m.protocol}"' in text
    keys = re.search(r"^def PARAMS_KEYS : Array String := #\[(.*)\]$", text, re.MULTILINE)
    assert keys and re.findall(r'"([a-z]+)"', keys[1]) == params
