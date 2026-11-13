"""The inference-only policy's manifest (`parties`): one party, the verifier, as sampled proofs declares it, with no
channel and no message; a placement holds it to the auditor's machine; and its id and its parameters' reader are the
Lean's (`NciCheck.Protocol`)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity.configuration.deploy.manifest import Role, place, site_of
from verity.configuration.wire import Refused
from verity.protocols.inference_only.verifier import parties as IO
from verity.protocols.spot.verifier import parties as SP

LEAN = Path(__file__).resolve().parents[1] / "nci" / "lean" / "NciCheck" / "Protocol.lean"
SITE = site_of([{"name": "auditor", "owner": "auditor", "provides": ["component"]},
                {"name": "node", "owner": "developer", "provides": ["component"]}])


def test_the_manifest_is_the_verifier_alone() -> None:
    m = IO.MANIFEST
    assert m.parties == (Role("verifier", "auditor", per="one", needs=("component",)),)
    assert m.channels == () and m.messages == () and not m.registry.types and m.peers(IO.VERIFIER) == ()


def test_the_verifier_is_sampled_proofs_party() -> None:
    mine, theirs = IO.MANIFEST.role(IO.VERIFIER), SP.MANIFEST.role(SP.VERIFIER)
    assert (mine.name, mine.principal, mine.per) == (theirs.name, theirs.principal, theirs.per)
    assert set(mine.needs) <= set(theirs.needs)
    assert IO.MANIFEST.protocol == SP.IDS["inference-only"]


def test_it_places_the_verifier_on_the_auditors_machine() -> None:
    assert place(IO.MANIFEST, SITE, {IO.VERIFIER: "auditor"}).mode == "placed"
    with pytest.raises(Refused) as e:
        place(IO.MANIFEST, SITE, {IO.VERIFIER: "node"})
    assert e.value.name == "principal"
    with pytest.raises(Refused) as e:
        place(IO.MANIFEST, SITE, {})
    assert e.value.name == "unbound"


def test_its_id_and_its_parameters_are_the_leans() -> None:
    text = LEAN.read_text(encoding="utf-8")
    assert f'def protocolId : String := "{IO.ID}"' in text
    module, _, name = IO.MANIFEST.params.partition(":")
    assert module == "NciCheck.Protocol" and f"namespace {module}" in text
    assert re.search(rf"^def {re.escape(name)} ", text, re.MULTILINE)
    keys = re.search(r"^def PARAMS_KEYS : Array String := #\[(.*)\]$", text, re.MULTILINE)
    assert keys and re.findall(r'"([a-z]+)"', keys[1]) == ["alpha", "gamma", "log", "parameters", "program"]
