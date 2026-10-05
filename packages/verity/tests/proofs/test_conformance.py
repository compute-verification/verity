"""``typed-obligation/v0``: the Python column of the conformance suite over ``fixtures/typed-obligation-v0``.

Every vector records what each interpreter must agree on for one object (canonical bytes, digests, verdict or
code, lowering, statement bytes/digest, bound public values, relation and witness verdicts).  The Rust column is
``backends/sp1/common/tests/typed_obligation_v0.rs`` over the same files.  Vectors of families this repository
does not carry (the ``int8-q7`` toy protocol) must be rejected as ``E_UNSUPPORTED_FAMILY`` -- or, for reject
vectors, with the recorded code -- rather than silently skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.proofs.conformance import interpret
from verity.proofs.typed_obligation import SUPPORTED_FAMILIES

FIXTURES = Path(__file__).resolve().parents[4] / "fixtures" / "typed-obligation-v0"


def fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("*.json"))


def test_fixture_directory_is_populated_and_every_family_is_carried() -> None:
    files = fixture_files()
    assert len(files) >= 140, FIXTURES
    families = set()
    for path in files:
        record = json.loads(path.read_text())
        if record.get("code") == "E_UNSUPPORTED_FAMILY":
            continue  # the vector that pins the rejection of an unknown family
        families.add(record["object"].get("family", {}).get("id"))
    families.discard(None)  # malformed-family reject vectors
    uncarried = families - set(SUPPORTED_FAMILIES)
    assert not uncarried, uncarried


@pytest.mark.parametrize("path", fixture_files(), ids=lambda p: p.stem)
def test_python_column_agrees_with_every_vector(path: Path) -> None:
    record = json.loads(path.read_text())
    got = interpret(record)
    assert got.canonical_ok and got.digest_ok
    assert got.subcircuits_well_formed == record.get("subcircuits_well_formed")
    assert got.verdict == record["verdict"]
    if record["verdict"] == "reject":
        assert got.code == record["code"]
        return
    assert got.lowering == record["lowering"], "lower() must reproduce the recorded backend data"
    assert got.statement == record["statement"]
    assert got.statement_digest == record["statement_digest"]
    binding = record["binding"]
    assert binding["object_digest"] == record["digest"] and binding["statement_digest"] == record["statement_digest"]
    assert got.bound_public_values == binding["object_digest"] + binding["statement_digest"] + "01"
    if "code" in binding:
        assert got.binding == binding["code"]
    else:
        assert binding["public_values"] == got.bound_public_values and got.binding == "accept"
    if "values" in record:
        assert got.relations == record["relations"]
    if "witness" in record:
        assert got.witness_verdict == record["witness_verdict"]
