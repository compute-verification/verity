"""The trusted verifiers keep copies of what the one table declares, since a verifier loads no declaration: the sampled
proofs' verifier its table of each composite's ids (`parties.SUBPROTOCOLS`, which `ids_of` reads), and each
verifier's manifest the parties and channels its scheme declares. Each copy equals its declaration."""
from __future__ import annotations

import pytest

from verity.protocols.compose import schemes
from verity.protocols.proofs.spot.verifier import parties as SP
from verity.protocols.workload_classification.training_detection.inference_only.verifier import parties as IO

#: the names the composites list that no scheme declares yet, each with the lane that declares it; only shrinks
UNDECLARED = {"certifier": "network-accounting (declared as network-attester; the rename is pending)",
              "pouw": "compute-accounting"}
MANIFESTS = {"sampled-proofs": SP.MANIFEST, "inference-only": IO.MANIFEST}


def _id(name: str) -> str:
    s = schemes.get(name)
    return s.composed_protocol or s.protocol


def test_the_verifiers_composite_ids_are_the_one_tables() -> None:
    wrong = [f"parties.SUBPROTOCOLS[{c!r}][{n!r}] is {i!r}, and the one table declares {_id(n)!r}: set it to "
             f"{_id(n)!r}, or the declaration's composed_protocol to {i!r}"
             for c, subs in sorted(SP.SUBPROTOCOLS.items()) for n, i in sorted(subs.items())
             if n in schemes.DECLARATIONS and i != _id(n)]
    assert not wrong, "\n".join(wrong)


def test_the_undeclared_composite_names_only_shrink() -> None:
    names = {n for subs in SP.SUBPROTOCOLS.values() for n in subs if n not in schemes.DECLARATIONS}
    assert not names - set(UNDECLARED), (f"{sorted(names - set(UNDECLARED))} are in parties.SUBPROTOCOLS and not in the "
                                         "one table (verity.protocols.compose.schemes): declare each one there")
    assert not set(UNDECLARED) - names, (f"{sorted(set(UNDECLARED) - names)} are declared now: delete each from "
                                         "UNDECLARED here")


@pytest.mark.parametrize("name", sorted(MANIFESTS))
def test_a_verifiers_manifest_is_its_declarations_copy(name: str) -> None:
    s, m = schemes.get(name), MANIFESTS[name]
    for field, mine, theirs in (("parties", m.parties, s.parties), ("channels", m.channels, s.channels)):
        assert mine == theirs, (f"{m.protocol}'s MANIFEST.{field} differs from the {name} declaration "
                                f"({schemes.DECLARATIONS[name]}), the source: copy the declaration's into the "
                                f"manifest\n  manifest:    {mine}\n  declaration: {theirs}")
