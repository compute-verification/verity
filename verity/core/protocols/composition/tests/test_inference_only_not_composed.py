"""No composite composes a protocol that publishes private data in the clear (`IN_THE_CLEAR`) unless `KNOWN` lists it
with its reason. The inference-only policy's public form, `verity/inference-only/v0` (`NciCheck.Protocol.protocol`),
publishes the declared division, and each inference block's gates and fresh-input bits, so it isn't zero-knowledge; both
composites compose it for the run of record of 28 Oct only, a stated exception by Daniel's ruling: "Let's run it in
the clear tonight as a stated exception." v1's private form replaces it after 15:00Z on 28 Oct. The
composites' components are read wherever they are listed: their Lean (`Core.Onsite.components` and
`Core.Remote.components`, with the ids each one's `#guard` pins), the reference's mirror of
them (`reference.ONSITE`, `REMOTE`), and the ids a configuration may name as components (`parties.IDS`, which step0's
`MANIFESTS` lists), read from its source: importing `parties` would read the deploy runtime, which this suite doesn't
declare."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

import reference as R

PROTOCOLS = Path(__file__).resolve().parents[2]
V0 = "verity/inference-only/v0"
PRIVATE_ONLY = ("only the private form (v1: division committed under hiding leaves, compliance proved, padded profile) "
                "may be composed")
COMPOSITES = {"Core.Onsite": PROTOCOLS / "onsite/CoreOnsite.lean", "Core.Remote": PROTOCOLS / "remote/CoreRemote.lean"}
#: The protocols that publish private data in the clear, each with what it publishes.
IN_THE_CLEAR = {V0: "the declared division, and each inference block's gates and fresh-input bits"}
#: The protocols of `IN_THE_CLEAR` a composite may compose, each with its reason. KNOWN only shrinks: an entry no
#: composite composes any more fails `test_known_only_shrinks`.
KNOWN = {"verity/inference-only/v0": "stated exception for tonight's run of record (Daniel's ruling); v1's private form "
                                     "replaces it"}


def v0_names() -> set[str]:
    """v0's Lean names: `NciCheck.Protocol.protocol` while its `protocolId` is v0."""
    text = (PROTOCOLS / "network/nci/lean/NciCheck/Protocol.lean").read_text()
    m = re.search(r'^def protocolId : String := "([^"]*)"', text, re.M)
    assert m, "NciCheck/Protocol.lean no longer defines `protocolId` as a string literal: read v0's id where it now is"
    return {"NciCheck.Protocol.protocol"} if m[1] == V0 else set()


def lean_components(path: Path) -> tuple[list[str], list[str]]:
    """A composite's `components`, the Lean names it lists, and the ids its `#guard` pins them to."""
    text = path.read_text()
    names = re.search(r"^def components\s*:[^\n]*?:=\s*\[(.*?)\]", text, re.M | re.S)
    ids = re.search(r"^#guard components\.map \(·\.id\) ==\s*\[(.*?)\]", text, re.M | re.S)
    assert names and ids, f"{path.name}: no `def components : ... := [...]` or no `#guard` on their ids"
    return [n.strip() for n in names[1].split(",")], re.findall(r'"([^"]*)"', ids[1])


def configuration_ids() -> list[str]:
    """The ids `parties.IDS` gives as string literals."""
    for node in ast.parse((PROTOCOLS / "sampled_proofs/parties.py").read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["IDS"]:
            assert isinstance(node.value, ast.Dict), "parties.IDS is no longer a dict literal: read its ids as it is now"
            return [v.value for v in node.value.values if isinstance(v, ast.Constant)]
    raise AssertionError("parties.py no longer assigns `IDS`: read a configuration's ids where they now are")


def unlisted(where: str, ids: list[str]) -> list[str]:
    """Each protocol of ``ids`` that publishes private data in the clear and `KNOWN` doesn't list, said."""
    return [f"{where} composes {i}, which publishes {IN_THE_CLEAR[i]} in the clear: "
            + (PRIVATE_ONLY if i == V0 else "compose its private form") + ", or list it in KNOWN with its reason"
            for i in ids if i in IN_THE_CLEAR and i not in KNOWN]


def composed() -> set[str]:
    """Every id a composite's Lean or the reference composes."""
    return {i for p in COMPOSITES.values() for i in lean_components(p)[1]} | {
        C.id for p in (R.ONSITE, R.REMOTE) for C in p.components}


@pytest.mark.parametrize("composite", sorted(COMPOSITES))
def test_lean_components(composite):
    names, ids = lean_components(COMPOSITES[composite])
    assert len(names) == len(ids) > 0, f"{composite}: {names} and the ids {ids} its `#guard` pins don't pair up"
    assert not unlisted(f"{composite}.components", ids), unlisted(f"{composite}.components", ids)
    assert not v0_names() & set(names) or V0 in ids, f"{composite}.components holds v0 under an id its `#guard` doesn't pin"


@pytest.mark.parametrize("product", [R.ONSITE, R.REMOTE], ids=lambda p: p.id)
def test_reference_components(product):
    ids = [C.id for C in product.components]
    assert not unlisted(f"the reference's {product.id}", ids), unlisted(f"the reference's {product.id}", ids)


def test_configuration_ids():
    ids = configuration_ids()
    assert ids, "parties.IDS gives no id as a literal"
    assert not unlisted("a configuration", ids), unlisted("a configuration", ids)


def test_an_unlisted_protocol_in_the_clear_is_refused(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "KNOWN", {})
    for composite, path in COMPOSITES.items():
        assert unlisted(composite, lean_components(path)[1]), f"{composite} composes v0 and nothing refuses it"
    assert unlisted("the reference", [C.id for C in R.REMOTE.components])


def test_known_only_shrinks():
    assert set(KNOWN) <= set(IN_THE_CLEAR), f"KNOWN lists {sorted(set(KNOWN) - set(IN_THE_CLEAR))}, which IN_THE_CLEAR doesn't"
    gone = sorted(set(KNOWN) - composed())
    assert not gone, f"no composite composes {gone} any more: delete it from KNOWN"
    assert all(KNOWN.values()), "every KNOWN entry gives its reason"
