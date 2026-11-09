"""No composite composes a protocol that publishes private data in the clear (`IN_THE_CLEAR`) unless `KNOWN` lists it
with its reason. The inference-only policy's public form, `verity/inference-only/v0` (`NciCheck.Protocol.protocol`),
publishes the division, and each inference block's gates and fresh-input bits, so it isn't zero-knowledge; every
composite composes it as a stated exception until v1's private form replaces it. The
composites' components are read wherever they are listed: their Lean (`Core.Onsite.components`, `Core.Remote.components`
and `Core.Remote.V1.components`, `Core.RemoteNetwork.components`, which extends `Core.Remote.components` with `::`, and
`Core.RemoteNetworkNoPouw.components`, with the ids each one's `#guard` pins), the reference's mirror of them
(`reference.ONSITE`, `REMOTE_V1`, `REMOTE`, `REMOTE_NETWORK`, `REMOTE_NETWORK_NO_POUW`), and the ids a configuration may
name as components (`parties.IDS`, which step0's
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
COMPOSITES = {"Core.Onsite": PROTOCOLS / "onsite/CoreOnsite.lean", "Core.Remote": PROTOCOLS / "remote/CoreRemote.lean",
              "Core.RemoteNetwork": PROTOCOLS / "onsite/CoreRemoteNetwork.lean",
              "Core.RemoteNetworkNoPouw": PROTOCOLS / "onsite/CoreRemoteNetworkNoPouw.lean"}
#: The protocols that publish private data in the clear, each with what it publishes.
IN_THE_CLEAR = {V0: "the division, and each inference block's gates and fresh-input bits"}
#: The protocols of `IN_THE_CLEAR` a composite may compose, each with its reason. KNOWN only shrinks: an entry no
#: composite composes any more fails `test_known_only_shrinks`.
KNOWN = {"verity/inference-only/v0": "stated exception for the demos' runs until v1's private form replaces it"}


def v0_names() -> set[str]:
    """v0's Lean names: `NciCheck.Protocol.protocol` while its `protocolId` is v0."""
    text = (PROTOCOLS / "network/nci/lean/NciCheck/Protocol.lean").read_text()
    m = re.search(r'^def protocolId : String := "([^"]*)"', text, re.M)
    assert m, "NciCheck/Protocol.lean no longer defines `protocolId` as a string literal: read v0's id where it now is"
    return {"NciCheck.Protocol.protocol"} if m[1] == V0 else set()


def lean_components(path: Path) -> tuple[list[str], list[str]]:
    """A composite module's `components` (each composite's it defines), the Lean names they list (a `P ::
    Core.X.components` through the first `components` of the module of `COMPOSITES` it extends), and the ids their
    `#guard`s pin them to."""
    text = path.read_text()
    lists = _listed(text)
    cons = re.search(r"^def components\s*:[^\n]*?:=\s*([\w.]+) :: ([\w.]+)\.components\s*$", text, re.M)
    if cons:
        lists.append([cons[1], *_listed(COMPOSITES[cons[2]].read_text())[0]])
    ids = re.findall(r"^#guard components\.map \(·\.id\) ==\s*\[(.*?)\]", text, re.M | re.S)
    assert lists and len(lists) == len(ids), (f"{path.name}: a `def components : ... := [...]` or `P :: "
                                              "Core.X.components` without a `#guard` on its ids")
    return [n for ns in lists for n in ns], [i for xs in ids for i in re.findall(r'"([^"]*)"', xs)]


def _listed(text: str) -> list[list[str]]:
    """Each `def components : ... := [...]` of a module's text, as the Lean names it lists."""
    return [[n.strip() for n in ns.split(",")]
            for ns in re.findall(r"^def components\s*:[^\n]*?:=\s*\[(.*?)\]", text, re.M | re.S)]


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
        C.id for p in (R.ONSITE, R.REMOTE_V1, R.REMOTE, R.REMOTE_NETWORK, R.REMOTE_NETWORK_NO_POUW) for C in p.components}


@pytest.mark.parametrize("composite", sorted(COMPOSITES))
def test_lean_components(composite):
    names, ids = lean_components(COMPOSITES[composite])
    assert len(names) == len(ids) > 0, f"{composite}: {names} and the ids {ids} its `#guard` pins don't pair up"
    assert not unlisted(f"{composite}.components", ids), unlisted(f"{composite}.components", ids)
    assert not v0_names() & set(names) or V0 in ids, f"{composite}.components holds v0 under an id its `#guard` doesn't pin"


@pytest.mark.parametrize("product", [R.ONSITE, R.REMOTE_V1, R.REMOTE, R.REMOTE_NETWORK, R.REMOTE_NETWORK_NO_POUW],
                         ids=lambda p: p.id)
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
