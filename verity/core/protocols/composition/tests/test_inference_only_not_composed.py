"""Neither composite composes the inference-only policy's public form, `verity/inference-only/v0`
(`NciCheck.Protocol.protocol`): it publishes the declared division, and each inference block's gates and fresh-input
bits, in the clear, so it isn't zero-knowledge. The composites' components are read wherever they are listed: their
Lean (`Core.Onsite.components`, which is also `Core.Product.registry`, and `Core.Remote.components`, with the ids each
one's `#guard` pins), the reference's mirror of them (`reference.ONSITE`, `REMOTE`), and the ids a configuration may
name as components (`parties.IDS`, which step0's `MANIFESTS` lists), read from its source: importing `parties` would
read the deploy runtime, which this suite doesn't declare."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import reference as R

PROTOCOLS = Path(__file__).resolve().parents[2]
V0 = "verity/inference-only/v0"
PRIVATE_ONLY = ("only the private form (v1: division committed under hiding leaves, compliance proved, padded profile) "
                "may be composed")
COMPOSITES = {"Core.Onsite": PROTOCOLS / "onsite/CoreOnsite.lean", "Core.Remote": PROTOCOLS / "remote/CoreRemote.lean"}


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


@pytest.mark.parametrize("composite", sorted(COMPOSITES))
def test_lean_components(composite):
    names, ids = lean_components(COMPOSITES[composite])
    assert len(names) == len(ids) > 0, f"{composite}: {names} and the ids {ids} its `#guard` pins don't pair up"
    assert not v0_names() & set(names) and V0 not in ids, f"{composite}.components holds {V0}: {PRIVATE_ONLY}"


@pytest.mark.parametrize("product", [R.ONSITE, R.REMOTE], ids=lambda p: p.id)
def test_reference_components(product):
    assert V0 not in [C.id for C in product.components], f"the reference's {product.id} composes {V0}: {PRIVATE_ONLY}"


def test_configuration_ids():
    ids = configuration_ids()
    assert ids, "parties.IDS gives no id as a literal"
    assert V0 not in ids, f"a configuration may list {V0} as a component: {PRIVATE_ONLY}"
