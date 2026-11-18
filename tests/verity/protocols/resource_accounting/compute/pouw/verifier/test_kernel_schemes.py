"""PoUW's one table of kernel schemes (`verifier.kernel_schemes`): every entry PoUW builds names itself as the table
does, a resource scheme on a card resolves to exactly one entry, and the readers of the table agree with it."""

from __future__ import annotations

import pytest

from verity.protocols.resource_accounting.compute.pouw import scheme as pouw_scheme
from verity.protocols.resource_accounting.compute.pouw.prover import circuit, schemes
from verity.protocols.resource_accounting.compute.pouw.prover.serving import pearl_c as PPD
from verity.protocols.resource_accounting.compute.pouw.verifier import kernel_schemes as K


@pytest.mark.parametrize("name", [k.name for k in K.KERNEL_SCHEMES if k.family in ("ncp", "pearl-kw", "pearl-c")])
def test_each_entry_pouw_builds_names_itself_as_the_table_does(name):
    assert schemes.get(name).name == name


def test_pearl_plus_on_the_h100_is_its_option():
    entry = K.resolve("pearl-plus", "h100", True)
    assert entry is K.resolve("pearl-plus", "h100", False)
    assert pouw_scheme.PEARL_PLUS.option == f"pouw:{entry.name}"
    assert schemes.of("pearl-plus", "h100", True).name == entry.name


@pytest.mark.parametrize(("scheme", "card"), [("pearl-plus", "rtx-pro-6000"), ("pearl-plus", "b200"), ("nope", "h100")])
def test_a_resource_scheme_on_a_card_without_an_entry_is_refused(scheme, card):
    with pytest.raises(KeyError, match="0 kernel schemes run"):
        K.resolve(scheme, card, True)


def test_the_readers_read_the_table():
    assert PPD.SCHEMES == tuple(k.name for k in K.of_family("pearl-c"))
    assert dict(circuit.SCHEMES) == {k.name: k.parameters["perm"] for k in K.of_family("ncp-v2")}
    with pytest.raises(KeyError, match="this device path runs"):
        PPD.scheme_of("ncp-v1")


def test_a_family_only_an_experimental_package_builds_is_registered_only_by_it():
    with pytest.raises(KeyError, match="not built by an experimental package"):
        schemes.register("pearl-c", schemes.BUILDERS["pearl-c"])
    with pytest.raises(KeyError, match="which no PoUWScheme builds"):
        schemes.get("ncp-v2")
