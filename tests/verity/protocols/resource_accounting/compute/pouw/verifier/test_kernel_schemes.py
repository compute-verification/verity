"""PoUW's one table of kernel schemes (`verifier.kernel_schemes`): every entry PoUW builds names itself as the table
does, a resource scheme on a card resolves to exactly one entry, and the readers of the table agree with it."""

from __future__ import annotations

from pathlib import Path

import pytest

from verity.device import gpu_card_called
from verity.protocols.compose import schemes as composed
from verity.protocols.resource_accounting.compute.pouw import scheme as pouw_scheme
from verity.protocols.resource_accounting.compute.pouw.prover import circuit, schemes
from verity.protocols.resource_accounting.compute.pouw.prover.serving import pearl_c as PPD
from verity.protocols.resource_accounting.compute.pouw.verifier import kernel_schemes as K
from verity.protocols.resource_accounting import scheme as S


@pytest.mark.parametrize("name", [k.name for k in K.KERNEL_SCHEMES if k.family in ("ncp", "pearl-kw", "pearl-c")])
def test_each_entry_pouw_builds_names_itself_as_the_table_does(name):
    assert schemes.get(name).name == name


KERNELS = S.declared(pouw_scheme.PEARL_PLUS, "kernels")
CARDS = Path(__file__).resolve().parents[7] / "calibration/hardware/cards"


def arch(short):
    return gpu_card_called(short, directory=CARDS)["arch"]


@pytest.mark.parametrize("card", ["h100", "h200"])
def test_pearl_plus_on_the_h100_or_h200_is_its_option(card):
    """Both cards run the one sm_90a build, under the one name its option, cubin and arm are found by."""
    name = KERNELS(card, pinned=True, arch=arch(card))
    assert name == KERNELS(card, pinned=False, arch=arch(card)) == pouw_scheme.KERNEL
    assert pouw_scheme.PEARL_PLUS.option == f"pouw:{name}"
    assert schemes.of(pouw_scheme.PEARL_PLUS, card, pinned=True, arch=arch(card)).name == name


@pytest.mark.parametrize("card", ["rtx-pro-6000", "b200"])
def test_pearl_plus_refuses_a_card_whose_architecture_it_has_no_build_for(card):
    """Pearl+ never had a kernel on the RTX PRO 6000 (its verifier checks pearl-c-h100-v1-h3 alone), nor on the B200."""
    with pytest.raises(LookupError, match=f"^{card}'s architecture sm_\\d+a has no build here: pearl-c-h100-v1-h3 is built "
                                          "for sm_90a$"):
        KERNELS(card, pinned=True, arch=arch(card))


def test_a_card_of_a_built_architecture_runs_it_only_if_its_row_lists_the_card():
    row = K.KernelScheme("probe", "pearl-c", arch="sm_90a", cards=("h200",), pinned=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(K, "BY_NAME", {"probe": row})
        kernels = schemes.runs_on("probe")
    with pytest.raises(LookupError, match=r"^0 builds for sm_90a run on h100 with B chosen: they run probe on h200 chosen$"):
        kernels("h100", pinned=False, arch="sm_90a")
    with pytest.raises(LookupError, match=r"^0 builds for sm_90a run on h200 with B pinned: they run probe on h200 chosen$"):
        kernels("h200", pinned=True, arch="sm_90a")
    assert kernels("h200", pinned=False, arch="sm_90a") == "probe"


def test_a_scheme_declaring_no_kernels_and_a_row_naming_cards_without_an_architecture_are_refused():
    with pytest.raises(LookupError, match="dither declares no kernels: experimental"):
        schemes.of(composed.get("dither"), "rtx-pro-6000", pinned=True, arch="sm_120a")
    with pytest.raises(ValueError, match="named together"):
        K.KernelScheme("probe", "pearl-c", cards=("h100",))


def test_the_readers_read_the_table():
    assert dict(circuit.SCHEMES) == {k.name: k.parameters["perm"] for k in K.of_family("ncp-v2")}
    with pytest.raises(KeyError, match="this device path runs"):
        PPD.refusals("ncp-v1", {})


def test_a_family_only_an_experimental_package_builds_is_registered_only_by_it():
    with pytest.raises(KeyError, match="not built by an experimental package"):
        schemes.register("pearl-c", schemes.BUILDERS["pearl-c"])
    with pytest.raises(KeyError, match="which no PoUWScheme builds"):
        schemes.get("ncp-v2")


def test_the_serving_facade_exports_the_tables_families_as_the_vllm_adapter_reads_them():
    from verity.protocols.resource_accounting.compute.pouw.prover import serving

    assert serving.NCP_CIRCUIT_SCHEMES == ("ncp-v2", "ncp-v2-shift24")
    assert serving.PEARL_C_SCHEMES == ("pearl-c-sm120-v1-h3", "pearl-c-sm120-unpromoted-v1-h3", "pearl-c-h100-v1-h3")
    assert serving.NCP_SCHEMES == ("ncp-v1", "ncp-v1-shift24", "pearl-fp8-v4", "pearl-c-sm120-v1-h3", "pearl-c-h100-v1-h3",
                                   "pearl-c-nvfp4-v0")
    assert serving.pearl_c_scheme is schemes.get and serving.pearl_c_scheme("pearl-c-h100-v1-h3").name == "pearl-c-h100-v1-h3"
