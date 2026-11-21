"""`machine-model`'s dispatch: which device a command reads, and the refusals when it is given none or a card no device
models."""
from __future__ import annotations

import pytest

from verity_machine_model.cli import main
from verity_machine_model.model import DATA, device, device_of


def test_a_card_reads_the_device_whose_model_is_that_cards():
    assert device_of("rtx-pro-6000-server") == DATA == device("sm_120")
    assert device_of("h200-sxm-141gb") == device("sm_90")
    assert device_of("b200-sxm-180gb") == device("sm_100")
    assert main(["--card", "h200-sxm-141gb", "sheet"]) == main(["--device", "sm_90", "sheet"]) == 0


def test_a_card_no_device_models_is_refused_by_name_not_read_as_another_of_its_target():
    with pytest.raises(SystemExit, match=r"no measured model for h100-sxm-80gb \(sm_90a\): the machine model's devices "
                                         r"are sm_120, the rtx-pro-6000-server, sm_90, the h200-sxm-141gb, sm_100, the "
                                         r"b200-sxm-180gb"):
        main(["--card", "h100-sxm-80gb", "sheet"])
    with pytest.raises(SystemExit, match="no card 'h100'"):
        device_of("h100")
    with pytest.raises(SystemExit, match="one of --device, --card or --data"):
        main(["--device", "sm_90", "--card", "h200-sxm-141gb", "sheet"])


def test_a_command_given_no_device_is_refused_by_name():
    """DESIGN.md "Devices": no device is the default; a command that reads one device's model names it, and the ones
    that read every device they cover take none."""
    for cmd in ("sheet", "eps"):
        with pytest.raises(SystemExit, match=rf"^machine-model {cmd} reads one device's model and was given none: name "
                                             r"it with --device sm_NN or --card SKU \(sm_120, the rtx-pro-6000-server; "
                                             r"sm_90, the h200-sxm-141gb; sm_100, the b200-sxm-180gb\)$"):
            main([cmd])
    assert main([]) == 0
    with pytest.raises(SystemExit, match="lean-headline writes every family's instances from this package's own data"):
        main(["--device", "sm_120", "lean-headline", "--check"])


def test_each_data_centre_reading_names_the_card_its_device_models():
    from verity_machine_model.dc_reading import FAMILIES
    from verity_machine_model.model import card

    assert {f.key: card(f.data)["short"] for f in FAMILIES} == {f.key: f.gpu.lower() for f in FAMILIES}
