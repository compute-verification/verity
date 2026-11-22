"""`machine-model attacks`: the cheapest attacks of a design at a vector's prices."""
from __future__ import annotations

import pytest

from verity_machine_model.cli import main
from verity_machine_model.model import device, devices, model


def test_attacks(capsys):
    assert main(["--device", "sm_120", "attacks", "VF3j-16", "--design", "U±64, F±1", "-n", "1"]) == 0
    assert "X 2.000" in capsys.readouterr().out


def test_attacks_says_a_device_has_no_price_vector():
    for d in [d for d in devices() if not model(device(d)).adopted]:
        with pytest.raises(SystemExit, match=f"^{d}: no price vector is solved on this device"):
            main(["--device", d, "attacks"])
