"""The space, bandwidth and wipe legs are their period hooks' ``leg``, loaded from their files as the runner loads them."""

import importlib.util
from pathlib import Path

from verity.core.protocols.pous import period as pous
from verity.core.protocols.pous import wipe_period as wipe
from verity.core.protocols.remote_bandwidth import period as bandwidth

LEGS = Path(__file__).resolve().parents[1]


def _leg(name: str):
    spec = importlib.util.spec_from_file_location(f"all_protocols_leg_{name}", LEGS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.leg


def test_the_space_bandwidth_and_wipe_legs_are_their_hooks():
    assert _leg("pous") is pous.leg
    assert _leg("bandwidth") is bandwidth.leg
    assert _leg("wipe") is wipe.leg
