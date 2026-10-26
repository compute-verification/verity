import json
import math
from pathlib import Path

import pytest
from rho import hardware
from rho.hardware import Fleet, Location
from rho.params import Registry
from rho.theorem import rho

REG = Registry.load()
CENSUS = {d["id"]: d for d in json.loads(
    (Path(__file__).resolve().parents[3] / "benchmarks" / "census" / "hardware.json").read_text())}


@pytest.mark.parametrize(("name", "device"), [("h100-hgx", "h100-sxm5-80gb"), ("h200-hgx", "h100-sxm5-80gb"),
                                              ("rtx-pro-6000-server", "rtx-pro-6000-server")])
def test_a_location_is_its_census_devices_where_the_census_has_them(name, device):
    loc = hardware.location(REG, name)
    assert math.isclose(loc.phi, loc.devices * CENSUS[f"{device}/e4m3"]["flop_per_second"])
    assert math.isclose(REG.param(f"location.{name}.phi").low, loc.devices * CENSUS[f"{device}/bf16"]["flop_per_second"])
    if name != "h200-hgx":
        assert math.isclose(loc.memory, loc.devices * CENSUS[f"{device}/e4m3"]["memory_bytes"] * 8)


@pytest.mark.parametrize("name", REG.names("fleet"))
def test_every_fleet_is_on_a_location(name):
    f = hardware.fleet(REG, name)
    assert hardware.location(REG, f.location).phi > 0 and f.count > 0 and 0 <= f.eps_dark < 1


@pytest.mark.parametrize(("name", "gpus"), [("reference-h100", 1e5), ("reference-h200", 1e5), ("reference-b200", 1e5),
                                            ("reference-gb200", 1e5), ("reference-h200-rack4", 1e5),
                                            ("reference-gb200-400k", 4e5)])
def test_the_reference_fleets_are_100k_or_400k_gpus(name, gpus):
    f = hardware.fleet(REG, name)
    assert math.isclose(f.count * hardware.location(REG, f.location).devices, gpus, rel_tol=1e-3)


def test_a_rack_of_four_servers_is_four_servers():
    one, four = hardware.location(REG, "h200-hgx"), hardware.location(REG, "h200-hgx-rack4")
    for field in ("devices", "phi", "memory", "hosts", "host_memory", "host_ops", "links"):
        assert math.isclose(getattr(four, field), 4 * getattr(one, field)), field


def test_a_location_without_hosts_is_one_host_with_nothing_counted():
    loc = Location("bare", 8, 1e16, 1e12)
    assert (loc.hosts, loc.host_memory, loc.host_ops, loc.links) == (1, 0.0, 0.0, math.inf)


def test_the_dark_pool_counts_at_its_own_mfu():
    assert Fleet("f", "h200-hgx", 1, 0.01, 0.4, 0.4).eps_dark_rate == 0.01
    assert Fleet("f", "h200-hgx", 1, 0.01, 0.4, 0.4, 0.4).eps_dark_rate == pytest.approx(0.01)
    fast = Fleet("f", "h200-hgx", 1, 0.01, 0.4, 0.4, 0.55)
    assert fast.eps_dark_rate > 0.01
    # ρ at the effective share is the ratio of progress rates, each pool at its own MFU, the monitored one slowed to g
    d, mt, md, g = fast.eps_dark, fast.mfu_train, fast.mfu_dark, 0.3
    assert rho(fast.eps_dark_rate, g) == pytest.approx((d * md + (1 - d) * mt) / (d * md + (1 - d) * mt * g))
