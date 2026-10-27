import json
import math
from pathlib import Path

import pytest
from rho import hardware
from rho.hardware import Fleet, Location
from rho.params import Registry

REG = Registry.load()
CENSUS_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "census"
CENSUS = {d["id"]: d for d in json.loads((CENSUS_DIR / "hardware.json").read_text())}


@pytest.mark.parametrize(("name", "device"), [("h100-hgx", "h100-sxm5-80gb"), ("h200-hgx", "h100-sxm5-80gb")])
def test_a_location_is_its_census_devices_where_the_census_has_them(name, device):
    loc, u = hardware.location(REG, name), hardware.units(REG)
    assert math.isclose(loc.phi, u.gates(loc.devices * CENSUS[f"{device}/e4m3"]["flop_per_second"]))
    assert math.isclose(REG.param(f"location.{name}.flops").low,
                        loc.devices * CENSUS[f"{device}/bf16"]["flop_per_second"])
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


@pytest.mark.parametrize("name", ["world-h200", "world-nvl72"])
def test_the_world_fleets_serve_the_census_adversary_in_its_monitoring_period(name):
    """`world-inference-2027` (`benchmarks/census/adversary.json`): its `total_flop` at the fleet's serving MFU."""
    world = next(a for a in json.loads((CENSUS_DIR / "adversary.json").read_text()) if a["id"] == "world-inference-2027")
    f = REG.table(f"fleet.{name}")
    flops = REG[f"location.{f['location']}.flops"]
    assert math.isclose(f["count"], world["total_flop"] / (flops * world["monitoring_period_s"] * f["mfu_serve"]),
                        rel_tol=1e-3)


def test_a_rack_of_four_servers_is_four_servers():
    one, four = hardware.location(REG, "h200-hgx"), hardware.location(REG, "h200-hgx-rack4")
    for field in ("devices", "phi", "memory", "hosts", "host_memory", "host_phi", "links"):
        assert math.isclose(getattr(four, field), 4 * getattr(one, field)), field
    assert math.isclose(four.server_memory, one.server_memory)


def test_a_location_without_hosts_is_one_host_with_nothing_counted():
    loc = Location("bare", 8, 1e16, 1e12)
    assert (loc.hosts, loc.host_memory, loc.host_phi, loc.links) == (1, 0.0, 0.0, math.inf)
    assert loc.server_memory == 1e12


def test_a_construction_gets_the_dark_leaves_at_their_own_mfu():
    assert Fleet("f", "h200-hgx", 1, 0.01, 0.4, 0.4).dark_mfu == 0.4
    assert Fleet("f", "h200-hgx", 1, 0.01, 0.4, 0.4, 0.55).dark_mfu == 0.55


def test_units_is_the_one_conversion_from_flops_to_gates():
    u = hardware.units(REG)
    p = REG.param("units.gates_per_flop")
    assert u.gates_per_flop == p.value and p.ranged and p.status
    assert u.gates(2.0) == 2 * p.value and u.flops(u.gates(3.0)) == pytest.approx(3.0)


def test_a_servers_memory_is_every_tier_over_its_hosts():
    loc = hardware.location(REG, "gb200-nvl72")
    assert loc.server_memory == pytest.approx((loc.memory + loc.host_memory + loc.disk) / loc.hosts)
