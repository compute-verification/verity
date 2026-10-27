import numbers
from dataclasses import dataclass, replace

import pytest
from rho.params import Param, Registry

REG = Registry.load()


def test_every_number_says_where_it_comes_from():
    unsourced = [p.key for p in REG.params() if isinstance(p.value, numbers.Number) and not isinstance(p.value, bool)
                 and not p.source and not p.key.startswith("horizon.")]
    assert not unsourced, f"give these a source in data/*.toml: {unsourced}"


def test_every_range_holds_its_value():
    bad = [p.key for p in REG.params() if p.ranged and not min(p.low, p.high) <= p.value <= max(p.low, p.high)]
    assert not bad


def test_an_override_of_a_key_that_does_not_exist_is_an_error():
    with pytest.raises(KeyError, match="no parameter"):
        REG.override({"fleet.reference-h200.drak": 0.1})
    assert REG.override({"fleet.reference-h200.eps_dark": 0.1})["fleet.reference-h200.eps_dark"] == 0.1
    assert REG["fleet.reference-h200.eps_dark"] != 0.1


def test_an_override_by_a_param_replaces_its_range_source_and_status():
    p = Param("any", 300.0, "gate/FLOP", source="a count on one design", status="derived")
    q = REG.override({"units.gates_per_flop": p}).param("units.gates_per_flop")
    assert q == replace(p, key="units.gates_per_flop") and not q.ranged and REG.param("units.gates_per_flop").ranged
    with pytest.raises(KeyError, match="no parameter"):
        REG.override({"units.gates_per_flp": p})


def test_a_table_with_a_key_its_class_lacks_is_refused():
    @dataclass(frozen=True)
    class Narrow:
        name: str
        count: float

    with pytest.raises(ValueError, match="not fields of Narrow"):
        REG.fields(Narrow, "fleet.reference-h200")
