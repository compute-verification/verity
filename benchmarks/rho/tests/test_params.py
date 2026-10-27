import numbers
from dataclasses import dataclass

import pytest
from rho.params import Registry

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


def test_a_table_with_a_key_its_class_lacks_is_refused():
    @dataclass(frozen=True)
    class Narrow:
        name: str
        count: float

    with pytest.raises(ValueError, match="not fields of Narrow"):
        REG.fields(Narrow, "fleet.reference-h200")
