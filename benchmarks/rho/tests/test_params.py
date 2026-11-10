import numbers
from dataclasses import dataclass, replace

import pytest
from rho import params
from rho.params import Param, Registry

from verity.ml.circuits import partition_v2

REG = Registry.load()


def test_every_number_says_where_it_comes_from():
    unsourced = [p.key for p in REG.params() if isinstance(p.value, numbers.Number) and not isinstance(p.value, bool)
                 and not p.source]
    assert not unsourced, f"give these a source in data/*.toml: {unsourced}"


def test_every_range_holds_its_value():
    bad = [p.key for p in REG.params() if p.ranged and not min(p.low, p.high) <= p.value <= max(p.low, p.high)]
    assert not bad


def test_a_blocks_output_bits_are_veritys_and_no_file_restates_them(tmp_path):
    """A proof block's outputs fit its Definition's public size class, and the bits an incorrect block injects are the
    widest (`partition_v2.UNIT_CLASS_MAX`), read from `verity` (`params.sourced`): every `step_bits` of the registry is
    one of `BLOCK_OUTPUT_BITS`, none set under `data/`, and a file that sets one is refused."""
    keys = sorted(p.key for p in REG.params() if p.key.rpartition(".")[2] == "step_bits")
    assert keys == sorted(params.BLOCK_OUTPUT_BITS), "name the key in params.BLOCK_OUTPUT_BITS; don't set it in data/"
    widest, smallest = partition_v2.UNIT_CLASS_MAX, partition_v2.UNIT_CLASSES[0]
    for key in keys:
        p = REG.param(key)
        assert (p.value, p.low, p.high, p.unit, p.status) == (widest, smallest, widest, "bit", "derived"), key
        assert "partition_v2 UNIT_CLASS_MAX" in p.source and "ruled" not in p.source
        table, _, leaf = key.rpartition(".")
        restated = tmp_path / f"{table}.toml"
        restated.write_text(f'[{table}]\n{leaf} = {{ value = {widest}, unit = "bit" }}\n')
        with pytest.raises(ValueError, match=f"{key} is set twice"):
            Registry.load([restated])


def test_an_override_of_a_key_that_does_not_exist_is_an_error():
    with pytest.raises(KeyError, match="no parameter"):
        REG.override({"fleet.reference-h200.drak": 0.1})
    assert REG.override({"fleet.reference-h200.eps_unreg": 0.1})["fleet.reference-h200.eps_unreg"] == 0.1
    assert REG["fleet.reference-h200.eps_unreg"] != 0.1


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
