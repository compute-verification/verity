import json
import numbers
import re
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from rho import params
from rho.params import Param, Registry

from verity.core.protocols import profile, reporting
from verity.ml.circuits import partition_v2

REG = Registry.load()
CENSUS = Path(__file__).resolve().parents[3] / "benchmarks" / "census"


def entries(path: Path) -> dict[str, dict]:
    return {e["id"]: e for e in json.loads(path.read_text())}


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


def test_what_another_file_states_is_read_from_it_and_no_file_restates_it(tmp_path):
    """Each parameter of `params.sourced` is the value its source states: core's monitoring periods, per-window δ (once
    for a window's coins and for SPOT's sample, twice for the window's cryptography) and reporting δs; the
    calibrations' `K`, `Γ`, and the least and greatest `K` as the range. A file under `data/` that sets one of their
    keys is refused. The session policy's `K` is its layer's, held to the calibration's, with the same range read: a
    file that gives it a range is refused.

    The census's numbers are the one exception, and it is a rule: `verity` takes the census's numbers as parameters and
    never reads it. So the registry reads no census. The layers state the reference datacenter's machines, the world's
    active parameters and the session policy's `Γ` at the world's model, each naming the census as its source, and this
    test holds each equal to the census, which it reads from here."""
    window, deltas = float(profile.WINDOW_DELTA), [float(d) for d in reporting.DELTAS]
    calibrations = entries(params.CALIBRATIONS)
    raw, ks = calibrations["kimi-k2-1m-raw"], [c["K"] for c in calibrations.values()]
    k = (raw["K"], min(ks), max(ks))
    widest, smallest = partition_v2.UNIT_CLASS_MAX, partition_v2.UNIT_CLASSES[0]
    want = {f"horizon.{name}.seconds": (seconds, None, None) for name, seconds in reporting.PERIODS}
    want |= {key: (widest, smallest, widest) for key in params.BLOCK_OUTPUT_BITS}
    want |= {
        "regime.remote.window_delta_coin": (window, None, None),
        "regime.remote.window_delta_crypto": (2 * window, None, None),
        "regime.remote-sampled.p_evade": (window, 1e-15, 2.0 ** -20),
        "incorrect.delta": (min(deltas), min(deltas), max(deltas)),
        "policy.kimi-k2-1m-raw.gamma": (raw["Gamma"], None, None),
        "policy.kimi-k2-1m-raw.k": k,
    }
    sourced = params.sourced()
    assert {p.key: (p.value, p.low, p.high) for p in sourced} == want
    for p in sourced:
        assert REG.param(p.key) == p and p.source and p.status in params.STATUSES, p.key
        table, _, leaf = p.key.rpartition(".")
        restated = tmp_path / f"{table}.toml"
        restated.write_text(f"[{table}]\n{leaf} = {{ value = {p.value!r} }}\n")
        with pytest.raises(ValueError, match=f"{re.escape(p.key)} is set twice"):
            Registry.load([restated])
    assert params.ranges() == {"policy.v4-pro-session.k": k[1:]}
    ruled = REG.param("policy.v4-pro-session.k")
    assert (ruled.value, ruled.low, ruled.high) == k and ruled.source and ruled.status == "ruled"
    ranged = tmp_path / "ranged.toml"
    ranged.write_text(f"[policy.v4-pro-session]\nk = {{ value = {ruled.value!r}, high = {ruled.high!r} }}\n")
    with pytest.raises(ValueError, match="policy.v4-pro-session.k's range is read from its source"):
        Registry.load([ranged])
    dc = entries(CENSUS / "deployments.json")[params.DATACENTER]
    world = entries(CENSUS / "adversary.json")[params.WORLD]
    assert world["model"].lower() == params.WORLD_MODEL
    stated = {f"fleet.{fleet}.count": dc["machines"] for fleet in params.DATACENTER_FLEETS}
    stated |= {
        f"model.{params.WORLD_MODEL}.n_active": world["active_params"],
        "policy.v4-pro-session.gamma": raw["K"] / raw["b_tok"] * world["flop_per_active_param_per_token"]
        * world["active_params"],
    }
    assert not set(stated) & {p.key for p in sourced}
    for key, value in stated.items():
        p = REG.param(key)
        assert (p.value, type(p.value), p.low, p.high) == (value, type(value), None, None), key
        assert "benchmarks/census/" in p.source and p.status in params.STATUSES, key


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
