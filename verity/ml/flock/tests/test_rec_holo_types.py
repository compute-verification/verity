"""rec-holo's unit types (``pod/rec_holo_types.py``, which the pod's route P and C-Flock jobs read): a type's gate list and its
C-Flock rows compute the same outputs from main's builders, a changed row is caught, and the file holds the header it says."""
from __future__ import annotations

import importlib.util
import random
from array import array
from pathlib import Path

import pytest


def _module():
    spec = importlib.util.spec_from_file_location("rec_holo_types", Path(__file__).parents[1] / "pod" / "rec_holo_types.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RH = _module()


def _inputs(name, net):
    rng = random.Random(name)
    return [rng.getrandbits(64) for _ in range(net.n_inputs)]


@pytest.mark.parametrize("name", ["f32_add", "f32_to_e4m3", "bf16_mul_f32"])
def test_a_types_gate_list_and_rows_agree_and_its_file_holds_them(name, tmp_path: Path):
    net, rows, fouts = RH.both(*RH.TYPES[name]())
    x = _inputs(name, net)
    assert RH.eval_net(net, x, 64) == RH.eval_rows(net.n_inputs, rows, fouts, x, 64)
    path = tmp_path / f"{name}.rhn"
    RH.write(path, net, rows, fouts)
    u = array("I", path.read_bytes())
    assert list(u[:5]) == [RH.MAGIC, net.n_inputs, len(net.gates), len(net.outputs), len(rows)]
    assert list(u[5 + 3 * len(net.gates):5 + 3 * len(net.gates) + len(net.outputs)]) == list(net.outputs)


def test_a_row_with_its_constant_flipped_disagrees_with_the_gate_list():
    net, rows, fouts = RH.both(*RH.TYPES["f32_add"]())
    (c, ids), b = rows[-1]
    bad = rows[:-1] + [((1 - c, ids), b)]
    x = _inputs("f32_add", net)
    assert RH.eval_net(net, x, 64) != RH.eval_rows(net.n_inputs, bad, fouts, x, 64)
