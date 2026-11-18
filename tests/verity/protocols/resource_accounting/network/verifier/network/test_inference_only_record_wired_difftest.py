"""The record form's wired reading (`InferenceOnlyVerifier.RecordWired`) against the Python reference: the Python half.

`verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts/record_wired_difftest_vectors.py`
writes the toys (`TOYS`), the record difftest's shaped cases and its first random ones, and on each
`scripts/inference_only_record_reference.py`'s wired reading. `InferenceOnlyVerifier/RecordWiredDifftestMain.lean`
holds `RecordWired` to it, and to `Record` where the root has no edge.

The first test is each toy's verdict: an edge the wired reading takes is refused by `read` (`own-rows`), and each
refusal is the stage the toy is made for. The second is that the wired reading is `read`'s where the root has no
edge, on every shaped case and the random ones, the whole verdict and every row's class. The third is what an edge's row counts as: inside its block, which leaves it
out of the block's fresh inputs, or a cross wire where the edge crosses slots.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LEAN = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "workload_classification" / "training_detection"
        / "inference_only" / "verifier" / "lean")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wired = _load("record_wired_difftest_vectors")
ref, V = wired.ref, wired.V
#: the refusals `read` gives before it reads the root's rows, which the wired reading gives as well
EARLIER = ("program", "not-topological", "templates")


def _detail(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "message"}


@pytest.fixture(scope="module")
def toys() -> dict[str, dict]:
    return {what: wired.toy(what) for what in wired.TOYS}


def test_each_toy_gives_its_verdict(toys):
    for what, c in toys.items():
        want = wired.TOYS[what][3]
        got = wired.results(c)["detail"]
        assert got.get("error") == want, what
        unwired = ref.check_detail(c["instance"], c["trusted"]).get("error")
        assert unwired == (want if want in EARLIER else "own-rows"), what
        assert wired._edge(c) == (want not in ("program", "not-topological")), what
    refused = {wired.TOYS[w][3] for w in toys}
    assert {"edge-rows", "inputs", "not-topological", "templates"} <= refused


def test_with_no_edge_it_is_read():
    made = [m["case"] for m in V.shaped()] + [V.case(seed)["case"] for seed in range(wired.RANDOM)]
    edgeless = [m for m in made if not wired._edge(m)]
    assert len(edgeless) > len(made) - 3
    for m in edgeless:
        inst, tr = m["instance"], m["trusted"]
        assert _detail(ref.check_detail(inst, tr, wired=True)) == _detail(ref.check_detail(inst, tr))
        assert ref.kinds(inst, tr, m["rows"], wired=True) == ref.kinds(inst, tr, m["rows"])


def test_an_edge_row_is_inside_or_a_cross_wire(toys):
    for what, c in toys.items():
        if wired.TOYS[what][3] is not None:
            continue
        r = wired.results(c)
        crossing = r["detail"]["cross_wires"] > 0
        assert crossing == (what == "an edge across slots"), what
        assert ("cross_wires" in r["kinds"]) == crossing, what
        assert (r["detail"]["fresh"] < r["detail"]["inputs"]) != crossing, what
