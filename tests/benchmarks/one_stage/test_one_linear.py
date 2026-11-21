"""`one_linear.py`'s circuit: one template's units, each proof block over the zero-knowledge floor and under 2^27 bits,
and its partition at two levels refused by name at the door, before any coin; and its window under ``--weights
committed`` read at its run input nodes' trees by the Lean verifier, each member at the record's one tree of each."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "one_stage"))

import one_linear as OL  # noqa: E402
from verity.circuits import laying  # noqa: E402
from verity.circuits.codec import encode_program  # noqa: E402
from verity.commitments.periods import interface as I  # noqa: E402
from verity.protocols.proof.spot.verifier.one_stage import members as M, one_circuit as OC, verifier as V  # noqa: E402


def test_one_template_its_blocks_between_the_floor_and_the_cap():
    c = OL.circuit()
    assert [n for _t, n in c.templates] == [OL.UNITS] and not c.wires
    k_log = I.check_proof_block(OL.gemm())
    assert laying.zk_min_units(k_log, 1) <= OL.CAP and k_log + (OL.CAP - 1).bit_length() <= I.K_MAX


def test_two_levels_are_refused_at_the_door_by_name():
    c, part = OL.two_levels()
    with pytest.raises(OC.Refused, match="partition: blocks-levels: the object has 2 levels of blocks; this reader takes one"):
        OC.block(c, part, 0)


def _checkpoint(_cache: Path) -> dict[str, np.ndarray]:
    """The checkpoint's three tensors at their shapes, finite BF16 words of a fixed generator, never downloaded."""
    rng = np.random.default_rng(7)

    def finite(*shape: int) -> np.ndarray:
        return rng.integers(0x3C00, 0x3F80, shape, dtype=np.uint16)
    return {OL.WEIGHT: finite(OL.E, OL.K), OL.NORM: finite(OL.K), OL.EMBED: finite(len(OL.TOKENS), OL.K)}


@pytest.fixture(scope="module")
def committed(tmp_path_factory):
    """`one_linear.py window --weights committed`'s files, as the honest run writes them, on `_checkpoint`'s values."""
    out = tmp_path_factory.mktemp("one-linear") / "window"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(OL, "checkpoint", _checkpoint)
        assert OL.window(argparse.Namespace(out=str(out), rule="rate:1/72", weights="committed", cache="")) == 0
    record = json.loads((out / "registration.json").read_text())
    units = [json.loads(p.read_text()) for p in out.glob("m*-units.json")]
    members = {u["member"]: [[u["base"], u["base"] + u["n"]]] for u in sorted(units, key=lambda u: u["base"])}
    program, query = OL.circuit(registered=False).program()
    return record, members, program, query


def _run_input_roots(record: dict, members: dict, program, query) -> dict:
    """The Lean verifier's `OneStageZk.runInputRoots` (`run_input_roots`) on the window's members, no tied gates."""
    return V.lean("run_input_roots", {"program": encode_program(program), "query": query, "roots": record["roots"],
                                      "members": [[n, us] for n, us in members.items()], "window_root": OL.RUN,
                                      "tied": []})


@pytest.mark.slow
def test_every_member_reads_an_untied_run_input_at_the_records_one_tree_and_without_it_is_refused(committed):
    """What `Properties.Spot.RunInputsAtOneTree` concludes, on the honest run's files: for every member, the inputs Lean's
    `memberInputs` gives it hold an untied run input node, and for each the member's slice of the record holds a root at
    its port, its node's one tree ``input/n<w>``, equal to the record's root of that tree. With one tree's root taken out
    of the record, Lean refuses the first member by `RUN_TREE_MISSING_WHY`. Slow (the window's commitments, seconds): no
    smaller window has one-linear's run input nodes and members."""
    record, members, program, query = committed
    got = _run_input_roots(record, members, program, query)
    assert got.get("refused") is None and len(got["roots"]) == len(members) >= 2, got
    of_tree = {r["port"]: r for r in record["roots"]}
    slices = M.slices(members, record, program=program, query=query, window_root=OL.RUN)
    for name, pairs in got["roots"]:
        assert pairs and all(t.startswith("input/n") and t in of_tree for _p, t in pairs), (name, pairs)
        for port, tree in pairs:
            (r,) = [r for r in slices[name]["roots"] if r.get("port") == port]
            assert r["tree"] == tree and r["root"] == of_tree[tree]["root"], (name, port, r)
    first, pairs = got["roots"][0]
    port, tree = pairs[-1]
    cut = {**record, "roots": [r for r in record["roots"] if r["port"] != tree]}
    refused = _run_input_roots(cut, members, program, query)
    assert refused == {"roots": None, "refused": M.run_tree_missing_why(first, port, int(tree.removeprefix("input/n")))}
