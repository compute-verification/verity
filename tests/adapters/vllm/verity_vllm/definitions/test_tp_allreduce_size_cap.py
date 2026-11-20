"""GLM-5.3's TP8 MNNVL all-reduce at three rows against SPOT's widest size class, through the spot difftest's Lean
pieces (a file of another suite, loaded by its path, with its directory on the path for the siblings it imports)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import verity_difftest as D

from verity.circuits import population as PN
from verity.commitments.periods import interface as IF
from verity.protocols.proofs.spot.verifier.one_stage import partition as P
from verity_vllm.definitions.registry.catalog import load

SPOT_DIFFTEST = Path(__file__).resolve().parents[5] / "tests/verity/protocols/proofs/spot/verifier/lean/test_difftest.py"
if "one_stage_difftest" not in sys.modules:
    sys.path.insert(0, str(SPOT_DIFFTEST.parent))
    _spec = importlib.util.spec_from_file_location("one_stage_difftest", SPOT_DIFFTEST)
    sys.modules["one_stage_difftest"] = _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
SD = sys.modules["one_stage_difftest"]

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

pieces = D.server_fixture(SD.DECODE)


def test_glm53s_all_reduce_at_three_rows_is_refused_past_the_widest_class(pieces):
    """GLM-5.3's TP8 MNNVL all-reduce at three rows, as `tp_allreduce.circuit-check.py`'s `_glm53_served` binds it
    (`AllReduceMnnvl_v1{WORLD=8,N=18432,SHOT=1}`), returns 294,912 bits: Lean's slots and the Python's partition refuse
    its instance by the widest class."""
    load()
    unit = IF.resolve("AllReduceMnnvl_v1{WORLD=8,N=18432,SHOT=1}")
    query = P.template_instance_query(unit.id)
    why = "commits 294912 bits, more than the widest class, 262144"
    with pytest.raises(P.QueryRefused, match=why):
        P.units(PN.population_program(unit, 1), query)
    refused = {"closure": None, "rule": {"law": "rate:1/1000", "rule": "law"},
               "source": hashlib.shake_256(b"refused").hexdigest(400), "work_table": None,
               "program": SD._template_population(unit, 1), "query": SD._json(query)}
    kind, body = D.replay(SD.SLOTS, refused, pieces)
    assert kind == "reject" and why in str(body), (kind, body)
    with pytest.raises(D.Reject, match=why):
        SD.slots(copy.deepcopy(refused))
