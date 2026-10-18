"""census/v1's `PRODUCTS` names every tensor-core step these Definitions register, word primitive or Boolean Definition, with its k,
and `PRECISION` each one's operand format; `RULED` names every family with a census rule, which a Program decoded in a process that
never imported the rule's module refuses rather than count by `PRODUCTS`."""
from __future__ import annotations

import importlib
import json
import pkgutil
import re
import subprocess
import sys

import verity.ml.circuits.ops as OPS
import verity.ml.circuits.ops.gemm  # noqa: F401  (registers the steps)
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.census import COUNTERS, FAMILY_COUNTS, PRECISION, PRODUCTS, RULED, census
from verity.ml.circuits.codec import encode_program
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.ops import s8 as S8
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program

DOT = re.compile(r"Dot(\d+)_v\d+$")


def test_every_tensor_core_step_has_its_products():
    steps = {fid: int(m[1]) for fid in REGISTRY.defs if (m := DOT.search(fid))}
    assert len(steps) >= 16
    assert {fid: PRODUCTS.get(fid) for fid in steps} == steps


def test_a_boolean_step_performs_its_word_steps_products():
    pairs = {d.id: word_of(d).id for d in (getattr(BG, n) for n in BG.__all__) if DOT.search(getattr(d, "id", ""))}
    assert len(pairs) == 9
    assert all(PRODUCTS[b] == PRODUCTS[w] and PRECISION[b] == PRECISION[w] for b, w in pairs.items()), pairs


def test_every_tensor_core_step_has_its_precision():
    assert set(PRECISION) == set(PRODUCTS) and set(PRECISION.values()) == {"bf16", "tf32", "fp8", "fp4", "int8"}
    assert {f: PRECISION[f] for f in PRODUCTS if "E5m2" in f or "Mxf4" in f or "U8S8" in f} == {
        "BlackwellE5m2QmmaDot32_v1": "fp8", "BlackwellE5m2QmmaDot32_v2": "fp8", "BlackwellMxf4OmmaDot64_v1": "fp4",
        "BlackwellMxf4OmmaDot64_v2": "fp4", "PouwDot16U8S8_v1": "int8", "BlackwellU8S8ImmaDot32_v1": "int8",
        "BlackwellU8S8ImmaDot32_v2": "int8"}


def test_every_definition_with_a_rule_is_ruled_under_its_module():
    """Every ops Definition with a census rule (`.counts`) is in `RULED` under the module defining it, which registers that rule."""
    for m in pkgutil.walk_packages(OPS.__path__, OPS.__name__ + "."):
        if ".tests" not in m.name:
            importlib.import_module(m.name)
    ruled = {d.id: d for d in REGISTRY.defs.values() if hasattr(d, "counts")}
    assert {f: d.body.__module__ for f, d in ruled.items()} == RULED
    assert all(FAMILY_COUNTS[f] is d.counts for f, d in ruled.items())


#: a fresh process: the primitives registered (as a verifier decoding the Program has them), then `argv[2:]` imported, then the census
DECODED = """
import json, sys
import verity.ml.circuits.ops.prims
for m in sys.argv[2:]:
    __import__(m)
from verity.ml.circuits.census import COUNTERS, census
from verity.ml.circuits.codec import decode_program
from verity.ml.circuits.defs import REGISTRY
try:
    totals = census(decode_program(json.load(open(sys.argv[1])), REGISTRY))["totals"]
    print(json.dumps({k: totals[k] for k in COUNTERS}))
except ValueError as e:
    print(json.dumps({"refused": str(e)}))
"""


def test_a_program_decoded_without_its_rules_module_refuses_its_census(tmp_path):
    """An edited s8 matmul (two S8Dot_v1 coordinates) decodes from `ops.prims`' primitives alone. Where `ops.s8` was never
    imported its census refuses, naming the module, instead of counting the tail's IMMA steps as served MACs (320, no tail atoms);
    with it imported, the decoded Program counts as the built one does."""
    P = Program(bind(S8.S8Matmul, N=2, **S8.SMOKE_EDITED))
    (tmp_path / "s8.json").write_text(json.dumps(encode_program(P)))

    def decoded(*modules: str) -> dict:
        run = [sys.executable, "-c", DECODED, str(tmp_path / "s8.json"), *modules]
        return json.loads(subprocess.run(run, capture_output=True, text=True, check=True).stdout)

    refused = decoded()["refused"]
    assert refused.startswith("S8Dot_v1{"), refused
    assert refused.endswith("census.FAMILY_COUNTS lacks: import verity.ml.circuits.ops.s8, which registers it, before taking the census")
    want = {k: census(P)["totals"][k] for k in COUNTERS}
    assert decoded("verity.ml.circuits.ops.s8") == want == {"served_macs": 128, "tail_atoms": 192, "receipt_folds": 0}
