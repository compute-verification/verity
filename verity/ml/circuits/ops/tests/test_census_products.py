"""census/v1's `PRODUCTS` names every tensor-core step these Definitions register, word primitive or Boolean Definition, with its k,
and `PRECISION` each one's operand format; `RULED` names every family with a census rule, which a Program decoded in a process that
never imported the rule's module refuses rather than count by `PRODUCTS`, and an s8 accumulator serves none of the products it
performs.  Every step is bound to its silicon model (`SILICON`), a Boolean step to its word step's, and a word GEMM counts its
Boolean version's work: K/16 steps' gates as `tc:bf16`, the cast's `cuda:gate`."""
from __future__ import annotations

import importlib
import json
import pkgutil
import re
import subprocess
import sys

import verity.ml.circuits.ops as OPS
import verity.ml.circuits.ops.gemm  # registers the steps
import verity.ml.circuits.ops.boolean.s8  # noqa: F401  (binds the u8 x s8 step)
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.census import COUNTERS, FAMILY_COUNTS, PRECISION, PRODUCTS, RULED, SILICON, census
from verity.ml.circuits.codec import descriptor_id, encode_program
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as P
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
    assert {f: d.body.__module__ for f, d in ruled.items()} == {f: m for f, m in RULED.items() if m.startswith(OPS.__name__ + ".")}
    assert all(FAMILY_COUNTS[f] is d.counts for f, d in ruled.items())


def test_an_s8_accumulator_performs_its_products_and_serves_none():
    """`GemmAccumulatorS8` (word and Boolean) has no committed words of its own, so Lean's `Accounting.Gates.roleOf` puts its
    steps at `Site.other`: under a linear's coordinate its K/32 IMMA steps keep their k, and it serves none of their products."""
    K = 64
    for coord, steps in ((bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM120_S8), "BlackwellS8ImmaDot32_v1"),
                         (bind(BG.ScaledMmS8Coordinate, K=K, **BG.GEMM_SM120_S8), "BlackwellS8ImmaDot32_v2")):
        c = census(Program(coord))
        (acc,) = [e for e in c["classes"].values() if e["family"].startswith("GemmAccumulatorS8_")]
        (step,) = [e for e in c["classes"].values() if e["family"] == steps]
        assert step["instances"] * PRODUCTS[steps] == K and step["served_macs"] == PRODUCTS[steps]
        assert {k: acc[k] for k in COUNTERS} == {k: c["totals"][k] for k in COUNTERS} == dict.fromkeys(COUNTERS, 0)
        assert acc["served_macs_by_precision"] == c["totals"]["served_macs_by_precision"] == {}


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


def test_every_tensor_core_step_is_bound_to_its_silicon_model():
    """Each registered step's family has its silicon model's precision, `PRECISION`'s, but Triton's MMAv2 e4m3 dot on sm_90,
    whose f16 HMMA multiplies in f16; a Boolean step is bound to the model of the word step it computes."""
    steps = {fid: d for fid, d in REGISTRY.defs.items() if DOT.search(fid)}
    f16 = {"HopperE4m3MmaV2Dot32_v1", "HopperE4m3MmaV2Dot32_v2"}
    assert {fid: SILICON.get(fid) for fid in steps} == {fid: "f16" if fid in f16 else PRECISION[fid] for fid in steps}
    assert f16 <= set(steps) and {PRECISION[fid] for fid in f16} == {"fp8"}
    boolean = {fid: d for fid, d in steps.items() if getattr(d, "word", None) is not None}
    assert len(boolean) == 10 and all(d.silicon is word_of(d).silicon for d in boolean.values()), sorted(boolean)
    assert {d.silicon.dtype for d in steps.values()} == {"bf16", "e4m3", "e5m2", "e2m1", "s8", "u8"}


def _gates(fn) -> tuple[int, int]:
    """`fn`'s two-input gates and ANDs, from its compiled circuit."""
    c = BS.compile(fn).counts
    return sum(c.values()), c["and"]


def test_a_word_gemm_counts_its_boolean_versions_work():
    """`Gemm_v2`'s coordinates through `GemmCoordinate_v3`: N x K/16 Hopper steps' gates are `tc:bf16`, and N casts (`F2fpBf16_v2`)
    the rest, `cuda:gate`; the Boolean GEMM's census counts the same."""
    K, N = 32, 3
    word = Program(bind(G.Gemm, K=K, N=N, DOT=P.HopperBF16WgmmaDot16))
    coord = bind(BG.GemmCoordinate, K=K, DOT=BG.HopperBF16WgmmaDot16)
    twins = {descriptor_id(word_of(coord)): coord}
    c = census(word, boolean=lambda fn: twins[descriptor_id(fn)])
    (g, a), (cg, ca) = _gates(BG.HopperBF16WgmmaDot16), _gates(BG.F2fpBf16)
    want = {"cuda:gate": {"gates": N * cg, "and": N * ca}, "tc:bf16": {"gates": N * K // 16 * g, "and": N * K // 16 * a}}
    assert c["totals"]["work"] == want == census(Program(bind(BG.Gemm, K=K, N=N, DOT=BG.HopperBF16WgmmaDot16)), boolean=True)["totals"]["work"]
    assert c["classes"][descriptor_id(word_of(coord))]["boolean_version"] == descriptor_id(coord)


#: a fresh process: the word primitives registered and bound, then `argv[2:]` imported, then the Boolean work of the decoded Program
WORK_DECODED = """
import json, sys
import verity.ml.circuits.ops.prims
for m in sys.argv[2:]:
    __import__(m)
from verity.ml.circuits.census import census
from verity.ml.circuits.codec import decode_program
from verity.ml.circuits.defs import REGISTRY
try:
    print(json.dumps(census(decode_program(json.load(open(sys.argv[1])), REGISTRY), boolean=True)["totals"]["work"]))
except ValueError as e:
    print(json.dumps({"refused": str(e)}))
"""


def test_a_boolean_program_decoded_without_its_steps_module_refuses_its_work(tmp_path):
    """Decoding keeps no binding: where `ops.boolean.gemm` was never imported the Boolean GEMM's work is refused, naming the step,
    instead of counting its gates as `cuda:gate`; with it imported, the decoded Program counts as the built one."""
    B = Program(bind(BG.Gemm, K=16, N=1, DOT=BG.HopperBF16WgmmaDot16))
    (tmp_path / "gemm.json").write_text(json.dumps(encode_program(B)))

    def decoded(*modules: str) -> dict:
        run = [sys.executable, "-c", WORK_DECODED, str(tmp_path / "gemm.json"), *modules]
        return json.loads(subprocess.run(run, capture_output=True, text=True, check=True).stdout)

    refused = decoded()["refused"]
    assert refused.startswith("HopperBF16WgmmaDot16_v2 is a tensor-core step (census.PRODUCTS) that census.SILICON binds to no "
                              "silicon model: import the module defining HopperBF16WgmmaDot16_v2"), refused
    assert decoded("verity.ml.circuits.ops.boolean.gemm") == census(B, boolean=True)["totals"]["work"]
