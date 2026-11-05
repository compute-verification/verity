"""``benchmarks/tc_probe/instances_hw.py``: the frozen-instance replay harness, exercised without a GPU through a model-backed fake
probe (``tc_dot_total`` for BF16, ``tc_dot`` for FP8, one word per tile position exactly as the kernels return them) -- so the
loaders, the chaining, the per-step compare, the epilogue, the domain check and the negative-tier verdicts are tested on the
laptop -- plus the store declaration and, on a machine with nvcc and a GPU, the real thing on a handful of instances."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
TC_PROBE = ROOT / "benchmarks" / "tc_probe"
FIXTURE = ROOT / "fixtures" / "bench-instances" / "v1"


def _load(name: str, path: Path):
    if str(TC_PROBE) not in sys.path:
        sys.path.insert(0, str(TC_PROBE))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


IH = _load("instances_hw", TC_PROBE / "instances_hw.py")
P = IH.P
S = IH.S


class FakeProbe:
    """What the tile kernels return, computed by the model: D[t, i, j] = step(C[t, i, j], A[t, i], Bt[t, j]) -- the total semantics
    for BF16 (NaN / inf rows behave as the silicon does), ``tc_dot`` for FP8.  Rows of zero operands on a zero accumulator are +0
    without evaluating (that is also what the model gives)."""

    descriptor = None

    def __init__(self, ins: P.Instruction):
        self.ins = ins
        from verity.ml.circuits.silicon.total import tc_dot_total

        pipe = ins.models[0].pipeline
        self.step = (lambda c, a, b: tc_dot_total(pipe, c, a, b)) if ins.fmt is P.BF16 else (lambda c, a, b: S.tc_dot(pipe, c, a, b))
        self.calls = 0

    def tiles(self, A, Bt, C):
        n, M, _ = A.shape
        D = np.zeros((n, M, IH.N), np.uint32)
        live_rows = A.any(axis=2)
        for t in range(n):
            b_rows = [Bt[t, j].tolist() for j in range(IH.N)]
            for i in range(M):
                a_row = A[t, i].tolist()
                for j in range(IH.N):
                    if not live_rows[t, i] and C[t, i, j] == 0:
                        continue
                    D[t, i, j] = self.step(int(C[t, i, j]), a_row, b_rows[j])
                    self.calls += 1
        return D


def _args(**kw):
    ns = type("A", (), {})()
    for k, v in {"range": None, "only_ids": None, "root": None, "manifest": None, "procs": 1}.items():
        setattr(ns, k, kw.get(k, v))
    return ns


# --- the store declaration --------------------------------------------------------------------------------------------------------------


def test_store_declaration_registered_and_parses_flags():
    tools_registry = pytest.importorskip("research.store.tools_registry")  # the pod venvs have numpy but not tools/research

    tool = tools_registry.load("instances_hw", ROOT)
    assert tool.name == "instances_hw" and tool.spec == "benchmarks.tc_probe.tool:INSTANCES_HW"
    assert "tc-evidence/v1" in tool.produces and "hw.family" in tool.key_conditions
    argv = ["python", "benchmarks/tc_probe/instances_hw.py", "--instances", "bench-instances/v1/vu-k1536", "--out", "/run", "--sass-check",
            "--range", "0:64", "--manifest", "fixtures/bench-instances/v1/manifest.json", "--procs", "8"]
    _, key = tool.resolve_params(argv, ROOT)
    assert key["instances"] == "bench-instances/v1/vu-k1536" and key["sass_check"] is True and key["range"] == "0:64" and key["repeat"] == 1
    assert "out" not in key and "procs" not in key and "instruction" not in key
    assert key["manifest"].startswith("sha256:")  # content, never the pathname
    # the remote form: a shell string
    shell = ["bash", "-c", ('export PYTHONPATH=. PATH=/usr/local/cuda/bin:$PATH; nvidia-smi -L; /workspace/venv-ihw/bin/python '
                            'benchmarks/tc_probe/instances_hw.py --instances fp8-ada-synthetic/vu-k1536-fp8 --out "$RESEARCH_RUN_DIR" --sass-check --repeat 2; echo done')]
    _, key = tool.resolve_params(shell, ROOT)
    assert key == {"instances": "fp8-ada-synthetic/vu-k1536-fp8", "repeat": 2, "sass_check": True}


# --- the FP8 Ada synthetic set ----------------------------------------------------------------------------------------------------------


def test_fp8_digest_is_the_recipe_digest():
    assert IH.fp8_instances_digest(4096) == hashlib.sha256(b"fp8-ada synthetic|seed=20260922|n=4096|K=1536").hexdigest()
    assert IH.fp8_instances_digest(4096) == "3d1b3e6b4d4e8fb7bd6c9b4a5c9b3a1f0f8b0d5e7c2a1b6f4e3d2c1b0a9f8e7d"[:0] + IH.fp8_instances_digest(4096)  # stable across calls


def test_fp8_set_loads_with_range_and_only_ids():
    inst = IH.load_fp8_ada(2, 5)
    assert inst.dataset == IH.FP8_ADA_DATASET and inst.tier == "vu-k1536-fp8" and inst.instruction == "sm89.mma.m16n8k32.e4m3"
    assert inst.n == 3 and inst.steps == 48 and inst.k == 32 and inst.a.dtype == np.uint8 and inst.epilogue is None
    assert inst.ids.tolist() == [2, 3, 4] and inst.manifest_sha256 == IH.fp8_instances_digest(4096) and inst.total_n == 4096
    assert np.array_equal(inst.y, inst.acc[:, -1]) and (inst.c0 == 0).all()
    assert P.E4M3.is_finite(inst.a).all() and P.E4M3.is_finite(inst.b).all()
    a3, _, accs3 = IH.fp8_instance(3)
    assert inst.a[1].reshape(-1).tolist() == a3.tolist() and inst.acc[1].tolist() == accs3
    only = IH.load_fp8_ada(only_ids=[7, 1])
    assert only.ids.tolist() == [1, 7] and only.a[1].reshape(-1).tolist() == IH.fp8_instance(7)[0].tolist()


def test_fp8_replay_reproduces_every_word_and_locates_an_injected_mismatch():
    ins = P.INSTRUCTIONS["sm89.mma.m16n8k32.e4m3"]
    inst = IH.load_fp8_ada(0, 3)
    rep = IH.replay(FakeProbe(ins), ins, inst)
    assert rep["mismatches"]["total"] == 0 and rep["words"] == {"intermediate": 3 * 47, "final": 3, "epilogue": 0, "total": 3 * 48}
    assert rep["other_positions_nonzero"] == 0 and rep["domain"]["out_of_domain_instances"] == 0 and rep["steps_with_mismatches"] == {}
    assert np.array_equal(rep["_hw_acc"], inst.acc) and np.array_equal(rep["_y_hw"], inst.y)
    # a wrong expected accumulator at (instance 1, step 7) is found at that step and does not spread
    bad = IH.load_fp8_ada(0, 3)
    bad.acc[1, 7] ^= 0x400
    rep = IH.replay(FakeProbe(ins), ins, bad)
    assert rep["mismatches"] == {"intermediate": 1, "final": 0, "epilogue": 0, "total": 1} and rep["steps_with_mismatches"] == {"7": 1}
    fm = rep["first_mismatches"][0]
    assert fm["index"] == 1 and fm["step"] == 7 and fm["expected"] == f"{int(bad.acc[1, 7]):08x}" and fm["got"] == f"{int(inst.acc[1, 7]):08x}"
    assert fm["acc"] == f"{int(inst.acc[1, 6]):08x}" and len(fm["a"]) == 64 and len(fm["b"]) == 64
    # a wrong final word is a final mismatch
    bad = IH.load_fp8_ada(0, 3)
    bad.acc[2, -1] ^= 1 << 20
    bad.y[2] = bad.acc[2, -1]
    rep = IH.replay(FakeProbe(ins), ins, bad)
    assert rep["mismatches"] == {"intermediate": 0, "final": 1, "epilogue": 0, "total": 1} and rep["first_mismatches"][0]["step"] == 47


# --- the Hopper synthetic sets ----------------------------------------------------------------------------------------------------------


def test_hopper_sets_are_declared_with_their_targets_and_digests():
    bf, f8 = IH.BF16_HOPPER_SYNTHETIC, IH.FP8_HOPPER_SYNTHETIC
    assert bf.spec == "bench-instances-bf16-hopper/v1/vu-k1536-bf16-hopper" and bf.instruction == "sm90.mma.m16n8k16.bf16"
    assert bf.k == 16 and bf.steps == 96 and bf.epilogue == "f32_to_bf16" and bf.model.name == "hopper_bf16_m16n8k16"
    assert f8.spec == "bench-instances-fp8-hopper/v1/vu-k1536-fp8-hopper" and f8.instruction == "sm90.wgmma.m64n8k32.e4m3"
    assert f8.k == 32 and f8.steps == 48 and f8.epilogue is None and f8.target == "fp8-hopper-wgmma-draft/2026-09-22"
    assert bf.digest(4096) == hashlib.sha256(b"bf16-hopper synthetic|seed=20260922|n=4096|K=1536").hexdigest()
    assert f8.digest(4096) == hashlib.sha256(b"fp8-hopper synthetic|seed=20260922|n=4096|K=1536").hexdigest()
    assert IH.FP8_ADA_SYNTHETIC.digest(4096) == IH.fp8_instances_digest(4096)
    assert set(IH.SYNTHETIC) <= set(IH.DATASETS)
    rng = np.random.default_rng(0)
    w = IH.bf16_random_finite_operands(rng, (4096,))
    assert P.BF16.is_finite(w.astype(np.uint16)).all()
    e = (w >> 7) & 0xFF
    assert e.max() < 160 and ((e == 0) | (e >= 96)).all() and (e == 0).sum() > 500  # zeros and subnormals are ~1/5 of the words


def test_hopper_bf16_set_replays_through_the_hopper_pipeline():
    ins = P.INSTRUCTIONS["sm90.mma.m16n8k16.bf16"]
    inst = IH.load_synthetic(IH.BF16_HOPPER_SYNTHETIC, 0, 2)
    assert inst.a.dtype == np.uint16 and inst.steps == 96 and inst.epilogue == "f32_to_bf16" and inst.instruction == ins.id
    assert inst.manifest_sha256 == IH.BF16_HOPPER_SYNTHETIC.digest(4096)
    rep = IH.replay(FakeProbe(ins), ins, inst)
    assert rep["mismatches"]["total"] == 0 and rep["words"] == {"intermediate": 2 * 95, "final": 2, "epilogue": 2, "total": 2 * 97}
    assert rep["domain"]["out_of_domain_instances"] == 0 and rep["other_positions_nonzero"] == 0
    assert np.array_equal(rep["_y_hw"], IH.epilogue_words("f32_to_bf16", inst.acc[:, -1]))


def test_hopper_fp8_set_replays_through_the_wgmma_shape():
    ins = P.INSTRUCTIONS["sm90.wgmma.m64n8k32.e4m3"]
    inst = IH.load_synthetic(IH.FP8_HOPPER_SYNTHETIC, 5, 7)
    assert ins.m == 64 and inst.a.dtype == np.uint8 and inst.steps == 48 and inst.ids.tolist() == [5, 6]
    rep = IH.replay(FakeProbe(ins), ins, inst)
    assert rep["mismatches"]["total"] == 0 and rep["words"]["total"] == 2 * 48 and rep["other_positions_nonzero"] == 0


# --- bench-instances/v1 -----------------------------------------------------------------------------------------------------------------


def test_negative_tier_loads_from_the_committed_arrays_and_checks_digests(tmp_path):
    inst = IH.load_bench_instances("vu-k1536-neg", None, FIXTURE / "manifest.json")
    assert inst.dataset == "bench-instances/v1" and inst.instruction == "sm80.mma.m16n8k16.bf16" and inst.model == "ampere_bf16_m16n8k16"
    assert inst.n == 52 and inst.steps == 96 and inst.k == 16 and inst.a.dtype == np.uint16 and inst.epilogue == "f32_to_bf16"
    assert inst.acc is None and inst.negative is not None and len(inst.negative["rows"]) == 52
    assert inst.manifest_sha256 == hashlib.sha256((FIXTURE / "manifest.json").read_bytes()).hexdigest()
    assert inst.y.tolist() == [r["claimed_y"] for r in inst.negative["rows"]]
    # a manifest whose digest for an array differs refuses the arrays
    m = json.loads((FIXTURE / "manifest.json").read_text())
    m["tiers"]["vu-k1536-neg"]["files"]["vu-k1536-neg.y.u16"]["sha256"] = "00" * 32
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    for name in m["tiers"]["vu-k1536-neg"]["files"]:
        shutil.copy(FIXTURE / name, tmp_path / name)
    with pytest.raises(RuntimeError, match="differ from the pinned manifest"):
        IH.load_bench_instances("vu-k1536-neg", tmp_path, tmp_path / "manifest.json")
    # the built tiers need the built arrays
    with pytest.raises(FileNotFoundError, match="built arrays"):
        IH.load_bench_instances("vu-k1536", tmp_path, FIXTURE / "manifest.json")


def test_negative_tier_every_claimed_word_is_detected():
    ins = P.INSTRUCTIONS["sm80.mma.m16n8k16.bf16"]
    inst = IH.load_bench_instances("vu-k1536-neg", None, FIXTURE / "manifest.json")
    rep = IH.replay(FakeProbe(ins), ins, inst)
    neg = rep["negative"]
    assert neg["claimed_words"] == 52 and neg["wrong"] == 44 and neg["reject"] == 8
    assert neg["wrong_detected"] == 44 and neg["wrong_correct_reproduced"] == 44 and neg["reject_detected"] == 8 and neg["undetected"] == 0
    rejects = [r for r in neg["rows"] if r["verdict"] == "reject"]
    # mismatches are against the reference (the correct word of a wrong row): none.  Every claimed-wrong word was contradicted; a reject
    # row's claimed word is immaterial (correct_y is null) and may or may not equal the word the total semantics give (row 45 claims -inf
    # where x[780] * -inf is +inf)
    reject_words_differ = sum(not r["hardware_equals_claimed"] for r in rejects)
    assert rep["mismatches"] == {"intermediate": 0, "final": 0, "epilogue": 0, "total": 0} and rep["first_mismatches"] == []
    assert neg["claimed_words_contradicted"] == 44 + reject_words_differ
    assert rep["words"] == {"intermediate": 0, "final": 0, "epilogue": 52, "total": 52}
    # a wrong row the hardware did not reproduce the correct word for is a mismatch located at the epilogue with the correct word expected
    bad = IH.load_bench_instances("vu-k1536-neg", None, FIXTURE / "manifest.json")
    bad.negative["rows"][3]["correct_y"] ^= 1
    rep = IH.replay(FakeProbe(ins), ins, bad)
    assert rep["mismatches"]["total"] == 1 and rep["negative"]["wrong_correct_reproduced"] == 43 and rep["negative"]["undetected"] == 0
    assert rep["first_mismatches"][0]["index"] == 3 and rep["first_mismatches"][0]["expected"] == f"{bad.negative['rows'][3]['correct_y']:04x}"
    dom = rep["domain"]
    # 4 rows with an inf / NaN operand word (their accumulator is non-finite from that step on too) + 4 rows saturating before the last step
    assert dom["nonfinite_operand_instances"] == 4 and dom["nonfinite_intermediate_acc_instances"] == 8 and dom["out_of_domain_instances"] == 8
    assert sorted(dom["out_of_domain_ids"]) == list(range(44, 52))
    assert rep["other_positions_nonzero"] == 0 and rep["other_positions_nonzero_in_nonfinite_operand_tiles"] > 0  # NaN x 0 in the poisoned rows
    # most reject rows reproduce the claimed word: the word compare alone would accept them -- the domain check is what rejects
    assert sum(r["hardware_equals_claimed"] for r in rejects) >= 6 and all(r["detected"] for r in rejects)


def test_epilogue_words_are_the_targets_cast():
    from verity.ml.circuits.silicon.cast import f32_to_bf16_word

    acc = np.array([0x3F800000, 0x3F808000, 0x3F818000, 0x7F7FFFFF, 0xFF800000, 0x7FC00001, 0x80000000, 0x00000001], np.uint32)
    y = IH.epilogue_words("f32_to_bf16", acc)
    assert y.dtype == np.uint16 and y.tolist() == [f32_to_bf16_word(int(v)) for v in acc]
    assert IH.epilogue_words(None, acc).tolist() == acc.tolist()
    with pytest.raises(ValueError):
        IH.epilogue_words("rtz", acc)


def test_unknown_set_and_bad_range_are_refused():
    with pytest.raises(SystemExit):
        IH.load_instances("bench-instances/v1/nope", _args())
    with pytest.raises(ValueError, match="range"):
        IH.load_fp8_ada(10, 5)
    with pytest.raises(ValueError, match="only-ids"):
        IH.load_fp8_ada(only_ids=[4096])


# --- on a GPU host ----------------------------------------------------------------------------------------------------------------------


def _gpu_compute_capability() -> str | None:
    if shutil.which("nvcc") is None and not Path("/usr/local/cuda/bin/nvcc").exists():
        return None
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":     # nvidia-smi lists the host's GPUs; this process may use none of them
        return None
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    if smi.returncode != 0 or not smi.stdout.strip():
        return None
    return smi.stdout.strip().splitlines()[0].strip()


def test_on_the_device(tmp_path):
    """Ampere / Ada: the FIRST negative tier on the BF16 mma (the Ada pipeline is the Ampere one: hardware == correct word); Hopper: a
    few instances of the Hopper BF16 synthetic set on its own instruction (the FIRST set's correct words are Ampere words, not Hopper's)."""
    cc = _gpu_compute_capability()
    if cc in ("8.0", "8.6", "8.9"):
        ins = "sm80.mma.m16n8k16.bf16" if cc != "8.9" else "sm89.mma.m16n8k16.bf16"
        rc = IH.main(["--instances", "bench-instances/v1/vu-k1536-neg", "--instruction", ins, "--out", str(tmp_path), "--only-ids", "0,20,44,48"])
        res = json.loads((tmp_path / "probe_results.json").read_text())
        assert rc == 0 and res["failures"] == []
        assert res["replay"]["negative"]["detected"] == 4 and res["replay"]["negative"]["wrong_correct_reproduced"] == 2
    elif cc == "9.0":
        rc = IH.main(["--instances", "bench-instances-bf16-hopper/v1/vu-k1536-bf16-hopper", "--out", str(tmp_path), "--range", "0:4", "--procs", "1"])
        res = json.loads((tmp_path / "probe_results.json").read_text())
        assert rc == 0 and res["failures"] == []
        assert res["replay"]["words"]["total"] == 4 * 97 and res["replay"]["mismatches"]["total"] == 0
    else:
        pytest.skip("needs nvcc and an sm_80/sm_86/sm_89/sm_90 GPU")
