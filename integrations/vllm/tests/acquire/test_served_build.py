"""The served Build (`acquire.served`) from a served-compiled process's own artifact: Llama-3.2-1B served at vLLM's defaults on an
H200 (`fixtures/served_llama.tar.gz`: the three Inductor wrappers its 17 pieces run, vLLM's split graph, the kernels' launch configs,
the cuBLAS routes probed on that GPU, and the four-request mixed workload).  The Build reads every wrapper line, binds every launch
to a Definition, and reproduces the CUDA-graph mode, padding and FA3 split count of each of the 14 steps the process served; a
line, a kernel source, a hook or a launch config it does not model is refused by name, and a GEMM route it lacks leaves it
incomplete."""
from __future__ import annotations

__models__ = ("meta-llama/Llama-3.2-1B",)

import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from verity_vllm.acquire import compiled as C
from verity_vllm.acquire import served as SV
from verity_vllm.acquire.inductor_templates import Unmodelled

FIXTURE = Path(__file__).parent / "fixtures" / "served_llama.tar.gz"
#: (mode, padded, tokens) of each engine step, as the serving process dispatched them (its capture's record)
SERVED = [("NONE", 137, 137), ("FULL", 2, 2), ("FULL", 2, 2), ("PIECEWISE", 16, 11), ("FULL", 4, 3), ("FULL", 4, 3),
          ("NONE", 303, 303), ("FULL", 4, 4), ("FULL", 4, 4), ("FULL", 4, 3), ("FULL", 4, 3), ("FULL", 2, 2), ("FULL", 1, 1),
          ("FULL", 1, 1)]


@pytest.fixture(scope="module")
def art(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("served")
    with tarfile.open(FIXTURE) as t:
        t.extractall(d, filter="data")
    return d / "served_llama"


def _build(d: Path, inductor: Path | None = None, **kw):
    wl = json.loads((d / "workload.json").read_text())
    args = dict(split_graph=str(d / "computation_graph.py"), launch_configs=json.loads((d / "launch_configs.json").read_text()),
                routes=json.loads((d / "routes.json").read_text()), schedule=C.schedule_of_workload(wl), steps=14,
                capture_sizes=[1, 2, 4, 8, 16], max_num_seqs=8, sms=132)
    return SV.build(str(inductor or d / "inductor"), **{**args, **kw})


def _copy(art: Path, tmp: Path) -> Path:
    shutil.copytree(art / "inductor", tmp / "inductor")
    return tmp / "inductor"


def _edit(inductor: Path, old: str, new: str, count: int = 1) -> None:
    hits = [p for p in sorted(inductor.rglob("*.py")) if old in p.read_text()]
    assert hits, old
    for p in hits:
        p.write_text(p.read_text().replace(old, new, count))


def test_the_build_is_the_served_computation(art: Path) -> None:
    m = _build(art)
    p = m["program"]
    assert m["complete"] and not m["routes_missing"]
    assert len(p["piece_wrappers"]) == 17 and len(set(p["piece_wrappers"])) == 3
    assert Counter(c.get("base", c["op"]) for c in p["calls"]) == {
        "mm": 64, "InductorRoPE_v1": 32, "InductorAddRMSNorm_v1": 16, "InductorSiluMul_v1": 16, "InductorAdd2RMSNorm_v1": 15,
        "Embedding_v1": 1, "InductorRMSNorm_v1": 1, "InductorAdd2RMSNormFinal_v1": 1}
    assert len(p["attention"]) == 16 and len(p["values"]) == 179 and p["final"]["value"] == "c16.mul/buf7"
    assert [(s["mode"], s["padded"], s["tokens"]) for s in m["steps_plan"]] == SERVED
    assert [s["step"] for s in m["steps_plan"] for r in s["requests"] if r["NS"] > 1] == [7, 8, 9, 10, 11]
    assert {r["NS"] for s in m["steps_plan"] for r in s["requests"]} == {1, 2}
    assert len(m["identities"]) == 6697
    assert m["program_digest"] == ("1f2df5e1029f878debf647a562314108b4e15159e90c5a852adefe9ade198707"
                                   "7c2b4cbf5621209148cdafbdc1113f15cbafd7a9c3f38d11a70659c291062a6b")


def test_the_gemm_route_follows_the_step(art: Path) -> None:
    plans = {s["step"]: s for s in _build(art)["steps_plan"]}
    down = lambda s: plans[s]["gemms"]["8192,2048"]
    assert down(1)["base"] == "CublasSplitKGemm_v1" and down(1)["statics"]["SPLIT_K"] == 3
    assert down(0)["statics"]["SPLIT_K"] == 2
    assert down(6)["base"] == "Gemm_v2"


def test_a_missing_route_leaves_the_build_incomplete(art: Path) -> None:
    m = _build(art, routes={})
    assert not m["complete"] and len(m["routes_missing"]) == 28
    assert all(g is None for s in m["steps_plan"] for g in s["gemms"].values())


@pytest.mark.parametrize("old,new,why", [
    ("        args.clear()\n", "        args.clear()\n        print(args)\n", "wrapper line not modelled"),
    ("tmp3 = libdevice.exp(tmp2)", "tmp3 = libdevice.exp2(tmp2)", "template"),
    ("'enable_fp_fusion': True", "'enable_fp_fusion': False", "enable_fp_fusion"),
])
def test_an_unmodelled_artifact_is_refused(art: Path, tmp_path: Path, old: str, new: str, why: str) -> None:
    ind = _copy(art, tmp_path)
    _edit(ind, old, new, count=-1)
    with pytest.raises(Unmodelled):
        _build(art, inductor=ind)


def test_an_unhooked_buffer_is_refused(art: Path, tmp_path: Path) -> None:
    ind = _copy(art, tmp_path)
    _edit(ind, "        run_intermediate_hooks('verity|mul|buf4', buf4)\n", "", count=-1)
    with pytest.raises(Unmodelled, match="no run_intermediate_hooks names them"):
        _build(art, inductor=ind)


def test_an_autotuned_kernel_is_refused(art: Path) -> None:
    lc = json.loads((art / "launch_configs.json").read_text())
    key, rec = next((k, v) for k, v in lc.items() if v["candidates"][0]["kwargs"].get("R0_BLOCK"))
    other = {**rec["candidates"][0], "num_warps": 8}
    with pytest.raises(Unmodelled, match="one candidate is required"):
        _build(art, launch_configs={**lc, key: {"candidates": [rec["candidates"][0], other], "executed": rec["executed"]}})


@pytest.mark.parametrize("n,uniform,want", [(3, True, ("FULL", 4)), (8, True, ("FULL", 8)), (9, True, ("PIECEWISE", 16)),
                                            (11, False, ("PIECEWISE", 16)), (17, True, ("NONE", 17)), (1, False, ("PIECEWISE", 1))])
def test_dispatch(n: int, uniform: bool, want: tuple[str, int]) -> None:
    assert SV.dispatch(n, uniform, [1, 2, 4, 8, 16], 8) == want


def test_the_full_graph_caps_fa3_splits() -> None:
    kw = dict(NH=32, KVH=8, D=64, sms=132)
    assert SV.fa3_splits([1, 1, 1, 1], [107, 44, 13, 301], split_cap=32, **kw)["NS"] == [1, 1, 1, 2]
    assert SV.fa3_splits([1, 1, 1, 300], [106, 43, 12, 300], split_cap=0, **kw)["NS"] == [1, 1, 1, 1]
