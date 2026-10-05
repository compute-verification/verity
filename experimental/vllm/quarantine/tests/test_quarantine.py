"""The experimental quarantine lanes, from where they now live: each module passes the quarantine's purity and identity lint, a
lane's re-export of a promoted Definition is the registry's own object, each difftest adapter builds its specialization
without torch, and the lanes here are the migration's `MOVE` while core's quarantine keeps none (the lanes core bound were
promoted into the registry)."""
from __future__ import annotations

import importlib
import pkgutil
import random
import re
from pathlib import Path

import pytest

import verity_quarantine
from verity_quarantine import act, collective, dense, ov_moe
from verity_quarantine.dense import softcap
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import act as R_act
from verity_vllm.program.registry import dense as R_dense
from verity_vllm.program.registry import moe as R_moe
from verity_vllm.program.registry import quarantine as core_quarantine
from verity_vllm.properties.quarantine_lint import lint

ROOT = Path(verity_quarantine.__file__).resolve().parent
MODULES = sorted(m.name for m in pkgutil.walk_packages(verity_quarantine.__path__, "verity_quarantine."))
ADAPTERS = [m for m in MODULES if m.endswith("_difftest")]


def _lanes(pkg) -> set[str]:
    return {m.name for m in pkgutil.iter_modules(pkg.__path__) if m.ispkg}


def test_every_module_imports():
    for name in MODULES:
        importlib.import_module(name)
    assert {"verity_quarantine.act.silu_mul", "verity_quarantine.collective.allreduce", "verity_quarantine.ov_moe.moe"} <= set(MODULES)


def test_every_module_passes_the_quarantine_lint():
    rep = lint(ROOT)
    assert rep["n_files"] == len([p for p in ROOT.rglob("*.py") if p != ROOT / "__init__.py"])
    assert rep["status"] == "pass", [(f["file"], f["problems"]) for f in rep["files"] if f["status"] != "pass"]


def test_the_re_exports_are_the_registry_s_definitions():
    assert act.silu_mul.SiluMul is R_act.SiluMul and act.silu_mul_bf16.SiluMulBf16 is R_act.SiluMulBf16
    assert [d.id for d in dense.ALL_DEFINITIONS + dense.ALL_PRIMITIVES
            if not (d is getattr(R_dense, d.name, None) or d is getattr(b1, d.name, None))] == []
    assert dense.mean_triton.MeanTriton is R_dense.MeanTriton and softcap.AttentionSoftcap is R_dense.AttentionSoftcap
    assert ov_moe.moe.MoeRouterTopK is R_moe.MoeRouterTopK and ov_moe.moe.MoeExpertGemm is R_moe.MoeExpertGemm


def test_the_collective_definitions_are_registered_under_their_ids():
    from verity.ir.defs import REGISTRY
    assert [d.id for d in collective.ALL_DEFINITIONS] == ["AllReduceSumBf16Elem_v1", "AllReduceSumBf16_v1"]
    assert all(d.id in REGISTRY for d in collective.ALL_DEFINITIONS)


@pytest.mark.parametrize("name", ADAPTERS)
def test_an_adapter_builds_its_definition_without_torch(name):
    mod = importlib.import_module(name)
    lane, defn, version = re.fullmatch(r"([\w-]+)/(\w+)@v(\d+)", mod.SEMANTIC_ID).groups()
    assert callable(mod.production)
    for i in range(6):
        sp = mod.spec(mod.cases(random.Random(7), i)["statics"])
        assert (sp.defn.name, sp.defn.version) == (defn, int(version))


def test_the_lanes_here_and_in_core_are_the_migration_s():
    assert _lanes(verity_quarantine) == {"act", "collective", "dense", "ov_moe"}
    assert _lanes(core_quarantine) == set()
