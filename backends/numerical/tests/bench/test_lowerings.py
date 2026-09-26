"""Per-template lowering modules (``bench.lowerings``): every family has one module per catalogue template, each declaring
what it lowers; today's GEMM cells resolve to the relation names their pod scripts take."""
from __future__ import annotations

import sys

import pytest

from verity_numerical.bench import lowerings as L
from verity_numerical.bench import templates as TM

FAMILIES = sorted(L.DIRECTORIES)
EXAMPLES = [TM.subcircuit(t.id, **t.example) for t in TM.TEMPLATES.values()]


@pytest.mark.parametrize("family", FAMILIES)
def test_one_module_per_template(family):
    reg = L.registry(family)
    assert reg.problems() == []
    assert set(reg.modules()) == set(TM.TEMPLATES)


@pytest.mark.parametrize("family", FAMILIES)
def test_unlowered_templates_say_why(family):
    reg = L.registry(family)
    for sub in EXAMPLES:
        why = reg.supports(sub, "frame-v3/blake3-keyed")
        if sub.template.id != "gemm-coordinate":
            assert isinstance(why, str) and family in why


CELLS = [("C-Flock", "gemm-coordinate/k1536/sm90-mma-bf16", "frame-v3/blake3-keyed", "bf16-hopper"),
         ("C-Flock", "gemm-coordinate/k1536/sm89-mma-e4m3", "frame-v3/blake3-keyed", "fp8-ada"),
         ("C-Flock", "gemm-coordinate/k1536/sm90-wgmma-e4m3", "frame-v3/blake3-keyed", "fp8-hopper"),
         ("B-Ligero", "gemm-coordinate/k1536/sm90-wgmma-e4m3", "frame-v3/blake3-keyed", "fp8-hopper+blake3"),
         ("B-Ligero", "gemm-coordinate/k1536/sm80-mma-bf16", "frame-v3/sha-256", "bf16-ampere+sha256"),
         ("A-GKR", "gemm-coordinate/k1536/sm80-mma-bf16", "frame-v3/blake3-keyed", "bf16-ampere+blake3"),
         ("D-SP1", "gemm-coordinate/k1536/sm80-mma-bf16", "frame-v3/sha-256", "bf16-ampere+frame-v3"),
         ("B-Ligero", "gemm-coordinate/k2048/sm80-mma-bf16", "frame-v3/blake3-keyed", "bf16-ampere-x4-k2048+blake3-xob"),
         ("B-Ligero", "gemm-coordinate/k8192/sm90-wgmma-bf16", "frame-v3/sha-256", "bf16-hopper-x4-k8192+sha256"),
         ("B-Ligero", "gemm-coordinate/k8192/sm89-mma-e4m3", "frame-v3/blake3-keyed", "fp8-ada-x4-k8192+blake3-xob")]


@pytest.mark.parametrize("family,sid,scheme,relation", CELLS)
def test_todays_gemm_cells(family, sid, scheme, relation):
    reg = L.registry(family)
    sub = TM.parse_subcircuit_id(sid)
    assert reg.supports(sub, scheme) is None
    assert reg.relation(sub, scheme) == relation


def test_relation_names_are_the_backends_own():
    from verity_flock import lowering as FL

    reg = L.registry("C-Flock")
    assert set(reg.module("gemm-coordinate").RELATIONS.values()) <= set(FL.PIPES)
    sys.path.insert(0, str(L.REPO / "backends" / "direct"))
    from ligero.relations import RELATIONS

    mod = L.registry("B-Ligero").module("gemm-coordinate")
    assert set(mod.RELATIONS.values()) <= set(RELATIONS)
    assert {f"{b}-x4-k{k}" for b in mod.REAL_K_RELATIONS.values() for k in mod.REAL_KS} <= set(RELATIONS)
    assert "vllm-v1" in L.registry("B-Ligero").supports(TM.parse_subcircuit_id("gemm-coordinate/k2048/sm80-mma-bf16"), "vllm-v1/sha-256")


def test_real_k_and_other_schemes_are_not_supported():
    reg = L.registry("C-Flock")
    k2048 = TM.subcircuit("gemm-coordinate", K=2048)
    assert "semantics" in reg.supports(k2048, "frame-v3/blake3-keyed")
    why = reg.supports(TM.subcircuit("gemm-coordinate", K=2048, semantics="sm90.mma.m16n8k16.bf16"), "frame-v3/blake3-keyed")
    assert "K = 2048" in why
    assert "sha-256" in reg.supports(TM.parse_subcircuit_id("gemm-coordinate/k1536/sm90-mma-bf16"), "frame-v3/sha-256")
    with pytest.raises(ValueError):
        reg.relation(k2048, "frame-v3/blake3-keyed")


def test_missing_and_misnamed_modules(tmp_path):
    (tmp_path / "rope_head.py").write_text('TEMPLATE = "silu-mul"\n\ndef supports(s, c):\n    return None\n')
    (tmp_path / "extra.py").write_text('TEMPLATE = "no-such"\n')
    reg = L.Registry("X-Test", tmp_path)
    probs = reg.problems()
    assert any("gemm_coordinate.py missing" in p for p in probs)
    assert any("rope_head.py lowers 'silu-mul'" in p for p in probs)
    assert any("extra.py names TEMPLATE 'no-such'" in p for p in probs)
    assert any("extra.py has no supports" in p for p in probs)
    assert "no lowering module" in reg.supports(TM.subcircuit("gemm-coordinate", K=64), "frame-v3/blake3-keyed")


def test_support_matrix():
    m = L.support_matrix([TM.parse_subcircuit_id("gemm-coordinate/k1536/sm80-mma-bf16"), TM.subcircuit("rope-head", D=64)],
                         ["frame-v3/sha-256"])
    assert m["D-SP1"]["gemm-coordinate/k1536/sm80-mma-bf16+frame-v3/sha-256"] is None
    assert isinstance(m["C-Flock"]["rope-head/d64/neox-bf16+frame-v3/sha-256"], str)
