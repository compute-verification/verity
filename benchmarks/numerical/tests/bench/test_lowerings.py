"""Per-template lowering modules (``bench.lowerings``): every family has one module per catalogue template, each declaring
what it lowers; today's GEMM cells resolve to the relation names their pod scripts take."""
from __future__ import annotations

import pytest

from verity_numerical.bench import lowerings as L
from verity_numerical.bench import templates as TM

FAMILIES = sorted(L.DIRECTORIES)
EXAMPLES = [TM.subcircuit(t.id, **t.example) for t in TM.TEMPLATES.values()]


@pytest.mark.parametrize("family", FAMILIES)
def test_one_module_per_template(family):
    reg = L.registry(family)
    assert reg.problems() == []
    assert set(reg.modules()) | set(reg.missing()) == set(TM.TEMPLATES) and not set(reg.modules()) & set(reg.missing())
    for tid in reg.missing():
        sub = TM.subcircuit(tid, **TM.TEMPLATES[tid].example)
        assert f"{L.module_name(tid)}.py" in reg.supports(sub, "frame-v3/blake3-keyed")


#: Templates a family proves under frame-v3/blake3-keyed besides the GEMM (C-Flock: verity/flock-ir-frame/v3; sampling under
#: verity/flock-ir-sampling/v1).
LOWERED = {"C-Flock": {"rope-head", "silu-mul", "silu-mul-v2", "rmsnorm-fused-cuda", "rmsnorm-triton", "attention-head",
                       "gumbel-top-p-token-select", "ingress-prompt", "token-select-chunk", "token-select-combine"}}


@pytest.mark.parametrize("family", FAMILIES)
def test_unlowered_templates_say_why(family):
    reg = L.registry(family)
    for sub in EXAMPLES:
        why = reg.supports(sub, "frame-v3/blake3-keyed")
        if sub.template.id in LOWERED.get(family, ()):
            assert why is None and isinstance(reg.supports(sub, "vllm-v1"), str)
        elif sub.template.id != "gemm-coordinate":
            assert isinstance(why, str) and family in why


CELLS = [("C-Flock", "gemm-coordinate/k1536/sm90-mma-bf16", "frame-v3/blake3-keyed", "bf16-hopper"),
         ("C-Flock", "gemm-coordinate/k1536/sm89-mma-e4m3", "frame-v3/blake3-keyed", "fp8-ada"),
         ("C-Flock", "gemm-coordinate/k1536/sm90-wgmma-e4m3", "frame-v3/blake3-keyed", "fp8-hopper"),
         ("C-Flock", "gemm-coordinate/k2048/sm89-mma-e4m3", "frame-v3/blake3-keyed", "fp8-ada"),
         ("C-Flock", "gemm-coordinate/k8192/sm90-wgmma-bf16", "frame-v3/blake3-keyed", "bf16-hopper-wgmma")]


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


def test_chunked_real_k_and_other_schemes():
    reg = L.registry("C-Flock")
    k2048 = TM.subcircuit("gemm-coordinate", K=2048)
    assert reg.supports(k2048, "frame-v3/blake3-keyed") is None
    assert reg.supports(TM.subcircuit("gemm-coordinate", K=2048, semantics="sm90.mma.m16n8k16.bf16"), "frame-v3/blake3-keyed") is None
    why = reg.supports(TM.subcircuit("gemm-coordinate", K=512, semantics="sm90.mma.m16n8k16.bf16"), "frame-v3/blake3-keyed")
    assert "CN3" in why
    assert "sha-256" in reg.supports(TM.parse_subcircuit_id("gemm-coordinate/k1536/sm90-mma-bf16"), "frame-v3/sha-256")
    with pytest.raises(ValueError):
        reg.relation(k2048, "frame-v3/sha-256")


def test_missing_and_misnamed_modules(tmp_path):
    (tmp_path / "rope_head.py").write_text('TEMPLATE = "silu-mul"\n\ndef supports(s, c):\n    return None\n')
    (tmp_path / "extra.py").write_text('TEMPLATE = "no-such"\n')
    reg = L.Registry("X-Test", tmp_path)
    probs = reg.problems()
    assert "gemm-coordinate" in reg.missing() and not any("gemm_coordinate" in p for p in probs)
    assert any("rope_head.py lowers 'silu-mul'" in p for p in probs)
    assert any("extra.py names TEMPLATE 'no-such'" in p for p in probs)
    assert any("extra.py has no supports" in p for p in probs)
    assert "no lowering module" in reg.supports(TM.subcircuit("gemm-coordinate", K=64), "frame-v3/blake3-keyed")


def test_support_matrix():
    m = L.support_matrix([TM.parse_subcircuit_id("gemm-coordinate/k1536/sm80-mma-bf16"), TM.subcircuit("rope-head", D=64)],
                         ["frame-v3/sha-256"])
    assert set(m) == {"C-Flock"}
    assert isinstance(m["C-Flock"]["rope-head/d64/neox-bf16+frame-v3/sha-256"], str)


def test_flock_stage_writes_an_input_set_range(tmp_path):
    import json

    import numpy as np

    from verity_numerical.bench import generate as G, input_sets as IS

    sub = TM.parse_subcircuit_id("gemm-coordinate/k2048/sm80-mma-bf16")
    stage = L.registry("C-Flock").stage(sub)
    d = tmp_path / "set"
    G.generate(sub, 4, 7, d)
    s = IS.InputSet.open(d)
    out = stage(s, 1, 3, tmp_path / "inst")
    h = json.loads(out.read_bytes().split(b"\n", 1)[0])
    assert (h["vus"], h["k"], h["instances"]["range"], h["instances"]["manifest_sha256"]) == (2, 2048, [1, 3], s.content_digest)
    assert np.fromfile(s.path / "x.u16", dtype="<u2").size == 4 * 2048
