"""`prove_leg.py defs`' served shapes: each family binds the Definition the frontend serves on the RTX PRO 6000 at a Llama
model's statics, a template the serving rows commit, with stand-in values for every parameter's words; and `units_for` is the
fewest units whose statement `stage` admits under --zk. Nothing is lowered or proved."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prove_leg as L  # noqa: E402

DOT = 'DOT={"fn":"HopperBF16WgmmaDot16_v1"}'
EPS = 'EPS={"f64":"0x1.4f8b588e368f1p-17"}'


def nbl(n: int, per_block: int) -> int:
    return max(3, (-(-n // per_block) - 1).bit_length())


@pytest.mark.parametrize("k_log,per_block", [(24, 1), (22, 1), (21, 1), (20, 2), (19, 4), (17, 8)])
def test_units_for_is_the_fewest_units_stage_admits(k_log, per_block):
    n = L.units_for(k_log, per_block)
    assert k_log + nbl(n, per_block) >= L.M_MIN
    assert n == 1 or k_log + nbl(n - 1, per_block) < L.M_MIN


@pytest.mark.parametrize("shape,want", [
    ("gemm:1b", f"Gemm_v2{{K=2048,N=1,{DOT}}}"),
    ("gemm:8b:4096", f"Gemm_v2{{K=4096,N=1,{DOT}}}"),
    ("attention:1b:129", f'Attention_v5{{T=129,NH=32,KVH=8,D=64,BN=128,{DOT},INV={{"fn":"Fa2InvSum_v1"}},MASKED_FROM=1}}'),
    ("attention:8b:65", f'Attention_v5{{T=65,NH=32,KVH=8,D=128,BN=64,{DOT},INV={{"fn":"Fa2InvSum_v1"}},MASKED_FROM=1}}'),
    ("attention-v18:1b:1", f'Attention_v18{{T=1,NH=32,KVH=8,D=64,BN=128,{DOT},INV={{"fn":"Fa2InvSum_v1"}},MASKED_FROM=0}}'),
    ("rmsnorm-fused:1b", f"RMSNormFusedCuda_v2{{N=2048,{EPS}}}"),
    ("rmsnorm-triton:8b", f"RMSNormTriton_v5{{N=4096,{EPS}}}"),
    ("embedding:1b:8", "Embedding_v1{V=8,H=2048}"),
    ("token-select:1b:200", "TokenSelect_v4{V=200}"),
    ("token-select-chunk:1b", "TokenSelectChunk_v2{C=64}"),
    ("token-select-combine:1b:256", "TokenSelectCombine_v2{V=256,C=64}"),
    ("rope:1b:8", "RoPE_v1{NHEADS=8,D=64}"),
])
def test_a_served_shape_binds_a_served_template_with_values_for_every_parameter(shape, want):
    from verity.ml.circuits.codec import descriptor_id
    from verity_vllm.pipeline import serving_rows as SRW

    fn, gen = L.served(shape)
    assert descriptor_id(fn) == want
    assert descriptor_id(SRW.template(want).fn) == want
    values = gen(np.random.default_rng(0), 2)
    params, _ret = fn.signature()
    assert {nm: v.shape for nm, v in values.items()} == {nm: (2, sum(t.leaf_widths()) // 16) for nm, t in params}
    assert all(v.dtype == np.uint16 for v in values.values())


def test_the_combines_partials_are_its_logits_rows_chunks():
    """The combine's stand-in is a logits row's: x0 its first logit, each chunk's partial its largest logit and that logit's
    first index in the chunk (the rows are finite, so `TokenSelectChunk_v2` is the first argmax)."""
    _fn, gen = L.served("token-select-combine:1b:256")
    v = gen(np.random.default_rng(0), 3)
    logits = L._gen(np.random.default_rng(0), 3, 256, 3.0)
    f = (logits.astype(np.uint32) << 16).view(np.float32).reshape(3, 4, 64)
    idx = v["idx"].reshape(3, 4, 2).astype(np.uint32)
    assert (v["x0"] == logits[:, :1]).all()
    assert ((idx[..., 0] | idx[..., 1] << 16) == f.argmax(-1)).all()
    assert (v["best"] == np.take_along_axis(logits.reshape(3, 4, 64), f.argmax(-1)[..., None], -1)[..., 0]).all()


def test_an_unknown_family_is_refused_by_name():
    with pytest.raises(SystemExit, match="no served family 'softmax'"):
        L.served("softmax:1b")


@pytest.mark.parametrize("reads", ["inline", "slots"])
def test_a_served_members_reads_reach_its_stager(monkeypatch, tmp_path, reads):
    """``--reads`` is the member's and its record's, and `stage` lays the unit's table reads out by it."""
    from verity_flock import class_statement as CS

    seen = {}

    class Built(Exception):
        pass

    def stager(*_a, **kw):
        seen.update(kw)
        raise Built

    monkeypatch.setattr(CS, "UnitStager", stager)
    m, info = L.served_member("rope:1b:8", 0, reads=reads)
    assert (m.reads, info["reads"]) == (reads, reads)
    with pytest.raises(Built):
        L.stage(m, 0, {"partition": "p", "program": "q"}, tmp_path, b"k" * 32, {})
    assert seen["reads"] == reads
