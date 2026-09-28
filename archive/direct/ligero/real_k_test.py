"""The x4 folds at vLLM's reduction lengths (``relations._at_k``): names and shapes, the same compiled column as K = 1536,
input-set instances (``relchain.set_instances``), and the leaf frames a real-K row composes with."""
from __future__ import annotations

import numpy as np
import pytest

from .relations import REAL_KS, RELATIONS, relation


def test_every_fold_is_registered_at_each_real_k() -> None:
    for base in ("bf16-ampere-x4", "bf16-hopper-x4", "fp8-ada-x4", "fp8-hopper-x4"):
        b = relation(base)
        for K in REAL_KS:
            r = relation(f"{base}-k{K}")
            assert (r.K, r.k, r.steps, r.fold, r.frozen_tier) == (K, b.k, K // b.k, 4, None)
            assert r.params == b.params and r.model == b.model and r.target is b.target
            assert r.tag == (b"" if base.startswith("bf16") else f"|rel={base}-k{K}|v1".encode())
            assert f"<{K}>" in r.relation_name and r.rust_verifiable and r.hooks.steps == r.steps
    assert b.K == 1536 and sum(1 for n in RELATIONS if "-k" in n) == 4 * len(REAL_KS)


def test_the_bare_column_is_the_folds_at_1536() -> None:
    """Column-uniform: the compiled system is a function of the params only, so K moves only the chain's shape (the
    statement's steps, which the verifier's steps pin checks) -- the Rust registry reuses the fold's bare pins."""
    from .protocol import system_id
    a, b = relation("fp8-ada-x4"), relation("fp8-ada-x4-k8192")
    assert system_id(a.compile(a.params, True)) == system_id(b.compile(b.params, True))


def _gemm_set(tmp_path, K: int, semantics: str, n: int = 6):
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.templates import gemm_coordinate
    return generate(gemm_coordinate(K, semantics), n, 20260926, tmp_path)["path"]


@pytest.mark.parametrize("name,semantics", [("bf16-ampere-x4-k2048", "sm80.mma.m16n8k16.bf16"),
                                            ("bf16-hopper-x4-k2048", "sm90.wgmma.m64n8k16.bf16")])
def test_set_instances_reproduce_the_sets_outputs(tmp_path, name, semantics) -> None:
    """The prover's instances of an input set: the set's rows, column-end accumulators recorded by the relation's model
    (every fold-th step of ``tc_dot``), whose last casts to each recorded output -- the Hopper mma model reproduces the
    wgmma set word for word (PR #49)."""
    from verity.ml.circuits.silicon.silicon import tc_dot
    from verity_numerical.bench.input_sets import InputSet

    from .relchain import set_instances, set_ref
    rel = relation(name)
    path = _gemm_set(tmp_path, rel.K, semantics)
    vus = set_instances(rel, path, 6)
    s = InputSet.open(path)
    x, w, y = s.port("x"), s.port("w"), s.port("y")
    assert np.array_equal(vus.x_rows, x) and np.array_equal(vus.w_cols, w)
    for i, (a, b, accs, yf) in enumerate(vus):
        c, want = 0, []
        for col in range(rel.steps):
            c = tc_dot(rel.model, c, a[rel.k * col:rel.k * col + rel.k], b[rel.k * col:rel.k * col + rel.k])
            want.append(c)
        assert accs == want and yf == want[-1] and rel.y_public(yf) == int(y[i, 0])
    ref = set_ref(path, 4, art="art:x")
    assert ref["range"] == [0, 4] and ref["content_digest"] == s.content_digest and ref["subcircuit"] == s.subcircuit.id
    with pytest.raises(ValueError, match="not recycled"):
        set_instances(rel, path, 7)
    with pytest.raises(ValueError, match="K = 2048"):
        set_instances(relation("bf16-ampere-x4-k8192"), path, 2)
    # a tampered output word is refused
    yfile = s.path / s.manifest["ports"]["y"]["file"]
    yb = bytearray(yfile.read_bytes())
    yb[0] ^= 1
    yfile.write_bytes(bytes(yb))
    with pytest.raises(ValueError, match="recorded outputs"):
        set_instances(rel, path, 6)


def test_real_k_rows_compose_with_a_sized_blake3_frame() -> None:
    from .leaf import registry
    from .serialize import Statement, _K_of
    for name, want in (("bf16-ampere-x4-k2048+blake3-xob", 4), ("bf16-hopper-x4-k8192+blake3", 16),
                       ("fp8-ada-x4-k8192+blake3-xob", 8), ("fp8-hopper-x4-k2048+blake3-xob", 3), ("bf16-ampere-x4+blake3", 3)):
        s = registry.for_relation(name)
        assert s.n_chunks_max == want and s.digest_elems == 1 + 16 * want and s.name == registry.split_relation_name(name)[1]
    assert registry.for_relation("bf16-hopper-x4-k8192+sha256").digest_elems == 17
    st = Statement.__new__(Statement)
    for rel_name, K in (("bf16-ampere-x4-k2048+sha256", 2048), ("fp8-ada-x4-k8192", 8192), ("fp8-ada-x4+blake3", 1536)):
        st.relation = rel_name
        assert _K_of(st) == K


@pytest.mark.parametrize("name,semantics", [("fp8-ada-x4-k2048", "sm89.mma.m16n8k32.e4m3"),
                                            ("fp8-hopper-x4-k2048", "sm90.wgmma.m64n8k32.e4m3")])
def test_fp8_sets_are_read_as_their_fp32_accumulators(tmp_path, name, semantics) -> None:
    """An E4M3 set records the FP32 accumulator itself (a u32 ``y``, no epilogue): the reader compares it word for word with the
    model's (not the 22-bit public packing of it), so a flip in any bit of the recorded word is refused."""
    from verity_numerical.bench.input_sets import InputSet

    from .relchain import set_instances
    rel = relation(name)
    path = _gemm_set(tmp_path, rel.K, semantics, n=4)
    s = InputSet.open(path)
    assert s.manifest["ports"]["y"]["dtype"] == "u32"
    vus = set_instances(rel, path, 4)
    assert [v[3] for v in vus] == [int(w) for w in s.port("y")[:, 0]]
    yfile = s.path / s.manifest["ports"]["y"]["file"]
    yb = bytearray(yfile.read_bytes())
    yb[0] ^= 1                                                   # bit 0 of instance 0's FP32 word
    yfile.write_bytes(bytes(yb))
    with pytest.raises(ValueError, match="FP32 accumulator"):
        set_instances(rel, path, 4)
