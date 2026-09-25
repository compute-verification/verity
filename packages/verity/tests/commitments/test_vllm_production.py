"""``vllm-production/v1``: the reference reproduces its conformance vectors, and its verifier accepts exactly the
vector openings and rejects the rejection vectors.  That vLLM's own code produces the same bytes is checked in
``integrations/vllm/tests/commit/test_production_vectors.py``."""

from __future__ import annotations

import pytest

from verity.commitments import vllm_production as ref
from verity.commitments.scheme import CommitmentScheme
from verity.commitments.vllm_production import vectors as V
from verity.errors import InvalidArtifact

VECTORS = V.load()


def _b(text: str) -> bytes:
    return bytes.fromhex(text)


def _path(entries: list[str | None]) -> list[bytes | None]:
    return [None if e is None else _b(e) for e in entries]


def test_vectors_file_is_what_the_reference_generates() -> None:
    assert V.main(["--check"]) == 0, "rerun `python -m verity.commitments.vllm_production.vectors`"


def test_scheme_implements_the_interface() -> None:
    assert isinstance(ref.SCHEME, CommitmentScheme)
    assert ref.SCHEME.name == VECTORS["scheme"] == "vllm-production/v1"


@pytest.mark.parametrize("case", VECTORS["trees"], ids=lambda c: f"n={c['leaf_count']}")
def test_every_vector_opening_verifies(case: dict) -> None:
    leaves, root = [_b(x) for x in case["leaves"]], _b(case["root"])
    for index, path in enumerate(case["paths"]):
        assert ref.SCHEME.verify_path(root, case["leaf_count"], index, leaves[index], _path(path))
        if case["leaf_count"] > 1:
            assert not ref.SCHEME.verify_path(root, case["leaf_count"], index, leaves[index - 1], _path(path))


@pytest.mark.parametrize("case", VECTORS["rejections"], ids=lambda c: c["why"])
def test_rejection_vectors_are_rejected(case: dict) -> None:
    assert not ref.verify_opening(_b(case["root"]), case["leaf_count"], case["index"], _b(case["leaf"]), _path(case["path"]))


def _domain(case: dict):
    d = dict(case["domain"])
    for key in [k for k in d if k.endswith("_digest") or k == "query_id"]:
        d[key] = _b(d[key])
    kind = case["kind"]
    if kind == "semantic":
        return ref.SemanticDomain(**d)
    if kind == "step":
        return ref.StepDomain(**d)
    if kind == "stream":
        return ref.StreamDomain(**d)
    return ref.ThreadDomain(**d)


def _opening_accepted(case: dict) -> bool:
    domain = _domain(case)
    try:
        if case["kind"] == "thread":
            p = case["position"]
            index = domain.index(p["slab"], p["m_block"], p["n_block"], p["tidx"])
            leaf = domain.leaf(_b(case["value"]), **p)
        else:
            index = case["index"]
            leaf = domain.leaf(index, _b(case["value"]))
    except InvalidArtifact:
        return False
    return index == case["index"] and ref.verify(domain, _b(case["root"]), index, leaf, _path(case["path"]))


@pytest.mark.parametrize("case", VECTORS["openings"], ids=lambda c: f"{c['kind']}-{c['index']}-{'accept' if c['accept'] else 'reject'}")
def test_domain_openings(case: dict) -> None:
    assert _opening_accepted(case) is case["accept"]


def test_every_domain_kind_has_a_positive_and_a_moved_negative() -> None:
    seen = {(c["kind"], c["accept"]) for c in VECTORS["openings"]}
    assert seen == {(k, a) for k in ("semantic", "step", "stream", "thread") for a in (True, False)}


def test_id_leaf_vectors() -> None:
    *leaves, ids = VECTORS["id_leaf"]
    for case in leaves:
        assert ref.id_leaf(case["canonical_id"], case["dtype"], tuple(case["shape"]), _b(case["value"])).hex() == case["leaf"]
    assert ref.ids_digest(ids["ids"]).hex() == ids["ids_digest"]


def test_binding_is_the_verifiers_not_the_provers() -> None:
    leaves = V.tree_leaves(5)
    tree_root = ref.SCHEME.root(leaves)
    program, ctx, geo, layout = (V.digest_of(k) for k in "pcgl")
    root = ref.step_root(program, ctx, geo, layout, 5, tree_root)
    path = ref.SCHEME.path(leaves, 2)
    assert ref.verify_opening(root, 5, 2, leaves[2], path, bind=lambda r: ref.step_root(program, ctx, geo, layout, 5, r))
    assert not ref.verify_opening(root, 5, 2, leaves[2], path, bind=lambda r: ref.step_root(program, geo, ctx, layout, 5, r))
    assert not ref.verify_opening(root, 5, 2, leaves[2], path)


def test_headers_refuse_fields_that_would_alias() -> None:
    fields = {"launch_tag": 0, "chunk_index": 0, "chunk_words": 64, "HB": 1, "M": 1, "NB": 1}
    with pytest.raises(InvalidArtifact):
        ref.chunk_header(src_mask=256, **fields)
    with pytest.raises(InvalidArtifact):
        ref.chunk_header(src_mask=0, **{**fields, "chunk_index": 1 << 32})
    with pytest.raises(InvalidArtifact):
        ref.thread_header(src_mask=1 << 8, slab=0, m_block=0, n_block=0, tidx=0, seqlen_q=1, seqlen_k=1, launch_tag=0, n_block_max=1,
                          ok0=True, ok1=True)
    with pytest.raises(InvalidArtifact):
        ref.step_root(b"\0" * 31, *(b"\0" * 32,) * 3, 1, b"\0" * 32)
