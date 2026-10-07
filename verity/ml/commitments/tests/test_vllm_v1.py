"""``vllm-v1``: the reference reproduces its conformance vectors, and its verifier accepts exactly the
vector openings and rejects the rejection vectors.  That vLLM's own code produces the same bytes is checked in
``integrations/vllm/tests/commit/test_production_vectors.py``."""

from __future__ import annotations

import hashlib

import pytest

from verity.ml.commitments import vllm_v1 as ref
from verity.ml.commitments.scheme import CommitmentScheme
from verity.ml.commitments.vllm_v1 import vectors as V
from verity.errors import InvalidArtifact

VECTORS = V.load()


def _b(text: str) -> bytes:
    return bytes.fromhex(text)


def _path(entries: list[str | None]) -> list[bytes | None]:
    return [None if e is None else _b(e) for e in entries]


def test_vectors_file_is_what_the_reference_generates() -> None:
    assert V.main(["--check"]) == 0, "rerun `python -m verity.ml.commitments.vllm_v1.vectors`"


def test_scheme_implements_the_interface() -> None:
    scheme = ref.VllmV1({})
    assert isinstance(scheme, CommitmentScheme)
    assert scheme.name == ref.NAME == VECTORS["scheme"] == "vllm-v1"
    assert scheme.hashes == ("sha256",)


@pytest.mark.parametrize("case", VECTORS["layouts"], ids=lambda c: c["leaf"])
def test_layout_vectors_are_the_leaf_rules(case: dict) -> None:
    value = _b(case["value"])
    assert hashlib.sha256(_b(case["prefix"]) + value).hexdigest() == case["digest"]
    if case["leaf"] == "pos":
        assert ref.pos_leaf(value).hex() == case["digest"] and len(_b(case["prefix"])) == 26
    elif case["leaf"] == "fa2h-chunk":
        assert len(_b(case["prefix"])) == 64
    elif case["leaf"] == "fa2h-thread":
        assert ref.thread_leaf(_b(case["prefix"]) + value).hex() == case["digest"]
    else:
        assert ref.id_leaf(case["canonical_id"], case["dtype"], tuple(case["shape"]), value).hex() == case["digest"]


def test_configured_scheme_proves_digests_and_verifies_bound_roots() -> None:
    values = [V.expand(f"cfg/{i}", 10 + i) for i in range(5)]
    domain = ref.StepDomain(*(V.digest_of(k) for k in "pcgl"), 5)
    stream = ref.StreamDomain(V.digest_of("sl"), 1, 3, 16, 3, 1, 4, 1)
    scheme = ref.VllmV1({"host": domain, "m1": stream}, value_nbytes={"host": lambda i: 10 + i})
    digests = [scheme.leaf_layout("host", i).digest(v) for i, v in enumerate(values)]
    leaves = [scheme.tree_leaf("host", i, d) for i, d in enumerate(digests)]
    assert leaves == [ref.pos_leaf(v) for v in values]
    root = scheme.root("host", leaves)
    assert root == domain.bind(ref.fold(leaves))
    for i in range(5):
        assert scheme.verify_path("host", root, 5, i, leaves[i], scheme.path("host", leaves, i))
        assert not scheme.verify_path("host", root, 6, i, leaves[i], scheme.path("host", leaves, i))
    chunks = [V.expand(f"cfg/m1/{i}", 64) for i in range(3)]
    m1 = [scheme.tree_leaf("m1", i, scheme.leaf_layout("m1", i).digest(c)) for i, c in enumerate(chunks)]
    assert m1 == [stream.leaf(i, c) for i, c in enumerate(chunks)]
    with pytest.raises(InvalidArtifact):
        scheme.leaf_layout("host", 5)
    with pytest.raises(InvalidArtifact):
        ref.VllmV1({"host": domain}).leaf_layout("host", 0)


@pytest.mark.parametrize("case", VECTORS["trees"], ids=lambda c: f"n={c['leaf_count']}")
def test_every_vector_opening_verifies(case: dict) -> None:
    leaves, root = [_b(x) for x in case["leaves"]], _b(case["root"])
    for index, path in enumerate(case["paths"]):
        assert ref.TREE.verify_path(root, case["leaf_count"], index, leaves[index], _path(path))
        if case["leaf_count"] > 1:
            assert not ref.TREE.verify_path(root, case["leaf_count"], index, leaves[index - 1], _path(path))


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


@pytest.mark.parametrize("case", [c for c in VECTORS["openings"] if c["accept"]], ids=lambda c: c["kind"])
def test_lowering_trace_is_hash_by_hash(case: dict) -> None:
    domain = _domain(case)
    message = _b(case["leaf_message"])
    assert message.startswith(_b(case["leaf_prefix"])) and hashlib.sha256(message).hexdigest() == case["leaf"]
    digest = _b(case["leaf"])
    for step, sibling in zip(case["trace"], _path(case["path"]), strict=True):
        msg = _b(step["message"])
        assert (step["rule"] == "lift") == (sibling is None)
        assert digest in msg and (sibling is None or sibling in msg)
        digest = hashlib.sha256(msg).digest()
        assert digest.hex() == step["digest"]
    assert digest.hex() == case["tree_root"]
    preimage = _b(case["binding_preimage"])
    assert preimage == domain.binding_preimage(digest) and hashlib.sha256(preimage).hexdigest() == case["root"]
    assert ref.domain_digest(domain).hex() == case["domain_digest"]


def test_domain_digest_separates_domains_over_one_tree() -> None:
    a = ref.StepDomain(*(V.digest_of(k) for k in "pcgl"), 5)
    b = ref.StepDomain(*(V.digest_of(k) for k in "pcgl"), 5, {"launch_tag": 0, "chunk_words": 16, "src_mask": 0, "HB": 0, "M": 0, "NB": 0})
    c = ref.SemanticDomain(V.digest_of("p"), b"", V.digest_of("c"), V.digest_of("g"), 0, 5)
    assert len({ref.domain_digest(d) for d in (a, b, c)}) == 3


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
    tree_root = ref.TREE.root(leaves)
    program, ctx, geo, layout = (V.digest_of(k) for k in "pcgl")
    root = ref.step_root(program, ctx, geo, layout, 5, tree_root)
    path = ref.TREE.path(leaves, 2)
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
