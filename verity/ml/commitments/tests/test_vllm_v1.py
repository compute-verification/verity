"""``vllm-v1``: the reference reproduces its conformance vectors, and its verifier accepts exactly the
vector openings and rejects the rejection vectors.  That vLLM's own code produces the same bytes is checked in
``integrations/vllm/tests/commit/test_production_vectors.py``.  ``vllm-v1-sha512``: the reference reproduces its vectors,
openings verify only under their own hash, and a configuration never mixes hashes."""

from __future__ import annotations

import hashlib

import pytest

from verity.ml.commitments import vllm_v1 as ref
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.leaves import LIFT_TAG, NODE_TAG, H
from verity.ml.commitments.scheme import CommitmentScheme
from verity.ml.commitments.vllm_v1 import vectors as V
from verity.ml.commitments.vllm_v1 import vectors_sha512 as V512
from verity.errors import InvalidArtifact

VECTORS = V.load()
VECTORS_SHA512 = V512.load()


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


# --- vllm-v1-sha512 ----------------------------------------------------------------------------------------------


def test_sha512_vectors_file_is_what_the_reference_generates() -> None:
    assert V512.main(["--check"]) == 0, "rerun `python -m verity.ml.commitments.vllm_v1.vectors_sha512`"


def test_sha512_leaves_and_tree_hash_vectors() -> None:
    for case in VECTORS_SHA512["hash"]:
        assert H(case["tag"], *[_b(p) for p in case["parts"]], hash="sha512").hex() == case["digest"]
    for case in VECTORS_SHA512["pos_leaves"]:
        value = _b(case["value"])
        assert ref.pos_leaf(value, "sha512").hex() == case["leaf"]
        assert ref.pos_leaf_layout(len(value), "sha512").digest(value).hex() == case["leaf"]
        assert case["leaf"] == hashlib.sha512(ref.POS_LEAF_TAG + len(value).to_bytes(8, "big") + value).hexdigest()
    for case in VECTORS_SHA512["chunk_leaves"]:
        assert ref.chunk_header(**case["fields"]).hex() == case["header"]
        assert ref.chunk_leaf(_b(case["header"]), _b(case["chunk"]), "sha512").hex() == case["leaf"]
    for case in VECTORS_SHA512["thread_leaves"]:
        assert ref.thread_leaf(_b(case["header"]) + _b(case["words"]), "sha512").hex() == case["leaf"]


@pytest.mark.parametrize("case", VECTORS_SHA512["trees"], ids=lambda c: f"n{c['leaf_count']}")
def test_sha512_tree_vectors(case: dict) -> None:
    leaves = [_b(x) for x in case["leaves"]]
    root = _b(case["root"])
    assert ref.TREE_SHA512.root(leaves) == root and len(root) == 64
    for i in range(case["leaf_count"]):
        path = tuple(_path(case["paths"][i]))
        assert ref.TREE_SHA512.path(leaves, i) == path
        assert ref.TREE_SHA512.verify_path(root, case["leaf_count"], i, leaves[i], path)
        assert not ref.TREE.verify_path(root, case["leaf_count"], i, leaves[i], path)
    if case["leaf_count"] == 3:
        a, b, c = leaves
        assert root == H(NODE_TAG, H(NODE_TAG, a, b, hash="sha512"), H(LIFT_TAG, c, hash="sha512"), hash="sha512")


def test_sha512_root_vectors() -> None:
    fns = {"semantic": ref.semantic_root, "step": ref.step_root, "run": ref.run_root, "weights": ref.weights_root,
           "stream": ref.stream_root, "thread": ref.thread_root}
    for case in VECTORS_SHA512["roots"]:
        i = case["inputs"]
        args = {
            "semantic": lambda: (_b(i["program_digest"]), _b(i["query_id"]), _b(i["template_digest"]), _b(i["ctx_digest"]), i["epoch"], i["n"],
                                 _b(i["tree_root"])),
            "step": lambda: (_b(i["program_digest"]), _b(i["ctx_digest"]), _b(i["geo_digest"]), _b(i["layout_digest"]), i["n"], _b(i["tree_root"])),
            "run": lambda: (_b(i["program_digest"]), _b(i["geo_digest"]), [_b(s) for s in i["step_roots"]]),
            "weights": lambda: (_b(i["geo_digest"]), _b(i["names_digest"]), [_b(t) for t in i["tensor_roots"]]),
            "stream": lambda: (_b(i["layout_digest"]), _b(i["ctx_digest"]), _b(i["tree_root"])),
            "thread": lambda: (_b(i["geo_digest"]), _b(i["ctx_digest"]), _b(i["tree_root"])),
        }[case["kind"]]()
        assert fns[case["kind"]](*args, hash="sha512").hex() == case["root"]
        if case["kind"] == "stream":
            assert ref.stream_ctx_digest(launch_tag=i["launch_tag"], src_mask=i["src_mask"], chunk_words=i["chunk_words"],
                                         leaf_count=i["leaf_count"], hash="sha512").hex() == i["ctx_digest"]
        if case["kind"] == "thread":
            assert ref.thread_ctx_digest(launch_tag=i["launch_tag"], src_mask=i["src_mask"], hash="sha512").hex() == i["ctx_digest"]


def test_sha512_openings_accept_exactly_the_honest_values() -> None:
    assert [(c["kind"], c["accept"]) for c in VECTORS_SHA512["openings"]] == [(k, a) for k in ("semantic", "step", "stream") for a in (True, False)]


def test_sha512_configuration_names_its_hash_and_never_mixes() -> None:
    d = hashlib.sha256(b"cfg").digest()
    s512 = ref.StepDomain(d, d, d, d, 3, hash="sha512")
    scheme = ref.VllmV1({"v": s512}, value_nbytes={"v": lambda _i: 8})
    assert isinstance(scheme, CommitmentScheme)
    assert (scheme.name, scheme.hashes) == (ref.NAME_SHA512, ("sha512",)) == (VECTORS_SHA512["scheme"], ("sha512",))
    leaves = [scheme.tree_leaf("v", i, scheme.leaf_layout("v", i).digest(bytes(8))) for i in range(3)]
    root = scheme.root("v", leaves)
    assert scheme.verify_path("v", root, 3, 1, leaves[1], scheme.path("v", leaves, 1))
    with pytest.raises(InvalidArtifact):
        scheme.tree_leaf("v", 0, bytes(32))
    with pytest.raises(InvalidArtifact):
        ref.VllmV1({"a": s512, "b": ref.StepDomain(d, d, d, d, 3)})
    assert ref.domain_digest(s512) != ref.domain_digest(ref.StepDomain(d, d, d, d, 3))
    assert ref.VllmV1({}).name == ref.NAME


def test_sha512_domain_digests_are_sha512() -> None:
    d = hashlib.sha256(b"dd").digest()
    s512 = ref.StepDomain(d, d, d, d, 3, hash="sha512")
    doc = {"kind": "step", "leaf_count": 3, "program_digest": d.hex(), "ctx_digest": d.hex(), "geo_digest": d.hex(), "layout_digest": d.hex(),
           "chunk": None, "hash": "sha512"}
    assert ref.domain_digest(s512) == identity_digest_sha512(ref.DOMAIN_DIGEST_TAG, doc)
    assert len(ref.domain_digest(s512)) == 64 and len(ref.domain_digest(ref.StepDomain(d, d, d, d, 3))) == 32
    assert all(len(_b(c["domain_digest"])) == 64 for c in VECTORS_SHA512["openings"])
