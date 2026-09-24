"""C0 tests: positions derived from static context name exactly the stored leaves; a position opening authenticates the
same bytes as the production id-carrying opening; tampering is rejected.

    PYTHONPATH=. pytest -q tests/commit/test_semantic_layout.py
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from verity_vllm.commit import identity
from verity_vllm.observe.capture_v1 import CaptureReader
from verity_vllm.commit.merkle import MerkleTree, Value, build_commitments, verify_opening
from verity_vllm.harness.synthetic import build_toy_run

from verity_vllm.commit import fasttree as ft
from verity_vllm.commit.semantic_layout import POS_LEAF_TAG, Layout, RowSpec, RUContext, StaticContext, Template, pos_leaf, verify_position_opening
from verity_vllm.commit.stream_merkle import StreamingMerkle


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    run = tmp_path_factory.mktemp("c0") / "toy"
    build_toy_run(run, requests=2, prompt_len=5, gen=4)
    build_commitments(CaptureReader(run), recompute_leaves=True, write=True)
    return CaptureReader(run)


def _static(store):
    keys = store.ru_keys()
    dec = next(k for k in keys if k[1] == 1)
    template = Template.from_decode_index(store.ru(*dec).index.entries)
    profile = identity.parse(store.ru(*dec).index.entries[0].canonical_id).profile_id
    plen = {rid: len({identity.parse(e.canonical_id).t for e in store.ru(rid, 0).index.entries}) for rid, s in keys if s == 0}
    return template, profile, plen


def test_every_position_names_a_stored_leaf_and_back(store):
    template, profile, plen = _static(store)
    for rid, step in store.ru_keys():
        lay = Layout(template, RUContext(profile, rid, plen[rid], step))
        ents = {e.canonical_id: e for e in store.ru(rid, step).index.entries}
        derived = [lay.canonical_id(i) for i in range(lay.count)]
        assert lay.count == len(ents)
        assert set(derived) == set(ents)
        assert [lay.rank(c) for c in derived] == list(range(lay.count))
        for i, c in enumerate(derived):
            _t, row = lay.unrank(i)
            assert (ents[c].dtype, tuple(ents[c].shape)) == (row.dtype, row.shape)


def test_position_opening_authenticates_the_same_bytes_as_production(store):
    template, profile, plen = _static(store)
    trees = store.commitments["trees"]
    for rid, step in store.ru_keys():
        ru = store.ru(rid, step)
        lay = Layout(template, RUContext(profile, rid, plen[rid], step))
        by = {e.canonical_id: e for e in ru.index.entries}
        mm = memoryview(np.memmap(ru.dir / "values.bin", dtype=np.uint8, mode="r"))
        order = [by[lay.canonical_id(i)] for i in range(lay.count)]
        # runtime: packed values only
        mroot = ft.fold([pos_leaf(mm[e.offset : e.offset + e.nbytes]) for e in order])
        sc = StaticContext(hashlib.sha256(b"program").digest(), hashlib.sha256(b"query").digest(), template, lay.ctx, 0, lay.count)
        R = sc.root(mroot)
        for i in (0, lay.count // 2, lay.count - 1):
            b = StreamingMerkle(track=i)
            for e in order:
                b.add(POS_LEAF_TAG, mm[e.offset : e.offset + e.nbytes])
            assert b.finish() == mroot
            e = order[i]
            value = bytes(mm[e.offset : e.offset + e.nbytes])
            ok, cid = verify_position_opening(R, sc, i, value, b.path)
            assert ok and cid == e.canonical_id
            # production opening of the same id authenticates the same bytes
            prod = MerkleTree({c: bytes.fromhex(by[c].leaf) for c in by}, ru.tree_id)
            popen = prod.open(Value(cid, e.dtype, tuple(e.shape), value, "captured"))
            assert verify_opening(bytes.fromhex(trees[ru.tree_id]["root"]), trees[ru.tree_id]["leaf_count"], popen)
            assert popen.value == value
            # tamper: value, position, context
            if any(value):
                assert not verify_position_opening(R, sc, i, bytes(len(value)), b.path)[0]
            if lay.count > 1:
                assert not verify_position_opening(R, sc, (i + 1) % lay.count, value, b.path)[0]
            sc2 = StaticContext(sc.program_digest, sc.query_id, template, RUContext(profile, rid, plen[rid] + 1, step), 0, lay.count)
            assert not verify_position_opening(R, sc2, i, value, b.path)[0]


def test_layout_arithmetic():
    t = Template((RowSpec("0", "layers.0.a", 0, "bfloat16", (4,)), RowSpec("0", "layers.0.b", 0, "bfloat16", (4,))),
                 (RowSpec("lm_head", "lm_head.gemm", 0, "bfloat16", (8,)),))
    pre = Layout(t, RUContext("ec44fc6dd686", "r0", 3, 0))
    assert pre.count == 3 * 2 + 1
    assert pre.canonical_id(0).endswith("/t=0/ph=prefill/l=0/op=layers.0.a/port=0")
    assert pre.canonical_id(5).endswith("/t=2/ph=prefill/l=0/op=layers.0.b/port=0")
    assert pre.canonical_id(6).endswith("/t=2/ph=prefill/l=lm_head/op=lm_head.gemm/port=0")
    dec = Layout(t, RUContext("ec44fc6dd686", "r0", 3, 4))
    assert dec.count == 3 and dec.canonical_id(0).endswith("/t=6/ph=decode/l=0/op=layers.0.a/port=0")
    assert all(dec.rank(dec.canonical_id(i)) == i for i in range(dec.count))
