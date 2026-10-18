"""The Commit's root of record (`commit.record`): core's frame-v3-sha512 / hm96-sha512 row reference on the host hasher's in-process and
forked paths, openings of one leaf and of a range against the run root, each named forgery refused, and the committer's retained leaves
(host per-tensor, GPU-tree block, padding) read as the bytes `open` returns."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from verity.ml.commitments import hm96
from verity.ml.commitments.frame_v3 import FrameV3, Port, RowLeaf
from verity.ml.commitments.rowleaf import ROLE_X, row_leaf, sha512_row_layout
from verity_vllm.commit import record as R

PROGRAM, GEO = hashlib.sha256(b"program").digest(), hashlib.sha256(b"geo").digest()


def rows_of(rng: random.Random, n: int, chunk: int) -> np.ndarray:
    out = np.zeros((n, R.row_bytes(chunk)), dtype=np.uint8)
    for i in range(n):
        k = chunk if i < n - 1 else rng.randint(1, chunk)
        out[i, :k] = np.frombuffer(rng.randbytes(k), dtype=np.uint8)
    return out


def record(shape: dict[int, int], chunk: int = 256, seed: int = 1, hasher=None) -> tuple[R.RunRecord, dict[int, np.ndarray]]:
    rng = random.Random(seed)
    rows = {s: rows_of(rng, n, chunk) for s, n in shape.items()}
    steps = [(s, hashlib.sha256(f"layout{s}".encode()).digest(), n, rows[s]) for s, n in shape.items()]
    return R.build(PROGRAM, GEO, steps, hasher=hasher or R.HostRows(1), chunk=chunk, salts=rng.randbytes), rows


def core_root(st: R.StepRecord, rows: np.ndarray) -> bytes:
    inst, key = hm96.SHA512, hm96.SHA512.default_key
    words = st.row // 2
    lay = sha512_row_layout(ROLE_X, 16, words)
    fv = FrameV3({"r": Port(st.domain, RowLeaf("sha512", ROLE_X, 16, words))})
    leaves = [row_leaf(st.domain, i, inst.tree_leaf(key, inst.commit_string(key, lay.digest(rows[i].tobytes()), st.salt(i, i + 1))), R.ROW_SCHEMA)
              for i in range(st.leaves)]
    return fv.root("r", leaves)


def test_row_bytes_are_whole_sha512_blocks():
    assert [R.row_bytes(c) for c in (1, 64, 128, 129, 256, 4096)] == [128, 128, 128, 256, 256, 4096]


@pytest.mark.parametrize("shape,chunk", [({0: 1}, 256), ({0: 5, 1: 3, 2: 8}, 256), ({0: 7, 1: 1}, 64), ({0: 3}, 300)])
def test_core_reference_on_the_in_process_path(shape, chunk):
    rec, rows = record(shape, chunk)
    for st in rec.steps:
        assert st.root == core_root(st, rows[st.step])
        assert st.binding == R.step_binding(PROGRAM, GEO, st.step, st.layout, st.leaves, st.row)
    assert rec.root == R.run_root(PROGRAM, GEO, [(st.root, st.leaves) for st in rec.steps]) and len(rec.root) == 64


def test_the_forked_path_is_the_in_process_path(monkeypatch):
    monkeypatch.setattr(R, "POOL_MIN", 1)
    monkeypatch.setattr(R, "TASK", 3)
    one, _ = record({0: 10, 1: 4}, seed=7)
    many, _ = record({0: 10, 1: 4}, seed=7, hasher=R.HostRows(3))
    assert many.root == one.root and [s.levels for s in many.steps] == [s.levels for s in one.steps]


def test_salts_are_fresh_by_default():
    rows = np.zeros((2, 256), dtype=np.uint8)
    a = R.build(PROGRAM, GEO, [(0, b"L", 2, rows)], hasher=R.HostRows(1), chunk=256)
    b = R.build(PROGRAM, GEO, [(0, b"L", 2, rows)], hasher=R.HostRows(1), chunk=256)
    assert a.root != b.root and a.steps[0].salts != b.steps[0].salts


def test_one_leaf_and_every_range_open_against_the_run_root():
    rec, rows = record({0: 6, 1: 1, 2: 5})
    for st in rec.steps:
        for i in range(st.leaves):
            assert rec.verify_leaves(st.step, i, [rows[st.step][i].tobytes()], st.salt(i, i + 1), R.path_pairs(i, st.path(i)), rec.root)
        for lo in range(st.leaves):
            for hi in range(lo + 1, st.leaves + 1):
                vals = [rows[st.step][i].tobytes() for i in range(lo, hi)]
                assert rec.verify_leaves(st.step, lo, vals, st.salt(lo, hi), st.range_path(lo, hi), rec.root)


def test_a_value_shorter_than_its_row_is_the_zero_padded_row():
    rec, rows = record({0: 3}, chunk=200)
    st = rec.steps[0]
    last = rows[0][2].tobytes().rstrip(b"\0")
    assert rec.verify_leaves(0, 2, [last], st.salt(2, 3), R.path_pairs(2, st.path(2)), rec.root)


def _forgeries(rec: R.RunRecord, rows):
    st, i = rec.steps[0], 2
    val, salt, path = rows[0][i].tobytes(), st.salt(i, i + 1), R.path_pairs(i, st.path(i))
    flip = lambda b, k=0: b[:k] + bytes([b[k] ^ 1]) + b[k + 1:]  # noqa: E731
    yield "a flipped value byte", (0, i, [flip(val, 5)], salt, path, rec.root), {}
    yield "a flipped salt byte", (0, i, [val], flip(salt, 100), path, rec.root), {}
    yield "a flipped sibling", (0, i, [val], salt, tuple((flip(a) if a else a, b) for a, b in path), rec.root), {}
    yield "another leaf's position", (0, i + 1, [val], salt, R.path_pairs(i + 1, st.path(i + 1)), rec.root), {}
    yield "another step", (1, i, [val], salt, path, rec.root), {}
    yield "a run root that is not the record's", (0, i, [val], salt, path, flip(rec.root)), {}
    yield "the step's root over another layout", (0, i, [val], salt, path, rec.root), {"layout": b"other"}
    yield "a value longer than its row", (0, i, [val + b"\0"], salt, path, rec.root), {}
    yield "a short salt", (0, i, [val], salt[:-1], path, rec.root), {}
    yield "a truncated path", (0, i, [val], salt, path[:-1], rec.root), {}
    yield "no value", (0, i, [], b"", path, rec.root), {}


@pytest.mark.parametrize("case", range(11))
def test_each_named_forgery_is_refused(case):
    rec, rows = record({0: 6, 1: 6})
    name, args, kw = list(_forgeries(rec, rows))[case]
    assert not rec.verify_leaves(*args, **kw), name


def test_a_record_reloads_from_its_doc_and_refuses_another_root():
    rec, _ = record({0: 4, 1: 2})
    doc = rec.doc()
    assert rec.steps[0].salt(0, 1)[:16].hex() not in str(doc) and doc["root"] == rec.root.hex() and doc["scheme"] == "frame-v3-sha512/hm96-sha512"
    back = R.from_doc(doc, lambda s: rec.steps[s].levels, lambda s: rec.steps[s].salts)
    assert back.root == rec.root
    bad = dict(doc, steps=[[s[0], "00" * 64, *s[2:]] for s in doc["steps"]])
    with pytest.raises(ValueError, match="record root"):
        R.from_doc(bad, lambda s: rec.steps[s].levels, lambda s: rec.steps[s].salts)


# --- the committer's leaves ----------------------------------------------------------------------------------------------

def _opened_rows(com, rc) -> dict[int, np.ndarray]:
    out = {}
    for sc in rc.steps:
        ops = com.open([(sc.step, i) for i in range(sc.n_leaves)])
        rows = np.zeros((sc.n_leaves, R.row_bytes(com.chunk)), dtype=np.uint8)
        for o in ops:
            rows[o.index, :len(o.value)] = np.frombuffer(o.value, dtype=np.uint8)
        out[sc.step] = rows
    return out


def test_the_host_committers_leaves_are_the_bytes_open_returns():
    from tests.check.opened import commit_host_steps
    com, rc = commit_host_steps({0: [("a", bytes(range(200))), ("b", b"\x05" * 70)], 1: [("c", b"\x09" * 64)]})
    want = _opened_rows(com, rc)
    for sc in rc.steps:
        assert (R.step_rows(com, sc.step, sc.n_leaves) == want[sc.step]).all()
    rec = R.of_committer(com, rc.steps)
    assert [s.leaves for s in rec.steps] == [sc.n_leaves for sc in rc.steps] and [s.layout for s in rec.steps] == [sc.layout_digest for sc in rc.steps]


def test_the_gpu_tree_blocks_leaves_are_the_bytes_open_returns():
    pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    com, rc = commit_steps({0: [("model.embed_tokens", bytes(range(256)) * 5), ("x", b"\x07" * 900)], 1: [("runner.sampler", b"\x01\x02\x03")]})
    want = _opened_rows(com, rc)
    for sc in rc.steps:
        assert (R.step_rows(com, sc.step, sc.n_leaves) == want[sc.step]).all()


def test_a_committer_that_dropped_bytes_is_refused_by_name():
    pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    com, rc = commit_steps({0: [("x", b"\x07" * 900)]}, retain="none")
    with pytest.raises(R.NotRetained, match="retain=none"):
        R.of_committer(com, rc.steps)
