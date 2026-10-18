"""The Commit's root of record (`commit.record`): core's frame-v3-sha512 / hm96-sha512 row reference on the host hasher's in-process and
forked paths, openings of one leaf and of a range against the run root, each named forgery refused, and the committer's retained leaves
(host per-tensor, GPU-tree block, padding) read as the bytes `open` returns."""
from __future__ import annotations

import dataclasses
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
    steps = [(s, hashlib.sha256(f"layout{s}".encode()).digest(), n, lambda s=s: rows[s]) for s, n in shape.items()]
    return R.build(PROGRAM, GEO, steps, hasher=hasher or R.HostRows(1), chunk=chunk, salt_key=rng.randbytes(32)), rows


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
    monkeypatch.setattr(R, "TASK", 4)
    one, _ = record({0: 10, 1: 4}, seed=7)
    many, _ = record({0: 10, 1: 4}, seed=7, hasher=R.HostRows(3))
    assert many.root == one.root
    assert all((x == y).all() for a, b in zip(many.steps, one.steps) for x, y in zip(a.levels, b.levels, strict=True))


def test_batches_and_salt_ranges_are_one_keystream(monkeypatch):
    one, _ = record({0: 9, 1: 2, 2: 5}, seed=4)
    monkeypatch.setattr(R, "BATCH_BYTES", 1)
    each, _ = record({0: 9, 1: 2, 2: 5}, seed=4)
    assert each.root == one.root
    st = one.steps[0]
    assert st.salt(3, 7) == st.salt(0, 9)[3 * R.SALT_BYTES:7 * R.SALT_BYTES] and st.salt(0, 1) != one.steps[1].salt(0, 1)


def test_salts_are_fresh_by_default():
    rows = np.zeros((2, 256), dtype=np.uint8)
    a = R.build(PROGRAM, GEO, [(0, b"L", 2, lambda: rows)], hasher=R.HostRows(1), chunk=256)
    b = R.build(PROGRAM, GEO, [(0, b"L", 2, lambda: rows)], hasher=R.HostRows(1), chunk=256)
    assert a.root != b.root and a.steps[0].salt(0, 2) != b.steps[0].salt(0, 2)


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
    key = rec.steps[0].key
    assert key.hex() not in str(doc) and doc["root"] == rec.root.hex() and doc["scheme"] == "frame-v3-sha512/hm96-sha512"
    back = R.from_doc(doc, lambda s: rec.steps[s].levels, key)
    assert back.root == rec.root and back.steps[1].salt(0, 2) == rec.steps[1].salt(0, 2) and back.steps[1].path(1) == rec.steps[1].path(1)
    public = R.from_doc(doc)
    with pytest.raises(ValueError, match="no salts"):
        public.steps[0].salt(0, 1)
    bad = dict(doc, steps=[[s[0], "00" * 64, *s[2:]] for s in doc["steps"]])
    with pytest.raises(ValueError, match="record root"):
        R.from_doc(bad)


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


# --- the committer of record ---------------------------------------------------------------------------------------------

HOST = {0: [("a", bytes(range(200))), ("b", b"\x05" * 70)], 1: [("c", b"\x09" * 64)], 2: [("d", bytes(i % 7 for i in range(333)))]}
PACKED = {0: [("model.embed_tokens", bytes(range(256)) * 5), ("model.layers.0.self_attn.qkv_proj", b"\x07" * 900)],
          1: [("runner.sampler", b"\x01\x02\x03")], 2: [("model.layers.0.mlp.down_proj", bytes(i % 251 for i in range(3000)))]}


def _runs(packed: bool):
    if packed:
        pytest.importorskip("torch")
        from tests.check.opened import commit_steps
        return commit_steps(PACKED, of_record=True), commit_steps(PACKED)
    from tests.check.opened import commit_host_steps
    return commit_host_steps(HOST, of_record=True), commit_host_steps(HOST)


@pytest.mark.parametrize("packed", [False, True])
def test_the_run_root_is_the_records_and_the_committer_root_a_diagnostic(packed):
    (com, rc), (legacy, lrc) = _runs(packed)
    assert len(rc.run_root) == 64 and rc.run_root == rc.record.root and rc.extra["record"]["root"] == rc.run_root.hex()
    assert rc.extra["committer_root"] == lrc.run_root.hex() and len(lrc.run_root) == 32 and lrc.record is None
    assert [s.root for s in rc.steps] == [s.root for s in lrc.steps] and "record" in rc.spans["throughput"]["stages"]


@pytest.mark.parametrize("packed", [False, True])
def test_every_leaf_opens_against_the_root_of_record(packed):
    (com, rc), (legacy, lrc) = _runs(packed)
    pos = [(s.step, i) for s in rc.steps for i in range(s.n_leaves)]
    ops, lops = com.open(pos), legacy.open(pos)
    assert [o.value for o in ops] == [o.value for o in lops] and [o.identity for o in ops] == [o.identity for o in lops]
    assert all(len(o.salt) == R.SALT_BYTES and all(len(p) == 64 for p in o.path) for o in ops)
    assert all(com.verify(rc, o) for o in ops) and all(legacy.verify(lrc, o) for o in lops)
    assert not any(com.verify(rc, o) for o in lops) and not any(legacy.verify(lrc, o) for o in ops)


@pytest.mark.parametrize("packed", [False, True])
def test_ranges_open_against_the_root_of_record(packed):
    (com, rc), _ = _runs(packed)
    roots: dict = {}
    checked = 0
    for s in rc.steps:
        for lo in range(s.n_leaves):
            for hi in range(lo + 1, s.n_leaves + 1):
                ro = com.open_range(s.step, lo, hi)
                if ro is None:                                       # the per-tensor path opens a range inside one tensor
                    continue
                checked += 1
                assert com.verify_range(rc, ro) and com.verify_range(rc, ro, roots)
                assert not com.verify_range(rc, dataclasses.replace(ro, value=bytes([ro.value[0] ^ 1]) + ro.value[1:]), roots)
    assert checked > 20 and com._run_root(rc) == rc.run_root


def test_each_forgery_against_the_committer_of_record_is_refused():
    (com, rc), _ = _runs(False)
    o = com.open([(2, 3)])[0]
    assert com.verify(rc, o)
    flip = lambda b: bytes([b[0] ^ 1]) + b[1:]  # noqa: E731
    bad = {"a flipped value byte": (rc, dataclasses.replace(o, value=flip(o.value))),
           "a flipped salt byte": (rc, dataclasses.replace(o, salt=flip(o.salt))),
           "a flipped sibling": (rc, dataclasses.replace(o, path=(flip(o.path[0]), *o.path[1:]))),
           "another identity": (rc, dataclasses.replace(o, identity={**o.identity, "name": "x"})),
           "a value padded with zeros": (rc, dataclasses.replace(o, value=o.value + b"\0")),
           "a claimed run root": (dataclasses.replace(rc, run_root=bytes(64)), o),
           "a 64-byte root with no record": (dataclasses.replace(rc, record=None), o),
           "a record of another run": (dataclasses.replace(rc, record=_runs(False)[0][1].record), o),
           "a run with another step's leaf count": (dataclasses.replace(rc, steps=[*rc.steps[:2], dataclasses.replace(rc.steps[2], n_leaves=99)]), o)}
    assert [k for k, (run, op) in bad.items() if com.verify(run, op)] == []


def test_the_opened_reader_reads_the_record_and_refuses_a_flipped_retained_byte():
    pytest.importorskip("torch")
    from tests.check.opened import commit_steps, retained_block
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.commit import opened as OV
    com, rc = commit_steps(PACKED, of_record=True)
    rd = OC.OpenedReader(com, rc)
    assert rd(0, "model.layers.0.self_attn.qkv_proj", 10, 20) == b"\x07" * 10
    assert rd.record()["run_root"] == rc.run_root.hex()
    retained_block(com, 0)[1300] ^= 1                                # inside qkv_proj (offset 1280)
    with pytest.raises(OV.OpeningNotVerified, match="qkv_proj"):
        OC.OpenedReader(com, rc)(0, "model.layers.0.self_attn.qkv_proj", 0, 100)


def test_a_hiding_committer_leaf_scheme_under_a_record_is_refused():
    from verity_vllm.commit import hiding
    from verity_vllm.commit.committer.native_host import NativeHostCommitter
    com = NativeHostCommitter(None, chunk=64, hash_threads=1, leaf_scheme=hiding.NAME)
    com.of_record = True
    with pytest.raises(ValueError, match="hm96-sha256"):
        com.commit_offline([[("a", b"\x01" * 100)]])
