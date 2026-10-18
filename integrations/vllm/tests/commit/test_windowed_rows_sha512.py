"""The windowed worker under ``rows-sha512`` (``commit_scheme``, ``verity_vllm.commit.step_rows``) commits what the reference
does, on the real collector, window ring and kernels: each step's root is ``step_rows``'s over the step's packed stream (the same
stream ``commit_block_offline`` hashes under vllm-v1) with the step's recorded salt key, on a learning run (each window hashed to
its rows' h in place, the leaves from them once the step's length is known; re-hashed from host when it keeps bytes), a hit run
(hashed in place, pad-only windows absorbed after the data windows), with and without the levels, and under bounded staging.
Openings of a run that kept no levels come from a replay that took its keys, and verify against its 64-byte roots; a tampered row,
salt or path does not.  ``test_windowed_retain_levels`` holds vllm-v1's roots, unchanged.
"""
from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("NATIVE_COLLECT_BUILD"),
                                reason="needs a GPU and the JIT-built native_collect extension (NATIVE_COLLECT_BUILD)")

from tests.commit import test_windowed_retain_levels as WR  # noqa: E402
from verity_vllm.commit import step_rows as R  # noqa: E402
from verity_vllm.commit.committer import native_collect as NC  # noqa: E402
from verity_vllm.commit.committer import native_host as nh  # noqa: E402

SCHED = WR.SCHED


def committer(mp, **kw) -> NC.NativeCollectCommitter:
    com = WR.committer(mp, commit_scheme=R.NAME, **kw)
    assert com.rows and com.name.endswith("+" + R.NAME)
    return com


def rows_root(step: int, tensors: list[tuple[str, torch.Tensor]], ks: bytes) -> bytes:
    """step_rows's step root over the stream `commit_block_offline` packs (each tensor at the next ALIGN offset, zero-padded to a
    row), its layout counted in rows."""
    ref = nh.NativeHostCommitter(None, run_id=WR.RUN_ID, program_digest=WR.PD, geo_digest=WR.GEO, retain_values=False, gpu_tree=True)
    stream, metas, first_leaf = bytearray(), [], 0
    for i, (name, t) in enumerate(tensors):
        raw = t.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        stream.extend(b"\0" * (-len(stream) % nh.NativeHostCommitter.ALIGN))
        n_leaves = -(-len(raw) // R.ROW_BYTES)
        m = nh.TensorMeta(i, name, nh._dtype_name(t), tuple(t.shape), len(raw), R.ROW_BYTES, first_leaf, n_leaves)
        m.dev_off = m.stream_off = len(stream)
        stream.extend(raw)
        metas.append(m)
        first_leaf += n_leaves
    stream.extend(b"\0" * (-len(stream) % R.ROW_BYTES))
    rows = [bytes(stream[o:o + R.ROW_BYTES]) for o in range(0, len(stream), R.ROW_BYTES)]
    ctx = ref._ctx_digest(step)
    tree_root = R.levels(R.domain(WR.PD, ctx, WR.GEO, len(rows)), rows, ks, 8 * R.ROW_BYTES)[-1][0]
    return R.step_root(WR.PD, ctx, WR.GEO, ref.layout_digest(metas), len(rows), tree_root)


def reference(keys: dict[int, bytes]) -> list[bytes]:
    return [rows_root(i, WR.step_tensors(kv), keys[i]) for i, kv in enumerate(SCHED)]


def check_run(rc, keys: dict[int, bytes]) -> None:
    roots = [s.root for s in rc.steps]
    assert sorted(keys) == list(range(len(SCHED))) and len(set(keys.values())) == len(SCHED), "a fresh 64-byte key per step"
    assert roots == reference(keys)
    assert rc.run_root == R.run_root(WR.PD, WR.GEO, roots) and len(rc.run_root) == R.ROOT_BYTES
    assert rc.extra["commit_scheme"] == R.NAME


@pytest.mark.parametrize("retain_levels", [True, False], ids=["levels", "no-levels"])
def test_learning_and_hit_runs_commit_step_rows_roots(monkeypatch, retain_levels):
    com = committer(monkeypatch, retain_levels=retain_levels)
    try:
        learn = WR.run(com, SCHED)
        learn_keys, reup, hfirst = dict(com.row_keys), com._win_reupload_bytes, com._win_hfirst_steps
        hit = WR.run(com, SCHED)
        assert reup == 0 and hfirst == len(SCHED), "a learning step that keeps no bytes hashes its windows to h in place (no re-upload)"
        assert com._win_reupload_bytes == 0 and com._win_hfirst_steps == 0, "the second run follows the first run's plans (in place)"
        check_run(learn, learn_keys)
        check_run(hit, com.row_keys)
        assert hit.run_root != learn.run_root and not set(learn_keys.values()) & set(com.row_keys.values()), "each run's keys are fresh"
        assert bool(com._gpu_levels) is retain_levels
    finally:
        com.close()


def test_capped_runs_commit_step_rows_roots(monkeypatch):
    """VERITY_ROWS_CAP_BLOCKS's policy on the worker (each window capped while the ring has room; test_rows_cuda splits launches)."""
    com = committer(monkeypatch)
    com.rows_cap_blocks, com.rows_cap_free = 1, 0
    try:
        learn = WR.run(com, SCHED)
        learn_keys = dict(com.row_keys)
        hit = WR.run(com, SCHED)
        check_run(learn, learn_keys)
        check_run(hit, com.row_keys)
    finally:
        com.close()


def test_bounded_record_run_commits_step_rows_roots(monkeypatch):
    com = committer(monkeypatch, retain_levels=False, bounded=True)
    try:
        com.learn_only = True
        warm = WR.run(com, SCHED)
        assert len(warm.extra["bounded_staging"]["learned_only_steps"]) == len(SCHED)
        com.learn_only = False
        rec = WR.run(com, SCHED)
        bs = rec.extra["bounded_staging"]
        assert not bs["learned_only_steps"] and bs["plan_mismatches"] == 0 and bs["transient_peak_bytes"] == 0, bs
        check_run(rec, com.row_keys)
    finally:
        com.close()


def test_a_run_without_levels_opens_from_a_replay_that_took_its_keys(monkeypatch):
    com = committer(monkeypatch, retain_levels=False)
    rep = committer(monkeypatch, retain="host")
    try:
        rc = WR.run(com, SCHED)
        rep.replay_row_keys = dict(com.row_keys)
        rrc = WR.run(rep, SCHED)
        assert rep.replay_row_keys is None and rep.row_keys == com.row_keys
        assert [s.root for s in rrc.steps] == [s.root for s in rc.steps] and rrc.run_root == rc.run_root
        sample = [(s, i) for s, sc in enumerate(rc.steps) for i in sorted({0, 1, sc.n_leaves // 2, sc.n_leaves - 1})]
        ops = com.open(sample, values_from=rep, paths_from_replay=True)
        assert len(ops) == len(sample) and all(com.verify(rc, o) for o in ops)
        assert all(len(o.salt) == R.SALT_BYTES and o.salt == R.salt(com.row_keys[o.step], o.index) for o in ops)
        o = ops[0]
        flip = bytes([o.value[0] ^ 1]) + o.value[1:]
        assert not com.verify(rc, replace(o, value=flip))
        assert not com.verify(rc, replace(o, salt=bytes([o.salt[0] ^ 1]) + o.salt[1:]))
        assert not com.verify(rc, replace(o, path=o.path[:-1]))
        assert not com.verify(rc, replace(o, index=o.index + 1))
        rep.reset()
        rep.replay_row_keys = {}
        with pytest.raises(RuntimeError, match="no salt key for step 0"):
            rep._row_key(0)
        rep.replay_row_keys = None
        unkeyed = WR.run(rep, SCHED[:1])
        assert unkeyed.steps[0].root != rc.steps[0].root, "a replay without the run's keys commits other trees"
    finally:
        com.close()
        rep.close()


def test_what_rows_sha512_refuses(monkeypatch):
    for k in WR.ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VERITY_WINDOW_MB", "0")
    eng = SimpleNamespace(model=torch.nn.Module(), runner=WR._Runner())
    base = dict(run_id=WR.RUN_ID, program_digest=WR.PD, geo_digest=WR.GEO, arena_bytes=None, wrap=True, gather=True, native_hooks=True,
                auto_blocks=True, native_worker=True)
    with pytest.raises(ValueError, match="windowed worker"):
        NC.NativeCollectCommitter(eng, commit_scheme=R.NAME, **base)
    monkeypatch.setenv("VERITY_WINDOW_MB", WR.WINDOW_MB)
    with pytest.raises(ValueError, match="windowed worker"):
        NC.NativeCollectCommitter(eng, layout="chunk-leaf-v2", commit_scheme=R.NAME, **base)
    with pytest.raises(ValueError, match="unknown commit scheme"):
        NC.NativeCollectCommitter(eng, commit_scheme="rows-sha256", **base)
    com = committer(monkeypatch)
    try:
        with pytest.raises(NotImplementedError, match=R.NAME):
            com.open_range(0, 0, 1)
        with pytest.raises(ValueError, match=R.NAME):
            com.add_padding_steps([object()])
    finally:
        com.close()
