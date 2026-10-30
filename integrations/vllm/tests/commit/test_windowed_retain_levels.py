"""The windowed worker (`native_collect_v2b`, `--window-mb`) with `retain_levels=False` commits the same bytes and roots as with the
levels kept, on the real collector, window ring and kernels.

A run that keeps no tree level leaves each window's subtree on the device and drops the per-window host copy and host sync, so a
slot goes back to the ring with only an event on the worker's side stream between the hash kernels that read it and the gather that
refills it. The windows here are 8 KiB and the ring holds four, so every step of ~56 KiB refills each slot while the worker still
has the previous window's kernels queued. Every step's root is checked against `commit_block_offline` (the CPU reference over the
same tensors packed into one block), on a learning run (re-hashed from host after the step), a hit run (hashed in place) and bounded
staging (warm-up learn-only, then a record run with no host copy). Openings of such a run come from a replay that kept its levels
(`paths_from_replay=True`) and verify against this run's roots; without them `open` refuses by name.
"""
from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("NATIVE_COLLECT_BUILD"),
                                reason="needs a GPU and the JIT-built native_collect extension (NATIVE_COLLECT_BUILD)")

from verity_vllm.commit.committer import native_collect as NC  # noqa: E402
from verity_vllm.commit.committer import native_host as nh  # noqa: E402

RUN_ID = "retain-levels"
PD, GEO = hashlib.sha256(b"program").digest(), hashlib.sha256(b"geo").digest()
LAYERS, H, BN, VOCAB = 3, 1000, 16, 1003
WINDOW_MB = str(8 / 1024)                                       # 8 KiB = 32 chunks: j = 5
ENV = ("VERITY_RETAIN", "VERITY_WINDOW_MB", "VERITY_WINDOW_SLOTS", "VERITY_STAGING_BOUNDED", "VERITY_LEARN_HOST_BUDGET_MB",
       "VERITY_RETAIN_EXCLUDE", "VERITY_FAULT", "VERITY_FOOTPRINT", "VERITY_HI_PRIO", "VERITY_CONSUMER_DELAY_MS", "VERITY_SKIP",
       "VERITY_INLINE_D2H", "VERITY_ABL")


class _Runner:
    def __init__(self) -> None:
        self.committer = None
        self.tensors: list[tuple[str, object]] = []
        self.input_batch = None

    def execute_model(self, scheduler_output, *a, **kw):
        com = self.committer
        for name, t in self.tensors:
            com._acquire(name, t, com._cur)


class _Sched:
    """A decode step: one new token for each request at its KV length."""

    def __init__(self, kv: tuple[int, ...]) -> None:
        self.req_ids = [f"r{i}" for i in range(len(kv))]
        self.num_scheduled_tokens = dict.fromkeys(self.req_ids, 1)
        self.total_num_scheduled_tokens = len(kv)
        self.scheduled_new_reqs = []
        self.scheduled_cached_reqs = SimpleNamespace(req_ids=self.req_ids, num_computed_tokens=list(kv))


def committer(mp, *, retain: str = "none", retain_levels: bool = True, bounded: bool = False,
              commit_scheme: str = "vllm-v1") -> NC.NativeCollectCommitter:
    for k in ENV:
        mp.delenv(k, raising=False)
    mp.setenv("VERITY_RETAIN", retain)
    mp.setenv("VERITY_WINDOW_MB", WINDOW_MB)
    mp.setenv("VERITY_WINDOW_SLOTS", "4")
    mp.setenv("VERITY_STAGING_BOUNDED", "1" if bounded else "0")
    if bounded:
        mp.setenv("VERITY_LEARN_HOST_BUDGET_MB", str(16 / 1024))   # under a step: a learn-only pass discards every learning step
    eng = SimpleNamespace(model=torch.nn.Module(), runner=_Runner())
    com = NC.NativeCollectCommitter(eng, run_id=RUN_ID, program_digest=PD, geo_digest=GEO, arena_bytes=None, wrap=True, gather=True,
                                    native_hooks=True, auto_blocks=True, native_worker=True, retain_levels=retain_levels,
                                    commit_scheme=commit_scheme)
    com.enable_windows(None)
    eng.runner.committer = com
    return com


def step_tensors(kv: tuple[int, ...]) -> list[tuple[str, torch.Tensor]]:
    """One decode step's tensors, odd-sized so window boundaries fall inside them; the hidden plane is sized by every request's KV
    blocks, so steps differ in length. The values are a function of the geometry."""
    ntok, seed = len(kv), sum(kv)
    shapes = [("model.embed_tokens/in", torch.int32, (ntok,))]
    for layer in range(LAYERS):
        shapes += [(f"model.layers.{layer}.self_attn.qkv_proj", torch.bfloat16, (ntok, 3 * H)),
                   (f"fa2.m1/L{layer}/acc", torch.float32, (sum(-(-k // BN) for k in kv), 4)),
                   (f"model.layers.{layer}.mlp.down_proj", torch.bfloat16, (ntok, H))]
    shapes += [("logits", torch.float32, (ntok, VOCAB)), ("runner.sampler/sampled_token_ids", torch.int32, (ntok,))]
    out = []
    for i, (name, dtype, shape) in enumerate(shapes):
        n = 1
        for d in shape:
            n *= d
        v = (torch.arange(n, dtype=torch.int64) * 7 + seed * 13 + i * 31) % 251 - 125
        out.append((name, v.to(dtype).reshape(shape).cuda()))
    # a pitched view 1000 B into each row of the first qkv output (RoPE's q): its segments straddle windows at a 4 B source offset
    qkv = out[1][1]
    out.append(("model.layers.0.self_attn.rotary_emb/q", qkv[:, H // 2:H // 2 + H]))
    return out


def schedule(kv0: int, steps: int, batch: int = 2) -> list[tuple[int, ...]]:
    return [tuple(kv0 + 37 * r + 9 * s for r in range(batch)) for s in range(steps)]


def run(com: NC.NativeCollectCommitter, sched: list[tuple[int, ...]]):
    """One commitment run (reset, install, the steps, finalize, uninstall); its blocks stay held until the next run's reset."""
    com.reset()
    com.install()
    for kv in sched:
        so = _Sched(kv)
        com.runner.tensors = step_tensors(kv)
        com.runner.input_batch = SimpleNamespace(req_ids=so.req_ids, num_reqs=len(kv), num_scheduled_tokens=None)
        com.runner.execute_model(so)
        com._end_step()
    rc = com.finalize()
    com.uninstall()
    assert com.errors == [], com.errors[:3]
    return rc


def one_block_root(step: int, tensors: list[tuple[str, torch.Tensor]]) -> bytes:
    ref = nh.NativeHostCommitter(None, run_id=RUN_ID, program_digest=PD, geo_digest=GEO, retain_values=False, gpu_tree=True)
    stream, metas, first_leaf = bytearray(), [], 0
    for i, (name, t) in enumerate(tensors):
        raw = t.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        stream.extend(b"\0" * (-len(stream) % nh.NativeHostCommitter.ALIGN))
        n_leaves = -(-len(raw) // nh.GPU_CHUNK_BYTES)
        m = nh.TensorMeta(i, name, nh._dtype_name(t), tuple(t.shape), len(raw), nh.GPU_CHUNK_BYTES, first_leaf, n_leaves)
        m.dev_off = len(stream)
        stream.extend(raw)
        metas.append(m)
        first_leaf += n_leaves
    return ref.commit_block_offline(step, stream, metas).root


SCHED = schedule(40, 6)


@pytest.fixture(scope="module")
def reference() -> list[bytes]:
    return [one_block_root(i, step_tensors(kv)) for i, kv in enumerate(SCHED)]


@pytest.mark.parametrize("retain_levels", [True, False], ids=["levels", "no-levels"])
def test_learning_and_hit_runs_commit_the_one_block_roots(monkeypatch, reference, retain_levels):
    """The first run learns every plan (each window copied to host and re-hashed with the step's length), the second hits them (each
    window hashed in place as it fills): both commit every step's one-block root, with or without the levels."""
    com = committer(monkeypatch, retain_levels=retain_levels)
    try:
        learn = run(com, SCHED)
        reup = com._win_reupload_bytes
        hit = run(com, SCHED)
        assert reup > 0 and com._win_reupload_bytes == 0, "the second run follows the first run's plans (in place, no re-upload)"
        assert [s.root for s in learn.steps] == reference
        assert [s.root for s in hit.steps] == reference
        assert hit.run_root == learn.run_root
        assert bool(com._gpu_levels) is retain_levels
    finally:
        com.close()


@pytest.mark.parametrize("fast", [True, False], ids=["native-end", "python-end"])
def test_the_step_end_paths_commit_the_same_roots_and_row_order(monkeypatch, reference, fast):
    """The serving thread's step end in one native call with the row order expanded by the worker, or as before: the same roots and
    the same `step_rows`."""
    com = committer(monkeypatch)
    com.native_end = com.rows_off_thread = fast
    try:
        rc = run(com, SCHED)
        assert [s.root for s in rc.steps] == reference
        assert list(com.step_rows.values()) == [[("r0", 0, 1), ("r1", 1, 2)]] * len(SCHED)
    finally:
        com.close()


def test_bounded_record_run_without_levels_commits_the_one_block_roots(monkeypatch, reference):
    """Bounded staging: the learn-only warm-up records the plans, then the record run hashes every window in place with no host copy
    and no host sync per window."""
    com = committer(monkeypatch, retain_levels=False, bounded=True)
    try:
        com.learn_only = True
        warm = run(com, SCHED)
        assert len(warm.extra["bounded_staging"]["learned_only_steps"]) == len(SCHED)
        com.learn_only = False
        rec = run(com, SCHED)
        bs = rec.extra["bounded_staging"]
        assert not bs["learned_only_steps"] and bs["plan_mismatches"] == 0 and bs["transient_peak_bytes"] == 0, bs
        assert [s.root for s in rec.steps] == reference and not com._gpu_levels
    finally:
        com.close()


def test_a_run_without_levels_opens_from_a_replay_that_kept_them(monkeypatch, reference):
    """Openings of a run that kept no levels take value and path from a replay (retain=host, levels kept) and verify against the
    run's own roots; with the replay's value and no replay path, `open` refuses by name."""
    rep = committer(monkeypatch, retain="host")
    com = committer(monkeypatch, retain_levels=False)
    try:
        rc = run(com, SCHED)
        rrc = run(rep, SCHED)
        assert [s.root for s in rc.steps] == [s.root for s in rrc.steps] == reference
        sample = [(s, i) for s, sc in enumerate(rc.steps) for i in sorted({0, 1, sc.n_leaves // 2, sc.n_leaves - 1})]
        ops = com.open(sample, values_from=rep, paths_from_replay=True)
        assert len(ops) == len(sample) and all(com.verify(rc, o) for o in ops)
        with pytest.raises(RuntimeError, match="kept no tree levels"):
            com.open(sample[:1], values_from=rep)
    finally:
        com.close()
        rep.close()
