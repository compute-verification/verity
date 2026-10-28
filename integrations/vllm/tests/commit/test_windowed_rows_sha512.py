"""The windowed worker (``native_collect_v2b``, ``--window-mb``) commits what the reference does under ``rows-sha512``
(``verity_vllm.commit.step_rows``, the collector's one scheme), on the real collector, window ring and kernels: each step's root is
``step_rows``'s over the step's packed stream (each tensor at the next ALIGN offset, zero-padded to a row) with the step's recorded
salt key, on a learning run (each window hashed to its rows' h in place, the leaves from them once the step's length is known;
re-hashed from host when it keeps bytes), a hit run (hashed in place, pad-only windows absorbed after the data windows), with and
without the levels, through either step end, and under bounded staging.

The windows here are 8 KiB and the ring holds four, so every step of ~56 KiB refills each slot while the worker still has the previous
window's kernels queued; a run that keeps no tree level leaves each window's subtree on the device and drops the per-window host copy
and host sync, so a slot goes back to the ring with only an event on the worker's side stream between the hash kernels that read it
and the gather that refills it.  Past ``rows_hold_bytes`` (what a step holds on the device: a learning step's h, a levelless step's
leaves) the trees are the same as within it.  Openings of a run that kept no levels come from a replay that took its keys, and verify
against its 64-byte roots, as the replay's range openings verify against its own; a tampered row, salt or path does not.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("NATIVE_COLLECT_BUILD"),
                                reason="needs a GPU and the JIT-built native_collect extension (NATIVE_COLLECT_BUILD)")

from verity_vllm.commit import step_rows as R  # noqa: E402
from verity_vllm.commit.committer import native_collect as NC  # noqa: E402
from verity_vllm.commit.committer import native_host as nh  # noqa: E402

RUN_ID = "windowed-rows"
PD, GEO = hashlib.sha256(b"program").digest(), hashlib.sha256(b"geo").digest()
LAYERS, H, BN, VOCAB = 3, 1000, 16, 1003
WINDOW_MB = str(8 / 1024)                                       # 8 KiB: two rows a window
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


def committer(mp, *, retain: str = "none", retain_levels: bool = True, bounded: bool = False) -> NC.NativeCollectCommitter:
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
                                    native_hooks=True, auto_blocks=True, native_worker=True, retain_levels=retain_levels)
    com.enable_windows(None)
    eng.runner.committer = com
    assert com.rows and com.name.endswith("+" + R.NAME)
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


SCHED = schedule(40, 6)


def rows_root(step: int, tensors: list[tuple[str, torch.Tensor]], ks: bytes) -> bytes:
    """step_rows's step root over the step's packed stream (each tensor at the next ALIGN offset, zero-padded to a row), its layout
    counted in rows."""
    ref = nh.NativeHostCommitter(None, run_id=RUN_ID, program_digest=PD, geo_digest=GEO, retain_values=False, gpu_tree=True)
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
    tree_root = R.levels(R.domain(PD, ctx, GEO, len(rows)), rows, ks, 8 * R.ROW_BYTES)[-1][0]
    return R.step_root(PD, ctx, GEO, ref.layout_digest(metas), len(rows), tree_root)


def reference(keys: dict[int, bytes]) -> list[bytes]:
    return [rows_root(i, step_tensors(kv), keys[i]) for i, kv in enumerate(SCHED)]


def check_run(rc, keys: dict[int, bytes]) -> None:
    roots = [s.root for s in rc.steps]
    assert sorted(keys) == list(range(len(SCHED))) and len(set(keys.values())) == len(SCHED), "a fresh 64-byte key per step"
    assert roots == reference(keys)
    assert rc.run_root == R.run_root(PD, GEO, roots) and len(rc.run_root) == R.ROOT_BYTES
    assert rc.extra["commit_scheme"] == R.NAME


@pytest.mark.parametrize("retain_levels", [True, False], ids=["levels", "no-levels"])
def test_learning_and_hit_runs_commit_step_rows_roots(monkeypatch, retain_levels):
    com = committer(monkeypatch, retain_levels=retain_levels)
    try:
        learn = run(com, SCHED)
        learn_keys, reup, hfirst = dict(com.row_keys), com._win_reupload_bytes, com._win_hfirst_steps
        hit = run(com, SCHED)
        assert reup == 0 and hfirst == len(SCHED), "a learning step that keeps no bytes hashes its windows to h in place (no re-upload)"
        assert com._win_reupload_bytes == 0 and com._win_hfirst_steps == 0, "the second run follows the first run's plans (in place)"
        check_run(learn, learn_keys)
        check_run(hit, com.row_keys)
        assert hit.run_root != learn.run_root and not set(learn_keys.values()) & set(com.row_keys.values()), "each run's keys are fresh"
        assert bool(com._gpu_levels) is retain_levels
    finally:
        com.close()


@pytest.mark.parametrize("fast", [True, False], ids=["native-end", "python-end"])
def test_the_step_end_paths_commit_the_same_roots_and_row_order(monkeypatch, fast):
    """The serving thread's step end in one native call with the row order expanded by the worker, or as before: the same roots and
    the same `step_rows`."""
    com = committer(monkeypatch)
    com.native_end = com.rows_off_thread = fast
    try:
        rc = run(com, SCHED)
        check_run(rc, com.row_keys)
        assert list(com.step_rows.values()) == [[("r0", 0, 1), ("r1", 1, 2)]] * len(SCHED)
    finally:
        com.close()


H_WINDOW = 64 * int(float(WINDOW_MB) * (1 << 20)) // R.ROW_BYTES     # a full window's h, and its leaves: 64 B a row (two rows here)


@pytest.mark.parametrize("retain_levels", [True, False], ids=["levels", "no-levels"])
@pytest.mark.parametrize("hold", [0, 3 * H_WINDOW], ids=["hold-0", "hold-3-windows"])
def test_steps_past_the_device_hold_commit_the_same_trees(monkeypatch, retain_levels, hold):
    """`rows_hold_bytes` bounds what a step holds on the device whatever its length (each step here is about seven windows): a learning
    step keeps its windows' h until the next would pass the hold, and re-hashes that window and the rest from host; a step that keeps
    no levels folds its held leaves into their windows' top nodes whenever the next window's would pass it.  Each run, learning then
    hit, commits the trees the default hold commits with the same keys (a replay that took them): the roots, and with the levels kept
    every level, leaves included."""
    com = committer(monkeypatch, retain_levels=retain_levels)
    ref = committer(monkeypatch, retain_levels=retain_levels)
    com.rows_hold_bytes = hold
    assert ref.rows_hold_bytes == NC.ROWS_HOLD_BYTES
    try:
        for phase in ("learn", "hit"):
            rc = run(com, SCHED)
            ref.replay_row_keys = dict(com.row_keys)
            want = run(ref, SCHED)
            check_run(rc, com.row_keys)
            assert [s.root for s in rc.steps] == [s.root for s in want.steps] and rc.run_root == want.run_root
            assert sorted(com._gpu_levels) == sorted(ref._gpu_levels) == (list(range(len(SCHED))) if retain_levels else [])
            for s in com._gpu_levels:
                got, ref_lv = com._gpu_levels[s], ref._gpu_levels[s]
                assert len(got) == len(ref_lv) and all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(got, ref_lv)), (phase, s)
            w, wr = rc.extra["windows"], want.extra["windows"]
            assert w["rows_hold_bytes"] == hold and wr["leaf_folds"] == 0
            assert (w["leaf_folds"] > 0) is not retain_levels, "a levelless step folds its held leaves past the hold"
            if phase == "learn":
                assert w["hfirst_steps"] == (len(SCHED) if hold else 0) and wr["hfirst_steps"] == len(SCHED)
                assert w["reupload_bytes"] > 0 and wr["reupload_bytes"] == 0, "the windows past the hold re-hash from host"
            else:
                assert w["hfirst_steps"] == w["reupload_bytes"] == 0, "the hit run follows the learning run's plans (in place)"
    finally:
        com.close()
        ref.close()


class _SlotRunner(_Runner):
    """A replayed graph's step: the per-token tensors land in persistent slots (the captured copies) and are staged from them in one
    `stage_slot_plan` call and flushed (the source's drain); the rest are acquired one by one, as the logits and the sampled tokens are."""

    def __init__(self, lag: bool = False) -> None:
        super().__init__()
        self.slots: list[torch.Tensor] = []
        self.handle = -1
        self.lag = lag

    def execute_model(self, scheduler_output, *a, **kw):
        com, n = self.committer, scheduler_output.total_num_scheduled_tokens
        per_token = [(nm, t) for nm, t in self.tensors if t.is_contiguous() and t.shape[0] == n and not nm.startswith(("logits", "runner."))]
        if self.handle < 0:
            self.slots = [torch.empty((8, *t.shape[1:]), dtype=t.dtype, device=t.device) for _nm, t in per_token]
            self.handle = com.slot_plan([nm for nm, _t in per_token], self.slots, [1] * len(per_token))
        for slot, (_nm, t) in zip(self.slots, per_token):
            slot[:n].copy_(t)
        com.stage_slot_plan(self.handle, n, n)
        side = com._nc.side_stream_for_test()
        if self.lag and side:
            with torch.cuda.stream(torch.cuda.ExternalStream(side)):
                torch.cuda._sleep(20_000_000)                       # the gather queued behind it lags the serving stream by milliseconds
        com._nc.flush()
        for nm, t in self.tensors:
            if all(nm != p for p, _t in per_token):
                com._acquire(nm, t, com._cur)


def test_side_stream_drain_commits_step_rows_roots(monkeypatch):
    """A replayed step's slots gathered on the side stream (VERITY_DRAIN_SIDE, on by default), the serving stream joining it before its
    next flush, window mark and step end, and the next step rewriting the slots on the serving stream: the same roots as one
    acquisition per tensor in the same order, also with the side stream held back behind a sleep before each step's gather (from the
    second step, once it exists), so that a missing join would bind stale or rewritten slots; with VERITY_DRAIN_SIDE=0, the same roots
    with every flush on the serving stream; and with VERITY_GATHER_DESC_DEV=0 (descriptors read from the host-mapped ring, not copied to
    the device ring ahead of each gather), the same roots."""
    for side, lag, desc_dev in ((True, False, True), (False, False, True), (True, True, True), (True, False, False)):
        monkeypatch.setenv("VERITY_DRAIN_SIDE", "1" if side else "0")
        monkeypatch.setenv("VERITY_GATHER_DESC_DEV", "1" if desc_dev else "0")
        com = committer(monkeypatch, retain_levels=False)
        runner = _SlotRunner(lag)
        runner.committer, com.runner = com, runner
        try:
            for phase in ("learn", "hit"):
                rc = run(com, SCHED)
                keys, roots = com.row_keys, [s.root for s in rc.steps]
                want, plain = [], []
                for i, kv in enumerate(SCHED):
                    ts = step_tensors(kv)
                    first = [(nm, t) for nm, t in ts if t.is_contiguous() and t.shape[0] == len(kv) and not nm.startswith(("logits", "runner."))]
                    want.append(rows_root(i, first + [(nm, t) for nm, t in ts if all(nm != p for p, _t in first)], keys[i]))
                    plain.append(rows_root(i, ts, keys[i]))
                st = rc.extra["native_collect"]
                assert roots == want, (side, lag, desc_dev, phase, [i for i in range(len(SCHED)) if roots[i] != want[i]],
                                       [i for i in range(len(SCHED)) if roots[i] == plain[i]], st)
                assert st["gather_side_flushes"] == (len(SCHED) if side else 0), st
                assert (st["gather_desc_copies"] > 0) is desc_dev, st
        finally:
            com.close()


def test_capped_runs_commit_step_rows_roots(monkeypatch):
    """rows_cap_blocks's policy on the worker (each window capped while the ring has room; test_rows_cuda splits launches)."""
    com = committer(monkeypatch)
    com.rows_cap_blocks, com.rows_cap_free = 1, 0
    try:
        learn = run(com, SCHED)
        learn_keys = dict(com.row_keys)
        hit = run(com, SCHED)
        check_run(learn, learn_keys)
        check_run(hit, com.row_keys)
    finally:
        com.close()


@pytest.mark.parametrize("hold", [NC.ROWS_HOLD_BYTES, H_WINDOW], ids=["within-hold", "past-hold"])
def test_bounded_record_run_commits_step_rows_roots(monkeypatch, hold):
    """Bounded staging, a 16 KiB transient budget under every step: within `rows_hold_bytes` a learning step that keeps no bytes copies
    nothing to host, so the warm-up commits it; past it (one window's h here) its later windows are copied to transient blocks, so the
    warm-up discards it (learned only) and the record run, hashing in place to the plans it learned, commits it."""
    com = committer(monkeypatch, retain_levels=False, bounded=True)
    com.rows_hold_bytes = hold
    try:
        com.learn_only = True
        warm = run(com, SCHED)
        if hold == NC.ROWS_HOLD_BYTES:
            assert not warm.extra["bounded_staging"]["learned_only_steps"] and com._win_hfirst_steps == len(SCHED), \
                "learning steps that keep no bytes copy nothing to host within rows_hold_bytes, so over the budget they are committed"
            check_run(warm, com.row_keys)
        else:
            assert warm.extra["bounded_staging"]["learned_only_steps"] == list(range(len(SCHED)))
        com.learn_only = False
        rec = run(com, SCHED)
        bs = rec.extra["bounded_staging"]
        assert not bs["learned_only_steps"] and bs["plan_mismatches"] == 0 and bs["transient_peak_bytes"] == 0, bs
        assert (rec.extra["windows"]["leaf_folds"] > 0) is (hold != NC.ROWS_HOLD_BYTES)
        check_run(rec, com.row_keys)
        assert not com._gpu_levels
    finally:
        com.close()


def test_a_run_without_levels_opens_from_a_replay_that_took_its_keys(monkeypatch):
    com = committer(monkeypatch, retain_levels=False)
    rep = committer(monkeypatch, retain="host")
    try:
        rc = run(com, SCHED)
        rep.replay_row_keys = dict(com.row_keys)
        rrc = run(rep, SCHED)
        assert rep.replay_row_keys is None and rep.row_keys == com.row_keys
        assert [s.root for s in rrc.steps] == [s.root for s in rc.steps] and rrc.run_root == rc.run_root
        sample = [(s, i) for s, sc in enumerate(rc.steps) for i in sorted({0, 1, sc.n_leaves // 2, sc.n_leaves - 1})]
        ops = com.open(sample, values_from=rep, paths_from_replay=True)
        assert len(ops) == len(sample) and all(com.verify(rc, o) for o in ops)
        with pytest.raises(RuntimeError, match="kept no tree levels"):
            com.open(sample[:1], values_from=rep)
        assert all(len(o.salt) == R.SALT_BYTES and o.salt == R.salt(com.row_keys[o.step], o.index) for o in ops)
        o = ops[0]
        flip = bytes([o.value[0] ^ 1]) + o.value[1:]
        assert not com.verify(rc, replace(o, value=flip))
        assert not com.verify(rc, replace(o, salt=bytes([o.salt[0] ^ 1]) + o.salt[1:]))
        assert not com.verify(rc, replace(o, path=o.path[:-1]))
        assert not com.verify(rc, replace(o, index=o.index + 1))
        ro = rep.open_range(0, 0, min(2, rrc.steps[0].n_leaves))
        assert rep.verify_range(rrc, ro) and ro.salts == b"".join(R.salt(rep.row_keys[0], i) for i in range(ro.lo, ro.hi))
        assert not rep.verify_range(rrc, replace(ro, salts=bytes(len(ro.salts))))
        rep.reset()
        rep.replay_row_keys = {}
        with pytest.raises(RuntimeError, match="no salt key for step 0"):
            rep._row_key(0)
        rep.replay_row_keys = None
        unkeyed = run(rep, SCHED[:1])
        assert unkeyed.steps[0].root != rc.steps[0].root, "a replay without the run's keys commits other trees"
    finally:
        com.close()
        rep.close()


def test_what_rows_sha512_refuses(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VERITY_WINDOW_MB", "0")
    eng = SimpleNamespace(model=torch.nn.Module(), runner=_Runner())
    base = dict(run_id=RUN_ID, program_digest=PD, geo_digest=GEO, arena_bytes=None, wrap=True, gather=True, native_hooks=True,
                auto_blocks=True, native_worker=True)
    with pytest.raises(ValueError, match="windowed worker"):
        NC.NativeCollectCommitter(eng, **base)
    monkeypatch.setenv("VERITY_WINDOW_MB", WINDOW_MB)
    for scheme in ("vllm-v1", "rows-sha256"):
        with pytest.raises(ValueError, match="unknown commit scheme"):
            NC.NativeCollectCommitter(eng, commit_scheme=scheme, **base)
