"""Openings stay possible after the raw staged bytes are released.

The GPU-tree committer used to keep a compact DEVICE copy of every step's packed bytes for lazy openings (the ~9.6 GiB per 1024-token
Llama prefill step that OOMs at B=2).  The retention policy (`VERITY_RETAIN=device|host|none`) moves that copy to host memory or drops it;
`release_staging()` frees whatever raw copy is still on the device; `open(..., values_from=replay)` answers a challenge from a second
deterministic execution, with the sibling path from the ORIGINAL levels and `verify()` against the ORIGINAL roots.  These tests run the
CPU reference of the same chunk tree (`commit_block_offline`) so they need no CUDA:

  * host retention: release frees nothing raw on the device, 64 challenge openings + forced picks verify;
  * no retention: openings need a replay source; a byte-identical replay opens every position against the original root;
  * a replay that differs in ONE byte fails exactly the opening whose chunk holds that byte -- the original root, not the replay, is the judge;
  * the run root of a replay equals the original (the check `commit_verdict` makes when the source is a replay);
  * `commit_verdict` fails closed when openings-after-release are short, unverified, from a non-matching replay, or bytes stayed on device.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.check.verdict import commit_verdict  # noqa: E402
from verity_vllm.commit.challenge import challenge_positions  # noqa: E402
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, TensorMeta, value_family  # noqa: E402


def _metas(step: int, seed: int) -> tuple[list[TensorMeta], bytes]:
    """A packed step block like the collector's: GEMM boundary rows, a norm, an FA2 hidden `fa2.m1/` row (the 82 % family), sampled ids."""
    import random

    rnd = random.Random(seed * 1000 + step)
    plan = [("model.layers.0.self_attn.qkv_proj", "bfloat16", (4, 3072), 4 * 3072 * 2), ("model.layers.0.self_attn.attn", "bfloat16", (4, 2048), 4 * 2048 * 2),
            ("model.layers.0.self_attn.o_proj", "bfloat16", (4, 2048), 4 * 2048 * 2), ("model.layers.0.post_attention_layernorm", "bfloat16", (4, 2048), 4 * 2048 * 2),
            ("fa2.m1/L0/g1x64x1", "uint32", (64, 130), 64 * 130 * 4), ("model.layers.0.mlp.down_proj", "bfloat16", (4, 2048), 4 * 2048 * 2),
            ("logits", "float32", (1, 1000), 4000), ("runner.sampler/sampled_token_ids", "int64", (1, 1), 8)]
    metas, off, first_leaf, buf = [], 0, 0, bytearray()
    for i, (name, dt, shape, nbytes) in enumerate(plan):
        n_leaves = -(-nbytes // GPU_CHUNK_BYTES)
        m = TensorMeta(i, name, dt, shape, nbytes, GPU_CHUNK_BYTES, first_leaf, n_leaves)
        m.dev_off = off
        metas.append(m)
        buf += rnd.randbytes(nbytes)
        pad = (-nbytes) % 4
        buf += b"\0" * pad
        off += nbytes + pad
        first_leaf += n_leaves
    return metas, bytes(buf)


def _committer(retain: str, run_id: str = "p0-test") -> NativeHostCommitter:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VERITY_RETAIN", retain)
        return NativeHostCommitter(None, run_id=run_id, gpu_tree=True)


def _run(com: NativeHostCommitter, seed: int = 1, steps: int = 3):
    for s in range(steps):
        metas, block = _metas(s, seed)
        com.commit_block_offline(s, block, metas)
    return com.finalize()


def _sample(rc, k: int = 64):
    sample = list(challenge_positions(rc.run_root, [s.n_leaves for s in rc.steps], k, salt=b"after-release/0"))
    # forced picks as the harness makes them: inside the FA2 hidden row of step 0, its last down_proj, leaf 0, and the last leaf of the last step
    return sample, k


def test_host_retention_opens_everything_after_release():
    com = _committer("host")
    rc = _run(com)
    assert all(b is not None and not b.is_cuda for b, _u in com._gpu_blocks.values())
    freed = com.release_staging()
    assert freed == {"raw_device_bytes": 0, "levels_device_bytes": 0}
    sample, k = _sample(rc)
    fa2 = next(m for m in com._layouts[0] if "fa2.m1/" in m.name)
    forced = [(0, (fa2.stream_off + fa2.nbytes // 2) // com.chunk), (0, 0), (2, rc.steps[2].n_leaves - 1)]
    ops = com.open(sample + forced)
    assert len(ops) == k + 3 and all(com.verify(rc, o) for o in ops)
    assert value_family(ops[k].identity["name"]) == "fa2_hidden_m1"


def test_no_retention_needs_a_replay_and_the_replay_opens_against_the_original_root():
    orig = _committer("none")
    rc = _run(orig)
    assert all(b is None for b, _u in orig._gpu_blocks.values())
    assert orig._retained_device_bytes() == 0
    sample, k = _sample(rc)
    with pytest.raises(RuntimeError, match="raw bytes released"):
        orig.open(sample[:1])
    replay = _committer("host")                       # the replay keeps its values on host for the duration of the opening
    rc2 = _run(replay)                                # same committed inputs -> same bytes (BI determinism stands in for re-execution here)
    assert rc2.run_root == rc.run_root
    ops = orig.open(sample, values_from=replay)
    assert len(ops) == k and all(orig.verify(rc, o) for o in ops)


def test_a_replay_that_differs_in_one_byte_fails_exactly_that_opening():
    orig = _committer("none")
    rc = _run(orig)
    replay = _committer("host")
    _run(replay)
    fa2 = next(m for m in orig._layouts[1] if "fa2.m1/" in m.name)
    victim_byte = fa2.stream_off + 777                # inside the FA2 hidden row of step 1
    blk, used = replay._gpu_blocks[1]
    blk[victim_byte] ^= 0x01
    victim_leaf = victim_byte // orig.chunk
    sample = [(1, victim_leaf), (1, victim_leaf + 1), (0, 3), (2, 0)]
    ops = orig.open(sample, values_from=replay)
    ok = [orig.verify(rc, o) for o in ops]
    assert ok == [False, True, True, True]


def test_replay_with_a_different_layout_is_refused():
    orig = _committer("none")
    rc = _run(orig)
    other = _committer("host")
    _run(other, seed=1)
    other._layouts[0][0].nbytes += 4                  # layout digest differs -> not the same committed step
    with pytest.raises(RuntimeError, match="replay layout differs"):
        orig.open([(0, 0)], values_from=other)


def test_device_release_then_frozen_view_survives_reset():
    """What the harness does: freeze the finished run's state, release the raw copies, reset the live committer for the replay."""
    if not torch.cuda.is_available():
        com = _committer("host")
    else:
        com = _committer("device")
    rc = _run(com)
    frozen = copy.copy(com)
    frozen.installed = True
    freed = frozen.release_staging()
    if torch.cuda.is_available():
        assert freed["raw_device_bytes"] > 0 and frozen._retained_device_bytes() == 0
        assert all(b is None for b, _u in frozen._gpu_blocks.values())
    com.reset()
    assert frozen._layouts and not com._layouts
    replay = _committer("host")
    _run(replay)
    ops = frozen.open([(0, 0), (2, 1)], values_from=replay)
    assert all(frozen.verify(rc, o) for o in ops)


def _rec(oar):
    return {"extra": {"expected_tensors": 8, "committed_tensors": 8, "errors": [], "errors_n": 0}, "openings": {"n": 64, "verified": 64},
            "openings_after_release": dict(oar) if oar is not None else None}


def test_verdict_fails_closed_on_openings_after_release():
    good = {"n": 68, "verified": 68, "source": "replay", "replay_root_equal": True, "retained_device_after": 0}
    assert commit_verdict(_rec(good), True) == (True, "PASS")
    assert not commit_verdict(_rec({**good, "verified": 67}), True)[0]
    assert not commit_verdict(_rec({**good, "n": 60, "verified": 60}), True)[0]
    assert not commit_verdict(_rec({**good, "replay_root_equal": False}), True)[0]
    assert not commit_verdict(_rec({**good, "retained_device_after": 4096}), True)[0]
    assert commit_verdict(_rec({**good, "source": "host", "replay_root_equal": None}), True)[0]


# ----------------------------------------------------------------------------------------------------------------------------------
# retain=host + excluded families: the excluded interior chunks are NOT on host (`_HostRanges`), every other leaf is;
# the excluded leaves open by replay -- regenerated value AND regenerated authentication path -- against the ORIGINAL root.

def _sparse(com: NativeHostCommitter, exclude: set[str]) -> dict[int, int]:
    """Turn the CPU fixture's contiguous host blocks into `_HostRanges` that omit the excluded families' interiors (what the bounded
    windowed worker builds); returns the excluded bytes per step."""
    from verity_vllm.commit.committer.native_collect import _HostRanges, _excluded_chunk_ranges

    out = {}
    for step, (blk, used) in list(com._gpu_blocks.items()):
        ex = _excluded_chunk_ranges(com._layouts[step], exclude, GPU_CHUNK_BYTES)
        ranges, pos, n = [], 0, int(blk.numel())
        for lo, hi, _fam in ex:
            if lo > pos:
                ranges.append((pos, lo, blk[pos:lo].clone()))
            pos = hi
        if pos < n:
            ranges.append((pos, n, blk[pos:n].clone()))
        com._gpu_blocks[step] = (_HostRanges(ranges, used, {f: sum(h - l for l, h, ff in ex if ff == f) for f in exclude}), used)
        out[step] = sum(h - l for l, h, _f in ex)
    return out


def test_excluded_chunk_ranges_are_the_chunk_aligned_interior():
    from verity_vllm.commit.committer.native_collect import _excluded_chunk_ranges

    metas, _ = _metas(0, 1)
    fa2 = next(m for m in metas if m.name.startswith("fa2.m1/"))
    ex = _excluded_chunk_ranges(metas, {"fa2_hidden_m1"}, GPU_CHUNK_BYTES)
    assert len(ex) == 1 and ex[0][2] == "fa2_hidden_m1"
    lo, hi, _ = ex[0]
    assert lo % GPU_CHUNK_BYTES == 0 and hi % GPU_CHUNK_BYTES == 0
    assert lo >= fa2.dev_off and hi <= fa2.dev_off + fa2.nbytes            # strictly inside the tensor: boundary chunks stay on host
    assert lo - fa2.dev_off < GPU_CHUNK_BYTES and (fa2.dev_off + fa2.nbytes) - hi < GPU_CHUNK_BYTES
    assert _excluded_chunk_ranges(metas, {"no_such_family"}, GPU_CHUNK_BYTES) == []


def test_sparse_host_ranges_hold_exactly_the_retained_bytes():
    from verity_vllm.commit.committer.native_collect import _HostRanges

    com = _committer("host")
    rc = _run(com)
    full = {s: bytes(b.numpy().tobytes()) for s, (b, _u) in com._gpu_blocks.items()}
    excl = _sparse(com, {"fa2_hidden_m1"})
    assert all(v > 0 for v in excl.values())
    for step, (blk, used) in com._gpu_blocks.items():
        assert isinstance(blk, _HostRanges) and blk.numel() == -(-used // GPU_CHUNK_BYTES) * GPU_CHUNK_BYTES
        assert blk.retained_bytes + excl[step] == blk.numel()
        fa2 = next(m for m in com._layouts[step] if m.name.startswith("fa2.m1/"))
        n_leaves = rc.steps[step].n_leaves
        held = [com.leaf_retained(step, i) for i in range(n_leaves)]
        # the excluded leaves are exactly the interior chunks of the FA2 row; every other leaf (incl. its two boundary chunks) is held
        for i in range(n_leaves):
            lo, hi = i * GPU_CHUNK_BYTES, (i + 1) * GPU_CHUNK_BYTES
            interior = lo >= fa2.dev_off and hi <= fa2.dev_off + fa2.nbytes
            assert held[i] == (not interior), (step, i)
            if held[i]:
                assert bytes(blk[lo:hi].numpy().tobytes()) == full[step][lo:hi]  # the same bytes as the contiguous copy
            else:
                with pytest.raises(KeyError, match="open by replay"):
                    blk[lo:hi]
        assert sum(1 for h in held if not h) == excl[step] // GPU_CHUNK_BYTES
        # a slice spanning a retained range boundary (two consecutive ranges) works; one touching the gap does not
        assert bytes(blk[0:2 * GPU_CHUNK_BYTES].numpy().tobytes()) == full[step][0:2 * GPU_CHUNK_BYTES]


@pytest.mark.slow
def test_sparse_retention_opens_held_leaves_from_host_and_excluded_leaves_by_replay_with_replay_paths():
    orig = _committer("host")
    rc = _run(orig)
    _sparse(orig, {"fa2_hidden_m1"})
    sample, k = _sample(rc)
    fa2 = next(m for m in orig._layouts[0] if m.name.startswith("fa2.m1/"))
    inside = (fa2.dev_off + fa2.nbytes // 2) // orig.chunk                  # an excluded (interior) leaf of step 0
    assert not orig.leaf_retained(0, inside)
    held = [p for p in sample if orig.leaf_retained(*p)]
    not_held = [p for p in sample if not orig.leaf_retained(*p)] + [(0, inside)]
    assert held, "the challenge sample must have host-held leaves"
    # host share: opens and verifies with no replay
    ops_h = orig.open(held)
    assert all(orig.verify(rc, o) for o in ops_h)
    # excluded share: no replay source -> a named error, never zeros
    with pytest.raises(RuntimeError, match="raw bytes released / not retained"):
        orig.open([(0, inside)])
    # a byte-identical replay: regenerated VALUES + regenerated AUTHENTICATION PATHS verify against the ORIGINAL roots
    replay = _committer("host")
    rc2 = _run(replay)
    assert rc2.run_root == rc.run_root
    ops_r = orig.open(not_held, values_from=replay, paths_from_replay=True)
    assert len(ops_r) == len(not_held) and all(orig.verify(rc, o) for o in ops_r)
    assert value_family(ops_r[-1].identity["name"]) == "fa2_hidden_m1"
    # the original root is the judge of BOTH regenerated parts: a replay VALUE that differs in one byte fails that opening (its
    # neighbour, whose regenerated value and path are right, still opens) ...
    victim = fa2.dev_off + 777
    blk, _u = replay._gpu_blocks[0]
    blk[victim] ^= 0x01
    victim_leaf = victim // orig.chunk
    ops_bad = orig.open([(0, victim_leaf), (0, victim_leaf + 1), (1, 0)], values_from=replay, paths_from_replay=True)
    assert [orig.verify(rc, o) for o in ops_bad] == [False, True, True]
    # ... and a replay TREE that differs in one node (a regenerated authentication path that is not the original's) fails every
    # opening whose path passes through that node -- the neighbour's does (the node is its sibling or in its sibling subtree)
    blk[victim] ^= 0x01
    lv0 = replay._gpu_levels[0][0]
    lv0[victim_leaf, 0] ^= 1
    sib = victim_leaf ^ 1                                                     # the Merkle sibling: its path's first node IS the victim's digest
    ops_bad = orig.open([(0, sib), (0, victim_leaf), (1, 0)], values_from=replay, paths_from_replay=True)
    assert [orig.verify(rc, o) for o in ops_bad] == [False, True, True]     # (the victim's own path does not contain its own digest)
    lv0[victim_leaf, 0] ^= 1


def test_schedule_stream_bound_reproduces_the_sizing_on_the_measured_b1_row():
    """Bounded admission: the declared-schedule stream model, fed the B1 1024/128 sizing record measured on cp-h100:
    its single prefill step must equal the sizing's prefill_step_bytes (same widths, same FA3 launch geometry), its decode steps the sizing's
    decode at the growing kv length, and the run total must sit under the measured run (25,075,721,724 B) by the factor the admission carries."""
    from verity_vllm.commit.committer.native_host import schedule_stream_bound
    asz = {"per_token_bytes": 2900996, "per_seq_bytes": 607744, "per_step_bytes": 93184,
           "model": {"hidden": 2560, "heads": 32, "kv_heads": 8, "head_dim": 128, "layers": 36, "hidden_m1": True, "attn_backend": "fa3"},
           "prefill_step_bytes": 20203618816, "prefill_hidden_bytes": 17232297984, "decode_step_bytes": 22531588, "decode_hidden_bytes": 18929664,
           "retained_bytes_bound": 23087662080}
    reqs = [{"prompt_token_ids": list(range(1024)), "arrive_step": 0, "max_tokens": 128}]
    s = schedule_stream_bound(asz, reqs, {"max_tokens": 128})
    assert s["steps"] == 128 and s["requests"] == 1
    big = s["largest_step"]
    assert big["step"] == 0 and big["tokens"] == 1024 and big["hidden_bytes"] == asz["prefill_hidden_bytes"] and big["bytes"] == asz["prefill_step_bytes"]
    # decode at kv 1151 (the last step) is the sizing's decode at max_model_len 1152 up to the tile: same order, never above it
    assert 0 < s["decode_step_bytes_max"] <= asz["decode_step_bytes"]
    assert s["hidden_prefill_bytes"] == asz["prefill_hidden_bytes"] and s["hidden_bytes"] > s["hidden_prefill_bytes"]
    measured_run = 25075721724
    assert s["run_bytes"] < measured_run < s["run_bytes"] * 1.15, (s["run_bytes"], measured_run)     # the 1.15 factor covers the measured run
    # B8: eight prompts arriving together prefill in ONE step (the row of record) -- the envelope bound counts three such steps
    reqs8 = [{"prompt_token_ids": list(range(443)), "arrive_step": 0, "max_tokens": 128} for _ in range(8)]
    s8 = schedule_stream_bound(asz, reqs8, {"max_tokens": 128})
    assert s8["largest_step"]["prefill_seqs"] == 8 and s8["largest_step"]["tokens"] == 8 * 443 and s8["steps"] == 128
    assert s8["run_bytes"] < 3 * s8["largest_step"]["bytes"]


def test_bounded_host_bound_term_by_term_covers_every_measured_b1_arm_and_admits_b8():
    """The bound is derived term by term (native_host.bounded_host_bound) and must sit ABOVE every
    RSS peak measured on cp-h100 for the B1 1024/128 bounded arms (rss_timeline 2 s samples) -- including the three runs
    that beat the earlier formulas -- while the per-component factors are the ones measured on the same workload (hidden exact -> 1.05;
    other x1.59 -> 2.0).  The B8 workload (declared schedule: one 3,537-token prefill) must be admitted under the 239,372 MiB cgroup with margin."""
    from verity_vllm.commit.committer.native_host import bounded_host_bound, schedule_stream_bound
    asz = {"per_token_bytes": 2900996, "per_seq_bytes": 607744, "per_step_bytes": 93184,
           "model": {"hidden": 2560, "heads": 32, "kv_heads": 8, "head_dim": 128, "layers": 36, "hidden_m1": True, "attn_backend": "fa3"},
           "prefill_step_bytes": 20203618816, "prefill_hidden_bytes": 17232297984, "decode_step_bytes": 22531588, "decode_hidden_bytes": 18929664,
           "retained_bytes_bound": 23087662080}
    MiB = 1 << 20
    s1 = schedule_stream_bound(asz, [{"prompt_token_ids": list(range(1024)), "arrive_step": 0, "max_tokens": 128}], {"max_tokens": 128})
    # measured components (runs.jsonl, bounded-x arm): excluded hidden 19,635,194,880 B (sized 19,636,365,312: exact), other kept 5,440,526,844 B
    assert abs(s1["hidden_bytes"] - 19635194880) < 2 * MiB
    assert 1.5 < 5440526844 / (s1["run_bytes"] - s1["hidden_bytes"]) < 2.0                       # the 'other' under-sizing the 2.0 factor covers
    # the measured B1 records ran on torch's pinned pool with no C2 Programs term: this part of the test keeps
    # that derivation (pool_exact=False, c2_programs_bytes=0) and the measured numbers; `pool_rounding` is removed from the sums below
    # to compare like with like -- the new terms are asserted in test_bound_phases.py
    common = dict(learn_budget_bytes=4096 * MiB, openings=64, chunk_bytes=256, window_bytes=256 * MiB, bounded=True, open_after_release=True,
                  host_cap_bytes=239372 * MiB, pool_exact=False, c2_programs_bytes=0)

    def _legacy(b):   # the §3 sum: engine + levels + kept + learn + replay + slack + pairs (no pow2 rounding term)
        return b["host_peak_bound_bytes"] - b["pool_rounding_bytes"]

    bx = bounded_host_bound(s1, engine_rss_bytes=2189 * MiB, retain="host", retain_exclude={"fa3_hidden_m1"}, **common)
    bn = bounded_host_bound(s1, engine_rss_bytes=2178 * MiB, retain="none", retain_exclude=set(), **common)
    assert bx["pool_rounding_bytes"] == bx["levels_bytes"] + bx["raw_kept_bytes"] + bx["learn_budget_bytes"]     # torch pool: x2 worst case, named
    # every measured peak of the arms (MiB): bounded-x try1 37,281 / try2 42,272 (window replay, superseded) / try3 25,643; bounded-none try1
    # 35,131 (window replay, superseded) / try3 23,325 -- the chunk-replay runs are the ones this formula is for; the window-replay peaks are
    # listed so the record shows what the formula does NOT cover (a window replay is not a bounded replay)
    assert _legacy(bx) > 25643 * MiB and _legacy(bx) // MiB == 36166, _legacy(bx) // MiB
    assert _legacy(bn) > 23325 * MiB and _legacy(bn) // MiB == 29615, _legacy(bn) // MiB
    assert bx["raw_kept_bytes"] > 5440526844 and bx["levels_bytes"] > 6268967168                   # kept raw and levels each cover their measurement
    assert bn["raw_kept_bytes"] == 0 and bn["replay_bytes"] == bn["levels_bytes"] + bn["learn_budget_bytes"] + bn["picks"] * 256
    # B8 row of record: 365/633/11/1024/817/449/165/73 at step 0 (one prefill of 3,537 tokens), caps up to 128
    reqs8 = [{"prompt_token_ids": list(range(n)), "arrive_step": 0, "max_tokens": c}
             for n, c in zip((365, 633, 11, 1024, 817, 449, 165, 73), (128, 113, 95, 35, 121, 3, 9, 2))]
    s8 = schedule_stream_bound(asz, reqs8, {"max_tokens": 128})
    b8 = bounded_host_bound(s8, engine_rss_bytes=2193 * MiB, retain="host", retain_exclude={"fa3_hidden_m1"}, **common)
    assert s8["largest_step"]["tokens"] == 3537 and s8["largest_step"]["prefill_seqs"] == 8
    assert _legacy(b8) < 239372 * MiB * 0.9 and (239372 * MiB - _legacy(b8)) / (239372 * MiB) > 0.45, (_legacy(b8) // MiB)
    assert b8["raw_kept_bytes"] // MiB > 13188                                                    # the blanket-1.15 kept term this replaces
    assert bx["pairs_extra_bytes"] == 0 and b8["pairs_extra_bytes"] == 0 and b8["pairs"] == 1     # a 1-pair bound has no pairs term
    # PAIRS >= 2: one more frozen pool (levels + kept + learn), once -- the B1 PAIRS=3 bounded-x curve (in-process per-step RSS, try 4
    # pairs3: 16.7 -> 25.6 -> 29.6 -> 30.2 -> 30.2 GiB) grew 4.6 GiB = 41 % of that pool after pair 0 and then held flat through pair 2
    bx3 = bounded_host_bound(s1, engine_rss_bytes=2189 * MiB, retain="host", retain_exclude={"fa3_hidden_m1"}, pairs=3, **common)
    assert bx3["pairs_extra_bytes"] == bx3["levels_bytes"] + bx3["raw_kept_bytes"] + bx3["learn_budget_bytes"]
    assert _legacy(bx3) == _legacy(bx) + bx3["pairs_extra_bytes"]
    assert _legacy(bx3) > 30189 * MiB and (30189 * MiB - 25634 * MiB) < 0.5 * bx3["pairs_extra_bytes"]   # measured growth < half the term
    # the B8 row's Commit of record runs row_pod's 3 pairs: the pairs term must still fit the cgroup at --host-admit 0.10
    b8x3 = bounded_host_bound(s8, engine_rss_bytes=2193 * MiB, retain="host", retain_exclude={"fa3_hidden_m1"}, pairs=3, **common)
    assert _legacy(b8x3) < 239372 * MiB * 0.9 and (239372 * MiB - _legacy(b8x3)) / (239372 * MiB) > 0.15, (_legacy(b8x3) // MiB)
    # ... and WITH the pow2 rounding term the same 3-pair torch-pool B8 run is NOT admissible -- it measured 223,776 MiB against
    # that 193,827 MiB sum; the derivation says so before the run (test_bound_phases.py reconstructs that run)
    assert b8x3["host_peak_bound_bytes"] > 239372 * MiB * 0.9
    # the dominant contributor is NAMED: the largest host term and the largest single materialization (the 3,537-token
    # prefill step's hidden export) with where it is (excluded from host, opened by replay) -- a refusal never reads merely "predicted OOM"
    dom = b8x3["dominant_contributor"]
    assert dom["host_term"] == "pairs" and dom["host_term_bytes"] == b8x3["pairs_extra_bytes"]                  # 3 pairs: the pairs term leads
    assert b8["dominant_contributor"]["host_term"] == "pool_rounding"      # 1 pair, torch pool: the pow2 rounding term (x2 worst case) leads
    b8e = bounded_host_bound(s8, engine_rss_bytes=2193 * MiB, retain="host", retain_exclude={"fa3_hidden_m1"}, **{**common, "pool_exact": True})
    assert b8e["dominant_contributor"]["host_term"] == "replay" and b8e["pool_rounding_bytes"] == 0          # exact pool: the fresh replay pool
    lm = dom["largest_materialization"]
    assert lm["step"] == s8["largest_step"]["step"] and lm["tokens"] == 3537 and lm["hidden_bytes"] == s8["largest_step"]["hidden_bytes"]
    assert lm["hidden_bytes"] > 100 * 1024 * MiB and "opened by replay" in lm["where"] and "excluded from host" in lm["where"]
    assert "step" in dom["text"] and "3537 tok" in dom["text"] and "excluded from host" in dom["text"]
    assert bn["dominant_contributor"]["largest_materialization"]["where"].startswith("not retained (retain none)")
    # the derivation names per_pick as the 256 B chunk and where the replay's transient windows are counted
    assert "256 B chunk" in b8x3["derivation"] and "learn term" in b8x3["derivation"]


def test_verdict_rules_for_host_plus_replay_and_bounded_records():
    good = {"n": 68, "verified": 68, "source": "host+replay", "replay_root_equal": True, "retained_device_after": 0, "replay_errors_n": 0}
    assert commit_verdict(_rec(good), True) == (True, "PASS")
    assert not commit_verdict(_rec({**good, "replay_root_equal": False}), True)[0]        # host+replay needs the replay root too
    assert not commit_verdict(_rec({**good, "replay_root_equal": None}), True)[0]
    assert not commit_verdict(_rec({**good, "replay_errors_n": 1}), True)[0]              # a replay with collector errors is not a replay of record
    # the in-run openings under exclude: `skipped` set with a host share -> the host share must verify; the after-release record carries the rest
    rec = _rec(good)
    rec["openings"] = {"n": 40, "verified": 40, "skipped": "retain=host excluding ['fa2_hidden_m1']: 24 of 64 -> replay", "deferred_replay_n": 24}
    rec["openings_after_release"]["deferred_received_n"] = 24                          # ... and the after-release phase received them all
    assert commit_verdict(rec, True)[0]
    rec["openings_after_release"]["deferred_received_n"] = 23                          # one deferred pick opened nowhere -> FAIL by name
    ok, why = commit_verdict(rec, True)
    assert not ok and "deferred to replay" in why and "received 23" in why, why
    rec["openings_after_release"]["deferred_received_n"] = 24
    rec["openings"]["verified"] = 39
    assert not commit_verdict(rec, True)[0]
    # the population-drawn openings (binding) partition the same way: host share verifies, deferred share must reach the replay phase.
    # A record WITH a binding block is a row-procedure record: C2 of record (replay / linkage / pin) must be on it for the PASS branch.
    from verity_vllm.check.commit_rules import binding_check
    po = {"n": 81, "bound": 81, "unbound_n": 0, "opened": 76, "verified": 76, "skipped": "retain=host excluding [...]: 5 of 81 -> replay", "deferred_replay_n": 5}
    bnd = {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 69761, "bound_n": 69761, "missing_n": 0}, "population": {"n": 69761}, "openings_population": po}
    assert binding_check(bnd) == (True, "PASS")
    ok, why = binding_check({**bnd, "openings_population": {**po, "verified": 75}})
    assert not ok and "host share" in why, why
    rec = _rec(good)
    rec["openings_after_release"]["deferred_received_n"] = 5
    rec["binding"] = bnd
    rec["sampled_replay"] = {"result": True, "grade": "complete", "population": {}}
    rec["boundary_linkage"] = {"result": True}
    rec["weights_pin"] = {"result": True}
    assert commit_verdict(rec, True)[0], commit_verdict(rec, True)
    rec["openings_after_release"]["deferred_received_n"] = 4                           # one population pick opened nowhere -> FAIL by name
    ok, why = commit_verdict(rec, True)
    assert not ok and "5 population pick(s) deferred" in why, why
    # by POSITION: 24 challenge + 5 population deferred, the after-release phase received 24 -- count-only would pass, the
    # positions say five picks opened nowhere
    pa = [[3, i] for i in range(24)]
    pb = [[7, 100 + i] for i in range(5)]
    rec["openings"] = {"n": 40, "verified": 40, "skipped": "...", "deferred_replay_n": 24, "deferred_positions": pa}
    rec["binding"]["openings_population"] = {**po, "deferred_positions": pb}
    rec["openings_after_release"]["deferred_received_n"] = 24
    rec["openings_after_release"]["deferred_received_positions"] = pa
    ok, why = commit_verdict(rec, True)
    assert not ok and "5 position(s) not received" in why and "[(7, 100)" in why, why
    rec["openings_after_release"]["deferred_received_positions"] = pa + pb            # all received (order/duplicates irrelevant)
    rec["openings_after_release"]["deferred_received_n"] = 29
    assert commit_verdict(rec, True)[0], commit_verdict(rec, True)
    rec["openings"]["deferred_positions"] = pa[:-1]                                     # a count without its positions is not a record
    ok, why = commit_verdict(rec, True)
    assert not ok and "positions are not on the record" in why, why
    # the count-only branch is only for records WITHOUT any position key: a positional writer that leaves the lists empty
    # (omission) is refused, a pre-positions record (no keys at all) still falls back to the count rule
    rec["openings"] = {"n": 40, "verified": 40, "skipped": "...", "deferred_replay_n": 24, "deferred_positions": []}
    rec["binding"]["openings_population"] = {**po, "deferred_positions": []}
    rec["openings_after_release"]["deferred_received_positions"] = []
    ok, why = commit_verdict(rec, True)
    assert not ok and "positions are not on the record" in why, why
    del rec["openings"]["deferred_positions"]; del rec["binding"]["openings_population"]["deferred_positions"]
    del rec["openings_after_release"]["deferred_received_positions"]
    assert commit_verdict(rec, True)[0], commit_verdict(rec, True)                      # received 29 >= max(24, 5): the older-record rule
    rec["openings_after_release"]["deferred_received_n"] = 23
    ok, why = commit_verdict(rec, True)
    assert not ok and "received 23" in why, why
    rec["openings"] = {"n": 40, "verified": 40, "skipped": "...", "deferred_replay_n": 24, "deferred_positions": pa}
    rec["binding"]["openings_population"] = {**po, "deferred_positions": pb}
    rec["openings_after_release"]["deferred_received_positions"] = pa + pb
    rec["openings_after_release"]["deferred_received_n"] = 29
    del rec["openings_after_release"]                                                  # deferred picks and no after-release phase at all -> FAIL
    ok, why = commit_verdict(rec, True)
    assert not ok and "no openings-after-release" in why, why                         # (the in-run `skipped` rule names it first)
    rec["openings"] = {"n": 64, "verified": 64}                                        # ... and with only POPULATION picks deferred, the new rule does
    ok, why = commit_verdict(rec, True)
    assert not ok and "deferred to replay" in why, why
    # a bounded record run must not carry a learned-only step (placeholder root) nor be finalized in learn-only mode
    rec = _rec(good)
    rec["extra"]["bounded_staging"] = {"enabled": True, "learn_only": False, "learned_only_steps": [], "plan_mismatches": 0}
    assert commit_verdict(rec, True)[0]
    rec["extra"]["bounded_staging"]["learned_only_steps"] = [0]
    ok, why = commit_verdict(rec, True)
    assert not ok and "learned-only" in why
    rec["extra"]["bounded_staging"] = {"enabled": True, "learn_only": True, "learned_only_steps": []}
    assert not commit_verdict(rec, True)[0]
