"""Host-backend round trips and negatives for the CMT-1 committer (no torch needed)."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from verity_capture.experimental.commit import Committer, Context, StepLayout, open_value, verify_opening
from verity_capture.experimental.commit.engine import production_leaf_check
from verity_capture.experimental.commit.positions import LEAF_BYTES

PD = hashlib.sha256(b"program").digest()
CTX = Context("profile-x", "req-1", 7, "SmolLM2-135M")


def _synthetic(steps: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    layouts, streams = [], []
    for s in range(steps):
        L = StepLayout(s, 7 if s == 0 else 1)
        parts = []
        for layer in range(2):
            for cls, inst, dt, shape in (("gemm1/out/gemmcoordinate_v1/f2fpbf16", f"model.layers.{layer}.self_attn.qkv_proj", "bf16", (L.n_tokens, 30)),
                                         ("rope1/out/ropeout_v1/ropeout", f"model.layers.{layer}.self_attn.rotary_emb.q", "bf16", (L.n_tokens, 18)),
                                         ("silumul1/out/silumulbf16_v1/silumulbf16", f"model.layers.{layer}.mlp.act_fn", "bf16", (L.n_tokens, 46))):
                seg = L.add(cls, inst, dt, shape)
                parts.append((seg, rng.integers(0, 256, size=seg.nbytes, dtype=np.uint8)))
        seg = L.add("tokenselect1/out/tokenselect_v1/selecti32", "sampler", "i32", (1,))
        parts.append((seg, np.frombuffer(np.int32(1234 + s).tobytes(), dtype=np.uint8).copy()))
        L.freeze()
        buf = np.zeros(L.stream_bytes, dtype=np.uint8)
        for sg, data in parts:
            buf[sg.offset: sg.offset + sg.nbytes] = data
        layouts.append(L)
        streams.append(buf)
    return layouts, streams


def _commit(layouts, streams, **kw):
    c = Committer(PD, CTX, backend="host", **kw)
    for L, st in zip(layouts, streams, strict=True):
        c.commit_step(L, st.tobytes())
    root, pos = c.finish()
    return c, root, pos


def test_roundtrip_every_segment_first_middle_last():
    layouts, streams = _synthetic()
    c, root, pos = _commit(layouts, streams)
    n_ok = 0
    for s, (L, st) in enumerate(zip(layouts, streams, strict=True)):
        for seg in L.segments:
            for e in sorted({0, seg.numel // 2, seg.numel - 1}):
                op = open_value(c, s, seg.class_id, seg.instance, e, stream=st.tobytes())
                ok, why = verify_opening(root, pos, op)
                assert ok, why
                ib = 4 if seg.dtype == "i32" else 2
                assert op.value == st[seg.offset + e * ib: seg.offset + (e + 1) * ib].tobytes()
                assert production_leaf_check(op)
                n_ok += 1
    assert n_ok == 57  # 3 steps x (6 bf16 segments x 3 elements + 1 i32 segment x 1 element)


def test_streaming_root_is_deterministic_and_layout_bound():
    layouts, streams = _synthetic()
    _, r1, p1 = _commit(layouts, streams)
    _, r2, p2 = _commit(layouts, streams)
    assert r1 == r2 and p1["steps"] == p2["steps"]
    # a different context (request id) -> different root for identical streams
    c3 = Committer(PD, Context("profile-x", "req-2", 7, "SmolLM2-135M"))
    for L, st in zip(layouts, streams, strict=True):
        c3.commit_step(StepLayout.from_json(L.to_json()), st.tobytes())
    r3, _ = c3.finish()
    assert r3 != r1


def test_negative_altered_value_rejected():
    layouts, streams = _synthetic()
    c, root, pos = _commit(layouts, streams)
    seg = layouts[1].segments[2]
    op = open_value(c, 1, seg.class_id, seg.instance, 5, stream=streams[1].tobytes())
    assert verify_opening(root, pos, op)[0]
    leaf = bytearray(op.leaf)
    _, off, _ = layouts[1].rank(seg.class_id, seg.instance, 5)
    leaf[off] ^= 0x01
    bad = op.replace(leaf=bytes(leaf), value=bytes(leaf[off: off + 2]))
    ok, why = verify_opening(root, pos, bad)
    assert not ok and "step root" in why
    # value claim inconsistent with the (genuine) leaf
    bad2 = op.replace(value=bytes([op.value[0] ^ 1, op.value[1]]))
    assert not verify_opening(root, pos, bad2)[0]


def test_negative_wrong_position_rejected():
    layouts, streams = _synthetic()
    c, root, pos = _commit(layouts, streams)
    seg = layouts[0].segments[0]
    op = open_value(c, 0, seg.class_id, seg.instance, 3, stream=streams[0].tobytes())
    # (a) claim the same bytes belong to another element / another instance / another step
    for kw in ({"element": 4}, {"instance": layouts[0].segments[3].instance}, {"step": 1}, {"leaf_index": op.leaf_index + 1}):
        assert not verify_opening(root, pos, op.replace(**kw))[0], kw
    # (b) a genuine opening of a value elsewhere presented under this identity
    other = open_value(c, 0, seg.class_id, seg.instance, 3 + LEAF_BYTES // 2, stream=streams[0].tobytes())
    forged = other.replace(element=3)
    assert not verify_opening(root, pos, forged)[0]


def test_negative_reordered_or_substituted_leaf_rejected():
    layouts, streams = _synthetic()
    c, root, pos = _commit(layouts, streams)
    st = streams[2].copy()
    # swap two leaves in the stream and try to open from the swapped stream with the ORIGINAL tree paths
    a, b = 0, 1
    st[a * 64:(a + 1) * 64], st[b * 64:(b + 1) * 64] = streams[2][b * 64:(b + 1) * 64].copy(), streams[2][a * 64:(a + 1) * 64].copy()
    seg = layouts[2].segments[0]
    op = open_value(c, 2, seg.class_id, seg.instance, 0, stream=st.tobytes())
    assert not verify_opening(root, pos, op)[0]
    # substitute a whole step: commit a different stream, present its opening against the original root
    layouts2, streams2 = _synthetic(seed=1)
    c2, root2, pos2 = _commit(layouts2, streams2)
    op2 = open_value(c2, 2, seg.class_id, seg.instance, 0, stream=streams2[2].tobytes())
    assert verify_opening(root2, pos2, op2)[0]
    assert not verify_opening(root, pos, op2)[0]
    assert not verify_opening(root, pos2, op2)[0]


def test_negative_tampered_positions_rejected():
    layouts, streams = _synthetic()
    c, root, pos = _commit(layouts, streams)
    seg = layouts[0].segments[1]
    op = open_value(c, 0, seg.class_id, seg.instance, 1, stream=streams[0].tobytes())
    import copy
    p2 = copy.deepcopy(pos)
    p2["steps"][0]["segments"][1]["instance"], p2["steps"][0]["segments"][2]["instance"] = \
        p2["steps"][0]["segments"][2]["instance"], p2["steps"][0]["segments"][1]["instance"]
    # positions digest changes -> root rebinding fails even though the tree paths are genuine
    assert not verify_opening(root, p2, op)[0]


def test_wrong_program_digest_rejected():
    layouts, streams = _synthetic()
    c, root, pos = _commit(layouts, streams)
    seg = layouts[0].segments[0]
    op = open_value(c, 0, seg.class_id, seg.instance, 0, stream=streams[0].tobytes())
    assert verify_opening(root, pos, op, program_digest=PD)[0]
    assert not verify_opening(root, pos, op, program_digest=hashlib.sha256(b"other").digest())[0]


@pytest.mark.skipif(os.environ.get("CMT_TORCH") != "1", reason="torch backend: run on the pod with CMT_TORCH=1")
def test_torch_backend_matches_host():
    import torch
    layouts, streams = _synthetic()
    c_h, root_h, pos_h = _commit(layouts, streams)
    c_t = Committer(PD, CTX, backend="torch")
    for L, st in zip(layouts, streams, strict=True):
        c_t.commit_step(StepLayout.from_json(L.to_json()), torch.from_numpy(st.copy()))
    root_t, pos_t = c_t.finish()
    assert root_t == root_h
    for s in range(3):
        for lv_h, lv_t in zip(c_h.steps[s].levels_host, c_t.steps[s].levels_torch, strict=True):
            assert b"".join(lv_h) == bytes(lv_t.cpu().numpy().tobytes())


def test_verifier_class_matches_verify_opening():
    """`Verifier` (parse once) gives the same verdicts as the stateless `verify_opening` on good and bad openings."""
    import dataclasses
    from verity_capture.experimental.commit import Verifier
    layouts, streams = _synthetic()
    c, root, pos = _commit(layouts, streams)
    V = Verifier(root, pos)
    n = 0
    for s, (L, st) in enumerate(zip(layouts, streams, strict=True)):
        for seg in L.segments[::3]:
            op = open_value(c, s, seg.class_id, seg.instance, seg.numel - 1, stream=st.tobytes())
            assert V.verify(op) == verify_opening(root, pos, op) == (True, "ok")
            bad = dataclasses.replace(op, value=bytes(len(op.value)))
            assert V.verify(bad)[0] is False and verify_opening(root, pos, bad)[0] is False
            bad2 = dataclasses.replace(op, leaf_index=op.leaf_index + 1)
            assert V.verify(bad2) == verify_opening(root, pos, bad2)
            n += 1
    assert n >= 6
    assert Verifier(bytes(32), pos).verify(op) == (False, "root mismatch")
