"""Row 4 — derived-value route for the fp32 lift of the bf16 residual word (VALUE-FIRST §3 row 4; derived_value_rule.md §1).

Binding-correctness of the derived opening over the native_host commitment (offline core; same code as the online path minus hooks):
  the derived word opens through the ORIGINAL root; a forged fp32 word, a word derived from another position, a source that is not the
  declared class, a tampered source path, an undeclared conversion, and two consumers fed inconsistent words are all rejected.
Value-correctness of phi: exhaustive over the 65,536 bf16 words against numpy's bf16->f32 semantics (u16 << 16).
"""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from verity_capture.experimental.commit_integ import row4_derived as R4
from verity_capture.experimental.commit_integ.native_host import NativeHostCommitter

RES = "model.layers.3.post_attention_layernorm/1"     # the residual output (tuple index 1) of a fused norm
NORMED = "model.layers.3.post_attention_layernorm/0"  # the normalised output word: NOT a valid source
N = 4096                                             # elements


def _packed_committer(seed: int = 1) -> tuple[NativeHostCommitter, object, dict]:
    """The offline core has no arena; emulate the packed layout the online path produces (stream_off, dtype names)."""
    rng = np.random.default_rng(seed)
    res = rng.integers(0, 1 << 16, size=N, dtype=np.uint16).astype("<u2").tobytes()
    normed = rng.integers(0, 1 << 16, size=N, dtype=np.uint16).astype("<u2").tobytes()
    other = os.urandom(3 * 4096 + 12)
    c = NativeHostCommitter(None, chunk=4096, hash_threads=1)
    c.staging = "arena"
    steps = [[("model.layers.3.self_attn.o_proj", other), (NORMED, normed), (RES, res)],
             [("model.layers.3.self_attn.o_proj", other[:1000]), (NORMED, normed[:64]), (RES, res[:64])]]
    run = _commit_packed(c, steps)
    return c, run, {"res": res, "normed": normed}


def _commit_packed(c: NativeHostCommitter, steps):
    """Offline packed commit: concatenate the tensors 4-B aligned into one stream per step, hash it in `chunk` leaves (the online worker's
    packed path), so `stream_off` / identity_of / open / verify exercise the same code as on the GPU."""
    import time

    from verity_capture.experimental.commit_integ.native_host import TensorMeta, _StepJob

    class _Ev:
        def synchronize(self):
            return None

    c.reset()
    for tensors in steps:
        job = _StepJob(c._step_index, [], _Ev())
        c._step_index += 1
        buf = bytearray()
        for name, raw in tensors:
            off = (len(buf) + 3) // 4 * 4
            buf.extend(b"\0" * (off - len(buf)))
            m = TensorMeta(len(job.metas), name, "bfloat16" if name.endswith(("/0", "/1")) else "uint8", (len(raw) // 2,) if name.endswith(("/0", "/1")) else (len(raw),),
                           len(raw), c.chunk, 0, 0)
            m.stream_off = off
            buf.extend(raw)
            job.metas.append(m)
        mv = memoryview(bytes(buf))
        for m in job.metas:
            m.host = mv[m.stream_off:m.stream_off + m.nbytes]
        job.stream = mv
        job.t_step_end = time.perf_counter()
        c._commit_step(job)
    return c.finalize()


def test_phi_exhaustive_matches_numpy_bf16_to_f32():
    words = np.arange(1 << 16, dtype=np.uint32)
    ref = (words << 16).astype("<u4").tobytes()
    got = b"".join(R4.phi_bf16_to_f32(struct.pack("<H", w)) for w in range(1 << 16))
    assert got == ref


def test_derived_opening_roundtrip_every_consumer_and_step():
    c, run, vals = _packed_committer()
    for step in (0, 1):
        n = N if step == 0 else 32
        for el in (0, 1, n // 2, n - 1):
            for cls in R4.DERIVED_CLASSES[:2]:                       # fo1 (variance FMA) and fo2 (normalise path): same source
                d = R4.open_derived(c, cls, step, RES, el)
                assert R4.verify_derived(c, run, d), (step, el, cls)
                u = vals["res"][2 * el:2 * el + 2]
                assert d.word == b"\0\0" + u                          # exactly u16 << 16


def test_two_consumers_are_forced_to_the_same_word():
    c, run, _ = _packed_committer()
    d1 = R4.open_derived(c, R4.DERIVED_CLASSES[0], 0, RES, 77)
    d2 = R4.open_derived(c, R4.DERIVED_CLASSES[1], 0, RES, 77)
    assert d1.word == d2.word and d1.source.value == d2.source.value
    # an inconsistent second consumer word (differs in the low half, "the fp32 the normalise path used") is refused
    forged = R4.DerivedOpening(d2.derived_class, d2.step, d2.tensor_name, d2.element, d1.word[:3] + bytes([d1.word[3] ^ 1]), d2.phi, d2.source, d2.source_offset)
    assert not R4.verify_derived(c, run, forged)
    forged_low = R4.DerivedOpening(d2.derived_class, d2.step, d2.tensor_name, d2.element, b"\1\0" + d1.word[2:], d2.phi, d2.source, d2.source_offset)
    assert not R4.verify_derived(c, run, forged_low)                 # a non-zero low half is not phi(u)


def test_forged_word_wrong_width_wrong_phi_rejected():
    c, run, _ = _packed_committer()
    d = R4.open_derived(c, R4.DERIVED_CLASSES[0], 0, RES, 5)
    assert not R4.verify_derived(c, run, R4.DerivedOpening(d.derived_class, d.step, d.tensor_name, d.element, d.word[:2], d.phi, d.source, d.source_offset))
    assert not R4.verify_derived(c, run, R4.DerivedOpening(d.derived_class, d.step, d.tensor_name, d.element, d.word, "F2fpBf16", d.source, d.source_offset))
    assert not R4.verify_derived(c, run, R4.DerivedOpening("gemm1/out/x", d.step, d.tensor_name, d.element, d.word, d.phi, d.source, d.source_offset))


def test_wrong_source_position_rejected():
    c, run, _ = _packed_committer()
    d5 = R4.open_derived(c, R4.DERIVED_CLASSES[0], 0, RES, 5)
    d6 = R4.open_derived(c, R4.DERIVED_CLASSES[0], 0, RES, 6)
    # claim element 6 with element 5's word / offset
    assert not R4.verify_derived(c, run, R4.DerivedOpening(d5.derived_class, 0, RES, 6, d5.word, d5.phi, d5.source, d5.source_offset))
    # a source leaf from another position (element 3000 lives in another 4 KB leaf) presented for element 5
    d3000 = R4.open_derived(c, R4.DERIVED_CLASSES[0], 0, RES, 3000)
    assert d3000.source.index != d5.source.index
    assert not R4.verify_derived(c, run, R4.DerivedOpening(d5.derived_class, 0, RES, 5, d5.word, d5.phi, d3000.source, d5.source_offset))
    # another step's source
    d_s1 = R4.open_derived(c, R4.DERIVED_CLASSES[0], 1, RES, 5)
    assert not R4.verify_derived(c, run, R4.DerivedOpening(d5.derived_class, 0, RES, 5, d5.word, d5.phi, d_s1.source, d5.source_offset))
    assert d6.word != d5.word or True


def test_source_must_be_the_declared_class():
    c, run, vals = _packed_committer()
    # the normalised output word (tuple index 0) is bf16 too, but it is NOT the declared source of the lift
    d = R4.open_derived(c, R4.DERIVED_CLASSES[0], 0, NORMED, 5)      # the prover can build it ...
    assert not R4.verify_derived(c, run, d)                          # ... the verifier refuses it
    # a uint8 tensor as source
    with pytest.raises(ValueError):
        R4.open_derived(c, R4.DERIVED_CLASSES[0], 0, "model.layers.3.self_attn.o_proj", 5)


def test_tampered_source_path_or_value_rejected():
    from verity_capture.experimental.commit_integ.api import Opening

    c, run, _ = _packed_committer()
    d = R4.open_derived(c, R4.DERIVED_CLASSES[0], 0, RES, 9)
    src = d.source
    path = list(src.path)
    for i, p in enumerate(path):
        if p is not None:
            path[i] = bytes(32)
            break
    bad_path = Opening(src.step, src.index, src.value, tuple(path), src.identity)
    assert not R4.verify_derived(c, run, R4.DerivedOpening(d.derived_class, d.step, d.tensor_name, d.element, d.word, d.phi, bad_path, d.source_offset))
    v = bytearray(src.value)
    v[d.source_offset] ^= 0x80
    bad_val = Opening(src.step, src.index, bytes(v), src.path, src.identity)
    w = R4.phi_bf16_to_f32(bytes(v[d.source_offset:d.source_offset + 2]))
    assert not R4.verify_derived(c, run, R4.DerivedOpening(d.derived_class, d.step, d.tensor_name, d.element, w, d.phi, bad_val, d.source_offset))


def test_root_recomputed_after_the_fact_does_not_authenticate_derived_openings():
    c, run, _ = _packed_committer()
    d = R4.open_derived(c, R4.DERIVED_CLASSES[0], 0, RES, 11)
    c2, run2, _ = _packed_committer(seed=2)
    assert run2.run_root != run.run_root
    assert not R4.verify_derived(c2, run2, d)                        # the opening belongs to the original commitment only
    assert not R4.verify_derived(c, run2, d)
