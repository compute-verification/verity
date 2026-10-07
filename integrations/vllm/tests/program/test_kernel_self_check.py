"""Every (Definition, kernel) pair registered through `verity_vllm.program.kernels.kernel_registry` — the replay row
kernels (`rows`) and the twins — against the reference by core `verity.ml.circuits.evaluation.self_check`, on each of the kernel's
check specialisations: no mismatch, and not every instance declined."""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from verity.ml.circuits.evaluation import kernels, self_check
from verity_vllm.program.kernels import kernel_registry as KR
from verity_vllm.program.kernels import rows, twins

N = 6
PAIRS = [(r.key, r.kernel) for r in KR.registered()]


def test_every_row_family_and_twin_is_registered_with_core():
    assert {k for k, kern in PAIRS if kern == rows.KERNEL} == set(rows.ROWS)
    assert {k for k, kern in PAIRS if kern == twins.KERNEL} == set(twins.FAMILIES)
    for key, kernel in PAIRS:
        assert kernels(key)[kernel] is next(r.core for r in KR.registered() if (r.key, r.kernel) == (key, kernel))


@pytest.mark.parametrize("key,kernel", PAIRS, ids=[f"{k}-{kern}" for k, kern in PAIRS])
def test_kernel_matches_the_reference(key, kernel):
    reg = next(r for r in KR.registered() if (r.key, r.kernel) == (key, kernel))
    targets = reg.check_targets(np.random.default_rng(zlib.crc32(f"{key}/{kernel}".encode())))
    assert targets, f"{key}: no check specialisations"
    for t in targets:
        assert KR.covers(t, kernel) is None, (t.id, KR.covers(t, kernel))
        sc = self_check(t, kernel=kernel, n=N, seed=0)
        assert sc.ok, str(sc)
        assert sc.declined < sc.n, f"{t.id}: {kernel} declined all {sc.n} instances"


def test_attention_kernels_on_scores_below_two_to_the_minus_63(monkeypatch):
    """Edge vectors for the attention kernels' self-check: scores near +-2^-66 put the P and rescale ex2 inputs at
    2^-87 <= |x| < 2^-63, where the C rule's shift count reached 64 (undefined; the host took it mod 64) and MUFU.EX2 reads
    T[0] = 1.0.  The twin and the row kernel equal the reference there, and the reference equals the same instance with
    q = 0, whose every ex2 input is 0."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.kernels import derived_rows as DR
    from verity_vllm.program.registry import b1

    statics = {"T": 20, "NH": 2, "KVH": 1, "D": 16, "BN": 16}
    target = bind(b1.AttentionV3, **statics)
    bf16 = lambda a: (np.asarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).reshape(-1)  # noqa: E731
    k, q = np.zeros((20, 16)), np.zeros((2, 16))
    k[:, 0] = 2.0 ** -33 * (1 + np.arange(19, -1, -1) / 64)   # head 0's max is in block 0, visited last: a tiny rescale input
    q[:, 0] = [2.0 ** -33, -(2.0 ** -33)]
    v = np.random.default_rng(3).standard_normal((20, 16))
    tiny, zero = [bf16(q), bf16(k), bf16(v)], [bf16(0 * q), bf16(k), bf16(v)]

    seen: list[np.ndarray] = []
    ex2 = DR.mufu_ex2_ftz
    monkeypatch.setattr(DR, "mufu_ex2_ftz", lambda x, table=None: seen.append(np.asarray(x).reshape(-1)) or ex2(x, table))
    got = rows.ROWS["Attention_v3"].fn(statics, tiny)["out"]
    monkeypatch.undo()
    xe = (np.concatenate(seen) >> 23) & 0xFF
    assert np.count_nonzero((xe >= 40) & (xe <= 63)) > 0 and not np.any((xe >= 104) & (xe < 255))

    ref = [evaluate(target, *(a.tolist() for a in inst)) for inst in (tiny, zero)]
    assert ref[0] == ref[1] and got.tolist() == list(ref[0])
    for kernel in (twins.KERNEL, rows.KERNEL):
        sc = self_check(target, kernel=kernel, inputs=[np.stack(pair) for pair in zip(tiny, zero)])
        assert sc.ok and sc.declined == 0, str(sc)
