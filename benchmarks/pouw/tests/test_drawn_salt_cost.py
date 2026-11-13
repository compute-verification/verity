"""`drawn_salt_cost.py`'s batch counts against the scalar references they stand for: the products cut at both ends and the
dead pairs (`pearl_c_debit.word_mend`), the products cut on each word's chain at its own codes (`pearl_c_debit.chain_cuts`),
the centre dead pairs from the scalar forming atom, and the debit now (`clear_work.tile_cap`), at the H100's step and
sm_120's."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
import drawn_salt_cost as DS  # noqa: E402
from verity.definitions.silicon import fp32  # noqa: E402
from verity.protocols.pouw.prover.schemes import pearl_c_debit as D  # noqa: E402
from verity.protocols.pouw.prover.schemes import pearl_c_work as PW  # noqa: E402
from verity.protocols.pouw.prover.schemes import pearl_kw as P  # noqa: E402
from verity.protocols.pouw.verifier.protocol import Matmul, Shape  # noqa: E402
from verity_vllm.protocols.pouw import pearl_c as PD  # noqa: E402

K = 384
SCHEMES = [s for s in PD.SCHEMES if s.startswith(("pearl-c-h100-v1", "pearl-c-sm120-v1"))]


def _codes(rng, rows: int) -> list[list[int]]:
    """E4M3 codes with zeros, subnormals and wide exponents in one atom, so that the adder cuts some products."""
    out = rng.integers(0, 0x7F, (rows, K)) | (rng.integers(0, 2, (rows, K)) << 7)
    out[rng.random((rows, K)) < 0.1] = 0
    small = rng.random((rows, K)) < 0.2
    out[small] = rng.integers(1, 16, small.sum())
    return out.tolist()


def _formed(rng, rows: int) -> list[tuple[int, list[int], int]]:
    one = 0x3F800000
    mags = 10.0 ** rng.uniform(-3, 5, (rows, K)) * rng.choice((-1, 1), (rows, K))
    beta = rng.uniform(0.01, 2.0, rows)
    return [(one, [int(w) for w in row.astype(np.float32).view(np.uint32)], int(np.float32(b).view(np.uint32)))
            for row, b in zip(mags, beta)]


def _centre(alpha: int, x: list[int], device) -> list[int]:
    zero = [0] * device.atom.k
    return [P.f32_to_fp8(P.f32_of_bits(device.atom.step(fp32.mul(alpha, w), zero, zero))) for w in x]


def _agree(c: int, d: int) -> bool:
    return D.ends_agree(c, d)


@pytest.mark.parametrize("name", SCHEMES)
def test_the_batch_counts_are_the_scalar_references(name):
    s = PD.scheme_of(name)
    d = s.device
    rng = np.random.default_rng(11)
    a, b = _codes(rng, 3), _codes(rng, 4)
    fa, fb = _formed(rng, 3), _formed(rng, 4)
    for i in range(2):
        b[i] = _centre(fb[i][0], fb[i][1], d)
        a[i][:K // 2] = _centre(fa[i][0], fa[i][1], d)[:K // 2]
    counts = DS._word_counts(np.array(a), np.array(b), DS._references(fa, d), DS._references(fb, d), d)
    ends_a, ends_b = [D.row_ends(*f, d) for f in fa], [D.row_ends(*f, d) for f in fb]
    ca, cb = [_centre(f[0], f[1], d) for f in fa], [_centre(f[0], f[1], d) for f in fb]
    for i in range(3):
        for j in range(4):
            cuts = D.chain_cuts(a[i], b[j], d)
            assert counts["ends"][i, j] == D.word_mend(ends_a[i], ends_b[j], d)
            assert counts["cut_drawn"][i, j] == sum(cuts)
            dead = [_agree(a[i][l], ca[i][l]) and _agree(b[j][l], cb[j][l]) for l in range(K)]
            assert counts["dead_centre"][i, j] == sum(dead)
            cp, cm = D.chain_cuts(ends_a[i][0], ends_b[j][0], d), D.chain_cuts(ends_a[i][1], ends_b[j][1], d)
            both = [(ends_a[i][2][l] and ends_b[j][2][l]) or (cp[l] and cm[l]) for l in range(K)]
            assert counts["mends_centre"][i, j] == sum(1 for l in range(K) if both[l] or cuts[l] or dead[l])
            assert counts["dead_any"][i, j] >= counts["dead_centre"][i, j]
    assert counts["cut_drawn"].sum() > 0 and counts["dead_centre"].sum() > 0

    record = PW.UnitRecord(Matmul(0, Shape(3, K, 4), 0), 3, K, (), (), (), ())
    side = lambda codes, formed: SimpleNamespace(codes=codes, alpha=[f[0] for f in formed], x=[f[1] for f in formed],
                                                 beta=[f[2] for f in formed])
    got = DS.measure(s, record, SimpleNamespace(a=side(a, fa), b=side(b, fb)), [0, 1, 2], [0, 1, 2, 3], True)
    assert got["cross_checked"] and got["mends_centre"] == int(counts["mends_centre"].sum())
    assert got["debit_centre"] - got["debit_now"] == got["mends_centre"] - got["ends"]
