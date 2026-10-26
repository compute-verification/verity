"""The lifecycle over any scheme (here an NCP test instance): honest runs accepted, negative controls rejected, draws
work-weighted and reproducible, the lottery selecting by digest."""

from __future__ import annotations

import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.protocols.pouw.audit import Epoch, Workload, tile_refs
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape
from verity.ml.protocols.pouw.schemes import ncp

# the clear audit is a reference tool (benchmarks/pouw/clear), imported by name
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "benchmarks" / "pouw" / "clear"))
from clear_audit import Lottery, Prover, Sampled, Verifier, reference_kernel, tickets  # noqa: E402

BEACON, SOURCE = b"beacon round 1", b"verifier secret"


def int7(rng: random.Random, rows: int, k: int) -> list[bytes]:
    return [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(k)]) for _ in range(rows)]


def setup(shapes, seed=1, kernel=reference_kernel):
    s, rng = ncp.NCP(k_min=64, n_min=64), random.Random(seed)
    work = Workload(tuple(Matmul(i, sh, w) for i, (sh, w) in enumerate(shapes)))
    weights = {w: int7(rng, n, k) for w, (n, k) in work.weights().items()}
    acts = {u.index: int7(rng, u.shape.m, u.shape.k) for u in work.matmuls}
    prover = Prover(s, weights, kernel)
    return s, work, weights, acts, prover, Epoch.unverified(work, prover.weight_roots, BEACON)


def signed(rows):
    return [[x - 256 if x >= 128 else x for x in r] for r in rows]


def test_honest_run_is_exact_and_accepted():
    s, work, weights, acts, prover, epoch = setup([(Shape(32, 64, 64), 7), (Shape(16, 64, 64), 7)])
    tr = prover.prove(epoch, acts)
    for pos, u in enumerate(work.matmuls):
        assert tr.useful[pos] == [[sum(p * q for p, q in zip(a, b)) for b in signed(weights[u.weight])] for a in signed(acts[u.index])]
    audit = Verifier(s, epoch).audit(prover, tr, Sampled(6), SOURCE)
    assert audit.accepted and len(audit.selected) == 6
    assert audit.profile.work == sum(s.wref(u.shape) for u in work.matmuls)
    assert audit.profile.delta_s(Fraction(1, 2)) == Fraction(1, 64)


def test_one_wrong_tile_is_caught():
    def kernel(scheme, ctx, noise, a, b, tiles):
        out = reference_kernel(scheme, ctx, noise, a, b, tiles)
        out[2] = (bytes(32), out[2][1])
        return out

    s, work, weights, acts, cheat, epoch = setup([(Shape(32, 64, 64), 1)], kernel=kernel)
    tr = cheat.prove(epoch, acts)
    v = Verifier(s, epoch)
    verdicts = [v.check_tile(tr, cheat.open(epoch, t.id)) for t in v.refs]
    assert [x.ok for x in verdicts] == [True, True, False, True, True, True, True, True]
    assert verdicts[2].reason == "checked values"


def test_stale_salt_is_refused():
    s, work, weights, acts, prover, epoch = setup([(Shape(16, 64, 64), 1)])
    tr = prover.prove(Epoch(work, epoch.weight_roots, b"\x00" * 32), acts)
    assert Verifier(s, epoch).check_tile(tr, prover.open(epoch, 0)).reason == "activation opening"


def test_malformed_rows_are_refused():
    s, work, weights, acts, prover, epoch = setup([(Shape(16, 64, 64), 1)])
    tr = prover.prove(epoch, acts)
    opening = prover.open(epoch, 0)
    forged = type(opening)(opening.ref, (type(opening.a[0])(0, bytes([100]) * 64, opening.a[0].path),) + opening.a[1:], opening.b, opening.leaf)
    assert not Verifier(s, epoch).check_tile(tr, forged).ok
    acts[0] = [bytes([100]) + r[1:] for r in acts[0]]
    with pytest.raises(DomainError):
        prover.prove(epoch, acts)


def test_workload_rules():
    with pytest.raises(DomainError):
        Workload((Matmul(0, Shape(16, 64, 64), 1), Matmul(1, Shape(16, 128, 64), 1)))
    with pytest.raises(DomainError):
        Workload((Matmul(0, Shape(16, 64, 64), 1), Matmul(0, Shape(16, 64, 64), 2)))
    with pytest.raises(DomainError):
        Matmul(1 << 64, Shape(16, 64, 64), 0)


def test_draws_are_work_weighted_and_reproducible():
    s, work, weights, acts, prover, epoch = setup([(Shape(64, 64, 64), 1), (Shape(16, 64, 64), 2)])
    tr = prover.prove(epoch, acts)
    refs = tile_refs(s, work)
    assert len(set(tickets(s, work, refs))) == 1                # equal tiles of equal shapes carry equal work
    one = Sampled(64).select(s, epoch, tr, SOURCE, refs, prover.digest)
    assert one == Sampled(64).select(s, epoch, tr, SOURCE, refs, prover.digest)
    assert one != Sampled(64).select(s, epoch, tr, b"another verifier", refs, prover.digest)
    assert all(0 <= t < len(refs) for t in one)


def test_lottery_selects_by_digest():
    s, work, weights, acts, prover, epoch = setup([(Shape(32, 64, 64), 1)])
    tr = prover.prove(epoch, acts)
    v = Verifier(s, epoch)
    assert Lottery(1 << 256).select(s, epoch, tr, SOURCE, v.refs, prover.digest) == tuple(r.id for r in v.refs)
    assert Lottery(0).select(s, epoch, tr, SOURCE, v.refs, prover.digest) == ()
    audit = v.audit(prover, tr, Lottery(1 << 256), SOURCE)
    assert audit.accepted and audit.profile.delta_s(Fraction(1, 2)) is None
