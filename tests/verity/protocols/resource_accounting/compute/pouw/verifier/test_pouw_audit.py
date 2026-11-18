"""The lifecycle over any scheme (here an NCP test instance): honest runs accepted, negative controls rejected, samples
work-weighted and reproducible, the lottery selecting by digest."""

from __future__ import annotations

import random
from fractions import Fraction
from types import SimpleNamespace

import pytest

from verity.commitments import MerkleTree
from verity.errors import ProtocolError
from verity.protocols.resource_accounting.compute.pouw.prover.clear.clear_audit import Lottery, Prover, Sampled, Verifier, reference_kernel, tickets
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import ncp
from verity.protocols.resource_accounting.compute.pouw.verifier.audit import (HM96_SALT, TILE_SCHEMA, Epoch, Workload, activation_binding,
                                              commit_rows, fresh_salts, tile_refs, transcript_domain, transcript_root,
                                              weight_binding)
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import DomainError, Matmul, Shape

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


def test_the_sample_is_work_weighted_and_reproducible():
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


def test_weight_and_activation_roots_are_matched_only_with_the_committers_salts():
    """Two weight sets committed under the committer's fresh salts give roots no one matches without the salts: not the
    plain SHA-256 or SHA-512 tree of either candidate set, nor either set under other salts; the salts' holder reproduces
    its own, and two commits of the same rows under fresh salts differ. `commit_rows` refuses a plain tree of values (only
    rows public by structure take ``public``), an hm96 tree with no salts, a salt short of a row and a salt of another
    size, so no PoUW weight or activation root is recomputable from candidate values."""
    rng = random.Random(7)
    sets = [int7(rng, 8, 64) for _ in range(2)]
    salts, others = fresh_salts(8), [fresh_salts(8), [bytes(HM96_SALT)] * 8]
    binding = weight_binding(3)

    def root(rows, ys):
        return commit_rows(binding, -2, rows, salts=ys).commitment.root

    roots = [root(rows, salts) for rows in sets]
    assert roots[0] != roots[1] and [root(rows, salts) for rows in sets] == roots
    assert root(sets[0], fresh_salts(8)) != roots[0]
    guesses = {root(rows, ys) for rows in sets for ys in others}
    guesses |= {commit_rows(binding, -2, rows, h, public=True).commitment.root for rows in sets for h in ("sha256", "sha512")}
    assert not guesses & set(roots)
    acts = [commit_rows(activation_binding(BEACON, 5), 5, rows, salts=ys).commitment.root
            for rows in sets for ys in (salts, *others)]
    assert len(set(acts)) == len(acts)
    for h in ("sha256", "sha512"):
        with pytest.raises(ProtocolError, match="not hiding"):
            commit_rows(binding, -2, sets[0], h)
    with pytest.raises(ProtocolError, match="takes the prover's salts"):
        commit_rows(binding, -2, sets[0])
    with pytest.raises(ProtocolError, match="takes 8 salts, not 7"):
        commit_rows(binding, -2, sets[0], salts=salts[:7])
    with pytest.raises(ProtocolError, match=f"row 2's salt is 64 bytes, not {HM96_SALT}"):
        commit_rows(binding, -2, sets[0], salts=[*salts[:2], bytes(64), *salts[3:]])


def test_the_transcript_root_salts_each_digest_as_commit_rows_salts_a_row():
    """`transcript_root` is the transcript domain's tree of the tiles' digests, each followed by its salt, and refuses
    what `commit_rows` refuses."""
    digests, ys = [bytes([i]) * 64 for i in range(5)], fresh_salts(5)
    salt, wr = bytes(range(32)), bytes(64)
    epoch = SimpleNamespace(salt=salt, weights_root=wr)
    want = MerkleTree(transcript_domain(epoch, 5), {i: d + y for i, (d, y) in enumerate(zip(digests, ys))},
                      lambda _: TILE_SCHEMA).commitment.root
    assert transcript_root(salt, wr, digests, salts=ys) == want != transcript_root(salt, wr, digests, salts=fresh_salts(5))
    with pytest.raises(ProtocolError, match="takes the prover's salts"):
        transcript_root(salt, wr, digests)
