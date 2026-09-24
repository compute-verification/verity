"""``fp4-nvf4+poseidon2`` through the pipelined prover (``HashedRelationRunner.prove_vus_many``, ``--pipeline N``).

The hashed runner's stage generator computes the base hints (the fp4 device hint graph, one instance per pipeline slot)
and the hash hint rows inside the prover's clock; the statement (public table, leaf indices, chain-end words) is built
before it.  Every pipelined proof must be accepted by the prover's own ``verify_vus`` (decode + sponges + the Merkle
multiproofs), and without ZK (no mask key: a deterministic transcript) its bytes and statement must equal the
sequential ``prove_vus`` proof under the same coins -- the pipeline must not change the protocol.  GPU only.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from .. import protocol as protocol_mod  # noqa: E402
from ..relchain import HashedRelationRunner, instances  # noqa: E402
from ..serialize import proof_bytes  # noqa: E402
from .hashed import FP4_HASHED  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the pipelined prover is a CUDA-stream matter")

BATCH = 1024                                   # l = 1024: 42 VUs per sub-batch
N_SUBS = 5                                     # 4 full sub-batches + a ragged one (the bench's shape)


def fixed_coins(i: int, tag: bytes = b"fp4-hashed-pipeline") -> protocol_mod.Coins:
    h = hashlib.shake_256(tag + i.to_bytes(4, "little")).digest(4 * protocol_mod.COIN_BYTES)
    c = protocol_mod.COIN_BYTES
    return protocol_mod.Coins(h[:c], h[c:2 * c], h[2 * c:3 * c], h[3 * c:])


def _setup(zk: bool):
    per_proof = BATCH // FP4_HASHED.steps
    total = per_proof * (N_SUBS - 1) + 3
    R = HashedRelationRunner(FP4_HASHED, "cuda", -128.0, 2, n_proofs=N_SUBS, zk=zk, mode=protocol_mod.MODE_INTERACTIVE)
    data = instances(FP4_HASHED, total)
    R.commit_vus(data)
    subs = [(lo, min(lo + per_proof, total)) for lo in range(0, total, per_proof)]
    return R, data, subs, R.layout(per_proof).l


def _many(R, data, subs, l, depth, coins):
    return R.prove_vus_many([data[lo:hi] for lo, hi in subs], coins, depth=depth, min_l=l,
                            vu_ids_list=[range(lo, hi) for lo, hi in subs])[0]


def test_pipelined_hashed_accepted_and_byte_identical():
    R, data, subs, l = _setup(zk=False)
    seq = []
    for i, (lo, hi) in enumerate(subs):
        proof, pubs, lay = R.prove_vus(data[lo:hi], check=False, min_l=l, coins=fixed_coins(i), vu_ids=range(lo, hi))
        seq.append((hashlib.sha256(proof_bytes(proof)).digest(), pubs))
    _many(R, data, subs, l, 4, lambda i: protocol_mod.Coins.sample())          # warm-up: every slot captures its graphs
    for depth in (4, 2, 4):
        for i, ((lo, hi), (V, proof, pubs, lay)) in enumerate(zip(subs, _many(R, data, subs, l, depth, fixed_coins))):
            ok, why = R.verify_vus(proof, pubs, lay, coins=V)
            assert ok, f"depth {depth} sub-batch [{lo}, {hi}): {why}"
            d_seq, p_seq = seq[i]
            assert np.array_equal(pubs[0], p_seq[0]) and np.array_equal(pubs[1], p_seq[1]), f"depth {depth} [{lo}, {hi}): statement differs"
            assert hashlib.sha256(proof_bytes(proof)).digest() == d_seq, f"depth {depth} [{lo}, {hi}): proof bytes differ from sequential"
            assert proof.timings["hints_host"] > 0 and "auth_openings" in proof.timings


def test_pipelined_hashed_zk_accepted():
    R, data, subs, l = _setup(zk=True)
    _many(R, data, subs, l, 4, lambda i: protocol_mod.Coins.sample())
    for (lo, hi), (V, proof, pubs, lay) in zip(subs, _many(R, data, subs, l, 4, lambda i: protocol_mod.Coins.sample())):
        ok, why = R.verify_vus(proof, pubs, lay, coins=V)
        assert ok, f"sub-batch [{lo}, {hi}): {why}"
