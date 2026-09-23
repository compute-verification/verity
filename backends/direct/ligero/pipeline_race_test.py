"""Regression test for the pipelined prover's cross-stream races (lane pipe-race).

``pipeline.prove_many`` runs up to ``depth`` sub-batches concurrently on their own CUDA streams; anything a stage
writes that is NOT per slot is shared by in-flight sub-batches.  The encoder's per-CTA global scratch (``encode_simt``:
CF_GLOBAL ``cf_scratch``, STAGE ``stg``) was such a buffer -- one encoder object per (l, n, t_pad, device), so two
encodes in flight on different streams corrupted each other's codewords and the verifier rejected honest depth-4 proofs
(``proximity test failed`` / ``chain constraints failed``; RTX 4090, fp8-ada, 4096 VUs, ``--batch 16384``).

The test proves a set of sub-batches at depth 4 (and 3) several times and verifies each proof with the prover's own
``verify_vus`` -- the same check the runners assert on -- and, without ZK (deterministic transcript: no mask key), checks
that the depth-1 and depth-4 proofs of the same coins are BYTE-IDENTICAL (the pipeline must not change the protocol).
GPU only (skipped on CPU); as a script it is the reproduction harness:

    python -m backends.direct.ligero.pipeline_race_test --relation fp8-ada --total-vus 4096 --batch 16384 --depth 4 --runs 5 [--no-zk]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from . import protocol as protocol_mod
from .relchain import RelationChainRunner, instances
from .relations import relation
from .serialize import proof_bytes

CACHE = os.environ.get("LIGERO_INSTANCES_CACHE")


def fixed_coins(i: int, tag: bytes = b"pipe-race") -> protocol_mod.Coins:
    """Deterministic step-0 coins for sub-batch ``i`` (the byte-identity check needs the same coins at every depth)."""
    h = hashlib.shake_256(tag + i.to_bytes(4, "little")).digest(4 * protocol_mod.COIN_BYTES)
    c = protocol_mod.COIN_BYTES
    return protocol_mod.Coins(h[:c], h[c:2 * c], h[2 * c:3 * c], h[3 * c:])


def setup(rel_name: str, total_vus: int, batch: int, zk: bool, device: str, procs: int = 8):
    rel = relation(rel_name)
    per_proof = batch // rel.steps
    n_proofs = -(-total_vus // per_proof)
    R = RelationChainRunner(rel, device, -128.0, 2, n_proofs=n_proofs, zk=zk, mode=protocol_mod.MODE_INTERACTIVE)
    data = instances(rel, total_vus, procs=procs, cache=CACHE)
    subs = [(lo, min(lo + per_proof, total_vus)) for lo in range(0, total_vus, per_proof)]
    l = R.layout(per_proof).l
    return R, data, subs, l


def prove_verify(R, data, subs, l, depth: int, coins_factory) -> tuple[list[str], list[bytes], float]:
    """Prove every sub-batch ``depth`` deep, verify each; ``(rejections, proof digests, wall)``."""
    many, wall = R.prove_vus_many([data[lo:hi] for lo, hi in subs], coins_factory, depth=depth, min_l=l)
    rejected, digests = [], []
    for (lo, hi), (V, proof, pubs, lay_i) in zip(subs, many):
        ok, why = R.verify_vus(proof, pubs, lay_i, coins=V)
        if not ok:
            rejected.append(f"[{lo}, {hi}): {why}")
        digests.append(hashlib.sha256(proof_bytes(proof)).digest())
    return rejected, digests, wall


def campaign(rel_name: str, total_vus: int, batch: int, zk: bool, depths: tuple[int, ...], runs: int, device: str = "cuda",
             log=print, identity: bool = True) -> dict:
    R, data, subs, l = setup(rel_name, total_vus, batch, zk, device)
    log(f"{rel_name}: {total_vus} VUs as {len(subs)} sub-batches of <= {subs[0][1] - subs[0][0]} (l = {l}), zk {zk}")
    # warm-up (compile, graph capture per slot) is untimed and unverified here: one pass at the deepest depth
    prove_verify(R, data, subs, l, max(depths), lambda i: protocol_mod.Coins.sample())
    table: dict = {}
    ref = None
    for depth in depths:
        rej_total, n_total, walls = 0, 0, []
        for run in range(runs):
            rej, digests, wall = prove_verify(R, data, subs, l, depth, (lambda i: fixed_coins(i)) if not zk else
                                              (lambda i: protocol_mod.Coins.sample()))
            rej_total += len(rej)
            n_total += len(subs)
            walls.append(wall)
            log(f"  depth {depth} run {run + 1}/{runs}: {len(subs) - len(rej)}/{len(subs)} accepted, wall {wall:.3f}s"
                + (f"  REJECTED: {'; '.join(rej)}" if rej else ""))
            if not zk and identity:
                if ref is None:
                    ref = digests
                elif digests != ref:
                    bad = [i for i, (a, b) in enumerate(zip(digests, ref)) if a != b]
                    log(f"  depth {depth} run {run + 1}: proof bytes differ from depth {depths[0]} run 1 at sub-batches {bad}")
                    table.setdefault("identity_mismatch", []).append((depth, run + 1, bad))
        table[depth] = {"rejected": rej_total, "sub_batches": n_total, "wall_median": sorted(walls)[len(walls) // 2]}
        log(f"depth {depth}: {rej_total} rejected of {n_total} sub-batches; median wall {table[depth]['wall_median']:.3f}s")
    return table


def encoder_concurrent_trials(l: int, R: int, trials: int, t_pad: int, device: str = "cuda") -> tuple[int, int]:
    """The SAME encoder object launched on two streams at once (two in-flight sub-batches) vs. the sequential reference:
    ``(trials with a wrong codeword, wrong rows)``.  On the unfixed encoder every trial at l = 16384 lost ~1 % of the rows
    (RTX 4090: 20/20 trials, 1886 of 150760 rows)."""
    from .encode_simt import encoder_for
    from .field import P
    enc = encoder_for(l, 4 * l, t_pad, device)
    g = torch.Generator(device=device).manual_seed(1)
    rows = [torch.randint(0, P, (R, l), dtype=torch.int64, device=device, generator=g) for _ in range(2)]
    masks = [torch.randint(0, P, (R, t_pad), dtype=torch.int64, device=device, generator=g) for _ in range(2)] if t_pad else [None, None]
    ref = [enc.encode(r, m)[0].clone() for r, m in zip(rows, masks)]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    bad_trials = bad_rows = 0
    for _ in range(trials):
        outs = []
        for i in range(2):
            with torch.cuda.stream(streams[i]):
                outs.append(enc.encode(rows[i], masks[i])[0])
        torch.cuda.synchronize()
        nb = sum(int((o != r).any(1).sum()) for o, r in zip(outs, ref))
        bad_trials += bool(nb)
        bad_rows += nb
    return bad_trials, bad_rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pipelined prover races are a CUDA-stream matter")
@pytest.mark.parametrize("l,t_pad", [(16384, 0), (16384, 256), (8192, 256), (4096, 0)])
def test_encoder_scratch_not_shared_across_streams(l, t_pad):
    """Two concurrent encodes of one encoder object (different streams) must both be bit-exact with the sequential
    reference: the per-CTA scratch is per launch, never shared (the root cause of the depth-4 rejections)."""
    from .encode_simt import available
    if not available():
        pytest.skip("no cupy / NVRTC")
    bad_trials, bad_rows = encoder_concurrent_trials(l, 3769, 10, t_pad)
    assert bad_trials == 0, f"l={l} t_pad={t_pad}: {bad_trials}/10 concurrent trials wrote wrong codewords ({bad_rows} rows)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pipelined prover races are a CUDA-stream matter")
@pytest.mark.parametrize("zk", [True, False])
def test_pipeline_depth4_accepted_and_byte_identical(zk):
    """8 sub-batches of fp8-ada proved 4 deep (twice) and 3 deep: every proof accepted; without ZK the depth-4 bytes equal
    the depth-1 bytes (same coins)."""
    rel_name = os.environ.get("LIGERO_PIPELINE_TEST_RELATION", "fp8-ada")
    rel = relation(rel_name)
    steps = rel.steps
    batch = 16384
    per_proof = batch // steps
    total_vus = 8 * per_proof                    # 8 sub-batches at l = 16384 (the failing operating point)
    table = campaign(rel_name, total_vus, batch, zk, depths=(1, 4, 3) if not zk else (4, 3), runs=2, log=lambda *a: None)
    for depth, row in table.items():
        if isinstance(depth, int):
            assert row["rejected"] == 0, f"depth {depth}: {row['rejected']} of {row['sub_batches']} sub-batches rejected"
    assert "identity_mismatch" not in table, f"pipelined transcript differs from the sequential one: {table['identity_mismatch']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--total-vus", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--depth", type=int, nargs="+", default=[4])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--no-zk", action="store_true")
    ap.add_argument("--device", default="cuda")
    ns = ap.parse_args(argv)
    t0 = time.perf_counter()
    table = campaign(ns.relation, ns.total_vus, ns.batch, not ns.no_zk, tuple(ns.depth), ns.runs, ns.device)
    print(f"TABLE relation={ns.relation} zk={not ns.no_zk} batch={ns.batch} vus={ns.total_vus}: "
          + "; ".join(f"depth {d}: {r['rejected']}/{r['sub_batches']} rejected, wall {r['wall_median']:.3f}s"
                      for d, r in table.items() if isinstance(d, int))
          + (f"; IDENTITY MISMATCH {table['identity_mismatch']}" if "identity_mismatch" in table else "; identity ok" if ns.no_zk else "")
          + f"  ({time.perf_counter() - t0:.0f}s)")
    bad = any(r["rejected"] for d, r in table.items() if isinstance(d, int)) or "identity_mismatch" in table
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
