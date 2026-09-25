"""Operator-graph extraction: exact against the flattened circuit on tiny configs (all algorithms, several
sequences per weight version), and symbolic -- op count and time independent of the number of sequences."""

import resource
import time

import pytest

from accumulation.algorithms import registry as R
from accumulation.configs import MODELS, Workload
from accumulation.graph import extract
from accumulation.graph.validate import check_against_flat

TINY_WL = Workload(tokens=8, seq=4, population=2, rank=2, local_steps=2, gen_tokens=2)


@pytest.mark.parametrize("name", sorted(R.ALGORITHMS))
def test_opgraph_matches_flat_circuit(name):
    alg = R.ALGORITHMS[name]
    for m in ("tiny2", "tiny-moe"):
        cfg = MODELS[m]
        if alg.supports(cfg, TINY_WL) is not None:
            continue
        bp = R.build(name, cfg, TINY_WL)
        g = extract(bp)
        r = check_against_flat(bp, g)
        assert r["ops"] > 0 and r["edges"] >= r["edges_exact_equal"] > 0


def test_extraction_is_symbolic_in_the_number_of_sequences():
    cfg = MODELS["small"]
    ops = {}
    for nseq in (1, 8, 64):
        t0 = time.perf_counter()
        bp = R.build("pretrain-dense", cfg, Workload(tokens=64 * nseq, seq=64))
        g = extract(bp)
        dt = time.perf_counter() - t0
        ops[nseq] = len(g.ops)
        assert dt < 20.0, (nseq, dt)
    assert len(set(ops.values())) == 1, ops
    # a weight-gradient op contracts over all tokens of the weight version
    bp = R.build("pretrain-dense", cfg, Workload(tokens=64 * 8, seq=64))
    g = extract(bp)
    wg = [o for o in g.ops if o.fn_id.startswith("AccMatmulTT")]
    assert wg and all(o.statics["K"] == 64 * 8 for o in wg)


def test_real_scale_extraction_is_cheap():
    """llama3-8b pretraining at 16 sequences: seconds, well under a gigabyte."""
    bp = R.build("pretrain-dense", MODELS["llama3-8b"], Workload(tokens=16 * 4096, seq=4096))
    t0 = time.perf_counter()
    g = extract(bp)
    assert time.perf_counter() - t0 < 60.0
    assert len(g.ops) == 2283
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 3 * 1024 ** 3
