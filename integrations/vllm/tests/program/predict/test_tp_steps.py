"""verity_vllm.program.predict.tp_steps: a node's advice compiled at every rank of φ.tp's world and composed (`compile_tp`).

The dense toy at TP2 compiles in seconds; GLM-5.3 at TP8, class G's registration, is the opt-in test, with a one-token prompt
and a two-token one, whose TP window (`pipeline.tp.wires`) places every collective at a grain C-Flock proves: each rank's
coordinates gathered into 96-word rows, one combine unit a chunk, and the chunks gathered into the call's row."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests.pipeline.test_tp_wires import in_order, parts_bits, peer_bits, proof_blocks, sent_at_own_slices
from tests.program.predict.test_steps import GREEDY, _class_g, _mixed, _registered, _session
from verity_vllm.pipeline.tp import wires as TW
from verity_vllm.program.predict.steps import compile_steps
from verity_vllm.program.predict.tp_steps import at_rank, compile_tp

from verity.ml.circuits.codec import descriptor_id
from verity.ml.windows import interface as I

ROOT = Path(__file__).resolve().parents[3]
ONCE = {**GREEDY, "max_tokens": 2, "ignore_eos": True}


def test_compile_tp_compiles_the_advice_at_every_rank_and_composes_them():
    """At TP2 `compile_tp` is `compile_steps` at each rank, `at_rank`'s registration that rank's own: each rank's workload and
    request Programs are its own compile's, a request shape binds once, and the composition's record holds each request's
    binding; a world of one rank is refused."""
    advice, ingress = _mixed()
    reg = _registered(tp={"world": 2, "rank": 1})
    c = compile_tp(reg, advice, ingress)
    for r in (0, 1):
        assert at_rank(reg, r) == _registered(tp={"world": 2, "rank": r})
        one = compile_steps(at_rank(reg, r), advice, ingress)
        assert c.ranks[r] == one.workload.digest and c.requests[r] == tuple(u.digest for u in one.requests)
    assert c.bindings[0] is c.bindings[2] and [len(b.sites) for b in c.bindings] == [45, 30, 45]
    assert c.workload.ranks == c.ranks and c.workload.record["bindings"] == [b.digest for b in c.bindings]
    with pytest.raises(ValueError, match="φ.tp's world is 1"):
        compile_tp(_registered(), advice, ingress)


def class_g_tp(hf_home: Path, layers: int | None = None) -> dict:
    """GLM-5.3 at TP8 under class G's registration (with ``layers``, its first ``layers`` decoder layers), a one-token prompt
    and a two-token one prefilled in one step, each decoding once: `compile_tp`'s digests and sites, and the TP window of
    both requests' sites: its units by template, its input bits, the bits of every rank's partial wired into what reads
    every rank's and of every combine's operand, the reads of a combine's output left free (by kind, `Sites.unwired` without
    the request, site, rank and node), the window's own units C-Flock's composer refuses (`proof_blocks`), whether each
    rank's sent partial is its own segment at every site's slice, and the first wire running backward in the public order
    (None)."""
    advice, ingress = _session([(1, 2, ONCE, [5]), (2, 2, ONCE, [5, 6])], [[(0, 0, 1), (1, 0, 2)], [(0, 1, 1), (1, 2, 1)]])
    c = compile_tp(_class_g(hf_home, layers=layers), advice, ingress)
    live = [b for b in c.bindings if b is not None]
    sites = TW.sites_circuit(c.bindings, "class-g")
    by = {descriptor_id(fn): fn for fn, _ in sites.circuit.templates}
    try:
        in_order(sites)
        backward = None
    except AssertionError as e:
        backward = str(e)[:2000]
    kinds = Counter(re.sub(r"\[\d+, \d+\)", "[..)", re.sub(r"request \d+'s collective \d+ \[\d+, \d+\): rank \d+'s node \d+ ",
                                                            "", u)) for u in sites.unwired)
    return {"workload": c.workload.digest, "ranks": list(c.ranks), "bindings": [x.digest for x in c.bindings],
            "sites": [len(x.sites) for x in live],
            "combines": sorted({b.fns[0][s.combine[0][0]].id for b in live for s in b.sites}),
            "units": dict(sorted(sites.circuit.units.items())),
            "count": I.input_count(I.descriptor_record(sites.circuit, 64), by.__getitem__),
            "parts_bits": sum(parts_bits(b) for b in live), "peer_bits": peer_bits(sites),
            "unwired": dict(sorted(kinds.items())), "proof_blocks": proof_blocks(sites),
            "sent": all(sent_at_own_slices(b) for b in live), "backward": backward,
            "wires": len(sites.circuit.wires), "templates": len(sites.circuit.templates)}


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("VERITY_VLLM_CLASS_G_WINDOW") != "1",
                    reason="VERITY_VLLM_CLASS_G_WINDOW=1: GLM-5.3's eight ranks compiled in each of two processes at once (about "
                           "45 minutes and 2.3 GB each, on one core)")
def test_class_g_composes_its_eight_ranks_with_stable_digests_and_wires_every_peer_row(tmp_path):
    """GLM-5.3 at TP8 (`class_g_tp`) in two fresh processes of other hash seeds: the same rank workloads, bindings, TP
    workload digest and window. The window of both requests, the two-token request's embedding all-reduce over both its
    rows among them, wires every bit of every rank's partial from its gathered rows, into its chunk combine units
    (embedding and attention all-reduces) and the MoE's fused row, or into each rank's token select (the logits'
    all-gather); each rank's sent partial is its own segment at every site's slice; every wire runs forward in the public
    order; no read of a combine's output is left free, the MoE row's new residual into the next attention all-reduce's
    chunk units among them; and C-Flock's composer refuses none of the window's own units. `at_rank` of the registration is
    each rank's own, so each rank's digest is `compile_steps`' there. What no cheaper test can: the binding of eight ranks
    of a served model's compiled Programs at the chunk grain, each rank's place among the others read from its Programs,
    and their digests' stability across processes; it compiles GLM-5.3 eight times in each of two processes."""
    code = ("import json, sys; from pathlib import Path; from tests.program.predict.test_tp_steps import class_g_tp; "
            "print(json.dumps(class_g_tp(Path(sys.argv[1]))))")
    seeds, runs = ("1", "2"), []
    for seed in seeds:
        with (tmp_path / f"{seed}.out").open("w") as out, (tmp_path / f"{seed}.err").open("w") as err:
            runs.append(subprocess.Popen([sys.executable, "-c", code, str(tmp_path / seed)], cwd=ROOT, stdout=out, stderr=err,
                                         env={**os.environ, "PYTHONHASHSEED": seed}))
    for seed, p in zip(seeds, runs):
        assert p.wait() == 0, (tmp_path / f"{seed}.err").read_text()[-3000:]
    got = [json.loads((tmp_path / f"{seed}.out").read_text().splitlines()[-1]) for seed in seeds]
    assert got[0] == got[1]
    g = got[0]
    assert len(set(g["ranks"])) == 8 and g["sites"][0] < g["sites"][1] and g["peer_bits"] == g["parts_bits"] <= g["count"]["wired"]
    assert g["sent"] and g["backward"] is None
    assert {c.split("{")[0] for c in g["combines"]} >= {"AllReduceRmsNormMnnvl_v1", "AllGather_v1"}
    assert g["unwired"] == {} and g["proof_blocks"] == {}
    reg = _class_g(tmp_path / "phi")
    assert all(at_rank(reg, r) == _class_g(tmp_path / "phi", rank=r) for r in range(8))
    print(json.dumps(g))
