"""verity_vllm.program.predict.tp_steps: a node's advice compiled at every rank of φ.tp's world and composed (`compile_tp`).

The dense toy at TP2 compiles in seconds; GLM-5.3 at TP8, class G's registration, is the opt-in test, on one-token prompts: at
node grain each rank's partial of a collective is one node's output only where the collective is per row, and the route
all-reduce of a prefill's rows (the embedding's) reads all of a rank's rows at once, so a prompt of two tokens or more is
refused by name (`pipeline.tp.wires`)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

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


def class_g_tp(hf_home: Path) -> dict:
    """GLM-5.3 at TP8 under class G's registration, a one-token prompt and a two-token one prefilled in one step, each decoding
    once: `compile_tp`'s digests and sites, the window of the one-token request's sites (its wired and free input bits, and
    the bits of every combine's parts rows), and why the window of both requests is refused."""
    advice, ingress = _session([(1, 2, ONCE, [5]), (2, 2, ONCE, [5, 6])], [[(0, 0, 1), (1, 0, 2)], [(0, 1, 1), (1, 2, 1)]])
    c = compile_tp(_class_g(hf_home), advice, ingress)
    one = TW.sites_circuit(c.bindings[:1], "class-g")
    by = {descriptor_id(fn): fn for fn, _ in one.circuit.templates}
    try:
        TW.sites_circuit(c.bindings, "class-g")
        refused = None
    except ValueError as e:
        refused = str(e)
    b = c.bindings[0]
    return {"workload": c.workload.digest, "ranks": list(c.ranks), "bindings": [x.digest for x in c.bindings],
            "sites": [len(x.sites) for x in c.bindings], "combines": sorted({b.fns[0][s.combine[0][0]].id for s in b.sites}),
            "count": I.input_count(I.descriptor_record(one.circuit, 64), by.__getitem__),
            "parts_bits": sum(b.world * b.world * (s.hi - s.lo) * 16 for s in b.sites),
            "wires": len(one.circuit.wires), "templates": len(one.circuit.templates), "refused": refused}


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("VERITY_VLLM_CLASS_G_WINDOW") != "1",
                    reason="VERITY_VLLM_CLASS_G_WINDOW=1: GLM-5.3's eight ranks compiled in each of two processes at once (about "
                           "45 minutes and 2.3 GB each, on one core)")
def test_class_g_composes_its_eight_ranks_with_stable_digests_and_wires_every_peer_row(tmp_path):
    """GLM-5.3 at TP8 (`class_g_tp`) in two fresh processes of other hash seeds: the same rank workloads, bindings and TP
    workload digest; the one-token request's window wires every combine's parts rows (embedding, attention and MoE
    all-reduces, the logits' all-gather), none free; the two-token request's embedding all-reduce, over both of a rank's rows,
    is refused by name. `at_rank` of the registration is each rank's own, so each rank's digest is `compile_steps`' there.
    What no cheaper test can: the binding of eight ranks of a served model's compiled Programs, each rank's place among the
    others read from its Programs, and their digests' stability across processes; it compiles GLM-5.3 eight times in each of
    two processes."""
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
    assert len(set(g["ranks"])) == 8 and g["sites"][0] < g["sites"][1] and g["count"]["wired"] == g["parts_bits"]
    assert {c.split("{")[0] for c in g["combines"]} >= {"AllReduceRmsNormMnnvl_v1", "AllGather_v1"}
    assert re.fullmatch(r"class-g: request 1's collective 0 \[0, 12288\): rank 0's partial is 2 nodes' rows \[\d+, \d+\], a collective "
                        r"over several rows, whose segment per rank no one unit writes at node grain; compile it per row", g["refused"])
    reg = _class_g(tmp_path / "phi")
    assert all(at_rank(reg, r) == _class_g(tmp_path / "phi", rank=r) for r in range(8))
    print(json.dumps(g))
