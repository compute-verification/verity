"""verity_vllm.program.predict.node: a tensor-parallel node's Program is its ranks' Programs joined at their collectives.

Each rank's run of the node's body is its rank Program's body node for node, every operand leaf the same leaf of the node's
parameters or of the rank's own nodes, except that a peer Input's leaf is a node of the peer rank: every rank's collective k reads
one operand where its rank Program reads its peers, each rank's rows its own (its other operands, a fused norm's residual and
weight, stay its own). So no peer Input is left, and an isolation block over the node reads only its prompt besides
θ (`nci_reference.check_desc`, `NciCheck.Desc`'s reference)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import is_input
from verity_vllm.pipeline import registration_log as RL
from verity_vllm.program.predict.build import build_program
from verity_vllm.program.predict.node import build_node_program, check_ranks
from verity_vllm.program.predict.phi import Phi
from tests.program.predict.test_predict import GLM53, TINY, _phi, _row_phi

_spec = importlib.util.spec_from_file_location(
    "nci_reference", Path(__file__).resolve().parents[5] / "verity/core/protocols/network/nci/lean/scripts/nci_reference.py")
NCI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NCI)


def _at(phi: Phi, rank: int) -> Phi:
    return Phi.from_json({**phi.to_json(), "tp": {**phi.tp, "rank": rank}})


def _joined(node, ranks: list, check_leaves: bool) -> None:
    """`node` is `ranks` (each rank's own Program) joined: see the module docstring. `check_leaves`: compare every operand leaf
    (else each rank's nodes' Definitions, in order)."""
    world, body = node.world, node.node_fn.body
    rank_of = node.rank_of()
    assert -1 not in rank_of, "every body node is a rank's"
    names = [n for n, _ in node.program.fn.signature()[0]]
    assert names[0] == "prompt" and not any(n.startswith("peers") for n in names)
    assert set(node.theta) == {n for n in names if ".weights[" in n}
    collectives: list[list] = []
    for r, rp in enumerate(ranks):
        mine = [j for j, q in enumerate(rank_of) if q == r]
        own = rp.program.fn.body.nodes
        assert [(body.nodes[j].form, body.nodes[j].fn.id) for j in mine] == [(n.form, n.fn.id) for n in own]
        rnames = [n for n, _ in rp.program.fn.signature()[0]]
        peer = {i: int(n[len("peers_"):]) for i, n in enumerate(rnames) if n.startswith("peers_")}
        others = [q for q in range(world) if q != r]
        reads = [(j, [x for x, a in enumerate(n.args) if any(s == "p" and i in peer for s, i, _lo, _hi in _runs(a))])
                 for j, n in zip(mine, own)]
        collectives.append([(j, xs) for j, xs in reads if xs])
        if not check_leaves:
            continue
        wi = rnames.index("weights")
        fields = [t.leaves for _, t in rp.program.fn.signature()[0][wi][1].fields]
        starts = [sum(fields[:f]) for f in range(len(fields))]
        for j, n in zip(mine, own):
            for a, b in zip(n.args, body.nodes[j].args, strict=True):
                assert a.type == b.type and a.refs.count == b.refs.count
                for (s, i, off), got in zip(a.refs, b.refs):
                    if s == "n":
                        assert got == ("n", mine[i], off)
                    elif rnames[i] == "prompt":
                        assert got == ("p", 0, off)
                    elif i == wi:
                        f = max(k for k, st in enumerate(starts) if st <= off)
                        assert got == ("p", names.index(f"rank{r}.weights[{f}]"), off - starts[f])
                    else:
                        q = others[peer[i] % (world - 1)]
                        assert got[0] == "n" and rank_of[got[1]] == q, f"rank {r}'s peer leaf is rank {q}'s node"
    assert len({len(c) for c in collectives}) == 1 and collectives[0]
    for calls in zip(*collectives):
        operands = [(xs, [_runs(body.nodes[j].args[x]) for x in xs]) for j, xs in calls]
        assert all(o == operands[0] for o in operands), "every rank's collective k reads one operand"


def _runs(c):
    from verity.ml.circuits.refs import runs
    return runs(c.refs)


@pytest.mark.parametrize("wrapper,shape", [("step", {"M": 1, "pos0": 8}), ("step", {"M": 3, "pos0": 2}), ("request", {"LP": 4, "T": 2})])
def test_tp2_node_is_its_ranks_joined_at_their_collectives(wrapper, shape):
    phi = _phi(TINY, tp={"world": 2, "rank": 0})
    node = build_node_program(phi, wrapper, with_desc=False, **shape)
    ranks = [build_program(_at(phi, r), wrapper, with_desc=False, **shape) for r in range(2)]
    _joined(node, ranks, check_leaves=True)
    n_events = 1 if wrapper == "step" else shape["T"] + 1
    assert node.program.fn.signature()[1].n == 2 * n_events
    assert sum(n.gates for n in node.node_fn.body.nodes) == sum(n.gates for rp in ranks for n in rp.program.fn.body.nodes)


@pytest.mark.parametrize("wrapper,shape", [("step", {"M": 1, "pos0": 8}), ("request", {"LP": 4, "T": 2})])
def test_the_root_is_the_input_nodes_and_one_call_of_the_body(wrapper, shape):
    """`NciCheck.Desc` keeps a root node's reads as pieces of gates, so the body is one Definition the root calls once: the root
    is the parameters' input nodes, in order, then that Call, and the descriptor holds the body's nodes in its callee."""
    node = build_node_program(_phi(TINY, tp={"world": 2, "rank": 0}), wrapper, **shape)
    lowered = sum(len(RL.lower(t)) for _, t in node.program.fn.signature()[0])
    root = node.program.root.body.nodes
    assert len(root) == lowered + 1 and all(is_input(n.fn) for n in root[:-1])
    assert root[-1].form == "call" and root[-1].fn is node.node_fn
    s = node.summary()
    assert (s["n_root_nodes"], s["body_nodes"], s["node_fn"]) == (lowered + 1, len(node.node_fn.body.nodes), node.node_fn.id)
    d = node.desc
    nodes = d["definitions"][d["root"]]["body"]["nodes"]
    assert len(nodes) == lowered + 1 and nodes[-1]["form"] == "call"
    callee = d["definitions"][d["callees"][nodes[-1]["fn"]]]
    assert len(callee["body"]["nodes"]) == len(node.node_fn.body.nodes) == sum(hi - lo for _r, lo, hi in node.spans)


def test_check_ranks_holds_each_ranks_run_of_the_body_to_its_own_program():
    phi = _phi(TINY, tp={"world": 2, "rank": 0})
    node = build_node_program(phi, "request", with_desc=False, LP=4, T=2)
    got = check_ranks(phi, node, LP=4, T=2)
    ranks = [build_program(_at(phi, r), "request", with_desc=False, LP=4, T=2) for r in range(2)]
    assert [(g["rank"], g["nodes"]) for g in got] == [(r, len(rp.program.fn.body.nodes)) for r, rp in enumerate(ranks)]
    with pytest.raises(AssertionError, match="rank 0: the node's run of"):
        check_ranks(phi, node, LP=4, T=1)


@pytest.mark.slow
def test_the_node_reads_only_its_prompt_besides_theta():
    """One IU over the node's computation: its novel reads are the prompt's 4 x 32 bits; the rank Programs' are their peers' too."""
    node = build_node_program(_phi(TINY, tp={"world": 2, "rank": 0}), "request", LP=4, T=2)
    text = canonical_json(node.desc).decode()
    inp, reg, spans = NCI.root_nodes(text)
    theta = node.desc["annotations"]["theta"]
    assert theta == [k for k in range(1, len(inp)) if inp[k]] and all(reg[k] for k in theta)
    computed = sum(hi - lo for k, (lo, hi) in enumerate(spans) if not inp[k])
    inst = {"program": text, "theta": theta, "owner": ["input" if i else 0 for i in inp], "gamma": node.program.gates,
            "alpha": [4 * 32, computed]}
    assert NCI.check_desc(inst) == {"gates": node.program.gates, "ius": [[computed, 4 * 32]], "complies": True, "failure": None}
    assert NCI.check_desc(inst | {"alpha": [4 * 32 - 1, computed]})["failure"] == [0, "nci"]


def test_a_world_of_one_has_no_node_program():
    from verity_vllm.program.predict.common import Unexpressible
    with pytest.raises(Unexpressible, match="φ's world is 1"):
        build_node_program(_phi(TINY), "step")


def _first_layers(phi: Phi, n: int) -> Phi:
    """`phi` with its checkpoint's config cut to the first `n` layers."""
    hf = phi.hf_config
    hf.update(num_hidden_layers=n, indexer_types=hf["indexer_types"][:n], mlp_layer_types=hf["mlp_layer_types"][:n])
    return Phi.from_json({**phi.to_json(), "model": {**phi.model, "hf_config": hf}})


@pytest.mark.slow
def test_glm53_tp8_node_step_is_its_ranks_joined(tmp_path):
    """GLM-5.3 B1's eight ranks at TP8 (`predict.dsa`): the node's step is each rank's step Program joined at the decoder's
    collectives, the FlashInfer fused all-reduce + norm (`peers.rows`), the routed all-reduce (`peers.whole`) and the logits
    gather, which the TP2 Llama test cannot reach. The checkpoint is cut to its first 4 layers, which hold every kind of layer the
    78 repeat (dense MLP with a full indexer, then MoE with a shared one), so the join is checked at each kind of collective without
    building all 78 (769 s on 12 workers)."""
    phi = _first_layers(_row_phi(GLM53, "GLM53", tmp_path), 4)
    assert phi.hf_config["mlp_layer_types"] == ["dense"] * 3 + ["sparse"] and phi.hf_config["indexer_types"][3] == "shared"
    node = build_node_program(phi, "step", M=1, pos0=8, with_desc=False)
    ranks = [build_program(_at(phi, r), "step", M=1, pos0=8, with_desc=False) for r in range(8)]
    _joined(node, ranks, check_leaves=False)
    assert node.world == 8 and len(node.theta) == 8 * len(ranks[0].program.fn.signature()[0][-1][1].fields)
