"""training_slowdown.capture.circuit: the training formalism's quantities on recorded/islands/toy/train.py (V = 16, H = 8, fp32,
4 x 8 tokens a step, measured at step 1): a dense update's u steps up by each matrix as one token's work first reads it
(emb, fc, head), however the program writes the update; a is a token's hidden state; Δ_grad is the fp32 gradient and the
state the forward doesn't read, or a coded gradient's codes; frozen layers above the update are its tail; MeZO's
gradient is one fp32 difference."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from training_slowdown.capture import script
from training_slowdown.capture.circuit import StepCircuit, check, measure, path, run_ops, vertex_cut
from training_slowdown.formalism import READ_BACKS

PROGRAM = Path(__file__).parent / "recorded" / "islands" / "toy" / "train.py"
V, H, B, CTX = 16, 8, 4, 8
T = B * CTX
EMB, FC, HEAD = V * H, H * H, V * H
VALUES = EMB + FC + HEAD


@pytest.fixture
def starts(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "STREAM_TOKENS", 10 ** 5)
    shapes = {"emb": [V, H], "fc": [H, H], "head": [V, H], "frozen": [V, H]}
    (tmp_path / "toy.start.json").write_text(json.dumps({"config": {"vocab": V, "d_model": H},
                                                         "tensors": {n: ["float32", s] for n, s in shapes.items()}}))
    return tmp_path


def recorded(starts, *args):
    return script.record(PROGRAM, ["--batch", str(B), "--update", *args], step=1, starts=starts, wall=[0.1] * 8,
                         step_tokens=T)


@pytest.mark.parametrize("update,state", [("inplace", 0), ("rebind", 0), ("adamw", 64 * VALUES),
                                          ("sign-ef", 32 * VALUES), ("adamw --micro 2", 64 * VALUES)])
def test_a_dense_update(starts, update, state):
    """τ: a token's id and its target's, 9 positions of each row per 8 consumed tokens. The token ops are every op but
    the optimizer's, all dependent; dW and the head's input gradient read no update value (t), and no frozen layer sits
    above the head. ``--micro 2``: one token's work is its micro-batch's, half the step's."""
    f = recorded(starts, *update.split()).formalism
    assert f["tokens"] == T and f["token_factor"] == 1 and f["tau_ids"] == 9 / 8 and f["tau_other_bits"] == 0
    assert f["a_bits"] == 32 * H
    assert [b for _, b in f["u_in"]] == [32 * b for b in (0, EMB, EMB, EMB + FC, EMB + FC, VALUES, VALUES)]
    assert f["u_in"][1][0] == 0 and f["u_in"][-1][0] == 1
    assert f["u1_bits"] == f["update_cut_bits"] == 32 * VALUES
    assert f["grad_cut_bits"] == 32 * VALUES and f["state_bits"] == state and f["delta_grad_bits"] == 32 * VALUES + state
    assert f["affected_share"] == f["sigma"] and f["affected_share"] > 0.9 and f["affected_all_share"] == 1
    assert f["tail_share"] == pytest.approx(1 / 3, abs=0.01) and f["frozen_tail_share"] == 0
    if "--micro" in update:
        assert f["group_gates"] == pytest.approx(recorded(starts, "adamw").formalism["group_gates"] / 2)


@pytest.mark.parametrize("micro", [1, 2])
@pytest.mark.parametrize("update,per_value", [("vote-packed", 1), ("vote-bytes", 1), ("vote-planes", 2 + 16 / 8)])
def test_a_coded_gradient_passes_at_its_code(starts, update, per_value, micro):
    """Each partial gradient passes coded and is summed decoded, so the cut is the codes once a partial: quant's int32
    words and a uint8 packing at 1 bit a vote, lowrank-restart's two bool planes at 1 bit an entry each and its f16
    scales at 16 bits per 8 entries (the dW products the codes are made from stay at 32 bits an element). Under the 32-bit
    floor on every cut value (training_slowdown 4.2) the uint8 words were 4 bits a vote and the planes 32 bits an entry. The
    codec's ops are the step's: |C| grows by their gates, all outside the token blocks."""
    plain = recorded(starts, "inplace").formalism
    f = recorded(starts, update, "--micro", str(micro)).formalism
    assert f["grad_cut_bits"] == micro * per_value * VALUES and f["delta_grad_bits"] == f["grad_cut_bits"]
    assert f["state_bits"] == 0 and f["u1_bits"] == 32 * VALUES and f["a_bits"] == 32 * H
    assert f["gates"] > plain["gates"] and f["sigma"] < plain["sigma"]


def test_reading_a_entries_along_a_dense_path(starts):
    """Reading A's inputs: entering at the start reads the token's id and the update the run of segments reads (emb and
    fc in the first, the whole update through the head); entering after it reads the token's hidden state, a's 32 H
    bits; a run of the backward alone reads the matrices its input gradients multiply. Without attention no op mixes
    positions, and one group's gradient partial is the whole step's (one batch)."""
    f = recorded(starts, "inplace").formalism
    e = f["entries"]
    assert e["tokens"] == f["group_tokens"] == T and e["positions"] == CTX
    assert sum(e["gates"]) * T == pytest.approx(f["group_gates"])
    assert e["frontier_bits"][0][-1] == 0 and e["ids"][0][0] == 1 and e["frontier_bits"][1][0] == 32 * H
    assert e["u_bits"][0][0] == 32 * (EMB + FC) and e["u_bits"][0][-1] == 32 * VALUES
    assert e["u_bits"][1][0] == 32 * HEAD and min(min(r) for r in e["u_bits"]) == 0
    assert e["context_fwd_bits"] == e["context_bwd_bits"] == [0.0] * e["segments"]
    assert f["partial_cut_bits"] == f["grad_cut_bits"] == 32 * VALUES


def test_reading_a_context_of_an_attention(starts):
    """One causal attention over the embeddings: its scores and their product with the values each read an earlier
    position's embedding (32 H bits) in the forward; the backward's four products read a later position's."""
    e = recorded(starts, "inplace", "--attn").formalism["entries"]
    assert sum(e["context_fwd_bits"]) == 2 * 32 * H and sum(e["context_bwd_bits"]) == 4 * 32 * H
    assert all(e["phase"][s] == 0 for s, b in enumerate(e["context_fwd_bits"]) if b)
    assert all(e["phase"][s] == 1 for s, b in enumerate(e["context_bwd_bits"]) if b)


def test_reading_a_enters_an_adapter_at_its_code():
    """wildcard's grown adapter on the tiny start (`fixtures/wildcard_adapter.npz`: islands/wildcard/train.py at
    9460fcd228, --grow 1 --adapter 8 --adapter-z 2 --set 'blocks\\.2\\.(mlp_norm|w_gate\\.weight|w_up\\.weight|w_down\\.weight)'
    --batch 2 --micro 2 --warmup 1, step 1). The frozen layers are θ's function of the tokens, so entering at the grown
    block reads no computed state, and its matrices; segments start inside the code (its 2-bit words and scale), where
    a block reaching the head reads at most 48 bits a token, but w_down's whole bf16 cast too (the program multiplies
    every column, the dead ones zero), and the dead units' slice that only a zeros_like reads is never charged."""
    f = measure(StepCircuit.load(Path(__file__).parent / "fixtures" / "wildcard_adapter.npz"))
    e = f["entries"]
    assert f["a_bits"] == 32 and f["tail_share"] == pytest.approx(0.989, abs=0.001)
    head = max(range(e["segments"]), key=lambda s: e["gates"][s] if e["phase"][s] == 0 else -1)
    assert e["frontier_bits"][0][head] == 0 and e["u_bits"][0][head] == 32 * 64 + 16 * 3 * 192 * 64
    best = min(range(1, head + 1), key=lambda i: e["frontier_bits"][i][head - i])
    assert e["frontier_bits"][best][head - best] <= 48 and e["u_bits"][best][head - best] == 16 * 64 * 192
    assert all(e["u_bits"][i][head - i] >= 16 * 64 * 192 for i in range(head + 1))
    assert max(e["frontier_bits"][i][0] for i in range(1, head)) < 16 * 192


def test_the_checker_counts_each_run_of_segments_as_the_entries_do():
    """Reading A's checker counts any set of the trace's ops on its own; on each run of the adapter's segments (the
    default packer's blocks, `run_ops`) its gates and reads are the entries'. A block of the grown block's w_down alone
    reads its whole cast."""
    c = StepCircuit.load(Path(__file__).parent / "fixtures" / "wildcard_adapter.npz")
    e, preds = measure(c)["entries"], c.preds()
    order, tokens = path(c, preds)
    assert tokens == e["tokens"] and len(e["first_ops"]) == e["segments"]
    for i in range(e["segments"]):
        for j in range(i, e["segments"]):
            got = check(c, run_ops(order, e, i, j), tokens, preds)
            assert got["gates"] == pytest.approx(sum(e["gates"][i:j + 1]))
            assert [got[k] for k in ("frontier_bits", "ids", "u_bits")] == pytest.approx(
                [e["frontier_bits"][i][j - i], e["ids"][i][j - i], e["u_bits"][i][j - i]])
    down = [o for o in order if check(c, [o], tokens, preds)["u_bits"] == 16 * 64 * 192]
    assert down and check(c, down, tokens, preds)["gates"] > 0


def test_a_partial_is_one_groups_gradient_at_its_code(starts):
    """With --micro 2 one token's group is its micro-batch: its partial is its own codes, half the step's cut."""
    f = recorded(starts, "vote-planes", "--micro", "2").formalism
    assert f["group_tokens"] == T // 2 and f["partial_cut_bits"] == 4 * VALUES == f["grad_cut_bits"] / 2


def test_frozen_layers_above_the_update_are_its_tail(starts):
    """With the head frozen, u(1) is emb's and fc's bits; the head's forward and input gradient are the frozen tail, and
    t adds fc's dW."""
    f = recorded(starts, "inplace", "--freeze", "head").formalism
    assert f["u1_bits"] == f["grad_cut_bits"] == 32 * (EMB + FC)
    work = FC + HEAD + HEAD + FC + FC
    assert f["frozen_tail_share"] == pytest.approx(2 * HEAD / work, abs=0.01)
    assert f["tail_share"] == pytest.approx((2 * HEAD + FC) / work, abs=0.01)


def test_an_index_computed_from_the_targets_is_no_token(starts):
    """A sampled head reads its rows at unique(y), an index computed from the targets: the consumed tokens are the
    token input's positions alone, so the token count and τ are per token as with the full head. a is the hidden state
    and the gathered rows over the tokens (θ' met by token data, as the embedding's lookup is), all T of them under fake
    tensors."""
    f = recorded(starts, "inplace", "--sampled-head").formalism
    assert f["tokens"] == T and f["token_factor"] == 1 and f["tau_ids"] == 9 / 8
    assert f["a_bits"] == 2 * 32 * H


def test_a_mezo_step(starts):
    """Both perturbed passes are the token's work, reading the update's perturbed copies (u(1) = 32 |θ'|) with nothing
    free of it; a is both passes' hidden states; Δ_grad is the one fp32 difference of the losses."""
    f = recorded(starts, "mezo").formalism
    assert f["u1_bits"] == 32 * VALUES and f["tail_share"] == 0
    assert f["a_bits"] == 2 * 32 * H
    assert f["delta_grad_bits"] == f["grad_cut_bits"] == 32 and f["state_bits"] == 0


@pytest.mark.parametrize("source", ["theta", "trained"])
def test_a_cache_rewritten_each_step(starts, source):
    """The side GEMM reads θ's table scaled by the buffer: outside A when the buffer comes from θ, and a second path a
    token's state is handed on when it comes from the trained embedding."""
    plain = recorded(starts, "inplace").formalism
    f = recorded(starts, "inplace", "--cache", source).formalism
    assert f["u1_bits"] == 32 * VALUES
    if source == "theta":
        assert f["affected_share"] < plain["affected_share"] and f["a_bits"] == 32 * H
    else:
        assert f["affected_share"] >= plain["affected_share"] and f["a_bits"] == 2 * 32 * H


@pytest.mark.parametrize("update", ["adamw", "mezo-clip"])
def test_a_saved_circuit_measures_alike(starts, tmp_path, update):
    """With the recording's read-back lists and product rows, which a saved circuit keeps (the scorer measures the saved
    one); one saved without the lists measures without them, which training_slowdown.formalism refuses as unchecked."""
    g = recorded(starts, update)
    g.circuit.save(tmp_path / "c.npz")
    assert measure(StepCircuit.load(tmp_path / "c.npz")) == g.formalism
    assert bool(g.formalism["data_dependent"]) is (update == "mezo-clip")
    replace(g.circuit, read_backs=None).save(tmp_path / "old.npz")
    old = measure(StepCircuit.load(tmp_path / "old.npz"))
    assert "data_dependent" not in old and old == {k: v for k, v in g.formalism.items() if k not in READ_BACKS}


def test_a_vertex_cut():
    """0 -> {1, 2} -> 3: the narrowest set meeting both paths is {0} or {3}, or {1, 2}; an uncuttable sink and source
    leave none."""
    preds = [[], [0], [0], [1, 2]]
    assert vertex_cut(preds, {0: 10, 1: 3, 2: 4, 3: 9}, [0], [3])[0] == 7
    assert vertex_cut(preds, {0: None, 1: 3, 2: None, 3: None}, [0], [3]) == (None, [])
