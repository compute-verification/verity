"""The operands of a Program row that are not one committed member's words (`verity_vllm.check.replay.driver`, `population`,
`evaluate`): the weights of the interior producers a pick composes are read in the parent ahead of the fork, each field once and
only under a producer the replay evaluates; the population names an interior producer that has no evaluator, and no other; an
operand slice no committed member holds, of a producer that is neither interior nor a Program constant, is unresolved by its row;
the Embedding is the first row that reads a prompt id, whatever parameter an earlier row reads; a MoE block's group quant that reads
a row of another module is the input's, and the activation's reads its runner's own activation row."""
from __future__ import annotations

import copy

import numpy as np

from tests.check import test_sampled_replay as T
from tests.query import test_glm53_moe_planes as GP
from verity_vllm.check.replay import driver as DR
from verity_vllm.check.replay import evaluate as EV
from verity_vllm.check.replay import index as IX
from verity_vllm.check.replay import population as POP
from verity_vllm.check.replay.opening import CommittedStore

moe_block = GP.instances


class _Index(IX.ProgramIndex):
    """An index over rows stated here.  `describe` answers what a test states of each row; `member_at` and `weight_slice` are the
    index's own, over five weight fields of ten words."""

    def __init__(self, rows: list[dict], described: dict[int, dict]) -> None:
        self.pnames, self.by_i, self._described = ["prompt", "weights"], {int(r["i"]): r for r in rows}, described
        self.fields = [(f"w{k}", 10, [10], 16) for k in range(5)]
        self.foff = [10 * k for k in range(5)]

    def describe(self, r: dict) -> dict | None:
        return self._described.get(int(r["i"]))


def _row(i: int, *operands: list) -> dict:
    return {"i": i, "spec": "X_v1{}", "name": f"x{i}", "args": [[item] for item in operands]}


def _field(k: int) -> list:
    """The first four words of weight field k."""
    return ["p", 1, 10 * k, 10 * k + 4]


def _pick() -> tuple[_Index, dict]:
    """A pick, row 0, that reads weight field 0 and eight words of each other row.  Row 1 commits them as one member.  Row 2 is
    interior, of a family with an evaluator: its fields 0 and 2 are the pick's too.  Field 3 is read by no row the pick composes:
    row 3 is interior, of a family without an evaluator; row 4's eight words span its two members; row 5 the index does not
    describe; row 9 is no row."""
    rows = [_row(0, _field(0), *(["n", producer, 0, 8] for producer in (1, 2, 3, 4, 5, 9))),
            _row(1, _field(1)), _row(2, _field(0), _field(2)), _row(3, _field(3)), _row(4, _field(3)), _row(5, _field(3))]
    described: dict[int, dict] = {
        1: {"family": "Gemm_v1", "members": [("out", 8, "bf16")]}, 2: {"family": "Gemm_v1", "members": []},
        3: {"family": "X_v1", "members": []}, 4: {"family": "Gemm_v1", "members": [("a", 4, "bf16"), ("b", 4, "bf16")]}}
    return _Index(rows, described), rows[0]


def test_the_prewarm_reads_each_weight_field_once_and_descends_only_into_an_interior_producer_the_replay_evaluates():
    index, pick = _pick()
    read: list[tuple] = []
    DR._prewarm_weights({"r0": index}, [{"request_id": "r0", "i": pick["i"]}], lambda field, lo, hi: read.append((field, lo, hi)))
    assert read == [("w0", 0, 4), ("w2", 0, 4)]


def test_the_population_names_the_interior_producer_without_an_evaluator_and_no_other_row_the_pick_reads():
    index, pick = _pick()
    assert POP._interior_feeders_without_evaluator(index, pick) == {"X_v1 (no registered evaluator)"}


def test_an_operand_slice_no_committed_member_holds_is_unresolved_by_its_row():
    """A Gemm of the sampled replay's CPU-served world (`tests.check.test_sampled_replay`), its operand widened one word past
    its producer's one member."""
    inst, woff = T._program()
    world = T.Honest(inst, woff, T._bf16(np.random.default_rng(7), max(hi for _lo, hi in woff.values()), 0.5), {"r0": [1, 2], "r1": [5, 3]})
    store, weights = CommittedStore(*world.bmap_and_reader()), T._weights(world)
    row = copy.deepcopy(next(r for r in world.pi.rows if r["spec"].startswith("Gemm_v1")))
    ((kind, producer, lo, hi),) = row["args"][0]
    outputs, why = EV.evaluated(world.pi, store, weights, row, "r0", 0, [])
    assert outputs is not None and why is None and world.pi.member_at(producer, lo, hi) is not None
    row["args"][0] = [[kind, producer, lo, hi + 1]]
    assert EV.evaluated(world.pi, store, weights, row, "r0", 0, []) == (
        None, f"arg slice ['n', {producer}, {lo}, {hi + 1}) is not inside one committed member of row {producer}")


def test_the_embedding_s_path_is_that_of_the_first_row_reading_a_prompt_id_whatever_parameter_an_earlier_row_reads():
    pi = IX.ProgramIndex(T._program()[0])
    assert pi.embed_op_path() == "model.embed_tokens"
    first = copy.copy(pi)
    first.rows = [{"i": -1, "spec": "X_v1{}", "name": "x", "args": [[["p", pi.pnames.index("weights"), 0, 4]]]}, *pi.rows]
    vars(first).pop("_embed_op", None)
    assert first.embed_op_path() == "model.embed_tokens"


def test_a_group_quant_that_reads_a_row_of_another_module_is_the_block_input_s(moe_block):
    """GLM-5.3's derived block (`tests.query.test_glm53_moe_planes`), whose input's quant reads a parameter, behind a norm row a
    token as a served block is: the input's quant reads that row.  A quant is the activation's only where the row it reads is an
    activation row of its own runner."""
    M, NV, H = GP.M, GP.NV, GP.H
    norm = f"RMSNormFusedCuda_v2{{N={H},EPS={GP.EPS}}}"
    before = [{"i": t, "spec": norm, "form": "call", "name": f"model.model.layers.3.post_attention_layernorm/fused_add_rms_norm[{t}]",
               "args": [[["p", 1, t * H, (t + 1) * H]], [["p", 1, t * H, (t + 1) * H]], [["p", 1, 0, H]]]} for t in range(M)]
    block = [dict(r, i=r["i"] + M, args=[[["n", ref[1] + M, *ref[2:]] if ref[0] == "n" else ref for ref in arg] for arg in r["args"]])
             for r in moe_block["rows"]]
    assert [r["args"] for r in block[:M]] == [[[["p", 1, t * H, (t + 1) * H]]] for t in range(M)]
    for t in range(M):
        block[t]["args"] = [[["n", t, 0, H]]]
    pi = IX.ProgramIndex(dict(moe_block, rows=before + block))
    planes = [d for r in before + block if (d := pi.describe(r)) and d["op_path"] == "moe/L3"]
    assert [d["plane_kind"] for d in planes] == (["in_quant"] * M + ["router"] * M + ["up"] * NV + ["act"] * NV + ["act_quant"] * NV
                                                 + ["down"] * NV)
    assert not [d["plane_error"] for d in planes if "plane_error" in d]
    assert [d["members"] for d in planes[:M]] == [[("in_q", H, "w8"), ("in_scale", H // GP.G, "f32")]] * M
