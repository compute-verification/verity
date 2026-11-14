"""What the readers of `verity_vllm.check.replay` answer for a field its document leaves out: a Program row without operands and
an instances document without supplied inputs (`verity_vllm.check.replay.index`, and the row's readers in `evaluate`, `population`
and `driver`), a binding map without entries and an entry without an identity or an encoding (`verity_vllm.check.replay.opening`),
a manifest without a query, aliases or a cross-check's counts, an engine schedule without observed arrive steps and a workload
request without steps or prompt ids (`verity_vllm.check.replay.linkage`), a population without reasons
(`verity_vllm.check.replay.replay_codes`), a row's description without statics, a stop rule's block without problems and an
executed prefix without its lag (`verity_vllm.check.replay.population`), a step's geometry without a plane row's module.  Each
field is given absent, null and empty, and the answer is the same for the three; where a document can state the field, the stated
answer stands beside them."""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.check import test_executed_prefix as XP
from tests.check import test_sampled_replay as T
from verity_vllm.check.replay import coverage as COV
from verity_vllm.check.replay import driver as DR
from verity_vllm.check.replay import evaluate as EV
from verity_vllm.check.replay import index as IX
from verity_vllm.check.replay import linkage as LK
from verity_vllm.check.replay import opening as OP
from verity_vllm.check.replay import population as POP
from verity_vllm.check.replay import replay_codes as RC
from verity_vllm.query.required import Q_MODULE_BODY_ID
from verity_vllm.records.required_manifest import SCHEMA
from verity_vllm.records.schema_ids import MOE_PLANE_ALIAS_CHECK_SCHEMA

ABSENT: Any = object()
NO_LIST = pytest.mark.parametrize("left_out", [ABSENT, None, []], ids=["absent", "null", "empty"])
NO_DICT = pytest.mark.parametrize("left_out", [ABSENT, None, {}], ids=["absent", "null", "empty"])


def _with(doc: dict, key: str, value: Any) -> dict:
    """`doc` with `key` holding `value`, or without the key."""
    return {k: v for k, v in doc.items() if k != key} | ({} if value is ABSENT else {key: value})


def _entry(encoding: Any = ABSENT, **identity: Any) -> dict:
    """A binding-map entry of request r0's `m/out` at step 0."""
    ident = {"request_id": "r0", "step": 0, "op_path": "m", "output_member": "out", **identity}
    return _with({"identity": ident, "name": "m/out"}, "encoding", encoding)


def _committed(step: int, name: str, lo: int, hi: int) -> bytes:
    return bytes(hi - lo)


@pytest.fixture(scope="module")
def inst() -> dict:
    return T._program()[0]


@pytest.fixture(scope="module")
def pi(inst: dict) -> IX.ProgramIndex:
    return IX.ProgramIndex(inst)


@pytest.fixture(scope="module")
def world() -> Any:
    """The Program served honestly to two requests, three steps each."""
    inst, woff = T._program()
    weights = T._bf16(np.random.default_rng(7), max(hi for _lo, hi in woff.values()), 0.5)
    return T.Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})


@NO_DICT
def test_a_cross_check_that_counts_no_identities_says_none_for_each_engine(left_out):
    cross = _with({"agree": True, "primary_engine": "traversal", "other_engine": "inventory"}, "identities", left_out)
    manifest = {"schema": SCHEMA, "program_digest": "p" * 8, "manifest_digest": "m" * 8, "requests": ["r0"], "identities": [], "cross_check": cross}
    line = COV.query_population(manifest, {}, set(), ranks=[])["scope"]
    assert line.endswith("cross-check AGREED (traversal vs inventory: None / None identities))")


@pytest.mark.parametrize("manifest", [None, {}, {"query": None}, {"query": {}}, {"query": {"query_id": "a query the population does not know"}}])
def test_a_manifest_that_names_no_query_it_knows_is_read_under_the_query_of_record(manifest):
    assert DR.manifest_query(manifest) is None
    assert DR.manifest_query({"query": {"query_id": Q_MODULE_BODY_ID}}) == Q_MODULE_BODY_ID


@NO_LIST
def test_a_row_that_states_no_operands_has_none(left_out):
    assert IX.row_args(_with({"i": 0}, "args", left_out)) == []
    assert IX.row_args(None) == [] and IX.row_args({}) == []
    assert IX.row_args({"args": [[["n", 3, 0, 4]]]}) == [[["n", 3, 0, 4]]]


@NO_DICT
def test_an_observed_schedule_that_states_no_arrive_steps_leaves_the_declaration(left_out):
    declared = {"a": 0, "b": 1}
    assert LK.arrivals_of_record(declared, _with({"use": "observed"}, "observed_arrive_steps", left_out)) == (declared, False)
    assert LK.arrivals_of_record(declared, {"use": "observed", "observed_arrive_steps": {"a": 2, "not declared": 9}}) == ({"a": 2, "b": 1}, True)


@NO_DICT
def test_a_manifest_without_aliases_gives_an_empty_table(left_out):
    assert OP.alias_table(_with({"schema": SCHEMA}, "op_path_aliases", left_out)) == {}
    assert OP.alias_table({"op_path_aliases": {"layers.7.rot": ["layers.0.rot", 7]}}) == {"layers.7.rot": ("layers.0.rot", 7)}


@NO_LIST
def test_a_binding_map_without_entries_binds_nothing(left_out):
    assert OP.CommittedStore(_with({"schema": "test"}, "entries", left_out), _committed).idx == {}


@NO_DICT
def test_an_entry_without_an_identity_is_not_indexed_and_is_invocation_0(left_out):
    entry = _with({"name": "m/out"}, "identity", left_out)
    assert OP.CommittedStore({"entries": [entry, _entry()]}, _committed).idx == {("r0", 0, "m", "out"): [_entry()]}
    assert OP._invocation(entry) == 0 and OP._invocation(_entry(invocation=3)) == 3


def test_an_entry_whose_identity_states_no_element_range_covers_no_slice():
    for entry in (_entry(encoding={"dtype": "bf16"}), _entry(encoding={"dtype": "bf16"}, element_range=None)):
        got = OP.CommittedStore({"entries": [entry]}, _committed).words("r0", 0, "m", "out", 0, 4)
        assert got == (None, "element slice [0, 4) outside the bound identity range [0, None] of (r0, step 0, m/out)")


@NO_DICT
def test_an_entry_without_an_encoding_has_no_dtype_the_store_knows(left_out):
    store = OP.CommittedStore({"entries": [_entry(encoding=left_out, element_range=[0, 8])]}, _committed)
    assert store.words("r0", 0, "m", "out", 0, 4) == (None, "unknown dtype '' of (r0, step 0, m/out)")
    words, dtype = OP.CommittedStore({"entries": [_entry(encoding={"dtype": "bf16"}, element_range=[0, 8])]}, _committed).words("r0", 0, "m", "out", 0, 4)
    assert words is not None and dtype == "bf16" and words.tolist() == [0, 0, 0, 0]


@NO_DICT
def test_plane_entries_without_an_identity_are_no_committed_plane(left_out):
    """The store indexes no such entry itself; the check reads whatever the index holds."""
    store = OP.CommittedStore({"entries": []}, _committed)
    entry = _with({"name": "moe/L0/up"}, "identity", left_out)
    store.idx = {("r0", 0, "moe/L0", "up"): [entry], ("r0", 0, "moe/L0", "down"): [entry]}
    fact = POP.moe_plane_alias_check(store)
    assert (fact["result"], fact["compared"], fact["per_layer"]) == (None, 0, {})


def test_a_plane_whose_entry_states_no_encoding_is_named_unread_and_its_layer_is_not_compared():
    """The store knows no dtype for such an entry, so the check reads no words of the plane.  Of thirteen it names the first
    twelve (`ALIAS_NAMED`), as it does of the pairs it reads and finds identical: here every pair, the store's bytes being zeros."""
    bf16 = {"dtype": "bf16"}

    def check(layers: int, up: Any = ABSENT, down: Any = ABSENT) -> dict:
        entries = [_entry(up if mem == "up" else down, op_path=f"moe/L{layer}", output_member=mem, element_range=[0, 8])
                   for layer in range(layers) for mem in ("up", "down")]
        return POP.moe_plane_alias_check(OP.CommittedStore({"entries": entries}, _committed))

    fact = check(1, up=bf16)
    assert (fact["schema"], fact["result"], fact["compared"], fact["read_errors_n"]) == (MOE_PLANE_ALIAS_CHECK_SCHEMA, None, 0, 1)
    assert fact["read_errors"] == [{"request_id": "r0", "step": 0, "layer": 0, "plane": "down",
                                    "why": "unknown dtype '' of (r0, step 0, moe/L0/down)"}]
    many = check(13)
    assert many["compared"] == 0 and [e["layer"] for e in many["read_errors"]] == list(range(12))
    assert {e["plane"] for e in many["read_errors"]} == {"up"}
    stated = check(13, up=bf16, down=bf16)
    assert (stated["result"], stated["compared"], stated["identical_n"], stated["read_errors"]) == (False, 13, 13, [])
    assert [pair["layer"] for pair in stated["identical"]] == list(range(12))


@NO_LIST
def test_a_binding_map_whose_entries_state_no_rank_is_rank_0s(left_out, monkeypatch):
    class Reached(Exception):
        pass

    def populate(programs, ranks, *args, **kw):
        raise Reached(ranks)

    monkeypatch.setattr(DR, "populate", populate)

    def ranks_of(bmap: dict) -> list[int]:
        with pytest.raises(Reached) as reached:
            DR.sampled_replay({}, bmap, _committed, lambda field, lo, hi: None, seed=1)
        return reached.value.args[0]

    assert ranks_of(_with({"schema": "test"}, "entries", left_out)) == [0]
    unstated: list[Any] = [ABSENT, None, {}]
    for identity in unstated:
        assert ranks_of({"entries": [_with({"name": "m/out"}, "identity", identity)]}) == [0]
    assert ranks_of({"entries": [_entry(), _entry(rank=None)]}) == [0]
    assert ranks_of({"entries": [_entry(rank=1), _entry(rank=0), _entry(rank=1)]}) == [0, 1]


@NO_LIST
def test_a_row_without_operands_composes_no_interior_producer_and_prewarms_no_weight(left_out, pi):
    row = _with({"i": 0, "spec": "X_v1{}", "name": "x"}, "args", left_out)
    assert POP._interior_feeders_without_evaluator(pi, row) == set()
    read: list[tuple] = []
    DR._prewarm_weights({"r0": SimpleNamespace(pnames=["prompt", "weights"], by_i={0: row})}, [{"request_id": "r0", "i": 0}],
                        lambda field, lo, hi: read.append((field, lo, hi)))
    assert read == []


@NO_LIST
def test_a_row_without_operands_gives_its_evaluator_no_input(left_out, pi):
    """No Program states such a row for a family that reads an operand; the evaluator is called on no inputs and refuses the first it asks for."""
    row = _with(pi.rows[0], "args", left_out)
    assert pi.describe(row)["family"] == "Embedding_v1"
    with pytest.raises(IndexError, match="list index out of range"):
        EV._evaluate_row(pi, OP.CommittedStore({}, _committed), lambda field, lo, hi: None, row, "r0", 0, [], depth=0)


@NO_DICT
def test_an_instances_document_without_supplied_inputs_supplies_none(left_out, inst):
    assert IX.ProgramIndex(_with(inst, "supplied", left_out)).supplied == {}


@NO_LIST
def test_supplied_inputs_without_run_entries_or_fields_supply_no_field_there(left_out):
    registered = T._program(registered=True)[0]
    field = {0: {"name": "eps", "width": 32, "bits": 0x3727C5AC, "offset": 0, "leaves": 1}}
    assert IX.ProgramIndex(registered).supplied == {1: ("registration", field)}
    no_run = copy.deepcopy(registered)
    no_run["supplied"] = _with(no_run["supplied"], "run", left_out)
    assert IX.ProgramIndex(no_run).supplied == {1: ("registration", field)}
    no_fields = copy.deepcopy(registered)
    no_fields["supplied"]["registration"] = _with(no_fields["supplied"]["registration"], "fields", left_out)
    assert IX.ProgramIndex(no_fields).supplied == {1: ("registration", {})}


@NO_LIST
def test_the_embedding_path_passes_over_a_row_without_operands(left_out, pi):
    first = copy.copy(pi)
    first.rows = [_with({"i": -1, "spec": "X_v1{}", "name": "x"}, "args", left_out), *pi.rows]
    vars(first).pop("_embed_op", None)
    assert first.embed_op_path() == "model.embed_tokens"


@NO_LIST
@pytest.mark.parametrize("kind, named", [
    ("up", "up row 3 does not read router row 1 id slot 1 (Program dataflow None): not pair-major"),
    ("act", "act row 3 does not read up row 3 (Program dataflow None): not pair-major"),
    ("act_quant", "act_quant row 3 does not read act row 3 (Program dataflow None): not pair-major"),
    ("down", "down row 3 does not read act row 3 (Program dataflow None): not pair-major"),
    ("sum", "sum row 3 reads 0 slices, not TOPK = 2"),
])
def test_a_plane_row_without_operands_is_named_as_not_in_pair_major_order(left_out, kind, named):
    index: Any = SimpleNamespace(by_i={5: _with({"i": 5}, "args", left_out)}, _moe_kind={}, _moe_ord={}, _moe_fused=set())
    assert IX.ProgramIndex._plane_dataflow_error(index, 5, kind, 3, 2, "moe/L0") == named


@NO_DICT
def test_a_population_without_reasons_has_no_codes(left_out):
    assert RC.not_evaluable_codes(_with({"evaluable": 4}, "not_evaluable", left_out)) == {}
    why = "no evaluator for family X_v1"
    assert RC.not_evaluable_codes({"not_evaluable": {why: 3}}) == {why: {"class": "no-evaluator", "family": None, "domain": False, "tags": []}}


@NO_LIST
def test_a_request_that_states_no_steps_is_linked_over_none(left_out, world):
    bmap, committed = world.bmap_and_reader()
    stated = {"request_id": "r0", "arrive_step": 0, "prompt_token_ids": [1, 2], "steps": [0, 1, 2], "max_tokens": 3, "seed": 3}
    request = _with(stated, "steps", left_out)
    lk = LK.boundary_linkage(bmap, committed, [request])
    assert (lk["pairs"], lk["linked"], lk["horizon"]["r0"]["linked_steps"]) == (0, 0, 0)
    seed = LK.prescribed_input_linkage(OP.CommittedStore(bmap, committed), [request])
    assert seed["seed"]["checked"] == 0 and [(u["kind"], u["engine_step"]) for u in seed["unlinked"]] == [("seed", None)]
    assert seed["unlinked"][0]["why"].startswith("r0: no committed sampled_token_ids at any of its 0 declared steps")
    lk = LK.boundary_linkage(bmap, committed, [stated])
    assert (lk["pairs"], lk["linked"], lk["horizon"]["r0"]["linked_steps"]) == (3, 3, 3)
    assert LK.prescribed_input_linkage(OP.CommittedStore(bmap, committed), [stated])["seed"]["checked"] == 3


@NO_LIST
def test_a_request_that_states_no_prompt_ids_declares_an_empty_prompt(left_out, world):
    """Its arrival step is still a pair; a later step's link reads no prompt."""
    bmap, committed = world.bmap_and_reader()
    stated = {"request_id": "r0", "arrive_step": 0, "prompt_token_ids": [1, 2], "steps": [0, 1, 2]}
    lk = LK.boundary_linkage(bmap, committed, [_with(stated, "prompt_token_ids", left_out)])
    assert (lk["pairs"], lk["linked"]) == (3, 3)
    assert LK.boundary_linkage(bmap, committed, [stated | {"prompt_token_ids": [1, 3]}])["unlinked"][0]["kind"] == "prompt"


@NO_LIST
def test_a_stop_rule_s_block_that_states_no_problems_found_none(left_out, tmp_path):
    """The mixed-stop record's first request, eleven tokens served under LAG 1, under a rule that returns the block stated here."""
    block = {"executed_prefix_len": 12, "eos_step": 10, "lag_steps": 1, "first_eos_index": 10, "served_past_first_eos": 0, "eos_stop": True}
    facts = XP._linked_facts(tmp_path)

    def prefix(problems: Any) -> dict:
        return POP.executed_prefix_of_record("r0", **facts, served=XP._served({"r0": 11}, {"r0"})["r0"], eos_ids=[XP.EOS],
                                             rule=lambda served, **stated: _with(block, "problems", problems))

    record = prefix(left_out)
    assert (record["faults"], record["n_exec"], record["r3_one_mechanism"]) == ([], 12, True)
    assert record["by_rule"]["problems"] == (None if left_out is ABSENT else left_out)
    assert prefix(["the last served token is no declared EOS"])["faults"] == ["Continuation_v1 rule: the last served token is no declared EOS"]


@pytest.mark.parametrize("left_out", [ABSENT, None, 0], ids=["absent", "null", "zero"])
def test_an_executed_prefix_that_states_no_lag_anchors_its_last_phase_under_lag_0(left_out, tmp_path):
    """The mixed-stop record's first request, of thirty-two steps: as served, stopping twenty steps short of its last; and served
    thirty-one tokens, where its lag step is its last step.  Under LAG 0 neither has a lag step, nor has the second where its
    record states no served length: it is then taken to have run to its last step."""
    facts = XP._linked_facts(tmp_path)
    short, reaching = (POP.executed_prefix_of_record("r0", **facts | {"a_r": served, "served_len_commit": served}) for served in (11, 31))
    for record, served in ((short, 11), (reaching, 31)):
        anchor, stated = POP.phase_anchor(32, _with(record, "lag", left_out)), POP.phase_anchor(32, record)
        assert (record["faults"], record["n_exec"]) == ([], served + 1)
        assert (anchor["last_step"], anchor["lag_step"], "is the lag step" in anchor["why"]) == (served, None, False)
        assert (stated["last_step"], stated["lag_step"], "is the lag step" in stated["why"]) == (served, served, True)
    assert POP.phase_anchor(32, _with(short, "lag", left_out))["why"].startswith(
        "the request stopped at a_r 11 under LAG 0: steps [12, 32) are not executed; its last EXECUTED step 11 is the EOS step (LAG 0: no lag step)")
    unsaid = POP.phase_anchor(32, _with(reaching, "a_r", ABSENT))
    assert (unsaid["last_step"], unsaid["lag_step"], "is the lag step" in unsaid["why"]) == (31, None, False)



@NO_DICT
def test_a_description_that_states_no_statics_keys_its_stratum_by_its_spec(left_out, pi):
    class NoStatics:
        def __getattr__(self, name: str) -> Any:
            return getattr(pi, name)

        def describe(self, row: dict) -> dict | None:
            d = pi.describe(row)
            return d if d is None else _with(d, "statics", left_out)

    index: Any = NoStatics()
    assert [v["stratum"] for v in POP.population(index, "r0")[0]] == [v["stratum"] for v in POP.population(pi, "r0")[0]]


@NO_DICT
def test_a_plane_row_of_a_module_its_step_s_geometry_does_not_hold_has_no_router_rows_by_name(left_out):
    module = "model.layers.3.mlp.experts"
    index: Any = SimpleNamespace(step_of=lambda i: 0, _moe_geo=[_with({}, module, left_out)], _moe_ord={5: 0})
    d = IX.ProgramIndex._describe_plane(index, 5, "MoeSum_v1", {}, module, "", None, "sum")
    assert (d["op_path"], d["members"]) == ("moe/L3", [])
    assert d["plane_error"] == f"MoE module {module!r} step 0: no router rows / TOPK in the Program (M=0, TOPK=0)"
