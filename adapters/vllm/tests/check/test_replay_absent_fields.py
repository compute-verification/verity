"""What the readers of `verity_vllm.check.replay` answer for a field its document leaves out: a Program row without operands and
an instances document without supplied inputs (`verity_vllm.check.replay.index`, and the row's readers in `evaluate`, `population`
and `driver`), a binding map without entries and an entry without an identity or an encoding (`verity_vllm.check.replay.opening`),
a manifest without a query, aliases or a cross-check's counts, an engine schedule without observed arrive steps
(`verity_vllm.check.replay.linkage`), a population without reasons (`verity_vllm.check.replay.replay_codes`).  Each field is given
absent, null and empty, and the answer is the same for the three; where a document can state the field, the stated answer stands
beside them."""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

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
