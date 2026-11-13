"""G3, G4 and G5 of the global match for one request (`verity_vllm.check.match.per_request.check_requests`), over folds built
by hand as a loader leaves them: no Program is decoded and no record is read.

A warm request: the record serves its first rows from the prefix cache, and the fold reads them where the request that
computed them left them (`verity_vllm.check.match.prefix_reads`, the fold of `test_prefix_reads.py`).  Compared with the
Program that reads them from its prefix input, the three checks pass and the request's entry carries the hit.  G5 names a hit
the Program does not declare, a served row count other than the declared one, cached rows no other request's row of the same
layer and position gives, and a block read from another request than the declared one.

The producers of a request's operands by engine step: G5 names the runs whose producer is in a later engine step than their
consumer and counts by family the ones whose producer is in an earlier one, a request that reads the weights is one whose
operand reads a weights node, and a run in the body's parameter space is none of these.  The record's loop gives this in the
record's form and in the compact one; under the fast implementation the record's form is counted by
`verity_vllm.check.match.global_match_fast.producer_chronology`, and the three agree."""
from __future__ import annotations

import pytest

from tests.check import test_match_operand_reads as reads
from tests.check import test_prefix_reads as warm
from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match import global_match_fast as gmf
from verity_vllm.check.match import per_request
from verity_vllm.check.match.attribution import Attribution
from verity_vllm.check.match.instance_sequence import RootView
from verity_vllm.check.match.match_run import MatchRun

FORMS = pytest.mark.parametrize("compact", [False, True], ids=["record", "compact"])


@pytest.fixture(autouse=True)
def _no_memo_is_left():
    yield
    gmf.reset()
    gmf.reset_projections()


def _checked(fold: RootView, request: dict, instances: list[int], step_of: dict[int, int], rows: dict[int, list[tuple]], *,
             weights: int, fast: bool = False, caching: bool = False, components: dict | None = None):
    """`check_requests` for the one declared `request`, whose `instances` of the fold are in the engine steps `step_of` gives by
    instance; `rows` are the record's rows by engine step.  Returns the document's checks by id, the request's entry and the facts."""
    res: dict = {"checks": [], "requests": []}
    rec = {"header": {"profile": {"engine_kwargs": {"enable_prefix_caching": caching}}}, "steps": {s: {"rows": r} for s, r in rows.items()},
           "samples": {}}
    declared = {"index": 0, "max_tokens": 4, "arrive_step": 0, **request}
    run = MatchRun(gp={"requests": [declared]}, oracle_dir="", tokens_path="", within_step="sequence", fast=fast, ops=bd.record_ops(),
                   loaded={"components": dict(components or {})}, tokens=None, tokens_optional=True, rec=rec, res=res, fold=fold,
                   declared={declared["request_id"]: declared})
    att = Attribution(weights_nodes=frozenset({weights}), weights_nodes_error=None,
                      meta_of={fold.inst[i]["k"]: {"step": s} for i, s in step_of.items()}, per_req={declared["request_id"]: instances},
                      unattributed=[], shared_idx=[], shared_gi=frozenset(), peer_index={})
    facts = per_request.check_requests(run, att, {declared["request_id"]: declared["prompt_node"]})
    (entry,) = res["requests"]
    return {c["id"]: c for c in res["checks"]}, entry, facts


# ------------------------------------------------------------------------------------------------ a warm request

R1 = {"request_id": "r1", "prompt_node": 1, "prompt_len": warm.LP}
HIT = {"rows": warm.C_ROWS, "producers": ["r0"]}
SERVED = "r1: the engine served {} prompt rows from the prefix cache (its first scheduled position), the Program declares {}"
UNLINKED = "r1: {} of its {} cached rows are read from no other request's row of the same layer and position"


def _warm(tmp_path, compact: bool, request: dict, *, first_row: int = warm.C_ROWS, component: bool = True, **fold):
    """r1 of the fold of `test_prefix_reads.py`, in engine step 1 after r0's rows of step 0; the record's first row of r1 is at
    position `first_row`, under prefix caching the rows the cache served it."""
    F, mine = warm.fold(compact, **fold)
    step_of = {i: int(i in mine) for i in range(len(F.inst))}
    rows = {0: [("r0", j) for j in range(warm.C_ROWS)], 1: [("r1", first_row)]}
    components = {"warm": warm.warm_program(tmp_path, compact)} if component else {}
    declared = {**request, "component": {"dir": "warm", "LP": warm.LP, "T": 0}} if component else request
    return _checked(F, declared, mine, step_of, rows, weights=2, caching=True, components=components)


@FORMS
def test_a_warm_request_passes_against_the_program_that_reads_its_prefix_input_and_its_entry_carries_the_hit(tmp_path, compact):
    checks, entry, facts = _warm(tmp_path, compact, {**R1, "prefix": HIT})
    assert [(cid, c["status"], c["problems"]) for cid, c in checks.items()] == [("G3", "PASS", []), ("G4", "PASS", []), ("G5", "PASS", [])]
    assert entry["prefix"] == {"rows": warm.C_ROWS, "declared_rows": warm.C_ROWS, "block_size": warm.C_ROWS, "declared_producers": ["r0"],
                               "read_from": [["r0"]], "linked_rows": warm.C_ROWS}
    assert (entry["executed_steps_equal"], entry["operands"], entry["token_selects"]) == (True, {"equal": 19, "different": 0}, 1)
    assert checks["G4"]["mapped"] == len(facts.glob2loc) == entry["n_instances"] == 9
    # each of the request's two attentions reads two cached rows' keys and values of step 0
    assert checks["G5"]["cross_step_producer_classes"] == {"Attention_v3": 2 * 2 * warm.C_ROWS}


@FORMS
@pytest.mark.parametrize(("request_", "how", "problems", "hit"), [
    pytest.param(R1, {}, [SERVED.format(2, 0)],
                 {"rows": 2, "declared_rows": 0, "block_size": None, "declared_producers": [], "read_from": [], "linked_rows": 2},
                 id="a hit the Program does not declare"),
    pytest.param({**R1, "prefix": HIT}, {"first_row": 3}, [SERVED.format(3, 2), UNLINKED.format(1, 3)],
                 {"rows": 3, "declared_rows": 2, "block_size": 2, "declared_producers": ["r0"], "read_from": [["r0"]], "linked_rows": 2},
                 id="a row more served than the fold reads"),
    pytest.param({**R1, "prefix": {"rows": 2, "producers": ["r2"]}}, {},
                 ["r1: cached block 0 was read from ['r0'], the Program declares 'r2' (1 block(s) differ)"],
                 {"rows": 2, "declared_rows": 2, "block_size": 2, "declared_producers": ["r2"], "read_from": [["r0"]], "linked_rows": 2},
                 id="another producer than the declared one"),
    pytest.param({**R1, "prefix": HIT}, {"read_row": lambda layer, j: (layer, warm.C_ROWS - 1 - j)},
                 ["r1: 8 cross-request operand reads {'prefix-unlinked': 8}", UNLINKED.format(2, 2),
                  "r1: cached block 0 was read from [], the Program declares 'r0' (1 block(s) differ)"],
                 {"rows": 2, "declared_rows": 2, "block_size": 2, "declared_producers": ["r0"], "read_from": [[]], "linked_rows": 0},
                 id="rows read from the producer's other position"),
])
def test_g5_names_a_hit_that_is_not_the_declared_one_by_rows_and_by_producer(tmp_path, compact, request_, how, problems, hit):
    """The request has no component here, which G3 says and G4 follows; G5's problems are the hit's alone, and the entry
    carries what was served, declared and read."""
    checks, entry, _facts = _warm(tmp_path, compact, request_, component=False, **how)
    assert checks["G5"]["problems"] == problems and entry["prefix"] == hit
    assert checks["G3"]["problems"] == [f"r1: no component Program in the global Program for shape (LP {warm.LP}, T_bound None)"]


@FORMS
def test_a_request_the_cache_served_nothing_and_the_program_declares_cold_has_no_prefix_entry(tmp_path, compact):
    """Without prefix caching in the record's header the first row's position is no hit: the reads of r0's rows are
    cross-request reads of nodes, and the entry has no `prefix`."""
    F, mine = warm.fold(compact)
    checks, entry, _facts = _checked(F, R1, mine, {i: int(i in mine) for i in range(len(F.inst))},
                                     {0: [("r0", 0), ("r0", 1)], 1: [("r1", warm.C_ROWS)]}, weights=2)
    assert checks["G5"]["problems"] == ["r1: 8 cross-request operand reads {'node': 8}"] and "prefix" not in entry


# ------------------------------------------------------------------------------------------------ producers by engine step

PROMPT, OTHER_PROMPT, WTS = range(3)
INPUTS = {PROMPT: ("Input32_v1", 3), OTHER_PROMPT: ("Input32_v1", 2), WTS: ("Input16_v1", 8)}
WEIGHTS_MAP = {"oracle_wmap": [(0, 8, "model.a.weight")], "_wlos": [0], "oracle_wmap_overlaps": [], "input_base": {WTS: 0}}
EMBED_0, READ_0, SELECT_0, EMBED_1, READ_NO_STEP, SELECT_1 = range(3, 9)                # the nodes; instance i is node i + 3
STEP_OF = {0: 0, 1: 0, 2: 0, 3: 1, 5: 1}                                              # the fifth instance has no engine step
ROWS = {0: [("r0", 0), ("r0", 1), ("r0", 2)], 1: [("r0", 3)]}
R0 = {"request_id": "r0", "prompt_node": PROMPT, "prompt_len": 3}
LATER = "r0: 2 operands produced in a LATER engine step than their consumer"
NO_WEIGHTS = "requests that never read the shared weights input: ['r0']"


def _two_steps(compact: bool, weights: list[tuple]) -> RootView:
    """One request over two engine steps; `weights` is the second operand of the first step's embedding."""
    return reads._view(INPUTS, [
        (EMBED_0, reads.EMBED, [[("n", PROMPT, 0, 3)], weights]),
        # an earlier instance of its step, then two windows of the embedding of step 1
        (READ_0, reads.READER, [[("n", EMBED_0, 0, 2), ("n", EMBED_1, 0, 2), ("n", EMBED_1, 4, 6)]]),
        (SELECT_0, reads.SELECT, [[("n", READ_0, 0, 8)]]),
        # the select of step 0, then a run in the body's parameter space whose index is the weights node's
        (EMBED_1, reads.EMBED, [[("n", SELECT_0, 0, 1)], [("p", WTS, 0, 8)]]),
        # in no engine step: a stride over three instances of step 0, then the select that follows it
        (READ_NO_STEP, reads.READER, [[("n", EMBED_0, 0, 1), ("n", READ_0, 0, 1), ("n", SELECT_0, 0, 1)], [("n", SELECT_1, 0, 1)]]),
        (SELECT_1, reads.SELECT, [[("n", EMBED_1, 0, 8)]]),
    ], compact=compact, **WEIGHTS_MAP)


@pytest.mark.parametrize(("compact", "fast"), [(False, False), (True, False), (False, True), (True, True)],
                         ids=["record", "compact", "fast", "fast-compact"])
@pytest.mark.parametrize(("weights", "reads_weights"), [([("n", WTS, 0, 8)], True), ([("p", WTS, 0, 8)], False)],
                         ids=["weights-node", "parameter-space"])
def test_g5_counts_a_requests_producers_by_engine_step_alike_in_both_forms_and_on_the_fast_path(compact, fast, weights, reads_weights):
    """A run of a later step's instance is named, a window each; a run of an earlier step's is counted under its consumer's
    family; a run of the same step's is neither, and an instance in no engine step counts nothing.  Only a read of the weights
    node makes the request one that reads the weights: a run in the body's parameter space does not, whatever its index."""
    checks, entry, facts = _checked(_two_steps(compact, weights), R0, list(range(6)), STEP_OF, ROWS, weights=WTS, fast=fast)
    assert checks["G5"]["problems"] == ([LATER] if reads_weights else [LATER, NO_WEIGHTS])
    assert checks["G5"]["cross_step_producer_classes"] == {"Embedding_v1": 1}
    assert (entry["n_instances"], entry["local_steps"], entry["token_selects"], entry["T_observed"]) == (6, 2, 2, 1)
    # the order inside a request's step is G8's: a record a run whose producer is emitted after its consumer
    assert facts.dep_viol == [{"request_id": "r0", "local_step": 0, "emission": 1, "consumer": 1, "producer": 3}] * 2 + \
        [{"request_id": "r0", "local_step": 1, "emission": 1, "consumer": 4, "producer": 5}]
    assert facts.engine_step_of == STEP_OF
