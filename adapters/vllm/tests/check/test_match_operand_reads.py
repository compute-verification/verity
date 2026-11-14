"""What a view's canon says of an operand's reads, in the two forms a loader gives an operand: the record's runs
`(space, idx, lo, hi)` and the compact form's progressions `(space, (idx0, lo0, hi0), (dI, dL, dH), count)`
(`verity_vllm.program.compact`).  A comparison holds two views' canonical operands equal tuple for tuple, so the compact form's
sources have to be a function of the record's: each source as a progression of one run, then the greedy RLE (`RootView.canon`).

Held here by kind of read: a parameter of the body, the prompt, the seed, a supplied field, a window of the fold's weights map, a
warm request's prefix input, a witness and peers (one read of peers is not such a function yet, and is held as it is); a top-p
select's split count; a read through a mux that the comparison unfolds; and, in one request's projection of a fold, the other
request's parameters and nodes.

The views are built by hand as a loader leaves them, the way `test_compare_tp_peer_split.py` builds its own: no Program is decoded."""
from __future__ import annotations

import pytest

from verity_vllm.check.match.canonical_operands import as_progression
from verity_vllm.check.match.instance_sequence import RootView
from verity_vllm.check.match.program_compare import compare_steps
from verity_vllm.check.match.projection import Projection
from verity_vllm.program import compact as K

READER, SELECT = "Concat_v1{N=8}", "TokenSelect_v1{V=8}"
FORMS = pytest.mark.parametrize("compact", [False, True], ids=["record", "compact"])


def _view(inputs: dict[int, tuple[str, int]], inst: list[tuple], *, compact: bool, **loaded) -> RootView:
    """A view as a loader leaves it.  `inputs`: the Input nodes; `inst`: the instances `(node, spec, operands)` in node order, an
    operand the record's runs, which `compact` gives as the compact loader does (`compact.from_runs`); `loaded`: what a loader binds."""
    P = RootView.__new__(RootView)
    P.label, P.digest, P.compact = "fold", "digest", compact
    P.inputs = inputs
    P.tok_node, P.seed_nodes, P.seed_index = 0, set(), {}
    P.weight_fields, P.param_names, P.oracle_rows, P.n_nodes, P.nodes = {}, [], None, 0, None
    P.supplied_fields, P.supplied_owner, P.peer_fields, P.peer_index, P.witness_fields, P.witness_index = {}, {}, {}, {}, {}, {}
    for name, value in loaded.items():
        setattr(P, name, value)
    P.inst = [{"k": k, "spec": spec, "form": "call", "args": [K.from_runs(a) if compact else list(a) for a in args], "out": "bf16"}
              for k, spec, args in inst]
    P.k2i = {r["k"]: i for i, r in enumerate(P.inst)}
    P._bind_splits()
    P._segment()
    return P


def _reader(inputs: dict[int, tuple[str, int]], operands: list[list[tuple]], *, compact: bool, **loaded) -> RootView:
    """One engine step whose first instance has `operands`; a select closes the step."""
    k = max(inputs) + 1
    return _view(inputs, [(k, READER, operands), (k + 1, SELECT, [[("n", k, 0, 8)]])], compact=compact, **loaded)


def _as_compact(record: list[tuple]) -> list[tuple]:
    """The compact form's sources that the record's give: each a progression of one run, then the greedy RLE."""
    return K.rle_mapped([as_progression(t) for t in record])


# ------------------------------------------------------------------------------------------------ one request's Program

TOK, SEED, WTS, VALUES, PREFIX, WIT = range(6)   # the prompt, the seed, the weights, the registered values, a warm request's prefix, a witness
INPUTS = {TOK: ("Input32_v1", 4), SEED: ("Input64_v1", 1), WTS: ("Input16_v1", 24), VALUES: ("Input32_v1", 3),
          PREFIX: ("Input16_v1", 32), WIT: ("Input32_v1", 2)}
# What a fold's loader binds: the weights map's two fields (leaves 0..8 and 8..24), three registered values of one leaf each, the
# prefix input's four planes of 8 leaves (two rows of 4 leaves: layer 3's key and value, then layer 5's) and the witness's name.
LOADED = {
    "seed_nodes": {SEED}, "seed_index": {SEED: 0},
    "oracle_wmap": [(0, 8, "model.a.weight"), (8, 24, "model.b.weight")], "_wlos": [0, 8], "oracle_wmap_overlaps": [], "input_base": {WTS: 0},
    "supplied_fields": {VALUES: [(0, "eos", 32), (1, "pad", 32), (2, "bos", 32)]}, "supplied_owner": {VALUES: None},
    "pfx": {PREFIX: (2, 4, [3, 5])},
    "witness_fields": {WIT: "schedule"},
}
READS = [   # (what the operand reads, its runs, the record's sources, the compact form's)
    ("a parameter of the body", [("p", 0, 0, 4)], [("p", 0, 0, 4)], [("p", (0, 0, 4), (0, 0, 0), 1)]),
    ("windows of the prompt", [("n", TOK, 0, 1), ("n", TOK, 2, 3)], [("TOK", 0, 1), ("TOK", 2, 3)], [("TOK", (0, 1), (2, 2), 2)]),
    ("the seed", [("n", SEED, 0, 1)], [("SEED", 0, 1)], [("SEED", (0, 1), (0, 0), 1)]),
    ("the prompt, then the seed", [("n", TOK, 0, 1), ("n", SEED, 0, 1)], [("TOK", 0, 1), ("SEED", 0, 1)],
     [("TOK", (0, 1), (0, 0), 1), ("SEED", (0, 1), (0, 0), 1)]),
    ("one registered value", [("n", VALUES, 1, 2)], [("SUPPLIED", "pad", 0, 1)], [("SUPPLIED", "pad", (0, 1), (0, 0), 1)]),
    ("one registered value, twice", [("n", VALUES, 1, 2), ("n", VALUES, 1, 2)], [("SUPPLIED", "pad", 0, 1), ("SUPPLIED", "pad", 0, 1)],
     [("SUPPLIED", "pad", (0, 1), (0, 0), 2)]),
    ("two registered values", [("n", VALUES, 0, 1), ("n", VALUES, 2, 3)], [("SUPPLIED", "eos", 0, 1), ("SUPPLIED", "bos", 0, 1)],
     [("SUPPLIED", "eos", (0, 1), (0, 0), 1), ("SUPPLIED", "bos", (0, 1), (0, 0), 1)]),
    ("a run over two registered values", [("n", VALUES, 0, 2)], [("SUPPLIED?", VALUES, 0, 2)], [("SUPPLIED?", (VALUES, 0, 2), (0, 0, 0), 1)]),
    ("windows of one weights field", [("n", WTS, 10, 12), ("n", WTS, 14, 16)], [("W", "model.b.weight", 2, 4), ("W", "model.b.weight", 6, 8)],
     [("W", "model.b.weight", (2, 4), (4, 4), 2)]),
    ("windows of two weights fields", [("n", WTS, 4, 6), ("n", WTS, 10, 12)], [("W", "model.a.weight", 4, 6), ("W", "model.b.weight", 2, 4)],
     [("W", "model.a.weight", (4, 6), (0, 0), 1), ("W", "model.b.weight", (2, 4), (0, 0), 1)]),
    ("a run over two weights fields", [("n", WTS, 6, 10)], [("W?", WTS, 6, 10)], [("W?", (WTS, 6, 10), (0, 0, 0), 1)]),
    ("a prefix plane", [("n", PREFIX, 16, 24)], [("PFX", 5, 0, 0, 8)], [("PFX", 5, 0, (0, 8), (0, 0), 1)]),
    ("windows of one prefix plane", [("n", PREFIX, 8, 10), ("n", PREFIX, 12, 14)], [("PFX", 3, 1, 0, 2), ("PFX", 3, 1, 4, 6)],
     [("PFX", 3, 1, (0, 2), (4, 4), 2)]),
    ("windows of two prefix planes", [("n", PREFIX, 0, 4), ("n", PREFIX, 8, 12)], [("PFX", 3, 0, 0, 4), ("PFX", 3, 1, 0, 4)],
     [("PFX", 3, 0, (0, 4), (0, 0), 1), ("PFX", 3, 1, (0, 4), (0, 0), 1)]),
    ("a run over two prefix planes", [("n", PREFIX, 4, 12)], [("PFX?", PREFIX, 4, 12)], [("PFX?", (PREFIX, 4, 12), (0, 0, 0), 1)]),
    ("a run past the prefix planes", [("n", PREFIX, 32, 36)], [("PFX?", PREFIX, 32, 36)], [("PFX?", (PREFIX, 32, 36), (0, 0, 0), 1)]),
    ("the witness", [("n", WIT, 0, 2)], [("WIT", "schedule", 0, 2)], [("WIT", "schedule", (0, 2), (0, 0), 1)]),
]


@pytest.mark.parametrize(("runs", "record", "compact"), [pytest.param(*case[1:], id=case[0]) for case in READS])
def test_a_read_has_the_records_sources_and_in_the_compact_form_their_progressions(runs, record, compact):
    """Each kind of read under its own head, relative to its field or plane; a read that lies in no one field keeps its node under
    a head that ends in `?`.  The compact form's sources are the record's, each a progression of one run, under the greedy RLE: a
    progression inside one field stays one, and one that crosses fields is the record's sources run by run."""
    assert _reader(INPUTS, [runs], compact=False, **LOADED).canon(0) == [record]
    assert _reader(INPUTS, [runs], compact=True, **LOADED).canon(0) == [compact]
    assert compact == _as_compact(record) and K.expand_mixed(compact) == record


# ------------------------------------------------------------------------------------------------ a rank's peers

PEERS = 2   # after the prompt and the weights
PEER_READS = [   # (what the operand reads, the Input's fields `(leaf offset, ordinal)`, its runs, the record's sources)
    ("windows of one field", [(0, 7), (8, 8)], [("n", PEERS, 8, 10), ("n", PEERS, 12, 14)], [("PEER", 8, 0, 2), ("PEER", 8, 4, 6)]),
    ("windows of two fields", [(0, 7), (8, 8)], [("n", PEERS, 4, 6), ("n", PEERS, 10, 12)], [("PEER", 7, 4, 6), ("PEER", 8, 2, 4)]),
    ("a run over two fields", [(0, 7), (8, 8)], [("n", PEERS, 4, 12)], [("PEER", 7, 4, 8), ("PEER", 8, 0, 4)]),
    ("a run below the first field", [(4, 7)], [("n", PEERS, 0, 2)], [("PEER?", PEERS, 0, 2)]),
]


@pytest.mark.parametrize(("fields", "runs", "record"), [pytest.param(*case[1:], id=case[0]) for case in PEER_READS])
def test_a_peers_read_is_under_its_fields_ordinal_and_the_compact_form_expands_to_the_records_sources(fields, runs, record):
    """A peers read is `("PEER", k, lo, hi)`, k the ordinal of the field it lies in and lo and hi relative to that field; a run
    over adjacent fields is one source a field.  The compact form's sources expand to the record's."""
    inputs = {TOK: ("Input32_v1", 4), 1: ("Input16_v1", 8), PEERS: ("Input16_v1", 16)}
    assert _reader(inputs, [runs], compact=False, peer_fields={PEERS: fields}).canon(0) == [record]
    assert K.expand_mixed(_reader(inputs, [runs], compact=True, peer_fields={PEERS: fields}).canon(0)[0]) == record


def _peers_fold(runs: list[tuple], *, compact: bool) -> RootView:
    """A fold's reader of a request's peers: one Record, a field of 8 leaves a peer, the ordinals 7 and 8."""
    return _reader({TOK: ("Input32_v1", 4), 1: ("Input16_v1", 8), PEERS: ("Input16_v1", 16)}, [runs], compact=compact,
                   peer_fields={PEERS: [(0, 7), (8, 8)]})


def _peers_program(runs: list[tuple], *, compact: bool) -> RootView:
    """A rank Program's reader of the same peers: one Input of 8 leaves a peer."""
    return _reader({TOK: ("Input32_v1", 4), 1: ("Input16_v1", 8), PEERS: ("Input16_v1", 8), PEERS + 1: ("Input16_v1", 8)}, [runs],
                   compact=compact, peer_fields={PEERS: [(0, 7)], PEERS + 1: [(0, 8)]})


@FORMS
@pytest.mark.parametrize(("fold_runs", "program_runs"), [
    pytest.param([("n", PEERS, 8, 10), ("n", PEERS, 12, 14)], [("n", PEERS + 1, 0, 2), ("n", PEERS + 1, 4, 6)], id="windows of one peer"),
    pytest.param([("n", PEERS, 0, 2), ("n", PEERS, 8, 10)], [("n", PEERS, 0, 2), ("n", PEERS + 1, 0, 2)], id="a window of each peer"),
    pytest.param([("n", PEERS, 0, 2), ("n", PEERS, 8, 10), ("n", PEERS, 12, 14)],
                 [("n", PEERS, 0, 2), ("n", PEERS + 1, 0, 2), ("n", PEERS + 1, 4, 6)], id="a window of one peer, then two of the next"),
])
def test_windows_of_peers_are_the_same_sources_in_a_folds_one_record_and_in_a_rank_programs_inputs(fold_runs, program_runs, compact):
    """The fold holds a request's peers in one Record, a field a peer at increasing offsets; the rank Program holds one Input a
    peer.  In these three reads the same leaves read from either are the same sources, in the record's form and in the compact one."""
    assert _peers_fold(fold_runs, compact=compact).canon(0) == _peers_program(program_runs, compact=compact).canon(0)


def test_a_stride_over_two_peers_fields_is_one_operand_in_the_records_form_and_counted_different_in_the_compact_one():
    """Two windows of each peer, one stride apart across the two fields.  The record's sources are the same from the fold's Record
    and from the rank Program's Inputs.  The compact sources are not, though both expand to the record's: the fold reads one
    progression over both fields, which is the record's sources run by run with the ordinal in each base, and the rank Program
    reads one progression an Input, which has the ordinal in its head (`PeerReads._peer_srcP`).  A comparison in the compact form
    then counts one different operand under equal hashes.  Today's answer, held so that a change of it shows; the two forms
    should agree."""
    fold_runs = [("n", PEERS, 0, 2), ("n", PEERS, 4, 6), ("n", PEERS, 8, 10), ("n", PEERS, 12, 14)]
    program_runs = [("n", PEERS, 0, 2), ("n", PEERS, 4, 6), ("n", PEERS + 1, 0, 2), ("n", PEERS + 1, 4, 6)]
    record = [("PEER", 7, 0, 2), ("PEER", 7, 4, 6), ("PEER", 8, 0, 2), ("PEER", 8, 4, 6)]
    fold, program = _peers_fold(fold_runs, compact=False), _peers_program(program_runs, compact=False)
    assert fold.canon(0) == program.canon(0) == [record]
    same = compare_steps(program, fold)
    assert same["totals"]["operands_different"] == 0 and same["canonical_sha256"]["equal"]

    fold, program = _peers_fold(fold_runs, compact=True), _peers_program(program_runs, compact=True)
    assert fold.canon(0) == [[("PEER", (7, 0, 2), (0, 4, 4), 2), ("PEER", (8, 0, 2), (0, 4, 4), 2)]]
    assert program.canon(0) == [[("PEER", 7, (0, 2), (4, 4), 2), ("PEER", 8, (0, 2), (4, 4), 2)]]
    assert K.expand_mixed(fold.canon(0)[0]) == K.expand_mixed(program.canon(0)[0]) == record
    counted = compare_steps(program, fold)
    assert counted["totals"]["operands_different"] == 1 and counted["canonical_sha256"]["equal"]


# ------------------------------------------------------------------------------------------------ a top-p select's split count

TOP_P = "GumbelTopPTokenSelect_v1{V=16}"
N_CONSTS = 3   # the constants each step holds before the ones `consts` adds


def _top_p_steps(consts: int, sixth, *, compact: bool) -> RootView:
    """Two engine steps of a top-p request.  A step is a gemm, three constants, `consts` more `Const32[0x20]_v1` and the select;
    `sixth(c)` is the select's sixth operand, the split count's, c the node of the first of those `consts`."""
    inst: list[tuple] = []
    k = max(INPUTS) + 1
    for _step in range(2):
        c = k + 1 + N_CONSTS
        inst += [(k, "Gemm_v1{M=1,N=16,K=8}", [[("n", TOK, 0, 4)], [("n", WTS, 0, 8)]]),
                 (k + 1, "Const32[0x3f733333]_v1", []), (k + 2, "Const64[0x3]_v1", []), (k + 3, "Const32[0x3f4ccccd]_v1", [])]
        inst += [(c + j, "Const32[0x20]_v1", []) for j in range(consts)]
        inst.append((c + consts, TOP_P, [[("n", k, 0, 16)], [("n", k + 1, 0, 1)], [("n", SEED, 0, 1)], [("n", k + 2, 0, 1)],
                                         [("n", k + 3, 0, 1)], sixth(c)]))
        k = c + consts + 1
    return _view(INPUTS, inst, compact=compact, **LOADED)


@FORMS
def test_a_top_p_selects_own_const32_is_its_bound_split_count_in_both_forms(compact):
    """The `Const32` a top-p select alone reads, whole, as its sixth operand is the select's split count: it leaves the sequence,
    its value is kept under the select's node, and the operand is `("SPLITS", e, e + 1)`, e the select's ordinal."""
    P = _top_p_steps(1, lambda c: [("n", c, 0, 1)], compact=compact)
    first = max(INPUTS) + 1
    assert (P.n_splits_consts_bound, P.splits_values) == (2, {first + 5: 0x20, first + 11: 0x20})
    assert [[r["spec"] for r in (P.inst[i] for i in step)].count("Const32[0x20]_v1") for step in P.steps] == [0, 0]
    assert [len(step) for step in P.steps] == [5, 5]
    sixths = [P.canon(step[-1])[5] for step in P.steps]
    assert sixths == ([[("SPLITS", (0, 1), (0, 0), 1)], [("SPLITS", (1, 2), (0, 0), 1)]] if compact else [[("SPLITS", 0, 1)], [("SPLITS", 1, 2)]])


@FORMS
@pytest.mark.parametrize(("consts", "sixth", "record"), [
    pytest.param(1, lambda c: [("n", c, 1, 2)], lambda e: [("N", e, 4, 1, 2)], id="a run that is not the whole Const32"),
    pytest.param(1, lambda c: [("n", c, 0, 1), ("n", c, 2, 3)], lambda e: [("N", e, 4, 0, 1), ("N", e, 4, 2, 3)], id="two windows of one Const32"),
    pytest.param(2, lambda c: [("n", c, 0, 1), ("n", c + 1, 0, 1)], lambda e: [("N", e, 4, 0, 1), ("N", e, 5, 0, 1)], id="two Const32s"),
    pytest.param(1, lambda c: [("p", 0, 0, 1)], lambda e: [("p", 0, 0, 1)], id="a parameter of the body"),
])
def test_a_sixth_operand_that_is_not_one_whole_const32_binds_no_split_count(consts, sixth, record, compact):
    """Such an operand names no one constant: nothing leaves the sequence, and the operand is compared as the reads it is."""
    P = _top_p_steps(consts, sixth, compact=compact)
    assert (P.n_splits_consts_bound, P.splits_values) == (0, {})
    assert [len(step) for step in P.steps] == [5 + consts, 5 + consts]
    sixths = [P.canon(step[-1])[5] for step in P.steps]
    assert sixths == [_as_compact(record(e)) if compact else record(e) for e in (0, 1)]


# ------------------------------------------------------------------------------------------------ a mux the comparison unfolds

EMBED, ZERO, MUX, MUX_ROWS = "Embedding_v1{V=8,H=2}", "Const32[0x0]_v1", "SelectI32_v1", "batch:SelectBf16_v1"
MUX_INPUTS = {TOK: ("Input32_v1", 4), 1: ("Input16_v1", 8)}
E, Z, M, M_ROWS, M_OF_M, READ = range(2, 8)   # the nodes; in the step, instance j is node j + 2
MUX_INST = [
    (E, EMBED, [[("n", TOK, 0, 4)], [("n", 1, 0, 8)]]),
    (Z, ZERO, []),
    (M, MUX, [[("n", Z, 0, 1)], [("n", E, 2, 6)], [("n", Z, 0, 1)]]),                           # live: the embedding's leaves 2..6
    (M_ROWS, MUX_ROWS, [[("n", Z, 0, 1)], [("n", E, 0, 2), ("n", E, 4, 6)], [("n", Z, 0, 1)]]),  # live: two runs
    (M_OF_M, MUX, [[("n", Z, 0, 1)], [("n", M, 1, 3)], [("n", Z, 0, 1)]]),                      # live: the first mux's leaves 1..3
]
MUX_READS = [   # (what the operand reads, its runs, the sources as the comparison unfolds, the sources as it does not)
    ("a mux", [("n", M, 0, 2)], [("N", 0, 0, 2, 4)], [("N", 0, 2, 0, 2)]),
    ("windows of a mux", [("n", M, 0, 1), ("n", M, 2, 3)], [("N", 0, 0, 2, 3), ("N", 0, 0, 4, 5)], [("N", 0, 2, 0, 1), ("N", 0, 2, 2, 3)]),
    ("the live value, then the mux", [("n", E, 0, 2), ("n", M, 0, 2)], [("N", 0, 0, 0, 2), ("N", 0, 0, 2, 4)], [("N", 0, 0, 0, 2), ("N", 0, 2, 0, 2)]),
    ("a mux of a mux", [("n", M_OF_M, 0, 1)], [("N", 0, 0, 3, 4)], [("N", 0, 4, 0, 1)]),
    ("a mux whose live operand is two runs", [("n", M_ROWS, 0, 2)], [("MUX-UNFOLD-FAILED", 3, 0, 2)], [("N", 0, 3, 0, 2)]),
]


@pytest.mark.parametrize(("runs", "unfolded", "folded"), [pytest.param(*case[1:], id=case[0]) for case in MUX_READS])
def test_a_read_through_a_mux_is_a_read_of_its_live_operand_when_the_comparison_unfolds(runs, unfolded, folded):
    """A padded Program holds a mux where a flat one holds the live value, `SelectI32_v1` or a batch of `SelectBf16_v1` whose second
    operand is the live one.  Unfolding, a read of the mux is the read of the same leaves of that operand, through a mux of a mux
    too; a mux whose live operand is not one run is named by its instance.  Otherwise a read of a mux is a read of an instance like
    any other.  In the record's form, the one `program-compare` unfolds."""
    P = _view(MUX_INPUTS, [*MUX_INST, (READ, READER, [runs]), (READ + 1, SELECT, [[("n", READ, 0, 8)]])], compact=False,
              weight_fields={1: "model.a.weight"})
    assert P.canon(READ - 2, True) == [unfolded]
    assert P.canon(READ - 2, False) == P.canon(READ - 2) == [folded]


# ------------------------------------------------------------------------------------------------ a request's projection of a fold

# A fold of two requests: each one's prompt, seed, `run_<i>` and witness, r1's peers, the registered values and the weights.
R0_PROMPT, R1_PROMPT, R0_SEED, R1_SEED, R0_RUN, R1_RUN, R0_WIT, R1_WIT, R1_PEERS, FOLD_VALUES, FOLD_WTS = range(11)
FOLD_INPUTS = {R0_PROMPT: ("Input32_v1", 3), R1_PROMPT: ("Input32_v1", 2), R0_SEED: ("Input64_v1", 1), R1_SEED: ("Input64_v1", 1),
               R0_RUN: ("Input32_v1", 3), R1_RUN: ("Input32_v1", 3), R0_WIT: ("Input32_v1", 2), R1_WIT: ("Input32_v1", 2),
               R1_PEERS: ("Input16_v1", 8), FOLD_VALUES: ("Input32_v1", 2), FOLD_WTS: ("Input16_v1", 8)}
RUN_FIELDS = [(0, "stop", 32), (1, "bias", 32), (2, "top_k", 32)]
FOLD_LOADED = {
    "seed_nodes": {R0_SEED, R1_SEED}, "seed_index": {R0_SEED: 0, R1_SEED: 1},
    "oracle_wmap": [(0, 8, "model.a.weight")], "_wlos": [0], "oracle_wmap_overlaps": [], "input_base": {FOLD_WTS: 0},
    "supplied_fields": {FOLD_VALUES: [(0, "eos", 32), (1, "pad", 32)], R0_RUN: RUN_FIELDS, R1_RUN: RUN_FIELDS},
    "supplied_owner": {FOLD_VALUES: None, R0_RUN: 0, R1_RUN: 1},
    "witness_fields": {R0_WIT: "schedule", R1_WIT: "schedule"}, "witness_index": {R0_WIT: 0, R1_WIT: 1},
    "peer_fields": {R1_PEERS: [(0, 5)]}, "peer_index": {R1_PEERS: 1},
}
R0_READ, R1_EMBED = 11, 12   # r0's instance that reads, r1's embedding
FOLD_READS: list[tuple[str, list[tuple], list[tuple]]] = [   # (what r0's operand reads, its runs, the record's sources in r0's projection)
    ("its own prompt", [("n", R0_PROMPT, 0, 3)], [("TOK", 0, 3)]),
    ("its own seed", [("n", R0_SEED, 0, 1)], [("SEED", 0, 1)]),
    ("fields of its own run parameter", [("n", R0_RUN, 0, 1), ("n", R0_RUN, 2, 3)], [("SUPPLIED", "stop", 0, 1), ("SUPPLIED", "top_k", 0, 1)]),
    ("a registered value", [("n", FOLD_VALUES, 0, 1)], [("SUPPLIED", "eos", 0, 1)]),
    ("its own witness", [("n", R0_WIT, 0, 2)], [("WIT", "schedule", 0, 2)]),
    ("the weights", [("n", FOLD_WTS, 0, 2), ("n", FOLD_WTS, 4, 6)], [("W", "model.a.weight", 0, 2), ("W", "model.a.weight", 4, 6)]),
    ("a parameter of the body", [("p", 0, 0, 3)], [("p", 0, 0, 3)]),
    ("r1's prompt", [("n", R1_PROMPT, 0, 2)], [("XREQ", "prompt", R1_PROMPT, 0, 2)]),
    ("r1's seed", [("n", R1_SEED, 0, 1)], [("XREQ", "seed", R1_SEED, 0, 1)]),
    ("fields of r1's run parameter", [("n", R1_RUN, 0, 1), ("n", R1_RUN, 2, 3)], [("XREQ", "run", R1_RUN, 0, 1), ("XREQ", "run", R1_RUN, 2, 3)]),
    ("r1's witness", [("n", R1_WIT, 0, 2)], [("XREQ", "witness", R1_WIT, 0, 2)]),
    ("windows of r1's peers", [("n", R1_PEERS, 0, 2), ("n", R1_PEERS, 4, 6)], [("XREQ", "peers", R1_PEERS, 0, 2), ("XREQ", "peers", R1_PEERS, 4, 6)]),
    ("r1's embedding", [("n", R1_EMBED, 0, 2)], [("XREQ", "node", R1_EMBED, 0, 2)]),
]
XREQ = {"prompt-param": 1, "seed-param": 1, "run-param": 2, "witness-param": 1, "peers-param": 2, "node": 1}   # a run of r1's, each


@FORMS
def test_a_requests_projection_names_its_own_parameters_by_kind_and_counts_each_read_of_the_other_requests(compact):
    """In r0's projection of the fold its own prompt, seed, run parameter and witness, the registered values and the weights read
    as in its own Program.  A read of r1's parameter or node is a cross-request read, `("XREQ", kind, node, lo, hi)`, counted a
    run by its kind, in the compact form as in the record's."""
    fold = _view(FOLD_INPUTS, [(R0_READ, READER, [runs for _what, runs, _record in FOLD_READS]),
                               (R1_EMBED, EMBED, [[("n", R1_PROMPT, 0, 2)], [("n", FOLD_WTS, 0, 8)]]),
                               (13, SELECT, [[("n", R0_READ, 0, 8)]]), (14, SELECT, [[("n", R1_EMBED, 0, 8)]])],
                 compact=compact, **FOLD_LOADED)
    P = Projection(fold, "fold[r0]", [0, 2], tok_node=R0_PROMPT, weights_node=FOLD_WTS, seed_node=R0_SEED)
    P._all_seed_nodes = {R0_SEED, R1_SEED}
    records = [record for _what, _runs, record in FOLD_READS]
    assert P.canon(0) == ([_as_compact(record) for record in records] if compact else records)
    assert dict(P.xreq) == XREQ
    assert P.xreq_examples == [{"kind": "node" if t[1] == "node" else f"{t[1]}-param", "ref": list(t[2:])}
                               for record in records for t in record if t[0] == "XREQ"]
