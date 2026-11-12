"""`program-compare`'s `main` run in process: the document it writes at `--out`, the brief it prints, and the padding audit of an
unfolded side. The Programs are the two-rank toy's (`test_tp2_d90_fixture`): rank 0's fold and its rank Program."""
from __future__ import annotations

import json

import pytest

from tests.check import test_tp2_d90_fixture as T
from verity_vllm.check.match import compiler_form as CF
from verity_vllm.check.match import program_compare as PC
from verity_vllm.check.match.instance_sequence import RootView
from verity_vllm.config import UsageError
from verity_vllm.records.schema_ids import PROGRAM_COMPARE_SCHEMA

FOLD_KEYS = ["fold_form", "oracle_segmentation_mismatches", "oracle_weight_map"]   # what `main` adds when B is a fold's record


def _dirs(tmp_path, **prog_kw) -> tuple[str, str]:
    """(the rank Program's directory, the fold's directory), rank 0's."""
    fd, dd = tmp_path / "fold", tmp_path / "build"
    fd.mkdir()
    dd.mkdir()
    T.persist_fold(T.fold_rank(0), fd, 0)
    T.write_derived(dd, T.rank_program(0, **prog_kw), 0)
    return str(dd), str(fd)


def _run(tmp_path, capsys, **kw) -> tuple[dict, dict, str]:
    """(the document, the brief on stdout, the document's text) of one run of `main`, which returns 0."""
    out = tmp_path / "match_compare.json"
    assert PC.main(PC.Options(out=str(out), **kw)) == 0
    text = out.read_text()
    return json.loads(text), json.loads(capsys.readouterr().out), text


def test_main_writes_the_comparison_of_a_rank_program_with_its_fold(tmp_path, capsys):
    a, b = _dirs(tmp_path)
    doc, brief, text = _run(tmp_path, capsys, a=a, b=b)
    # the whole document, key for key in this order, from the two Programs loaded here and the module's own parts
    A, B = PC.load_derived(a, "derived"), PC.load_oracle(b, "oracle")
    want = {"schema": "cb-a/compare/v1", "utc": doc["utc"], "a_dir": a, "b_dir": b, "unfold_a": False,
            "fold_form": CF.fold_in_form(B, [a]),
            "digests": {"a": A.digest, "b": B.digest, "equal": False},
            "histograms": {"a": PC.spec_histograms(A), "b": PC.spec_histograms(B)},
            "histogram_diff_T_normalised": {},
            "templates": {"a": PC.per_step_templates(A), "b": PC.per_step_templates(B)},
            "oracle_segmentation_mismatches": 0,
            "oracle_weight_map": {"n_ranges": 4, "first": B.oracle_wmap, "overlaps": [], "n_overlaps": 0},
            "compare": PC.compare_steps(A, B),
            "seconds": doc["seconds"]}
    assert text == json.dumps(want, indent=1, default=str)                                 # one space a level, and no other layout
    assert doc["schema"] == PROGRAM_COMPARE_SCHEMA and len(B.oracle_wmap) == 4
    cmp = doc["compare"]
    assert (cmp["a"]["label"], cmp["b"]["label"]) == ("derived", "oracle")
    # what a row's verdict reads: equal canonical sequences, as many steps on both sides, no operand that differs
    assert cmp["canonical_sha256"]["equal"] and cmp["a"]["steps"] == cmp["b"]["steps"] == 2 and cmp["totals"]["operands_different"] == 0
    assert brief == {"digests_equal": False, "canonical_equal": True, "steps_a": 2, "steps_b": 2,
                     "instances": [cmp["a"]["instances"], cmp["b"]["instances"]], "totals": cmp["totals"],
                     "histogram_diff_T_normalised": {}, "seconds": doc["seconds"]}


def test_main_reports_a_rank_program_that_reads_another_collectives_peer(tmp_path, capsys):
    a, b = _dirs(tmp_path, peer_of={1: 0})
    doc, brief, _ = _run(tmp_path, capsys, a=a, b=b, label_a="rank", label_b="fold")
    cmp = doc["compare"]
    assert (cmp["a"]["label"], cmp["b"]["label"]) == ("rank", "fold")
    assert not cmp["canonical_sha256"]["equal"] and cmp["totals"]["operands_different"] == 2
    wiring = cmp["steps"][0]["wiring"]
    assert wiring["different_by_spec_arg"] == {"AllReduce2_v1{N=16}#arg0": 2} and [e["arg"] for e in wiring["examples"]] == [0, 0]
    assert brief["canonical_equal"] is False and brief["totals"] == cmp["totals"]


def test_two_rank_program_directories_are_compared_without_the_folds_keys(tmp_path, capsys):
    a, _b = _dirs(tmp_path)
    doc, _brief, _ = _run(tmp_path, capsys, a=a, b=a)
    assert not set(FOLD_KEYS) & set(doc)
    assert doc["digests"]["equal"] and doc["compare"]["canonical_sha256"]["equal"]


def test_unfolding_a_adds_its_padding_audit(tmp_path, capsys):
    a, b = _dirs(tmp_path)
    doc, _brief, _ = _run(tmp_path, capsys, a=a, b=b, unfold_a=True)
    assert doc["unfold_a"] is True and list(doc)[-2:] == ["padding_audit_a", "seconds"]
    assert doc["padding_audit_a"] == {"per_step": [{"step": 0, "muxes": 0}, {"step": 1, "muxes": 0}]}


def test_progressions_refuse_an_unfolded_side_before_a_directory_is_read(tmp_path):
    out = tmp_path / "match_compare.json"
    with pytest.raises(UsageError, match="compares without unfolding only"):
        PC.main(PC.Options(a=str(tmp_path / "absent"), b=str(tmp_path / "absent"), out=str(out), operands="progressions", unfold_a=True))
    assert not out.exists()


def test_the_padding_audit_counts_a_steps_muxes_their_live_bit_and_zero_row_and_its_guards():
    """A view of what `audit_padding` reads and nothing more: the second step holds one event's guard and its three muxes."""
    live, zero = [("n", 3, 0, 1)], ("n", 0, 0, 16)
    P = RootView.__new__(RootView)
    P.compact = False
    P.inst = [{"spec": "TokenSelect_v1{V=8}", "args": [[("n", 1, 0, 8)]]},
              {"spec": "I32Eq_v1", "args": [[("n", 2, 0, 1)], [("n", 2, 1, 2)]]},
              {"spec": "BitNot_v1", "args": [[("n", 5, 0, 1)]]},
              {"spec": "BitAnd_v1", "args": [[("n", 6, 0, 1)], [("n", 2, 2, 3)]]},
              {"spec": "SelectI32_v1", "args": [live, [("n", 2, 3, 4)], [("n", 2, 4, 5)]]},
              {"spec": "SelectBf16_v1{N=16}", "args": [live, [("n", 8, 0, 16)], [zero, ("n", 9, 0, 16)]]},
              {"spec": "SelectBf16_v1{N=16}", "args": [live, [("n", 8, 16, 32)], [zero, ("n", 9, 16, 32)]]},
              {"spec": "TokenSelect_v1{V=8}", "args": [[("n", 10, 0, 8)]]}]
    P.steps = [[0], [1, 2, 3, 4, 5, 6, 7]]
    assert PC.audit_padding(P) == {"per_step": [
        {"step": 0, "muxes": 0},
        {"step": 1, "muxes": 3, "mux_kinds": {"SelectI32_v1": 1, "SelectBf16_v1{N=16}": 2}, "distinct_live_refs": 1, "distinct_zero_refs": 1,
         "guards_and_consts": {"I32Eq_v1": 1, "BitNot_v1": 1, "BitAnd_v1": 1}}]}
