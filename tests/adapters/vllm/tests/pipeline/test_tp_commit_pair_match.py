"""A pair of `tp-commit` (`verity_vllm.pipeline.tp.commit`) given a Match record (`--match-dir`): the record's generated ids reach
every rank by the workload's logical request, each rank's attribution of its committed sampled ids against them and its match
oracle (its committed TP partials against the capture's stored collectives) are in its verdict by name, and the Commit writes
them folded: the summary's `attribution`, `match_oracle` and `oracle_compare`, and `coverage.json`'s value correspondence where
no value check ran.  With it: a record that is not this workload's, a Match run that failed its own check, `--no-match-oracle`,
and the fold Match's peer binding, which a Commit with no fold Match leaves unbound and `--require-fold-match` fails on.
CPU-only, in the frame of `tests.pipeline.test_tp_commit_pair`: its ranks hold no Match record, so each test gives a rank's
finalize the attribution and the oracle a rank that compared answers with."""
from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.pipeline import test_tp_commit_options as TO
from tests.pipeline import test_tp_commit_pair as TP
from verity_vllm.acquire.sources.oracle_witness import SUMMARY_KEYS
from verity_vllm.errors import Refused

driver, pair = TO.driver, TP.pair

#: the capture arm's generated ids of the workload's two requests (`TO.WORKLOAD`: q0, q1)
GENERATED = {"q0": [5, 6, 7, 8], "q1": [9, 9]}
ATTRIBUTION: dict[str, Any] = {"ok": True, "steps_checked": 3, "rows_checked": 6, "mismatches_n": 0, "why": None, "requests": 2}
ORACLE: dict[str, Any] = {"verdict": "PASS", "steps_compared": 3, "tensors": 6, "compared": 6, "equal": 6, "mismatch_n": 0, "missing_n": 0,
                          "not_in_oracle_n": 0, "by_member": {"part": 6}, "witnesses": {"equal": 0, "unpaired_n": 0, "differs": []}}
NO_FOLD_MATCH = "fold Match record missing (tp2_fold_match.json / tp2_rank_match.json): the peer leaves have no Match-stage binding"


def _match(root: Path, generated: dict[str, list[int]] = GENERATED, verdict: bool | None = True) -> Path:
    """A Match dir: the capture arm's tokens, and the check's verdict unless ``verdict`` is None."""
    match = root / "match"
    (match / "capture").mkdir(parents=True)
    (match / "capture" / "tokens.json").write_text(json.dumps({"requests": [{"request_id": q, "generated_token_ids": ids} for q, ids in generated.items()]}))
    if verdict is not None:
        (match / "tp2_match.json").write_text(json.dumps({"pass": verdict}))
    return match


def _compared(attribution: dict[str, Any] | None = ATTRIBUTION, oracle: dict[str, Any] | None = ORACLE) -> Callable[[dict[str, Any]], None]:
    """A change of a rank's finalize answer: its run of record (a warm-up is compared with nothing) holds this attribution and oracle."""
    def change(answer: dict[str, Any]) -> None:
        if answer["binding"] is not None:
            answer["attribution"] = copy.deepcopy(attribution)
            answer["match_oracle"] = copy.deepcopy(oracle) | {"rank": answer["rank"]} if oracle is not None else None
    return change


def _agreed(pair: TP._Pair, **rank1: Any) -> None:
    """Rank 0 agrees with the Match record; rank 1 does too, but for ``rank1``'s records."""
    pair.alter("tp2_commit_finalize", 0, _compared())
    pair.alter("tp2_commit_finalize", 1, _compared(**rank1))


def test_a_pair_whose_ranks_agree_with_the_match_record_passes_and_leaves_the_comparison_folded(pair, driver, tmp_path, capsys):
    _agreed(pair)
    match = _match(tmp_path)
    assert pair.run("--match-dir", str(match)) == 0
    said = capsys.readouterr().out.splitlines()
    # every rank is made with the record's ids by logical request, and with the capture its oracle reads
    made = driver[0].made
    assert (made["match_tokens"], made["match_capture_dir"]) == ({"r0": [5, 6, 7, 8], "r1": [9, 9]}, str(match / "capture"))
    reference = {"path": str(match / "capture" / "tokens.json"), "requests": 2, "generated_total": 6, "tp2_match_verdict": True,
                 "capture_dir": str(match / "capture"), "stored_steps_per_rank": {"0": 0, "1": 0}}
    assert pair.read("versions.json")["match_record"] == reference
    c = pair.rows()[1]["commit"]
    for r in "01":
        per = c["per_rank"][r]
        assert (per["commit_pass"], per["attribution"], per["match_oracle"]) == (True, ATTRIBUTION, ORACLE | {"rank": int(r)})
    ranks = "; ".join(f"rank {r} PASS steps 3 tensors 6 equal 6 mismatch 0 missing 0 not_in_oracle 0 witnesses equal 0 unpaired 0 {{'part': 6}} " for r in (0, 1))
    assert f"[tp2commit] match oracle (tp partials vs capture), 0 differing MoE align witness(es): {ranks}" in said

    summary = pair.read("summary.json")
    assert summary["pass"] is True
    by_rank = {r: {"ok": True, "steps_checked": 3, "rows_checked": 6, "mismatches_n": 0, "why": None} for r in "01"}
    assert summary["attribution"] == {"match_record": reference, "per_rank": by_rank}
    assert summary["match_oracle"] == {r: {k: ORACLE.get(k) for k in SUMMARY_KEYS} for r in "01"}
    compared = summary["oracle_compare"]
    assert [compared[k] for k in ("result", "complete", "oracle_match_verdict", "oracle_is_failed_match_run", "why")] == [True, False, True, False, None]
    assert compared["attribution"] == {"complete": True, "result": True, "per_rank": {"0": True, "1": True}, "rows_checked_per_rank": {"0": 6, "1": 6}}
    assert compared["tp_partials"] == {"result": True, "per_rank": summary["match_oracle"]}
    assert compared["identities"] == {"families_covered": ["tp_rank_partials", "sampled_token_ids"],
                                      "families_not_covered": ["instance_outputs", "fa2_hidden_m1_stream", "input_token_ids", "weights"],
                                      "covered_n": None, "total_n": None}

    cov = pair.read("coverage.json")
    assert cov["attribution_per_rank"] == {r: by_rank[r] | {"swap_requests_negative": None} for r in "01"} and cov["attribution_reference"] == reference
    vc = cov["value_correspondence"]
    assert vc["method"] == ("Match-record compare only (no --value-check this attempt): sampled-ids attribution + TP partials vs the capture's stored "
                            "collectives") and vc["file"] is None
    assert vc["components"]["tap_reexec"] == {"result": None, "reference": "not run (no --value-check)"}
    assert vc["components"]["oracle_compare"] == compared
    linkage = vc["components"]["linkage"]
    assert [linkage[k] for k in ("result", "checked", "why")] == [True, 12, None]
    assert linkage["per_rank"] == {r: {"ok": True, "steps_checked": 3, "rows_checked": 6, "mismatches_n": 0, "requests": 2} for r in "01"}
    assert vc["components"]["root_compare"] == vc["components"]["replay"] == {"result": None, "reference": "not run"}
    # the comparison held on every rank; no rank pinned its weights (no manifest), so the correspondence is not a pass
    assert (vc["components"]["weights_pin"]["result"], vc["result"]) == (None, None)

    binding = pair.read("fold_match_binding.json")
    assert (binding["tp"], binding["result"], binding["n_bindings"], binding["why"]) == (2, None, 0, f"{NO_FOLD_MATCH}; Match is collective-level only (◐)")
    assert summary["fold_match_binding"] == {k: binding.get(k) for k in ("result", "n_bindings", "n_recorder", "n_bad_bindings", "why")}


@pytest.mark.parametrize("rank1,why,folded", [
    ({"attribution": ATTRIBUTION | {"ok": False, "mismatches_n": 1, "why": "step 2 row 1: 9 != 7"}},
     "request attribution vs Match record: step 2 row 1: 9 != 7", "rank 1: attribution step 2 row 1: 9 != 7"),
    ({"attribution": None}, "request attribution vs Match record: not run", "rank 1: attribution None"),
    ({"oracle": ORACLE | {"verdict": "FAIL", "equal": 5, "mismatch_n": 1, "why": "1 of 6 tensors differ"}},
     "match oracle (tp partials vs tp2_match capture): 1 of 6 tensors differ", "rank 1: match oracle FAIL 1 of 6 tensors differ mismatch 1 missing 0 not_in_oracle 0"),
    ({"oracle": {"verdict": "FAIL", "error": "FileNotFoundError('coll.rank1.step0.pt')"}},
     "match oracle (tp partials vs tp2_match capture): FileNotFoundError('coll.rank1.step0.pt')",
     "rank 1: match oracle FAIL FileNotFoundError('coll.rank1.step0.pt') mismatch None missing None not_in_oracle None"),
    ({"oracle": None}, "match oracle (tp partials vs tp2_match capture): not run", ""),
    ({"attribution": ATTRIBUTION | {"ok": False, "why": "a request the record does not hold"}, "oracle": ORACLE | {"verdict": "FAIL", "why": "nothing stored"}},
     "request attribution vs Match record: a request the record does not hold; also: match oracle (tp partials vs tp2_match capture): nothing stored",
     "rank 1: attribution a request the record does not hold; rank 1: match oracle FAIL nothing stored mismatch 0 missing 0 not_in_oracle 0"),
], ids=["a sampled id that is not the record's", "no attribution", "a partial that differs", "an oracle that could not read", "no oracle",
        "neither holds: each is named, the first fault first"])
def test_one_rank_that_does_not_agree_with_the_match_record_fails_alone_by_name(pair, tmp_path, rank1, why, folded):
    _agreed(pair, **rank1)
    assert pair.run("--match-dir", str(_match(tmp_path))) == 1
    c = pair.rows()[1]["commit"]
    assert (c["per_rank"]["0"]["commit_pass"], c["per_rank"]["0"]["commit_fail_reason"]) == (True, "PASS")
    assert (c["per_rank"]["1"]["commit_pass"], c["per_rank"]["1"]["commit_fail_reason"]) == (False, why)
    assert (c["commit_pass"], c["commit_fail_reason"]) == (False, f"rank 1: {why}")
    summary = pair.read("summary.json")
    # a rank with no oracle beside a rank that passed its own is not a rank whose oracle failed: the fold names no reason for it
    assert (summary["pass"], summary["oracle_compare"]["result"], summary["oracle_compare"]["why"]) == (False, False, folded)
    # the correspondence is the oracle's compare with the linkage, not the linkage alone: a rank's attribution may hold where its oracle does not
    vc = pair.read("coverage.json")["value_correspondence"]
    assert (vc["result"], vc["components"]["oracle_compare"]["result"]) == (False, False)
    assert vc["components"]["linkage"]["result"] is ("attribution" not in rank1)


def test_a_rank_that_failed_a_gate_before_the_match_records_keeps_that_reason_first(pair, tmp_path):
    def no_map(answer: dict[str, Any]) -> None:
        _compared(attribution=None)(answer)
        if answer["binding"] is not None:
            answer["binding"]["error"] = "no map"
    pair.alter("tp2_commit_finalize", 0, _compared())
    pair.alter("tp2_commit_finalize", 1, no_map)
    assert pair.run("--match-dir", str(_match(tmp_path))) == 1
    c = pair.rows()[1]["commit"]
    assert c["per_rank"]["1"]["commit_fail_reason"] == "binding map / coverage error: no map; also: request attribution vs Match record: not run"
    assert (c["per_rank"]["0"]["commit_pass"], c["commit_fail_reason"]) == (True, f"rank 1: {c['per_rank']['1']['commit_fail_reason']}")


def test_the_match_oracle_line_names_the_first_moe_align_witness_that_differs_over_the_ranks(pair, tmp_path, capsys):
    def witness(rank: int, step: int, k: int) -> dict[str, Any]:
        return {"rank": rank, "step": step, "k": k, "module": f"m.moe{k}", "implies": "another route"}
    pair.alter("tp2_commit_finalize", 0, _compared(oracle=ORACLE | {"witnesses": {"equal": 1, "unpaired_n": 0, "differs": [witness(0, 3, 0)]}}))
    pair.alter("tp2_commit_finalize", 1, _compared(oracle=ORACLE | {"witnesses": {"equal": 0, "unpaired_n": 2, "differs": [witness(1, 3, 1), witness(1, 2, 4)]}}))
    assert pair.run("--match-dir", str(_match(tmp_path))) == 0
    line = next(said for said in capsys.readouterr().out.splitlines() if said.startswith("[tp2commit] match oracle"))
    # a witness that differs is not the oracle's verdict: the line says which comes first, by step, then k, then rank
    assert line == ("[tp2commit] match oracle (tp partials vs capture), 3 differing MoE align witness(es), first rank 1 step 2 k 4 m.moe4 implies another route: "
                    "rank 0 PASS steps 3 tensors 6 equal 6 mismatch 0 missing 0 not_in_oracle 0 witnesses equal 1 unpaired 0 {'part': 6} ; "
                    "rank 1 PASS steps 3 tensors 6 equal 6 mismatch 0 missing 0 not_in_oracle 0 witnesses equal 0 unpaired 2 {'part': 6} ")


def test_with_no_match_oracle_a_rank_is_made_with_no_capture_and_its_attribution_alone_decides(pair, driver, tmp_path, capsys):
    for r in (0, 1):
        pair.alter("tp2_commit_finalize", r, _compared(oracle=None))
    assert pair.run("--match-dir", str(_match(tmp_path)), "--no-match-oracle") == 0
    made = driver[0].made
    assert (made["match_tokens"], made["match_capture_dir"]) == ({"r0": [5, 6, 7, 8], "r1": [9, 9]}, None)
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["match_oracle"]) == (True, None)
    assert summary["oracle_compare"]["result"] is True and summary["oracle_compare"]["identities"]["families_covered"] == ["sampled_token_ids"]
    assert summary["oracle_compare"]["tp_partials"] == {"result": None, "reference": "not run (--no-match-oracle or no stored capture steps)"}
    assert not any(line.startswith("[tp2commit] match oracle") for line in capsys.readouterr().out.splitlines())


@pytest.mark.parametrize("generated,refusal", [
    (GENERATED | {"q9": [1]}, "fail-closed: Match record request q9 is not a request of this workload"),
    ({"q0": [5, 6, 7, 8]}, r"fail-closed: Match record covers \['r0'\] but the workload declares \['r0', 'r1'\]"),
], ids=["a request the workload does not hold", "a request of the workload the record lacks"])
def test_a_match_record_that_is_not_the_workloads_is_refused_before_any_rank_is_made(pair, driver, tmp_path, generated, refusal):
    with pytest.raises(Refused, match=refusal):
        pair.run("--match-dir", str(_match(tmp_path, generated)))
    assert driver[0].made == {} and pair.sent == []


@pytest.mark.parametrize("verdict", [False, None], ids=["a Match run that failed its check", "a Match run with no verdict"])
def test_a_commit_that_equals_a_match_run_that_did_not_pass_is_compared_and_named(pair, tmp_path, verdict):
    """The capture of a Match run that failed its own check still compares; that the Commit equals it is said, and is no pass."""
    _agreed(pair)
    assert pair.run("--match-dir", str(_match(tmp_path, verdict=verdict))) == 0
    compared = pair.read("summary.json")["oracle_compare"]
    assert (compared["oracle_match_verdict"], compared["oracle_is_failed_match_run"]) == (verdict, verdict is False)
    assert compared["result"] is (None if verdict is False else True)
    assert compared.get("oracle_note") == ("oracle is a FAILED Match run (tp2_match.json pass False): compared, not a PASS ingredient" if verdict is False else None)
    assert compared["attribution"]["result"] is True and compared["tp_partials"]["result"] is True and compared["why"] is None


@pytest.mark.parametrize("with_match", [False, True], ids=["no Match dir", "a Match dir with no fold Match"])
def test_a_required_fold_match_that_is_not_there_fails_the_commit_by_name(pair, tmp_path, capsys, with_match):
    if with_match:
        _agreed(pair)
    assert pair.run("--require-fold-match", *(("--match-dir", str(_match(tmp_path))) if with_match else ())) == 1
    why = f"{NO_FOLD_MATCH} -- REQUIRED" if with_match else "no --match-dir: no fold Match record to bind the peer inputs to -- REQUIRED"
    binding = pair.read("fold_match_binding.json")
    assert (binding["tp"], binding["result"], binding["why"]) == (2, False, why)
    # every rank passed its own checks: the Commit fails on the binding alone
    c = pair.rows()[1]["commit"]
    assert (c["commit_pass"], c["commit_fail_reason"]) == (True, "PASS")
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["commit_pass"], summary["stopped"]) == (False, [True], None)
    assert (summary["fold_match_binding"]["result"], summary["fold_match_binding"]["why"]) == (False, why)
    vc = pair.read("coverage.json")["value_correspondence"]
    assert (vc["result"], vc["components"]["fold_match_binding"]["result"], vc["components"]["fold_match_binding"]["artifact"]) == (False, False, "fold_match_binding.json")
    said = capsys.readouterr().out.splitlines()
    assert any(line.startswith("[tp2commit] fold-Match peer binding (D90, R1 at Commit): result False ") and line.endswith(why) for line in said)
    assert said[-1].startswith("[tp2commit] FAIL delta ")
