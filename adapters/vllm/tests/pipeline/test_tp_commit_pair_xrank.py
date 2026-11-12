"""A pair of `tp-commit` (`verity_vllm.pipeline.tp.commit`) with its cross-rank collectives pass on (`--xrank-per-stratum N`): the
driver asks every rank for its picks under the pair's TP run root, holds each received output to the definition over the ranks'
committed partials (`check.match.xrank_collectives`), fails the ranks the pass names, or every rank where it names none, and
writes the pass (`xrank_pair<N>.json`) and its fold (`xrank_collectives.json`).  A family a rank's replay has no evaluator for is
waived while the pass runs.  With it, the negatives a Commit is asked to inject (`--flip-partial`, `--swap-requests`,
`--omit-partial-site`): one its target rank did not inject makes the run an error (exit 3), never a verdict.
CPU-only, in the frame of `tests.pipeline.test_tp_commit_pair`: a rank's picks are read from a GPU's committed bytes, so each
test gives the driver the dump a rank answers with."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from tests.pipeline import test_tp_commit_options as TO
from tests.pipeline import test_tp_commit_pair as TP
from tests.pipeline import test_tp_commit_pair_replay as PR
from verity_vllm.check.replay import replay_codes as RC

driver, pair = TO.driver, TP.pair

FAMILY, SITE = "AllReduce2_v1", "m.proj"
#: each rank's committed partial of one all-reduce, and what every rank received of it: their sum, rounded once
PARTS = {r: torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.25, 4.0]][r], dtype=torch.bfloat16).reshape(1, 3) for r in (0, 1)}
RECEIVED = (PARTS[0].float() + PARTS[1].float()).to(torch.bfloat16)
ON = ("--xrank-per-stratum", "2")
PASSED = {"result": True, "picks": 1, "equal": 1, "mismatch_n": 0, "missing_n": 0, "why": None}


def _dump(pair: TP._Pair, rank: int, *, received: Any = RECEIVED, part: bool = True, flipped: int = 0, sites: tuple[str, ...] = (SITE,),
          written: bool = True) -> Callable[[dict[str, Any]], None]:
    """A change of a rank's answer to `tp2_commit_xrank_dump` into a rank's (`xrank_dump.dump`): its one pick, in the file the answer
    names unless not ``written``; ``sites`` are the sites of the family its Program names; ``flipped``, the collective inputs it flipped."""
    def change(answer: dict[str, Any]) -> None:
        file = pair.out / f"xrank_pair0_rank{rank}.pt"
        pick = {"step": 1, "site": SITE, "kind": "all_reduce", "family": FAMILY, "stratum": f"{SITE}|all_reduce|all", "rows": [0],
                "part": PARTS[rank] if part else None, "recv": received, "why": None if part else "no such member in the run"}
        if written:
            torch.save({"rank": rank, "picks": [pick]}, file)
        answer.clear()
        answer.update(rank=rank, file=str(file), picks=1, unreplayable_sites={FAMILY: list(sites)}, flipped_collective_inputs=flipped)
    return change


def _dumped(pair: TP._Pair, **rank1: Any) -> None:
    """Rank 0 dumps its pick as it committed and received it; rank 1 does too, but for ``rank1``."""
    pair.alter("tp2_commit_xrank_dump", 0, _dump(pair, 0))
    pair.alter("tp2_commit_xrank_dump", 1, _dump(pair, 1, **rank1))


def _asked(pair: TP._Pair) -> list[dict[str, Any]]:
    return [kw for method, kw in pair.asked if method == "tp2_commit_xrank_dump"]


def test_a_pair_whose_ranks_received_the_sum_of_their_committed_partials_passes_the_cross_rank_pass(pair, capsys):
    _dumped(pair)
    assert pair.run(*ON) == 0
    c = pair.rows()[1]["commit"]
    # one call, after every rank's root is fixed: the picks are seeded by the pair's TP run root
    assert _asked(pair) == [{"out_dir": str(pair.out), "seed_hex": c["tp_run_root"], "pair": 0, "per_stratum": 2}]
    assert [c["per_rank"][r]["cross_rank_collectives"] for r in "01"] == [PASSED, PASSED] and c["commit_pass"] is True
    one = pair.read("xrank_pair0.json")
    assert (one["schema"], one["tp"], one["pair"], one["commit_started"]) == ("verity/tp2/xrank-collectives/v1", 2, 0, pair.rows()[1]["commit_started"])
    assert [one[k] for k in ("result", "world", "picks", "equal", "mismatch_n", "missing_n", "mismatches", "missing")] == [True, 2, 1, 1, 0, 0, [], []]
    assert one["by_family"] == {FAMILY: {"picks": 1, "equal": 1, "mismatch": 0, "missing": 0}}
    assert one["by_stratum"] == {f"{SITE}|all_reduce|all": {"picks": 1, "equal": 1, "families": [FAMILY]}}
    assert one["dumps"] == [{"rank": r, "file": str(pair.out / f"xrank_pair0_rank{r}.pt"), "picks": 1, "unreplayable_sites": {FAMILY: [SITE]},
                             "flipped_collective_inputs": 0} for r in (0, 1)]
    assert one["coverage_of_unreplayable"]["complete"] is True
    assert one["coverage_of_unreplayable"]["families"] == {FAMILY: {"sites": 1, "covered": 1, "missing": [], "picks": 1, "equal": 1}}
    # the verdict is the evidence: the ranks' picks are not kept
    assert not list(pair.out.glob("xrank_pair0_rank*.pt"))
    fold = pair.read("xrank_collectives.json")
    assert (fold["schema"], fold["tp"]) == ("verity/tp2/xrank-collectives-fold/v1", 2)
    assert [fold[k] for k in ("result", "pairs", "picks", "equal", "mismatch_n", "missing_n", "families", "why")] == [True, 1, 1, 1, 0, 0, [FAMILY], None]
    assert fold["per_pair_artifacts"] == [str(pair.out / "xrank_pair0.json")]
    assert fold["coverage_of_unreplayable"] == {"complete": True, "per_pair": [one["coverage_of_unreplayable"]]}
    summary = pair.read("summary.json")
    assert summary["pass"] is True and summary["cross_rank_collectives"] == {k: v for k, v in fold.items() if k not in ("schema", "tp")}
    assert pair.read("coverage.json")["value_correspondence"]["components"]["cross_rank_collectives"] == summary["cross_rank_collectives"]
    assert ("[tp2commit] cross-rank collectives pair 0: result True picks 1 equal 1 mismatch 0 missing 0 families ['AllReduce2_v1'] flipped [0, 0]; "
            in capsys.readouterr().out.splitlines())


def test_a_rank_that_received_another_value_than_the_definitions_fails_alone_by_the_site(pair):
    other = RECEIVED.clone()
    other[0, 2] += 1
    _dumped(pair, received=other)
    assert pair.run(*ON) == 1
    c = pair.rows()[1]["commit"]
    found = "1 pick(s) where a committed received output != the definition over the committed partials, 0 not evaluable -- first: m.proj step 1 all_reduce "
    why = f"cross-rank collective outputs (TP-12): {found}"
    assert (c["per_rank"]["0"]["commit_pass"], c["per_rank"]["0"]["commit_fail_reason"]) == (True, "PASS")
    assert (c["per_rank"]["1"]["commit_pass"], c["per_rank"]["1"]["commit_fail_reason"]) == (False, why)
    assert (c["commit_pass"], c["commit_fail_reason"]) == (False, f"rank 1: {why}")
    failed = {"result": False, "picks": 1, "equal": 0, "mismatch_n": 1, "missing_n": 0, "why": found}
    assert [c["per_rank"][r]["cross_rank_collectives"] for r in "01"] == [failed, failed]
    one = pair.read("xrank_pair0.json")
    assert one["mismatches"] == [{"pick": 0, "site": SITE, "step": 1, "kind": "all_reduce", "family": FAMILY, "rows": [0],
                                  "received_equals_definition_per_rank": {"0": True, "1": False}, "received_equal_across_ranks": False,
                                  "n_diff_per_rank": {"0": 0, "1": 1}, "elements": 3}]
    summary, cov = pair.read("summary.json"), pair.read("coverage.json")
    assert (summary["pass"], summary["cross_rank_collectives"]["result"], summary["cross_rank_collectives"]["why"]) == (False, False, found)
    assert (cov["verdict"], cov["reason"], cov["value_correspondence"]["result"]) == ("FAIL", f"rank 1: {why}", False)


NOT_COMMITTED = ("0 pick(s) where a committed received output != the definition over the committed partials, 1 not evaluable -- first: m.proj step 1 "
                 "all_reduce rank 1: partial not committed (no such member in the run)")
NO_STRATUM = "coverage of the unreplayable collective sites incomplete -- sites without a stratum: AllReduce2_v1: ['m.other']"
NO_DUMP = "cross-rank collective pass did not run: FileNotFoundError("


@pytest.mark.parametrize("rank1,found,kept", [({"part": False}, NOT_COMMITTED, True), ({"sites": (SITE, "m.other")}, NO_STRATUM, True),
                                              ({"written": False}, NO_DUMP, False)],
                         ids=["a pick a rank did not commit", "a site of the family with no pick", "a dump that is not there"])
def test_a_pass_that_names_no_rank_fails_every_rank(pair, capsys, rank1, found, kept):
    _dumped(pair, **rank1)
    assert pair.run(*ON) == 1
    c = pair.rows()[1]["commit"]
    for r in "01":
        per = c["per_rank"][r]
        assert per["commit_pass"] is False and per["commit_fail_reason"].startswith(f"cross-rank collective outputs (TP-12): {found}")
        assert per["cross_rank_collectives"]["result"] is False and per["cross_rank_collectives"]["why"].startswith(found)
    assert c["commit_fail_reason"] == "; ".join(f"rank {r}: {c['per_rank'][r]['commit_fail_reason']}" for r in "01")
    # a pass that did not run leaves no record of its own; the fold says it failed, and why
    assert (pair.out / "xrank_pair0.json").exists() is kept
    fold = pair.read("xrank_collectives.json")
    assert (fold["result"], fold["pairs"]) == (False, 1) and fold["why"].startswith(found)
    assert fold["per_pair_artifacts"] == ([str(pair.out / "xrank_pair0.json")] if kept else [])
    assert pair.read("summary.json")["pass"] is False
    errors = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[tp2commit] cross-rank collectives pair 0: ERROR FileNotFoundError(")]
    assert len(errors) == (0 if kept else 1)


def test_a_commit_that_binds_no_partials_runs_no_cross_rank_pass(pair):
    assert pair.run(*ON, "--no-tp-partials") == 0
    assert _asked(pair) == []
    off = {"result": None, "picks": None, "equal": None, "mismatch_n": None, "missing_n": None, "why": "not run (--xrank-per-stratum 0)"}
    assert [pair.rows()[1]["commit"]["per_rank"][r]["cross_rank_collectives"] for r in "01"] == [off, off]
    assert pair.read("versions.json")["negatives"]["no_tp_partials"] is True and not (pair.out / "xrank_pair0.json").exists()


@pytest.mark.parametrize("flags,code,waived", [((), 1, []), (ON, 0, [FAMILY])], ids=["the pass off", "the pass on"])
def test_a_family_a_ranks_replay_has_no_evaluator_for_fails_the_rank_unless_the_cross_rank_pass_evaluates_it(pair, tmp_path, flags, code, waived):
    reason = f"no registered evaluator for {FAMILY}"
    population = PR.REPLAY["population"] | {"not_evaluable": {reason: 3}, "not_evaluable_codes": {reason: RC.reason_code("no-evaluator", FAMILY)}}
    PR._replayed(pair, replay=PR.REPLAY | {"population": population})
    _dumped(pair)
    assert pair.run(*PR._of_record(tmp_path), *flags) == code
    c = pair.rows()[1]["commit"]
    gap = c["per_rank"]["1"]["no_evaluator_gap"]
    assert (gap["cross_rank_pass"], gap["waived_cross_rank"], gap["fail"] is None) == (bool(flags), waived, bool(flags))
    assert c["per_rank"]["0"]["commit_pass"] is True and c["per_rank"]["1"]["commit_pass"] is bool(flags)
    if not flags:
        assert c["per_rank"]["1"]["commit_fail_reason"].startswith("no-evaluator families (rank 1): ") and FAMILY in c["per_rank"]["1"]["commit_fail_reason"]
    else:
        families = pair.read("sampled_replay.json")["unreplayable_collectives_evaluated_by"]
        assert (families["result"], families["by_family"]) == (True, {FAMILY: {"picks": 1, "equal": 1, "mismatch": 0, "missing": 0}})


# a negative the Commit was asked to inject

def _made(**fields: Any) -> Callable[[dict[str, Any]], None]:
    return lambda answer: answer.update(fields)


@pytest.mark.parametrize("flags,made,applied", [
    (("--flip-partial", f"1:{SITE}", *ON), {}, {"flip_partial": {"rank": "1", "flipped_collective_inputs_per_pair": [0], "applied": False}}),
    (("--flip-partial", f"1:{SITE}"), {}, {"flip_partial": {"rank": "1", "flipped_collective_inputs_per_pair": [], "applied": None}}),
    (("--swap-requests", "1:q0:q1"), {}, {"swap_requests": {"rank": "1", "swap_requests_applied": [None], "applied": False}}),
    (("--omit-partial-site", f"1:{SITE}"), {"tp_partial_source": {"sites": 1}}, {"omit_partial_site": {"rank": "1", "omit_site_per_rank": [None], "applied": False}}),
], ids=["a flip the rank's dump does not count", "a flip with no cross-rank pass to count it", "a swap the rank did not make", "a site the rank did not omit"])
def test_a_negative_its_rank_did_not_inject_is_an_error_run_and_never_a_verdict(pair, capsys, flags, made, applied):
    _dumped(pair)
    pair.alter("tp2_commit_make", 1, _made(**made))
    assert pair.run(*flags) == 3
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["negative_not_applied"], summary["negative_applied"]) == (None, sorted(applied), applied)
    why = f"NEGATIVE NOT APPLIED: {sorted(applied)} -- {json.dumps(applied)}; the run is an error, not a verdict"
    control, run = pair.rows()
    # every rank passed, and the run's row says neither that nor a failure
    assert (run["commit"]["commit_pass"], run["commit"]["commit_fail_reason"]) == (None, why)
    assert [run["commit"]["per_rank"][r]["commit_pass"] for r in "01"] == [True, True] and (control["arm"], "commit" in control) == ("control", False)
    assert summary["commit_pass"] == [True]
    assert capsys.readouterr().out.splitlines()[-1] == f"[tp2commit] {why}"


@pytest.mark.parametrize("flags,made,dump,applied,code", [
    (("--flip-partial", f"1:{SITE}", *ON), {}, {"flipped": 2, "received": RECEIVED + 1},
     {"flip_partial": {"rank": "1", "flipped_collective_inputs_per_pair": [2], "applied": True}}, 1),
    (("--swap-requests", "1:q0:q1:2-5"), {"swap_requests_applied": True}, {}, {"swap_requests": {"rank": "1", "swap_requests_applied": [True], "applied": True}}, 0),
    (("--omit-partial-site", f"1:{SITE}"), {"tp_partial_source": {"omit_site": SITE}}, {},
     {"omit_partial_site": {"rank": "1", "omit_site_per_rank": [SITE], "applied": True}}, 0),
], ids=["a flip the cross-rank pass catches", "a swap the rank made", "a site the rank omitted"])
def test_a_negative_its_rank_injected_leaves_the_commit_its_verdict(pair, flags, made, dump, applied, code):
    """The verdict is the Commit's own: these ranks' stand-in records catch the flip alone."""
    _dumped(pair, **dump)
    pair.alter("tp2_commit_make", 1, _made(**made))
    assert pair.run(*flags) == code
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["negative_applied"], "negative_not_applied" in summary) == (code == 0, applied, False)
    assert pair.rows()[1]["commit"]["commit_pass"] is (code == 0)
