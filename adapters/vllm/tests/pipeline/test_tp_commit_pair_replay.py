"""A pair of `tp-commit` (`verity_vllm.pipeline.tp.commit`) under a required manifest, where a rank's finalize holds more than its
root and its binding: its sampled replay with its boundary linkage, and its weights pin.  The driver takes each into the rank's
verdict by name, says each rank's replay and pin as it reads them, and writes the ranks' records folded: the summary's
`sampled_replay`, `boundary_linkage`, `query_population` and `weights_pin`, `sampled_replay.json`, `weights_pin.json` and the
value correspondence of `coverage.json`.  With it: ranks that replayed and pinned nothing, a Commit with no sampled replay, a
rank that skipped its binding under a manifest, and a deferred replay (`--replay-deferred`), whose ranks are pending and whose
bundles the Commit seals unless a rank is decided already.
CPU-only, in the frame of `tests.pipeline.test_tp_commit_pair`: its ranks commit a host committer's stream and replay nothing, so
each test gives a rank's finalize the records a rank that replayed and pinned answers with."""
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
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline.tp import c2 as C2
from verity_vllm.program.frontend.collective_kinds import cross_rank_families

driver, pair = TO.driver, TP.pair

#: a rank's replay that passed, its population reconciled against the query; its linkage; its pin
REPLAY: dict[str, Any] = {"result": True, "grade": "complete", "population": {"vus": 6, "strata_n": 2, "query_checks": {"ok": True}},
                          "sample": {"picked": 4, "evaluated": 4, "equal": 4, "mismatch_n": 0, "not_evaluated_n": 0},
                          "weights": {"equal": True}, "timing": {"wall_s": 1.5}, "strata": {"a": {}, "b": {}}, "trace": ["t"]}
LINKAGE: dict[str, Any] = {"result": True, "pairs": 3, "linked": 3}
PIN: dict[str, Any] = {"result": True, "root_of_record": "ab" * 32, "compared": 2, "mismatched": [], "missing": [], "unpinned": [],
                       "all_shards_pinned": True}
PROGRAMS = {str(r): TP._program(r).hex() for r in (0, 1)}
NOT_RECONCILED = "NOT reconciled against Q(P) with a required manifest supplied: no `population.query_checks` on the rank's sampled replay"


def _of_record(root: Path) -> tuple[str, ...]:
    """The flags of a Commit under a manifest that names each rank's Program, with each rank's Build."""
    return ("--required-manifest", TO._manifest(root, PROGRAMS), "--build-rank0", TO._build(root, 0, PROGRAMS["0"]),
            "--build-rank1", TO._build(root, 1, PROGRAMS["1"]))


def _records(replay: dict[str, Any] | None = REPLAY, linkage: dict[str, Any] | None = LINKAGE,
             pin: dict[str, Any] | None = PIN) -> Callable[[dict[str, Any]], None]:
    """A change of a rank's finalize answer: its run of record (a warm-up replays and pins nothing) holds these records."""
    def change(answer: dict[str, Any]) -> None:
        if answer["binding"] is not None:
            answer["sampled_replay"] = copy.deepcopy({"sampled_replay": replay, "boundary_linkage": linkage}) if replay is not None else None
            answer["weights_pin"] = copy.deepcopy(pin)
    return change


def _replayed(pair: TP._Pair, **rank1: Any) -> None:
    """Rank 0 replayed, linked and pinned; rank 1 did too, but for ``rank1``'s records."""
    pair.alter("tp2_commit_finalize", 0, _records())
    pair.alter("tp2_commit_finalize", 1, _records(**rank1))


def test_a_pair_whose_ranks_replayed_linked_and_pinned_passes_and_leaves_their_records_folded(pair, tmp_path, capsys):
    _replayed(pair)
    flags = _of_record(tmp_path)
    assert pair.run(*flags) == 0
    said = capsys.readouterr().out.splitlines()
    c = pair.rows()[1]["commit"]
    assert (c["commit_pass"], c["commit_fail_reason"]) == (True, "PASS")
    for r in "01":
        per = c["per_rank"][r]
        assert (per["commit_pass"], per["commit_fail_reason"]) == (True, "PASS")
        # the rank's replay without its strata and its trace, which stay in its finalize
        assert per["sampled_replay"] == {k: v for k, v in REPLAY.items() if k not in ("strata", "trace")} | {"strata_n": 2}
        assert per["boundary_linkage"] == LINKAGE and per["weights_pin"] == PIN | {"mismatched_n": 0}
        assert [per["query_population"][k] for k in ("rank", "present", "ok", "fail")] == [int(r), True, True, None]
        assert [per["no_evaluator_gap"][k] for k in ("rank", "cross_rank_pass", "waived_cross_rank", "fail")] == [int(r), False, [], None]
        assert per["cublas_route_pin"] is None and "replay_deferred" not in per
        assert (f"[tp2commit] sampled replay rank {r}: result True grade complete population 6 strata 2 picked 4 evaluated 4 equal 4 mismatch 0 "
                "not_evaluated 0 weights self-consistent True (not a pin) wall 1.5s; boundary linkage True 3/3; ") in said
        assert (f"[tp2commit] weights pin rank {r}: result True root_of_record abababababababab compared 2 mismatched 0 missing 0 unpinned 0 "
                "shards_pinned True; ") in said
    assert "[tp2commit] C2 population vs Q(P): result True per rank {'0': (True, {}), '1': (True, {})} " in said
    assert [f["sampled_replay"]["sampled_replay"]["strata"] for f in pair.read("finalize_pair0.json")] == [REPLAY["strata"]] * 2

    record = pair.read("sampled_replay.json")
    assert (record["schema"], record["tp"]) == ("verity-sweep/tp-sampled-replay/v1", 2)
    assert record["per_pair_artifacts"] == [str(pair.out / "finalize_pair0.json")]
    assert record["unreplayable_collectives_evaluated_by"] == {"pass": "cross_rank_collectives (TP-12)", "families": sorted(cross_rank_families(2)),
                                                               "result": None, "by_family": {}, "artifact": "xrank_collectives.json"}
    replay, linkage = record["sampled_replay"], record["boundary_linkage"]
    assert (replay["result"], replay["why"], replay["n"]) == (True, None, 8)
    assert replay["sample"] == {"picked": 8, "evaluated": 8, "equal": 8, "mismatch_n": 0, "not_evaluated_n": 0, "per_stratum": None}
    assert replay["population"] == {"n": 12, "per_rank": {"0": 6, "1": 6}, "source": None, "by_family": {}}
    assert (replay["strata"]["n"], replay["strata"]["ranks"]) == (4, ["rank 0", "rank 1"])
    assert replay["weights"] == replay["weights_self_consistent"] == {"0": True, "1": True} and replay["weights_pinned"] == {"0": None, "1": None}
    assert replay["per_rank"] == {r: {"result": True, "grade": "complete", "why": None, "rank": None, "rank_note": None} for r in "01"}
    assert replay["per_run"] == [{r: {"sampled_replay": True, "grade": "complete", "boundary_linkage": True} for r in "01"}]
    assert (replay["unreplayable"], replay["mismatch"]) == ([], [])
    assert [linkage[k] for k in ("result", "checked", "linked", "why")] == [True, 6, 6, None]
    assert linkage["per_rank"] == {r: {"result": True, "pairs": 3, "linked": 3, "why": None} for r in "01"}

    summary = pair.read("summary.json")
    assert (summary["pass"], summary["commit_pass"]) == (True, [True]) and "replay_deferred" not in summary
    assert summary["sampled_replay"] == {k: replay.get(k) for k in C2.SR_SUMMARY_KEYS} and summary["boundary_linkage"] == linkage
    population = summary["query_population"]
    assert (population["result"], population["why"]) == (True, None)
    assert population["per_rank"] == {r: {"present": True, "ok": True, "counts": {}, "fail": None} for r in "01"}
    assert [summary["weights_pin"][k] for k in ("result", "why", "checkpoint_dir")] == [True, None, None]
    assert summary["weights_pin"]["per_rank"] == {r: {"result": True, "root_of_record": "ab" * 32, "compared": 2, "mismatched_n": 0, "missing": [],
                                                      "unpinned": [], "all_shards_pinned": True, "committed_weights_root": None, "why": None} for r in "01"}
    assert summary["required_manifest"] == {"path": flags[1], "manifest_digest": "m" * 64, "identities": None, "program_digest_per_rank": PROGRAMS}
    pin = pair.read("weights_pin.json")
    assert (pin["schema"], pin["tp"], pin["result"], pin["pairs"]) == ("verity-sweep/tp-weights-pin/v1", 2, True, 1)
    assert pin["per_rank"] == summary["weights_pin"]["per_rank"]

    cov = pair.read("coverage.json")
    assert (cov["verdict"], cov["identity_coverage"], cov["population_source"], cov["replay_pending_ranks"]) == ("PASS", True, "required-manifest", None)
    assert cov["checker"].startswith("verity/coverage/binding-r15-tp04 (verity_vllm.commit.binding per TP rank; population of record = the required manifest ")
    assert cov["required_manifest"] == {"path": flags[1], "manifest_digest": "m" * 64, "identities": None, "by_family": None, "complete": None, "ranks": 2,
                                        "program_digest_per_rank": PROGRAMS, "query_digest": None, "tp_rule": None}
    # the manifest's digest names the population; the ranks' bindings here are of their structure, so each states no manifest rows
    assert (cov["manifest_digest"], cov["manifest_rows_per_rank"], cov["structural_coverage_ok_per_rank"]) == ("m" * 64, {"0": None, "1": None}, {"0": None, "1": None})
    vc = cov["value_correspondence"]
    assert (vc["result"], vc["file"]) == (True, None)
    assert vc["method"] == "sampled-exact-replay(complete, per rank) + boundary-linkage + no value check / Match record this attempt"
    assert vc["c2_of_record"] == {"form": "A (sampled replay) per rank", "result": True,
                                  "rule": "MAN-06 O1: result true on EVERY rank and pair, else the Commit FAILs named"}
    assert sorted(vc["components"]) == ["boundary_linkage", "cross_rank_collectives", "fold_match_binding", "query_population", "replay", "sampled_replay",
                                        "weights_pin"]
    assert vc["components"]["sampled_replay"] == replay and vc["components"]["boundary_linkage"] == linkage
    assert vc["components"]["query_population"] == population and vc["components"]["replay"]["result"] is True
    assert vc["components"]["weights_pin"]["result"] is True and vc["components"]["cross_rank_collectives"]["result"] is None


@pytest.mark.parametrize("rank1,why,component,folded", [
    ({"replay": REPLAY | {"result": False, "grade": "mismatch", "why": "1 of 4 differ", "mismatch": [{"vu": "m.proj@1"}]}},
     "sampled replay (C2 of record, rank 1): mismatch -- 1 of 4 differ", "sampled_replay", "rank 1: mismatch 1 of 4 differ"),
    ({"linkage": {"result": False, "pairs": 3, "linked": 2, "why": "step 2: input ids differ"}},
     "boundary linkage (rank 1): step 2: input ids differ", "boundary_linkage", "rank 1: step 2: input ids differ"),
    ({"pin": PIN | {"result": False, "why": "1 served weight differs", "mismatched": [{"served_name": "w.down"}]}},
     "weights of record (rank 1): 1 served weight differs (first: w.down)", "weights_pin", "served weights differ from the checkpoint of record on a rank"),
    ({"pin": PIN | {"result": False, "why": "1 weight missing", "missing": ["w.up"]}},
     "weights of record (rank 1): 1 weight missing (first: w.up)", "weights_pin", "served weights differ from the checkpoint of record on a rank"),
    ({"replay": REPLAY | {"population": {"vus": 6, "strata_n": 2, "query_checks": {"ok": False, "vu_not_required": {"n": 2}}}}},
     "C2 population (rank 1): population.query_checks.ok False -- vu_not_required x2", "query_population",
     "rank 1: population.query_checks.ok False -- vu_not_required x2"),
    ({"replay": REPLAY | {"population": {"vus": 6, "strata_n": 2}}},
     f"C2 population (rank 1): {NOT_RECONCILED}", "query_population", f"rank 1: {NOT_RECONCILED}"),
], ids=["a replay that found a mismatch", "a linkage that does not hold", "a served weight that differs", "a weight that is missing",
        "a population outside the query", "a population not reconciled"])
def test_one_ranks_record_that_did_not_pass_fails_that_rank_alone_by_name(pair, tmp_path, rank1, why, component, folded):
    _replayed(pair, **rank1)
    assert pair.run(*_of_record(tmp_path)) == 1
    c = pair.rows()[1]["commit"]
    assert (c["per_rank"]["0"]["commit_pass"], c["per_rank"]["0"]["commit_fail_reason"]) == (True, "PASS")
    assert (c["per_rank"]["1"]["commit_pass"], c["per_rank"]["1"]["commit_fail_reason"]) == (False, why)
    assert (c["commit_pass"], c["commit_fail_reason"]) == (False, f"rank 1: {why}")
    summary, cov = pair.read("summary.json"), pair.read("coverage.json")
    assert (summary["pass"], summary["commit_pass"]) == (False, [False])
    assert (summary[component]["result"], summary[component]["why"]) == (False, folded)
    assert {name: summary[name]["result"] for name in ("sampled_replay", "boundary_linkage", "query_population", "weights_pin")} == \
        {name: name != component for name in ("sampled_replay", "boundary_linkage", "query_population", "weights_pin")}
    assert (cov["verdict"], cov["identity_coverage"], cov["reason"]) == ("FAIL", False, f"rank 1: {why}")
    assert cov["value_correspondence"]["components"][component]["result"] is False
    # the pin is not the replay: a served weight that differs leaves the C2 of record as the replay and the linkage found it
    assert cov["value_correspondence"]["c2_of_record"]["result"] is (component in ("query_population", "weights_pin"))
    assert cov["value_correspondence"]["result"] is (component == "weights_pin")


def test_under_a_manifest_ranks_that_replayed_and_pinned_nothing_fail_by_each_name(pair, tmp_path, capsys):
    assert pair.run(*_of_record(tmp_path)) == 1
    c = pair.rows()[1]["commit"]
    whys = {r: (f"weights unpinned (MAN-09, rank {r}): no weights of record; also: sampled replay (C2 of record, rank {r}): not run -- no result; "
                f"also: C2 population (rank {r}): {NOT_RECONCILED}") for r in "01"}
    for r in "01":
        per = c["per_rank"][r]
        assert (per["commit_pass"], per["commit_fail_reason"]) == (False, whys[r])
        assert (per["sampled_replay"], per["boundary_linkage"], per["weights_pin"]) == (None, None, None)
        assert (per["query_population"]["present"], per["query_population"]["ok"]) == (False, None)
    assert c["commit_fail_reason"] == f"rank 0: {whys['0']}; rank 1: {whys['1']}"
    summary = pair.read("summary.json")
    assert summary["pass"] is False
    assert (summary["sampled_replay"]["result"], summary["sampled_replay"]["why"]) == (None, "rank 0: None no result; rank 1: None no result")
    assert (summary["boundary_linkage"]["result"], summary["boundary_linkage"]["why"]) == (None, "rank 0: no result; rank 1: no result")
    assert (summary["query_population"]["result"], summary["query_population"]["why"]) == (False, f"rank 0: {NOT_RECONCILED}; rank 1: {NOT_RECONCILED}")
    assert (summary["weights_pin"]["result"], summary["weights_pin"]["why"]) == (None, "weights unpinned (MAN-09) on a rank")
    vc = pair.read("coverage.json")["value_correspondence"]
    assert (vc["result"], vc["c2_of_record"]["result"]) == (False, False)
    assert vc["method"] == "sampled-exact-replay(not-evaluated, per rank) + boundary-linkage + no value check / Match record this attempt"
    said = capsys.readouterr().out.splitlines()
    for r in "01":
        assert (f"[tp2commit] sampled replay rank {r}: result None grade None population None strata None picked None evaluated None equal None "
                "mismatch None not_evaluated None weights self-consistent None (not a pin) wall Nones; boundary linkage None None/None; ") in said
        assert (f"[tp2commit] weights pin rank {r}: result None root_of_record None compared None mismatched 0 missing 0 unpinned 0 shards_pinned None; ") in said


@pytest.mark.parametrize("pinned,code", [(True, 0), (False, 1)], ids=["every rank pinned", "no rank pinned"])
def test_a_manifest_with_no_sampled_replay_decides_on_the_pin_alone_and_writes_no_replay_record(pair, tmp_path, capsys, pinned, code):
    """`--no-weights-pin` records that the Commit asked for no pin (`versions.json`); under a manifest a rank with none still fails."""
    if pinned:
        for r in (0, 1):
            pair.alter("tp2_commit_finalize", r, _records(replay=None))
    assert pair.run(*_of_record(tmp_path), "--no-sampled-replay", "--no-weights-pin") == code
    c = pair.rows()[1]["commit"]
    for r in "01":
        per = c["per_rank"][r]
        assert (per["commit_pass"], per["commit_fail_reason"]) == ((True, "PASS") if pinned else (False, f"weights unpinned (MAN-09, rank {r}): no weights of record"))
        assert (per["sampled_replay"], per["boundary_linkage"]) == (None, None) and "query_population" not in per and "no_evaluator_gap" not in per
    assert not (pair.out / "sampled_replay.json").exists()
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["sampled_replay"], summary["boundary_linkage"]) == (pinned, None, None)
    assert (summary["query_population"]["result"], summary["query_population"]["why"], summary["weights_pin"]["result"]) == (None, None, True if pinned else None)
    assert pair.read("versions.json")["weights_pin"]["required"] is False
    vc = pair.read("coverage.json")["value_correspondence"]
    assert vc["method"] == "no value check / Match record this attempt" and "c2_of_record" not in vc and "replay" not in vc["components"]
    assert vc["components"]["sampled_replay"] == {"method": "sampled-exact-replay per rank (sampled_replay)", "population": None, "n": 0, "strata": None, "seed": None,
                                                  "unreplayable": [], "result": None, "why": "not run (--no-sampled-replay or no --required-manifest)"}
    assert not any(line.startswith("[tp2commit] sampled replay rank") for line in capsys.readouterr().out.splitlines())


def test_under_a_manifest_a_rank_that_skipped_its_binding_fails_and_the_coverage_with_it(pair, tmp_path):
    """Without a manifest a binding no rank ran fails no rank (`test_tp_commit_pair`); under one the rank's identities are required."""
    def skipped(answer: dict[str, Any]) -> None:
        _records()(answer)
        TP._binding(skipped="no expected population on this rank")(answer)

    pair.alter("tp2_commit_finalize", 0, _records())
    pair.alter("tp2_commit_finalize", 1, skipped)
    assert pair.run(*_of_record(tmp_path)) == 1
    c = pair.rows()[1]["commit"]
    why = "identity coverage not run: no expected population on this rank"
    assert (c["per_rank"]["0"]["commit_pass"], c["per_rank"]["1"]["commit_pass"], c["per_rank"]["1"]["commit_fail_reason"]) == (True, False, why)
    assert c["per_rank"]["1"]["binding_note"] == "no expected population on this rank" and "binding_note" not in c["per_rank"]["0"]
    cov = pair.read("coverage.json")
    assert (cov["verdict"], cov["identity_coverage"], cov["reason"], cov["skipped"]) == ("FAIL", False, f"rank 1: {why}", ["no expected population on this rank"])


def test_a_family_a_ranks_replay_could_not_evaluate_is_said_and_named_to_the_cross_rank_pass(pair, tmp_path, capsys):
    """The family is one the cross-rank pass evaluates; with the pass off (`PAIR`) its entry names the pass and no result of it."""
    family = {"family": "AllReduce2_v1", "n": 3, "why": "fed by a peer rank's partial"}
    _replayed(pair, replay=REPLAY | {"unreplayable": [family, {"family": "Odd_v1", "n": 1, "why": "no rule"}]})
    assert pair.run(*_of_record(tmp_path)) == 0
    said = capsys.readouterr().out.splitlines()
    assert f"[tp2commit]   rank 1 unreplayable: {family}" in said and not any(line.startswith("[tp2commit]   rank 0 unreplayable") for line in said)
    by_pass = {"pass": "cross_rank_collectives (TP-12)", "result": None, "picks": None, "equal": None, "sites": None, "sites_covered": None,
               "sites_missing": None, "artifact": "xrank_pair0.json",
               "rule": "sampled strata over the committed collective outputs (>= per_stratum per site x kind x phase), not every VU"}
    listed = pair.read("finalize_pair0.json")[1]["sampled_replay"]["sampled_replay"]["unreplayable"]
    assert listed == [family | {"evaluated_by": by_pass}, {"family": "Odd_v1", "n": 1, "why": "no rule"}]
    assert pair.read("finalize_pair0.json")[0]["sampled_replay"]["sampled_replay"].get("unreplayable") is None
    record = pair.read("sampled_replay.json")
    assert record["sampled_replay"]["unreplayable"] == [family | {"ranks": [1], "evaluated_by": by_pass}, {"family": "Odd_v1", "n": 1, "why": "no rule", "ranks": [1]}]
    assert record["unreplayable_collectives_evaluated_by"]["families"] == sorted(cross_rank_families(2))


# a deferred replay

def _deferred(pair: TP._Pair, rank: int, pin: dict[str, Any] = PIN) -> Callable[[dict[str, Any]], None]:
    """A change of a rank's finalize answer to what a rank that deferred its replay answers (`rank_replay.defer`): the deferred
    records and its bundle, whose partial dir holds the rank's binding map."""
    def change(answer: dict[str, Any]) -> None:
        if answer["binding"] is None:
            return
        part = RB.partial(pair.out, 0, rank=rank)
        part.mkdir(parents=True)
        (part / "binding_map.json").write_text(json.dumps(pair.ranks[rank]._tp2_commit["bmap"], default=str) + "\n")
        replay, linkage = RB.deferred_records(RB.path(pair.out, 0, rank=rank))
        answer["sampled_replay"] = {"sampled_replay": replay | {"rank": rank}, "boundary_linkage": linkage | {"rank": rank},
                                    "bundle": {"partial": str(part), "hashed": {}, "engine_of_record": {"vllm": "a stand-in"}, "programs": {}}}
        answer["weights_pin"] = copy.deepcopy(pin)
    return change


def test_a_deferred_pair_is_pending_on_every_rank_and_seals_each_ranks_bundle(pair, tmp_path, capsys):
    for r in (0, 1):
        pair.alter("tp2_commit_finalize", r, _deferred(pair, r))
    flags = _of_record(tmp_path)
    assert pair.run(*flags, "--replay-deferred") == 0
    # the warm-up's finalize, then the run of record's, which is told where its bundle goes
    assert [kw for method, kw in pair.sent if method == "tp2_commit_finalize"] == [{}, {"defer": {"out": str(pair.out), "pair": 0}}]
    c, roots = pair.rows()[1]["commit"], pair.roots()
    assert (c["commit_pass"], c["commit_fail_reason"]) == (None, RB.PENDING)
    for r in (0, 1):
        per = c["per_rank"][str(r)]
        assert (per["commit_pass"], per["commit_fail_reason"]) == (None, RB.PENDING)
        assert per["replay_deferred"] == {"pre": [True, "PASS"], "bundle": f"replay_bundle_p0_rank{r}"}
        assert (per["sampled_replay"]["grade"], per["sampled_replay"]["result"], per["boundary_linkage"]["result"]) == ("deferred", None, None)
        assert per["weights_pin"]["result"] is True and "query_population" not in per
    # the bundle a rank named is the driver's alone: the rank's finalize on disk does not hold it
    assert ["bundle" in f["sampled_replay"] for f in pair.read("finalize_pair0.json")] == [False, False]
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["commit_pass"]) == (None, [None])
    why = "deferred: ranks ['0', '1'] replay on CPU from their bundles (`verity-vllm tp-commit-replay`)"
    assert (summary["sampled_replay"]["result"], summary["sampled_replay"]["why"]) == (None, why)
    assert (summary["boundary_linkage"]["result"], summary["boundary_linkage"]["deferred"]) == (None, {"ranks": ["0", "1"]})
    assert (summary["query_population"]["result"], summary["query_population"]["why"]) == (None, "deferred: ranks ['0', '1'] replay from their bundles")
    bundles = summary["replay_deferred"]["bundles"]
    assert (summary["replay_deferred"]["record"], summary["replay_deferred"]["pending_pairs"]) == (C2.DEFERRED, [0])
    assert [(b["pair"], b["rank"], b["path"]) for b in bundles] == [(0, r, str(RB.path(pair.out, 0, rank=r))) for r in (0, 1)]
    for b in bundles:
        sealed = RB.open_bundle(b["path"])
        context = sealed["context"]
        assert (sealed["manifest_sha256"], b["files"]) == (b["manifest_sha256"], 2)
        assert (context["tp"], context["pair"], context["run_root"], context["tp_run_root"]) == \
            ({"world": 2, "rank": b["rank"]}, 0, roots[b["rank"]].hex(), c["tp_run_root"])
        assert (context["required_manifest"], context["engine_of_record"], context["programs"]) == (flags[1], {"vllm": "a stand-in"}, {})
        assert context["replay"]["programs"] == {"0": flags[3], "1": flags[5]} and context["replay"]["of_record"] == {r: [PROGRAMS[r]] for r in "01"}
        assert json.loads((Path(b["path"]) / "binding_map.json").read_text())["digest"] == c["per_rank"][str(b["rank"])]["binding"]["map_digest"]
    assert not list(pair.out.glob("*.partial"))
    record = pair.read(C2.DEFERRED)
    assert (record["world"], record["pending_pairs"], record["bundles"], record["all_pass"], record["replayed"]) == (2, [0], bundles, True, [])
    assert (record["required_manifest"], record["xrank"], record["xrank_families"]) == (flags[1], {"0": {"result": None, "by_family": None}},
                                                                                         sorted(cross_rank_families(2)))
    assert sorted(record["components"]) == ["cross_rank_collectives", "fold_match_binding", "weights_pin"]
    cov = pair.read("coverage.json")
    assert (cov["verdict"], cov["replay_pending_ranks"], cov["value_correspondence"]["result"]) == ("PASS", [0, 1], None)
    assert record["vc_base"] is None and cov["value_correspondence"]["c2_of_record"]["result"] is None
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith("[tp2commit] PENDING delta ")
    assert last.endswith(f"; the C2 replay of every rank runs on CPU from {[b['path'] for b in bundles]}")


def test_a_deferred_pair_one_rank_of_which_is_decided_fails_and_keeps_no_bundle(pair, tmp_path, capsys):
    pair.alter("tp2_commit_finalize", 0, _deferred(pair, 0))
    pair.alter("tp2_commit_finalize", 1, _deferred(pair, 1, PIN | {"result": False, "why": "1 served weight differs", "mismatched": [{"served_name": "w.down"}]}))
    assert pair.run(*_of_record(tmp_path), "--replay-deferred") == 1
    c = pair.rows()[1]["commit"]
    why = "weights of record (rank 1): 1 served weight differs (first: w.down)"
    assert (c["per_rank"]["0"]["commit_pass"], c["per_rank"]["0"]["commit_fail_reason"]) == (None, RB.PENDING)
    assert (c["per_rank"]["1"]["commit_pass"], c["per_rank"]["1"]["commit_fail_reason"]) == (False, why)
    assert (c["commit_pass"], c["commit_fail_reason"]) == (False, f"rank 1: {why}")
    summary = pair.read("summary.json")
    assert summary["pass"] is False and "replay_deferred" not in summary
    assert not (pair.out / C2.DEFERRED).exists() and not list(pair.out.glob("replay_bundle_*"))
    assert pair.read("coverage.json")["replay_pending_ranks"] == [0]
    assert capsys.readouterr().out.splitlines()[-1].startswith("[tp2commit] FAIL delta ")
