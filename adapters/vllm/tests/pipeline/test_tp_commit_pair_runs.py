"""`tp-commit` (`verity_vllm.pipeline.tp.commit`) around its pairs: what the ranks' answers to the driver's make and installs gate
(a call that did not reach every rank, an MoE block source, a rank whose Program is not the manifest's), the checkpoint of record
it has every rank pin its weights against (`--checkpoint-dir`), the value-check run after the pairs (`--value-check`), the traced
run (`--trace-dir`) and a run of one arm (`--only-arm`).
CPU-only, in the frames of `tests.pipeline.test_tp_commit_options` (to the ranks' make) and `tests.pipeline.test_tp_commit_pair`
(a whole Commit over two ranks)."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.pipeline import test_tp_commit_options as TO
from tests.pipeline import test_tp_commit_pair as TP
from verity_vllm.acquire import as_loaded as AL
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline.tp import commit as TPCM
from verity_vllm.pipeline.tp import lease as TPL

driver, commit, pair = TO.driver, TO.commit, TP.pair

#: the classes a TP Commit requires of every rank when it is given none
DEFAULT_CLASSES = ["input_token_ids", "instance_outputs_native", "sampled_token_ids", "weights", "fa2_hidden_m1_stream", "tp_rank_partials"]
MOE = {"layers": 2, "planes": 4}


def _says(**fields: Any) -> Callable[[dict[str, Any]], None]:
    return lambda answer: answer.update(fields)


def _printed(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


# what the ranks answer the driver's make and installs

def test_a_make_that_did_not_reach_every_rank_is_refused_before_any_run(pair):
    pair.lose("tp2_commit_make", 1)
    with pytest.raises(SystemExit, match="^fail-closed: committer constructed on 1/2 ranks$"):
        pair.run()
    assert pair.sent == [] and not (pair.out / "committer_make.json").exists() and not (pair.out / "runs.jsonl").exists()


def test_an_moe_block_source_on_some_ranks_only_is_refused(pair):
    pair.alter("tp2_commit_make", 1, _says(moe_source=MOE))
    with pytest.raises(SystemExit, match=r"^fail-closed: the MoE block source attached on ranks \[1\] of 2 -- every rank of an MoE model binds its planes$"):
        pair.run()
    assert pair.sent == []


def test_an_moe_block_source_on_every_rank_makes_its_class_required_of_a_commit_that_names_none(pair, capsys):
    for r in (0, 1):
        pair.alter("tp2_commit_make", r, _says(moe_source=MOE))
    # no --required-classes: these ranks declare none of the default classes, and the verdict says so of each
    assert TPCM.main(TO._options(pair.out, ("--pairs", "1", "--openings", "4", "--xrank-per-stratum", "0"))) == 1
    required = [*DEFAULT_CLASSES, "moe_block_stream"]
    versions, summary = pair.read("versions.json"), pair.read("summary.json")
    assert versions["required_classes"] == summary["required_classes"] == required
    assert versions["required_classes_source"] == ("default (REL-02) + tp_rank_partials (R14 TP-03), checked per rank"
                                                   " + moe_block_stream (R15 TP-MOE-SRC: MoE-runner modules on every rank)")
    assert versions["moe_source"] == {"0": MOE, "1": MOE}
    assert f"[tp2commit] MoE block source attached on every rank ({MOE}); required classes {required}" in _printed(capsys)
    c = pair.rows()[1]["commit"]
    assert [c["per_rank"][r]["commit_fail_reason"] for r in "01"] == ["required class 'input_token_ids' not declared: its producer never fired / was not attached (R-14 omitted producer)"] * 2


def test_an_moe_block_source_adds_no_class_to_the_classes_a_commit_names(pair):
    for r in (0, 1):
        pair.alter("tp2_commit_make", r, _says(moe_source=MOE))
    assert pair.run() == 0
    versions = pair.read("versions.json")
    assert (versions["required_classes"], versions["required_classes_source"], "moe_source" in versions) == (["instance_outputs_native"], "cli", False)


def test_a_rank_whose_program_is_not_the_manifests_is_refused_once_every_rank_is_made(pair):
    pair.alter("tp2_commit_make", 1, _says(required_manifest={"manifest_digest": "m" * 64, "identities": 7, "program_digest_matches": False, "complete": True}))
    with pytest.raises(SystemExit, match="^fail-closed: rank 1: manifest Program digest != this rank's Program digest$"):
        pair.run()
    assert pair.sent == [] and [m["rank"] for m in pair.read("committer_make.json")] == [0, 1]


def test_an_install_that_did_not_reach_every_rank_is_refused(pair):
    pair.lose("tp2_commit_install", 1)
    with pytest.raises(SystemExit, match="^fail-closed: install did not reach every rank$"):
        pair.run()
    # the warm-up's run, the pair's control, then the install that is refused
    assert [method for method, _kw in pair.sent] == ["tp2_commit_install", "tp2_commit_finalize", "tp2_commit_reset", "tp2_commit_install"]
    assert [row["arm"] for row in pair.rows()] == ["control"]


# the checkpoint of record

def test_a_checkpoint_dir_is_pinned_once_and_handed_to_every_rank_with_its_program_dir(commit, tmp_path, monkeypatch):
    pin, taken = {"shards": {"model-00001.safetensors": "5" * 64}}, []
    monkeypatch.setattr(WR, "record_shard_pin", lambda d, **kw: taken.append((d, kw)) or pin)
    ckpt, builds = str(tmp_path / "ckpt"), {str(r): TO._build(tmp_path, r, str(r) * 64) for r in (0, 1)}
    llm, out = commit("--checkpoint-dir", ckpt, "--checkpoints-manifest", "checkpoints.json", "--checkpoint-model", "org/model", "--build-rank0", builds["0"],
                      "--build-rank1", builds["1"], "--serve-shard-from", "0:w:1")
    assert taken == [(ckpt, {"checkpoints_manifest": "checkpoints.json", "model": "org/model"})]
    assert llm.made["weights_pin"] == {"checkpoint_dir": ckpt, "checkpoints_manifest": "checkpoints.json", "model": "org/model", "program_dirs": builds, "shard_pin": pin}
    # each rank registers its weights as it loaded them, composed from the same checkpoint
    loaded = llm.made["weights_as_loaded"]
    assert (loaded.func, loaded.args, sorted(loaded.keywords)) == (AL.attach_rank, (), ["composer"])
    composer = loaded.keywords["composer"]
    assert (composer.func, composer.args, composer.keywords) == (WR.rank_composer, (), {"checkpoint_dir": ckpt, "checkpoints_manifest": "checkpoints.json", "model": "org/model"})
    versions = json.loads((out / "versions.json").read_text())
    assert (versions["weights_pin"]["checkpoint_dir"], versions["weights_pin"]["checkpoints_manifest"], versions["weights_pin"]["required"]) == (ckpt, "checkpoints.json", False)
    assert llm.made["serve_shard_from"] == versions["negatives"]["serve_shard_from"] == {"0": [{"param": "w", "from_rank": 1}]}


@pytest.mark.parametrize("built,have", [((), []), ((1,), ["1"])], ids=["no rank's Build", "one rank's Build"])
def test_a_checkpoint_dir_needs_every_ranks_build(driver, tmp_path, monkeypatch, built, have):
    monkeypatch.setattr(WR, "record_shard_pin", lambda d, **kw: pytest.fail("pinned a checkpoint no rank composes from"))
    flags = [flag for r in built for flag in (f"--build-rank{r}", TO._build(tmp_path, r, str(r) * 64))]
    with pytest.raises(SystemExit) as refused:
        TPCM.main(TO._options(driver[1], ("--checkpoint-dir", str(tmp_path / "ckpt"), *flags)))
    assert str(refused.value) == f"--checkpoint-dir needs a rank Program dir for every rank (--build-rank<r>): have {have} of 2"
    assert driver[0].made == {}


# the value-check run

def test_a_value_check_install_that_did_not_reach_every_rank_is_refused_after_the_pairs(pair):
    pair.lose("tp2_commit_value_check_install", 1)
    with pytest.raises(SystemExit, match="^fail-closed: value-check install did not reach every rank$"):
        pair.run("--value-check")
    assert [row["arm"] for row in pair.rows()] == ["control", "instrumented"] and not (pair.out / "value_check.json").exists()


def test_a_value_check_one_rank_fails_fails_the_commit_whose_pair_passed(pair, capsys):
    pair.alter("tp2_commit_value_check", 1, _says(verdict="FAIL", method="T + R", modules={"verdict": "FAIL", "tensors": 3, "tap_mismatch_n": 1},
                                                  tp_partials={"verdict": "PASS", "tensors": 2}))
    assert pair.run("--value-check") == 1
    check = pair.read("value_check.json")
    assert (check["verdict"], check["tokens_equal_ref"], check["ranks_answered"]) == ("FAIL", True, [0, 1])
    assert {r: v["verdict"] for r, v in check["per_rank"].items()} == {"0": "PASS", "1": "FAIL"}
    summary, cov = pair.read("summary.json"), pair.read("coverage.json")
    # the pair's verdict stands; the Commit's does not
    assert (summary["pass"], summary["commit_pass"], summary["value_check"]) == (False, [True], {"verdict": "FAIL", "per_rank": {"0": "PASS", "1": "FAIL"}})
    assert pair.rows()[1]["commit"]["commit_pass"] is True and (cov["verdict"], cov["value_correctness"]) == ("PASS", False)
    vc = cov["value_correspondence"]
    assert (vc["result"], vc["file"], vc["components"]["tap_reexec"]["result"], vc["components"]["tap_reexec"]["per_rank"]) == \
        (False, "value_check.json", False, {"0": "PASS", "1": "FAIL"})
    assert vc["identities_checked"] == {"0": {"module_outputs": None, "tp_partials": None}, "1": {"module_outputs": 3, "tp_partials": 2}}
    assert vc["per_rank"]["1"] == {"verdict": "FAIL", "modules": "FAIL", "tp_partials": "PASS"}
    assert ("[tp2commit] VALUE CHECK rank 1: FAIL | modules FAIL tensors 3 tap-eq None tap-mismatch 1 reexec None/None missing 0 | tp partials PASS tensors 2 "
            "tap-eq None reexec None/None missing None by_member None") in _printed(capsys)


def test_a_value_check_one_rank_did_not_answer_fails_the_commit(pair):
    pair.alter("tp2_commit_value_check", 0, _says(t6_4=LATE_READ))
    pair.lose("tp2_commit_value_check", 1)
    assert pair.run("--value-check") == 1
    check = pair.read("value_check.json")
    assert (check["verdict"], check["ranks_answered"], sorted(check["per_rank"])) == ("FAIL", [0], ["0"])
    assert pair.read("summary.json")["value_check"] == {"verdict": "FAIL", "per_rank": {"0": "PASS"}}
    # the late-read record of the rank that answered holds; the Commit's, a rank short, does not
    late = pair.read("t6_4_check.json")
    assert (late["per_rank"]["0"]["result"], late["result"]) == (True, False)


LATE_READ = {"applicable": True, "result": True, "layers_compared": 2, "equal_pairs": 4, "missing_pairs": 0, "clone_mismatches": 0, "out_tensors_vs_clone": 6,
             "clone_rule": "clone at the hook", "rule": "T6.4", "pairs": [{"layer": 0, "equal": True}]}
DENSE = {"applicable": False, "result": None}


@pytest.mark.parametrize("rank1,applicable,result,counted,mismatches", [
    (LATE_READ, True, True, [4, 8, 0, 12], 0),
    (LATE_READ | {"result": False, "missing_pairs": 1, "clone_mismatches": None}, True, False, [4, 8, 1, 12], None),
    (DENSE, True, False, [2, 4, 0, 6], None),
    (None, True, False, [2, 4, 0, 6], None),
], ids=["every rank's holds", "one rank's does not", "one rank's does not apply", "one rank has none"])
def test_the_value_checks_late_read_record_is_every_ranks_folded(pair, rank1, applicable, result, counted, mismatches):
    pair.alter("tp2_commit_value_check", 0, _says(t6_4=LATE_READ))
    pair.alter("tp2_commit_value_check", 1, _says(**({"t6_4": rank1} if rank1 is not None else {})))
    assert pair.run("--value-check") == 0
    record = pair.read("t6_4_check.json")
    assert (record["schema"], record["tp"], record["applicable"], record["result"]) == ("verity/commit-integ/t6_4-check/v1", 2, applicable, result)
    assert [record[k] for k in ("layers_compared", "equal_pairs", "missing_pairs", "out_tensors_vs_clone")] == counted and record["clone_mismatches"] == mismatches
    assert (record["clone_rule"], record["rule"]) == ("clone at the hook", "T6.4")
    assert record["per_rank"]["1"] == (rank1 if rank1 is not None else {"result": False, "error": "no t6_4 in the rank's value-check result"})
    assert record["pairs"] == [{"layer": 0, "equal": True, "rank": r} for r in ((0, 1) if rank1 and "pairs" in rank1 else (0,))]


def test_a_late_read_record_that_applies_to_no_rank_is_no_result(pair):
    for r in (0, 1):
        pair.alter("tp2_commit_value_check", r, _says(t6_4=DENSE))
    assert pair.run("--value-check") == 0
    record = pair.read("t6_4_check.json")
    assert (record["applicable"], record["result"], record["per_rank"]) == (False, None, {"0": DENSE, "1": DENSE})


# the traced run and a run of one arm

def test_a_traced_run_follows_the_pairs_and_keeps_each_arms_profile_and_roots(pair, tmp_path, capsys):
    trace = tmp_path / "trace"
    assert pair.run("--trace-dir", str(trace), "--trace-ranks", "0,1") == 0
    control, traced = [json.loads(line) for line in (trace / "traced.jsonl").read_text().splitlines()]
    assert (control["arm"], "run_roots" in control) == ("control", False)
    # the traced run commits as a pair's does, without its checks: its roots are the ranks'
    assert (traced["arm"], traced["run_roots"]) == ("instrumented", {str(r): root.hex() for r, root in pair.roots().items()})
    assert [(method, kw) for method, kw in pair.sent[-3:]] == [("tp2_commit_install", {}), ("tp2_commit_finalize", {"checks": False}), ("tp2_commit_reset", {})]
    profiled = [(method, kw) for method, kw in pair.asked if str(method).startswith("tp2_profiler")]
    assert profiled == [step for arm in ("control", "instrumented") for step in (
        ("tp2_profiler_start", {}), ("tp2_profiler_stop", {"out_dir": str(trace / f"torch.{arm}"), "keep_trace": True, "keep_ranks": [0, 1]}))]
    assert [row["profiler"] == pair._stand_in("tp2_profiler_stop") for row in (control, traced)] == [True, True]
    printed = _printed(capsys)
    assert [f"[tp2commit] traced torch {arm}: 0.500s steps 4 -> {trace / f'torch.{arm}'}" in printed for arm in ("control", "instrumented")] == [True, True]
    # the Commit's verdict is its pair's: the traced run is no row of it
    assert [row["arm"] for row in pair.rows()] == ["control", "instrumented"] and pair.read("summary.json")["pass"] is True


@pytest.mark.parametrize("flags,stops", [((), ["work.trace"]), (("--value-check",), [])], ids=["the Commit's last run", "a value check after it"])
def test_a_traced_run_gives_the_gpus_back_unless_a_value_check_runs_after_it(pair, tmp_path, monkeypatch, flags, stops):
    stopped: list[str] = []
    monkeypatch.setattr(TPL, "stop", lambda llm, rpc_timeout, after: stopped.append(after))
    assert pair.run("--trace-dir", str(tmp_path / "trace"), "--trace-ranks", "0", *flags) == 0
    assert stopped == stops


@pytest.mark.parametrize("arm,fails,code,decided", [("control", False, 0, None), ("instrumented", False, 0, True), ("instrumented", True, 1, False)],
                         ids=["control", "instrumented", "instrumented, one run failing"])
def test_a_run_of_one_arm_is_that_arm_alone_and_its_summary_says_what_was_decided(pair, arm, fails, code, decided):
    if fails:
        pair.alter("tp2_commit_open", 1, lambda answer: answer.update(verified=3) if pair.ranks[1]._tp2_commit["com"].runs == 3 else None)
    assert pair.run("--only-arm", arm, "--pairs", "2", "--all-pairs") == code
    rows, summary = pair.rows(), pair.read("summary.json")
    assert [(row["arm"], row["pair"], "commit" in row) for row in rows] == [(arm, 0, arm != "control"), (arm, 1, arm != "control")]
    assert (summary["only_arm"], summary["runs"], summary["commit_pass"], summary["pass"]) == (arm, 2, decided, not fails)
