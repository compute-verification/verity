"""The config run (docs/vllm-config-sweep-plan.md §2-§4): the row mode (`--config-run 1`), the config record, the derived-Program cache,
the family-stratified k-unit replay sample.  CPU only: row dirs, workloads and processes are stand-ins."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.pipeline.test_row import _row
from verity_vllm.check.replay.driver import family_strata
from verity_vllm.check.replay.sample import sample
from verity_vllm.engine import served as SV
from verity_vllm.pipeline import config_record as CR
from verity_vllm.pipeline import program_cache as PC
from verity_vllm.pipeline import row_config as RC
from verity_vllm.pipeline import row_records as R_
from verity_vllm.pipeline.row_driver import Stop

ROW = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"


# ---------------------------------------------------------------------------------------------------------------- the row mode

def test_a_config_run_skips_the_match_and_word_checks_the_build(tmp_path, monkeypatch):
    r = _row(tmp_path, "--config-run", "1")
    calls = []
    for name in ("build", "match", "commit", "research_result", "timeline_report"):
        monkeypatch.setattr(r, name, lambda n=name: calls.append(n))
    monkeypatch.setattr(r, "prepare", lambda: (calls.append("prepare"), setattr(r, "trap", True)))
    monkeypatch.setattr(RC, "word_check", lambda row: calls.append("word_check"))
    monkeypatch.setattr(RC, "config_record", lambda row: calls.append("config_record"))
    assert r.run() == 0
    assert calls == ["prepare", "build", "word_check", "commit", "research_result", "config_record", "timeline_report"]


def test_the_word_check_gets_the_row_s_allowed_max_gates(tmp_path):
    from types import SimpleNamespace
    seen = {}
    env = {"VERITY_QWORD_MAX_GATES": "GumbelTopPTokenSelect_v2=30000000", "VERITY_QWORD_MAX_GATES_ALLOWED": "GumbelTopPTokenSelect_v2=30000000"}
    row = SimpleNamespace(env=SimpleNamespace(environ=env), d=str(tmp_path), row=ROW, cli=lambda *a: list(a),
                          run_tl=lambda name, to, log, cmd: seen.setdefault("cmd", cmd) and 0, verdict=lambda *a: None)
    RC.word_check(row)
    cmd = seen["cmd"]
    assert cmd[cmd.index("--build-max-gates") + 1] == cmd[cmd.index("--allowed-max-gates") + 1] == "GumbelTopPTokenSelect_v2=30000000"


def test_a_config_run_that_stops_still_writes_its_record(tmp_path, monkeypatch):
    r = _row(tmp_path, "--config-run", "1")
    calls = []
    monkeypatch.setattr(r, "prepare", lambda: setattr(r, "trap", True))
    monkeypatch.setattr(r, "build", lambda: None)

    def word_fail(row):
        raise Stop(20)

    monkeypatch.setattr(RC, "word_check", word_fail)
    monkeypatch.setattr(r, "research_result", lambda: None)
    monkeypatch.setattr(RC, "config_record", lambda row: calls.append("config_record"))
    assert r.run() == 20 and calls == ["config_record"]


def test_the_config_run_s_commit_is_one_instrumented_run_under_bounded_staging(tmp_path):
    r = _row(tmp_path, "--config-run", "1", "--replay-k", "128")
    r.hidden_so, r.B = "/x/verity_fa2_matReqG.so", 1
    cmd = r.commit_cmd([])
    assert cmd[cmd.index("--pairs") + 1] == "1" and cmd[cmd.index("--warmup") + 1] == "1"
    assert cmd[cmd.index("--only-arm") + 1] == "instrumented" and "--bounded-staging" in cmd
    assert cmd[cmd.index("--replay-k") + 1] == "128" and "--vu-export-dir" not in cmd
    assert cmd[cmd.index("--replay-draw") + 1] == "uniform", "the gate's k units are a uniform draw unless the row asks for family"
    fam = _row(tmp_path / "fam", "--config-run", "1", "--replay-draw", "family")
    fam.hidden_so, fam.B = "/x/verity_fa2_matReqG.so", 1
    assert fam.commit_cmd([])[fam.commit_cmd([]).index("--replay-draw") + 1] == "family"
    row = _row(tmp_path / "row")
    row.hidden_so, row.B = "/x/verity_fa2_matReqG.so", 1
    plain = row.commit_cmd([])
    assert "--only-arm" not in plain and plain[plain.index("--pairs") + 1] == "3", "a row of record keeps its pairs"


def test_the_commit_s_own_parser_takes_every_flag_the_config_run_passes(tmp_path):
    pytest.importorskip("torch")                      # commit.py imports torch at module level (check runs with CPU torch)
    from verity_vllm.pipeline import cli
    r = _row(tmp_path, "--config-run", "1", "--replay-k", "128")
    r.hidden_so, r.B = "/x/verity_fa2_matReqG.so", 1
    cmd = r.commit_cmd([])
    args = cli.parse("commit", cmd[cmd.index("commit") + 1:], {})
    assert (args.only_arm, args.bounded_staging, args.replay_k, args.replay_draw, args.pairs, args.warmup) == ("instrumented", True, 128,
                                                                                                                "uniform", 1, 1)


@pytest.mark.parametrize("commit_pass,rc,expect", [(True, 0, None), (False, 0, 12), (True, 3, 12)])
def test_the_config_run_s_commit_decides_by_its_one_run(tmp_path, commit_pass, rc, expect):
    r = _row(tmp_path, "--config-run", "1")
    (Path(r.d) / "commit").mkdir()
    (Path(r.d) / "commit" / "summary.json").write_text(json.dumps({"only_arm": "instrumented", "runs": 1, "commit_pass": commit_pass}))
    if expect is None:
        RC.commit_verdict(r, rc, 60)
        assert "commit PASS" in (Path(r.d) / "stages.txt").read_text()
    else:
        with pytest.raises(Stop) as e:
            RC.commit_verdict(r, rc, 60)
        assert e.value.rc == expect


def test_a_deferred_config_run_passes_the_flag_and_parses(tmp_path):
    r = _row(tmp_path, "--config-run", "1", "--replay-deferred", "1")
    r.hidden_so, r.B = "/x/verity_fa2_matReqG.so", 1
    cmd = r.commit_cmd([])
    assert "--replay-deferred" in cmd
    plain = _row(tmp_path / "plain", "--config-run", "1")
    plain.hidden_so, plain.B = "/x/verity_fa2_matReqG.so", 1
    assert "--replay-deferred" not in plain.commit_cmd([])
    of_record = _row(tmp_path / "row", "--replay-deferred", "1")
    of_record.hidden_so, of_record.B = "/x/verity_fa2_matReqG.so", 1
    assert "--replay-deferred" not in of_record.commit_cmd([]), "only a config run defers its replay"
    pytest.importorskip("torch")
    from verity_vllm.pipeline import cli
    assert cli.parse("commit", cmd[cmd.index("commit") + 1:], {}).replay_deferred == "full"
    slim = _row(tmp_path / "slim", "--config-run", "1", "--replay-deferred", "slim")
    slim.hidden_so, slim.B = "/x/verity_fa2_matReqG.so", 1
    sc = slim.commit_cmd([])
    assert cli.parse("commit", sc[sc.index("commit") + 1:], {}).replay_deferred == "slim"


def _pending_commit(r, *, bundles):
    (Path(r.d) / "commit").mkdir()
    (Path(r.d) / "commit" / "summary.json").write_text(json.dumps({"only_arm": "instrumented", "runs": 1, "commit_pass": None}))
    (Path(r.d) / "commit" / "verdict.json").write_text(json.dumps({"commit_pass": None, "replay_deferred": {
        "pending_pairs": [1], "bundles": [{"pair": 1, "path": "/b/replay_bundle_p1", "manifest_sha256": "ab" * 32}] * bundles}}))


def test_a_deferred_commit_is_pending_with_its_bundle_and_fails_without_one(tmp_path):
    r = _row(tmp_path, "--config-run", "1", "--replay-deferred", "1")
    _pending_commit(r, bundles=1)
    assert RC.commit_verdict(r, 0, 60) is True
    assert "commit PENDING" in (st := (Path(r.d) / "stages.txt").read_text()) and "/b/replay_bundle_p1" in st
    with pytest.raises(Stop):
        RC.commit_verdict(r, 3, 60)
    bare = _row(tmp_path / "bare", "--config-run", "1", "--replay-deferred", "1")
    _pending_commit(bare, bundles=0)
    with pytest.raises(Stop):
        RC.commit_verdict(bare, 0, 60)


def test_a_pending_config_run_writes_no_record(tmp_path, monkeypatch):
    r = _row(tmp_path, "--config-run", "1", "--replay-deferred", "1")
    calls = []
    for name in ("build", "research_result", "timeline_report"):
        monkeypatch.setattr(r, name, lambda n=name: calls.append(n))
    monkeypatch.setattr(r, "prepare", lambda: setattr(r, "trap", True))
    monkeypatch.setattr(r, "commit", lambda: setattr(r, "replay_pending", True))
    monkeypatch.setattr(RC, "word_check", lambda row: None)
    monkeypatch.setattr(RC, "config_record", lambda row: calls.append("config_record"))
    assert r.run() == 0 and "config_record" not in calls


def test_a_failed_commit_and_a_decided_replay_delete_the_bundle_and_a_crashed_replay_keeps_it(tmp_path, monkeypatch):
    r = _row(tmp_path, "--config-run", "1", "--replay-deferred", "1")
    _pending_commit(r, bundles=1)
    (Path(r.d) / "commit" / "replay_bundle_p1" / "store").mkdir(parents=True)
    (Path(r.d) / "commit" / "replay_bundle_p2.partial").mkdir()
    assert RC.commit_verdict(r, 0, 60) is True and (Path(r.d) / "commit" / "replay_bundle_p1").is_dir(), "a pending Commit keeps its bundle"
    r.ck = str(tmp_path)
    slim = Path(r.d) / "commit" / "replay_slim_p1"

    def replay(st, to, lf, cmd, rc, keeps):
        if keeps:                                                    # the replay writes its slim keep before it exits
            slim.mkdir(exist_ok=True)
            (slim / "slim.json").write_text("{}")
        return rc

    for rc, keeps, kept in ((2, False, True), (3, False, True), (3, True, False)):
        monkeypatch.setattr(r, "run_tl", lambda st, to, lf, cmd, rc=rc, keeps=keeps: replay(st, to, lf, cmd, rc, keeps))
        with pytest.raises(Stop):
            RC.replay(r)
        assert (Path(r.d) / "commit" / "replay_bundle_p1").is_dir() is kept, (rc, keeps)
    assert "left no slim keep" in (Path(r.d) / "row.log").read_text(), "a decided replay without its slim keep keeps the bundle, by name"
    assert not list((Path(r.d) / "commit").glob("replay_bundle_p*")) and (slim / "slim.json").is_file()
    f = _row(tmp_path / "fail", "--config-run", "1", "--replay-deferred", "1")
    _pending_commit(f, bundles=1)
    (Path(f.d) / "commit" / "replay_bundle_p1.partial").mkdir()
    with pytest.raises(Stop):
        RC.commit_verdict(f, 3, 60)
    assert not list((Path(f.d) / "commit").glob("replay_bundle_p*")), "a failed Commit leaves no bundle"


def test_the_replay_stage_replays_the_one_bundle_beside_the_row_on_cpu(tmp_path, monkeypatch):
    r = _row(tmp_path, "--config-run", "1", "--replay-deferred", "1")
    _pending_commit(r, bundles=1)
    r.ck = str(tmp_path)
    seen = {}
    monkeypatch.setattr(r, "run_tl", lambda st, to, lf, cmd: seen.update(st=st, cmd=cmd) or 0)
    RC.replay(r)
    cmd = seen["cmd"]
    assert seen["st"] == "replay.commit_replay" and cmd[cmd.index("commit-replay") + 1] == f"{r.d}/commit/replay_bundle_p1"
    assert cmd[cmd.index("--row-dir") + 1] == r.d and cmd[cmd.index("--checkpoint-dir") + 1] == str(tmp_path)
    assert "replay PASS" in (Path(r.d) / "stages.txt").read_text()
    monkeypatch.setattr(r, "run_tl", lambda st, to, lf, cmd: 3)
    with pytest.raises(Stop) as e:
        RC.replay(r)
    assert e.value.rc == 12 and "replay FAIL" in (Path(r.d) / "stages.txt").read_text()


@pytest.mark.parametrize("config_run,bundles,ck,why", [("0", 1, True, "config run"), ("1", 0, True, "0 replay bundles"),
                                                       ("1", 1, False, "no checkpoint dir")])
def test_the_replay_stage_refuses_what_it_cannot_decide(tmp_path, monkeypatch, config_run, bundles, ck, why):
    r = _row(tmp_path, "--config-run", config_run, "--replay-deferred", "1")
    _pending_commit(r, bundles=bundles)
    r.ck = str(tmp_path) if ck else ""
    monkeypatch.setattr(r, "run_tl", lambda *a: pytest.fail("nothing to replay"))
    with pytest.raises(Stop):
        RC.replay(r)
    assert why in (Path(r.d) / "stages.txt").read_text()


def test_a_replay_stage_writes_the_record_the_pending_commit_left_unwritten(tmp_path, monkeypatch):
    r = _row(tmp_path, "--config-run", "1", "--replay-deferred", "1")
    r.stages = "replay"
    calls = []
    for name in ("research_result", "timeline_report"):
        monkeypatch.setattr(r, name, lambda n=name: calls.append(n))
    monkeypatch.setattr(r, "prepare", lambda: setattr(r, "trap", True))
    monkeypatch.setattr(RC, "replay", lambda row: calls.append("replay"))
    monkeypatch.setattr(RC, "config_record", lambda row: calls.append("config_record"))
    assert r.run() == 0 and calls == ["replay", "research_result", "config_record", "timeline_report"]


def test_a_config_record_carries_the_replay_s_weights_attest(tmp_path):
    d = _row_dir(tmp_path)
    run = json.loads((d / "commit" / "runs.jsonl").read_text())
    run["commit"]["weights_attest"] = {"root_openings": "3/3", "sources": {"checkpoint": 3, "bundle": 0}}
    (d / "commit" / "runs.jsonl").write_text(json.dumps(run) + "\n")
    assert CR.record(d, ROW)["commit"]["weights_attest"]["root_openings"] == "3/3"
    assert "weights_attest" not in CR.record(_row_dir(tmp_path / "old"), ROW)["commit"], "a record without one is unchanged"


TP2 = "olmoe-1b-7b__bf16__l40s__tp2__b8__i1024__o128__mixed__greedy__bi-eager"


@pytest.mark.parametrize("W", [2, 4, 8])
def test_the_tp_path_runs_a_config_run_at_worlds_2_4_and_8(W, tmp_path, monkeypatch):
    from verity_vllm.pipeline import cli, row
    from verity_vllm.pipeline.row_tp import TpRow
    ran = []
    monkeypatch.setattr(TpRow, "run", lambda self: ran.append((self.W, self.config_run)) or 0)
    o = cli.parse("row", ["run", TP2, "OLMOE", "r", "v", "--sweep-dir", str(tmp_path), "--config-run", "1", "--world", str(W)], {})
    assert row.run_row(o, o.stages) == 0 and ran == [(W, True)]


def _tp(tmp_path, *argv):
    from tests.pipeline.test_row import TP_DENSE, _tp_row
    import os
    t = _tp_row(tmp_path, TP_DENSE, *argv)
    os.makedirs(t.d, exist_ok=True)
    return t


def test_a_tp2_config_run_skips_the_match_and_word_checks_the_build(tmp_path, monkeypatch):
    t = _tp(tmp_path, "--config-run", "1")
    calls = []
    for name in ("build", "match", "commit", "research_result", "timeline_report"):
        monkeypatch.setattr(t, name, lambda n=name: calls.append(n))
    monkeypatch.setattr(t, "prepare", lambda: (calls.append("prepare"), setattr(t, "trap", True)))
    monkeypatch.setattr(RC, "word_check", lambda row: calls.append("word_check"))
    monkeypatch.setattr(RC, "config_record", lambda row: calls.append("config_record"))
    assert t.run() == 0
    assert calls == ["prepare", "build", "word_check", "commit", "research_result", "config_record", "timeline_report"]
    calls.clear()
    monkeypatch.setattr(t, "build", lambda: (_ for _ in ()).throw(Stop(10)))
    assert t.run() == 10 and calls == ["prepare", "config_record", "research_result"], "a stopped config run still writes its record"


def test_the_tp2_word_check_counts_the_components_of_both_ranks(tmp_path, monkeypatch):
    t = _tp(tmp_path, "--config-run", "1")
    seen = []
    monkeypatch.setattr(t, "run_tl", lambda stage, to, log, cmd: seen.append(cmd) or 0)
    RC.word_check(t)
    assert seen[0][seen[0].index("--world") + 1] == "2"
    r = _row(tmp_path / "tp1", "--config-run", "1")
    monkeypatch.setattr(r, "run_tl", lambda stage, to, log, cmd: seen.append(cmd) or 0)
    RC.word_check(r)
    assert seen[1][seen[1].index("--world") + 1] == "1"


def test_the_tp2_config_run_s_commit_is_one_instrumented_run_with_no_match(tmp_path):
    t = _tp(tmp_path, "--config-run", "1", "--replay-k", "460")
    (Path(t.M) / "fold_match").mkdir(parents=True)
    (Path(t.M) / "fold_match" / "tp2_fold_match.json").write_text("{}")
    t.fold_profile = "derived_LLAMA32_1B_tp2"
    cmd = t.commit_cmd([])
    assert cmd[cmd.index("--pairs") + 1] == "1" and cmd[cmd.index("--warmup") + 1] == "1"
    assert cmd[cmd.index("--only-arm") + 1] == "instrumented" and cmd[cmd.index("--replay-k") + 1] == "460"
    assert not {"--match-dir", "--require-fold-match", "--value-check"} & set(cmd)
    assert cmd.count("--build-rank") == 2 and cmd[cmd.index("--required-manifest") + 1] == f"{t.d}/manifest.json"
    row = _tp(tmp_path / "row")
    (Path(row.M) / "fold_match").mkdir(parents=True)
    (Path(row.M) / "fold_match" / "tp2_fold_match.json").write_text("{}")
    row.fold_profile = "derived_LLAMA32_1B_tp2"
    plain = row.commit_cmd([])
    i = plain.index("--required-manifest")
    assert plain[i + 2:i + 4] == ["--match-dir", row.M] and plain[plain.index("--pairs") + 1:plain.index("--pairs") + 6] == ["3", "--warmup", "2", "--value-check", "--replay-workers"]
    assert "--only-arm" not in plain and plain[-1] == "--require-fold-match", "a TP row of record keeps its Match, pairs and value check"


def test_tp_commit_s_own_parser_takes_every_flag_the_tp2_config_run_passes(tmp_path):
    from verity_vllm.pipeline import cli
    from verity_vllm.pipeline.tp.config_run import only_arm_summary, replay_k_per_rank
    t = _tp(tmp_path, "--config-run", "1", "--replay-k", "460")
    cmd = t.commit_cmd([])
    args = cli.parse("tp-commit", cmd[cmd.index("tp-commit") + 1:], {})
    assert (args.only_arm, args.replay_k, args.pairs, args.warmup, args.match_dir) == ("instrumented", 460, 1, 1, None)
    assert replay_k_per_rank(460, 2) == {"0": 230, "1": 230} and replay_k_per_rank(461, 2) == {"0": 231, "1": 230}
    assert replay_k_per_rank(None, 2) is None
    one = [{"arm": "instrumented"}]
    assert only_arm_summary("instrumented", one, one, [True]) == {"only_arm": "instrumented", "runs": 1, "commit_pass": True}
    assert only_arm_summary("instrumented", one, one, [False])["commit_pass"] is False
    assert only_arm_summary("instrumented", one, [], [])["commit_pass"] is False, "no instrumented run is not a pass"


@pytest.mark.parametrize("commit_pass,expect", [(True, None), (False, 12)])
def test_the_tp2_config_run_s_commit_decides_by_its_one_run(tmp_path, commit_pass, expect, monkeypatch):
    t = _tp(tmp_path, "--config-run", "1")
    t.to_k = 1
    (Path(t.d) / "commit").mkdir()
    (Path(t.d) / "commit" / "summary.json").write_text(json.dumps({"only_arm": "instrumented", "runs": 1, "commit_pass": commit_pass, "pass": commit_pass}))
    monkeypatch.setattr(R_, "tp_rank_workload_digest", lambda bdir, r: f"wd{r}")
    monkeypatch.setattr(R_, "tp_manifest_complete", lambda d: True)
    monkeypatch.setattr(R_, "applicable_taps", lambda man, ck: [])
    t.hidden_so = str(tmp_path / "verity_fa2_matReqG.so")      # never the host's pod-layout build (unreadable to other users on node 1)
    Path(t.hidden_so).touch()
    ran = []
    monkeypatch.setattr(t, "manifest_global", lambda *a: ran.append("manifest_global") or 0)
    monkeypatch.setattr(t, "run_tl", lambda stage, to, log, cmd: ran.append(stage) or 0)
    if expect is None:
        t.commit()
        assert "commit PASS" in (Path(t.d) / "stages.txt").read_text()
    else:
        with pytest.raises(Stop) as e:
            t.commit()
        assert e.value.rc == expect
    assert ran == ["commit.tp_commit"], "the config run's Commit reuses the Build's complete manifest"
    assert t.child_env["VERITY_STAGING_BOUNDED"] == "1", "the ranks stage bounded, as the single-GPU config run's --bounded-staging"


# ---------------------------------------------------------------------------------------------------------------- the config record

#: The served switches of a Commit whose every rank read back what its row declares (`engine.served.record`).
SERVED = {"schema": SV.SCHEMA, "ok": True, "problems": []}


def _row_dir(tmp_path: Path, *, mismatch: int = 0, commit_pass: bool = True, word_ok: bool = True) -> Path:
    d = tmp_path / ROW
    (d / "commit").mkdir(parents=True)
    (d / "build_summary.json").write_text(json.dumps({"step": {"ok": True, "program_digest": "aa"}, "request": {"ok": True, "program_digest": "bb"},
                                                      "workload": {"ok": True, "workload_digest": "ww"}}))
    (d / "manifest_header.json").write_text(json.dumps({"manifest_digest": "mm", "query": {"query_id": "Q_word_v1{X=16,W=32,R=no-recompute}"}}))
    (d / "strict_word.json").write_text(json.dumps({"ok": word_ok, "check": "fast", "word_lines": 1, "components": 1, "problems": []}))
    sr = {"result": mismatch == 0, "grade": "complete" if mismatch == 0 else "mismatch",
          "sample": {"strata_by": "family", "k": 256, "picked": 256, "evaluated": 256, "equal": 256 - mismatch, "mismatch_n": mismatch,
                     "not_evaluated_n": 0, "seed": 7, "seed_source": "run root"},
          "mismatch": [{"i": 3, "family": "Gemm_v1", "request_id": "r0", "step": 2, "op_path": "model.layers.0.mlp"}] * mismatch,
          "mismatch_by": {"by_family": {"Gemm_v1": mismatch} if mismatch else {}}}
    run = {"arm": "instrumented", "wall_s": 12.5, "engine_steps": 32,
           "commit": {"commit_pass": commit_pass, "commit_fail_reason": None if commit_pass else "C2 mismatch", "run_root": "rr",
                      "openings": {"n": 64, "verified": 64}, "expected_tensors": 9, "committed_tensors": 9, "sampled_replay": sr}}
    (d / "commit" / "runs.jsonl").write_text(json.dumps(run) + "\n")
    (d / "commit" / "summary.json").write_text(json.dumps({"versions": {"served": SERVED}}))
    return d


def test_a_clean_config_run_s_record_is_pass_and_small(tmp_path):
    rec = CR.write(_row_dir(tmp_path), ROW, tree="abc")
    assert rec["outcome"] == "PASS" and rec["why"] is None
    assert rec["config"] == {"row": ROW, "model": "llama32-1b", "dtype": "bf16", "gpu": "l40s", "tp": 1, "batch": 1, "input": 256, "output": 32,
                             "arrivals": "mixed", "sampling": "greedy", "engine": "bi-eager"}
    assert rec["build"]["programs"] == {"request": "bb", "step": "aa"} and rec["build"]["manifest_digest"] == "mm"
    assert (rec["commit"]["run_root"], rec["replay"]["k"], rec["replay"]["equal"]) == ("rr", 256, 256)
    assert (tmp_path / ROW / CR.NAME).stat().st_size < 8192


@pytest.mark.parametrize("kw,why", [({"mismatch": 2, "commit_pass": False}, "commit FAIL: C2 mismatch"), ({"word_ok": False}, "word check FAIL")])
def test_a_config_run_that_failed_says_why_first(tmp_path, kw, why):
    rec = CR.record(_row_dir(tmp_path, **kw), ROW)
    assert rec["outcome"] == "FAIL" and rec["why"] == why
    if kw.get("mismatch"):
        assert rec["replay"]["differed"][0]["family"] == "Gemm_v1" and rec["replay"]["by_family"] == {"Gemm_v1": 2}


@pytest.mark.parametrize("gm, outcome, why", [("PASS", "PASS", None), ("FAIL", "PASS", "match FAIL"), ("PASS", "FAIL", "commit FAIL")])
def test_a_single_gpu_row_of_record_is_its_global_matchs_verdict_and_its_commits_decision_document(tmp_path, gm, outcome, why):
    """GM-01 writes ``verdict`` (`check.match.global_match`), never the ``ok`` the record read, and a single-GPU Commit's summary
    decides nothing (its decision document is the row dir's verdict.json): a row chain is never refused as "match FAIL" or
    "commit FAIL" over a passing Match and Commit."""
    d = _row_dir(tmp_path)
    (d / "global_match.json").write_text(json.dumps({"schema": "cb-a/global-match/v1", "verdict": gm, "failed_checks": []}))
    (d / "verdict.json").write_text(json.dumps({"schema": "vllm-verdict/v1", "outcome": outcome, "manifest_digest": "mm"}))
    rec = CR.record(d, ROW, summary={"versions": {"served": SERVED}})
    assert (rec["outcome"], rec["why"]) == ("PASS" if why is None else "FAIL", why)
    assert rec["commit"]["manifest"] == {"manifest_digest": "mm"} and rec["commit"]["run_root"] == "rr"


def test_a_config_run_with_no_commit_is_not_pass(tmp_path):
    d = _row_dir(tmp_path)
    (d / "commit" / "runs.jsonl").unlink()
    assert CR.record(d, ROW)["why"] == "commit not run"


def test_a_commit_that_read_no_switches_back_is_not_pass(tmp_path):
    d = _row_dir(tmp_path)
    (d / "commit" / "summary.json").unlink()
    rec = CR.record(d, ROW)
    assert (rec["outcome"], rec["why"], rec["served"]) == ("FAIL", "served switches not run", None)


def test_a_commit_its_switches_failed_closed_says_so_first(tmp_path):
    d = _row_dir(tmp_path)
    (d / "commit" / "runs.jsonl").unlink()
    bad = {**SERVED, "ok": False, "problems": ["rank 0: VLLM_BATCH_INVARIANT=True where row file declares False"]}
    (d / "commit" / "summary.json").write_text(json.dumps({"schema": SV.SCHEMA, "pass": False, "served": bad}))
    rec = CR.record(d, ROW)
    assert rec["why"] == "served switches FAIL: rank 0: VLLM_BATCH_INVARIANT=True where row file declares False"
    assert rec["served"] == bad


ROW_TP2 = "llama32-1b__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager"


def _tp_sr(rank: int, k: int, mismatch: int = 0) -> dict:
    return {"result": mismatch == 0, "grade": "complete" if mismatch == 0 else "mismatch",
            "sample": {"strata_by": "family", "k": k, "picked": k, "evaluated": k, "equal": k - mismatch, "mismatch_n": mismatch, "not_evaluated_n": 0,
                       "seed": 10 + rank, "seed_source": "this rank's run root"},
            "mismatch": [{"i": 5, "family": "Gemm_v1", "request_id": "r0", "step": 1, "op_path": "model.layers.0.mlp.down_proj"}] * mismatch,
            "mismatch_by": {"by_family": {"Gemm_v1": mismatch} if mismatch else {}}}


def _tp_row_dir(tmp_path: Path, *, mismatch_r1: int = 0, ranks: int = 2) -> Path:
    d = tmp_path / ROW_TP2
    (d / "commit").mkdir(parents=True)
    (d / "build_summary.json").write_text(json.dumps({"tp": {"world": 2}, "step": {"program_digest": "s0"}, "ranks": {
        f"rank{r}": {"build_step": {"ok": True, "program_digest": f"s{r}"}, "build_request_LP256_T31": {"ok": True, "program_digest": f"q{r}"},
                     "workload": {"ok": True, "workload_digest": f"w{r}"}} for r in range(2)}}))
    (d / "manifest.json").write_text(json.dumps({"manifest_digest": "mm", "ranks": 2, "program_digest_per_rank": {"0": "w0", "1": "w1"},
                                                 "query": {"query_id": "Q_word_v1{X=16,W=32,R=no-recompute}"}}))
    (d / "strict_word.json").write_text(json.dumps({"ok": True, "check": "fast", "word_lines": 2, "components": 2, "problems": []}))
    fail = mismatch_r1 > 0
    per = {str(r): {"run_root": f"root{r}", "commit_pass": not (fail and r == 1), "commit_fail_reason": "sampled replay mismatch" if fail and r == 1 else None,
                    "openings": {"n": 64, "verified": 64}, "extra": {"expected_tensors": 9, "committed_tensors": 9},
                    "binding": {"coverage": {"ok": True, "expected_n": 9, "bound_n": 9, "missing_n": 0}}, "sampled_replay": _tp_sr(r, 230, mismatch_r1 if r == 1 else 0)} for r in range(ranks)}
    run = {"arm": "instrumented", "wall_s": 20.0, "engine_steps": 32,
           "commit": {"tp_run_root": "tproot", "world": 2, "per_rank": per, "openings": {"n": 128, "verified": 128}, "tensors": 18,
                      "commit_pass": not fail, "commit_fail_reason": "rank 1: sampled replay mismatch" if fail else "PASS"}}
    (d / "commit" / "runs.jsonl").write_text(json.dumps(run) + "\n")
    (d / "commit" / "summary.json").write_text(json.dumps({"served": SERVED}))
    return d


def test_a_clean_tp2_config_run_s_record_is_per_rank_and_in_total(tmp_path):
    rec = CR.write(_tp_row_dir(tmp_path), ROW_TP2, tree="abc")
    assert rec["outcome"] == "PASS" and rec["why"] is None and rec["config"]["tp"] == 2
    b = rec["build"]
    assert b["per_rank"]["1"] == {"ok": True, "programs": {"build_request_LP256_T31": "q1", "build_step": "s1"}, "census_digests": {},
                                  "workload_digest": "w1"}
    assert (b["manifest_digest"], b["program_digest_per_rank"]) == ("mm", {"0": "w0", "1": "w1"})
    c = rec["commit"]
    assert (c["run_root"], c["reason"], c["openings"]) == ("tproot", None, {"n": 128, "verified": 128})
    assert [c["per_rank"][r]["run_root"] for r in ("0", "1")] == ["root0", "root1"]
    rp = rec["replay"]
    assert (rp["k"], rp["evaluated"], rp["equal"], rp["mismatch_n"]) == (460, 460, 460, 0), "the replay's k units over both ranks"
    assert [rp["per_rank"][r]["seed"] for r in ("0", "1")] == [10, 11]
    assert (tmp_path / ROW_TP2 / CR.NAME).stat().st_size < 16384


def test_a_tp2_config_run_that_differed_on_one_rank_names_the_rank(tmp_path):
    rec = CR.record(_tp_row_dir(tmp_path, mismatch_r1=2), ROW_TP2)
    assert rec["outcome"] == "FAIL" and rec["why"] == "commit FAIL: rank 1: sampled replay mismatch"
    rp = rec["replay"]
    assert (rp["ok"], rp["equal"], rp["mismatch_n"], rp["by_family"]) == (False, 458, 2, {"Gemm_v1": 2})
    assert rp["differed"][0]["rank"] == 1 and rec["commit"]["per_rank"]["0"]["ok"] is True


def test_a_tp2_config_run_missing_a_rank_is_not_pass(tmp_path):
    d = _tp_row_dir(tmp_path, ranks=1)
    rec = CR.record(d, ROW_TP2)
    assert rec["replay"]["ok"] is False and rec["outcome"] == "FAIL"


MANIFEST_OF_RECORD = "art:" + "6" * 64


@pytest.mark.parametrize("line,programs_ok,manifest,ok,superseded", [
    ("build FAIL 2026-10-15T19:12:49Z rc=1 manifest_rc=137", True, MANIFEST_OF_RECORD, True, True),
    ("build FAIL 2026-10-15T19:12:49Z rc=1 manifest_rc=137", True, None, False, False),
    ("build FAIL 2026-10-15T19:12:49Z rc=1", True, MANIFEST_OF_RECORD, False, False),
    ("build FAIL 2026-10-15T19:12:49Z rc=1 manifest_rc=137", False, MANIFEST_OF_RECORD, False, False),
    ("build PASS 2026-10-15T19:12:49Z rc=0 manifest_rc=0", True, None, True, False),
])
def test_a_build_whose_in_stage_manifest_alone_failed_passes_only_superseded_by_the_manifest_of_record(
        tmp_path, line, programs_ok, manifest, ok, superseded):
    d = _tp_row_dir(tmp_path)
    (d / "stages.txt").write_text(f"precheck PASS 2026-10-15T18:00:00Z rc=0\n{line}\ncommit PASS 2026-10-16T23:26:33Z rc=0\n")
    if not programs_ok:
        bs = json.loads((d / "build_summary.json").read_text())
        bs["ranks"]["rank1"]["build_step"]["ok"] = False
        (d / "build_summary.json").write_text(json.dumps(bs))
    rec = CR.record(d, ROW_TP2, evidence={"manifest": manifest})
    st = rec["build"]["stage"]
    assert rec["build"]["ok"] is ok and rec["outcome"] == ("PASS" if ok else "FAIL")
    assert st["status"] == line.split()[1] and st.get("superseded_by") == (manifest if superseded else None)
    assert {s: v["status"] for s, v in rec["stages"].items()} == {"precheck": "PASS", "build": line.split()[1], "commit": "PASS"}
    if not ok:
        stranded = programs_ok and "manifest_rc=137" in line
        assert rec["why"] == ("build FAIL: the in-stage manifest exited 137 and no manifest of record supersedes it (evidence.manifest)"
                              if stranded else "build FAIL")


def test_a_row_dir_without_a_build_verdict_keeps_the_programs_verdict(tmp_path):
    rec = CR.record(_tp_row_dir(tmp_path), ROW_TP2)
    assert rec["build"]["stage"] is None and rec["stages"] == {} and rec["outcome"] == "PASS"


def test_the_record_s_slowdown_is_instrumented_over_control_per_phase(tmp_path):
    d = _row_dir(tmp_path)
    assert CR.record(d, ROW)["timing"]["slowdown"] is None, "no control arm: no baseline, no slowdown"
    inst = json.loads((d / "commit" / "runs.jsonl").read_text())
    inst.update(step_wall_prefill_s=0.30, step_wall_decode_mean_s=0.050)
    ctrl = {"arm": "control", "pair": 0, "wall_s": 2.5, "engine_steps": 32, "step_wall_prefill_s": 0.02, "step_wall_decode_mean_s": 0.010}
    (d / "commit" / "runs.jsonl").write_text(json.dumps(ctrl) + "\n" + json.dumps(inst) + "\n")
    rec = CR.record(d, ROW)
    assert rec["outcome"] == "PASS" and rec["commit"]["runs"] == 1, "the control arm is timing only; the decision is the instrumented run's"
    sd = rec["timing"]["slowdown"]
    assert sd["prefill"] == pytest.approx(15.0) and sd["decode"] == pytest.approx(5.0) and sd["wall"] == pytest.approx(5.0)
    assert rec["timing"]["control"]["decode_mean_s"] == 0.010


def test_the_record_carries_the_engine_s_cublas_workspace_pin(tmp_path):
    d = _row_dir(tmp_path)
    assert CR.record(d, ROW)["engine"] is None, "no engine facts recorded: none claimed"
    run = json.loads((d / "commit" / "runs.jsonl").read_text())
    run["commit"]["sampled_replay"]["engine_of_record"] = {"VLLM_BATCH_INVARIANT": "1", "CUBLAS_WORKSPACE_CONFIG": ":16:8",
                                                           "CUBLASLT_WORKSPACE_SIZE": "1", "rule": "measured in the engine's process"}
    (d / "commit" / "runs.jsonl").write_text(json.dumps(run) + "\n")
    eng = CR.record(d, ROW)["engine"]
    assert eng["CUBLAS_WORKSPACE_CONFIG"] == ":16:8" and eng["CUBLASLT_WORKSPACE_SIZE"] == "1" and "rule" not in eng


def test_a_config_baseline_keeps_the_control_arm(tmp_path):
    r = _row(tmp_path, "--config-run", "1", "--config-baseline", "1")
    r.hidden_so, r.B = "/x/verity_fa2_matReqG.so", 1
    cmd = r.commit_cmd([])
    assert "--only-arm" not in cmd and cmd[cmd.index("--pairs") + 1] == "1" and "--bounded-staging" in cmd


# ---------------------------------------------------------------------------------------------------------------- the Program cache

def _workload(tmp_path: Path, name: str, *, batch: int = 1, prompt: list[int] | None = None) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"sweep": {"sampling": "greedy", "sampling_params": {"temperature": 0.0}, "batch": batch, "sampling_seeds": []},
                             "requests": [{"prompt_token_ids": prompt or [1, 2, 3]}] * batch}))
    return p


def _derive(out: str, wl: Path, lp: int = 256) -> list[str]:
    return ["/venv/bin/python", "-m", "verity_vllm.pipeline.cli", "build", "--model", "m", "--revision", "r", "--wrapper", "request",
            "--LP", str(lp), "--T", "31", "--workload", str(wl), "--out", out]


DEV = {"name": "NVIDIA L40S", "compute_cap": "8.9"}


def test_the_cache_key_is_the_derive_without_its_out_dir_and_workload_path(tmp_path):
    a, b = _workload(tmp_path, "a"), _workload(tmp_path, "b", prompt=[9, 9, 9])
    k = PC.key(_derive("/x/a/build_request", a), workload=a, tree="t1", device=DEV)
    assert k == PC.key(_derive("/y/b/build_request", b), workload=b, tree="t1", device=DEV), "the out dir and the prompts a derive never reads"
    assert k != PC.key(_derive("/x/a/build_request", a, lp=512), workload=a, tree="t1", device=DEV), "another shape"
    assert k != PC.key(_derive("/x/a/build_request", a), workload=a, tree="t2", device=DEV), "another tree"
    assert k != PC.key(_derive("/x/a/build_request", a), workload=a, tree="t1", device={**DEV, "name": "NVIDIA H100"}), "another GPU"
    b8 = _workload(tmp_path, "b8", batch=8)
    assert k != PC.key(_derive("/x/a/build_request", b8), workload=b8, tree="t1", device=DEV), "the sampler geometry's batch"
    assert PC.key(_derive("/x", a), workload=a, tree=None, device=DEV) is None, "no tree commit, no cache"


def test_the_derive_key_s_target_is_the_declared_one_and_a_cpu_host_needs_no_nvidia_smi(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def no_nvidia_smi(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(RC.subprocess, "run", no_nvidia_smi)
    target = '{"compute_capability": [12, 0], "num_sms": 188}'
    assert RC.device_id(SimpleNamespace(tgt=target)) == {"declared_target": target}
    assert RC.device_id(SimpleNamespace(tgt="")) is None, "no target and no GPU: no key, the plain derive"


def test_a_published_derive_is_restored_whole_and_the_first_publisher_stands(tmp_path):
    derived = tmp_path / "row" / "build_request"
    derived.mkdir(parents=True)
    (derived / "result.json").write_text(json.dumps({"ok": True}))
    (derived / "instances.json.gz").write_bytes(b"program")
    root = tmp_path / "cache"
    entry = PC.publish(root, "k1", derived, {"row": "r"})
    assert entry == root / "k1" and PC.lookup(root, "k1") == entry
    (derived / "instances.json.gz").write_bytes(b"other")
    assert PC.publish(root, "k1", derived, {"row": "r2"}) == entry and (entry / "program" / "instances.json.gz").read_bytes() == b"program"
    out = tmp_path / "row2" / "build_request"
    PC.restore(entry, out)
    assert (out / "instances.json.gz").read_bytes() == b"program" and not (out / "instances.json.gz").samefile(entry / "program" / "instances.json.gz")
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "result.json").write_text(json.dumps({"ok": False}))
    assert PC.publish(root, "k2", failed, {}) is None and PC.lookup(root, "k2") is None


# ---------------------------------------------------------------------------------------------------------------- the replay sample

def test_family_strata_draw_k_units_with_every_family_at_least_once():
    vus = [{"i": i, "family": fam, "stratum": f"{fam}|r{i % 3}|rank0|mid|op{i}|fresh-window"}
           for fam, n in (("Gemm_v1", 900), ("RoPE_v1", 400), ("Embedding_v1", 3)) for i in range(n)]
    keyed, per = family_strata(vus, 256)
    assert per == 256 and {v["stratum"] for v in keyed} == {"Gemm_v1|all|all|all|all|all", "RoPE_v1|all|all|all|all|all",
                                                           "Embedding_v1|all|all|all|all|all"}
    picked, strata, tgt = sample(keyed, per, seed=11)
    kept = picked[:min(max(256, len(strata)), tgt["stratified"])]
    fams = [v["family"] for v in kept]
    assert len(kept) == 256 and fams.count("Embedding_v1") == 3, "the small family is used up, the others fill k"
    assert abs(fams.count("Gemm_v1") - fams.count("RoPE_v1")) <= 1
    assert family_strata(vus, None, 2) == (vus, 2), "without k: the row's per-stratum sample, untouched"


def test_the_uniform_draw_is_k_units_from_one_stratum_of_every_unit():
    vus = [{"i": i, "family": fam, "stratum": f"{fam}|r0|rank0|mid|op{i}|fresh-window"}
           for fam, n in (("Gemm_v1", 900), ("RoPE_v1", 400), ("Embedding_v1", 3)) for i in range(n)]
    keyed, per = family_strata(vus, 460, draw="uniform")
    assert per == 460 and {v["stratum"] for v in keyed} == {"all|all|all|all|all|all"}
    assert {v["family"] for v in keyed} == {"Gemm_v1", "RoPE_v1", "Embedding_v1"}, "each unit keeps its family for the record"
    picked, strata, tgt = sample(keyed, per, seed=11)
    assert len(picked) == tgt["stratified"] == 460 and len({(v["family"], v["i"]) for v in picked}) == 460, "without replacement"
    assert strata == {"all|all|all|all|all|all": {"population": 1303, "sampled": 460}}
    assert picked != sample(keyed, per, seed=12)[0]


def test_a_record_that_names_no_draw_keeps_the_family_draw_and_an_unknown_draw_is_refused():
    from types import SimpleNamespace

    from verity_vllm.pipeline.c2_replay import draw_of

    assert [draw_of(a) for a in (SimpleNamespace(), SimpleNamespace(replay_draw=None), SimpleNamespace(replay_draw="uniform"))] == [
        "family", "family", "uniform"]
    vus = [{"i": i, "family": fam, "stratum": f"{fam}|r0|rank0|mid|op{i}|fresh-window"} for fam in ("Gemm_v1", "RoPE_v1") for i in range(3)]
    assert family_strata(vus, 4, draw=draw_of(SimpleNamespace())) == family_strata(vus, 4)
    with pytest.raises(ValueError, match=r"replay draw 'stratified' is not one of \('family', 'uniform'\)"):
        family_strata(vus, 4, draw="stratified")
