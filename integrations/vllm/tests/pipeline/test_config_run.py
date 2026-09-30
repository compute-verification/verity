"""The config run (docs/vllm-config-sweep-plan.md §2-§4): the row mode (`--config-run 1`), the config record, the derived-Program cache,
the family-stratified k-unit replay sample and the sweep driver's packing.  CPU only: row dirs, workloads and processes are stand-ins."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.pipeline.test_row import _row
from verity_vllm.check.replay.driver import family_strata
from verity_vllm.check.replay.sample import sample
from verity_vllm.pipeline import config_record as CR
from verity_vllm.pipeline import program_cache as PC
from verity_vllm.pipeline import row_config as RC
from verity_vllm.pipeline import sweep as SW
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
    assert (args.only_arm, args.bounded_staging, args.replay_k, args.pairs, args.warmup) == ("instrumented", True, 128, 1, 1)


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


def test_the_tp_path_refuses_a_config_run_by_name(tmp_path):
    from verity_vllm.config import UsageError
    from verity_vllm.pipeline import cli, row
    o = cli.parse("row", ["run", "olmoe-1b-7b__bf16__l40s__tp2__b8__i1024__o128__mixed__greedy__bi-eager", "OLMOE", "r", "v",
                          "--sweep-dir", str(tmp_path), "--config-run", "1"], {})
    with pytest.raises(UsageError, match="tensor-parallel row"):
        row.run_row(o, o.stages)


# ---------------------------------------------------------------------------------------------------------------- the config record

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


def test_a_config_run_with_no_commit_is_not_pass(tmp_path):
    d = _row_dir(tmp_path)
    (d / "commit" / "runs.jsonl").unlink()
    assert CR.record(d, ROW)["why"] == "commit not run"


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


# ---------------------------------------------------------------------------------------------------------------- the sweep driver

def test_estimates_follow_the_plan_s_classes():
    assert SW.estimate(ROW) == SW.Need(gpus=1, ram_gb=150, cpus=16, minutes=60)
    assert SW.estimate("llama32-1b__bf16__l40s__tp1__b64__i1024__o128__mixed__greedy__bi-eager").ram_gb == 500
    assert SW.estimate("qwen25-15b__bf16__l40s__tp1__b1__i4096__o512__mixed__greedy__bi-eager").ram_gb == 500
    assert SW.estimate("olmoe-1b-7b__bf16__l40s__tp2__b8__i1024__o128__mixed__greedy__bi-eager").gpus == 2, "the TP world's GPUs"


def _cfg(row: str, ram: int, cpus: int = 16, minutes: int = 60, gpus: int = 1) -> SW.Config:
    return SW.Config(row=row, need=SW.Need(gpus=gpus, ram_gb=ram, cpus=cpus, minutes=minutes))


def test_the_plan_starts_the_large_builds_first_and_packs_the_small_ones_around_them():
    host = SW.Host(gpus=[0, 1, 2, 3], ram_gb=1000, cpus=list(range(64)))
    cfgs = [_cfg("s1", 150), _cfg("s2", 150), _cfg("big1", 500, 32, 150), _cfg("big2", 500, 32, 150), _cfg("s3", 150), _cfg("huge", 2000)]
    p = SW.plan(cfgs, host)
    assert p["refused"] == [{"row": "huge", "why": "needs 2000 GB of RAM, the host has 1000"}]
    by = {c["row"]: c for c in p["configs"]}
    assert by["big1"]["start_min"] == 0 and by["big2"]["start_min"] == 0, "the large Builds start first"
    assert all(by[s]["start_min"] == 150 for s in ("s1", "s2", "s3")), "RAM, not GPUs, holds the small ones until a Build ends"
    assert p["makespan_min"] == 210


def test_the_driver_frees_each_share_as_its_run_ends_and_records_it(tmp_path):
    host = SW.Host(gpus=[0, 1], ram_gb=400, cpus=list(range(32)))
    cfgs = [_cfg("a", 300, 16), _cfg("b", 100, 16), _cfg("c", 100, 16)]
    started, order = [], iter(["a", "b", "c"])
    clock = iter(float(t) for t in range(100))

    def start(c, gpus, cpus):
        started.append((c.row, gpus, (cpus[0], cpus[-1])))
        (tmp_path / "rows" / c.row).mkdir(parents=True, exist_ok=True)
        (tmp_path / "rows" / c.row / "config_record.json").write_text(json.dumps({"outcome": "PASS" if c.row != "c" else "FAIL", "why": None}))
        return c.row

    doc = SW.run(cfgs, host, tmp_path, start=start, wait=lambda: (next(order), 0), now=lambda: next(clock))
    assert started == [("a", [0], (0, 15)), ("b", [1], (16, 31)), ("c", [0], (0, 15))]
    assert [(d["row"], d["outcome"]) for d in doc["done"]] == [("a", "PASS"), ("b", "PASS"), ("c", "FAIL")]
    assert json.loads((tmp_path / "sweep.json").read_text())["done"] == doc["done"] and doc["pending"] == [] and doc["running"] == []


def test_each_run_gets_its_share_as_the_build_budget_and_the_host_s_caches(tmp_path):
    e = SW.child_env({"PATH": "/bin"}, _cfg("a", 300, 16), [1], list(range(16, 32)), tmp_path, "/cache/rules")
    assert (e["CUDA_VISIBLE_DEVICES"], e["OMP_NUM_THREADS"], e["BUILD_RAM_BUDGET_GB"], e["VERITY_UNIT_RULE_CACHE"]) == ("1", "16", "300", "/cache/rules")
    assert e["SWEEP_DIR"] == str(tmp_path / "rows") and e["PATH"] == "/bin"
    assert "VERITY_UNIT_RULE_CACHE" not in SW.child_env({}, _cfg("a", 300), [0], [0], tmp_path, None)
