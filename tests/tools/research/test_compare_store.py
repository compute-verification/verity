"""`research compare` over the evidence store: a replay's verdict against its originals, read by run id from a store whose run dirs
are gone (the remote is an FsRemote vault), and each rule of `compare.verdict`."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from research import compare as C
from research.cli import main
from research.store import attempt as A
from research.store.local import LocalStore

RUN_FIXTURE = Path(__file__).parent / "fixtures" / "run_v0.2"
O1, O2, O3, REPLAY = "r20261004-100000-0001", "r20261004-100000-0002", "r20261004-100000-0003", "r20261004-110000-00aa"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "pod-store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    return tmp_path


def _publish(env: Path, run_id: str, *, prove_wall: float = 1.5, proof: bytes = b"proof", build: bytes = b"build", state: str = "done",
             rc: int = 0, validation: str = "passed", detail: str = "5/5 verified", why: str = "rc 0", derivation: str | None = None,
             run_record: bool = True) -> None:
    """A run dir from the fixture under `run_id`, published (and pushed to the vault) from the pod store, then deleted."""
    d = env / "runs" / run_id
    shutil.copytree(RUN_FIXTURE, d)
    (d / "state.json").rename(d / "status.json")
    job = json.loads((d / "job.json").read_text())
    job["run"]["id"] = job["ownership"]["job"] = run_id
    if derivation:
        job["store"]["derivation"] = derivation
    (d / "job.json").write_text(json.dumps(job))
    status = json.loads((d / "status.json").read_text())
    (d / "status.json").write_text(json.dumps({**status, "state": state, "rc": rc}))
    (d / "failure.json").write_text(json.dumps({**json.loads((d / "failure.json").read_text()), "why": why}))
    res = json.loads((d / "result.json").read_text())
    res.update(run_id=run_id, measurements=[{"name": "prove_wall", "value": prove_wall, "unit": "s"}, {"name": "proof_bytes", "value": 4096, "unit": "B"}])
    res["validation"].update(status=validation, detail=detail)
    (d / "result.json").write_text(json.dumps(res))
    (d / "out" / "proof.bin").write_bytes(proof)
    (d / "stdout.log").write_text(f"{run_id} started at its own time\n")                 # the runner's own log: never compared
    (d / "out" / "build.bin").write_bytes(build)
    (d / "outputs.json").write_text(json.dumps({"schema": "research/outputs/v0.1", "outputs": [{"name": "build", "kind": "vllm-build/v1", "file": "out/build.bin"}]}))
    out = A.publish(d, store=LocalStore(), run_record=run_record)
    assert out["published"] and out.get("preserved"), out
    shutil.rmtree(d)


def _here(env: Path) -> LocalStore:
    """A store on another machine: empty, with the same vault."""
    return LocalStore(env / "here-store")


def _verdict(env: Path, originals: list[str], replay: str, expect: str | None = None) -> C.Verdict:
    st = _here(env)
    return C.verdict([C.load(st, r) for r in originals], C.load(st, replay), expect)


def test_a_replay_of_identical_bytes_is_same_read_from_the_remote_with_the_run_dirs_gone(env, capsys):
    for r in (O1, O2, REPLAY):
        _publish(env, r)
    assert not (env / "runs" / O1).exists() and not (env / "here-store" / "attempts").exists()
    v = _verdict(env, [O1, O2], REPLAY)
    assert v.verdict == C.SAME, v.reasons
    assert v.reasons == ["equal: 1 derivation, 3 status fields, 1 output, 4 run files, 2 metrics"], "the build output net of its run id"
    assert v.band["prove_wall"] == {"n": 2, "lo": 1.5, "hi": 1.5, "value": 1.5, "unit": "s"}
    rc = main(["compare", O1, O2, REPLAY, "--json", "--store", str(env / "here-store")])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0 and doc["verdict"] == "same" and doc["originals"] == [O1, O2] and doc["replay"] == REPLAY and doc["expect"] is None


def test_metrics_inside_the_originals_band_are_within_band_and_outside_it_differ(env, capsys):
    _publish(env, O1, prove_wall=1.5)
    _publish(env, O2, prove_wall=1.7)
    _publish(env, REPLAY, prove_wall=1.8)
    _publish(env, O3, prove_wall=2.0)
    v = _verdict(env, [O1, O2], REPLAY)
    assert v.verdict == C.WITHIN_BAND and v.reasons[0].startswith("metric prove_wall: 1.8 s inside the band [1.3, 1.9] of 2 originals")
    v = _verdict(env, [O1, O2], O3)
    assert v.verdict == C.DIFFERS and v.reasons == ["metric prove_wall: 2.0 s outside the band [1.3, 1.9] of 2 originals", v.reasons[1]]
    assert main(["compare", O1, O2, O3, "--store", str(env / "here-store")]) == 1
    assert capsys.readouterr().out.startswith(f"differs  replay {O3} against {O1}, {O2}\n  - metric prove_wall: 2.0 s outside")


def test_with_one_original_a_moved_metric_differs_and_says_the_band_is_unknown(env):
    _publish(env, O1, prove_wall=1.5)
    _publish(env, REPLAY, prove_wall=1.6)
    v = _verdict(env, [O1], REPLAY)
    assert v.verdict == C.DIFFERS and v.reasons[0] == "metric prove_wall: 1.5 -> 1.6 s; band unknown: one original (a band needs at least two)"
    assert v.band["prove_wall"]["lo"] is None


def test_a_changed_file_or_output_differs_by_name_and_runner_logs_never_count(env):
    _publish(env, O1)
    _publish(env, O2)
    _publish(env, REPLAY, proof=b"other proof", build=b"other build")
    v = _verdict(env, [O1, O2], REPLAY)
    assert v.verdict == C.DIFFERS
    assert any(r.startswith("output build: content ") for r in v.reasons)
    files = next(r for r in v.reasons if r.startswith("run files:"))
    assert files.startswith("run files: 2 differ: out/build.bin (") and "out/proof.bin (" in files
    assert "stdout.log" not in " ".join(v.reasons) and "job.json" not in " ".join(v.reasons)


def test_what_varies_across_the_originals_is_within_band_not_checked_bit_for_bit(env):
    _publish(env, O1, proof=b"a")
    _publish(env, O2, proof=b"b")
    _publish(env, REPLAY, proof=b"c")
    v = _verdict(env, [O1, O2], REPLAY)
    assert v.verdict == C.WITHIN_BAND
    assert "run files: 1 vary across the originals, not checked bit for bit: out/proof.bin" in v.reasons


def test_a_different_derivation_or_status_differs(env):
    _publish(env, O1)
    _publish(env, REPLAY, derivation="drv:" + "9" * 64, state="failed", rc=1)
    v = _verdict(env, [O1], REPLAY)
    assert v.verdict == C.DIFFERS
    assert f"derivation: {'drv:' + '1' * 64!r} -> {'drv:' + '9' * 64!r}" in v.reasons
    assert "status state: 'done' -> 'failed'" in v.reasons and "status rc: 0 -> 1" in v.reasons


def test_a_matching_refusal_is_the_expected_change_and_anything_else_differs(env, capsys):
    _publish(env, O1)
    _publish(env, REPLAY, validation="failed", detail="refused: out_net != unit net (tail form)")
    _publish(env, O2)
    v = _verdict(env, [O1], REPLAY, expect="refused: tail form")
    assert v.verdict == C.EXPECTED_CHANGE and v.reasons[0].startswith("expected refused: tail form: the replay was refused (")
    v = _verdict(env, [O1], REPLAY, expect="blocked: V3")
    assert v.verdict == C.DIFFERS and v.reasons[0].startswith("expected blocked: V3: the replay was refused, but not naming 'V3'")
    v = _verdict(env, [O1], O2, expect="refused: tail form")
    assert v.verdict == C.DIFFERS and v.reasons == ["expected refused: tail form: the replay was not refused", v.reasons[1]]
    with pytest.raises(ValueError, match="neither"):
        _verdict(env, [O1], REPLAY, expect="blocked")

    here = _here(env)
    C.load(here, O1)
    here.add_label(O1, C.EXPECT_KEY, "refused: tail form", by="lane-x", ref="note:x/y")
    assert main(["compare", O1, REPLAY, "--json", "--store", str(env / "here-store")]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "expected-change", "the CLI reads the original's replay_expect label"
    assert main(["compare", O1, REPLAY, "--no-expect", "--json", "--store", str(env / "here-store")]) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "differs"
    assert main(["compare", O1, REPLAY, "--expect", "blocked", "--store", str(env / "here-store")]) == 2


def test_without_a_run_record_only_the_results_listed_files_are_compared(env):
    _publish(env, O1, run_record=False)
    _publish(env, REPLAY, build=b"other build")
    v = _verdict(env, [O1], REPLAY)
    assert C.load(_here(env), O1).files_scope == "run_files"
    assert f"run files: only the result's listed files are compared ({O1} has no run record)" in v.reasons
    assert not any(r.startswith("run files: 1 differ") for r in v.reasons), "out/build.bin is not a listed file"
    assert v.verdict == C.DIFFERS and any(r.startswith("output build:") for r in v.reasons)


def test_an_attempt_nowhere_falls_back_to_the_run_dirs_or_fails(env, capsys):
    assert main(["compare", O1, O2, REPLAY, "--store", str(env / "here-store")]) == 1
    assert f"{O1}: no attempt in the store" in capsys.readouterr().err
    assert main(["compare", O1, O2, "--store", str(env / "here-store"), "--runs-dir", str(env / "runs")]) == 1
    err = capsys.readouterr().err
    assert "comparing the run dirs instead" in err
