"""A tensor-parallel row before its first stage, and the order its stages run in (`verity_vllm.pipeline.row_tp`: `TpRow.prepare`,
`TpRow.run`), on CPU.

`prepare` runs for real over the row's own file: the stages' environment, the row's directories, its shape and the line it starts
with.  What it would start a process for, or read the host for, is a stand-in on the row object (`_Host`): the release line, the tree's
export commit, the source identity, the target family, the signal handlers and the checkpoint dir of the host.  The driver's clock is
injected, so each line a row writes is compared whole.

`run` is held to the order of its stages with each stage a stand-in."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.pipeline import cli, row_driver
from verity_vllm.pipeline import row_config as RC
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.row_tp import TpRow

DENSE = "llama32-1b__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager"
MOE = "qwen3-30b-a3b__bf16__l40s__tp2__b2__i1024__o128__mixed__greedy__bi-eager"
GLM53 = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
SM120 = "llama32-1b__bf16__rtxpro6000__tp2__b1__i256__o32__mixed__greedy__bi-eager"
ROLE, REPO, REV = "LLAMA32_1B", "unsloth/Llama-3.2-1B", "9535bd9b"
SHA = "c0ffee" * 6 + "abcd"
RELEASE = "r16-int-3|commit=a2dabe493 tree=/x"
FAMILY_OK = "TARGET-FAMILY-OK row family 8.9 == Build target 8.9"
PENDING = "Commit PENDING: every rank's C2 replay runs on CPU from its sealed bundle (`row stage replay`, `tp-commit-replay`)"


def _tp_row(tmp: Path, *argv: str, row: str = DENSE, world: int = 2) -> TpRow:
    """The row as `verity-vllm row run` makes it, under an environment that names nothing but the path."""
    o = cli.parse("row", ["run", row, ROLE, REPO, REV, "--sweep-dir", str(tmp), "--py", sys.executable,
                          "--native-collect-default", str(tmp / "nc_build"), *argv], {"PATH": os.environ.get("PATH", "")})
    return TpRow(row=o.row, role=o.role, repo=o.repo, rev=o.rev, stages=o.stages, env=o.env, world=world, gpu_util=None,
                 build=o.build, match=o.match, commit=o.commit)


class _Host:
    """`t`'s prechecks that start a process and the host's checkpoint dir, as stand-ins: `asked` holds what each was asked, in order."""

    def __init__(self, t: TpRow, monkeypatch: pytest.MonkeyPatch, *, release: str = RELEASE, sha: str = SHA, identity: int = 0,
                 family: tuple[int, str] = (0, FAMILY_OK), ck: str = "", requests: int | None = None) -> None:
        self.asked: list[tuple[Any, ...]] = []
        shape = R.tp_shape

        def release_line(fallback: str) -> tuple[str, str]:
            self.asked.append(("release_line", fallback))
            return release, "-"

        def source_identity(pys: list[str], out_of: Any) -> int:
            self.asked.append(("source_identity", pys, [out_of(py) for py in pys]))
            return identity

        def target_family(*extra: str, out: str) -> tuple[int, str]:
            self.asked.append(("target_family", extra, out))
            return family

        def tp_shape(wl: str, checkpoints: str, hf_home: str) -> dict[str, Any]:
            self.asked.append(("tp_shape", wl, checkpoints, hf_home))
            return {**shape(wl, checkpoints, hf_home), "ck": ck, **({} if requests is None else {"requests": requests})}

        for name, fn in (("release_line", release_line), ("export_commit", lambda: sha), ("source_identity", source_identity),
                         ("target_family", target_family), ("install_signal_forwarding", lambda: self.asked.append(("signals",)))):
            monkeypatch.setattr(t, name, fn)
        monkeypatch.setattr(R, "tp_shape", tp_shape)
        monkeypatch.setattr(row_driver, "utc", lambda: "T")

    def names(self) -> list[str]:
        return [a[0] for a in self.asked]


def _stages(t: TpRow) -> list[str]:
    return Path(t.d, "stages.txt").read_text().splitlines()


def _log(t: TpRow) -> list[str]:
    return Path(t.d, "row.log").read_text().splitlines()


def _events(t: TpRow, stage: str) -> list[dict[str, Any]]:
    return [e for e in map(json.loads, Path(t.timeline).read_text().splitlines()) if e["stage"] == stage]


def _ck(tmp: Path) -> str:
    ck = tmp / "ck"
    ck.mkdir()
    return str(ck)


# ---------------------------------------------------------------------------------------------------------------- a precheck that passes

def test_a_row_that_passes_its_precheck_has_its_shape_its_engine_and_its_first_line(tmp_path, monkeypatch, capsys):
    t = _tp_row(tmp_path, "--py312", "/x/python3.12")
    ck = _ck(tmp_path)
    host = _Host(t, monkeypatch, ck=ck)
    monkeypatch.chdir(tmp_path)                                      # the row's file and its pins are the tree's, wherever the row runs from
    t.prepare()
    d = t.d
    assert host.asked == [("release_line", ""), ("source_identity", [sys.executable], [f"{d}/source_identity.json"]),
                          ("target_family", (), f"{d}/target_family.json"),
                          ("tp_shape", t.path(t.wl), t.path("manifests/checkpoints.json"), t.env.hf_home), ("signals",)]
    assert (t.B, t.I, t.O, t.shapes, t.requests, t.MML, t.MNBT, t.ck) == (1, 256, 32, [(256, 31)], 1, 288, 257, ck)
    assert (t.bdir, t.M, t.ncb) == (f"{d}/build", f"{d}/match", str(tmp_path / "nc_build"))
    assert all(os.path.isdir(p) for p in (d, t.bdir, t.M, t.ncb))
    assert (t.nccl, t.hidden_so, t.tgt_args, t.fold_profile, t.code9, t.trap) == ("1", t.env.hidden_so_fa2_guarded, [], f"derived_{ROLE}_tp2", SHA[:9], True)
    assert t.engine == ["--engine-arg", "max_model_len=288", "--engine-arg", "max_num_batched_tokens=257",
                        "--engine-arg", "disable_custom_all_reduce=true"]
    pins = {"MATCH_IMPL": "fast", "MATCH_PIPELINE": "shared", "NATIVE_COLLECT_BUILD": t.ncb, "VERITY_INSTANCES_FORM": "progressions",
            "NCCL_P2P_DISABLE": "1", "VERITY_TREE": t.tree, "HF_HUB_OFFLINE": "1"}
    assert {k: t.child_env[k] for k in pins} == pins
    assert _stages(t) == ["release r16-int-3 T commit=a2dabe493 tree=/x tp=2",
                          f"precheck PASS T source identity == {t.tree} @ {SHA} (verity verity_vllm research)",
                          f"precheck-target PASS T {FAMILY_OK}"]
    start = (f"T [row {DENSE}] start tp=2 B=1 I=256 O=32 MML=288 mnbt=257 shapes=[256,31] ckpt={ck} stages={t.stages} "
             f"fold_profile=derived_{ROLE}_tp2 gpu_util=0.6 hidden_so={t.env.hidden_so_fa2_guarded} code={SHA[:9]} NCCL_P2P_DISABLE=1 "
             "disable_custom_all_reduce=true")
    assert _log(t) == [start] and capsys.readouterr().out == start + "\n"      # the three verdicts of a passing precheck are not printed
    assert [(e["ev"], e.get("rc"), e.get("fail_class")) for e in _events(t, "precheck")] == [("start", None, None), ("end", 0, None)]
    (row,) = _events(t, "row")
    assert (row["ev"], row["stages"], row["B"], row["tp"], row["code"], row["pod"], row["proc"]) == (
        "start", t.stages, 1, 2, SHA[:9], os.uname().nodename, "tp_stage")


def test_a_row_of_two_shapes_names_both_and_its_batch_bounds_the_tokens_of_a_step(tmp_path, monkeypatch):
    t = _tp_row(tmp_path, row=MOE)
    _Host(t, monkeypatch, ck=_ck(tmp_path))
    t.prepare()
    assert (t.B, t.I, t.O, t.shapes, t.requests, t.MML, t.MNBT) == (2, 1024, 128, [(9, 1), (1024, 127)], 2, 1152, 2048)
    assert " B=2 I=1024 O=128 MML=1152 mnbt=2048 shapes=[9,1 1024,127] " in _log(t)[-1]
    assert t.engine[1::2] == ["max_model_len=1152", "max_num_batched_tokens=2048", "disable_custom_all_reduce=true"]


def test_a_row_that_declares_a_target_names_it_to_its_derives_and_keeps_its_own_engine_environment(tmp_path, monkeypatch):
    """GLM-5.3's row declares its target and names no NCCL_P2P_DISABLE, at a world of 8: its derives get ``--target``, its stages run
    with the variable unset, and its engines keep the custom all-reduce."""
    t = _tp_row(tmp_path, row=GLM53, world=8)
    _Host(t, monkeypatch, ck=_ck(tmp_path))
    t.prepare()
    (flag, target) = t.tgt_args
    assert flag == "--target" and target == R.declared_target(t.path(t.wl), row=GLM53)
    assert json.loads(target) == {"batch_invariant": False, "compute_capability": [9, 0], "moe_backend": "triton", "num_sms": 132}
    assert t.nccl is None and "NCCL_P2P_DISABLE" not in t.child_env and t.fold_profile == f"derived_{ROLE}_tp8"
    assert _stages(t)[0].endswith(" tp=8") and _log(t)[-1].endswith(" NCCL_P2P_DISABLE=unset disable_custom_all_reduce=false")
    assert _log(t)[-1].startswith(f"T [row {GLM53}] start tp=8 B=1 I=256 O=32 MML=288 mnbt=288 shapes=[256,31] ")
    assert _events(t, "row")[0]["tp"] == 8


def test_a_row_whose_file_declares_no_target_gets_the_one_of_its_ids_device_class(tmp_path, monkeypatch):
    t = _tp_row(tmp_path, row=SM120)
    _Host(t, monkeypatch, ck=_ck(tmp_path))
    t.prepare()
    assert t.tgt_args == ["--target", json.dumps({"compute_capability": [12, 0], "num_sms": 188})]
    assert "target" not in json.loads(Path(t.path(t.wl)).read_text())["engine"]


def test_a_rows_requests_are_its_shapes_count_of_them_which_its_batch_is_not(tmp_path, monkeypatch):
    t = _tp_row(tmp_path)
    _Host(t, monkeypatch, ck=_ck(tmp_path), requests=5)
    t.prepare()
    assert (t.B, t.requests) == (1, 5)


@pytest.mark.parametrize("argv, ncb", [((), "nc_build"), (("--native-collect-build", "{tmp}/nc_own"), "nc_own")])
def test_the_collectors_build_dir_is_the_callers_else_the_default_and_the_row_makes_it(tmp_path, monkeypatch, argv, ncb):
    t = _tp_row(tmp_path, *(a.format(tmp=tmp_path) for a in argv))
    _Host(t, monkeypatch, ck=_ck(tmp_path))
    t.prepare()
    assert t.ncb == t.child_env["NATIVE_COLLECT_BUILD"] == str(tmp_path / ncb) and os.path.isdir(t.ncb)
    assert not os.path.exists(tmp_path / {"nc_build": "nc_own", "nc_own": "nc_build"}[ncb])


@pytest.mark.parametrize("argv, profile, said", [
    ((), f"derived_{ROLE}_tp2", f"derived_{ROLE}_tp2"),             # the default: the rank profile of the role at this world
    (("--fold-profile", "mine"), "mine", "mine"),
    (("--fold-profile", ""), "", "none"),                           # '' = collective-level only, and the row says so
])
def test_a_rows_fold_profile_is_the_callers_else_the_one_derived_for_its_role_and_world(tmp_path, monkeypatch, argv, profile, said):
    t = _tp_row(tmp_path, *argv)
    _Host(t, monkeypatch, ck=_ck(tmp_path))
    t.prepare()
    assert t.fold_profile == profile and f" fold_profile={said} gpu_util=" in _log(t)[-1]


@pytest.mark.parametrize("release, line", [
    (RELEASE, "release r16-int-3 T commit=a2dabe493 tree=/x tp=2"),
    ("UNRELEASED", "release UNRELEASED T UNRELEASED tp=2"),          # a line with no bar is its own detail
    ("r16|", "release r16 T  tp=2"),
])
def test_the_release_line_is_the_rows_first_verdict_its_tag_before_the_bar(tmp_path, monkeypatch, release, line):
    t = _tp_row(tmp_path)
    _Host(t, monkeypatch, release=release, ck=_ck(tmp_path))
    t.prepare()
    assert _stages(t)[0] == line


def test_a_tree_that_states_no_commit_passes_its_precheck_under_no_commit(tmp_path, monkeypatch):
    t = _tp_row(tmp_path)
    _Host(t, monkeypatch, sha="", ck=_ck(tmp_path))
    t.prepare()
    assert _stages(t)[1] == f"precheck PASS T source identity == {t.tree} (verity verity_vllm research)"
    assert t.code9 == "" and " code= NCCL_P2P_DISABLE=1 " in _log(t)[-1]


# ---------------------------------------------------------------------------------------------------------------- a precheck that stops

def test_a_row_with_no_file_stops_before_it_writes_anything(tmp_path, monkeypatch, capsys):
    t = _tp_row(tmp_path)
    host = _Host(t, monkeypatch)
    t.wl = "workloads/absent.json"
    monkeypatch.setattr(t, "research_result", lambda: pytest.fail("a result envelope from a row that never started"))
    assert t.run() == 2
    assert capsys.readouterr().out == "no manifest workloads/absent.json\n"
    assert host.asked == [] and t.trap is False and not os.path.exists(t.d)


def test_a_source_identity_that_fails_stops_the_row_at_its_precheck_by_name(tmp_path, monkeypatch):
    t = _tp_row(tmp_path)
    host = _Host(t, monkeypatch, identity=3)
    results: list[int] = []
    monkeypatch.setattr(t, "research_result", lambda: results.append(1))
    assert t.run() == 3
    assert _stages(t)[1:] == [f"precheck FAIL T PRECHECK_FAIL_SOURCE_IDENTITY (see {t.d}/source_identity.json and row.log)"]
    assert [(e["ev"], e.get("rc"), e.get("fail_class")) for e in _events(t, "precheck")] == [
        ("start", None, None), ("end", 3, "PRECHECK_FAIL_SOURCE_IDENTITY")]
    assert host.names() == ["release_line", "source_identity"] and results == [1], "a row that stops past its setup still writes its result"
    assert not _events(t, "row") and not os.path.exists(f"{t.d}/row.log")


def test_a_target_family_that_differs_stops_the_row_before_it_reads_its_shape(tmp_path, monkeypatch):
    line = "PRECHECK_FAIL_TARGET_FAMILY: Build target 9.0 != row family 8.9"
    t = _tp_row(tmp_path)
    host = _Host(t, monkeypatch, family=(3, line))
    with pytest.raises(Stop) as stop:
        t.prepare()
    assert stop.value.rc == 3 and _stages(t)[1:] == [
        f"precheck PASS T source identity == {t.tree} @ {SHA} (verity verity_vllm research)", f"precheck-target FAIL T {line}"]
    assert host.names() == ["release_line", "source_identity", "target_family"] and not hasattr(t, "shapes")
    assert [(e["ev"], e.get("rc")) for e in _events(t, "precheck")] == [("start", None), ("end", 0)]


@pytest.mark.parametrize("ck, said", [("", "-"), ("absent", "absent"), ("a_file", "a_file")])
def test_a_host_without_the_checkpoint_dir_fails_the_build_after_the_row_has_started(tmp_path, monkeypatch, ck, said):
    """The checkpoint's pin names no local dir, one that is not there, or a file: the row has started (its line, its timeline event, its
    signal handlers) and stops as a failed Build."""
    (tmp_path / "a_file").write_text("")
    path = str(tmp_path / ck) if ck else ""
    t = _tp_row(tmp_path)
    host = _Host(t, monkeypatch, ck=path)
    with pytest.raises(Stop) as stop:
        t.prepare()
    assert stop.value.rc == 10 and host.names()[-2:] == ["tp_shape", "signals"]
    assert _stages(t)[-1] == (f"build FAIL T no local checkpoint dir for {REPO} @ {REV} (checkpoints.json local_path / HF snapshot absent; "
                              "HF_HUB_OFFLINE=1)")
    assert f" ckpt={path if ck else said} stages=" in _log(t)[-1] and [e["ev"] for e in _events(t, "row")] == ["start"]


# ---------------------------------------------------------------------------------------------------------------- the order of the stages

class _Stages:
    """`t` with each stage, its precheck and what it writes on its way out as stand-ins: `ran` holds their names in order."""

    def __init__(self, t: TpRow, monkeypatch: pytest.MonkeyPatch, *, pending: bool = False, stops: dict[str, int] | None = None) -> None:
        self.ran: list[str] = []
        halt = stops or {}
        os.makedirs(t.d, exist_ok=True)

        def stage(name: str) -> None:
            self.ran.append(name)
            if name == "prepare":
                t.trap = True
            if name == "commit" and pending:
                t.replay_pending = True
            if name in halt:
                raise Stop(halt[name])

        for name in ("prepare", "build", "match", "commit", "replay", "research_result", "timeline_report"):
            monkeypatch.setattr(t, name, lambda name=name: stage(name))
        for name in ("word_check", "config_record", "commit_verdict"):
            monkeypatch.setattr(RC, name, lambda row, *a, name=name: self.ran.append(name))
        monkeypatch.setattr(row_driver, "utc", lambda: "T")


@pytest.mark.parametrize("stages, ran", [
    ("build,match,commit,replay", ["prepare", "build", "match", "commit", "replay"]),
    ("build,match,commit", ["prepare", "build", "match", "commit"]),
    ("match", ["prepare", "match"]),
    ("commit,replay", ["prepare", "commit", "replay"]),
    ("replay", ["prepare", "replay"]),
    ("plan", ["prepare"]),                                           # a stage a TP row has none of
])
def test_a_row_runs_the_stages_it_was_asked_for_in_their_order_and_ends_done(tmp_path, monkeypatch, stages, ran):
    t = _tp_row(tmp_path, "--stages", stages)
    row = _Stages(t, monkeypatch)
    assert t.run() == 0
    assert row.ran == [*ran, "research_result", "timeline_report"]
    assert _log(t) == [f"T [row {DENSE}] DONE"]
    (end,) = _events(t, "row")
    assert (end["ev"], end["ok"]) == ("end", True)


@pytest.mark.parametrize("stopped, rc, ran", [
    ("build", 10, ["prepare", "build"]),
    ("match", 11, ["prepare", "build", "match"]),
    ("commit", 12, ["prepare", "build", "match", "commit"]),
    ("replay", 12, ["prepare", "build", "match", "commit", "replay"]),
])
def test_a_stage_that_stops_ends_the_row_with_its_exit_and_no_stage_after_it_runs(tmp_path, monkeypatch, stopped, rc, ran):
    t = _tp_row(tmp_path, "--stages", "build,match,commit,replay")
    row = _Stages(t, monkeypatch, stops={stopped: rc})
    assert t.run() == rc
    assert row.ran == [*ran, "research_result"] and not os.path.exists(f"{t.d}/row.log") and not os.path.exists(t.timeline)


def test_a_commit_that_deferred_its_replay_leaves_the_row_pending_until_the_replay_stage_decides_it(tmp_path, monkeypatch):
    """A Commit that sealed its ranks' replay bundles leaves the row pending, which the row says before it ends; the replay stage in the
    same run decides it, and the row says nothing of a pending Commit."""
    t = _tp_row(tmp_path, "--stages", "commit")
    row = _Stages(t, monkeypatch, pending=True)
    assert t.run() == 0 and t.replay_pending is True
    assert row.ran == ["prepare", "commit", "research_result", "timeline_report"]
    assert _log(t) == [f"T [row {DENSE}] {PENDING}", f"T [row {DENSE}] DONE"]

    t = _tp_row(tmp_path / "decided", "--stages", "commit,replay")
    row = _Stages(t, monkeypatch, pending=True)
    assert t.run() == 0 and t.replay_pending is False
    assert row.ran == ["prepare", "commit", "replay", "research_result", "timeline_report"] and _log(t) == [f"T [row {DENSE}] DONE"]

    t = _tp_row(tmp_path / "stopped", "--stages", "commit,replay")
    row = _Stages(t, monkeypatch, pending=True, stops={"replay": 12})
    assert t.run() == 12 and t.replay_pending is True, "a replay that stopped decided nothing"


def test_a_config_run_has_no_match_and_one_that_is_not_writes_no_config_record(tmp_path, monkeypatch):
    t = _tp_row(tmp_path, "--stages", "build,match,commit", "--config-run", "1")
    row = _Stages(t, monkeypatch)
    assert t.run() == 0 and row.ran == ["prepare", "build", "word_check", "commit", "research_result", "config_record", "timeline_report"]

    t = _tp_row(tmp_path / "of_record", "--stages", "build,match,commit")
    row = _Stages(t, monkeypatch, stops={"commit": 12})
    assert t.run() == 12 and row.ran == ["prepare", "build", "match", "commit", "research_result"]
