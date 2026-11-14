"""`verity_vllm.pipeline.commit`'s small parts that `test_commit_main` stands in for or never reaches: the source's commit
(`git_head`), the device's memory in use (`nvidia_smi_used_mib`), an engine row's request (`_rid_of`) and the staging's
environment (`staging_environ`) where it refuses, excludes families, traces the footprint or drops a flush point; the Programs'
forecast (`c2_programs_forecast`) where an instance file can't be read, and the check that the engine is left as it was
(`assert_pristine`) where a runner's method or a module's forward is still wrapped, or the sampler keeps a hook.  The two
commands each reader runs are stood in.  CPU only."""
from __future__ import annotations

import dataclasses
import gzip
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from verity_vllm import settings
from verity_vllm.commit.committer import native_collect as NC
from verity_vllm.pipeline import cli, commit

#: The environment names `staging_environ` writes: absent at a test's start, and taken back at its end.
ENVIRON = ("VERITY_RETAIN", "VERITY_WINDOW_MB", "VERITY_WINDOW_SLOTS", "VERITY_STAGING_BOUNDED", "VERITY_LEARN_HOST_BUDGET_MB",
           "VERITY_RETAIN_EXCLUDE", "VERITY_FOOTPRINT")


def runs(monkeypatch: pytest.MonkeyPatch, outcome: str | Exception) -> list[list[str]]:
    """`subprocess.run` prints ``outcome`` (raises it if it is an exception); the commands it was asked for."""
    asked: list[list[str]] = []

    def run(cmd: list[str], **_: Any) -> SimpleNamespace:
        asked.append(cmd)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(stdout=outcome)
    monkeypatch.setattr(subprocess, "run", run)
    return asked


def test_the_source_s_commit_is_the_export_s_when_the_directory_holds_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    asked = runs(monkeypatch, "f" * 40 + "\n")
    (tmp_path / "EXPORT.json").write_text(json.dumps({"commit": "a" * 40}))
    assert commit.git_head() == "a" * 40 and asked == []


@pytest.mark.parametrize("export, printed, head", [
    (None, "f" * 40 + "\n", "f" * 40),
    ("{not json", "f" * 40 + "\n", "f" * 40),
    (None, "", None),
    (None, FileNotFoundError("git"), None),
])
def test_else_it_is_git_s_head_and_none_where_git_names_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export: str | None,
                                                              printed: str | Exception, head: str | None) -> None:
    monkeypatch.chdir(tmp_path)
    if export is not None:
        (tmp_path / "EXPORT.json").write_text(export)
    asked = runs(monkeypatch, printed)
    assert commit.git_head() == head and asked == [["git", "rev-parse", "HEAD"]]


@pytest.mark.parametrize("printed, mib", [("1234\n5678\n", 1234), ("", None), ("[N/A]\n", None), (FileNotFoundError("nvidia-smi"), None)])
def test_the_device_s_memory_in_use_is_the_first_device_s_and_none_where_it_cannot_be_read(
        monkeypatch: pytest.MonkeyPatch, printed: str | Exception, mib: int | None) -> None:
    asked = runs(monkeypatch, printed)
    assert commit.nvidia_smi_used_mib() == mib
    assert asked == [["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]]


@pytest.mark.parametrize("rid, named", [("abc", "r1"), ("abc-0123abcd", "r1"), ("abc-0123ABCD", "abc-0123ABCD"),
                                        ("xyz-0123abcd", "xyz-0123abcd"), ("def-89abcdef", "r3"), (7, "r7")])
def test_an_engine_row_s_request_is_the_workload_s_by_its_id_or_the_id_vllm_suffixed_else_the_raw_id(rid: Any, named: str) -> None:
    order = {"abc": "r1", "def-89abcdef": "r3", "7": "r7"}
    assert commit._rid_of(order, rid) == named


@pytest.fixture
def environ(monkeypatch: pytest.MonkeyPatch) -> Callable[[], dict[str, str | None]]:
    """The names `staging_environ` writes, absent now; what each holds when called."""
    for name in ENVIRON:
        monkeypatch.delenv(name, raising=False)
    return lambda: {name: settings.raw(name) for name in ENVIRON}


def staged(tmp_path: Path, **resolved: Any) -> commit.Options:
    """The options of a Commit with its staging resolved: bounded, retained on the host and opened after release, unless given."""
    args = cli.parse("commit", ["--workload", str(tmp_path / "workload.json"), "--out", str(tmp_path / "out")], {})
    return dataclasses.replace(args, **{"window_mb": 256, "window_slots": None, "bounded_staging": True, "retain": "host"} | resolved)


@pytest.mark.parametrize("unmet", [{"bounded_staging": False}, {"retain": "device"}, {"retain": "none"}, {"open_after_release": False}])
def test_families_are_excluded_only_from_bounded_staging_retained_on_the_host_and_opened_after_release(
        tmp_path: Path, environ: Callable[[], dict[str, str | None]], unmet: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match=r"^--retain-exclude needs --bounded-staging --retain host --open-after-release "):
        commit.staging_environ(staged(tmp_path, retain_exclude="mlp_up", **unmet))
    assert environ()["VERITY_RETAIN_EXCLUDE"] is None


def test_the_window_s_slots_the_excluded_families_and_the_footprint_s_file_go_to_the_environment(
        tmp_path: Path, environ: Callable[[], dict[str, str | None]]) -> None:
    footprint = tmp_path / "a" / "b" / "footprint.jsonl"
    commit.staging_environ(staged(tmp_path, retain_exclude="mlp_up,mlp_down", footprint=str(footprint), learn_budget_mb=512,
                                window_slots=4))
    assert environ() == {"VERITY_RETAIN": "host", "VERITY_WINDOW_MB": "256", "VERITY_WINDOW_SLOTS": "4", "VERITY_STAGING_BOUNDED": "1",
                         "VERITY_LEARN_HOST_BUDGET_MB": "512", "VERITY_RETAIN_EXCLUDE": "mlp_up,mlp_down",
                         "VERITY_FOOTPRINT": str(footprint)}
    assert footprint.parent.is_dir() and not footprint.exists()


def test_a_late_read_drops_its_leaf_from_the_gather_flush_points_and_says_so(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        environ: Callable[[], dict[str, str | None]]) -> None:
    before = NC.GATHER_FLUSH_LEAVES
    leaf = sorted(before)[0]
    monkeypatch.setattr(NC, "GATHER_FLUSH_LEAVES", before)
    commit.staging_environ(staged(tmp_path, late_read=leaf))
    left = NC.GATHER_FLUSH_LEAVES
    assert left == before - {leaf} and leaf in before
    assert capsys.readouterr().out == (f"[delta] LATE-READ NEGATIVE TEST: gather flush points {sorted(left)} (dropped {leaf!r}); "
                                       "roots must NOT match arm 2\n")
    assert environ()["VERITY_RETAIN_EXCLUDE"] is None


def test_an_instance_file_that_cannot_be_read_is_named_and_counts_no_rows(tmp_path: Path) -> None:
    good, bad = tmp_path / "build_request_a" / "instances.json.gz", tmp_path / "build_request_b" / "instances.json.gz"
    text = json.dumps({"rows": [{"i": 0}, {"i": 1}]}).encode()
    for f in (good, bad):
        f.parent.mkdir()
    with gzip.open(good, "wb") as fh:
        fh.write(text)
    bad.write_bytes(text)
    with pytest.raises(gzip.BadGzipFile) as raised, gzip.open(bad, "rb") as fh:
        fh.read()
    fc = commit.c2_programs_forecast(str(tmp_path), None, factor=2.0)
    assert (fc["rows"], fc["json_bytes"], fc["bytes"], fc["why"]) == (2, len(text), 2 * len(text), None)
    assert fc["files"] == [{"file": str(good), "rows": 2, "json_bytes": len(text), "gz_bytes": good.stat().st_size},
                           {"file": str(bad), "error": repr(raised.value)[:200]}]


def engine(**runner: Any) -> Any:
    """An engine whose runner holds ``runner`` and whose model has two modules, as `assert_pristine` reads one."""
    return SimpleNamespace(runner=SimpleNamespace(**runner), model=torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU()))


BASELINE: dict[str, Any] = {"runner_had": {}, "hooks": 0, "sampler_hooks": 0}


@pytest.mark.parametrize("nm", ["execute_model", "sample"])
def test_a_runner_method_still_wrapped_is_refused_unless_the_runner_had_it_before(nm: str) -> None:
    eng = engine(**{nm: lambda *a: None})
    with pytest.raises(AssertionError, match=rf"^runner\.{nm} still wrapped$"):
        commit.assert_pristine(eng, BASELINE)
    commit.assert_pristine(eng, BASELINE | {"runner_had": {nm: True}})


def test_a_module_forward_still_wrapped_is_refused_unless_the_baseline_counts_it() -> None:
    eng = engine()
    commit.assert_pristine(eng, BASELINE)
    eng.model[0].forward = lambda x: x
    with pytest.raises(AssertionError, match=r"^wrapped module\.forward present: 1 != baseline 0$"):
        commit.assert_pristine(eng, BASELINE)
    commit.assert_pristine(eng, BASELINE | {"wrapped_forwards": 1})


def test_a_hook_left_on_the_sampler_is_refused_unless_the_baseline_counts_it() -> None:
    sampler = torch.nn.Identity()
    eng = engine(sampler=sampler)
    commit.assert_pristine(eng, BASELINE)
    sampler.register_forward_hook(lambda *a: None)
    with pytest.raises(AssertionError, match=r"^sampler hooks present$"):
        commit.assert_pristine(eng, BASELINE)
    commit.assert_pristine(eng, BASELINE | {"sampler_hooks": 1})
