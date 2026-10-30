"""`circuit-check TARGET...`: named targets (a Definition, or a sidecar's own) run one worker per core and are reused from
check's own cache while their inputs are unchanged, unless told otherwise; each target's result and log line, and the run's
last line, say where its time went (`checks.spent`)."""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from circuit_check import cache as CC
from circuit_check import checks as CK
from circuit_check import cli
from circuit_check import sidecars as SC
from circuit_check import targets as TG

GF2K = TG.REPO / "verity" / "ml" / "circuits" / "ops" / "boolean" / "gf2k.circuit-check.py"


def _target(name: str, seconds: float, by: dict, **more) -> dict:
    return {"kind": "definition", "target": name, "failures": [], "warnings": [], "seconds": seconds, "seconds_by_check": by, **more}


@pytest.fixture
def runs(monkeypatch, tmp_path):
    """`cli.main` with `cli.run` recorded instead of run, and neither the registries' prebuild nor the tracer started."""
    got: list[dict] = []
    traced: list[Path] = []

    def run(specs, opts, **kw):
        got.append({"specs": specs, **kw})
        return CK.report([_target(s, 2.0, {"correspondence": 1.5, "lowering": 0.4, "ports": 0.01}) for s in specs], opts)

    monkeypatch.setattr(cli, "run", run)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(CC, "trace", traced.append)
    monkeypatch.setattr(cli, "update_pins", lambda rep: {})
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "check"))
    monkeypatch.setattr(cli, "cores", lambda root=Path("/"): 6)
    monkeypatch.setattr("circuit_check.known.stale", lambda failures: [])
    return got, traced


def test_named_targets_run_on_every_core_and_reuse_checks_cache(runs, tmp_path, capsys):
    got, traced = runs
    assert cli.main(["F32Add_v1", "F32Mul_v1"]) == 0
    assert got[-1]["jobs"] == 6 and got[-1]["cache"] == tmp_path / "check" / "circuit-check" == cli.default_cache()
    assert traced == [cli.REPO]
    out = capsys.readouterr().out
    assert "circuit-check: " in out and "at -j 6: 2 target(s) checked in 4.0s (correspondence 3.0, lowering 0.8" in out
    assert cli.main(["F32Add_v1", "--no-cache", "-j", "2"]) == 0
    assert got[-1]["jobs"] == 2 and got[-1]["cache"] is None and len(traced) == 1
    assert cli.main(["F32Add_v1", "--update-pins"]) == 0 and got[-1]["cache"] is None
    assert cli.main(["F32Add_v1", "--cache", str(tmp_path / "mine"), "-j", "0"]) == 0
    assert got[-1]["cache"] == tmp_path / "mine" and got[-1]["jobs"] == cli.auto_jobs()


def test_as_call_says_a_targets_call_either_way_or_leaves_it_to_is_call(runs):
    """``--as-call`` and ``--no-as-call`` say whether a Definition target is a Call, which is all a target `targets.is_call` can't
    decide without the catalog needs; with neither, `is_call` decides it."""
    got, _ = runs
    for flags, want in (([], None), (["--as-call"], True), (["--no-as-call"], False)):
        assert cli.main(["F32Add_v1", *flags]) == 0 and got[-1]["as_call"] is want


def test_all_keeps_one_job_and_no_cache_unless_given(runs, tmp_path):
    """check.py gives `--all` its slot's jobs and cache explicitly; a bare `--all` stays as it was."""
    got, _ = runs
    assert cli.main(["--all", "-q"]) == 0
    assert got[-1]["jobs"] == 1 and got[-1]["cache"] is None
    assert cli.main(["--all", "-q", "--jobs", "8", "--cache", str(tmp_path / "c")]) == 0
    assert got[-1]["jobs"] == 8 and got[-1]["cache"] == tmp_path / "c"


def test_cores_keeps_to_the_cgroups_quota(tmp_path):
    import os

    (tmp_path / "sys/fs/cgroup").mkdir(parents=True)
    n = len(os.sched_getaffinity(0))
    assert cli.cores(tmp_path) == n
    (tmp_path / "sys/fs/cgroup/cpu.max").write_text("max 100000\n")
    assert cli.cores(tmp_path) == n
    (tmp_path / "sys/fs/cgroup/cpu.max").write_text("250000 100000\n")
    assert cli.cores(tmp_path) == min(n, 2)
    (tmp_path / "sys/fs/cgroup/cpu.max").write_text("50000 100000\n")
    assert cli.cores(tmp_path) == 1


def test_a_targets_time_is_split_by_check_net_of_the_checks_inside(monkeypatch):
    """`checks.spent` charges each check its own time: the lowering's two seconds inside the correspondence's five leave it
    three; a target's ``seconds_by_check`` is heaviest first, and its log line names those over `cli.SHOWN_SECONDS`."""
    ticks = iter([0.0, 1.0, 3.0, 5.0, 5.0, 5.05])
    monkeypatch.setattr(CK, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))

    def check() -> dict:
        with CK.spent("correspondence"):
            with CK.spent("lowering"):
                pass
        with CK.spent("ports"):
            pass
        return {"failures": [], "warnings": []}

    r = cli._one("definition", "X", check)
    assert r["seconds_by_check"] == {"correspondence": 3.0, "lowering": 2.0, "ports": 0.05}
    assert list(r["seconds_by_check"]) == ["correspondence", "lowering", "ports"]
    assert cli._where_time_went({**r, "seconds": 5.1}) == "5.1s: correspondence 3.0, lowering 2.0"


def test_the_last_line_sums_the_checked_targets_and_counts_the_reused():
    rep = CK.report([_target("A", 3.0, {"partition": 2.0, "lowering": 1.0}), _target("B", 1.0, {"lowering": 1.0}),
                     _target("C", 9.0, {"partition": 9.0}, cache={"hit": "e1"})], CK.Options())
    assert cli.time_summary(rep, 3.5, 4) == ("circuit-check: 3.5s wall at -j 4: 2 target(s) checked in 4.0s "
                                              "(partition 2.0, lowering 2.0), 1 reused")


def test_a_warning_is_printed_unchecked_apart_from_the_passes_and_counted(runs, monkeypatch, capsys):
    """A warning never fails: a target with one is logged ``unchecked``, not ``ok``; the warnings are printed under ``unchecked``
    by check, with counts, and the last line says how many there are, on how many Definitions."""
    F = CK.Findings()
    F.warn("redundant-gates", "ir", "F32Add_v1", 3, "values computed again inside their own unit")
    F.warn("lowering", "unpinned", "F32Add_v1", 1, "[F32Add_v1]: no pin")
    F.warn("partition", "unchecked", "F32Mul_v1", 1, "6000000 gates: over the partition budget 5000000")
    by = {"F32Add_v1": F.warnings[:2], "F32Mul_v1": F.warnings[2:]}
    monkeypatch.setattr(cli, "run", lambda specs, opts, **kw: CK.report([_target(s, 1.0, {}, warnings=by[s]) for s in specs], opts))
    assert cli.main(["F32Add_v1", "F32Mul_v1"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [s for s in out if s.startswith(("unchecked ", "warn "))] == [
        "unchecked redundant-gates/ir: 3 over 1 target(s)", "unchecked lowering/unpinned: 1 over 1 target(s)",
        "unchecked partition/unchecked: 1 over 1 target(s)"]
    assert "3 unchecked (warnings, which never fail) on 2 Definition(s)" in out[-1]
    assert cli.main(["F32Add_v1", "--fail-on-warnings"]) == 1
    statuses = [cli._status(t) for t in (_target("A", 0, {}, warnings=by["F32Add_v1"]), _target("B", 0, {}),
                                          _target("C", 0, {}, failures=[{}], warnings=by["F32Mul_v1"]))]
    assert [s.strip() for s in statuses] == ["unchecked", "ok", "FAIL"]


def test_a_sidecar_names_the_targets_its_bindings_reach_in_its_module():
    """``circuit-check path/to/<module>.circuit-check.py`` checks what that sidecar binds, not the catalog: the Definitions its
    root functions reach that its module defines, selected as the suite selects them, each with its binding unit, and the
    tasks are keyed by their roots' descriptors, as ``--all``'s are."""
    assert TG.kind_of(str(GF2K)) == "sidecar" and TG.kind_of("GfScale_v1{L=2}") == "definition"
    found, templates = TG.sidecar_targets(str(GF2K))
    assert not templates and {"GfScale_v1{L=2}", "Gf128Mul_v1"} <= set(found)
    assert {SC.owner(fn) for fn in found.values()} == {"verity.ml.circuits.ops.boolean.gf2k"}
    unit = "verity.ml.circuits.ops.boolean.gf2k.circuit-check#_gf2k_roots"
    assert all(unit in TG.BINDINGS[fid] for fid in found)
    tasks = cli._tasks([str(GF2K)], CK.Options(), False, None)
    assert [t[1] for t in tasks] == list(found)
    assert all(t[0] == "definition" and t[3]["roots"] == [found[t[1]]] and unit in t[3]["units"] for t in tasks)
