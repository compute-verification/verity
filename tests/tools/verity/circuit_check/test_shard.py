"""``circuit-check --shard K/N`` and ``--combine`` (`circuit_check.shard`): which shard a target is, what a shard writes, what
``--combine`` refuses, and that a combined run's report, verdict and output are the unsharded run's.  And ``--binding NAME``
and ``--bindings``: each catalog binding's targets, together the whole list."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from circuit_check import cache as CC
from circuit_check import cli
from circuit_check import shard as SH
from circuit_check import targets as TG
from test_cache import REPO, _bindings, _edit, _tree
from test_units import BINDINGS as UNIT_BINDINGS
from test_units import RECORDS_ONLY


def test_a_target_s_shard_is_its_id_s_hash_and_never_its_place():
    """Pinned: a change to the hash moves every target, which the check's shard plans and their measured walls assume it doesn't."""
    assert [SH.shard_of("definition", "F32Add_v1", n) for n in (2, 3, 4, 7)] == [1, 3, 1, 1]
    assert [SH.shard_of("definition", "RMSNormTriton_v1{N=64,EPS=1e-05}", n) for n in (2, 3, 4, 7)] == [1, 1, 3, 4]
    assert [SH.shard_of("template", "rmsnorm-cuda-vec8/n6144-b256-eps1e-05/bf16", n) for n in (2, 3, 4, 7)] == [2, 2, 2, 6]
    assert SH.shard_of("definition", "F32Add_v1", 1) == 1
    ids = [("definition", f"Syn{i}_v1") for i in range(200)] + [("template", f"t/{i}") for i in range(50)]
    for n in (1, 2, 3, 4, 7):
        shards = [SH.mine(ids, (k, n)) for k in range(1, n + 1)]
        assert sorted(i for s in shards for i in s) == list(range(len(ids)))          # total: each target in exactly one
        assert all(s == sorted(s) for s in shards)
        moved = [("definition", "SynNew_v1"), *ids[:120], ("template", "t/new"), *ids[120:]]
        again = {moved[i]: k for k in range(1, n + 1) for i in SH.mine(moved, (k, n))}
        assert all(again[t] == k for k in range(1, n + 1) for t in (ids[i] for i in shards[k - 1]))   # a new target moves none


@pytest.mark.parametrize("spec", ["0/4", "5/4", "4", "a/b", "-1/3", "1/0"])
def test_a_shard_is_one_of_1_to_n(spec):
    with pytest.raises(ValueError, match="1 <= K <= N"):
        SH.parse(spec)
    assert SH.parse("1/4") == (1, 4) and SH.parse(" 4/4 ") == (4, 4)


def _fail(label: str, code: str, check: str = "lowering") -> dict:
    return {"check": check, "code": code, "definition": label, "name": f"{check}/{code} {label}"}


def _result(failures=(), warnings=(), **more) -> dict:
    return {"failures": list(failures), "warnings": list(warnings), **more}


#: the stub targets: a pass, a failure, a warning, a Boolean Call recomputing across units, a by-calls root whose callee (another
#: shard's at N = 3) fails, so only the joined report fails the root, and three heavy ones
CALLEE, ROOT = "SynCallee_v1", "SynRoot_v1"
TARGETS = {
    "SynPass_v1": lambda: _result(),
    "SynNew_v1": lambda: _result([_fail("SynNew_v1", "lowering-mismatch")]),
    "SynWarn_v1": lambda: _result(warnings=[{**_fail("SynWarn_v1", "and", "redundant"), "count": 3}]),
    "SynAcross_v1": lambda: _result(partition={"q_word": 2, "recomputed_across": 5}),
    ROOT: lambda: _result(mode="by-calls", by_calls={"callees": [{"callee": "c0", "binding": CALLEE}]}),
    CALLEE: lambda: _result([_fail(CALLEE, "dead")]),
    "SynHeavyA_v1": lambda: _result(),
    "SynHeavyB_v1": lambda: _result(),
    "SynHeavyC_v1": lambda: _result(),
}
HEAVY = {"SynHeavyA_v1": 30, "SynHeavyB_v1": 20, "SynHeavyC_v1": 40}


@pytest.fixture
def stub(monkeypatch):
    """`cli.run` over `TARGETS` (and the template ``t/stub``), each checked in this process in the order `cli.parallel` was
    handed them, which ``order`` records per run: no Definition is built and no process forked."""
    tasks = [("definition", label, fn) for label, fn in TARGETS.items()] + [("template", "t/stub", lambda: _result())]
    order: list[list[str]] = []
    monkeypatch.setattr(cli, "_tasks", lambda specs, opts, everything, as_call: list(tasks))
    monkeypatch.setattr(cli, "Zygote", lambda *a, **k: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli, "heavy", lambda: dict(HEAVY))

    def parallel(fn, items, jobs, initializer=None, initargs=(), **_):
        order.append([cli._TASKS[i][1] for i in items])
        for i in items:
            yield i, (i, cli._one(*cli._TASKS[i][:3]), None, [], None), 1.0, 0.5

    monkeypatch.setattr(cli, "parallel", parallel)
    return types.SimpleNamespace(tasks=tasks, order=order)


#: what a record holds that two runs of one target needn't share: how long it took and how much memory it used
TIMING = ("seconds", "seconds_by_check", "peak_gb", "rss_start_gb")


def _untimed(rep: dict) -> list[dict]:
    return [{k: v for k, v in t.items() if k not in TIMING} for t in rep["targets"]]


def _shards(capsys, tmp_path: Path, n: int, base: list[str]) -> tuple[list[Path], list[int]]:
    dirs, codes = [], []
    for k in range(1, n + 1):
        d = tmp_path / f"shard{k}"
        codes.append(cli.main([*base, "--shard", f"{k}/{n}", "--shard-dir", str(d)]))
        dirs.append(d)
    capsys.readouterr()
    return dirs, codes


def test_combining_three_shards_gives_the_unsharded_run_s_report_verdict_and_output(stub, tmp_path, capsys):
    """Each shard runs only the targets its hash gives it, heaviest first, and writes its records unjoined; ``--combine`` of
    the three, in any order, prints what the unsharded ``--all`` prints, byte for byte, writes its report (its summary byte
    for byte, every record but its timings) and exits as it does: here 1, on its failures, among them a by-calls root whose
    callee is another shard's."""
    out = tmp_path / "r.json"
    base = ["--all", "--no-cache", "-q", "-j", "2"]
    whole = cli.main([*base, "--out", str(out)])
    said = capsys.readouterr().out
    rep = json.loads(out.read_text())
    assert whole == 1 and f"FAIL lowering/dead {CALLEE}" in said
    assert f"by-calls/callee-failed {ROOT}" in " ".join(rep["summary"]["failure_names"])
    out.unlink()

    dirs, codes = _shards(capsys, tmp_path, 3, base)
    ids = [(kind, label) for kind, label, _ in stub.tasks]
    for k, d in enumerate(dirs, 1):
        m = json.loads((d / SH.MANIFEST).read_text())
        mine = [ids[i] for i in SH.mine(ids, (k, 3))]
        assert (m["shard"], m["shards"]) == (k, 3) and [tuple(r[1:]) for r in m["ran"]] == mine
        assert m["list"] == {"targets": len(ids), "sha256": SH.digest(ids)}
        assert [(r["kind"], r["target"]) for r in json.loads((d / SH.RECORDS).read_text())] == mine
    assert sorted(t for o in stub.order[1:] for t in o) == sorted(label for _, label in ids)
    for o in stub.order[1:]:                                                    # each shard's heavy targets first, heaviest first
        heavy = [t for t in o if t in HEAVY]
        assert o[:len(heavy)] == heavy == sorted(heavy, key=lambda t: -HEAVY[t])
    assert [SH.shard_of("definition", t, 3) for t in (CALLEE, ROOT, "SynNew_v1")] == [1, 2, 3] and codes == [1, 0, 1]

    random.Random(3).shuffle(dirs)
    combined = cli.main(["--combine", *map(str, dirs), "-q", "--out", str(out)])
    assert (combined, capsys.readouterr().out) == (whole, said)
    got = json.loads(out.read_text())
    assert json.dumps(got["summary"]) == json.dumps(rep["summary"])
    assert _untimed(got) == _untimed(rep) and got["schema"] == rep["schema"] and got["options"] == rep["options"]
    assert got["combined"]["shards"] == 3 and sum(w["targets"] for w in got["combined"]["walls"]) == len(ids)
    assert all(0 <= w["prologue_s"] <= w["wall_s"] for w in got["combined"]["walls"]), got["combined"]["walls"]


def test_a_shard_fails_on_its_own_failures_and_only_the_combine_on_the_joined_ones(stub, tmp_path, capsys):
    """A shard fails on its own targets' failures alone: the by-calls root whose callee is in another shard passes in its own
    shard, and the combined run fails it as the unsharded one does, which only the whole list shows."""
    stub.tasks[:] = [t for t in stub.tasks if t[1] != "SynNew_v1"]
    base = ["--all", "--no-cache", "-q", "-j", "2"]
    whole = cli.main(base)
    said = capsys.readouterr().out
    assert whole == 1 and f"by-calls/callee-failed {ROOT}" in said
    dirs, codes, shards = [], [], []
    for k in (1, 2, 3):
        dirs.append(tmp_path / f"shard{k}")
        codes.append(cli.main([*base, "--shard", f"{k}/3", "--shard-dir", str(dirs[-1])]))
        shards.append(capsys.readouterr().out)
    assert codes == [1, 0, 0] and not any(f"by-calls/callee-failed {ROOT}" in o for o in shards)
    assert (cli.main(["--combine", *map(str, dirs), "-q"]), capsys.readouterr().out) == (whole, said)


def _refused(capsys, dirs, why: str) -> None:
    assert cli.main(["--combine", *map(str, dirs), "-q"]) == 2
    err = capsys.readouterr().err
    assert "circuit-check --combine: refused" in err and why in err, err


def _rewrite(d: Path, edit) -> None:
    m = json.loads((d / SH.MANIFEST).read_text())
    edit(m)
    (d / SH.MANIFEST).write_text(json.dumps(m))


def test_combine_refuses_a_missing_shard_an_extra_target_and_mismatched_lists(stub, tmp_path, capsys):
    base = ["--all", "--no-cache", "-q", "-j", "2"]
    dirs, _ = _shards(capsys, tmp_path, 3, base)
    assert cli.main(["--combine", *map(str, dirs), "-q"]) == 1
    capsys.readouterr()
    _refused(capsys, dirs[:2], "shard(s) 3 of 3 missing")
    _refused(capsys, [dirs[0], dirs[1], dirs[1]], "given twice")
    ids = [(kind, label) for kind, label, _ in stub.tasks]

    def copy(name: str) -> list[Path]:
        new = [tmp_path / name / d.name for d in dirs]
        for d, e in zip(dirs, new):
            shutil.copytree(d, e)
        return new

    unfinished = copy("unfinished")
    (unfinished[1] / SH.MANIFEST).unlink()
    _refused(capsys, unfinished, "did not finish")

    extra = copy("extra")                       # shard 1 ran a target its hash gives another shard
    k2 = SH.mine(ids, (2, 3))[0]
    recs = json.loads((extra[0] / SH.RECORDS).read_text()) + [{"kind": ids[k2][0], "target": ids[k2][1], "failures": []}]
    data = json.dumps(recs).encode()
    (extra[0] / SH.RECORDS).write_bytes(data)
    _rewrite(extra[0], lambda m: m.update(ran=[*m["ran"], [k2, *ids[k2]]], records_sha256=hashlib.sha256(data).hexdigest()))
    _refused(capsys, extra, "an extra target")

    dropped = copy("dropped")                   # shard 2 left one of its targets out
    recs = json.loads((dropped[1] / SH.RECORDS).read_text())[1:]
    data = json.dumps(recs).encode()
    (dropped[1] / SH.RECORDS).write_bytes(data)
    _rewrite(dropped[1], lambda m: m.update(ran=m["ran"][1:], records_sha256=hashlib.sha256(data).hexdigest()))
    _refused(capsys, dropped, "a missing target")

    tampered = copy("tampered")
    (tampered[2] / SH.RECORDS).write_text("[]")
    _refused(capsys, tampered, "not the records its manifest names")

    other = tmp_path / "other"                  # a shard of a run over another list (a target more)
    stub.tasks.append(("definition", "SynMore_v1", lambda: _result()))
    cli.main([*base, "--shard", "2/3", "--shard-dir", str(other)])
    capsys.readouterr()
    _refused(capsys, [dirs[0], other, dirs[2]], "disagree on the whole list")
    stub.tasks.pop()

    four = tmp_path / "four"                    # a shard of a run with another N
    cli.main([*base, "--shard", "3/4", "--shard-dir", str(four)])
    capsys.readouterr()
    _refused(capsys, [*dirs[:2], four], "disagree on N")

    seeded = tmp_path / "seeded"                # a shard of another request: its checks' options differ
    cli.main([*base, "--seed", "7", "--shard", "3/3", "--shard-dir", str(seeded)])
    capsys.readouterr()
    _refused(capsys, [*dirs[:2], seeded], "disagree on the request")

    swapped = copy("swapped")                   # a list of the same targets in another order: another list
    for d in swapped:
        _rewrite(d, lambda m: m["list"].update(sha256=SH.digest(ids[::-1])))
    _refused(capsys, swapped, "not the list their manifests name")


def test_the_shard_options_are_refused_where_they_mean_nothing(stub, tmp_path, capsys):
    for argv, why in ((["--all", "--shard", "0/3", "--shard-dir", "d"], "1 <= K <= N"),
                      (["--all", "--shard", "1/3"], "--shard-dir"),
                      (["--all", "--shard-dir", "d"], "give --shard K/N"),
                      (["--all", "--shard", "1/3", "--shard-dir", "d", "--update-pins"], "--update-pins"),
                      (["--combine", "d", "--all"], "--combine joins"),
                      (["--combine", "d", "--ambient"], "five modes"),
                      (["--combine", "d", "--records-only"], "five modes"),
                      (["--all", "--cache", "c", "--records-only", "--bindings"], "five modes"),
                      (["--bindings", "--cache", "c"], "--bindings names"),
                      (["--all", "--bindings"], "--bindings names"),
                      (["--all", "--bindings", "--cache", "c", "--binding", "_core_roots"], "--bindings names"),
                      (["--all", "--catalog-units", "--binding", "_core_roots"], "--binding checks"),
                      (["--all", "--cache", "c", "--binding", "_core_roots"], "--binding checks"),
                      (["--all", "--catalog-units", "--cache", "c", "--binding", "_core_roots", "SynPass_v1"], "--binding checks"),
                      (["--all", "--catalog-units", "--cache", "c", "--binding", "_core_roots", "--shard", "1/3",
                        "--shard-dir", "d"], "--binding checks"),
                      (["--records-only", "--cache", "c"], "--records-only stores"),
                      (["--all", "--records-only"], "--records-only stores"),
                      (["--all", "--records-only", "--cache", "c", "SynPass_v1"], "--records-only stores"),
                      (["--all", "--records-only", "--cache", "c", "--no-cache"], "--records-only stores"),
                      (["--all", "--records-only", "--cache", "c", "--update-pins"], "--records-only stores"),
                      (["--all", "--records-only", "--cache", "c", "--shard", "1/3", "--shard-dir", "d"], "--records-only stores"),
                      (["--reached", "main", "--all"], "--reached names"),
                      (["--reached", "main", "--shard", "1/3", "--shard-dir", "d"], "--reached names"),
                      (["--reached", "main", "--update-pins"], "--update-pins but with --check"),
                      (["--all", "--check"], "--check checks")):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err, argv
    _refused(capsys, [tmp_path / "nothing"], "did not finish")


def test_a_change_reaches_no_target_unless_it_touches_a_path_a_check_can_read(tmp_path, monkeypatch, capsys):
    """`shard.reaching`: a file modified outside the checker and the places its suite declares it reads, or prose anywhere,
    reaches no target, and ``--reached`` says so, building nothing; a path added or deleted anywhere may, as a module lookup
    finds it by its name."""
    checks = f"{SH.CHECKER}/src/circuit_check/checks.py"
    paths = [("M", "verity/circuits/ops/argmax.py"), ("M", "verity/errors.py"), ("M", checks), ("M", "verity/circuits/PROTOCOL.md"),
             ("M", "benchmarks/x.py"), ("M", "verity/circuitsx/y.py"), ("D", "adapters/vllm/x/README.md"), ("A", "tests/x/test_y.py")]
    assert SH.reaching(paths, ["verity/circuits", "verity/errors.py"]) == [
        "verity/circuits/ops/argmax.py", "verity/errors.py", checks, "adapters/vllm/x/README.md", "tests/x/test_y.py"]
    root = _tree(tmp_path, {SH.SUITE: '[tool.verity.tests]\ninputs = ["syn"]\n', "syn/README.md": "syn\n", "bench/run.py": "x = 1\n"})

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", *args], check=True,
                       capture_output=True)

    git("add", "-A")
    git("commit", "-qm", "base")
    git("branch", "base")
    _edit(root, "syn/README.md", "syn", "the synthetic registry")
    git("commit", "-qam", "prose")
    _edit(root, "bench/run.py", "1", "2")
    assert SH.changed(root, "base") == [("M", "bench/run.py"), ("M", "syn/README.md")]
    monkeypatch.setattr(cli, "REPO", root)
    assert cli.main(["--reached", "base"]) == 0
    assert "no target reached" in capsys.readouterr().out
    (root / "syn" / "new.py").write_text("")
    assert SH.changed(root, "base")[-1] == ("A", "syn/new.py")
    assert SH.reaching(SH.changed(root, "base"), SH.declared(root)) == ["syn/new.py"]


def test_the_modules_whose_source_inspect_looked_for_are_every_binding_s_loaded_probes(tmp_path, monkeypatch):
    """`cli._read_sets` takes the modules whose source `inspect` looked for (``present``) out of each binding's own set: their
    paths, but the files the set holds by their bytes, are ``loaded``, every binding's, since torch's import asks for the
    source of every module the process loaded and which those are depends on how the run went.  A target with no entry to
    take its reads from is its binding's ``incomplete``."""
    entries = {"kx": {"files": ["a.py"], "present": ["m", "own"]}, "ky": {"files": ["b.py"], "present": ["n"]}}
    where = {"m": "m.py", "n": "n.py", "own": "a.py"}
    monkeypatch.setattr(CC, "recorded", lambda cache, key, repo: entries.get(key))
    monkeypatch.setattr(CC, "declared", lambda root, reads: {
        "files": sorted(reads.get("files", ())), "listed": [], "probes": sorted(where[m] for m in reads.get("present", ())),
        "registry": False})
    group = {"files": [], "listed": [], "probes": []}
    ids, keys = [("definition", "X_v1"), ("definition", "Y_v1"), ("definition", "Z_v1")], ["kx", "ky", "kz"]
    got = cli._read_sets(tmp_path, {"x": [0], "y": [1]}, ids, keys, group)
    assert got == {"reads": {"registries": group, "loaded": ["m.py", "n.py"],
                             "bindings": {"x": {"files": ["a.py"], "listed": [], "probes": [], "registry": False},
                                          "y": {"files": ["b.py"], "listed": [], "probes": [], "registry": False}}}}
    entries["ky"] = {"files": ["b.py"], "present": ["n", "m"]}
    got = cli._read_sets(tmp_path, {"y": [1, 2]}, ids, keys, group)["reads"]
    assert got["loaded"] == ["m.py", "n.py"] and got["bindings"]["y"]["probes"] == [] and got["incomplete"] == {"y": ["Z_v1"]}


def test_a_binding_s_name_is_its_root_function_its_files_models_or_its_template_entry():
    m = "verity.circuits.ops.boolean.gf2k.circuit-check#"
    assert [SH.binding(u) for u in ((f"{m}_gf2k_roots",), (f"{m}_x_roots", f"{m}_x_bindings"), ("_kernel_roots",),
                                    ("_vocabulary_roots", "_vocabulary_bindings"), (f"{m}_model_roots:Gf2kMul_v1",),
                                    ("_model_roots:SynOther_v1",), (f"{m}TEMPLATE_PARAMS:token-select-combine",),
                                    ("TEMPLATE_PARAMS:rmsnorm-cuda-vec8",), ("<registered>",))] == [
        f"{m}_gf2k_roots", f"{m}_x_roots", "_kernel_roots", "_vocabulary_roots", f"{m}_model_roots", "_model_roots",
        f"{m}TEMPLATE_PARAMS:token-select-combine", "TEMPLATE_PARAMS", "<registered>"]


def test_a_template_entry_at_a_name_is_a_binding_of_its_instances_alone():
    """``<template>@<name>`` entries (`targets.template_params`) split one template's instances into bindings of their own:
    each GLM-5.3 RMSNorm launch is one."""
    m = "verity_flock.subcircuit_templates.circuit-check#TEMPLATE_PARAMS:"
    got = {label: sorted(TG.BINDINGS[label]) for label, sub, _ in TG.templates() if sub.template.id == "rmsnorm-cuda-vec8"}
    assert got == {"rmsnorm-cuda-vec8/n6144-b768-eps1e-05/bf16": [f"{m}rmsnorm-cuda-vec8@b768"],
                   "rmsnorm-cuda-vec8/n6144-b256-eps1e-05/bf16": [f"{m}rmsnorm-cuda-vec8@b256"]}
    assert [SH.binding((u,)) for (u,) in got.values()] == [f"{m}rmsnorm-cuda-vec8@b768", f"{m}rmsnorm-cuda-vec8@b256"]
    assert TG.template_of("rmsnorm-cuda-vec8@b256") == TG.template_of("rmsnorm-cuda-vec8") == "rmsnorm-cuda-vec8"


def test_the_shard_module_is_outside_the_checker_s_version(tmp_path, monkeypatch):
    """`cache.SHARD` decides which targets a run checks, never what a check computes: an edit to it reruns no target."""
    monkeypatch.setattr(CC, "_HERE", tmp_path)
    monkeypatch.setattr(CC, "SHARD", tmp_path / "shard.py")
    (tmp_path / "checks.py").write_text("def f(x):\n    return 2 * x\n")
    (tmp_path / "shard.py").write_text("def g(x):\n    return x % 4\n")
    here = CC.version()
    (tmp_path / "shard.py").write_text("def g(x):\n    return x % 5\n")
    assert CC.version() == here
    (tmp_path / "checks.py").write_text("def f(x):\n    return 3 * x\n")
    assert CC.version() != here


#: a small sidecar set: argmax's chunks and combines and the Boolean scalars, each checked as a Call (`--as-call`), so neither
#: builds the catalog
SIDECARS = [str(TG.REPO / "verity/circuits/ops/argmax.circuit-check.py"),
            str(TG.REPO / "verity/circuits/ops/boolean/scalar.circuit-check.py")]


def test_three_shards_of_a_sidecar_set_combine_to_the_unsharded_run(tmp_path, capsys):
    """The real checks: the combined report's summary, its output and its exit status are the unsharded run's, byte for byte.
    Uncached: a cache would trace this test process for good (`cache.trace`)."""
    out = tmp_path / "r.json"
    base = [*SIDECARS, "--as-call", "--no-cache", "-q", "-j", "2"]
    whole = cli.main([*base, "--out", str(out)])
    said = capsys.readouterr().out
    rep = json.loads(out.read_text())
    out.unlink()
    assert len(rep["targets"]) >= 8
    dirs, _ = _shards(capsys, tmp_path, 3, base)
    combined = cli.main(["--combine", *map(str, dirs[::-1]), "-q", "--out", str(out)])
    assert (combined, capsys.readouterr().out) == (whole, said)
    got = json.loads(out.read_text())
    assert json.dumps(got["summary"]) == json.dumps(rep["summary"])
    assert [(t["kind"], t["target"], t["failures"]) for t in got["targets"]] == [(t["kind"], t["target"], t["failures"])
                                                                                 for t in rep["targets"]]


def _py(root: Path, harness: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", harness, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


#: `test_units`' ``--records-only`` with each binding's slice and the shared part beside the tree (`cli.records`' ``slices``)
SLICED_RECORDS = RECORDS_ONLY.replace("jobs=2)", "jobs=2, slices=root.parent / f'{root.name}-slices')")


def _slices(root: Path) -> tuple[dict, Path]:
    """``root``'s records into its cache and its slices beside it (`SLICED_RECORDS`): the report, the slices' directory."""
    assert SLICED_RECORDS.count("slices=") == 1
    got = _py(root, SLICED_RECORDS)
    assert got.returncode == 0, got.stderr
    rep, cut = json.loads(got.stdout.strip().splitlines()[-1]), root.parent / f"{root.name}-slices"
    assert "why" not in rep and rep["slices"]["bindings"] == len(json.loads((cut / "index.json").read_text())["bindings"]), rep
    for f in cut.glob("*.json"):
        assert f.read_bytes() == json.dumps(json.loads(f.read_bytes()), sort_keys=True, separators=(",", ":")).encode(), f
    return rep, cut


def _digests(cut: Path) -> dict[str, str]:
    """Each binding's slice's digest by name, and the shared part's (``<shared>``): what a graph keys each binding's task on."""
    index = json.loads((cut / "index.json").read_text())
    return {name: hashlib.sha256((cut / f).read_bytes()).hexdigest()
            for name, f in [("<shared>", index["shared"]), *index["bindings"].items()]}


@pytest.mark.slow
def test_one_binding_s_edit_changes_its_own_slice_alone(tmp_path):
    """The slices are a function of the tree's bytes alone: two trees of the same bytes at two paths cut the same bytes.  An
    edit to a file only ``_xor_roots`` reads (its Definition's body) changes that binding's slice and no other's, nor the
    shared part: a graph keying each binding's task on its slice reruns that binding's alone.  ``_model_roots``' binding asks
    `inspect` for the module of its frame, as torch's import does, which looks for the source of every module loaded, that
    file's among them: its record holds that module by whether it is there (``present``, `units._unchanged`), so its slice is
    unchanged too, and the edited tree's records cut from the first tree's rebuild ``_xor_roots``' alone, into the same slices
    as from none."""
    trees = {}
    for name in ("a", "b", "c", "warm"):
        trees[name] = _tree(tmp_path / name)
        _bindings(trees[name], *UNIT_BINDINGS)
        _edit(trees[name], "syn/catalog.py", "def model():\n",
              "def model():\n    import inspect\n    import sys\n\n    inspect.getmodule(sys._getframe())\n")
    for name in ("b", "warm"):
        _edit(trees[name], "syn/xdef.py", "B.call(SynXor, a, b)", "B.call(SynXor, b, a)")
    cuts = {name: _slices(root)[1] for name, root in trees.items() if name != "warm"}
    got = {name: _digests(cut) for name, cut in cuts.items()}
    index = json.loads((cuts["a"] / "index.json").read_text())["bindings"]
    (model,) = json.loads((cuts["a"] / index["_model_roots"]).read_text())["entries"]
    assert model["reads"]["present"].get("syn.xdef") is True, model["reads"]["present"]
    assert got["a"] == got["c"] and len(got["a"]) == 6, got
    assert {k for k in got["a"] if got["a"][k] != got["b"][k]} == {"_xor_roots"}, got
    shutil.copytree(trees["a"] / "cache", trees["warm"] / "cache")
    rep, cut = _slices(trees["warm"])
    assert rep["rebuilt"] == ["_xor_roots"] and _digests(cut) == got["b"], (rep, _digests(cut))
