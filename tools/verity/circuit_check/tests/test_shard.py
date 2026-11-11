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
from circuit_check import units as UN
from test_cache import REPO, _bindings, _edit, _tree
from test_units import BINDINGS as UNIT_BINDINGS
from test_units import FORKED as UNITS_FORKED
from test_units import HARNESS as UNITS
from test_units import RECORDS_ONLY, _copied, _fresh, _records, _records_only, _words_tree
from test_units import _run as _run_units


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
                      (["--combine", "d", "--batch"], "six modes"),
                      (["--combine", "d", "--records-only"], "six modes"),
                      (["--all", "--cache", "c", "--records-only", "--bindings"], "six modes"),
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
                      (["--all", "--cache", "c", "--update-reads"], "--update-reads writes"),
                      (["--all", "--catalog-units", "--update-reads"], "--update-reads writes"),
                      (["--all", "--catalog-units", "--cache", "c", "--update-reads", "--shard", "1/3", "--shard-dir", "d"],
                       "--update-reads writes")):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err, argv
    _refused(capsys, [tmp_path / "nothing"], "did not finish")


def test_reads_holds_each_path_once_and_gives_each_binding_its_whole_read_set_back():
    """`cli.grouped` writes each path once, in the group of exactly the bindings whose read sets hold it, those of the first
    binding first, each binding's places as ranges, and `cli.expanded` gives every set back as it was, with its other fields."""
    regs = {"files": ["r.py"], "listed": [], "probes": ["q"]}
    sets = {"a": {"files": ["x.py", "y.py"], "listed": ["d"], "probes": ["p"], "registry": True},
            "b": {"files": ["x.py"], "listed": [], "probes": ["p", "z"], "registry": False, "untracked": ["lib.so"]},
            "c": {"files": [], "listed": [], "probes": [], "registry": False}}
    got = cli.grouped(sets)
    assert got["groups"] == [{"files": ["x.py"], "probes": ["p"]}, {"files": ["y.py"], "listed": ["d"]}, {"probes": ["z"]}]
    assert got["bindings"] == {"a": {"groups": "0-1", "registry": True},
                               "b": {"groups": "0,2", "registry": False, "untracked": ["lib.so"]},
                               "c": {"groups": "", "registry": False}}
    assert cli.expanded({"registries": regs, "loaded": ["m.py"], **json.loads(json.dumps(got))}) == \
        {"registries": regs, "loaded": ["m.py"], "bindings": sets}
    assert cli._ranges([0, 1, 2, 3, 7, 9, 10]) == "0-3,7,9-10" and cli._places("0-3,7,9-10") == [0, 1, 2, 3, 7, 9, 10]


def test_the_modules_whose_source_inspect_looked_for_are_every_binding_s_loaded_probes(tmp_path, monkeypatch):
    """`cli._read_sets` takes the modules whose source `inspect` looked for (``present``) out of each binding's own set: their
    paths, but the files the set holds by their bytes, are ``loaded``, every binding's, since torch's import asks for the
    source of every module the process loaded and which those are depends on how the run went.  A run reports what it holds
    beyond `cli.READS`' ``loaded`` as undeclared ``loaded``, and another binding's run that loaded the same module, nothing."""
    monkeypatch.setattr(cli, "READS", tmp_path / "reads")
    entries = {"kx": {"files": ["a.py"], "present": ["m", "own"]}, "ky": {"files": ["b.py"], "present": ["n"]}}
    where = {"m": "m.py", "n": "n.py", "own": "a.py"}
    monkeypatch.setattr(CC, "recorded", lambda cache, key, repo: entries[key])
    monkeypatch.setattr(CC, "declared", lambda root, reads: {
        "files": sorted(reads.get("files", ())), "listed": [], "probes": sorted(where[m] for m in reads.get("present", ())),
        "registry": False})
    group = {"files": [], "listed": [], "probes": []}
    ids, keys = [("definition", "X_v1"), ("definition", "Y_v1")], ["kx", "ky"]
    got = cli._read_sets(tmp_path, {"x": [0], "y": [1]}, ids, keys, group)
    assert got["reads"]["bindings"] == {"x": {"files": ["a.py"], "listed": [], "probes": [], "registry": False},
                                        "y": {"files": ["b.py"], "listed": [], "probes": [], "registry": False}}
    assert got["reads"]["loaded"] == ["m.py", "n.py"] and got["undeclared"]["x"] == {"files": ["a.py"], "loaded": ["m.py"]}
    cli.update_reads(got, every=True)
    assert cli._read_sets(tmp_path, {"x": [0], "y": [1]}, ids, keys, group)["undeclared"] == {}
    entries["ky"] = {"files": ["b.py"], "present": ["n", "m"]}
    assert cli._read_sets(tmp_path, {"y": [1]}, ids, keys, group)["undeclared"] == {}


def test_update_reads_adds_what_each_run_read_to_what_reads_declares(tmp_path, monkeypatch):
    """`cli.update_reads` adds what a run read to each binding's declared set, and to ``registries`` and ``loaded``: a
    binding's runs read otherwise by how they ran (``--binding`` makes only its own binding's roots), so its declaration holds
    each way.  A run of every binding drops a binding the catalog no longer has.  A probe is written as `cache.probed`
    declares it, one `READS` held already too."""
    monkeypatch.setattr(cli, "READS", tmp_path / "reads")

    def rep(bindings: dict, loaded: list[str], files: tuple[str, ...] = ("r.py",)) -> dict:
        return {"reads": {"registries": {"files": list(files), "listed": [], "probes": []}, "loaded": loaded,
                          "bindings": bindings}}

    a1 = {"files": ["x.py"], "listed": [], "probes": ["p"], "registry": False}
    a2 = {"files": ["y.py"], "listed": ["d"], "probes": [], "registry": True}
    b = {"files": ["z.py"], "listed": [], "probes": [], "registry": False}
    assert cli.update_reads(rep({"a": a1, "b": b}, ["m"]), every=True) == []
    cli.update_reads(rep({"a": a2}, ["n"], ("r.py", "s.py")), every=False)
    got = cli._reads_file()
    assert got["bindings"] == {"a": {"files": ["x.py", "y.py"], "listed": ["d"], "probes": ["p"], "registry": True}, "b": b}
    assert got["loaded"] == ["m", "n"] and got["registries"]["files"] == ["r.py", "s.py"]
    cli.update_reads(rep({"a": a1}, []), every=True)
    assert set(cli._reads_file()["bindings"]) == {"a"}
    under = {**b, "probes": ["no-such-dir/pins/X_v1.json", "no-such-dir/pins/Y_v1.json"]}
    cli.update_reads(rep({"b": under}, []), every=False)
    assert cli._reads_file()["bindings"]["b"]["probes"] == ["no-such-dir"]


def test_a_binding_s_name_is_its_root_function_its_files_models_or_its_template_entry():
    m = "verity.ml.circuits.ops.boolean.gf2k.circuit-check#"
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
SIDECARS = [str(TG.REPO / "verity/ml/circuits/ops/argmax.circuit-check.py"),
            str(TG.REPO / "verity/ml/circuits/ops/boolean/scalar.circuit-check.py")]


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


#: `test_units`' cached ``--all`` over the synthetic catalog as shard SHARD of 3 (None: unsharded), with each target
#: `cache.descriptor` describes (``described``) and, as its checks start (`cli._checked_all`), the ones described before them
#: (``before``) and the targets it checks (``checks``)
PROLOGUE = """
described, before, checks = [], [], []
desc, checked = CC.descriptor, cli._checked_all
CC.descriptor = lambda fn: described.append(fn.id) or desc(fn)
def entered(*a):
    before.extend(described)
    checks.extend(a[0][i][1] for i in (range(len(a[0])) if a[17] is None else a[17]))
    return checked(*a)
cli._checked_all = entered
"""
SHARDED = (UNITS.replace("rep = cli.run(", PROLOGUE + "rep = cli.run(").replace("units=True)", "units=True, shard=SHARD)")
           .replace('print(json.dumps({"stats"', 'print(json.dumps({"described": described, "before": before, "checks": checks, "stats"'))
UNIT_IDS = ["SynAdd_v1", "SynK_v1{K=2}", "SynOther_v1", "SynSub_v1", "SynTop_v1", "SynX_v1", "SynXor_v1"]


def test_a_shard_describes_only_its_own_targets_before_its_checks(tmp_path):
    """Describing every target (its key) before forking a worker is slow over the whole catalog, so a shard of ``--all``
    describes only its own.  A full build builds the whole catalog, as the ids come from it, describes only shard 1's targets
    before its checks and still stores every binding's record; shard 2, from those records, describes and makes only its own;
    shard 1 again reuses its task list and every target it checked, describing nothing.  Unsharded, the full build describes
    every target before its checks."""
    assert SHARDED.count("shard=SHARD") == 1 and SHARDED.count('"described": described') == 1
    own = {k: sorted(t for t in UNIT_IDS if SH.shard_of("definition", t, 3) == k) for k in (1, 2)}
    assert all(0 < len(o) < len(UNIT_IDS) for o in own.values())
    root = _tree(tmp_path / "t")
    _bindings(root, *UNIT_BINDINGS)
    first = _run_units(root, SHARDED.replace("SHARD)", "(1, 3))"))
    assert sorted(first["before"]) == sorted(first["checks"]) == sorted(first["values"]) == own[1]
    assert first["hits"] == [] and first["stats"]["units"] == {"stored": 5}
    second = _run_units(root, SHARDED.replace("SHARD)", "(2, 3))"))
    assert sorted(second["described"]) == sorted(second["checks"]) == sorted(second["values"]) == own[2]
    assert second["hits"] == [] and second["stats"]["units"] == {"reused": 5, "rebuilt": [], "stored": 0}
    again = _run_units(root, SHARDED.replace("SHARD)", "(1, 3))"))
    assert (again["described"], again["checks"], again["hits"]) == ([], [], own[1])

    whole = _tree(tmp_path / "w")
    _bindings(whole, *UNIT_BINDINGS)
    out = _run_units(whole, SHARDED.replace("SHARD)", "None)"))
    assert sorted(out["before"]) == sorted(out["checks"]) == UNIT_IDS and out["stats"]["units"] == {"stored": 5}


@pytest.mark.slow
def test_every_shard_from_a_copy_of_one_records_only_run_s_cache_builds_no_binding(tmp_path):
    """One ``--records-only`` run's cache, copied to three trees (three machines), gives each forked shard its targets with
    no binding built and no body built in its own process, and the shards' verdicts are together the full build's."""
    assert UNITS_FORKED.count("fork=True)") == 1
    root = _tree(tmp_path / "t")
    _bindings(root, *UNIT_BINDINGS)
    assert _records_only(root)["stored"] == 5
    got: dict = {}
    for k in (1, 2, 3):
        out = _run_units(_copied(root, tmp_path / f"s{k}"), UNITS_FORKED.replace("fork=True)", f"fork=True, shard=({k}, 3))"))
        assert out["stats"]["units"] == {"reused": 5, "rebuilt": [], "stored": 0} and out["here"] == [], out["stats"]["units"]
        assert sorted(out["values"]) == sorted(t for t in UNIT_IDS if SH.shard_of("definition", t, 3) == k)
        got.update(out["values"])
    assert got == _fresh(root)["values"]


#: `test_units`' setup, then ``circuit-check --all --bindings -j 2`` (`cli.bindings`) from the records in the tree's cache
BINDINGS_ONLY = UNITS[:UNITS.index("rep = cli.run(")] + 'print(json.dumps(cli.bindings(root / "cache", log=None, jobs=2)))\n'
#: `test_units`' bindings, each root function writing its units to ``<root>-roots.jsonl`` (beside the tree) as it runs
ROOTS_LOGGED = ("def bindings():\n    for units, make in catalog.BINDINGS:\n        yield units, make,",
                "def rooted(units, make):\n"
                "    def run():\n"
                "        with CC.internal(), open(root.parent / f'{root.name}-roots.jsonl', 'a') as f:\n"
                "            f.write(json.dumps(units) + '\\n')\n"
                "        return make()\n"
                "    return run\n\n\n"
                "def bindings():\n    for units, make in catalog.BINDINGS:\n        yield units, rooted(units, make),")


def _roots_made(log: Path) -> set[str]:
    return {SH.binding(json.loads(ln)) for ln in log.read_text().splitlines()} if log.exists() else set()


def _py(root: Path, harness: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", harness, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def _forked(name: str) -> str:
    """`test_units`' forked ``--all`` as ``--binding name`` runs it, its bindings' roots logged (`ROOTS_LOGGED`)."""
    # the real registry, which `targets.load_registries` loads before any check and the harness's stand-in doesn't: else the
    # first check to load it records the kernels it registers as reads this process can't digest
    return UNITS_FORKED.replace(*ROOTS_LOGGED).replace("fork=True)", f"fork=True, binding={name!r})").replace(
        "rep = cli.run(", "from verity.ml.definitions.registry import catalog as RC\nRC.load()\nrep = cli.run(", 1)


@pytest.mark.slow
def test_each_binding_from_a_copy_of_one_records_only_run_s_cache_checks_its_own_targets_and_together_the_list(tmp_path):
    """``--bindings`` names each catalog binding with the targets ``--binding`` checks under it: together the whole list, each
    target once.  Each forked ``--binding`` run, in a tree of its own from a copy of one ``--records-only`` run's cache (a task
    on another machine), builds no binding and no body in its own process, makes only its own binding's roots (the catalog's
    roots record lists the others'), keys and checks exactly its targets, with the verdicts a full build gives; run again, it
    reuses every target.  With the roots record stale (a module no binding reads edited) it makes every binding's roots, with
    the same verdicts.  With no record it refuses rather than build, and given a name no catalog binding has, rather than
    check nothing."""
    assert UNITS_FORKED.count(ROOTS_LOGGED[0]) == 1
    root = _tree(tmp_path / "t")
    _bindings(root, *UNIT_BINDINGS)
    cold = _py(root, UNITS.replace("units=True)", "units=True, binding='_core_roots')"))
    assert cold.returncode and "the bindings' records give no task list" in cold.stderr, cold.stderr
    assert _records_only(root)["stored"] == 5
    got = _py(root, BINDINGS_ONLY)
    assert got.returncode == 0, got.stderr
    named = json.loads(got.stdout.strip().splitlines()[-1])
    assert sorted(named["bindings"]) == ["<registered>", "_core_roots", "_model_roots", "_sized_roots", "_xor_roots"]
    assert sorted(label for ts in named["bindings"].values() for _, label in ts) == UNIT_IDS and named["root_errors"] == []
    values: dict = {}
    for i, (name, targets) in enumerate(named["bindings"].items()):
        tree = _copied(root, tmp_path / f"b{i}")
        out = _run_units(tree, _forked(name))
        assert out["stats"]["units"] == {"reused": 5, "rebuilt": [], "stored": 0} and out["here"] == [], (name, out["stats"])
        assert sorted(out["values"]) == sorted(label for _, label in targets) and out["hits"] == [], name
        assert _roots_made(tmp_path / f"b{i}-roots.jsonl") == {name} - {"<registered>"}, name
        assert len(out["keys"]) == len(out["values"]), (name, out["keys"])
        values.update(out["values"])
        if name == "_core_roots":
            again = _run_units(tree, _forked(name))
            assert again["stats"]["units"] == out["stats"]["units"] and again["here"] == [], again["stats"]
            assert again["hits"] == sorted(again["values"]) == sorted(out["values"]), again
    assert values == _fresh(root)["values"]
    stale = _copied(root, tmp_path / "stale")
    _edit(stale, "syn/idle.py", "B.call(SynAdd, a, a)", "B.call(SynAdd, b, b)")
    out = _run_units(stale, _forked("_core_roots"))
    assert out["stats"]["units"] == {"reused": 5, "rebuilt": [], "stored": 0} and out["here"] == [], out["stats"]
    assert _roots_made(tmp_path / "stale-roots.jsonl") == set(named["bindings"]) - {"<registered>"}
    assert out["values"] == {t: values[t] for _, t in named["bindings"]["_core_roots"]}
    unknown = _py(root, UNITS.replace("units=True)", "units=True, binding='_no_roots')"))
    assert unknown.returncode and cli.NO_BINDING in unknown.stderr, unknown.stderr


def _reads_run(name: str, write: bool = False) -> str:
    """`_forked` with `cli.READS` in the tree: prints the report's ``reads`` and ``undeclared`` too, with ``write`` once
    ``--update-reads`` wrote them."""
    return (_forked(name).replace("rep = cli.run(", "cli.READS = root / 'reads'\nrep = cli.run(", 1)
            .replace('print(json.dumps({"stats"', ("cli.update_reads(rep, every=False)\n" if write else "")
                     + 'print(json.dumps({"stats"', 1)
            .replace('"values": {', '"reads": rep["reads"], "undeclared": rep["undeclared"], "values": {', 1))


@pytest.mark.slow
def test_a_binding_s_report_holds_its_read_set_and_what_reads_does_not_declare_of_it(tmp_path):
    """A ``--binding`` run's report holds its binding's read set (`cli._read_sets`) beside the registries' (each module they
    imported: here all of ``syn``, which `load_registries` enumerates): its own targets' pins files, no other binding's, and
    the package listing every check holds.  With no `cli.READS` all of it is undeclared; once ``--update-reads`` wrote it,
    nothing is, and another binding's run written beside it keeps the first's: each of those runs makes only its own binding's
    roots, the others' are the catalog's roots record's (`units.plan`'s ``binding``).  A module the process loaded after the
    registries' inputs and before its checks, which a check's `inspect` lookup asks for, is undeclared in ``loaded``, the probes
    every binding's set holds (`cli.READS`), not in the binding's own set: which modules a process loaded depends on how it
    runs.  A module only one target's check imports, new, is
    undeclared for that binding alone; a module the registries import anew, for every binding, until it is written."""
    from test_cache import _pin

    root = _tree(tmp_path / "t")
    _bindings(root, *UNIT_BINDINGS)
    assert _records_only(root)["stored"] == 5
    pre = _copied(root, tmp_path / "pre")
    core = _run_units(root, _reads_run("_core_roots"))
    regs, mine = core["reads"]["registries"], core["reads"]["bindings"]["_core_roots"]
    assert {"syn/top.py", "syn/sub.py", "syn/xdef.py", "syn/catalog.py"} <= set(regs["files"]), regs
    assert mine["registry"] and "syn" in mine["listed"] and "syn/top.py" not in mine["files"], mine
    assert _pin("SynTop_v1") in mine["files"] and _pin("SynOther_v1") not in mine["files"], mine
    assert {"files", "listed", "registry"} <= set(core["undeclared"]["_core_roots"]), core["undeclared"]
    _run_units(root, _reads_run("_core_roots", write=True))
    again = _run_units(root, _reads_run("_core_roots"))
    assert again["undeclared"] == {} and again["hits"] == sorted(again["values"]), again["undeclared"]
    model = _run_units(root, _reads_run("_model_roots", write=True))
    assert _roots_made(tmp_path / "t-roots.jsonl") == {"_core_roots", "_model_roots"}
    theirs = model["reads"]["bindings"]["_model_roots"]
    assert _pin("SynOther_v1") in theirs["files"] and _pin("SynTop_v1") not in theirs["files"], theirs
    written = {k: v for part in ("registries.json", "loaded.json", "bindings.json")
               for k, v in json.loads((root / "reads" / part).read_text()).items()}
    assert cli.expanded(written) == {"registries": regs, "loaded": sorted({*core["reads"]["loaded"], *model["reads"]["loaded"]}),
                                     "bindings": {"_core_roots": mine, "_model_roots": theirs}}
    paths = [(k, p) for g in written["groups"] for k, ps in g.items() for p in ps]
    assert len(paths) == len(set(paths)) and set(paths) >= {("files", _pin("SynTop_v1")), ("files", _pin("SynOther_v1"))}
    # a module the process loaded before its checks, whose source `syn.check`'s `inspect` lookup asks for as torch's import
    # asks for every loaded module's: in a tree with the records alone, `loaded` holds it, and the binding's own set doesn't.
    # It is first imported after the registries' inputs are taken, in a recording no target holds, as making another
    # binding's roots imports what only that binding reads (one imported before them is the registries')
    shutil.copytree(root / "reads", pre / "reads")
    (pre / "synpre").mkdir()
    (pre / "synpre/__init__.py").write_text("")
    late = ("_inputs = cli.CC.registry_inputs\n\n\ndef _then(root):\n    got = _inputs(root)\n"
            "    with cli.CC.recording(merge=False):\n        import synpre  # noqa: F401\n    return got\n\n\n"
            "cli.CC.registry_inputs = _then\nrep = cli.run(")
    first = _run_units(pre, _reads_run("_core_roots").replace("rep = cli.run(", late, 1))
    own, u = first["reads"]["bindings"]["_core_roots"], first["undeclared"]["_core_roots"]
    assert (own["files"], own["listed"]) == (mine["files"], mine["listed"]) and "files" not in u, (own, u)
    assert "synpre/__init__.py" in u["loaded"] and "synpre/__init__.py" not in own["probes"], (own, u)
    (root / "synlib").mkdir()
    (root / "synlib/__init__.py").write_text("")
    (root / "synlib/helper.py").write_text("H = 1\n")
    # by a computed name: an import statement would be a site of `syn.top`, which the registries import
    _edit(root, "syn/top.py", "    from syn.shape import operands\n",
          "    from syn.shape import operands\n    __import__('synlib.helper')\n")
    core = _run_units(root, _reads_run("_core_roots"))
    (u,) = core["undeclared"].values()
    assert list(core["undeclared"]) == ["_core_roots"] and set(u) <= {"files", "probes"}, core["undeclared"]
    assert u["files"] == ["synlib/__init__.py", "synlib/helper.py"], u
    assert _run_units(root, _reads_run("_model_roots"))["undeclared"] == {}
    (root / "syn/fresh.py").write_text("Y = 1\n")
    model = _run_units(root, _reads_run("_model_roots"))
    assert model["undeclared"]["_model_roots"]["registries"]["files"] == ["syn/fresh.py"], model["undeclared"]


#: two bindings past `test_units`' whose roots register beyond themselves: ``_late_roots`` a primitive of a module only it
#: imports (`syn.late.lz`: the registries import `syn`'s top-level modules) and a member of a family its id makes
#: (`SynWide{n}`, `codec.register_lazy_family`); ``_minted_roots`` a composite its factory makes, which no id makes
LATE = {
    "syn/wide.py": """
from verity.ml.circuits.codec import register_lazy_family
from verity.ml.circuits.defs import PrimitiveDefinition

from syn.prims import V

_MADE = {}


def wide(n):
    if n not in _MADE:
        _MADE[n] = PrimitiveDefinition(f"SynWide{n}", 1, (("a", V), ("b", V)), V, lambda a, b: a | b)
    return _MADE[n]


register_lazy_family(r"^SynWide(\\d+)_v1$", lambda m: wide(int(m[1])))
""",
    "syn/late/__init__.py": "",
    "syn/late/lz.py": """
from verity.ml.circuits import primitive

from syn.prims import V


@primitive("SynLz", 1, [("a", V), ("b", V)], V)
def SynLz(a, b):
    return a & b
""",
    "syn/fresh.py": """
from verity.ml.circuits import composite

from syn.prims import SynAdd, V

_MADE = []


def made():
    if not _MADE:
        @composite("SynFresh", 1, [], lambda S: ((("a", V), ("b", V)), V))
        def SynFresh(B, S, a, b):
            return B.call(SynAdd, a, b)
        _MADE.append(SynFresh)
    return _MADE[0]
""",
}
LATE_ROOTS = ("\ndef late():\n    from syn import wide\n    from syn.late import lz\n    return [lz.SynLz, wide.wide(7)]\n\n\n"
              "def minted():\n    from syn import fresh\n    return [bind(fresh.made())]\n\n")
#: what the forked run's own process has registered of the tree's Definitions, in order, and imported of its modules
HELD_HERE = ('"here": here,', '"here": here, "registry": [f for f in dict.keys(__import__("verity.ml.circuits.defs", '
             'fromlist=["REGISTRY"]).REGISTRY.defs) if f.startswith("Syn")], "modules": sorted(m for m in sys.modules '
             'if m.startswith("syn")),')


def _late_tree(at: Path) -> Path:
    root = _tree(at)
    for rel, text in LATE.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text.lstrip("\n"))
    _edit(root, "syn/catalog.py", "\nBINDINGS = [", LATE_ROOTS + "\nBINDINGS = [")
    _bindings(root, *UNIT_BINDINGS, '(("_late_roots",), late)', '(("_minted_roots",), minted)')
    return root


@pytest.mark.slow
def test_a_binding_run_does_in_each_other_binding_s_place_what_making_its_roots_did_beyond_them(tmp_path):
    """A ``--binding`` run from a copy of one ``--records-only`` run's cache makes another binding's roots only where what
    making them did beyond them (the catalog's roots record notes it) can't be done in their place: ``_late_roots``' import of
    `syn.late.lz` and its ``SynWide7_v1``, made by its id, are done (`units._replay`), and ``_minted_roots``' composite, which
    no id makes, is made.  So this process's registry and its in-tree modules are, in order, the ones a run that makes every
    binding's roots has (the roots record stale), with that run's verdicts: Flock lowers a word primitive by the first
    Boolean Definition registered of that word, which another binding's roots can be the first to register."""
    root = _late_tree(tmp_path / "t")
    assert _records_only(root)["stored"] == 7
    lazy, stale = _late_tree(tmp_path / "lazy"), _late_tree(tmp_path / "stale")
    for tree in (lazy, stale):
        shutil.copytree(root / "cache", tree / "cache")
    _edit(stale, "syn/idle.py", "B.call(SynAdd, a, a)", "B.call(SynAdd, b, b)")
    harness = _forked("_core_roots").replace(*HELD_HERE)
    assert harness.count('"registry": [') == 1
    got, full = _run_units(lazy, harness), _run_units(stale, harness)
    assert _roots_made(tmp_path / "lazy-roots.jsonl") == {"_core_roots", "_minted_roots"}
    assert _roots_made(tmp_path / "stale-roots.jsonl") == {"_core_roots", "_model_roots", "_sized_roots", "_xor_roots",
                                                           "_late_roots", "_minted_roots"}
    assert got["registry"] == full["registry"] and got["modules"] == full["modules"], (got, full)
    assert [f for f in got["registry"] if f in ("SynLz_v1", "SynWide7_v1", "SynFresh_v1")] == \
        ["SynLz_v1", "SynWide7_v1", "SynFresh_v1"] and "syn.late.lz" in got["modules"], got
    assert got["stats"]["units"] == full["stats"]["units"] == {"reused": 7, "rebuilt": [], "stored": 0}, (got, full)
    assert got["here"] == full["here"] == [] and got["values"] == full["values"] and got["values"], (got, full)


#: `test_units`' ``--records-only`` with each binding's slice and the shared part beside the tree (`cli.records`' ``slices``)
SLICED_RECORDS = RECORDS_ONLY.replace("jobs=2)", "jobs=2, slices=root.parent / f'{root.name}-slices')")
#: the real registry loaded before the run, as `_forked` loads it
LOADED = ("rep = cli.run(", "from verity.ml.definitions.registry import catalog as RC\nRC.load()\nrep = cli.run(")


def _slices(root: Path) -> tuple[dict, Path]:
    """``root``'s records into its cache and its slices beside it (`SLICED_RECORDS`): the report, the slices' directory."""
    assert SLICED_RECORDS.count("slices=") == 1
    got = _py(root, SLICED_RECORDS)
    assert got.returncode == 0, got.stderr
    rep, cut = json.loads(got.stdout.strip().splitlines()[-1]), root.parent / f"{root.name}-slices"
    assert "why" not in rep and rep["slices"]["bindings"] == len(json.loads((cut / "index.json").read_text())["bindings"]), rep
    return rep, cut


def _sliced(name: str, cut: Path) -> str:
    """`_forked`'s ``--binding name`` from that binding's slice in ``cut`` and the shared part alone (`units.read_slice`)."""
    piece = cut / json.loads((cut / "index.json").read_text())["bindings"][name]
    return _forked(name).replace(f"binding={name!r})", f"binding={name!r}, sliced=UN.read_slice(Path({str(piece)!r}), "
                                 f"Path({str(cut / 'shared.json')!r}), {name!r}))", 1)


def _bare(root: Path, to: Path) -> Path:
    """A copy of ``root`` at ``to`` with no cache: a binding's task on a machine that holds none of the catalog's records."""
    shutil.copytree(root, to, ignore=shutil.ignore_patterns("cache"))
    return to


def _digests(cut: Path) -> dict[str, str]:
    """Each binding's slice's digest by name, and the shared part's (``<shared>``): what a graph keys each binding's task on."""
    index = json.loads((cut / "index.json").read_text())
    return {name: hashlib.sha256((cut / f).read_bytes()).hexdigest()
            for name, f in [("<shared>", index["shared"]), *index["bindings"].items()]}


@pytest.mark.slow
def test_each_binding_from_its_slice_alone_checks_what_it_checks_from_every_record(tmp_path):
    """``--records-only --slices`` cuts each binding's slice and the part they share (`units.slices`).  Each forked ``--binding
    --slice`` run, in a copy of the tree with no record at all (a task that reads its slice and the shared part alone), makes
    only its own binding's roots, holds its own record to its reads and stores no record, with the values and keys of the run
    from every record (`units.plan`'s ``binding``); together they are the targets, values and keys of a forked ``--all`` from
    the records.  An edit to a file only one binding reads leaves every other binding's slice counting, and that one's is
    refused; so is another binding's slice, and a slice cut with another shared part.  Only a run in a tree with no record
    shows that a slice holds all a binding's run reads of the catalog, and only the bindings' own runs that it gives their
    keys and verdicts."""
    root = _tree(tmp_path / "t")
    _bindings(root, *UNIT_BINDINGS)
    rep, cut = _slices(root)
    index = json.loads((cut / "index.json").read_text())
    assert rep["stored"] == 5 and sorted(index["bindings"]) == ["<registered>", "_core_roots", "_model_roots", "_sized_roots",
                                                                "_xor_roots"], (rep, index)
    whole = _run_units(_copied(root, tmp_path / "whole"), UNITS_FORKED.replace(*LOADED, 1))
    values, keys = {}, set()
    for i, name in enumerate(sorted(index["bindings"])):
        bare = _bare(root, tmp_path / f"s{i}")
        out = _run_units(bare, _sliced(name, cut))
        assert out["stats"]["units"] == {"reused": 1, "rebuilt": [], "stored": 0} and out["here"] == [], (name, out["stats"])
        assert _roots_made(tmp_path / f"s{i}-roots.jsonl") == {name} - {"<registered>"}, name
        assert _records(bare) == [] and out["hits"] == [] and len(out["keys"]) == len(out["values"]), name
        if name in ("_core_roots", "_xor_roots"):
            full = _run_units(_copied(root, tmp_path / f"r{i}"), _forked(name))
            assert (out["values"], out["keys"]) == (full["values"], full["keys"]), name
        assert not set(out["values"]) & set(values), name
        values.update(out["values"])
        keys.update(out["keys"])
    assert values == whole["values"] and keys == set(whole["keys"]), (values, whole["values"])

    edited = _bare(root, tmp_path / "edited")
    _edit(edited, "syn/xdef.py", "B.call(SynXor, a, b)", "B.call(SynXor, b, a)")
    kept = _run_units(edited, _sliced("_core_roots", cut))
    assert kept["values"] and kept["values"] == {t: values[t] for t in kept["values"]}, kept["values"]
    refused = _py(edited, _sliced("_xor_roots", cut))
    assert refused.returncode and "its record in the slice doesn't count here" in refused.stderr, refused.stderr
    with pytest.raises(UN.Stale, match="binding _core_roots's slice, not _xor_roots's"):
        UN.read_slice(cut / index["bindings"]["_core_roots"], cut / "shared.json", "_xor_roots")
    (other := tmp_path / "other.json").write_bytes(b" " + (cut / "shared.json").read_bytes())
    with pytest.raises(UN.Stale, match="cut with another shared part"):
        UN.read_slice(cut / index["bindings"]["_xor_roots"], other, "_xor_roots")


@pytest.mark.slow
def test_the_registered_roots_slice_holds_each_of_their_records(tmp_path):
    """Two registered roots take a record each (`units.registered_entries`), and the ``<registered>`` binding's slice holds both,
    each at its place: its ``--binding --slice`` run in a tree with no record holds each to its reads and checks what its run
    from every record checks, with the same keys.  The one-record tree of the test above can't tell a slice that holds only the
    first."""
    from test_units import TWO_REGISTERED

    root = _tree(tmp_path / "t")
    _bindings(root, *UNIT_BINDINGS)
    _edit(root, "syn/catalog.py", "BINDINGS = [", TWO_REGISTERED + "BINDINGS = [")
    _, cut = _slices(root)
    piece = json.loads((cut / json.loads((cut / "index.json").read_text())["bindings"]["<registered>"]).read_text())
    assert [x["at"] for x in piece["entries"]] == [[["<registered>"], "registered", str(i), i] for i in range(2)], piece
    out = _run_units(_bare(root, tmp_path / "s"), _sliced("<registered>", cut))
    full = _run_units(Path(shutil.copytree(root, tmp_path / "r")), _forked("<registered>"))
    assert out["stats"]["units"] == {"reused": 2, "rebuilt": [], "stored": 0} and out["values"], out["stats"]
    assert (out["values"], out["keys"]) == (full["values"], full["keys"]), (out["values"], full["values"])


@pytest.mark.slow
def test_a_binding_from_its_slice_registers_and_imports_what_it_does_from_every_record(tmp_path):
    """From its slice and the shared part alone, in a copy of the tree with no record, a binding's run does in each other
    binding's place what making its roots did beyond them where the shared part notes it, and makes them where that can't be
    done (``_minted_roots``' composite): this process's registry and in-tree modules are, in order, those of a run that makes
    every binding's roots, with its verdicts.  An edit to what one binding registers beyond its roots changes the shared part,
    so every slice, which holds the shared part's digest, and no binding's own part but that one's."""
    root = _late_tree(tmp_path / "t")
    rep, cut = _slices(root)
    assert rep["stored"] == 7, rep
    noted = {at[2] for at, *_ in json.loads((cut / "shared.json").read_text())["replay"]}
    assert {"_late_roots", "_minted_roots"} <= noted, noted
    lazy, stale = _bare(root, tmp_path / "lazy"), _late_tree(tmp_path / "stale")
    shutil.copytree(root / "cache", stale / "cache")
    _edit(stale, "syn/idle.py", "B.call(SynAdd, a, a)", "B.call(SynAdd, b, b)")
    got = _run_units(lazy, _sliced("_core_roots", cut).replace(*HELD_HERE))
    full = _run_units(stale, _forked("_core_roots").replace(*HELD_HERE))
    assert _roots_made(tmp_path / "lazy-roots.jsonl") == {"_core_roots", "_minted_roots"}
    assert got["registry"] == full["registry"] and got["modules"] == full["modules"], (got, full)
    assert got["values"] == full["values"] and got["values"] and got["here"] == full["here"] == [], (got, full)
    assert got["stats"]["units"] == {"reused": 1, "rebuilt": [], "stored": 0}, got["stats"]

    more = _late_tree(tmp_path / "more")
    _edit(more, "syn/late/lz.py", "def SynLz(a, b):\n    return a & b\n",
          'def SynLz(a, b):\n    return a & b\n\n\n@primitive("SynLz2", 1, [("a", V), ("b", V)], V)\ndef SynLz2(a, b):\n'
          "    return a | b\n")
    _, moved = _slices(more)
    before, after = _digests(cut), _digests(moved)
    assert before.keys() == after.keys() and all(before[k] != after[k] for k in before), (before, after)

    def own(d: Path) -> dict:
        return {name: {k: v for k, v in json.loads((d / f).read_text()).items() if k != "shared"}
                for name, f in json.loads((d / "index.json").read_text())["bindings"].items()}
    was, now = own(cut), own(moved)
    assert {name for name in was if was[name] != now[name]} == {"_late_roots"}, now["_late_roots"]


@pytest.mark.slow
def test_a_binding_from_its_slice_finds_the_boolean_definitions_and_statics_a_full_build_does(tmp_path):
    """From its slice and the shared part alone, in a copy of the tree with no record, a word primitive's check finds the
    Boolean Definition of its word that only another binding's body build registers (made here along the path that binding's
    record reaches it by: the shared part's ``words``) or a module registers as it is imported, and a reference two bindings
    bind at two statics is checked at the kept binding's: each with the value and key a full build gives."""
    root, here = _words_tree(tmp_path / "t"), _words_tree(tmp_path / "here")
    whole = _run_units(here, UNITS_FORKED.replace(", jobs=2, isolate=True, fork=True)", ")").replace(*LOADED, 1))
    _, cut = _slices(root)
    words = {w["id"]: w for w in json.loads((cut / "shared.json").read_text())["words"]}
    assert words["SynWBits_v1"]["via"][0][2] == "_wtop_roots" and words["SynWBits_v1"]["node"], words
    for i, (name, target) in enumerate((("_wprim_roots", "SynW_v1"), ("_vtop_roots", "SynV_v1"),
                                        ("_scale3_roots", "syn.scale"))):
        out = _run_units(_bare(root, tmp_path / f"s{i}"), _sliced(name, cut))
        assert target in out["values"] and out["values"] == {t: whole["values"][t] for t in out["values"]}, (name, out)
        assert set(out["keys"]) <= set(whole["keys"]) and len(out["keys"]) == len(out["values"]), name
    assert whole["values"]["SynW_v1"][1] and whole["values"]["SynV_v1"][1], whole["values"]


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
