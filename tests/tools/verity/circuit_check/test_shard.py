"""``circuit-check --shard K/N`` and ``--combine`` (`circuit_check.shard`): what a shard spec, a shard's failures and ``--combine``
refuse, and the shard options refused where they mean nothing."""
from __future__ import annotations

import hashlib
import json
import shutil
import types
from pathlib import Path

import pytest

from circuit_check import cli
from circuit_check import shard as SH


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


def _shards(capsys, tmp_path: Path, n: int, base: list[str]) -> tuple[list[Path], list[int]]:
    dirs, codes = [], []
    for k in range(1, n + 1):
        d = tmp_path / f"shard{k}"
        codes.append(cli.main([*base, "--shard", f"{k}/{n}", "--shard-dir", str(d)]))
        dirs.append(d)
    capsys.readouterr()
    return dirs, codes


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
