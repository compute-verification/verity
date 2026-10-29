"""``circuit-check --shard K/N`` and ``--combine`` (`circuit_check.shard`): which shard a target is, what a shard writes, what
``--combine`` refuses, and that a combined run's report, verdict and output are the unsharded run's."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import types
from pathlib import Path

import pytest

from circuit_check import cache as CC
from circuit_check import cli
from circuit_check import known as KN
from circuit_check import shard as SH
from circuit_check import targets as TG


def test_a_target_s_shard_is_its_id_s_hash_and_never_its_place():
    """Pinned: a change to the hash moves every target, which ci's shard plans and the measured walls assume it doesn't."""
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


#: the stub targets: a pass, a new and a known failure, a warning, a Boolean Call recomputing across units, a by-calls root whose
#: callee (another shard's at N = 3) fails with a known failure, so only the joined report fails the root, and three heavy ones
CALLEE, ROOT = "SynCallee_v1", "SynRoot_v1"
TARGETS = {
    "SynPass_v1": lambda: _result(),
    "SynNew_v1": lambda: _result([_fail("SynNew_v1", "lowering-mismatch")]),
    "SynKnown_v1": lambda: _result([_fail("SynKnown_v1", "dead")]),
    "SynWarn_v1": lambda: _result(warnings=[{**_fail("SynWarn_v1", "and", "redundant"), "count": 3}]),
    "SynAcross_v1": lambda: _result(partition={"q_word": 2, "recomputed_across": 5}),
    ROOT: lambda: _result(mode="by-calls", by_calls={"callees": [{"callee": "c0", "binding": CALLEE}]}),
    CALLEE: lambda: _result([_fail(CALLEE, "dead")]),
    "SynHeavyA_v1": lambda: _result(),
    "SynHeavyB_v1": lambda: _result(),
    "SynHeavyC_v1": lambda: _result(),
}
KNOWN = {"lowering/dead SynKnown_v1": "a known failure", f"lowering/dead {CALLEE}": "the callee's known failure",
         "lowering/dead SynGone_v1": "no longer fails: --all drops it"}
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
    monkeypatch.setattr(KN, "KNOWN", dict(KNOWN))
    monkeypatch.setattr(KN, "KNOWN_PIECES", {})
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
    for byte, every record but its timings) and exits as it does: here 1, on a new failure, a by-calls root whose callee is
    another shard's and a known entry that no longer fails."""
    out = tmp_path / "r.json"
    base = ["--all", "--no-cache", "-q", "-j", "2"]
    whole = cli.main([*base, "--out", str(out)])
    said = capsys.readouterr().out
    rep = json.loads(out.read_text())
    assert whole == 1 and "lowering/dead SynGone_v1" in said
    assert f"by-calls/callee-failed {ROOT}" in " ".join(rep["summary"]["new_failures"])
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
    assert [SH.shard_of("definition", t, 3) for t in (CALLEE, ROOT, "SynNew_v1")] == [1, 2, 3] and codes == [0, 0, 1]

    random.Random(3).shuffle(dirs)
    combined = cli.main(["--combine", *map(str, dirs), "-q", "--out", str(out)])
    assert (combined, capsys.readouterr().out) == (whole, said)
    got = json.loads(out.read_text())
    assert json.dumps(got["summary"]) == json.dumps(rep["summary"])
    assert _untimed(got) == _untimed(rep) and got["schema"] == rep["schema"] and got["options"] == rep["options"]
    assert got["combined"]["shards"] == 3 and sum(w["targets"] for w in got["combined"]["walls"]) == len(ids)


def test_a_shard_fails_on_its_own_new_failures_and_only_the_combine_on_the_joined_ones(stub, tmp_path, capsys, monkeypatch):
    """Without a new failure of its own, every shard passes, while the combined run fails as the unsharded one does: on the
    by-calls root whose callee is in another shard, and on the known entry that no longer fails, which only the whole list
    shows."""
    stub.tasks[:] = [t for t in stub.tasks if t[1] != "SynNew_v1"]
    base = ["--all", "--no-cache", "-q", "-j", "2"]
    whole = cli.main(base)
    said = capsys.readouterr().out
    dirs, codes = _shards(capsys, tmp_path, 3, base)
    assert codes == [0, 0, 0] and whole == 1
    assert (cli.main(["--combine", *map(str, dirs), "-q"]), capsys.readouterr().out) == (whole, said)
    monkeypatch.setitem(KN.KNOWN, f"by-calls/callee-failed {ROOT}", "the root's known join")
    monkeypatch.delitem(KN.KNOWN, "lowering/dead SynGone_v1")
    assert cli.main(base) == 0
    said = capsys.readouterr().out
    assert (cli.main(["--combine", *map(str, dirs), "-q"]), capsys.readouterr().out) == (0, said)


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
                      (["--combine", "d", "--batch"], "four modes")):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err, argv
    _refused(capsys, [tmp_path / "nothing"], "did not finish")


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
