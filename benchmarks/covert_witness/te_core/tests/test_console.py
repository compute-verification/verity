"""te_core.console cuts a panel to the panels route's limits, the first cut that fits, and names the cut in its note;
its PUT refuses, before any request, a panel te_core doesn't PUT, a te_core object that isn't its source's, and a te_core
tree that isn't the integrator's branch's head."""
import functools
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from te_core import algorithm_policy, console, formalism, ledger

P1 = ["start", "start_id", "size", "saturation", "curve", "policy", "basis", "target_pct", "compute_gates",
      "compute_frac", "loss", "loss_lo", "loss_hi", "record", "checkpoint", "method", "family"]


def p1(points: int, band: int) -> dict:
    rows = []
    for curve, policies in (("honest", [""]), ("covert", ["c=4096w k=10kB"]),
                            ("covert-policy", [f"c1e1{i}/k1e{i}" for i in range(band)])):
        for pol in policies:
            for i in range(points):
                loss = 1.2 * (1 - 0.0001 * i)
                rows.append(["34M", "s1", 4.5e7, 3.42, curve, pol, "target", None, 1e17 * (1 + i), None, loss, None,
                             None, f"record-{i:06d}", i, "ref-optim/muon", "full"])
    return {"id": "training/p1-loss-vs-compute", "title": "t", "kind": "table", "updated_at": "now", "source": "s",
            "note": "x" * 2000, "columns": P1, "rows": rows}


def test_a_small_panel_is_kept_whole_with_the_short_note():
    q = console.cut(p1(10, 2))
    assert len(q["rows"]) == 40 and "loss_lo" not in q["columns"]
    assert q["note"].startswith(console.SHORT["p1-loss-vs-compute"]) and len(q["note"]) <= console.NOTE_MAX


def test_a_large_panel_takes_the_first_cut_that_fits():
    q = console.cut(p1(2000, 9))
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX
    rows = [dict(zip(q["columns"], r)) for r in q["rows"]]
    assert "apart" in q["note"] and {r["curve"] for r in rows} >= {"honest", "covert"}
    ends = [r for r in rows if r["curve"] == "honest"]
    assert ends[0]["loss"] == pytest.approx(1.2) and ends[-1]["record"] == "record-001999"


def test_the_route_refuses_more_than_24_columns_and_a_repeated_name():
    """The panels route answers 400 to 25 columns ("columns must be 1 to 24 distinct names of up to 60 characters")."""
    p = p1(2, 1)
    extra = [f"c{i}" for i in range(25 - len(P1))]
    wide = p | {"columns": [*P1, *extra], "rows": [r + [0] * len(extra) for r in p["rows"]]}
    assert console.refused(wide) == ["its 25 columns aren't 1 to 24 distinct names of at most 60 characters"]
    assert not console.refused(wide | {"columns": wide["columns"][:-1], "rows": [r[:-1] for r in wide["rows"]]})
    assert console.refused(p | {"columns": [*P1[:-1], "start"]}) and console.refused(p | {"columns": []})


def test_the_route_refuses_a_source_over_300_characters():
    """The panels route answers 400 to it ("source must be 1 to 300 characters: a run, commit, PR, command or URL")."""
    p = p1(2, 1)
    assert not console.refused(p | {"source": "s" * 300})
    assert console.refused(p | {"source": "s" * 301}) == ["its source is 301 characters, not 1 to 300"]
    assert console.refused(p | {"source": ""}) == console.refused(p | {"source": None}) == [
        "its source is 0 characters, not 1 to 300"]


def test_the_route_refuses_a_title_over_120_characters():
    """The panels route answers 400 to it ("title must be 1 to 120 characters")."""
    p = p1(2, 1)
    assert not console.refused(p | {"title": "t" * 120})
    assert console.refused(p | {"title": "t" * 121}) == ["its title is 121 characters, not 1 to 120"]
    assert console.refused(p | {"title": ""}) == ["its title is 0 characters, not 1 to 120"]


def test_write_cuts_every_panel(tmp_path):
    from te_core.plots import NAMES
    for n in NAMES:
        p = p1(5, 1) | {"id": f"training/{n}"}
        (tmp_path / f"{n}.json").write_text(json.dumps(p))
    made = console.write(tmp_path, tmp_path / "console")
    assert sorted(made) == sorted(NAMES) and all((tmp_path / "console" / f"{n}.json").exists() for n in NAMES)


@pytest.mark.parametrize("pid, why", [
    ("training/x", "not a panel te_core PUTs"), ("training/frontier-branches", "retired: frontier-branches is stored"),
    *((f"training/{n}", "retired: te_core.plots' panels are off the console") for n in console.NAMES)])
def test_put_refuses_a_panel_te_core_doesnt_put_before_any_request(pid, why):
    with pytest.raises(SystemExit, match=f"console.put {pid}: {why}"):
        console.put({"id": pid})


def test_every_training_panel_is_one_put_puts():
    """The four training panels `put` PUTs: algorithm-policy's two, standard-workloads' and size-scaling (ladder-v3's),
    none of them retired."""
    assert set(console.PANELS) == {"training/algorithm-policy", "training/frontier-fit", "training/standard-workloads",
                                   "training/size-scaling"} and not set(console.PANELS) & set(console.RETIRED)


@pytest.mark.parametrize("patch, named", [
    (lambda: (formalism, "published", lambda *a, **k: None), "te_core.formalism.published isn't the function"),
    (lambda: (formalism, "published", functools.partial(formalism.published, reading="B")),
     "te_core.formalism.published isn't the function"),
    (lambda: (formalism, "published", functools.wraps(formalism.published)(lambda *a, **k: None)),
     "te_core.formalism.published isn't the function"),
    (lambda: (formalism, "P_CELL", False), "te_core.formalism.P_CELL is False, its source's True"),
    (lambda: (formalism.Measured, "gates", property(lambda self: 0.0)),
     "te_core.formalism.Measured.gates isn't the method"),
    (lambda: (algorithm_policy, "console", ledger), "te_core.algorithm_policy.console isn't te_core.console"),
    (lambda: (console, "put", lambda p: "200"), "te_core.console.put isn't the function")])
def test_put_refuses_a_wrapped_swapped_or_reset_te_core_object(monkeypatch, patch, named):
    """The front-doors ruling: a ruling reaches a published number only as a te_core VERSION, so a function replaced
    by a lambda, a partial or a wrapper that names it, a constant set at run time, a method, a module's import of
    another's name, or `put` itself wrapped (as a scorer's wrapper did) stops the PUT."""
    put = console.put
    monkeypatch.setattr(ledger, "SHARED", {})
    monkeypatch.setattr(*patch())
    assert any(b.startswith(named) for b in console.replaced())
    with pytest.raises(SystemExit, match="te_core objects aren't their source's"):
        put({"id": "training/algorithm-policy"})


def test_a_clean_te_core_has_nothing_replaced_and_its_put_fetches_the_branchs_head_before_any_request():
    """In a process of its own, so no other test's state reaches it: every te_core module imported (torch's wrapping of
    a dispatch mode's `__torch_dispatch__` and the standard library's decorators are the source's), `replaced` finds
    nothing, and a PUT goes on to fetch `BRANCH` from origin, which here fails and refuses the PUT."""
    script = """
import importlib, pkgutil, subprocess, sys
import te_core
from te_core import console, ledger
for m in pkgutil.walk_packages(te_core.__path__, "te_core."):
    if ".tests" not in m.name:
        importlib.import_module(m.name)
print(console.replaced())
calls = []
subprocess.run = lambda argv, **k: calls.append(argv) or subprocess.CompletedProcess(argv, 1, "", "no network")
ledger.SHARED.clear()
try:
    console.put({"id": "training/algorithm-policy"})
except SystemExit as e:
    print(e)
print(calls)
"""
    here = Path(console.__file__).resolve().parent
    out = subprocess.run([sys.executable, "-c", script], cwd=here.parent, capture_output=True, text=True, check=True)
    lines = out.stdout.splitlines()
    assert lines[0] == "[]", out.stdout + out.stderr
    assert lines[1].startswith(f"console.put training/algorithm-policy: te_core's tree can't be held to {console.BRANCH}")
    assert lines[2] == str([["git", "-C", str(here), "fetch", "--quiet", "origin",
                             f"+refs/heads/{console.BRANCH}:refs/remotes/origin/{console.BRANCH}"]])


def test_off_branch_names_uncommitted_and_untracked_files_and_the_branchs_files_head_lacks_or_changes(tmp_path):
    """A te_core tree is the branch's when it is clean and has every file of te_core at the branch's head; a file
    only it has is its publisher's own."""
    repo, ref = tmp_path / "repo", "refs/remotes/origin/b"
    te = repo / "cw" / "te_core"
    te.mkdir(parents=True)

    def git(*a):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                        "-c", "core.hooksPath=/dev/null", *a], check=True, capture_output=True)

    def commit():
        git("add", "-A")
        git("commit", "-qm", "c")

    (te / "a.py").write_text("A = 1\n")
    (te / "b.py").write_text("B = 1\n")
    git("init", "-q")
    commit()
    git("update-ref", ref, "HEAD")
    assert console.off_branch(ref, te) == []
    (te / "own.py").write_text("OWN = 1\n")
    assert console.off_branch(ref, te) == ["uncommitted changes or untracked files in cw/te_core: cw/te_core/own.py"]
    commit()
    assert console.off_branch(ref, te) == []
    (te / "a.py").write_text("A = 2\n")
    (te / "b.py").unlink()
    assert console.off_branch(ref, te)[0].startswith("uncommitted changes or untracked files in cw/te_core: ")
    commit()
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=10", ref], capture_output=True, text=True,
                          check=True).stdout.strip()
    assert console.off_branch(ref, te) == [f"2 files of cw/te_core at {ref} ({head}) differ here or are missing: a.py, b.py"]
