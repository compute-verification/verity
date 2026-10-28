"""Every suite that builds or runs a Lake package is one `check` runs after its Lean steps (`AFTER_LEAN`): Lake builds a
package's path requirements in their own `.lake`, so a `lake build` from the pytest step writes the verifier's build directory
while lean-build does (lean-build can find a `Flock` `.olean` gone while a suite builds sampled-proofs-verifier, which
requires flock, from the pytest step).

What says so is the suite's metadata alone, which every suite may read: no scan of test code tells a suite that builds the Lean
from one that only imports the modules that would (import reach from the tests to a module that runs `lake` flags several
times as many suites as run `lake`). A suite whose tests build a Lake package lists
its lakefile under `inputs` (what `lake` reads, which no tracer sees: the testing skill), and that is what this holds; a suite
that lists one only to read it is in `READS_ONLY`, with the reason."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("suites_after_lean", HERE / "suites.py")
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S                                     # dataclasses resolve their module by name
spec.loader.exec_module(S)

LAKEFILES = ("lakefile.toml", "lakefile.lean")
#: suites whose metadata names a Lake package that their tests only read, never build or run: suite name -> why
READS_ONLY = {
    "verity/core/protocols/pous": "test_pous_trusted hashes Security's lakefiles against the lock; it starts no program",
}


def lake_builds(suite) -> list[str]:
    """What in a suite's metadata says its tests build or run a Lake package: its directory is one (a lakefile in it), it lists
    one's lakefile under `[tool.verity.tests] inputs`, or it lists a path in a `.lake` directory under `inputs` or `artifacts`.
    Reading a package's sources, its manifest or a directory that holds one is not building it. Whether a suite's directory
    holds a lakefile is a stat, which the guard doesn't record: a lakefile added beside an unchanged pyproject is the nightly
    uncached check's to catch."""
    why = [f"{suite.path}/{f}" for f in LAKEFILES if (S.ROOT / suite.path / f).is_file()]
    why += [p for p in suite.inputs if Path(p).name in LAKEFILES]
    return why + [p for p in (*suite.inputs, *suite.artifacts) if ".lake" in Path(p).parts]


def test_every_suite_that_builds_a_lake_package_runs_after_the_lean_steps():
    missing = {s.name: why for s in S.discover()
               if (why := lake_builds(s)) and s.name not in S.AFTER_LEAN and s.name not in READS_ONLY}
    assert not missing, "\n".join(
        f"suite {name!r} builds or runs a Lake package ({', '.join(why)}), and check would run it in its pytest step, beside "
        f"lean-build: add {name!r} to AFTER_LEAN in tools/verity/check/suites.py (or, if its tests only read that package's "
        f"files, to READS_ONLY in {Path(__file__).name} with the reason)" for name, why in missing.items())


def test_after_lean_and_reads_only_name_only_suites_they_hold():
    suites = {s.name: s for s in S.discover()}
    assert not (stale := [n for n in S.AFTER_LEAN if n not in suites]), (
        f"AFTER_LEAN in tools/verity/check/suites.py names no suite: {stale} (a suite's name is its [project] name, else its path)")
    assert not (stale := [n for n in READS_ONLY if n not in suites or not lake_builds(suites[n]) or n in S.AFTER_LEAN]), (
        f"READS_ONLY in {Path(__file__).name} names a suite that is gone, declares no Lake package or is in AFTER_LEAN: "
        f"remove {stale}")


def test_a_lake_build_is_read_from_each_form_the_metadata_states_it_in(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ROOT", tmp_path)
    for d, f in (("port", "lakefile.toml"), ("old", "lakefile.lean"), ("lib", "lakefile.toml")):
        (tmp_path / d).mkdir()
        (tmp_path / d / f).write_text("")

    def suite(path: str, inputs: tuple = (), artifacts: tuple = ()):
        return S.Suite(path, path, (), inputs, artifacts=artifacts)

    assert lake_builds(suite("port")) == ["port/lakefile.toml"]
    assert lake_builds(suite("old")) == ["old/lakefile.lean"]
    assert lake_builds(suite("b", ("lib/lakefile.toml", "lib/Lib"))) == ["lib/lakefile.toml"]
    assert lake_builds(suite("b", ("lib/.lake/build/bin/x",))) == ["lib/.lake/build/bin/x"]
    assert lake_builds(suite("b", artifacts=("lib/.lake/build/bin/v",))) == ["lib/.lake/build/bin/v"]
    assert lake_builds(suite("b", ("lib", "lib/Lib/A.lean", "lib/lake-manifest.json", "lib/lean-toolchain", "lean-audit.json"))) == []
