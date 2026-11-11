"""`served_piece.sh`, PoUS's piece of a served window (served_verdicts.sh's POUS_CMD), on a CPU: `lake`, `gpu-lease`
and the piece's Python are stand-ins on PATH, the library is one named already (P2_LIB) and HOME holds no elan. The two
Lean verifiers that decide, `pous-run-protocol` and `remote-audit-p2-protocol`, are built in a scratch copy of
``verity/`` before the card is asked for and go to `served_piece.py` as ``--case-exe`` and ``--bw-case-exe``; ones
named already (POUS_EXE, BW_CASE_EXE) are passed as they are and nothing is built; one that doesn't build is no answer,
and neither the card nor the piece is started."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "benchmarks" / "pous" / "served_piece.sh"
CHECK, AUDIT = "verity/protocols/pous/verifier/check", "verity/protocols/poub/verifier/lean"
RUN_EXE, BW_EXE = "pous-run-protocol", "remote-audit-p2-protocol"

LAKE = """#!/usr/bin/env bash
echo "lake $PWD $* $([ -f lakefile.toml ] || [ -f lakefile.lean ] && echo package)" >> "$FAKE_LOG"
[ "$2" = "${FAKE_LAKE_FAILS:-}" ] && { echo "error: $2 has a hole"; exit 1; }
mkdir -p .lake/build/bin && printf '#!/bin/sh\\n' > ".lake/build/bin/$2" && chmod +x ".lake/build/bin/$2"
"""
LEASE = """#!/usr/bin/env bash
echo "lease $*" >> "$FAKE_LOG"
while [ "$1" != -- ]; do shift; done
shift
GPU_LEASE_UUID=GPU-own0 exec "$@"
"""
PYTHON = """#!/usr/bin/env bash
echo "piece ${CUDA_VISIBLE_DEVICES:-}" >> "$FAKE_LOG"
printf '%s\\n' "$@" > "$FAKE_ARGV"
for a in "$@"; do case $a in */.lake/build/bin/*) [ -x "$a" ] && echo "$a" >> "$FAKE_BUILT" ;; esac; done
exit 0
"""


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def node(tmp_path) -> dict[str, str]:
    """The environment of a node with a stand-in `lake` and `gpu-lease` first on PATH, a stand-in for the piece's
    Python, a library built already, and a HOME with no elan, so the stand-in `lake` is the one found."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "lake", LAKE)
    _exe(bin_ / "gpu-lease", LEASE)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GPU_LEASE_", "POUS_", "BW_", "CUDA_VISIBLE")) and k not in ("LEASE_MIN", "MEM_GB")}
    return env | {"PATH": f"{bin_}:{env['PATH']}", "HOME": str(tmp_path / "home"), "P2_LIB": str(tmp_path / "lib"),
                  "PY": str(_exe(tmp_path / "python", PYTHON)), "POUS_NODE": "gpu-node",
                  "FAKE_LOG": str(tmp_path / "log"), "FAKE_ARGV": str(tmp_path / "argv"),
                  "FAKE_BUILT": str(tmp_path / "built")}


def _piece(env: dict[str, str], d: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), str(d.parent / "serve-run"), "3", str(d)], env=env, cwd=ROOT,
                          capture_output=True, text=True)


def _log(env: dict[str, str]) -> list[list[str]]:
    log = Path(env["FAKE_LOG"])
    return [line.split() for line in log.read_text().splitlines()] if log.exists() else []


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_the_two_leans_are_built_in_a_scratch_copy_before_the_card_is_asked_for(node, tmp_path):
    """Each verifier is built by `lake build` in its package of a copy of ``verity/`` outside the tree, whose ``.lake``
    is never written; then the card is asked for, at the lease's defaults; and the piece's Python gets the two builds,
    there while it runs and removed with the scratch copy when it ends."""
    d = tmp_path / "pieces" / "pous"
    r = _piece(node, d)
    assert r.returncode == 0, r.stdout + r.stderr
    (_, check, *run), (_, audit, *bw), lease, piece = _log(node)
    assert run == ["build", RUN_EXE, "package"] and bw == ["build", BW_EXE, "package"]
    scratch = Path(check[:-len(CHECK)])
    assert check.endswith("/" + CHECK) and audit == str(scratch / AUDIT) and ROOT not in scratch.parents
    assert lease[:7] == ["lease", "1", "--wait", "--max-min", "90", "--mem-gb", "72"]
    assert piece == ["piece", "GPU-own0"]
    argv = Path(node["FAKE_ARGV"]).read_text().splitlines()
    exes = [str(scratch / CHECK / ".lake/build/bin" / RUN_EXE), str(scratch / AUDIT / ".lake/build/bin" / BW_EXE)]
    assert [_option(argv, "--case-exe"), _option(argv, "--bw-case-exe")] == exes
    assert Path(node["FAKE_BUILT"]).read_text().splitlines() == exes and not scratch.exists()
    assert argv[:4] == [str(ROOT / "benchmarks/pous/served_piece.py"), str(d.parent / "serve-run"), "3", str(d)]
    assert _option(argv, "--node") == "gpu-node" and (d / "lake-build.log").exists()


def test_leans_built_already_are_passed_as_they_are_and_nothing_is_built(node, tmp_path):
    """POUS_EXE and BW_CASE_EXE name the two verifiers: no `lake` runs, and in the caller's lease no card is asked
    for. With one of them named, only the other is built."""
    d = tmp_path / "pieces" / "pous"
    named = {"POUS_EXE": "/built/" + RUN_EXE, "BW_CASE_EXE": "/built/" + BW_EXE}
    r = _piece(node | named | {"GPU_LEASE_UUID": "GPU-serve0"}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _log(node) == [["piece", "GPU-serve0"]] and not (d / "lake-build.log").exists()
    argv = Path(node["FAKE_ARGV"]).read_text().splitlines()
    assert [_option(argv, "--case-exe"), _option(argv, "--bw-case-exe")] == list(named.values())

    Path(node["FAKE_LOG"]).unlink()
    r = _piece(node | {"POUS_EXE": named["POUS_EXE"], "GPU_LEASE_UUID": "GPU-serve0"}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    (_, audit, *bw), piece = _log(node)
    assert audit.endswith("/" + AUDIT) and bw == ["build", BW_EXE, "package"] and piece == ["piece", "GPU-serve0"]
    argv = Path(node["FAKE_ARGV"]).read_text().splitlines()
    assert _option(argv, "--case-exe") == named["POUS_EXE"]
    assert _option(argv, "--bw-case-exe") == audit + "/.lake/build/bin/" + BW_EXE


@pytest.mark.parametrize("exe, builds", [(RUN_EXE, 1), (BW_EXE, 2)])
def test_a_lean_that_does_not_build_is_no_answer_and_starts_neither_the_card_nor_the_piece(node, tmp_path, exe, builds):
    """Exit 7, saying which verifier doesn't build, with the build's last lines: the Python never decides in its place."""
    d = tmp_path / "pieces" / "pous"
    r = _piece(node | {"FAKE_LAKE_FAILS": exe}, d)
    assert r.returncode == 7 and f"POUS: no answer: {exe} doesn't build (lake-build.log)" in r.stdout
    assert f"error: {exe} has a hole" in r.stdout and f"error: {exe} has a hole" in (d / "lake-build.log").read_text()
    log = _log(node)
    assert [line[0] for line in log] == ["lake"] * builds and not Path(node["FAKE_ARGV"]).exists()
    assert not Path(log[0][1][:-len(CHECK)]).exists() and not (d / "answer.json").exists()


@pytest.mark.parametrize("env, want", [
    ({}, "120m"), ({"PIECE_TIMEOUT": "45m"}, "45m"), ({"POUS_TIMEOUT": "3h", "PIECE_TIMEOUT": "45m"}, "3h"),
    ({"POUS_CMD": ""}, "")])
def test_the_composed_run_gives_the_piece_time_for_its_leans(tmp_path, env, want):
    """`onsite_piece_commands` sets the piece's limit with its command, exported for `served_verdicts.py pieces`:
    POUS_TIMEOUT where set, else PIECE_TIMEOUT's, else its own default; with no PoUS piece it sets none."""
    tree, home = tmp_path / "tree", tmp_path / "home"
    (tree / "benchmarks/pous").mkdir(parents=True)
    (tree / "benchmarks/pous/served_piece.sh").write_text("")
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh/vy-auditor2-pous").write_text("")
    got = subprocess.run(["bash", "-uc", f'. {SCRIPT.parent / "onsite_pieces.sh"}; onsite_piece_commands; '
                          'exec env'], env={"PATH": os.defpath, "HOME": str(home), "TREE": str(tree)} | env,
                         check=True, capture_output=True, text=True).stdout.splitlines()
    assert [line for line in got if line.startswith("POUS_TIMEOUT=")] == ([f"POUS_TIMEOUT={want}"] if want else [])


def test_the_scratch_copy_holds_every_package_the_two_leans_require():
    """`served_piece.sh` copies ``verity/`` alone: each package the two verifiers' manifests require is a path
    requirement inside it, so the copy builds with no network and nothing of the tree outside it."""
    for package in (CHECK, AUDIT):
        manifest = json.loads((ROOT / package / "lake-manifest.json").read_text())
        for p in manifest["packages"]:
            assert p["type"] == "path", f"{package} requires {p['name']} from {p.get('url')}, which no copy holds"
            assert ROOT / "verity" in (ROOT / package / p["dir"]).resolve().parents, (package, p["name"], p["dir"])
