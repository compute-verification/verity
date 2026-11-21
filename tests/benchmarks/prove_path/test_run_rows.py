"""run.sh's served arm refuses a window absent from this node before any build, and outside a research run writes
under the clone."""
from __future__ import annotations

import subprocess
from pathlib import Path

RUN_SH = Path(__file__).resolve().parents[3] / "benchmarks" / "prove_path" / "run.sh"
REPO = RUN_SH.parents[2]


def run_sh(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    unset = [a for v in ("PP_TP", "PP_SERVE_RUN", "PP_SERVE_ROWS", "OUT") for a in ("-u", v)]
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return subprocess.run(["env", *unset, f"RESEARCH_RUN_DIR={out}", *(f"{k}={v}" for k, v in env.items()), "bash", str(RUN_SH)],
                          cwd=REPO, capture_output=True, text=True)


def test_a_served_window_absent_from_this_node_is_refused_before_any_build(tmp_path):
    got = run_sh(tmp_path, PP_SERVE_RUN=str(tmp_path / "gone"), PP_SERVE_ROWS=str(tmp_path / "gone" / "windows" / "w0"))
    assert got.returncode == 8 and "no serve-openai run directory on this node" in got.stdout
    (serve := tmp_path / "serve").mkdir()
    got = run_sh(tmp_path, PP_SERVE_RUN=str(serve), PP_SERVE_ROWS=str(serve / "windows" / "w0"))
    assert got.returncode == 8 and "no sealed window row dir on this node" in got.stdout
    assert list((tmp_path / "out").iterdir()) == []


def test_outside_a_research_run_the_outputs_go_under_the_clone_and_say_so(tmp_path):
    """With neither OUT nor RESEARCH_RUN_DIR set, run.sh's outputs go to out/prove_path under the clone (git ignores out/), said
    in one line, and no variable is unbound."""
    lines = RUN_SH.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("REPO=$(pwd)"))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("mkdir -p $O"))
    got = subprocess.run(["env", "-u", "OUT", "-u", "RESEARCH_RUN_DIR", "bash", "-c",
                          "set -u\n" + "\n".join(lines[i:j + 1]) + '\necho "O=$O"'], cwd=tmp_path, capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert got.stdout.splitlines() == [f"OUT unset outside a research run: the outputs go to {tmp_path}/out/prove_path",
                                       f"O={tmp_path}/out/prove_path"]
    assert (tmp_path / "out" / "prove_path").is_dir()
