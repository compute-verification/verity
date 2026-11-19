"""run.sh's served arm refuses a window absent from this node before any build, settles its forged partition before the
builds, says in a line which arm PP_ONLY leaves out, and outside a research run writes under the clone."""
from __future__ import annotations

import json
import subprocess
import sys
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


def _lines(first: str, last: str) -> list[str]:
    lines = RUN_SH.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(first))
    return lines[i:next(k for k in range(i, len(lines)) if lines[k].startswith(last)) + 1]


def test_the_served_arms_forged_partition_is_settled_before_any_build(tmp_path):
    """A served window of steps 1 to 2 has no unit at step 24: the served arm's forged partition is derived from the honest
    one, or a given one outside its steps is refused (exit 8, the fix named), after the install and before the builds,
    where its commit would raise only after the honest arm ran."""
    text = RUN_SH.read_text()
    assert text.index("forged-partition --honest") < min(text.index("lake build"), text.index("60-circuit.sh"))
    assert text.index("PY=$REPO/.venv/bin/python") < text.index("forged-partition --honest")
    assert 'run $1-forged "$src" - $O/forged-partition.json' in text
    snippet = "\n".join(_lines("PLAIN=(); SAMPLING_DESIGN=", "fi"))
    part = REPO / "benchmarks/prove_path"

    def served(**env: str) -> subprocess.CompletedProcess:
        unset = [a for v in ("PP_TP", "PP_PLAIN", "PP_PARTITION", "PP_FORGED_PARTITION", "PP_ONLY") for a in ("-u", v)]
        return subprocess.run(["env", *unset, f"PY={sys.executable}", f"O={tmp_path}", f"PP_SERVE_RUN={tmp_path}",
                               *(f"{k}={v}" for k, v in env.items()), "bash", "-c", snippet + "\necho settled"],
                              cwd=REPO, capture_output=True, text=True)

    got = served(PP_PLAIN="1", PP_PARTITION=str(part / "llama32_1b_silu_mul_v2_i8192_l0_step1_all_stratified.json"))
    assert got.stdout.strip() == "settled", got.stderr
    assert json.loads((tmp_path / "forged-partition.json").read_text()) == json.loads(
        (part / "llama32_1b_silu_mul_v2_i8192_l0_step1_all_stratified.json").read_text())
    got = served(PP_FORGED_PARTITION=str(part / "llama32_1b_silu_mul_v2_i8192_l0_step24.json"))
    assert got.returncode == 8 and "settled" not in got.stdout
    assert "outside the honest partition's [1, 2]" in got.stderr and "Unset PP_FORGED_PARTITION" in got.stderr
    assert served(PP_ONLY="honest", PP_FORGED_PARTITION=str(part / "llama32_1b_silu_mul_v2_i8192_l0_step24.json")).returncode == 0


def test_an_arm_pp_only_leaves_out_says_so_in_a_line_the_printer_reads(tmp_path):
    body = _lines("run() {", "  local adv=")[:-1]
    for only, name, args, word in (("honest", "served-forged", ' --forge 0:0', "forged"), ("forged", "served", "", "honest")):
        got = subprocess.run(["bash", "-c", "\n".join(body) + f'\n  echo ran\n}}\nPP_ONLY={only}\nrun {name} src - p ""{args}'],
                             capture_output=True, text=True)
        assert got.returncode == 0, got.stderr
        tag, _, doc = got.stdout.strip().partition(" ")
        assert tag == "PROVE-PATH" and json.loads(doc) == {"name": name, "ran": False, "why": f"PP_ONLY={only}: no {word} run"}
    got = subprocess.run(["bash", "-c", "\n".join(body) + '\n  echo ran\n}\nPP_ONLY=honest\nrun served src - p ""'],
                         capture_output=True, text=True)
    assert got.stdout.strip() == "ran"
