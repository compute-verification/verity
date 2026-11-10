"""Neither Commit script publishes a value. `examples/vllm/commit.sh` and `a5_llama1b_h200.sh` copy a row dir's records into
$OUT, which `research run` publishes as the run's record: never a file of the row's commit/ (the Commit's token record and
sampled replays, the sealed replay bundle, the slim keep), nor the Match's token records and capture. Each script's `records`
runs over a row dir with those files planted, and the demo's Commit runs whole in a toy tree under a stub of vLLM's venv: it
puts nothing in the evidence store, and its bundle stays in the row dir."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
COMMIT_SH = ROOT / "examples" / "vllm" / "commit.sh"
A5_SH = ROOT / "benchmarks" / "one_stage" / "a5_llama1b_h200.sh"
ROW = "llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager"
#: what every file that holds values carries here, and no path can
PLANTED = "a planted request value"

BUNDLE, KEEP = "commit/replay_bundle_p0", "commit/replay_slim_p0"
#: a row dir's files by the stage that leaves them. The ones that hold values, which stay on the node:
PRIVATE = {
    "match": ("match/capture/tokens.json", "match/control/tokens.json", "match/capture/log.jsonl.gz"),
    "commit": ("commit/tokens_p0.json", f"{BUNDLE}/MANIFEST.json", f"{BUNDLE}/context.json", f"{BUNDLE}/store/salts/s0.bin",
               f"{BUNDLE}/weights/tree.json"),
    "replay": ("commit/sampled_replay_p0.json", f"{KEEP}/slim.json", f"{KEEP}/context.json", f"{KEEP}/store/raw/r0.bin"),
}
#: the Commit's other records under commit/, which no script copies either
UNDER_COMMIT = {"commit": ("commit/verdict.json", "commit/versions.json", "commit/runs.jsonl")}
#: the row's top-level records and logs, which both scripts copy
RECORDS = {"build": ("build_summary.json", "build.log"), "match": ("match.log",), "commit": ("verdict.json", "commit.log"),
           "replay": ("config_record.json", "replay.log")}
#: match/'s top-level files, which the demo's Commit copies too
MATCH = {"match": ("match/program.json",)}
STAGES = ("build", "match", "commit", "replay")

FAKE_CLI = '''import json, os, shutil, sys
from pathlib import Path
a = sys.argv[1:]
d, stage = Path(a[a.index("--sweep-dir") + 1]) / a[2], a[a.index("--stages") + 1]
for rel, text in json.loads(Path(os.environ["LEAVES"]).read_text())[stage].items():
    (d / rel).parent.mkdir(parents=True, exist_ok=True)
    (d / rel).write_text(text)
if stage == "replay":
    shutil.rmtree(d / "commit" / "replay_bundle_p0")
with open(d / "stages.txt", "a") as f:
    f.write(stage + " PASS\\n")
'''


def leaves() -> dict[str, dict[str, str]]:
    """Each stage's files with their text: ``PLANTED`` in the ones that hold values, the file's own name in the others."""
    return {s: {**{rel: PLANTED for rel in PRIVATE.get(s, ())},
                **{rel: rel for group in (UNDER_COMMIT, RECORDS, MATCH) for rel in group.get(s, ())}} for s in STAGES}


def names(*groups: dict[str, tuple[str, ...]]) -> list[str]:
    return [rel for group in groups for rels in group.values() for rel in rels]


def published(out: Path) -> list[str]:
    """$OUT's files by name, after holding that none carries a planted value (named, never printed)."""
    files = {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    leaked = sorted(name for name, data in files.items() if PLANTED.encode() in data)
    assert not leaked, f"values reached $OUT in {leaked}"
    return sorted(files)


def function(script: Path, name: str) -> str:
    """The script's shell function ``name``: its lines from ``name() {`` through the next line that is ``}`` alone."""
    lines = script.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    j = next(k for k in range(i, len(lines)) if lines[k] == "}")
    return "\n".join(lines[i:j + 1]) + "\n"


@pytest.mark.parametrize("script, copied", [(COMMIT_SH, names(RECORDS, MATCH)), (A5_SH, names(RECORDS))],
                         ids=["commit.sh", "a5_llama1b_h200.sh"])
def test_records_copies_no_file_of_the_commit_nor_the_matchs_tokens(tmp_path, script, copied):
    d, out, window = tmp_path / ROW, tmp_path / "out", tmp_path / "window"
    for rel, text in {"stages.txt": "commit PASS\n", **{k: v for files in leaves().values() for k, v in files.items()}}.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    out.mkdir()
    window.mkdir()
    code = f"D={shlex.quote(str(d))}; OUT={shlex.quote(str(out))}; WD={shlex.quote(str(window))}\n{function(script, 'records')}records\n"
    got = subprocess.run(["bash", "-c", code], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert published(out) == sorted([*copied, "stages.txt", "sizes.txt"])


def test_the_demos_commit_stores_nothing_and_leaves_its_bundle_on_the_node(tmp_path):
    tree, venv, bin_, out, root = tmp_path / "tree", tmp_path / "venv" / "bin", tmp_path / "bin", tmp_path / "out", tmp_path / "llama"
    for d in (tree / "integrations" / "vllm", venv, bin_):
        d.mkdir(parents=True)
    (tmp_path / "cli.py").write_text(FAKE_CLI)
    (tmp_path / "leaves.json").write_text(json.dumps(leaves()))
    research = 'echo "$*" >> "$CALLS"'
    python = (f'case "$1 $2" in\n"-m research") shift 2; {research}; exit 0;;\n'
              f'"-m verity_vllm.pipeline.cli") shift 2; exec {sys.executable} {tmp_path / "cli.py"} "$@";;\nesac\nexit 64')
    for path, body in ((venv / "python", python), (bin_ / "research", research)):
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "OUT": str(out), "RESEARCH_RUN_ID": "r-toy",
           "RESEARCH_RUN_DIR": str(tmp_path / "runs" / "r-toy"), "CALLS": str(tmp_path / "calls.txt"), "LEAVES": str(tmp_path / "leaves.json"), "ROW": ROW, "ROLE": "LLAMA32_1B",
           "REPO": "unsloth/Llama-3.2-1B", "REV": "9535bd9b", "VENV": str(venv.parent), "CUDA_HOME": str(tmp_path),
           "HF_HOME": str(tmp_path), "LLAMA_ROOT": str(root)}
    got = subprocess.run(["bash", str(COMMIT_SH)], cwd=tree, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    calls = [line.split()[0] for line in (tmp_path / "calls.txt").read_text().splitlines()]
    assert calls == ["keep"], f"commit.sh's `research` calls are its keep of the node's root alone, never a put: {calls}"
    assert got.returncode == 0, got.stdout + got.stderr
    assert published(out) == sorted([*names(RECORDS, MATCH), "stages.txt", "sizes.txt", "machine.txt", "steps.txt",
                                     "bundle.sha256", *(f"{s}.out" for s in STAGES)])
    d = root / "r-toy" / ROW
    assert all((d / rel).read_text() == PLANTED for rel in names(PRIVATE)), "the bundle, the keep and the token records stay in the row dir"
