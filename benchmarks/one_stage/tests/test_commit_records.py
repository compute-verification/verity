"""A Commit run publishes only listed files. `research run` publishes what a script leaves under $OUT, its stdout and the
other files of its run dir, so `demos/vllm/commit.sh` and `a5_llama1b_h200.sh` put a row's file there only when
`demos/vllm/published.sh` lists it with the reason it holds nothing of a request, and name the files they leave on the node.
Here the list never matches a file known to hold a value; each script's `row` and `records` run over a stand-in row dir with
values planted in every file a stage leaves; and the demo's Commit runs whole in a toy tree under a stub of vLLM's venv, where
its run dir gains only listed files, its stdout no planted value, the evidence store nothing, and its bundle stays in the row
dir. A file the published set gains without an entry fails `unlisted`."""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PUBLISHED_SH = ROOT / "demos" / "vllm" / "published.sh"
COMMIT_SH = ROOT / "demos" / "vllm" / "commit.sh"
A5_SH = ROOT / "benchmarks" / "one_stage" / "a5_llama1b_h200.sh"
ROW = "llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager"
#: what every file a stage leaves carries here, and no path can
PLANTED = "a planted request value"

BUNDLE, KEEP = "commit/replay_bundle_p0", "commit/replay_slim_p0"
#: a row dir's files that hold values, by the stage that leaves them: the prompts and the access trace, the token records
#: and the capture, the Commit's records, bundle and keep
VALUES = {
    "match": ("match/spec.json", "match/accesses.jsonl.gz", "match/capture/tokens.json", "match/control/tokens.json",
              "match/capture/log.jsonl.gz"),
    "commit": ("commit/tokens_p0.json", "commit/verdict.json", "commit/versions.json", "commit/runs.jsonl",
               f"{BUNDLE}/MANIFEST.json", f"{BUNDLE}/context.json", f"{BUNDLE}/store/salts/s0.bin", f"{BUNDLE}/weights/tree.json"),
    "replay": ("commit/sampled_replay_p0.json", f"{KEEP}/slim.json", f"{KEEP}/context.json", f"{KEEP}/store/raw/r0.bin"),
}
#: the row's records and logs whose failing copy can quote a served token, and the ones that carry the requests' shape
QUOTING = {
    "build": ("build.log", "build_summary.json", "manifest.json", "manifest.log", "strict_word.json", "effective_config.json",
              "build_request_LP4096_T32/digest.txt"),
    "match": ("match.log", "match/run.json", "match/program.json", "match/instances.jsonl", "match/CARD.md", "match/resolution.md"),
    "commit": ("commit.log", "verdict.json"),
    "replay": ("replay.log", "config_record.json"),
}
#: the records every stage appends to
EVERY_STAGE = ("stages.txt", "row.log")
STAGES = ("build", "match", "commit", "replay")
#: a name with no entry, as a stage, a later change or a stray tool could leave it
STRAYS = ("stray.json", "match/stray.json")

FAKE_CLI = '''import json, os, shutil, sys
from pathlib import Path
a = sys.argv[1:]
d, stage = Path(a[a.index("--sweep-dir") + 1]) / a[2], a[a.index("--stages") + 1]
leaves, planted = json.loads(Path(os.environ["LEAVES"]).read_text()), os.environ["PLANTED"]
for rel, text in leaves[stage].items():
    (d / rel).parent.mkdir(parents=True, exist_ok=True)
    (d / rel).write_text(text)
if stage == "replay":
    shutil.rmtree(d / "commit" / "replay_bundle_p0")
with open(d / "row.log", "a") as f:
    f.write(planted + "\\n")
with open(d / "stages.txt", "a") as f:
    f.write(f"{stage} PASS 2026-01-01T00:00:00Z {planted}\\n{stage} T_r=17 2026-01-01T00:00:00Z {planted}\\n{planted}\\n")
# what `row run` leaves in its run dir (the one it is given, else the runner's) and a stage that writes into $OUT
run_dir = Path(a[a.index("--research-run-dir") + 1] if "--research-run-dir" in a else os.environ["RESEARCH_RUN_DIR"])
for rel in ("result.json", "evidence/row.log", f"serving-rows/{a[2]}/m0-private.bin"):
    (run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (run_dir / rel).write_text(planted)
if os.environ.get("STRAY_IN_OUT"):
    (Path(os.environ["OUT"]) / f"{stage}.stray").write_text(planted)
Path(os.environ["ARGS"]).write_text(json.dumps(a))
print(planted)
'''


def entries() -> list[tuple[str, str, str]]:
    """published.sh's list, as bash reads it: (whose, pattern, why) per entry."""
    got = subprocess.run(["bash", "-c", f'. {shlex.quote(str(PUBLISHED_SH))} && printf "%s\\n" "${{PUBLISHED[@]}}"'],
                         capture_output=True, text=True, check=True)
    return [tuple(line.split("|", 2)) for line in got.stdout.splitlines()]  # type: ignore[misc]


def has_entry(rel: str, whose: str | None = None) -> bool:
    return any(fnmatch.fnmatchcase(rel, pattern) for w, pattern, _ in entries() if whose in (None, w))


def row_records() -> list[str]:
    """The row dir's files the list names outright, which a stand-in row dir is given."""
    return sorted(pattern for whose, pattern, _ in entries() if whose == "row" and not re.search(r"[*?\[]", pattern))


def leaves() -> dict[str, dict[str, str]]:
    """Each stage's files with their text: ``PLANTED`` in the ones that hold values, the records that can quote one and a stray,
    and on the build the listed records, each holding its own name."""
    return {s: {**{rel: PLANTED for rel in (*VALUES.get(s, ()), *QUOTING.get(s, ()), *(STRAYS if s == "match" else ()))},
                **{rel: rel for rel in (row_records() if s == "build" else ())}} for s in STAGES}


def unlisted(published: Path) -> list[str]:
    """The published dir's files with no entry in published.sh, and any file at all that carries a planted value."""
    files = {p.relative_to(published).as_posix(): p for p in sorted(published.rglob("*")) if not p.is_dir()}
    return sorted(name for name, p in files.items()
                  if p.is_symlink() or not has_entry(name) or PLANTED.encode() in p.read_bytes())


def function(script: Path, name: str) -> str:
    """The script's shell function ``name``: its lines from ``name() {`` through the next line that is ``}`` alone."""
    lines = script.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    j = next(k for k in range(i, len(lines)) if lines[k] == "}")
    return "\n".join(lines[i:j + 1]) + "\n"


def stub_venv(tmp: Path) -> tuple[Path, Path, dict[str, str]]:
    """A venv whose python runs ``FAKE_CLI`` for the row CLI and records `research` calls, a `research` on PATH that does the
    same, and the environment both read."""
    venv, bin_ = tmp / "venv" / "bin", tmp / "bin"
    venv.mkdir(parents=True), bin_.mkdir()
    (tmp / "cli.py").write_text(FAKE_CLI)
    (tmp / "leaves.json").write_text(json.dumps(leaves()))
    research = 'echo "$*" >> "$CALLS"'
    python = (f'case "$1 $2" in\n"-m research") shift 2; {research}; exit 0;;\n'
              f'"-m verity_vllm.pipeline.cli") shift 2; exec {sys.executable} {tmp / "cli.py"} "$@";;\nesac\nexit 64')
    for path, body in ((venv / "python", python), (bin_ / "research", research)):
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    return venv, bin_, {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp), "CALLS": str(tmp / "calls.txt"), "ARGS": str(tmp / "args.json"),
                        "LEAVES": str(tmp / "leaves.json"), "PLANTED": PLANTED}


def run_function(tmp: Path, script: Path, body: str, **environ: str) -> tuple[subprocess.CompletedProcess, Path, Path, Path]:
    """``body`` under bash after published.sh and the script's `row` and `records`, over a row dir a stub's stages fill: the
    result, the row dir, $OUT and KEPT."""
    tree, base, out, run_dir = tmp / "tree", tmp / "base", tmp / "run" / "out", tmp / "run"
    d, kept, window = base / ROW, base / "unpublished", base / "serving-rows"
    for p in (tree / "adapters" / "vllm", d, out, kept, window):
        p.mkdir(parents=True)
    venv, _, env = stub_venv(tmp)
    q = shlex.quote
    code = (f". {q(str(PUBLISHED_SH))}\nTREE={q(str(tree))} PY={q(str(venv / 'python'))} ROW={ROW} ROLE=R REPO=r REV=0 BASE={q(str(base))}\n"
            f"D={q(str(d))} KEPT={q(str(kept))} WD={q(str(window))} FA3=a FA3G=b NT=c\n"
            f"{function(script, 'row')}{function(script, 'records')}{body}\n")
    got = subprocess.run(["bash", "-uc", code], env={**env, **environ, "OUT": str(out), "RESEARCH_RUN_DIR": str(run_dir)}, stdin=subprocess.DEVNULL,
                         capture_output=True, text=True)
    return got, d, out, kept


SCRIPTS = pytest.mark.parametrize("script", [COMMIT_SH, A5_SH], ids=["commit.sh", "a5_llama1b_h200.sh"])


def test_every_entry_says_why_and_none_matches_a_file_that_holds_a_value():
    es = entries()
    assert es and all(len(e) == 3 for e in es)
    for whose, pattern, why in es:
        assert whose in ("row", "run") and pattern and len(why.split()) >= 4, f"an entry is WHOSE|PATTERN|why: {whose}|{pattern}|{why}"
    private = [*(rel for group in (VALUES, QUOTING) for rels in group.values() for rel in rels), *EVERY_STAGE, *STRAYS,
               *(f"{s}.out" for s in STAGES), "sizes.txt", "result.json", "evidence/row.log", "evidence/match/run.json"]
    assert [rel for rel in private if has_entry(rel)] == []
    assert row_records(), "the stand-in row dir gets every record the list names"


def test_a_published_file_with_no_entry_is_found(tmp_path):
    """What the tests below hold a published dir to: a file with no entry, a link, or a listed file carrying a value."""
    (tmp_path / "match").mkdir()
    for rel, text in {"machine.txt": "a node", "stray.json": "{}", "match/spec.json": "{}", "steps.txt": PLANTED}.items():
        (tmp_path / rel).write_text(text)
    (tmp_path / "verdicts.txt").symlink_to(tmp_path / "stray.json")
    assert unlisted(tmp_path) == ["match/spec.json", "steps.txt", "stray.json", "verdicts.txt"]


@SCRIPTS
def test_a_stage_gets_a_run_dir_on_the_node_and_prints_verdict_words_alone(tmp_path, script):
    got, d, out, kept = run_function(tmp_path, script, "row build --stages build")
    assert got.returncode == 0, got.stderr
    args = json.loads((tmp_path / "args.json").read_text())
    assert args[args.index("--research-run-dir") + 1] == str(kept), "the stage's result and record copies go to KEPT, on the node"
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["out"], "and never into the run dir, which the run publishes"
    assert unlisted(out) == [] and sorted(p.name for p in out.iterdir()) == ["steps.txt", "verdicts.txt"]
    assert (out / "verdicts.txt").read_text() == "build PASS 2026-01-01T00:00:00Z\n"
    assert PLANTED in (kept / "build.out").read_text() and PLANTED not in got.stdout + got.stderr
    assert "build PASS 2026-01-01T00:00:00Z" in got.stdout.splitlines()


@SCRIPTS
def test_records_copies_only_listed_files_and_names_the_ones_it_leaves(tmp_path, script):
    body = "".join(f"row {s} --stages {s} > /dev/null\n" for s in STAGES) + "records\n"
    # only the demo's Commit holds all of $OUT to the list: a5's $OUT also keeps its audit's and its window's files
    holds = script == COMMIT_SH
    got, d, out, kept = run_function(tmp_path, script, body, **({"STRAY_IN_OUT": "1"} if holds else {}))
    assert got.returncode == 0, got.stderr
    assert unlisted(out) == []
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == sorted([*row_records(), "steps.txt", "verdicts.txt"])
    assert all((d / rel).is_file() for rel in leaves()["match"]), "a file left out stays where it was"
    assert all((kept / "out" / f"{s}.stray").is_file() for s in STAGES) == holds, "and one found in $OUT moves to KEPT"
    assert (out / "verdicts.txt").read_text() == "".join(f"{s} PASS 2026-01-01T00:00:00Z\n" for s in STAGES)
    left = next(line for line in got.stdout.splitlines() if line.startswith("not published (no entry in published.sh), left in " + str(d)))
    for rel in ("match/spec.json", "stray.json", "match/stray.json", "stages.txt", "row.log", "commit.log", "build_request_LPN_TN/digest.txt"):
        assert f" {rel} " in left + " ", f"the run names {rel} among the files it left: {left}"
    assert "4096" not in left.split(": ", 1)[1], "a number in a left file's name is masked: a row dir's names carry lengths"
    assert PLANTED not in got.stdout + got.stderr
    assert (kept / "sizes.txt").is_file() and not (out / "sizes.txt").exists()


def test_the_demos_commit_publishes_listed_files_alone_and_leaves_its_bundle_on_the_node(tmp_path):
    tree, run_dir, root = tmp_path / "tree", tmp_path / "runs" / "r-toy", tmp_path / "llama"
    (tree / "adapters" / "vllm").mkdir(parents=True)
    venv, _, env = stub_venv(tmp_path)
    env.update({"OUT": str(run_dir / "out"), "STRAY_IN_OUT": "1", "RESEARCH_RUN_ID": "r-toy", "RESEARCH_RUN_DIR": str(run_dir), "ROW": ROW, "ROLE": "LLAMA32_1B",
                "REPO": "unsloth/Llama-3.2-1B", "REV": "9535bd9b", "VENV": str(venv.parent), "CUDA_HOME": str(tmp_path),
                "HF_HOME": str(tmp_path), "LLAMA_ROOT": str(root)})
    got = subprocess.run(["bash", str(COMMIT_SH)], cwd=tree, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    calls = [line.split()[0] for line in (tmp_path / "calls.txt").read_text().splitlines()]
    assert calls == ["keep"], f"commit.sh's `research` calls are its keep of the node's root alone, never a put: {calls}"
    assert got.returncode == 0, got.stdout + got.stderr
    assert sorted(p.name for p in run_dir.iterdir()) == ["out"], "the run dir, which the run publishes beside $OUT, gains no other file"
    assert unlisted(run_dir / "out") == []
    assert sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()) == sorted(
        f"out/{rel}" for rel in [*row_records(), "machine.txt", "steps.txt", "verdicts.txt", "bundle.sha256"])
    assert PLANTED not in got.stdout + got.stderr, "the run's stdout is published too"
    d, kept = root / "r-toy" / ROW, root / "r-toy" / "unpublished"
    assert all((d / rel).read_text() == PLANTED for rels in VALUES.values() for rel in rels), "the prompts, the bundle, the keep and the token records stay in the row dir"
    on_node = [*(f"{s}.out" for s in STAGES), *(f"out/{s}.stray" for s in STAGES), "result.json", "evidence/row.log", "sizes.txt"]
    assert all((kept / rel).is_file() for rel in on_node), "each stage's output, what it wrote for the runner and what it left in $OUT stay on the node"
    assert sum(line.startswith("not published (no entry in published.sh), left in ") for line in got.stdout.splitlines()) == 2
