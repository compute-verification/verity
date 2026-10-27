"""a5_tp.sh's store, on a dry run (``A5_DRY``: a scratch dir's sessions, nothing built or proved): each session's circuit.txt
files and lookup tables go to the store as one fixture/v1 tree keyed by SHA-512 (``sha512/<hex>``, the Lean verifier's
``--archive`` layout) through store_trees.sh, the tree preserved and the run labelled ``circuits`` with its art id, while $OUT keeps
the records and neither a circuit, a table nor a prover's file; and a run with no custody key fails instead of keeping nothing."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ART = "art:" + "ab" * 32
CLI = """#!/bin/bash
# stand-in for `research`: logs every call; `data put` copies the tree it was given and prints an artifact id
echo "$*" >> "$CALLS"
if [ "$1 $2" = "data put" ]; then
  while [ $# -gt 0 ]; do [ "$1" = --tree ] && cp -r "$2" "$PUT_TREES/"; shift; done
  echo '{"artifact": "%s"}'
fi
exit 0
""" % ART
VV = """#!/bin/bash
# stand-in for `verity-vllm`: `epoch custody-env` exports nothing
exit 0
"""


def _sha512(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def _job(tmp: Path, custody: bool = True) -> tuple[dict[str, str], dict[str, bytes]]:
    run, scratch, b = tmp / "research" / "runs" / "r1", tmp / "scratch", tmp / "bin"
    for d in (run, b, tmp / "puts", scratch / "glm" / "rank0" / "tables", scratch / "glm" / "rank0" / "archive" / "sha512"):
        d.mkdir(parents=True)
    if custody:
        c = tmp / "research" / "requests" / "r1" / "custody"
        c.mkdir(parents=True)
        (c / "store.toml").write_text("[remote]\n")
        (c / "cred.json").write_text(json.dumps({"AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": "s", "AWS_SESSION_TOKEN": "t"}))
    files = {"circuit.txt": b"g 1 2\n", "rank0/circuit.txt": b"g 3 4\n", "rank0/tables/exp2.u32": bytes(range(16))}
    for rel, data in files.items():
        (scratch / "glm" / rel).write_bytes(data)
    (scratch / "glm" / "rank0" / "archive" / "sha512" / _sha512(b"g 1 2\n")).write_bytes(b"g 1 2\n")
    (scratch / "glm" / "rank0" / "inst-4.bin").write_bytes(b"the prover's rows")
    (scratch / "glm" / "rank0" / "records.json").write_text("{}")
    for name, text in (("research", CLI), ("verity-vllm", VV)):
        (b / name).write_text(text)
        (b / name).chmod(0o755)
    env = {**os.environ, "RESEARCH_RUN_DIR": str(run), "RESEARCH_RUN_ID": "r1", "OUT": str(tmp / "out"), "A5_DRY": str(scratch),
           "PY": sys.executable, "VV": str(b / "verity-vllm"), "RESEARCH_CLI": str(b / "research"), "CALLS": str(tmp / "calls.txt"),
           "PUT_TREES": str(tmp / "puts")}
    return env, files


def _run(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(HERE.parent / "a5_tp.sh")], cwd=REPO, env=env, capture_output=True, text=True)


def test_a_dry_session_stores_its_circuits_and_tables_by_sha512_and_labels_the_run(tmp_path):
    env, files = _job(tmp_path)
    r = _run(env)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert f"STORED glm {ART} PRESERVED" in r.stdout and f"PRESERVED-CHECK ok {ART}" in r.stdout
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    out = tmp_path / "out"
    assert calls == [f"data put --kind fixture/v1 --tree {tmp_path}/scratch/store/glm --meta @{out}/store/meta-glm.json --preserve --json",
                     f"data preserved {ART}", f"data label r1 circuits {ART} --by proofs"]
    tree = tmp_path / "puts" / "glm"
    held = {p.relative_to(tree).as_posix(): p.read_bytes() for p in tree.rglob("*") if p.is_file()}
    assert held == {f"sha512/{_sha512(b)}": b for b in files.values()}
    meta = json.loads((out / "store" / "meta-glm.json").read_text())
    assert meta["files"] == {rel: _sha512(b) for rel, b in files.items()} and (meta["name"], meta["run"]) == ("a5-circuits/glm", "r1")
    kept = {p.relative_to(out / "glm").as_posix() for p in (out / "glm").rglob("*") if p.is_file()}
    assert kept == {"rank0/records.json"}
    assert not (tmp_path / "scratch" / "store").exists() and (tmp_path / "scratch" / "glm" / "circuit.txt").exists()


def test_a_run_with_no_custody_key_fails(tmp_path):
    env, _ = _job(tmp_path, custody=False)
    r = _run(env)
    assert r.returncode == 1 and "no custody key at" in r.stdout, (r.stdout, r.stderr)
    assert not (tmp_path / "calls.txt").exists()


def test_store_trees_refuses_a_label_key_without_its_asserter_before_any_put(tmp_path):
    env, _ = _job(tmp_path)
    (tmp_path / "ev").mkdir()
    env.update({"EV": str(tmp_path / "ev"), "LABEL_KEY": "circuits"})
    env.pop("LABEL_BY", None)
    r = subprocess.run(["bash", str(REPO / "integrations/vllm/verity_vllm/ops/store_trees.sh"), str(tmp_path / "scratch"), "glm"],
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "LABEL_BY" in r.stderr and not (tmp_path / "calls.txt").exists(), (r.stdout, r.stderr)
