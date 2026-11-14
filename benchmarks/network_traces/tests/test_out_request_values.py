"""No request value reaches a job's $OUT, which custody uploads. Each script's copy into $OUT runs on stand-in records,
with a prompt planted in every file a request's values are in (a rehearsal's plan.json, probe.json, io.json and
out-witness.jsonl, a network certifier rehearsal's traffic.json and client.json): $OUT gets the public records and no file holding the prompt or a request's id
fields."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROMPT = [918273, 827364, 736455]
NEEDLES = (b"918273", b"prompt_ids", b"streamed_ids")
LOG = 'log() { echo "$*" >> "$OUT/job.log"; }\n'


def block(script: str, first: str, last: str) -> str:
    """``script``'s lines from the one starting with ``first`` through the next one starting with ``last``."""
    lines = (HERE / script).read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(first))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith(last))
    return "\n".join(lines[i:j + 1]) + "\n"


def put(root: Path, files: dict[str, object]) -> None:
    for name, doc in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        text = doc if isinstance(doc, str) else "".join(json.dumps(d) + "\n" for d in doc) if name.endswith(".jsonl") \
            else json.dumps(doc)
        (root / name).write_text(text)


def run(code: str, **env: Path | str) -> subprocess.CompletedProcess:
    return subprocess.run(["env", *(f"{k}={v}" for k, v in env.items()), "bash", "-c", LOG + code],
                          capture_output=True, text=True)


def published(out: Path) -> list[str]:
    """$OUT's files by name, after holding that none carries a planted request value (named, never printed)."""
    files = {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    leaked = sorted(name for name, data in files.items() if any(n in data for n in NEEDLES))
    assert not leaked, f"request values reached $OUT in {leaked}"
    return sorted(files)


REHEARSAL = {
    "plan.json": {"windows": 2, "requests": [{"session": "s0", "id_request": {"prompt": PROMPT, "max_tokens": 4}}]},
    "probe.json": {"body": {"prompt": PROMPT}, "responses": [{"prompt_tokens": 3, "last": PROMPT[1]}]},
    "calibration.json": {"rho0": "1"},
    "honest/io.json": {"windows": 2, "requests": [{"user": "s0", "prompt": PROMPT, "output": [5, 6]}]},
    "honest/out-witness.jsonl": [{"window": 0, "rows": [PROMPT], "salts": ["00"]}],
    "honest/deployment.json": {"windows": 2},
    "honest/certified.jsonl": [{"window": 0, "head": "cd"}],
    "honest/out-records.jsonl": [{"window": 0, "root": "ef"}],
    "honest/check.json": {"accepted": True},
    "honest/client/certificates.jsonl": [{"window": 0}],
}


def test_chain_record_publishes_no_rehearsals_request_values(tmp_path):
    g, out = tmp_path / "G", tmp_path / "OUT"
    put(g / "g1" / "certifier-x", {"end.json": {"files": {}}, "request.json": {"args": ["--requests", "24"]},
                                   **{f"out/{k}": v for k, v in REHEARSAL.items()}})
    (out / "gates").mkdir(parents=True)
    got = run(block("chain_record.sh", '(cd "$G" && find', "  ! -name plan.json"), G=g, OUT=out)
    assert got.returncode == 0, got.stderr
    assert published(out) == sorted(f"gates/g1/certifier-x/{f}" for f in (
        "end.json", "request.json", "out/calibration.json", "out/honest/deployment.json", "out/honest/certified.jsonl",
        "out/honest/out-records.jsonl", "out/honest/check.json", "out/honest/client/certificates.jsonl"))


def test_n2_zk_publishes_a_network_certifier_runs_public_files_and_not_its_traffic_or_responses(tmp_path):
    r, out = tmp_path / "run", tmp_path / "OUT"
    public = {"deployment.json": {"windows": 2}, "cells.jsonl": [{"registration": {"root": "ab"}}],
              "certified.jsonl": [{"window": 0, "head": "cd"}], "check.json": {"accepted": True},
              "warden.json": {"process": {}}}
    put(r, {**public, "traffic.json": [{"at": 0, "body": {"prompt": PROMPT}}],
            "client.json": {"connections": [{"index": 0, "response": {"ids": PROMPT}}]}})
    out.mkdir()
    got = run(block("n2_zk.sh", 'mkdir -p "$OUT/run"', "done"), RUN=r, OUT=out)
    assert got.returncode == 0, got.stderr
    assert published(out) == sorted(f"run/{f}" for f in public)


def test_scan_zk_publishes_a_warden_rehearsals_public_files_and_not_its_traffic_or_responses(tmp_path):
    w, out = tmp_path / "W", tmp_path / "OUT"
    public = {"deployment.json": {"windows": 2}, "cells.jsonl": [{"registration": {"root": "ab"}}],
              "certified.jsonl": [{"window": 0, "head": "cd"}], "check.json": {"accepted": True},
              "warden.json": {"process": {}}, "serve.log": "serving\n"}
    put(w / "warden", {**public, "traffic.json": [{"at": 0, "body": {"prompt": PROMPT}}],
                       "client.json": {"connections": [{"index": 0, "response": {"ids": PROMPT}}]}})
    out.mkdir()
    got = run(block("scan_zk.sh", '  mkdir -p "$OUT/warden"', "  done"), W=w, OUT=out)
    assert got.returncode == 0, got.stderr
    assert published(out) == sorted(f"warden/{f}" for f in public)
