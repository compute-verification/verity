"""No request value reaches a job's $OUT, which custody uploads. Each script's copy into $OUT runs on stand-in records,
with a prompt planted in every file a request's values are in (a rehearsal's plan.json, probe.json, io.json and
out-witness.jsonl, a network attester rehearsal's traffic.json and client.json): $OUT gets the public records and no
file holding the prompt or a request's id fields. A network attester run reaches $OUT only through the gate's
publisher. Nor does a row registration whole reach it, with its wiring's digest: the chain's copy of a step's files puts
the files that hold one on its private side."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = (Path(__file__).resolve().parents[3] / "benchmarks" / "network_traces")
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
    put(g / "g1" / "attester-x", {"end.json": {"files": {}}, "request.json": {"args": ["--requests", "24"]},
                                   **{f"out/{k}": v for k, v in REHEARSAL.items()}})
    (out / "gates").mkdir(parents=True)
    got = run(block("chain_record.sh", '(cd "$G" && find', "  ! -name plan.json"), G=g, OUT=out)
    assert got.returncode == 0, got.stderr
    assert published(out) == sorted(f"gates/g1/attester-x/{f}" for f in (
        "end.json", "request.json", "out/calibration.json", "out/honest/deployment.json", "out/honest/certified.jsonl",
        "out/honest/out-records.jsonl", "out/honest/check.json", "out/honest/client/certificates.jsonl"))


#: a network attester run's files, the gate's publisher's input: a record head with its frames' root and the
#: attester's salt, a registration with its wiring and a line's cost, the attester's own records, and the run's
#: request values
NETWORK_ATTESTER_RUN = {
    "deployment.json": {"windows": 2, "origin_raw": 7}, "groups.json": {"groups": []},
    "cells.jsonl": [{"registration": {"root": "ab" * 64, "wiring": "17" * 64}, "cost": {"rows": 1}}],
    "certified.jsonl": [{"window": 0, "head": {"root": "cd" * 64}, "salt_c": "ef" * 64}],
    "check.json": {"accepted": True}, "warden.json": {"process": {}}, "endpoints.json": {"listen": ["127.0.0.1", 1]},
    "serve.log": "serving\n", "traffic.json": [{"at": 0, "body": {"prompt": PROMPT}}],
    "client.json": {"connections": [{"index": 0, "response": {"ids": PROMPT}}]},
}


def holds_none_of_the_run(out: Path, at: str) -> None:
    """$OUT holds what the gate's publisher gives of a run with no accepted lines, and no head, salt, wiring, cost or
    deployment key the gate keeps."""
    assert published(out) == [f"{at}/deployment.json", f"{at}/groups.json", f"{at}/outcome.json"]
    assert json.loads((out / at / "deployment.json").read_text()) == {"windows": 2}
    text = "".join(p.read_text() for p in (out / at).iterdir())
    assert not any(v in text for v in ("cd" * 64, "ef" * 64, "17" * 64, "salt_c", "wiring", "cost", "head"))


def test_n2_zk_publishes_what_the_gate_publishes_of_a_network_attester_run(tmp_path):
    r, out = tmp_path / "run", tmp_path / "OUT"
    put(r, NETWORK_ATTESTER_RUN)
    out.mkdir()
    got = run(block("n2_zk.sh", 'mkdir -p "$OUT/run"', '"$PY" -m'), RUN=r, OUT=out, PY=sys.executable)
    assert got.returncode == 0, got.stderr
    holds_none_of_the_run(out, "run")


def test_scan_zk_publishes_what_the_gate_publishes_of_its_own_rehearsal(tmp_path):
    w, out = tmp_path / "W", tmp_path / "OUT"
    put(w / "warden", NETWORK_ATTESTER_RUN)
    out.mkdir()
    got = run(block("scan_zk.sh", '  mkdir -p "$OUT/warden"', '  "$PY" -m'), W=w, OUT=out, PY=sys.executable)
    assert got.returncode == 0, got.stderr
    holds_none_of_the_run(out, "warden")


def test_the_chain_keeps_a_step_s_whole_registrations_on_its_private_side(tmp_path):
    """chain_record.sh's copy of a step's files into $OUT, which the leak guard reads and custody uploads, leaves
    out the files that hold a row registration whole, its wiring's digest with it: the registrations a verifier takes
    (the tampered one too) and the scan's answers, which carry them, go to the chain's private side, the quarantine's
    witness/STEP, under their own paths. The step's other files are copied as before, and a prover's file, a circuit
    and an archive's files go to neither."""
    step, out, q = tmp_path / "W" / "scan", tmp_path / "held", tmp_path / "q"
    whole = {"wiring": "17" * 64, "root": "ab" * 64}
    put(step, {"summary.json": {"ok": True}, "answers.json": {"windows": [{"registrations": [whole]}]},
               "honest/registration.json": {"roots": [{"root": "ab" * 64}]}, "honest/registered.json": {"ports": {}},
               "honest/verifier/warden-registration-egress-0.json": whole,
               "honest/verifier/warden-registration-ingress-0.json": whole,
               "tampered/verifier/warden-registration-egress-0-tampered.json": whole,
               "egress/warden-registration.json": whole, "tie/w0-registration.json": {"roots": []},
               "honest/circuit.txt": "c", "honest/inst-4.bin": "i", "honest/archive/answers.json": whole,
               "honest/archive/session.json": {}})
    copy = block("chain_record.sh", "WHOLE=(", "}")
    got = run(copy + 'keep "$W/scan" "$OUT/scan"', W=step.parent, OUT=out, Q=q)
    assert got.returncode == 0, got.stderr
    private = ["scan/answers.json", "scan/egress/warden-registration.json",
               "scan/honest/verifier/warden-registration-egress-0.json",
               "scan/honest/verifier/warden-registration-ingress-0.json",
               "scan/tampered/verifier/warden-registration-egress-0-tampered.json"]
    assert sorted(p.relative_to(q / "witness").as_posix() for p in (q / "witness").rglob("*") if p.is_file()) == private
    assert published(out) == ["scan/honest/registered.json", "scan/honest/registration.json", "scan/summary.json",
                              "scan/tie/w0-registration.json"]
    assert not [p for p in out.rglob("*") if p.is_file() and b"17" * 64 in p.read_bytes()]
