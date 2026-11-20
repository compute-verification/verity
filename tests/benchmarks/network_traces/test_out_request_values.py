"""No request value reaches a job's $OUT, which custody uploads. Each script's copy into $OUT runs on stand-in records,
with a prompt planted in every file a request's values are in (a rehearsal's plan.json, probe.json, io.json and
out-witness.jsonl, a network attester rehearsal's traffic.json and client.json): $OUT gets the public records and no
file holding the prompt or a request's id fields. A network attester run reaches $OUT only through the gate's
publisher. Nor does a row registration whole reach it, with its wiring's digest: the scan keeps it in its SC_KEEP,
where the period's reader reads it."""
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


def test_scan_zk_publishes_what_the_gate_publishes_of_its_own_rehearsal(tmp_path):
    w, out = tmp_path / "W", tmp_path / "OUT"
    put(w / "warden", NETWORK_ATTESTER_RUN)
    out.mkdir()
    got = run(block("scan_zk.sh", '  mkdir -p "$OUT/warden"', '  "$PY" -m'), W=w, OUT=out, PY=sys.executable)
    assert got.returncode == 0, got.stderr
    holds_none_of_the_run(out, "warden")


def test_scan_zk_keeps_its_summary_and_programs_and_nothing_the_verifier_received(tmp_path):
    """Of its driver's directory the scan keeps the summary, which holds each answer's verdict and public part, and each
    round's Program; ``answers.json`` and the verifier's files (the registrations with their wiring, the openings,
    the registered reads) stay in $W."""
    w, out = tmp_path / "W", tmp_path / "OUT"
    whole = {"wiring": "17" * 64, "root": "ab" * 64}
    put(w / "sczk", {"summary.json": {"answers": {"0": {"verdict": "accept", "public": {"proved": [0]}}}},
                     "answers.json": {"windows": [{"registration": whole, "registrations": [whole, whole]}]},
                     "honest-0/verifier/program.json": {"units": 3},
                     "honest-0/verifier/warden-registration-ingress-0.json": whole,
                     "honest-0/verifier/openings.json": [{"unit": 0, "o0": 1}],
                     "honest-0/verifier/registration.json": {"window": {"units": {"cells": [[1, 4]]}}},
                     "honest-0/registered.json": {"ports": {"p": {"rows": [3]}}}})
    (out / "sczk").mkdir(parents=True)
    got = run(block("scan_zk.sh", '(cd "$W/sczk" && find', "  | xargs"), W=w, OUT=out)
    assert got.returncode == 0, got.stderr
    assert published(out) == ["sczk/honest-0/verifier/program.json", "sczk/summary.json"]
    assert "17" * 64 not in "".join(p.read_text() for p in out.rglob("*.json"))


def test_network_attester_zk_keeps_its_summary_alone_of_its_drivers_directory(tmp_path):
    """A member's registrations and the verifier's registered reads open the wiring: of the driver's directory the
    job keeps only its summary."""
    w, out = tmp_path / "W", tmp_path / "OUT"
    put(w / "wzk", {"summary.json": {"ok": True}, "pad/warden-registration.json": {"wiring": "17" * 64},
                    "pad/registration.json": {"window": {"units": {"cells": [[1, 4]]}}},
                    "pad/verifier/registered.json": {"ports": {"p": {"rows": [3]}}}})
    got = run(block("network_attester_zk.sh", 'mkdir -p "$OUT/wzk"', '[ -f "$W/wzk/summary.json" ]'), W=w, OUT=out)
    assert got.returncode == 0, got.stderr
    assert published(out) == ["wzk/summary.json"]


def test_the_records_endpoint_step_keeps_its_verdict_and_counts_of_windows_alone():
    """The endpoint step (chain_record.py's, from the private report) names no request and counts no window's
    requests: the attested I/O's violations by their count, the advice's binding by whether it is bound and rebuilt."""
    sys.path.insert(0, str(HERE))
    import chain_record as cr

    bits = {"bound": False, "rebuilt": False, "bound_why": "2 certified requests are in none (s0-needle, s1-needle)",
            "commit_windows": [{"window": 0, "requests": 7, "bound": True, "rebuilt": True},
                               {"window": 1, "requests": 5, "bound": False, "why": "s1-needle"}]}
    for io in ({"accepted": True}, {"accepted": False, "violations": [[3, "a request of s0-needle"], [9, "x"]]}):
        step = cr.network_attester_endpoint_step({"warden_run": "r", "io": io, "chain": {"bits": bits}})
        assert step["verdict"] == "refused" and "needle" not in json.dumps(step) and '"requests"' not in json.dumps(step)
        assert step["advice"] == {"bound": False, "rebuilt": False}
        assert step["io"] == {"accepted": io["accepted"], "violations": len(io.get("violations") or [])}


def test_scan_period_reads_every_file_it_opens_from_the_scans_private_keep(tmp_path, monkeypatch):
    """A stub run of scan_period.sh (its loop of scan_zk.sh, each run's scan_zk.sh its own copy into $OUT and its
    SC_KEEP, then its call of scan_period.py) leaves the period's reader every file it opens, and all of them on the
    scan's private side (0700, 0600 files, outside both run dirs): $OUT keeps what the gate publishes alone. The
    reader runs whole, with the mirror, network-period and the parameters' checks stood in."""
    from types import SimpleNamespace

    from verity.protocols.resource_accounting.network.prover import scan_period as SP
    from verity.protocols.resource_accounting.network.verifier import relay_check as RC

    inner, w, out = tmp_path / "inner", tmp_path / "W", tmp_path / "OUT"
    certs = [{"certificate": {"link": link, "direction": d.value, "window": 0, "digest": "ab", "key_id": 0,
                              "signature": "00"}, "head": {"root": "cd" * 64}} for link, d in RC.CERTIFIED.values()]
    deployment = {"windows": 1, "session": "s", "start": {"signed": 1}, "key": {"params": [1, 2, 3], "public": "aa"},
                  "params": {"egress": "id", "lanes": 1}}
    reg = {"program": "ab" * 64, "wiring": "17" * 64}
    for run_ in ("honest", "late"):
        put(inner / run_ / "warden", {**NETWORK_ATTESTER_RUN, "deployment.json": deployment, "certified.jsonl": certs})
        put(inner / run_ / "sczk", {
            "answers.json": {"windows": [{"window": 0, "registration": reg, "answer": {"verdict": "accept"},
                                          "registrations": [reg]}]},
            "summary.json": {"rounds": {f"{r}-0": {"audit": {"why": "x"}} for r in ("tampered", "chain")}},
            "joint/program.json": {"templates": []}, "tampered-0/registration.json": reg,
            "chain-0/registration.json": reg, "honest-0/verifier/program.json": {"units": 3},
            "honest-0/verifier/registration.json": reg, "honest-0/session-start/prove.log": "LIVE\t{}\n",
            "honest-0/session-start/lean-verify.log": "inputs: 1 ms\n", "honest-0/session-start/sessions/s.json": {}})
    stub = tmp_path / "scan_zk_stub.sh"
    stub.write_text(LOG + 'W=$INNER/$(basename "$RESEARCH_RUN_DIR") OUT=$RESEARCH_RUN_DIR/out PY=$REAL_PY rc=0\n'
                    'mkdir -p "$OUT"\n' + block("scan_zk.sh", '  mkdir -p "$OUT/warden"', "  SC_WARDEN_RUN")
                    + block("scan_zk.sh", 'mkdir -p "$OUT/sczk"', 'log "sczk: done'))
    argv = tmp_path / "argv"
    (capture := tmp_path / "py").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {argv}\n')
    capture.chmod(0o755)
    loop = block("scan_period.sh", "declare -A rc", "p=${PIPESTATUS[0]}").replace(
        "benchmarks/network_traces/scan_zk.sh", str(stub))
    out.mkdir()
    got = run(loop, W=w, OUT=out, INNER=inner, P=tmp_path / "P", SRC=tmp_path / "SRC", PYTHONPATH="", PY=capture,
              REAL_PY=sys.executable)
    assert got.returncode == 0, got.stderr
    args = argv.read_text().splitlines()
    assert args[:2] == ["-m", "verity.protocols.resource_accounting.network.prover.scan_period"], args

    opened: list[Path] = []
    read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k: (opened.append(p), read_text(p, *a, **k))[1])
    monkeypatch.setattr(SP, "mirror", lambda path: SimpleNamespace(
        reference=lambda case: {"verdict": "reject"}, UNWIRED="unwired", UNSTATED="unstated", _states=lambda r: True))
    monkeypatch.setattr(SP, "lean_check", lambda lean, case, at: {"verdict": "reject"})
    monkeypatch.setattr(SP, "NetworkAttesterParams", lambda **k: SimpleNamespace(rows_cells=lambda n: []))
    monkeypatch.setattr(SP, "INTEGER_FIELDS", ("lanes",))
    monkeypatch.setattr(SP.SA, "window_answer", lambda *a, **k: {"verdict": "reject"})
    SP.main(args[2:])
    read = sorted({p.relative_to(w).as_posix() for p in opened if p.is_relative_to(w)})
    assert read == sorted(f"keep/{r}/{f}" for r, fs in (
        ("honest", ["network_attester/certified.jsonl", "network_attester/deployment.json", "sczk/answers.json",
                    "sczk/summary.json", "sczk/joint/program.json", "sczk/tampered-0/registration.json",
                    "sczk/chain-0/registration.json"]),
        ("late", ["network_attester/certified.jsonl", "network_attester/deployment.json", "sczk/answers.json",
                  "sczk/joint/program.json"])) for f in fs)
    assert not [p for p in opened if p.is_relative_to(out)]
    keep = w / "keep"
    assert sorted(p.relative_to(keep).as_posix() for p in keep.rglob("*.log")) == [   # benchmarks/scan_cost's stages
        f"{r}/sczk/honest-0/session-start/{f}" for r in ("honest", "late") for f in ("lean-verify.log", "prove.log")]
    assert {oct(p.stat().st_mode & 0o777) for p in keep.rglob("*") if p.is_file()} == {"0o600"}
    assert {oct(p.stat().st_mode & 0o777) for p in [keep / "honest", *keep.rglob("*")] if p.is_dir()} == {"0o700"}
    for run_ in ("honest", "late"):
        assert published(out / run_) == ["job.log", "sczk/honest-0/verifier/program.json", "sczk/summary.json",
                                         "warden/deployment.json", "warden/groups.json", "warden/outcome.json"]
        assert "17" * 64 not in "".join(p.read_text() for p in (out / run_).rglob("*.json"))


def test_each_runner_fails_by_name_when_the_publisher_fails(tmp_path):
    """The three runners that publish a network attester run stop, by name, when the publisher exits nonzero, rather
    than going on with an empty outcome.json."""
    (py := tmp_path / "py").write_text("#!/bin/sh\nexit 3\n")
    py.chmod(0o755)
    for script, first in (("capture_loopback.sh", 'mkdir -p "$OUT/run"'), ("network_attester_zk.sh", 'mkdir -p "$OUT/run"'),
                          ("scan_zk.sh", '  mkdir -p "$OUT/warden"')):
        out = tmp_path / script / "OUT"
        out.mkdir(parents=True)
        got = run(block(script, first, first.split("mkdir")[0] + '"$PY" -m') + "echo went on\n",
                  W=tmp_path / script / "W", RUN=tmp_path / script / "run", OUT=out, PY=py)
        assert got.returncode == 6 and "went on" not in got.stdout, (script, got.returncode, got.stdout)
        assert "the publisher (auditor.py publish) exited 3" in (out / "job.log").read_text(), script
