"""No request value reaches a job's $OUT, which custody uploads. Each script's copy into $OUT runs on stand-in records,
with a prompt planted in every file a request's values are in (a rehearsal's plan.json, probe.json, io.json and
out-witness.jsonl, a network attester rehearsal's traffic.json and client.json): $OUT gets the public records and no
file holding the prompt or a request's id fields. A network attester run reaches $OUT only through the gate's
publisher. Nor does a row registration whole reach it, with its wiring's digest: the chain's copy of a step's files puts
the files that hold one on its private side."""
from __future__ import annotations

import json
import re
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
    "honest/cells.jsonl": [{"registration": {"wiring": "17"}, "cost": {"rows": 1}}],
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
        "end.json", "request.json", "out/calibration.json", "out/honest/deployment.json", "out/honest/out-records.jsonl",
        "out/honest/check.json", "out/honest/client/certificates.jsonl"))


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
    got = run(block("chain_record.sh", "WHOLE=(", "}") + kept("SCANOUT", "scan"), W=step.parent, OUT=out, Q=q,
              SCANOUT=step)
    assert got.returncode == 0, got.stderr
    private = ["scan/answers.json", "scan/egress/warden-registration.json",
               "scan/honest/verifier/warden-registration-egress-0.json",
               "scan/honest/verifier/warden-registration-ingress-0.json",
               "scan/tampered/verifier/warden-registration-egress-0-tampered.json"]
    assert sorted(p.relative_to(q / "witness").as_posix() for p in (q / "witness").rglob("*") if p.is_file()) == private
    assert published(out) == ["scan/summary.json"]
    assert not [p for p in out.rglob("*") if p.is_file() and b"17" * 64 in p.read_bytes()]


#: a tie run's directory: its summary and public parameters, and what its verifier received, the inputs among them
TIE_RUN = {"summary.json": {"ok": True, "tie": {"honest": [True]}}, "stager/program.json": {"units": 1},
           "stager/partition.json": {"program": "p"},
           "honest/verifier/inputs.json": {"places": [0], "lengths": [64], "received": [2], "digest": "ab" * 64},
           "honest/verifier/w000-outcome.json": {"public": {"inputs": "ab" * 64}},
           "honest/verifier/w000-registered.json": {"ports": {"pub": {"known": [64]}}},
           "honest/verifier/w000-delivered.json": {"salts": {}}}
TIE_KEPT = ["stager/partition.json", "stager/program.json", "summary.json"]


def test_n2_tie_zk_keeps_its_verdicts_and_public_parameters_and_never_its_inputs(tmp_path):
    """The tie's inputs (each request's place, its prompt length, the cells each window received) and the outcome records
    naming their digest are what the verifier received, not public: $OUT/n2tie gets the summary and the tie's
    Program and partition alone."""
    w, out = tmp_path / "W", tmp_path / "OUT"
    put(w / "n2tie", TIE_RUN)
    got = run(block("n2_tie_zk.sh", 'mkdir -p "$OUT/n2tie"', "  [ -f"), W=w, OUT=out)
    assert got.returncode == 0, got.stderr
    assert published(out) == [f"n2tie/{f}" for f in TIE_KEPT]


def test_the_chains_tie_step_keeps_its_verdicts_and_public_parameters_and_never_its_inputs(tmp_path):
    w, out = tmp_path / "W", tmp_path / "OUT"
    put(w / "tie", TIE_RUN)
    got = run(block("chain_record.sh", "WHOLE=(", "}") + kept('W/tie', "tie"), W=w, OUT=out, Q=tmp_path / "q")
    assert got.returncode == 0, got.stderr
    assert published(out) == [f"tie/{f}" for f in TIE_KEPT]


def kept(source: str, step: str) -> str:
    """chain_record.sh's own call of ``keep`` for ``step``, whose files are under ``$source``."""
    (line,) = re.findall(rf'keep "\${source}" "\$OUT/{step}"[^;\n]*', (HERE / "chain_record.sh").read_text())
    return line + "\n"


#: an N2 run's directory: its summary and public parameters, and what its verifier received
N2_RUN = {"summary.json": {"ok": True}, "stager/program.json": {"units": 1}, "stager/partition.json": {"program": "p"},
          "honest/verifier/w000-registration.json": {"roots": [{"root": "ab" * 64}]},
          "honest/verifier/w000-certificates.json": [{"window": 0}], "honest/verifier/w000-syncs.json": [],
          "honest/verifier/w000-delivered.json": {"salts": {"0": {"pub": "cd" * 32}}},
          "honest/verifier/w000-outcome.json": {"public": {"verdict": 1}}, "honest/verifier/period.json": {"units": [0]},
          "honest/verifier/w000-pub-1.bin": "p", "honest/verifier/w000-session-s-session.json": {}}


def test_n2_zk_keeps_its_summary_and_public_parameters_and_nothing_the_verifier_received(tmp_path):
    w, out = tmp_path / "W", tmp_path / "OUT"
    put(w / "n2zk", N2_RUN)
    got = run(block("n2_zk.sh", 'mkdir -p "$OUT/n2zk"', "  [ -f"), W=w, OUT=out)
    assert got.returncode == 0, got.stderr
    assert published(out) == [f"n2zk/{f}" for f in TIE_KEPT]


def test_the_chains_n2_and_scan_steps_keep_their_summaries_and_the_scans_program_alone(tmp_path):
    """The chain releases N2's summary alone (its one file `released_keys.json` classifies) and the scan's summary and
    joint Program (the attester piece's period case reads both); what each step's verifier received stays in $W."""
    w, out, q = tmp_path / "W", tmp_path / "OUT", tmp_path / "q"
    put(w / "n2", N2_RUN)
    put(w / "scan", {"summary.json": {"ok": True}, "joint/program.json": {"units": 2},
                     "honest-0/verifier/program.json": {"units": 3}, "honest-0/verifier/openings.json": [{"o0": 1}],
                     "honest-0/registration.json": {"window": {"units": {"cells": [[1, 4]]}}}})
    code = block("chain_record.sh", "WHOLE=(", "}") + kept("N2OUT", "n2") + kept("SCANOUT", "scan")
    got = run(code, OUT=out, Q=q, N2OUT=w / "n2", SCANOUT=w / "scan")
    assert got.returncode == 0, got.stderr
    assert published(out) == ["n2/summary.json", "scan/joint/program.json", "scan/summary.json"]


ENDPOINT_PY = """#!/bin/sh
out=; while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
echo '{"chain": {"bits": {"bound_why": "certified requests are in none (s0-needle)", "commit_windows": [{"requests": 7}]}}}' > "$out"
echo "the endpoint's stdout: s0-needle"
"""


def test_the_chains_endpoint_step_keeps_its_report_and_logs_on_the_private_side(tmp_path):
    """The endpoint's report holds its advice rebuilt in the clear and each Commit window's counts, and its log and the
    advice's what it printed: whether the step rebuilds it or takes the served node's, none of it reaches $OUT."""
    py = tmp_path / "py"
    py.write_text(ENDPOINT_PY)
    py.chmod(0o755)
    code = ('unpack() { mkdir -p "$2"; cp -r "$1"/. "$2"; }\n'
            + block("chain_record.sh", 'if [ -n "${CR_ENDPOINT:-}" ]', "fi"))
    for k, files in enumerate(({"r/advice_trace.jsonl": "{}\n", "r/advice.log": "s0-needle\n"},
                               {"r/report.json": '{"bits": "s0-needle"}'})):
        ep, out, q, w = (tmp_path / f"{x}{k}" for x in ("ep", "OUT", "Q", "W"))
        put(ep, files)
        out.mkdir()
        got = run(code, CR_ENDPOINT=ep, CR_EOS_IDS="1", NETWORK_CERTIFIER="g/run", G="g", PY=py, CR="cr", OUT=out, Q=q,
                  W=w)
        assert got.returncode == 0, got.stderr
        assert published(out) == ["job.log"] and b"s0-needle" not in (out / "job.log").read_bytes()
        kept = sorted(p.name for p in (q / "witness" / "endpoint").iterdir())
        assert kept == (["advice.log", "endpoint.log", "report.json"] if k == 0 else ["report.json"])


def test_the_records_endpoint_step_keeps_its_verdict_and_counts_of_windows_alone():
    """The report's endpoint step (chain_record.py's, from the private report) names no request and counts no window's
    requests: the attested I/O's violations by their count, the advice's binding by its Commit windows'."""
    sys.path.insert(0, str(HERE.parents[1]))
    from benchmarks.network_traces import chain_record as cr

    bits = {"bound": False, "rebuilt": False, "bound_why": "2 certified requests are in none (s0-needle, s1-needle)",
            "commit_windows": [{"window": 0, "requests": 7, "bound": True, "rebuilt": True},
                               {"window": 1, "requests": 5, "bound": False, "why": "s1-needle"}]}
    for io in ({"accepted": True}, {"accepted": False, "violations": [[3, "a request of s0-needle"], [9, "x"]]}):
        step = cr.network_attester_endpoint_step({"warden_run": "r", "io": io, "chain": {"bits": bits}})
        assert step["verdict"] == "refused" and "needle" not in json.dumps(step) and '"requests"' not in json.dumps(step)
        assert step["advice"] == {"bound": False, "rebuilt": False, "commit_windows": 2, "bound_windows": 1}
        assert step["io"] == {"accepted": io["accepted"], "violations": len(io.get("violations") or [])}
        e2e = cr.endpoint_step({"chain": {"bits": bits, "tie": {}}})
        assert "needle" not in json.dumps(e2e) and e2e["advice"] == step["advice"]
