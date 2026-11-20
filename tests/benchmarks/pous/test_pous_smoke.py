"""The storage piece's smallest runs that reach `pous-run-protocol`, honest and dishonest, on a processor: the piece's
sessions over the tiny int8 checkpoint on the CPU, the parties played in one process (so nothing here certifies a
deployment), the signer the PoUS tests' own, and the Lean the built `pous-run-protocol` under `--test`, which takes that
signer's schemes where the deployed protocol takes the attester's. The honest case is accepted by it; the same case with
its trusted key another's is refused by it, by name."""
from __future__ import annotations

import hashlib
import json
import os
import re

import pytest
from test_pous_served_piece import RUN_EXE, TEST_PROTOCOL, _run, case_exe, run_exe, served  # noqa: F401  (fixtures)

from verity.protocols.resource_accounting.memory.pous.verifier import period

NOT_TRUSTED = "the session is signed by a key that is not a trusted memory challenger's"


def _digest(exe) -> str:
    """The SHA-512 of the Lean file the script of `run_exe` execs."""
    real = re.search(r"exec (\S+)", exe.read_text()).group(1)
    return hashlib.sha512(open(real, "rb").read()).hexdigest()


@pytest.mark.slow
def test_the_honest_case_is_accepted_and_the_same_case_under_another_key_is_refused_by_its_lean(
        served, tmp_path, monkeypatch, run_exe):
    answer, _ = _run(served, tmp_path, monkeypatch, case_exe=str(run_exe))
    assert answer["verdict"] == "accept", answer["reason"]
    doc = json.loads((served.out / "pous.json").read_text())
    assert doc["decided_by"]["program"] == RUN_EXE
    case = json.loads((served.private / "case.json").read_text())
    honest = period.serve(run_exe, [case])[0]
    assert honest["verdict"] == "accept" and honest["protocol"] == TEST_PROTOCOL
    other = case | {"params": case["params"] | {"trusted": ["00" * 64]}}
    refused = period.serve(run_exe, [other])[0]
    assert refused["verdict"] == "reject" and refused["reason"]
    line = (f"SMOKE storage honest: {({k: honest[k] for k in ('verdict', 'protocol')})} dishonest (trusted key swapped): "
            f"{({k: refused[k] for k in ('verdict', 'reason')})} by {RUN_EXE} sha512 {_digest(run_exe)}")
    if out := os.environ.get("SMOKE_OUT"):
        with open(out, "a") as f:
            f.write(line + "\n")


@pytest.mark.slow
def test_the_bandwidth_audits_honest_session_is_accepted_and_two_dishonest_are_refused_by_its_lean(tmp_path, case_exe):
    """`bandwidth.py session` over the toy's store on the CPU, against `p2_party_at_leaf.py` in the auditor gate's shape
    (`python -S -E`, its close signed at the deployed parameters under a test key made without a keygen): the parties in one
    machine, so it certifies no deployment. The Lean is the built `poub-p2-protocol`. The honest session is accepted by it;
    a prover that skips a layer is rejected by it for its receipts; and the same honest session under a party trusting another
    key is refused by it, by name."""
    import hashlib as H
    import sys
    from pathlib import Path

    import test_pous_p2_bandwidth as T

    reg = T.BT.toy(tmp_path / "toy")
    (tmp_path / "challenger").mkdir()
    T.mc.SessionLog.create(tmp_path / "challenger" / "sessions")
    lines = []

    def run(name: str, trusted: str, more: list[str], expect: str) -> dict:
        session = H.sha512(name.encode()).hexdigest()[:64]
        party = (f"cd {T.ROOT} && exec {sys.executable} -S -E {Path(T.__file__).resolve().parent / 'p2_party_at_leaf.py'} "
                 f"--device {tmp_path / ('party-' + name)} --listen 127.0.0.1:0 "
                 f"--registration {tmp_path / 'toy' / 'registration.json'} {T.held_to(1, 1, 8)} "
                 f"--challenger {tmp_path / 'challenger'} --session {session} --trusted-sha512 {trusted}")
        private = tmp_path / "private" / name
        assert T.B.main(["session", "--store", str(tmp_path / "toy" / "store"), "--layout", str(tmp_path / "toy" / "layout.json"),
                         "--node", reg.node.decode(), "--party", party, "--party-host", "127.0.0.1", "--device", "cpu",
                         "--ref-decode", "--samples", "8", "--windows", "1", "--passes", "1", "--window-s", "0.001",
                         "--deadline-s", "0.001", "--workers", "1", "--out", str(tmp_path / name), "--expect", expect,
                         "--case-dir", str(private), "--case-exe", str(case_exe), *more]) == 0
        answer = json.loads((private / T.p2_case.ANSWER).read_text())
        lines.append(f"SMOKE bandwidth {name}: {({k: answer.get(k) for k in ('verdict', 'reason', 'protocol')})}")
        return answer

    honest = run("honest", T.AT_LEAF, [], "accept")
    assert honest["verdict"] == "accept", honest["reason"]
    cheat = run("cheat", T.AT_LEAF, ["--skip-layer", "1"], "reject")
    assert cheat["verdict"] == "reject"
    swapped = run("swapped", "00" * 64, [], "accept")
    assert swapped["verdict"] in ("reject", "undecided") and swapped["reason"]
    if out := os.environ.get("SMOKE_OUT"):
        with open(out, "a") as f:
            f.write("\n".join(lines) + f"\nby poub-p2-protocol sha512 {hashlib.sha512(open(case_exe, 'rb').read()).hexdigest()}\n")


@pytest.mark.slow
def test_the_wipe_launcher_on_a_processor_is_accepted_by_wipe_protocol_and_two_dishonest_cases_are_refused(tmp_path):
    """`served_wipe.sh` as the application's hook runs it (`--run-keys`: the memory attester on this machine under a key made
    for the run, so a rehearsal that certifies nothing), the fill in host memory and a stand-in `gpu-lease`, which a processor
    has none of; the Lean it builds in a scratch copy is the declared `wipe-protocol`, whose accept ends the launcher with exit 0.
    The case it keeps is then asked of the same executable built beside: unchanged, accepted; with a key that is not the
    pinned one, and with one word of its signed session changed, each refused by name. It costs minutes (two Lean builds):
    what it catches that the suite's stand-in judge cannot is a launcher or a verdict that has stopped reaching the real
    executable, which no cheaper test touches."""
    import shutil
    import stat
    import subprocess
    import sys

    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    from test_pous_served_piece import ROOT as TREE
    from test_pous_served_piece import _built

    bin_ = tmp_path / "bin"
    bin_.mkdir()
    lease = bin_ / "gpu-lease"
    lease.write_text('#!/usr/bin/env bash\nwhile [ "$1" != -- ]; do shift; done\nshift\nGPU_LEASE_UUID=GPU-standin exec "$@"\n')
    lease.chmod(lease.stat().st_mode | stat.S_IXUSR)
    serve, out = tmp_path / "serve", tmp_path / "serve" / "out"
    serve.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPU_LEASE_", "WIPE_", "VERITY_"))}
    env |= {"PATH": f"{bin_}:{os.environ['PATH']}", "PY": sys.executable, "DEVELOPMENT": "1"}
    ran = subprocess.run(["bash", str(TREE / "benchmarks/pous/streamed_wipe/served_wipe.sh"), str(serve), "0", str(out),
                          '{"run_keys": true, "store": "host", "streamer": "python"}'],
                         env=env, cwd=TREE, capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-2000:]
    said = json.loads(next(x for x in ran.stdout.splitlines() if x.startswith("WIPE ")).removeprefix("WIPE "))
    assert said["verdict"] == "accept" and said["decided_by"]["program"] == "wipe-protocol" and said["key_pin"] == "run"
    exe = _built(TREE / "verity/protocols/resource_accounting/memory/pous/attester/lean", "wipe-protocol")
    assert hashlib.sha512(exe.read_bytes()).hexdigest() == said["executables"]["wipe-protocol"]
    case = json.loads((serve / "private" / "wipe" / "wipe.case.json").read_text())
    node = bytes.fromhex(case["params"]["server"]).decode()
    swapped = case | {"params": case["params"] | {"trusted": ["00" * 64]}}
    evidence = json.loads(bytes.fromhex(case["evidence"]))
    evidence["signed"]["batches"][0][next(k for k, v in evidence["signed"]["batches"][0].items() if isinstance(v, int))] += 1
    changed = case | {"evidence": json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode().hex()}
    answers = {n: period.serve(exe, [c], "--server", node)[0] for n, c in (("honest", case), ("swapped", swapped), ("changed", changed))}
    assert answers["honest"]["verdict"] == "accept"
    assert answers["swapped"]["verdict"] == "reject" and NOT_TRUSTED in answers["swapped"]["reason"]
    assert answers["changed"]["verdict"] == "reject" and "the signature does not verify" in answers["changed"]["reason"]
    if target := os.environ.get("SMOKE_OUT"):
        with open(target, "a") as f:
            f.write("SMOKE wipe: " + json.dumps({n: {k: a[k] for k in ("verdict", "reason")} for n, a in answers.items()})
                    + f" by wipe-protocol sha512 {said['executables']['wipe-protocol']}\n")
