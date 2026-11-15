"""This directory's scripts held to PoUB's verifier host (`verity/protocols/resource_accounting/memory/poub/attester/live_verifier/`): the
prover's tunnel client (`run.sh`'s `ssh -T`) asks no pty while the challenger's forces one, `auditor_line.py` runs on the
host's own python with no site-packages from the pinned tree, and the out/ a host run leaves (the stub verifier's, a real
paced session) is the one `auditor_line.py` reads."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
LIVE = ROOT / "verity/protocols/resource_accounting/memory/poub/attester/live_verifier/host.py"
sys.path.insert(0, str(HERE.parent))

import auditor_line  # noqa: E402


def test_the_provers_tunnel_client_asks_no_pty_and_the_challengers_forces_one():
    """ssh ends a session whose forced pty (`-tt`) the server refuses (OpenSSH 9.6, checked against a real sshd), so the
    prover's key, whose client is run.sh's `ssh -T`, has no `pty`, and the challenger's keeps it only while serve.py
    forces one; infra's packaging test holds the keys `setup.sh` writes to the same."""
    verifier_host = (HERE.parent / "run.sh").read_text().split('[ -n "${VERIFIER_SSH:-}" ]; then')[1].split("\nelif ")[0]
    assert re.search(r"\bssh -T\b", verifier_host) and "-tt" not in verifier_host
    serve = (ROOT / "verity/protocols/compose/verifier/live_challenger/serve.py").read_text()
    assert re.search(r'^SSH = \["ssh", "-tt"', serve, re.M), (
        "serve.py no longer forces a pty: drop `pty` from the challenger's tunnel key in verifier/setup.sh, and this check")


def test_auditor_line_runs_on_the_hosts_python_with_no_site_packages(tmp_path):
    """The host runs its units with its own python3 (`-I`), where neither `verity` nor any other package is installed:
    `auditor_line.py` from the host's pinned tree (`host.TREE`) resolves every import within it."""
    code = ("import runpy, sys; sys.path.insert(0, %r); import host; sys.argv = ['auditor_line.py', '--help']; "
            "runpy.run_path(str(host.TREE / 'benchmarks' / 'remote_session' / 'auditor_line.py'), run_name='__main__')"
            % str(LIVE.parent))
    p = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "site-packages" not in p.stdout


@pytest.mark.slow
def test_auditor_line_reads_the_out_a_host_run_leaves(tmp_path):
    """The stub verifier the host's slow test runs (with the Lean verdict, so it builds the Lean verifier): what the
    cheaper tests of `auditor_line` can't show is that the out/ a host run leaves, from a real paced session, is one it
    reads, its registration and its session's verdict as the run's record says."""
    o = tmp_path / "out"
    p = subprocess.run([sys.executable, str(LIVE.parent / "tests" / "stub_verifier.py"), "--out", str(o),
                        "--transcript", str(tmp_path / "transcript.jsonl"), "--registry", str(tmp_path / "registry.json"),
                        "--port", "0"], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    v = json.loads((o / "verifier.json").read_text())
    line = auditor_line.analyse(o, host_bytes=1843153879040, pin=None, period=None, heads=None, peer="auditor-host")
    assert line["registration"] == v["registration"]["digest"]
    assert [(r["k"], r["accepted"], r["skipped"]) for r in line["sessions"]] == [
        (0, True, "no challenger log: coins drawn in the verifier's process")]
