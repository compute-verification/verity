"""`gate.sh probe`: before a gate run writes anything to either auditor host, it reads vy-challenger-1's key, pins and
ledger and vy-auditor-1's status, and refuses (exit 3, the reason in probe.json) a key that isn't PIN, a used period, a
period of the lane still open, a host pinned elsewhere or busy, and, before any request, a CIRCUIT that isn't COMMIT's
and a COMMIT off origin/main's first-parent line (or no line to check it on). A host pinned elsewhere on that line is
told to rerun at its pin; one pinned off it, to repin. Under DRY=1 nothing reaches ssh. The gates are a stub ssh on
PATH; COMMIT is the head of a throwaway repository whose bare origin's main holds it and its parent."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "gate.sh"
FP = "SHA256:s8UdOpJZGMXJR1DjgmOyTsWpTqQX7GDJq/yQCYhSJhs"
PUBLIC = bytes(range(80))
PIN = hashlib.sha512(PUBLIC).hexdigest()


def entry(seq: int, kind: str, period: str, lane: str) -> dict:
    return {"seq": seq, "kind": kind, "body": {"period": period, "lane": lane, "by": lane}}


CLOSED_7 = [entry(1, "period-open", "ma-remote-coins-7", "memory-accounting"),
            entry(2, "period-open", "kernels-dev-1", "kernels"),
            entry(3, "period-close", "ma-remote-coins-7", "memory-accounting")]


GIT = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@t"}


def git(r: Path, *cmd: str) -> str:
    return subprocess.run(["git", *cmd], cwd=r, env=GIT, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    r, origin = tmp_path / "repo", tmp_path / "origin.git"
    (r / "verity/core/protocols/remote_bandwidth").mkdir(parents=True)
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(r, "init", "-q", "-b", "main")
    for x in ("0", "1"):
        (r / "verity/core/protocols/remote_bandwidth/store.py").write_text(f"x = {x}\n")
        git(r, "add", ".")
        git(r, "commit", "-q", "-m", f"c{x}")
    git(r, "remote", "add", "origin", str(origin))
    git(r, "push", "-q", "origin", "main")
    return r, git(r, "rev-parse", "HEAD")


def off_main(repo: tuple[Path, str]) -> tuple[Path, str]:
    """The repository checked out at a commit on top of main that origin/main doesn't hold."""
    r, _ = repo
    git(r, "checkout", "-q", "-b", "side")
    (r / "verity/core/protocols/remote_bandwidth/store.py").write_text("x = 2\n")
    git(r, "commit", "-q", "-am", "side")
    return r, git(r, "rev-parse", "HEAD")


def probe(tmp_path: Path, repo: tuple[Path, str], *, ledger=CLOSED_7, pinned=None, busy=(), public=PUBLIC,
          period="ma-remote-coins-8", circuit=None, dry=False) -> tuple[subprocess.CompletedProcess, Path, list[str]]:
    """gate.sh probe against stub gates answering from these records; its result, $OUT and the calls ssh got."""
    r, head = repo
    g, out, bin_ = tmp_path / "gates", tmp_path / "out", tmp_path / "bin"
    for d in (g, out, bin_):
        d.mkdir()
    (g / "pubkey").write_text(json.dumps({"W": 4, "H": 10, "public": public.hex()}))
    (g / "pins").write_text(json.dumps({"commit": head, "main": head}) + "\n")
    (g / "ledger").write_text("".join(json.dumps(e) + "\n" for e in ledger))
    (g / "status").write_text(json.dumps({"pinned": pinned or head, "busy": list(busy), "pins": [], "runs": []}))
    (bin_ / "ssh").write_text(f'#!/bin/sh\nfor a; do last=$a; done\necho "$*" >> {g}/calls\ncat "{g}/$last"\n')
    (bin_ / "ssh-keygen").write_text(f'#!/bin/sh\necho "# Host 81.85.1.145 found: line 1"\necho "81.85.1.145 ED25519 {FP}"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}", "OUT": str(out), "COMMIT": head,
           "PERIOD": period, "PIN": PIN, "GATE_KEY": str(tmp_path / "key"), "AUDITOR_FP": FP}
    if circuit:
        env["CIRCUIT"] = circuit
    if dry:
        env["DRY"] = "1"
    p = subprocess.run(["bash", str(GATE), "probe"], cwd=r, env=env, capture_output=True, text=True)
    calls = (g / "calls").read_text().splitlines() if (g / "calls").exists() else []
    return p, out, calls


def test_the_probe_passes_reading_only(tmp_path, repo):
    p, out, calls = probe(tmp_path, repo)
    assert p.returncode == 0, p.stdout + p.stderr
    tree = subprocess.run(["git", "rev-parse", f"{repo[1]}:verity/core/protocols/remote_bandwidth"], cwd=repo[0],
                          capture_output=True, text=True, check=True).stdout.strip()
    doc = json.loads((out / "probe.json").read_text())
    assert doc["refused"] == [] and doc["challenger_pinned_at_commit"]
    assert doc["circuit"] == hashlib.sha512(f"verity/remote-bandwidth/v0 verifier tree {tree}".encode()).hexdigest()
    assert [c.split()[-1] for c in calls] == ["pubkey", "pins", "ledger", "status"]


@pytest.mark.parametrize("kw, why", [
    ({"period": "ma-remote-coins-7"}, "period ma-remote-coins-7 is used (seq [1, 3]): name a new PERIOD"),
    ({"ledger": CLOSED_7 + [entry(4, "period-open", "ma-remote-coins-x", "memory-accounting")]},
     "lane memory-accounting has period ma-remote-coins-x open: close it first (gate.sh close, PERIOD=ma-remote-coins-x)"),
    ({"public": bytes(80)}, f"the challenger's public key is {hashlib.sha512(bytes(80)).hexdigest()[:16]}..., "
                            f"not PIN {PIN[:16]}..."),
    ({"pinned": "{parent}"}, "vy-auditor-1 is pinned at {parent}, not COMMIT {head}: rerun with COMMIT={parent}"),
    ({"pinned": "9b062d027f612c9f39faae87da61fcc554c84b81"},
     "vy-auditor-1 is pinned at 9b062d027f612c9f39faae87da61fcc554c84b81, off main's first-parent line, where no "
     "session credits: repin it at COMMIT {head} (vy-verifier-pin) and rerun"),
    ({"busy": ["v20261024T000000Z-0000"]}, "vy-auditor-1 runs ['v20261024T000000Z-0000']: one unit at a time"),
], ids=["used-period", "lane-open", "wrong-key", "host-elsewhere-on-main", "host-off-main", "host-busy"])
def test_the_probe_refuses_before_any_write(tmp_path, repo, kw, why):
    at = {"head": repo[1], "parent": git(repo[0], "rev-parse", "HEAD~1")}
    p, out, calls = probe(tmp_path, repo, **{k: v.format(**at) if isinstance(v, str) else v for k, v in kw.items()})
    assert p.returncode == 3, p.stdout + p.stderr
    doc = json.loads((out / "probe.json").read_text())
    assert doc["refused"] == [why.format(**at)]
    assert doc["main_commits"] == 2 and doc["host_pinned_on_main"] == (kw.get("pinned") in (None, "{parent}"))
    assert [c.split()[-1] for c in calls] == ["pubkey", "pins", "ledger", "status"]


def test_a_circuit_not_commits_is_refused_before_any_request(tmp_path, repo):
    p, out, calls = probe(tmp_path, repo, circuit="85f38fae")
    assert p.returncode == 3 and "CIRCUIT 85f38fae is not COMMIT's" in p.stdout
    assert calls == []


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_a_commit_off_mains_first_parent_line_is_refused_before_any_request(tmp_path, repo, dry):
    side = off_main(repo)
    p, out, calls = probe(tmp_path, side, pinned=side[1], dry=dry)
    assert p.returncode == 3
    assert f"COMMIT {side[1]} is not on main's first-parent line, so no session would credit" in p.stdout
    assert calls == [] and not (out / "probe.json").exists()


def test_no_main_line_to_check_commit_on_refuses_before_any_request(tmp_path, repo):
    git(repo[0], "remote", "remove", "origin")
    p, out, calls = probe(tmp_path, repo)
    assert p.returncode == 3 and "can't read origin/main's first-parent history" in p.stdout
    assert calls == []


def test_a_dry_probe_sends_nothing(tmp_path, repo):
    p, out, calls = probe(tmp_path, repo, dry=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert calls == []
    assert [line.split(" ", 2)[2] for line in (out / "gate.requests").read_text().splitlines()] == \
        ["pubkey", "pins", "ledger", "status"]
