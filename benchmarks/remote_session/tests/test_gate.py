"""`gate.sh probe`: before a gate run writes anything to either auditor device, it reads the challenger host's key, pins
and ledger and the verifier host's status, and refuses (exit 3, the reason in probe.json) a key that isn't PIN, a used
period, a period of the lane still open, a host pinned elsewhere or busy, and, before any request, a CIRCUIT that isn't COMMIT's
and a COMMIT off origin/main's first-parent line (or no line to check it on). A host pinned elsewhere on that line is
told to rerun at its pin; one pinned off it, to repin. Under DRY=1 nothing reaches ssh. The gates are a stub ssh on
PATH; COMMIT is the head of a throwaway repository whose bare origin's main holds it and its parent.

`gate.sh host`: the run it asks the verifier host for is PACED's paced sessions, or with SESSIONS that many sessions of
one coin a window, and never both: every verb refuses both forms at once, and `form` does nothing else.

The two hosts are CHALLENGER and AUDITOR, by default research@ the addresses the environment holds: a verb that
finds neither ends naming the variable, before any request."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "gate.sh"
AUDITOR, WHATIF, MAIN_LINE = (GATE.parent / n for n in ("auditor.sh", "gate_whatif.sh", "main_line.sh"))
URL = "https://user:tok@127.0.0.1:9/x.git"
FP = "SHA256:s8UdOpJZGMXJR1DjgmOyTsWpTqQX7GDJq/yQCYhSJhs"
PUBLIC = bytes(range(80))
PIN = hashlib.sha512(PUBLIC).hexdigest()
#: the two hosts' addresses as the launcher's environment holds them, here documentation addresses
HOSTS = {"VY_CHALLENGER_1_HOST": "192.0.2.21", "VY_AUDITOR_1_HOST": "192.0.2.22"}


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
    (r / "verity/protocols/accounting/poub/verifier").mkdir(parents=True)
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(r, "init", "-q", "-b", "main")
    for x in ("0", "1"):
        (r / "verity/protocols/accounting/poub/verifier/store.py").write_text(f"x = {x}\n")
        git(r, "add", ".")
        git(r, "commit", "-q", "-m", f"c{x}")
    git(r, "remote", "add", "origin", str(origin))
    git(r, "push", "-q", "origin", "main")
    return r, git(r, "rev-parse", "HEAD")


def off_main(repo: tuple[Path, str]) -> tuple[Path, str]:
    """The repository checked out at a commit on top of main that origin/main doesn't hold."""
    r, _ = repo
    git(r, "checkout", "-q", "-b", "side")
    (r / "verity/protocols/accounting/poub/verifier/store.py").write_text("x = 2\n")
    git(r, "commit", "-q", "-am", "side")
    return r, git(r, "rev-parse", "HEAD")


def probe(tmp_path: Path, repo: tuple[Path, str], *, ledger=CLOSED_7, pinned=None, busy=(), public=PUBLIC,
          period="ma-remote-coins-8", circuit=None, dry=False, verb="probe",
          **more: str) -> tuple[subprocess.CompletedProcess, Path, list[str]]:
    """gate.sh VERB (default probe) against stub gates answering from these records, with MORE set; its result, $OUT
    and the calls ssh got."""
    r, head = repo
    g, out, bin_ = tmp_path / "gates", tmp_path / "out", tmp_path / "bin"
    for d in (g, out, bin_):
        d.mkdir()
    (g / "pubkey").write_text(json.dumps({"W": 4, "H": 10, "public": public.hex()}))
    (g / "pins").write_text(json.dumps({"commit": head, "main": head}) + "\n")
    (g / "ledger").write_text("".join(json.dumps(e) + "\n" for e in ledger))
    (g / "status").write_text(json.dumps({"pinned": pinned or head, "busy": list(busy), "pins": [], "runs": []}))
    (bin_ / "ssh").write_text(f'#!/bin/sh\nfor a; do last=$a; done\necho "$*" >> {g}/calls\ncat "{g}/$last"\n')
    (bin_ / "ssh-keygen").write_text(f'#!/bin/sh\necho "# Host 192.0.2.7 found: line 1"\necho "192.0.2.7 ED25519 {FP}"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}", "OUT": str(out), "COMMIT": head,
           "PERIOD": period, "PIN": PIN, "GATE_KEY": str(tmp_path / "key"), "AUDITOR_FP": FP,
           "AUDITOR": "research@192.0.2.7", "CHALLENGER": "research@192.0.2.8"}
    if circuit:
        env["CIRCUIT"] = circuit
    if dry:
        env["DRY"] = "1"
    env |= more
    p = subprocess.run(["bash", str(GATE), verb], cwd=r, env=env, capture_output=True, text=True)
    calls = (g / "calls").read_text().splitlines() if (g / "calls").exists() else []
    return p, out, calls


def test_the_probe_passes_reading_only(tmp_path, repo):
    p, out, calls = probe(tmp_path, repo)
    assert p.returncode == 0, p.stdout + p.stderr
    tree = subprocess.run(["git", "rev-parse", f"{repo[1]}:verity/protocols/accounting/poub/verifier"], cwd=repo[0],
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
    ({"pinned": "{parent}"}, "the verifier host is pinned at {parent}, not COMMIT {head}: rerun with COMMIT={parent}"),
    ({"pinned": "9b062d027f612c9f39faae87da61fcc554c84b81"},
     "the verifier host is pinned at 9b062d027f612c9f39faae87da61fcc554c84b81, off main's first-parent line, where no "
     "session credits: repin it at COMMIT {head} (vy-verifier-pin) and rerun"),
    ({"busy": ["v20261024T000000Z-0000"]}, "the verifier host runs ['v20261024T000000Z-0000']: one unit at a time"),
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
    assert p.returncode == 3 and "can't fetch main's first-parent history from (no origin)" in p.stdout
    assert calls == []


def userinfo_origin(tmp_path: Path, monkeypatch) -> None:
    """A git on PATH ahead of the real one that answers origin's URL with URL, a credential as its userinfo, and fails
    the fetch from it naming the URL whole, as a git that doesn't redact it would: no git config holds the URL (an
    agent's VM refuses a remote URL with a credential in it)."""
    stub = tmp_path / "git-stub"
    stub.mkdir()
    (stub / "git").write_text(
        f'#!/bin/sh\n[ "$1 $2 $3" = "remote get-url origin" ] && {{ echo "{URL}"; exit 0; }}\n'
        f'for a; do [ "$a" = "{URL}" ] && {{ echo "fatal: unable to access \'{URL}/\': refused" >&2; exit 128; }}; done\n'
        f'exec {shutil.which("git")} "$@"\n')
    (stub / "git").chmod(0o755)
    monkeypatch.setenv("PATH", str(stub), prepend=os.pathsep)


def holds_no_userinfo(p: subprocess.CompletedProcess, out: Path) -> bool:
    texts = (p.stdout, p.stderr, *(f.read_text(errors="replace") for f in out.rglob("*") if f.is_file()))
    return not any("user" in t or "tok" in t for t in texts)


def test_a_refused_fetch_prints_no_userinfo_of_origins_url_anywhere(tmp_path, repo, monkeypatch):
    """On an agent's VM origin's URL carries a credential as its userinfo: neither the refusal nor any file under $OUT
    holds it, while the refusal still names the URL without it."""
    userinfo_origin(tmp_path, monkeypatch)
    p, out, calls = probe(tmp_path, repo)
    assert p.returncode == 3, p.stdout + p.stderr
    assert "can't fetch main's first-parent history from https://127.0.0.1:9/x.git (fetch.probe.log)" in p.stdout
    assert (out / "fetch.probe.log").exists() and calls == []
    assert holds_no_userinfo(p, out)


def test_unauth_takes_out_every_urls_userinfo_and_nothing_else(tmp_path):
    """The log is stripped too, whatever git writes into it, for any git version that names the URL whole."""
    lines = "fatal: unable to access 'https://user:tok@h/x.git/': no\nfrom ssh://git@h:22/x and /srv/git/verity.git\n"
    p = subprocess.run(["bash", "-c", f'source {MAIN_LINE}; unauth'], input=lines,
                       capture_output=True, text=True, check=True)
    assert p.stdout == "fatal: unable to access 'https://h/x.git/': no\nfrom ssh://h:22/x and /srv/git/verity.git\n"


def run_clone(tmp_path: Path, repo: tuple[Path, str], *, at: str) -> tuple[Path, Path]:
    """A run's clone as research makes one on a node: the node's bare repository (the fixture's origin, its HEAD naming
    no branch, as `git init --bare` leaves it there), a source root cloned from it while its main stood at `at` (`remote.py`'s
    `_git_ship_script`), and the run's clone of that source root (`clone.py`'s `make`), detached at the fixture's head.
    The bare's main is the fixture's head again afterwards, as a later launch's push leaves it."""
    r, head = repo
    bare, root, clone = tmp_path / "origin.git", tmp_path / "src", tmp_path / "clone"
    git(bare, "symbolic-ref", "HEAD", "refs/heads/ship")
    git(bare, "update-ref", "refs/heads/main", at)
    git(tmp_path, "clone", "-q", "--shared", "--no-checkout", str(bare), str(root))
    git(root, "checkout", "-q", "--detach", head)
    git(bare, "update-ref", "refs/heads/main", head)
    git(tmp_path, "clone", "-q", "--shared", "--no-checkout", str(root), str(clone))
    git(clone, "checkout", "-q", "--detach", head)
    return root, clone


def test_a_run_clone_reads_main_from_the_bare_repository_not_its_source_roots_frozen_copy(tmp_path, repo):
    """The source root's origin/main stood at the head's parent when it was made; the bare's main has the head since."""
    parent = git(repo[0], "rev-parse", "HEAD~1")
    root, clone = run_clone(tmp_path, repo, at=parent)
    assert git(root, "rev-parse", "refs/remotes/origin/main") == parent
    p, out, calls = probe(tmp_path, (clone, repo[1]))
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"main from {tmp_path / 'origin.git'} at {repo[1]}" in p.stdout
    assert (out / "main.probe.txt").read_text().split() == [repo[1], parent]
    assert [c.split()[-1] for c in calls] == ["pubkey", "pins", "ledger", "status"]


def test_a_run_clone_whose_bare_repository_cant_be_reached_refuses_and_never_reads_the_frozen_copy(tmp_path, repo):
    """The source root's frozen origin/main holds COMMIT, so a fallback to it would pass: the probe refuses instead."""
    root, clone = run_clone(tmp_path, repo, at=repo[1])
    gone = tmp_path / "gone.git"
    git(root, "remote", "set-url", "origin", str(gone))
    p, out, calls = probe(tmp_path, (clone, repo[1]))
    assert p.returncode == 3, p.stdout + p.stderr
    assert f"can't fetch main's first-parent history from {gone} (fetch.probe.log)" in p.stdout
    assert "relaunch from a checkout fetched just before (git fetch origin main)" in p.stdout
    assert calls == [] and not (out / "main.probe.txt").exists()


def audit(tmp_path: Path, r: Path, script: Path = AUDITOR) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """auditor.sh (or gate_whatif.sh) from the clone r's root, as gate.sh's audit and gate_run.sh's whatif run them,
    against a gate serving an empty ledger and pins log and a uv standing in for the store and auditor_line.py; its
    result, $OUT and the file of the commands uv got (absent where it got none)."""
    run, out, bin_, uv = tmp_path / "host-run", tmp_path / "aud-out", tmp_path / "aud-bin", tmp_path / "uv.calls"
    for d in (run / "out", out, bin_):
        d.mkdir(parents=True)
    (bin_ / "ssh").write_text("#!/bin/sh\nexit 0\n")
    (bin_ / "uv").write_text(f'#!/bin/sh\necho "$*" >> {uv}\necho "--main FILE"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {**os.environ, "OUT": str(out), "RUN": str(run), "GATE": "research@192.0.2.8", "GATE_KEY": str(tmp_path / "k"),
           "PERIOD": "ma-remote-coins-8", "PIN": PIN, "COMMIT": git(r, "rev-parse", "HEAD"), "RUN_ART": "art:r",
           "AUDITOR_ART": "art:a", "PROVER_ART": "art:p"}
    env["PATH"] = f"{bin_}{os.pathsep}{env['PATH']}"
    return subprocess.run(["bash", str(script)], cwd=r, env=env, capture_output=True, text=True), out, uv


def test_the_auditor_reads_main_from_the_bare_repository_not_its_source_roots_frozen_copy(tmp_path, repo):
    """The fire's audit runs auditor.sh in the run's clone: its --main is the bare's main, as the probe's is."""
    parent = git(repo[0], "rev-parse", "HEAD~1")
    _, clone = run_clone(tmp_path, repo, at=parent)
    p, out, uv = audit(tmp_path, clone)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "main.txt").read_text().split() == [repo[1], parent]
    assert f"main from {tmp_path / 'origin.git'}, 2 first-parent commits" in (out / "fetch.log").read_text()
    assert f"--main {out / 'main.txt'}" in uv.read_text()


def test_an_auditor_whose_bare_repository_cant_be_reached_refuses_and_writes_no_line(tmp_path, repo):
    """The source root's frozen origin/main holds COMMIT: the auditor refuses rather than check pins against it."""
    root, clone = run_clone(tmp_path, repo, at=repo[1])
    gone = tmp_path / "gone.git"
    git(root, "remote", "set-url", "origin", str(gone))
    p, out, uv = audit(tmp_path, clone)
    assert p.returncode == 3, p.stdout + p.stderr
    assert (f"can't fetch main's first-parent history from {gone} (fetch.log), so no session's pin can be checked on "
            "it: no line was written") in p.stdout
    assert not uv.exists() and not (out / "main.txt").exists() and not (out / "auditor_line.json").exists()


@pytest.mark.parametrize("script", [AUDITOR, WHATIF], ids=["auditor", "whatif"])
def test_a_refused_fetch_of_main_by_the_auditor_or_the_whatif_prints_no_userinfo_anywhere(tmp_path, repo, monkeypatch,
                                                                                            script):
    userinfo_origin(tmp_path, monkeypatch)
    p, out, uv = audit(tmp_path, repo[0], script)
    assert p.returncode == 3, p.stdout + p.stderr
    assert "can't fetch main's first-parent history from https://127.0.0.1:9/x.git (fetch.log)" in p.stdout
    assert "--host-bytes" not in (uv.read_text() if uv.exists() else "") and not (out / "main.txt").exists()
    assert holds_no_userinfo(p, out)


def test_a_dry_probe_sends_nothing(tmp_path, repo):
    p, out, calls = probe(tmp_path, repo, dry=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert calls == []
    assert [line.split(" ", 2)[2] for line in (out / "gate.requests").read_text().splitlines()] == \
        ["pubkey", "pins", "ledger", "status"]


def fire(tmp_path: Path, status: str, lease_by: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    """gate.sh fire with LEASE_BY against a stub ssh that answers every request with the prover node's `gpu-lease status`
    STATUS; its result and the calls ssh got. Fire opens nothing for a prover lease that began after LEASE_BY or still
    waits 2 minutes past it."""
    out, bin_ = tmp_path / "out", tmp_path / "bin"
    for d in (out, bin_):
        d.mkdir()
    (tmp_path / "status").write_text(status)
    (bin_ / "ssh").write_text(f'#!/bin/sh\necho "$*" | tr "\\n" " " >> {tmp_path}/calls\necho >> {tmp_path}/calls\n'
                              f'cat {tmp_path}/status\n')
    (bin_ / "ssh").chmod(0o755)
    env = {**os.environ, **HOSTS, "PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}", "OUT": str(out), "PROVER_RUN": "r1",
           "PROVER_SSH": "research@n2", "ENGINES": "8", "LEASE_BY": lease_by, "GATE_KEY": str(tmp_path / "key"),
           "COMMIT": "0" * 40, "PERIOD": "ma-remote-coins-9", "PIN": PIN}
    p = subprocess.run(["bash", str(GATE), "fire"], cwd=tmp_path, env=env, capture_output=True, text=True)
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return p, calls


def held(since: str) -> str:
    return "".join(f"{i} held 1m pid=7 run=r1 who=w uuid=GPU-{i} since={since} until=2026-10-24T14:25:00Z\n"
                   for i in range(8))


@pytest.mark.parametrize("status, lease_by, why", [
    (held("2026-10-24T13:15:01Z"), "2026-10-24T13:15Z",
     "r1's lease began at 2026-10-24T13:15:01Z, after LEASE_BY 2026-10-24T13:15Z"),
    ("0 free\nwaiting 30m pid=7 n=8 run=r1 who=w since=2019-12-31T23:30:00Z max=100\n", "2020-01-01T00:00Z",
     "r1 holds 0 of 8 GPUs, 2 minutes past LEASE_BY 2020-01-01T00:00Z"),
], ids=["leased-late", "still-waiting"])
def test_fire_opens_nothing_for_a_lease_past_lease_by(tmp_path, status, lease_by, why):
    p, calls = fire(tmp_path, status, lease_by)
    assert p.returncode == 3, p.stdout + p.stderr
    assert f"{why}: nothing opened" in p.stdout
    assert len(calls) == 1 and "gpu-lease status" in calls[0]
    assert (tmp_path / "out" / "gate.requests").read_text() == ""


def test_fire_goes_on_for_a_lease_begun_by_lease_by(tmp_path):
    p, calls = fire(tmp_path, held("2026-10-24T12:45:02Z"), "2026-10-24T13:15Z")
    assert re.search(r"^\S+Z leased$", p.stdout, re.M) and "nothing opened" not in p.stdout, p.stdout
    assert len(calls) == 1 and "gpu-lease status" in calls[0]


FORM = ("PACED", "PACE_PASS_MS", "SESSIONS", "WINDOWS", "WINDOW_MS", "SAMPLES")
REQUEST = ("MODEL", "BIND", "ENGINES", "ARENA_BLOCKS", "PORT", "MINUTES", "JITTER_MS", "QUANTILE", "PROVER_NODE", "AUDITOR")
PACED = ("--paced 1:60000:33273:2 1:60000:33273:2 1:1000:2773:7 1:1000:2773:7 --pace-pass-ms 85 --minutes 30 "
         "--jitter-ms 50 --quantile max --model QWEN25_32B_BF16 --engines 8 --arena-blocks 16252928")


def host(tmp_path: Path, verb: str = "host", **form: str) -> tuple[subprocess.CompletedProcess, list[str] | None]:
    """gate.sh VERB under DRY=1 with these of FORM's variables and no other: its result, and the requests it would
    have sent, each without its time and its host (None when it wrote no gate.requests). `form` gets no OUT."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in FORM + REQUEST + ("OUT", "CHALLENGER")}
    env |= {**HOSTS, "DRY": "1", "PERIOD": "ma-remote-coins-9", "PIN": PIN, "GATE_KEY": str(tmp_path / "key"),
            "COMMIT": "0" * 40, **form}
    if verb != "form":
        env["OUT"] = str(out)
    p = subprocess.run(["bash", str(GATE), verb], cwd=tmp_path, env=env, capture_output=True, text=True)
    sent = out / "gate.requests"
    return p, [line.split(" ", 2)[2] for line in sent.read_text().splitlines()] if sent.exists() else None


@pytest.mark.parametrize("form, want", [
    ({}, PACED),
    ({"PACED": "64:60000:33273:7", "PACE_PASS_MS": "46"},
     "--paced 64:60000:33273:7 --pace-pass-ms 46 --minutes 30 --jitter-ms 50 --quantile max --model QWEN25_32B_BF16 "
     "--engines 8 --arena-blocks 16252928"),
    ({"SESSIONS": "1", "WINDOWS": "7", "WINDOW_MS": "60000", "SAMPLES": "33273"},
     "--sessions 1 --windows 7 --window-ms 60000 --samples 33273 --minutes 30 --jitter-ms 50 --quantile max "
     "--model QWEN25_32B_BF16 --engines 8 --arena-blocks 16252928"),
    ({"SESSIONS": "2"},
     "--sessions 2 --minutes 30 --jitter-ms 50 --quantile max --model QWEN25_32B_BF16 --engines 8 "
     "--arena-blocks 16252928"),
    ({"SESSIONS": "1", "WINDOWS": "7", "WINDOW_MS": "60000", "SAMPLES": "33273", "BIND": "0"},
     "--sessions 1 --windows 7 --window-ms 60000 --samples 33273 --minutes 30 --jitter-ms 50 --quantile max"),
    ({"BIND": "0"}, PACED.removesuffix(" --model QWEN25_32B_BF16 --engines 8 --arena-blocks 16252928")),
    ({"BIND": "1", "MODEL": "TINY"}, PACED.replace("QWEN25_32B_BF16", "TINY")),
], ids=["default", "paced", "one-coin-a-window", "sessions-alone", "one-coin-a-window-unbound", "paced-unbound",
        "bound-named"])
def test_the_hosts_run_is_asked_for_paced_sessions_or_with_sessions_for_one_coin_a_window(tmp_path, form, want):
    """The request names one form: PACED's words at PACE_PASS_MS, by default the four paced sessions, or SESSIONS
    sessions with the WINDOWS, WINDOW_MS and SAMPLES that are set and nothing of a pace. With BIND=1, the default, it
    binds the store to MODEL with its engines' arenas; with BIND=0 it names no model, engines or arenas, so the
    verifier holds no checkpoint."""
    p, sent = host(tmp_path, **form)
    assert p.returncode == 0, p.stdout + p.stderr
    assert sent == [f"run --peer vy-nebius-2 --port 7951 --period ma-remote-coins-9 --pin {PIN} {want}"]


@pytest.mark.parametrize("verb", ["host", "probe", "fire", "form"])
@pytest.mark.parametrize("form, why", [
    ({"SESSIONS": "1", "PACED": "1:1000:2773:7"}, "the host's run takes one form, so unset SESSIONS or both of the others"),
    ({"SESSIONS": "1", "PACE_PASS_MS": "85"}, "the host's run takes one form, so unset SESSIONS or both of the others"),
    ({"WINDOWS": "7"}, "SESSIONS is unset: set it, or unset them for PACED's sessions"),
    ({"WINDOW_MS": "60000", "SAMPLES": "33273"}, "SESSIONS is unset: set it, or unset them for PACED's sessions"),
], ids=["sessions-and-paced", "sessions-and-a-pace", "windows-alone", "a-window-alone"])
def test_both_forms_at_once_or_a_window_without_sessions_is_refused_before_any_request(tmp_path, verb, form, why):
    p, sent = host(tmp_path, verb, **form)
    assert p.returncode == 2 and why in p.stdout, p.stdout + p.stderr
    assert sent is None


@pytest.mark.parametrize("bind, model", [({}, " --model QWEN25_32B_BF16"), ({"BIND": "1", "MODEL": "TINY"}, " --model TINY"),
                                         ({"BIND": "0"}, "")], ids=["default", "bound", "unbound"])
def test_the_period_signs_the_model_only_when_the_run_binds_the_store_to_it(tmp_path, bind, model):
    """`open` asks for `period-open` with MODEL where the host's run binds the store to it (BIND=1, the default), and
    with no model under BIND=0, whose period the auditor's line then finds no model to bind the store to in."""
    p, sent = host(tmp_path, "open", CIRCUIT="ab" * 64, **bind)
    assert p.returncode == 0, p.stdout + p.stderr
    assert sent == ["ledger", f"period-open --name ma-remote-coins-9 --cap 4 --circuit {'ab' * 64}{model}", "ledger"]


@pytest.mark.parametrize("verb", ["host", "probe", "fire", "form"])
@pytest.mark.parametrize("bind", ["2", "yes", ""], ids=["two", "a-word", "set-empty"])
def test_a_bind_other_than_0_or_1_is_refused_before_any_request(tmp_path, verb, bind):
    """An empty BIND is the default, 1, as every unset variable of gate.sh's is."""
    p, sent = host(tmp_path, verb, BIND=bind)
    if bind == "":
        assert verb != "host" or p.returncode == 0, p.stdout + p.stderr
        assert verb != "host" or "--model QWEN25_32B_BF16" in sent[0]
        return
    assert p.returncode == 2 and f"BIND is {bind}: 1 binds the store to MODEL's pinned checkpoint, 0 to none" in p.stdout
    assert sent is None


@pytest.mark.parametrize("verb", ["probe", "open", "host", "close", "audit", "fire"])
@pytest.mark.parametrize("unset, given", [("VY_CHALLENGER_1_HOST", "CHALLENGER"), ("VY_AUDITOR_1_HOST", "AUDITOR")])
def test_a_host_with_no_address_in_the_environment_ends_the_verb_naming_the_variable(tmp_path, verb, unset, given):
    """The script holds neither host's address: without the variable and without the host given whole, every verb but
    `form` ends at once, the variable's name in what it prints, and no request is written; with the host given whole
    the variable isn't read."""
    p, sent = host(tmp_path, verb, **{unset: ""})
    assert p.returncode != 0 and f"{unset}: the address of the" in p.stderr and f"or {given} as USER@HOST" in p.stderr
    assert sent is None
    p, sent = host(tmp_path / "given", "open", CIRCUIT="ab" * 64, **{unset: "", given: "research@192.0.2.23"})
    assert p.returncode == 0 and len(sent) == 3, p.stdout + p.stderr


def test_form_reads_no_hosts_address(tmp_path):
    p, sent = host(tmp_path, "form", VY_CHALLENGER_1_HOST="", VY_AUDITOR_1_HOST="")
    assert p.returncode == 0 and sent is None, p.stdout + p.stderr


def test_a_dry_fire_that_cant_read_the_provers_leases_says_so_once_and_writes_every_request(tmp_path, repo):
    """A stopped prover node answers no lease status: a dry fire says so, asks once and sends nothing, so the
    requests it would send in the session are all in gate.requests."""
    p, out, calls = probe(tmp_path, repo, dry=True, verb="fire", PROVER_SSH="research@192.0.2.9", PROVER_RUN="r1",
                          ENGINES="8", BIND="0", SESSIONS="1", WINDOWS="7", WINDOW_MS="60000", SAMPLES="33273")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "DRY: can't read vy-nebius-2's leases; would wait until r1 holds 8" in p.stdout
    asked = " ".join(calls)
    assert asked.count("BatchMode") == 1 and "gpu-lease status" in asked, \
        "one read of the leases, and no request"
    sent = [line.split(" ", 2)[2] for line in (out / "gate.requests").read_text().splitlines()]
    assert sent[:4] == ["pubkey", "pins", "ledger", "status"]
    (run,) = [r for r in sent if r.startswith("run ")]
    assert run.endswith("--sessions 1 --windows 7 --window-ms 60000 --samples 33273 --minutes 30 --jitter-ms 50 "
                        "--quantile max")
    (opened,) = [r for r in sent if r.startswith("period-open --name ma-remote-coins-8 ")]
    assert "--model" not in opened, "a period of an unbound run signs no model"
    assert any(r.startswith("period-close") for r in sent)


@pytest.mark.parametrize("form", [{}, {"PACED": "1:1000:2773:7", "PACE_PASS_MS": "46"}, {"SESSIONS": "1"},
                                  {"SESSIONS": "1", "WINDOWS": "7", "WINDOW_MS": "60000", "SAMPLES": "33273"}],
                         ids=["default", "paced", "sessions-alone", "one-coin-a-window"])
def test_form_passes_one_form_and_writes_and_asks_nothing(tmp_path, form):
    p, sent = host(tmp_path, "form", **form)
    assert (p.returncode, p.stdout, p.stderr, sent) == (0, "", "", None)
    assert list((tmp_path / "out").iterdir()) == []
