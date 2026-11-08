"""The witnesses against the auditor ledger v1, ported to v2 as the refusals they now meet, where they hit the auditor
host's code: the gate, the ledger and its publisher.

* A root gets one session, whatever period a v1 open named for it, and an open's `by` is its gate key's lane; `serve.py` is
  told the window's root (its refusal of a salt after any other root is `test_serve.py`'s).
* `end` counts the coins each family released, so a client that takes the salt and one of three round coins and then closes
  ends `complete: false`.
* A restored key state signs at a leaf the ledger has passed, which `append` and `verify_ledger` refuse; a restored key state
  and ledger fork the chain, which `extends` and `publish.sh` refuse.
* The gate's checks are whole-string, so a trailing newline opens nothing, and a root is kept in lowercase.

The verifier's halves of the first three, and one `verify_window` joining the ledger, the end's seal of out/ and the key pin,
are the verifier's tests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs import challenger as Ch, remote as R
from verity.ml.crypto import lms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402

P = lms.Params(64, 4, 4)
T0 = datetime(2026, 10, 20, 9, 30, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 20, 9, 40, 0, tzinfo=timezone.utc)
COMMIT = "ab" * 64
PIN = "c" * 40
SALT, ROUND = "pouw/salt", "pouw/round"
FAMILIES = [[SALT, 1], [ROUND, 3]]
SECRET = bytes(range(32))


@pytest.fixture
def host(tmp_path) -> A.Host:
    (tmp_path / "var" / "sessions").mkdir(parents=True)
    (tmp_path / "peers").write_text("# peers\nvy-challenger-1 research@10.0.0.1\nvy-nebius-1 research@10.0.0.2\n")
    (tmp_path / "coin-ports").write_text("vy-nebius-1 7811-7899\n")
    (tmp_path / "pins.jsonl").write_text(A.pin_line(PIN, "d" * 40, "2026-10-20T09:00:00Z") + "\n")
    return A.Host(root=tmp_path / "var", peers=tmp_path / "peers", pins=tmp_path / "pins.jsonl")


@pytest.fixture
def signer(host):
    return A.key_signer(host, P, workers=1)


def verify_under(host):
    public = (host.key / "public").read_bytes()
    return public[:16], lambda m, sig: lms.verify(P, public, m, lms.Signature.decode(P, sig))


def ssh(host, signer, words, key="vllm", token="ab12", **hooks):
    return A.gate(["--ssh", words, "--key", key], host, now=T0, token=token, own=lambda d: None, start=lambda s: None,
                  active=lambda s: False, secret=SECRET,
                  reserve=lambda s: A.reserve(host, s, signer=signer, pinned=PIN, now=T0),
                  admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=T0, active=lambda s: False), **hooks)


RESERVE = f"reserve --peer vy-nebius-1 --scheme verity/test/coins/v1 --families '{json.dumps(FAMILIES)}'"


def reserve(host, signer, *, token, key="vllm") -> str:
    rc, out = ssh(host, signer, RESERVE, key, token)
    assert rc == 0, out
    return json.loads(out)["session"]


def open_window(host, signer, *, token, commit=COMMIT, key="vllm"):
    session = reserve(host, signer, token=token, key=key)
    rc, out = ssh(host, signer, f"open --session {session} --root {commit}", key)
    assert rc == 0, out
    return session


def v1_open(host, signer, session, period, commit=COMMIT):
    """A v1 open entry, as the host's ledger holds before v2: its window the requester's (period, commit)."""
    I, sign, key_sha512 = signer
    lines = A.read_ledger(host.ledger)
    e = {"schema": A.SCHEMA_V1, "seq": len(lines), "prev": hashlib.sha512(lines[-1]).hexdigest() if lines else A.ZERO,
         "kind": "open", "session": session,
         "body": {"window": {"period": period, "commit": commit}, "key_sha512": key_sha512, "by": "research"}}
    e["sig"] = sign(A.entry_message(I, e)).hex()
    with open(host.ledger, "ab") as f:
        f.write(json.dumps(e, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def serve(host, signer, session, client, tmp_path, monkeypatch) -> dict:
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
    argv = A.serve_argv(host, session, body)
    argv[argv.index("--port") + 1] = "0"
    argv = [sys.executable, str(HERE / "stub_serve.py"), *argv[2:], "--params", "4,4", "--keygen-workers", "1"]
    out = host.sessions / session / "out"

    def wait(proc, seconds):
        address = (out / "stdout.log").read_text().split()[1]
        with R.RemoteChallenger.connect(address, SECRET) as c:
            client(c)
        return proc.wait()

    return A.run(host, session, signer=signer, pinned=PIN, now=lambda: T1, argv=argv, wait=wait)


def test_rt4_int_24_a_root_gets_one_session_whatever_period_a_v1_open_named_and_its_prover_is_the_keys_lane(host, signer):
    v1_open(host, signer, "s20261020T092000Z-0001", "p0")
    held = reserve(host, signer, token="cd34", key="hash")
    for root, why in ((COMMIT, "one session a root, ever"), (COMMIT.upper(), "128 lowercase hex digits")):
        rc, out = ssh(host, signer, f"open --session {held} --root {root}", "hash")
        assert rc == 3 and why in json.loads(out)["refused"]
    vllm = open_window(host, signer, token="ab12", commit="cd" * 64, key="vllm")
    hash_ = open_window(host, signer, token="ef56", commit="ef" * 64, key="hash")

    I, verify = verify_under(host)
    windows = A.verify_ledger(A.read_ledger(host.ledger), I, verify)
    assert list(windows) == [COMMIT, "cd" * 64, "ef" * 64]
    assert [w["open"]["schema"] for w in windows.values()] == [A.SCHEMA_V1, A.SCHEMA, A.SCHEMA]
    assert [w["open"]["body"]["by"] for w in windows.values()] == ["research", "vllm", "hash"]
    assert [w["reserve"] and w["reserve"]["session"] for w in windows.values()] == [None, vllm, hash_]
    for session, root in ((vllm, "cd" * 64), (hash_, "ef" * 64)):
        argv = A.serve_argv(host, session, A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"])
        assert argv[argv.index("--root") + 1] == root


def test_rt4_int_24_the_v1_witness_one_commit_under_two_period_labels_is_a_second_open_for_one_root(host, signer):
    v1_open(host, signer, "s20261020T092000Z-0001", "p0")
    v1_open(host, signer, "s20261020T092000Z-0002", "p0-again")
    I, verify = verify_under(host)
    with pytest.raises(ValueError, match="a second open for root"):
        A.verify_ledger(A.read_ledger(host.ledger), I, verify)


def test_rt4_int_25_a_client_that_takes_the_salt_and_one_round_coin_then_closes_ends_incomplete(host, signer, tmp_path,
                                                                                                 monkeypatch):
    short = open_window(host, signer, token="ab12")

    def dropped(c):
        c.open(SALT, 0, bytes.fromhex(COMMIT))
        c.open(ROUND, 0, hashlib.sha512(b"round 0").digest())

    end = serve(host, signer, short, dropped, tmp_path, monkeypatch)
    assert end["body"]["closed"] is True and end["body"]["ended"] == "closed by the client"
    assert end["body"]["released"] == {SALT: 1, ROUND: 1} and end["body"]["complete"] is False
    out = host.sessions / short / "out"
    assert json.loads((out / "challenger.json").read_text())["ended"] is None

    full = open_window(host, signer, token="cd34", commit="cd" * 64)

    def window(c):
        c.open(SALT, 0, bytes.fromhex("cd" * 64))
        for s in range(3):
            c.open(ROUND, s, hashlib.sha512(b"round %d" % s).digest())

    end = serve(host, signer, full, window, tmp_path, monkeypatch)
    assert end["body"]["released"] == {SALT: 1, ROUND: 3} and end["body"]["complete"] is True
    I, verify = verify_under(host)
    assert A.verify_ledger(A.read_ledger(host.ledger), I, verify)["cd" * 64]["end"] == end


def test_the_open_answer_status_and_end_carry_the_coin_commitment_the_hello_answers(host, signer, tmp_path, monkeypatch):
    """The serving side checks, before its first coin, that the challenger it reached is its session's."""
    session = reserve(host, signer, token="ab12")
    rc, out = ssh(host, signer, f"open --session {session} --root {COMMIT}", ready=lambda s: "ef" * 64)
    assert rc == 0 and json.loads(out)["commitment"] == "ef" * 64
    assert A.await_commitment(host, session, active=lambda s: False) is None
    hello = []
    end = serve(host, signer, session, lambda c: hello.append(c.commitment.root.hex()), tmp_path, monkeypatch)
    assert end["body"]["commitment"] == hello[0] == A.await_commitment(host, session, active=lambda s: True)
    rc, out = ssh(host, signer, f"status {session}")
    assert rc == 0 and json.loads(out)["commitment"] == hello[0]


def test_a_withheld_sessions_gate_gives_no_coin_commitment_root_and_fetches_no_record_before_its_end(host, signer, tmp_path,
                                                                                                    monkeypatch):
    """A sampled-proofs window's session (`reserve --withhold`): its reserve and open say `withheld: true` (so
    `challenger.withheld_on` takes its window), serve.py runs it `--withhold`, and its hello answers the layout alone.
    Until its end entry the gate hands out no root: the open's answer, though `ready` has one, and `status` carry
    `commitment` null, and `fetch` refuses, though serve.py has written out/record.json. Once its end is in, `status`
    gives the root the challenger released at the client's close, and `fetch` the directory with its record."""
    rc, out = ssh(host, signer, RESERVE + " --withhold")
    assert rc == 0 and json.loads(out)["withheld"] is True, out
    session = json.loads(out)["session"]
    rc, out = ssh(host, signer, f"open --session {session} --root {COMMIT}", ready=lambda s: "ef" * 64)
    got = json.loads(out)
    assert rc == 0 and got["withheld"] is True and got["signed"] is True and got["commitment"] is None, got
    assert "ef" * 64 not in out.decode()
    lines = A.read_ledger(host.ledger)
    s = A.sessions_of(lines)[session]
    assert s["reserve"]["body"]["withheld"] is True and s["open"]["body"]["withheld"] is True
    assert Ch.withheld_on([ln.decode() for ln in lines], COMMIT)
    assert A.serve_argv(host, session, s["open"]["body"])[-1] == "--withhold"

    seen = {}

    def client(c):
        seen.update(client=c, hello=(c.withheld, c.commitment))
        c.open(SALT, 0, bytes.fromhex(COMMIT))
        seen["record"] = (host.sessions / session / "out" / R.RECORD).is_file()
        seen["status"] = json.loads(ssh(host, signer, f"status {session}")[1])
        seen["fetch"] = ssh(host, signer, f"fetch {session}")

    end = serve(host, signer, session, client, tmp_path, monkeypatch)
    assert seen["hello"] == (True, None) and seen["record"]
    assert seen["status"]["end"] is None and seen["status"]["commitment"] is None, seen["status"]
    rc, out = seen["fetch"]
    assert rc == 3 and "withheld, and its end is not in the ledger" in json.loads(out)["refused"]

    released = seen["client"].commitment.root.hex()
    assert end["body"]["closed"] is True and end["body"]["commitment"] == released
    rc, out = ssh(host, signer, f"status {session}")
    assert rc == 0 and json.loads(out)["commitment"] == released
    rc, out = ssh(host, signer, f"fetch {session}")
    assert rc == 0
    with tarfile.open(fileobj=io.BytesIO(out)) as tar:
        assert f"{session}/out/{R.RECORD}" in tar.getnames()
    I, verify = verify_under(host)
    assert A.verify_ledger(A.read_ledger(host.ledger), I, verify)[COMMIT]["end"] == end


def test_rt4_int_30_a_trailing_newline_or_a_period_opens_nothing_and_an_uppercase_root_is_refused(host, signer, capsys):
    rest = f"--scheme verity/test/coins/v1 --families '{json.dumps(FAMILIES)}'"
    for words, why in ((f"reserve --peer 'vy-nebius-1\n' {rest}", "--peer"),
                       (f"reserve --period 'p0\n' --peer vy-nebius-1 {rest}", "no --commit, --period"),
                       (f"reserve --period p0 --peer vy-nebius-1 {rest}", "no --commit, --period"),
                       (f"reserve --commit {COMMIT} --peer vy-nebius-1 {rest}", "no --commit, --period")):
        rc, out = ssh(host, signer, words)
        assert rc == 3 and why in json.loads(out)["refused"], words
    assert A.read_ledger(host.ledger) == [] and capsys.readouterr() == ("", "")
    session = reserve(host, signer, token="ab12")
    for words in (f"open --session {session} --root '{COMMIT}\n'", f"open --session {session} --root '{COMMIT} '",
                  f"open --session '{session}\n' --root {COMMIT}", f"open --session {session} --root {COMMIT.upper()}"):
        rc, out = ssh(host, signer, words)
        assert rc == 3 and ("128 lowercase hex digits" in json.loads(out)["refused"]
                            or "--session" in json.loads(out)["refused"]), words
    assert len(A.read_ledger(host.ledger)) == 1
    rc, out = ssh(host, signer, f"open --session {session} --root {COMMIT}")
    assert rc == 0 and json.loads(out)["root"] == COMMIT


def snapshot(host, to: Path, ledger: bool) -> list[Path]:
    saved = [host.key / "state.json", host.key / "state.json.journal", *([host.ledger] if ledger else [])]
    to.mkdir()
    for p in saved:
        shutil.copy2(p, to / p.name)
    return saved


def restore(saved: list[Path], frm: Path) -> None:
    for p in saved:
        os.chmod(p, 0o644)
        shutil.copy2(frm / p.name, p)


def test_rt4_int_27_a_restored_key_state_signs_at_a_used_leaf_which_append_and_the_check_refuse(host, signer, tmp_path):
    open_window(host, signer, token="ab12")
    saved = snapshot(host, tmp_path / "key-only", ledger=False)
    open_window(host, signer, token="cd34", commit="cd" * 64)
    head = A.read_ledger(host.ledger)
    restore(saved, tmp_path / "key-only")

    rc, out = ssh(host, signer, RESERVE, token="ef56")
    assert rc == 3 and "not after the ledger's last entry's" in json.loads(out)["refused"]
    assert A.read_ledger(host.ledger) == head

    restore(saved, tmp_path / "key-only")
    I, sign, _ = signer
    e = {"schema": A.SCHEMA, "seq": 4, "prev": hashlib.sha512(head[-1]).hexdigest(), "kind": "open",
         "session": "s20261020T093000Z-0a0b", "body": {"window": {"root": "ef" * 64}}}
    e["sig"] = sign(A.entry_message(I, e)).hex()
    forged = [*head, json.dumps(e, sort_keys=True, separators=(",", ":")).encode()]
    _, verify = verify_under(host)
    with pytest.raises(ValueError, match="entry 4: signed at leaf 2, not after the last entry's 3"):
        A.verify_ledger(forged, I, verify)


def test_rt4_int_27_a_record_leaf_outside_its_session_is_refused(host, signer):
    session = open_window(host, signer, token="ab12")
    I, sign, _ = signer
    opened = A.leaf_of(A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["sig"])
    A.append(host, "end", session, lambda lines: {"record_leaf": opened}, I, sign)
    _, verify = verify_under(host)
    with pytest.raises(ValueError, match="record leaf 1 isn't between"):
        A.verify_ledger(A.read_ledger(host.ledger), I, verify)


def test_rt4_int_27_a_restored_key_state_and_ledger_fork_the_chain_and_neither_the_check_nor_the_publisher_takes_both(
        host, signer, tmp_path):
    open_window(host, signer, token="ab12")
    saved = snapshot(host, tmp_path / "both", ledger=True)
    open_window(host, signer, token="cd34", commit="cd" * 64)
    head_a = A.read_ledger(host.ledger)
    os.chmod(host.ledger, 0o644)
    restore(saved, tmp_path / "both")
    open_window(host, signer, token="ef56", commit="ef" * 64)
    head_b = A.read_ledger(host.ledger)

    I, verify = verify_under(host)
    assert len(A.verify_ledger(head_a, I, verify)) == len(A.verify_ledger(head_b, I, verify)) == 2
    assert not A.extends(head_a, head_b) and not A.extends(head_b, head_a)
    assert A.extends(head_a[:1], head_a) and A.extends(head_a, head_a)

    log = tmp_path / "published.log"
    published = publish(tmp_path, head_a[:1], log)
    assert published.returncode == 0 and kinds(log) == ["put", "label"]
    assert publish(tmp_path, head_a, log).returncode == 0
    refused = publish(tmp_path, head_b, log)
    assert refused.returncode == 4 and "doesn't extend the last published head" in refused.stderr
    assert kinds(log) == ["put", "label"] * 2
    assert publish(tmp_path, head_a[:1], log).returncode == 4


def kinds(log: Path) -> list[str]:
    return [line.split()[0] for line in log.read_text().splitlines()]


def publish(tmp_path: Path, lines: list[bytes], log: Path) -> subprocess.CompletedProcess:
    """publish.sh's pass against a host whose ledger is `lines`, with ssh and research stubbed: ssh drains its stdin, as the
    real one does, and serves `ledger` and `fetch SESSION` (from `host-ledger.sessions/`); `research data put` prints the
    file's sha256 as its art and logs it, `label` logs."""
    bin_dir = tmp_path / "bin"
    if not bin_dir.exists():
        bin_dir.mkdir()
        (bin_dir / "ssh").write_text(
            '#!/bin/sh\ncat > /dev/null\nfor a; do w=$v; v=$a; done\n[ "$v" = ledger ] && exec cat "$LEDGER"\n'
            '[ "$w" = fetch ] && exec tar -czf - -C "$LEDGER.sessions" "$v"\nexit 9\n')
        (bin_dir / "research").write_text(
            '#!/bin/sh\nif [ "$2" = put ]; then while [ $# -gt 0 ]; do case $1 in --file|--tree) f=$2;; esac; shift; done\n'
            '  echo "put $f" >> "$PUBLISHED"; sha256sum "$f" | cut -c1-64\nelse echo "label $*" >> "$PUBLISHED"; fi\n')
        for p in bin_dir.iterdir():
            p.chmod(0o755)
    (tmp_path / "host-ledger").write_bytes(b"".join(ln + b"\n" for ln in lines))
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "LEDGER": str(tmp_path / "host-ledger"),
           "PUBLISHED": str(log), "VY_CHALLENGER_STATE": str(tmp_path / "state")}
    return subprocess.run(["bash", str(HERE.parent / "publish.sh"), "research@10.0.0.1"], env=env, capture_output=True,
                          text=True)


def test_publish_puts_every_session_that_ended_since_its_last_pass(tmp_path):
    def entry(kind, session):
        return json.dumps({"schema": A.SCHEMA, "kind": kind, "session": session, "body": {"key_sha512": "k"}}).encode()

    sessions = ["s20261020T105206Z-9738", "s20261020T105232Z-1281"]
    for s in sessions:
        (tmp_path / "host-ledger.sessions" / s).mkdir(parents=True)
        (tmp_path / "host-ledger.sessions" / s / "request.json").write_text("{}")
    log = tmp_path / "published.log"
    done = publish(tmp_path, [entry(k, s) for s in sessions for k in ("open", "end")], log)
    assert done.returncode == 0, done.stderr
    assert [ln.split()[1].rsplit("/", 1)[1] for ln in log.read_text().splitlines()[2:]] == sessions
    assert (tmp_path / "state" / "sessions").read_text().split() == sessions
