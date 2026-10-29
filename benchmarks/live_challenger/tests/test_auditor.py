"""The auditor host's service (`auditor.py`) on a test key, with the gate's root steps (chown, systemctl) and the clock injected:
a window is reserved, then opened at its root (one session a root), its entries signed before serve.py runs; the session's end entry seals its out/
and counts the coins each family released; a session whose pin moved doesn't run; the gate refuses what it can't check; and the
reference check refuses a ledger that was edited, cut, reordered or given a second session for a root. The
witnesses against v1 are `test_auditor_rt4.py`'s."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import shlex
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs import challenger as C
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402

P = lms.Params(64, 4, 4)
T0 = datetime(2026, 10, 20, 9, 30, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 20, 9, 40, 0, tzinfo=timezone.utc)
COMMIT = "ab" * 64
PIN = "c" * 40
FAMILIES = [["positions", 2], ["replay", 1]]


@pytest.fixture
def host(tmp_path) -> A.Host:
    (tmp_path / "var" / "sessions").mkdir(parents=True)
    (tmp_path / "peers").write_text("# peers\nvy-challenger-1 research@10.0.0.1\nvy-nebius-1 research@10.0.0.2\n")
    (tmp_path / "coin-ports").write_text("# peer, its certifier coin link's ports\nvy-nebius-1 7811-7899\n")
    (tmp_path / "pins.jsonl").write_text(A.pin_line(PIN, "d" * 40, "2026-10-20T09:00:00Z") + "\n")
    return A.Host(root=tmp_path / "var", peers=tmp_path / "peers", pins=tmp_path / "pins.jsonl")


@pytest.fixture
def signer(host):
    return A.key_signer(host, P, workers=1)


def verify_under(host):
    public = (host.key / "public").read_bytes()
    return public[:16], lambda m, sig: lms.verify(P, public, m, lms.Signature.decode(P, sig))


SECRET = bytes(range(32))


RESERVE = ["reserve", "--peer", "vy-nebius-1", "--scheme", "verity/test/coins/v1", "--families", json.dumps(FAMILIES)]


def call(host, signer, argv, *, by="vllm", token="ab12", active=lambda s: False, now=T0):
    return A.gate(argv, host, by=by, now=now, token=token, own=lambda p: None, start=lambda s: None, active=active,
                  secret=SECRET, reserve=lambda s: A.reserve(host, s, signer=signer, pinned=PIN, now=now),
                  admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=now, active=active))


def reserve(host, signer, *extra, by="vllm", token="ab12", now=T0) -> str:
    rc, out = call(host, signer, RESERVE + list(extra), by=by, token=token, now=now)
    assert rc == 0, out
    return json.loads(out)["session"]


def open_window(host, signer, *extra, token="ab12", active=lambda s: False, commit=COMMIT, by="vllm"):
    """reserve, then open at `commit`: the open's (rc, stdout)."""
    session = reserve(host, signer, *extra, by=by, token=token)
    return call(host, signer, ["open", "--session", session, "--root", commit], by=by, active=active)


def test_a_window_is_reserved_then_opened_and_its_open_binds_session_and_root_before_it_serves(host, signer):
    rc, out = call(host, signer, RESERVE)
    assert rc == 0 and json.loads(out) == {"session": "s20261020T093000Z-ab12", "seq": 0, "peer": "vy-nebius-1",
                                           "by": "vllm", "expires_utc": "2026-10-20T11:30:00Z", "period": None}
    session = json.loads(out)["session"]
    assert not (host.sessions / session / "secret").exists()
    rc, out = call(host, signer, ["open", "--session", session, "--root", COMMIT], now=T1)
    doc = json.loads(out)
    assert rc == 0 and doc == {"session": session, "port": 7811, "peer": "vy-nebius-1", "seq": 1, "reserve": 0,
                               "host": f"vy-challenger-1.{session}", "root": COMMIT, "by": "vllm", "opened_by": "vllm",
                               "secret": SECRET.hex(), "commitment": None}
    I, verify = verify_under(host)
    windows = A.verify_ledger(A.read_ledger(host.ledger), I, verify)
    assert list(windows) == [COMMIT] and windows[COMMIT]["end"] is None
    r, e = windows[COMMIT]["reserve"], windows[COMMIT]["open"]
    assert (r["kind"], r["seq"], r["session"], e["kind"], e["session"]) == ("reserve", 0, session, "open", session)
    assert r["body"] == {"by": "vllm", "peer": "vy-nebius-1", "scheme": "verity/test/coins/v1", "epoch": 0,
                         "families": FAMILIES, "minutes": 120, "pinned": PIN, "key_sha512": e["body"]["key_sha512"],
                         "host": f"vy-challenger-1.{session}", "started_utc": "2026-10-20T09:30:00Z",
                         "expires_utc": "2026-10-20T11:30:00Z"}
    assert e["body"]["window"] == {"root": COMMIT} and e["body"]["reserve"] == 0 and e["body"]["pinned"] == PIN
    assert {k: e["body"][k] for k in ("by", "peer", "scheme", "epoch", "families", "minutes")} == {
        k: r["body"][k] for k in ("by", "peer", "scheme", "epoch", "families", "minutes")}
    assert e["body"]["opened_by"] == "vllm" and e["body"]["started_utc"] == "2026-10-20T09:40:00Z"
    assert e["body"]["secret_sha512"] == hashlib.sha512(SECRET).hexdigest() and "secret" not in json.dumps(e).replace(
        "secret_sha512", "")
    assert (host.sessions / session / "secret").read_text() == SECRET.hex() + "\n"
    for f in ("request.json", A.OPEN_REQUEST):
        assert SECRET.hex() not in (host.sessions / session / f).read_text()
    assert e["body"]["key_sha512"] == hashlib.sha512((host.key / "public").read_bytes()).hexdigest()
    assert r["prev"] == A.ZERO and e["prev"] == hashlib.sha512(A.read_ledger(host.ledger)[0]).hexdigest()

    other = reserve(host, signer, by="kernels", token="cd34")
    rc, out = call(host, signer, ["open", "--session", other, "--root", COMMIT], by="kernels")
    assert rc == 3 and "one session a root, ever" in json.loads(out)["refused"]
    assert not (host.sessions / other / "secret").exists() and not (host.sessions / other / A.OPEN_REQUEST).exists()

    # an open that passed the gate's look at the ledger before another took the root is refused by admit, under its lock
    (host.sessions / other / A.OPEN_REQUEST).write_text(json.dumps(
        {"session": other, "root": COMMIT, "by": "kernels", "secret_sha512": "0" * 128}))
    with pytest.raises(A.Refused, match="one session a root"):
        A.admit(host, other, signer=signer, pinned=PIN, now=T0, active=lambda s: False)
    assert len(A.read_ledger(host.ledger)) == 3

    rc, out = open_window(host, signer, token="0a0b", commit="cd" * 64, active=lambda s: True)
    assert rc == 0 and json.loads(out)["port"] == 7812
    rc, out = open_window(host, signer, token="0c0d", commit="ef" * 64, active=lambda s: False)
    assert rc == 0 and json.loads(out)["port"] == 7811


def test_an_open_is_refused_unless_its_session_is_reserved_unopened_unexpired_and_the_openers_own(host, signer):
    def opens(session, root, by="vllm", now=T0):
        return call(host, signer, ["open", "--session", session, "--root", root], by=by, now=now)

    rc, out = opens("s20261020T093000Z-ffff", COMMIT)
    assert rc == 3 and "has no reserve in the ledger: reserve first" in json.loads(out)["refused"]
    mine = reserve(host, signer, "--minutes", "10")
    rc, out = opens(mine, COMMIT, by="kernels")
    assert rc == 3 and f"session {mine} is vllm's reserve, not kernels's" in json.loads(out)["refused"]
    rc, out = opens(mine, COMMIT, now=datetime(2026, 10, 20, 9, 40, 1, tzinfo=timezone.utc))
    assert rc == 3 and "reserve expired at 2026-10-20T09:40:00Z" in json.loads(out)["refused"]
    rc, out = opens(mine, COMMIT, now=datetime(2026, 10, 20, 9, 40, 0, tzinfo=timezone.utc))
    assert rc == 0, out
    rc, out = opens(mine, "cd" * 64)
    assert rc == 3 and "is opened already: a reserve opens once" in json.loads(out)["refused"]

    dev = reserve(host, signer, "--for", "kernels", by="vy-admin", token="cd34")
    rc, out = call(host, signer, ["open", "--session", dev, "--root", "cd" * 64], by="hash")
    assert rc == 3 and "kernels's reserve, not hash's" in json.loads(out)["refused"]
    rc, out = call(host, signer, ["open", "--session", dev, "--root", "cd" * 64], by="kernels")
    assert rc == 0 and json.loads(out)["by"] == "vy-admin" and json.loads(out)["for"] == "kernels"
    assert json.loads(out)["opened_by"] == "kernels"
    I, verify = verify_under(host)
    assert len(A.verify_ledger(A.read_ledger(host.ledger), I, verify)) == 2
    rc, out = call(host, signer, ["status", dev])
    assert rc == 0 and json.loads(out)["reserve"]["body"]["for"] == "kernels" and not json.loads(out)["expired"]


def test_a_session_serves_from_the_pinned_tree_and_its_end_entry_seals_its_out(host, signer, tmp_path, monkeypatch):
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    session = json.loads(open_window(host, signer)[1])["session"]
    body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
    argv = A.serve_argv(host, session, body)
    argv[argv.index("--port") + 1] = "0"
    argv = [sys.executable, str(HERE / "stub_serve.py"), *argv[2:], "--params", "4,4", "--keygen-workers", "1"]
    out = host.sessions / session / "out"

    def client(proc, seconds):
        assert seconds == 120 * 60
        address = (out / "stdout.log").read_text().split()[1]
        with pytest.raises(C.CoinError, match="the session's secret"):
            R.RemoteChallenger.connect(address)
        R.RemoteChallenger.connect(address, SECRET).close()
        return proc.wait()

    end = A.run(host, session, signer=signer, pinned=PIN, now=lambda: T1, argv=argv, wait=client)
    assert end["body"]["rc"] == 0 and end["body"]["closed"] and end["body"]["released"] == {"positions": 0, "replay": 0}
    assert end["body"]["refused_hellos"] == 1 and "hellos.jsonl" in end["body"]["files"]

    session = json.loads(open_window(host, signer, token="cd34", commit="cd" * 64)[1])["session"]
    body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
    argv = A.serve_argv(host, session, body)
    assert argv[argv.index("--root") + 1] == "cd" * 64 and argv[argv.index("--secret-file") + 1].endswith(f"{session}/secret")
    argv[argv.index("--port") + 1] = "0"
    argv = [sys.executable, str(HERE / "stub_serve.py"), *argv[2:], "--params", "4,4", "--keygen-workers", "1"]
    out = host.sessions / session / "out"

    def client(proc, seconds):
        assert seconds == 120 * 60
        address = (out / "stdout.log").read_text().split()[1]
        with R.RemoteChallenger.connect(address, SECRET) as c:
            record = C.LiveRecord(c, "positions", itertools.count().__next__)
            record.record(bytes(64))
        return proc.wait()

    end = A.run(host, session, signer=signer, pinned=PIN, now=lambda: T1, argv=argv, wait=client)
    assert end["body"]["rc"] == 0 and end["body"]["closed"] and end["body"]["ended_utc"] == "2026-10-20T09:40:00Z"
    assert end["body"]["released"] == {"positions": 1, "replay": 0} and end["body"]["complete"] is False
    record_leaf = int(json.loads((out / R.RECORD).read_text())["signature"][:8], 16)
    assert end["body"]["record_leaf"] == record_leaf and A.leaf_of(end["sig"]) > record_leaf
    assert not (host.sessions / session / "secret").exists()
    assert set(end["body"]["files"]) >= {"record.json", "key.json", "releases.jsonl", "challenger.json"}
    assert all(hashlib.sha512((out / n).read_bytes()).hexdigest() == h for n, h in end["body"]["files"].items())

    I, verify = verify_under(host)
    assert A.verify_ledger(A.read_ledger(host.ledger), I, verify)["cd" * 64]["end"] == end
    key = C.ChallengerKey.from_json(json.loads((out / R.KEY).read_text()))
    assert key.public == (host.key / "public").read_bytes()
    commitment, roots = R.trusted_session(out, [host.ledger], {"root": "cd" * 64}, key,
                                          lambda k, m, s: lms.verify(P, k.public, m, lms.Signature.decode(P, s)))
    assert roots == {"positions": (bytes(64),), "replay": ()} and commitment.root.hex() == end["body"]["commitment"]
    assert json.loads((out / R.RECORD).read_text())["host"] == body["host"] == f"vy-challenger-1.{session}"

    rc, tgz = A.gate(["fetch", session], host)
    with tarfile.open(fileobj=io.BytesIO(tgz)) as tar:
        assert {f"{session}/request.json", f"{session}/out/record.json", f"{session}/out/releases.jsonl"} <= set(tar.getnames())
        assert json.loads(tar.extractfile(f"{session}/out/challenger.json").read())["session"] == session
    rc, st = A.gate(["status", session], host, active=lambda s: True)
    assert rc == 0 and json.loads(st)["end"] == end and not json.loads(st)["active"]
    with pytest.raises(A.Refused, match="has ended"):
        A.run(host, session, signer=signer, pinned=PIN, now=lambda: T1, argv=argv, wait=client)


def test_a_session_whose_pin_moved_after_its_admission_does_not_run(host, signer):
    session = json.loads(open_window(host, signer)[1])["session"]
    end = A.run(host, session, signer=signer, pinned="d" * 40, now=lambda: T1, argv=["/nonexistent/serve"])
    assert end["body"]["rc"] == 2 and not end["body"]["closed"] and "the pin moved" in end["body"]["ended"]
    assert end["body"]["files"] == {} and not (host.sessions / session / "out").exists()
    with pytest.raises(A.Refused, match="not a pinned tree"):
        A.pinned_commit(Path("/workspace"))
    assert A.pinned_commit(Path("/opt/vy-challenger") / PIN) == PIN


def test_each_open_seals_the_pins_log_so_the_verifiers_pinned_session_takes_the_session(host, signer):
    session = json.loads(open_window(host, signer)[1])["session"]
    pins = A.read_pins(host.pins)
    body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
    assert body["pins"] == C.pins_seal(pins) == {"lines": 1, "sha512": C.log_digest(pins)}
    assert C.pins_reason(session, body, pins) is None
    A.record_pin(host.pins, "e" * 40, "e" * 40, T1, seal=lambda p: None)
    assert C.pins_reason(session, body, A.read_pins(host.pins)) is None
    assert "doesn't record as a commit of main's history" in C.pins_reason(session, {**body, "pinned": "e" * 40},
                                                                            A.read_pins(host.pins))


@pytest.mark.parametrize("log, why", [
    ("", "records no pin, so no session here would count"),
    (C.pin_line("e" * 40, "e" * 40, "2026-10-20T09:00:00Z") + "\n", "doesn't record as a commit of main's history"),
    ('{"commit": "' + PIN + '"}\n', "pins log line 0 is malformed"),
])
def test_a_reserve_or_an_open_is_refused_unless_the_pins_log_records_this_trees_pin(host, signer, log, why):
    good = host.pins.read_text()
    host.pins.write_text(log)
    rc, out = call(host, signer, RESERVE)
    assert rc == 3 and why in json.loads(out)["refused"] and A.read_ledger(host.ledger) == []
    host.pins.write_text(good)
    session = reserve(host, signer, token="cd34")
    host.pins.write_text(log)
    rc, out = call(host, signer, ["open", "--session", session, "--root", COMMIT])
    assert rc == 3 and why in json.loads(out)["refused"] and len(A.read_ledger(host.ledger)) == 1
    assert not (host.sessions / session / "secret").exists()


def test_a_pin_is_logged_append_only_in_the_verifiers_form_or_refused(tmp_path):
    log, sealed = tmp_path / "pins.jsonl", []
    assert A.record_pin(log, PIN, "d" * 40, T0, seal=sealed.append) == C.pin_line(PIN, "d" * 40, "2026-10-20T09:30:00Z")
    A.record_pin(log, "e" * 40, "f" * 40, T1, seal=sealed.append)
    assert A.read_pins(log) == [C.pin_line(PIN, "d" * 40, "2026-10-20T09:30:00Z"),
                                C.pin_line("e" * 40, "f" * 40, "2026-10-20T09:40:00Z")]
    assert sealed == [log, log] and log.stat().st_mode & 0o777 == 0o644
    for commit, main in ((PIN.upper(), "d" * 40), (PIN, "d" * 39), (PIN + "\n", "d" * 40), ("", "")):
        with pytest.raises(A.Refused, match="pin"):
            A.record_pin(log, commit, main, T1, seal=sealed.append)
    assert len(A.read_pins(log)) == 2 and len(sealed) == 2


@pytest.mark.parametrize("bad, why", [
    (["--commit", COMMIT], "unrecognized arguments: --commit"),
    (["--peer", "vy-challenger-1"], "--peer"),
    (["--peer", "vy-nebius-9"], "--peer"),
    (["--peer", "vy-nebius-1\n"], "--peer"),
    (["--families", "[]"], "--families"),
    (["--families", '[["positions", 0]]'], "--families"),
    (["--families", '[["positions", 1], ["positions", 2]]'], "--families"),
    (["--families", '[["positions\\n", 1]]'], "--families"),
    (["--families", "not json"], "--families"),
    (["--period", "p0"], "no --commit, --period or --lane"),
    (["--lane", "vllm"], "no --commit, --period or --lane"),
    (["--minutes", "0"], "--minutes"),
    (["--minutes", "721"], "--minutes"),
    (["--epoch", "-1"], "--epoch"),
    (["--scheme", "Has Caps"], "--scheme"),
    (["--scheme", "verity/test/coins/v1\n"], "--scheme"),
    (["--for", "kernels"], "unrecognized arguments: --for"),
])
def test_the_gate_refuses_a_reserve_it_cant_check(host, signer, bad, why):
    good = {"--peer": "vy-nebius-1", "--scheme": "verity/test/coins/v1", "--families": json.dumps(FAMILIES)}
    good.update(dict(zip(bad[::2], bad[1::2])))
    argv = ["--ssh", shlex.join(["reserve", *itertools.chain.from_iterable(good.items())]), "--key", "vllm"]
    rc, out = A.gate(argv, host, now=T0, token="ab12", own=lambda p: None, active=lambda s: False,
                     reserve=lambda s: pytest.fail("reserved"))
    assert rc == 3 and why in json.loads(out)["refused"]
    assert A.read_ledger(host.ledger) == [] and list(host.sessions.iterdir()) == []


@pytest.mark.parametrize("ports", [None, "", "vy-nebius-2 7811-7899\n", "vy-nebius-1\n", "vy-nebius-1 7811-7899 x\n",
                                   "vy-nebius-1 7810-7820\n", "vy-nebius-1 7890-7900\n", "vy-nebius-1 7820-7811\n",
                                   "vy-nebius-1 78a1\n"])
def test_a_reserve_for_a_peer_without_a_certifier_coin_link_is_refused(host, signer, ports):
    """A window's coins reach its peer only through the peer's certifier coin link (`CoinsConfidential`), so a peer
    with no line of /etc/vy/coin-ports, or one whose ports aren't the gate's, gets no window."""
    if ports is None:
        host.coin_ports.unlink()
    else:
        host.coin_ports.write_text(ports)
    rc, out = call(host, signer, RESERVE)
    assert rc == 3 and "has no certifier coin link" in json.loads(out)["refused"] and A.read_ledger(host.ledger) == []


def test_an_open_takes_a_port_of_its_peers_coin_link_and_its_serve_py_opens_no_tunnel(host, signer):
    host.coin_ports.write_text("vy-nebius-1 7850-7851\n")
    live: set[str] = set()
    for token, commit, port in (("ab12", COMMIT, 7850), ("cd34", "cd" * 64, 7851)):
        rc, out = open_window(host, signer, token=token, commit=commit, active=live.__contains__)
        assert rc == 0 and json.loads(out)["port"] == port
        live.add(json.loads(out)["session"])
    session = min(live)
    rc, out = open_window(host, signer, token="ef56", commit="ef" * 64, active=live.__contains__)
    assert rc == 3 and "every port of 7850..7851 has a live session" in json.loads(out)["refused"]

    argv = A.serve_argv(host, session, A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"])
    assert argv[argv.index("--port") + 1] == "7850" and not {"--peer", "--peers"} & set(argv) and "--root" in argv

    reserved = reserve(host, signer, token="0a0b")
    host.coin_ports.write_text("")
    rc, out = call(host, signer, ["open", "--session", reserved, "--root", "0a" * 64])
    assert rc == 3 and "has no certifier coin link" in json.loads(out)["refused"]


@pytest.mark.parametrize("bad, why", [
    (["--root", "ab" * 32], "128 lowercase hex digits"),
    (["--root", "g" * 128], "128 lowercase hex digits"),
    (["--root", COMMIT + "\n"], "128 lowercase hex digits"),
    (["--root", " " + COMMIT], "128 lowercase hex digits"),
    (["--root", COMMIT.upper()], "128 lowercase hex digits"),
    (["--session", "s20261020T093000Z-ab12\n"], "--session"),
    (["--session", "../../etc"], "--session"),
    (["--commit", COMMIT], "unrecognized arguments: --commit"),
])
def test_the_gate_refuses_an_open_it_cant_check(host, signer, bad, why):
    good = {"--session": reserve(host, signer), "--root": COMMIT}
    good.update(dict(zip(bad[::2], bad[1::2])))
    rc, out = A.gate(["open", *itertools.chain.from_iterable(good.items())], host, by="vllm", now=T0, own=lambda p: None,
                     start=lambda s: None, active=lambda s: False, admit=lambda s: pytest.fail("admitted"))
    assert rc == 3 and why in json.loads(out)["refused"]
    assert len(A.read_ledger(host.ledger)) == 1


def ssh(host, signer, words, key=None, token="ab12"):
    """The forced command's call: `--ssh WORDS --key NAME` (a lane's key), or `--ssh WORDS` alone (a key with no lane)."""
    argv = ["--ssh", words, *(["--key", key] if key else [])]
    return A.gate(argv, host, by="vy-admin", now=T0, token=token, own=lambda p: None, start=lambda s: None,
                  active=lambda s: False, secret=SECRET,
                  reserve=lambda s: A.reserve(host, s, signer=signer, pinned=PIN, now=T0),
                  admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=T0, active=lambda s: False))


def test_the_gate_takes_its_request_from_ssh_with_the_keys_lane_as_its_prover_and_refuses_any_other_verb(host, signer):
    words = shlex.join(RESERVE)
    for key in (None, "-"):
        for w in (words, f"open --session s20261020T093000Z-ab12 --root {COMMIT}"):
            rc, out = ssh(host, signer, w, key)
            assert rc == 3 and "this key reads only" in json.loads(out)["refused"]
    assert A.read_ledger(host.ledger) == []
    rc, out = ssh(host, signer, words, "kernels")
    assert rc == 0 and json.loads(out)["by"] == "kernels"
    session = json.loads(out)["session"]
    rc, out = ssh(host, signer, f"open --session {session} --root {COMMIT}", "vllm")
    assert rc == 3 and "kernels's reserve, not vllm's" in json.loads(out)["refused"]
    rc, out = ssh(host, signer, f"open --session {session} --root {COMMIT}", "kernels")
    doc = json.loads(out)
    assert rc == 0 and doc["session"] == "s20261020T093000Z-ab12" and doc["by"] == "kernels" and doc["secret"] == SECRET.hex()
    e = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]
    assert e["body"]["by"] == e["body"]["opened_by"] == "kernels"

    rc, tgz = ssh(host, signer, f"fetch {doc['session']}")
    with tarfile.open(fileobj=io.BytesIO(tgz)) as tar:
        names = tar.getnames()
        assert {f"{session}/request.json", f"{session}/{A.OPEN_REQUEST}"} <= set(names) and f"{session}/secret" not in names
        assert all(SECRET.hex().encode() not in tar.extractfile(m).read() for m in tar.getmembers() if m.isfile())
    assert ssh(host, signer, "ledger") == (0, host.ledger.read_bytes())
    assert ssh(host, signer, "pins") == (0, host.pins.read_bytes()) and ssh(host, signer, "pins", "kernels")[0] == 0
    assert json.loads(ssh(host, signer, f"status {doc['session']}")[1])["open"] == e

    for argv in (["--ssh", "open --session 'ab"], ["--ssh", "x" * 8193], ["--ssh", "bash -c id"], ["--ssh", ""],
                 ["--ssh", "ledger", "--key", "a/b"], ["--ssh", "ledger", "--by", "kernels"], ["--ssh", "ledger", "--key"],
                 ["status", "../../etc"], ["status", "s20261020T093000Z-ffff"], ["status", "s20261020T093000Z-ab12\n"],
                 ["fetch", "s20261020T093000Z-ffff"]):
        rc, out = A.gate(argv, host, active=lambda s: False)
        assert rc == 3 and json.loads(out)["refused"]


def test_the_reference_check_refuses_an_edited_cut_reordered_or_doubled_ledger(host, signer, tmp_path):
    for token, commit in (("ab12", COMMIT), ("cd34", "cd" * 64)):
        assert open_window(host, signer, token=token, commit=commit)[0] == 0
    lines = A.read_ledger(host.ledger)
    I, verify = verify_under(host)
    assert len(A.verify_ledger(lines, I, verify)) == 2

    e = json.loads(lines[1])
    e["body"]["port"] = 7899
    edited = [lines[0], json.dumps(e, sort_keys=True, separators=(",", ":")).encode()]
    for bad, why in ((edited, "signature"), (lines[1:], "entry 0"), (lines[::-1], "entry 0")):
        with pytest.raises(ValueError, match=why):
            A.verify_ledger(bad, I, verify)

    _, sign, _ = signer
    first = A.sessions_of(lines)["s20261020T093000Z-ab12"]["open"]["body"]
    A.append(host, "open", "s20261020T093000Z-ef56", lambda ls: first, I, sign)
    with pytest.raises(ValueError, match="a second open"):
        A.verify_ledger(A.read_ledger(host.ledger), I, verify)
    other = A.Host(root=tmp_path / "other", peers=host.peers, pins=host.pins)
    other.root.mkdir()
    A.append(other, "end", "s20261020T093000Z-0000", lambda ls: {}, I, sign)
    with pytest.raises(ValueError, match="no open"):
        A.verify_ledger(A.read_ledger(other.ledger), I, verify)
