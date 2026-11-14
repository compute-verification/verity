"""The gate's warden rehearsal (`warden-rehearse`, `auditor.py`), with its unit, the tunnel and the rehearsal injected:
the gate refuses a request it can't check, one whose minutes the traffic doesn't fit in, a read-only key, and a second
rehearsal (a certifier's or a warden's) while one runs; a lane's own key starts one, whose run (as `auditor`) runs the
network certifier through the tunnel into `witness/`, publishes to `out/` only its deployment, its certificates, a
row registration's among them, and each cells' registration as the witness writes it less its wiring and its cost
(a line the gate's one list of published keys doesn't rule fails the end and publishes no registration), and writes
its end; `fetch` gives `witness/` to the run's own lane and
vy-admin alone."""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import socket
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402

T0 = datetime(2026, 10, 24, 11, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 24, 11, 20, 0, tzinfo=timezone.utc)
PIN = "c" * 40
REQUEST = ["warden-rehearse", "--peer", "vy-nebius-1", "--port", "8011", "--minutes", "20"]
TUNNEL = [sys.executable, "-c", "import signal; signal.pause()"]
LANE = "network-accounting"


@pytest.fixture
def host(tmp_path) -> A.Host:
    (tmp_path / "var").mkdir()
    (tmp_path / "peers").write_text("# peers\nvy-challenger-1 research@10.0.0.1\n"
                                    "vy-nebius-1 -i /var/lib/vy-challenger/.ssh/vy-challenger-tunnel research@10.0.0.2\n")
    return A.Host(root=tmp_path / "var", peers=tmp_path / "peers", pins=tmp_path / "pins.jsonl")


def gate(host, argv, *, key=LANE, active=lambda r: False, started=None, token="ab12", stdin=None, own=None):
    argv = ["--ssh", shlex.join(argv)] + ([] if key is None else ["--key", key])
    return A.gate(argv, host, now=T0, token=token, own=own or (lambda p: None), certifier_active=active,
                  start_certifier=(started.append if started is not None else lambda r: None),
                  read=lambda n: pytest.fail("read stdin") if stdin is None else stdin[:n])


def test_a_lane_key_starts_a_warden_rehearsal_and_one_rehearsal_of_either_device_runs_at_a_time(host):
    started = []
    rc, out = gate(host, REQUEST, started=started)
    doc = json.loads(out)
    assert rc == 0 and doc == {"run": "c20261024T110000Z-ab12", "device": "warden", "peer": "vy-nebius-1", "port": 8011,
                               "minutes": 20, "requests": 100, "height": 20, "temperature": 0.0, "top_p": 1.0,
                               "by": LANE,
                               "requested_utc": "2026-10-24T11:00:00Z",
                               "unit": "vy-certifier@c20261024T110000Z-ab12.service"}
    assert started == [doc["run"]]
    assert json.loads((host.certifier / doc["run"] / "request.json").read_text()) == \
        {k: v for k, v in doc.items() if k != "unit"}
    assert A.read_ledger(host.ledger) == []

    certifier = ["certifier-rehearse", "--peer", "vy-nebius-1", "--port", "8011", "--minutes", "30"]
    for argv in (REQUEST, certifier):
        rc, out = gate(host, argv, active=lambda r: r == doc["run"], started=started, token="cd34")
        assert rc == 3 and "rehearsal c20261024T110000Z-ab12 still runs: one at a time" in json.loads(out)["refused"]
    assert started == [doc["run"]]
    rc, out = gate(host, certifier, started=started, token="cd34")
    assert rc == 0 and "device" not in json.loads(out)
    small = ["--tokens", "64", "--requests", "4", "--minutes", "4", "--height", "10"]
    rc, out = gate(host, REQUEST + small, active=lambda r: r == "c20261024T110000Z-cd34", started=started, token="ef56")
    assert rc == 3 and "c20261024T110000Z-cd34 still runs" in json.loads(out)["refused"]
    rc, out = gate(host, REQUEST + small, started=started, token="ef56")
    assert rc == 0 and {k: json.loads(out)[k] for k in ("requests", "tokens", "minutes", "height")} == \
        {"requests": 4, "tokens": 64, "minutes": 4, "height": 10}


@pytest.mark.parametrize("bad, why", [
    (["--peer", "vy-challenger-1"], "--peer"),
    (["--peer", "vy-nebius-9"], "--peer"),
    (["--port", "80"], "--port '80': 1024 to 65535"),
    (["--port", "08011"], "--port"),
    (["--minutes", "0"], "--minutes '0': 1 to 170"),
    (["--minutes", "171"], "--minutes '171': 1 to 170"),
    (["--minutes", "10"], "--minutes 10: 100 requests take 11 minutes"),
    (["--minutes", "10"], "the check and the key at height 20): give --minutes 11 or more"),
    (["--requests", "4", "--minutes", "7"], "give --minutes 8 or more"),
    (["--requests", "400", "--minutes", "20"], "give --minutes 21 or more, or fewer --requests"),
    (["--minutes", "6", "--height", "10"], "--minutes 6: 100 requests take 7 minutes"),
    (["--requests", "4", "--minutes", "3", "--height", "10"], "give --minutes 4 or more"),
    (["--requests", "400", "--minutes", "16", "--height", "10"], "give --minutes 17 or more, or fewer --requests"),
    *[(["--height", v], f"--height {v!r}: 10 to 20") for v in ("9", "21", "010", "20.0", "H", "")],
    (["--requests", "3"], "--requests '3': 4 to 1700"),
    (["--requests", "1701"], "--requests '1701': 4 to 1700"),
    (["--requests", "1700", "--minutes", "59", "--height", "13"], "--minutes 59: 1700 requests take 60 minutes"),
    (["--tokens", "15"], "--tokens '15': 16 to 256"),
    (["--tokens", "257"], "--tokens '257': 16 to 256"),
    (["--tokens", "+32"], "--tokens '+32': 16 to 256"),
    (["--tamper", "late"], "warden-rehearse: unrecognized arguments: --tamper"),
    (["--tamper", "token"], "warden-rehearse: unrecognized arguments: --tamper"),
    (["--in-flight", "2"], "unrecognized arguments: --in-flight"),
    (["--profile", "served"], "unrecognized arguments: --profile"),
    (["--upstream", "10.0.0.2:8011"], "unrecognized arguments: --upstream"),
    (["--rate", "5"], "unrecognized arguments: --rate"),
    (["--command", "id"], "unrecognized arguments: --command"),
    (["--ids", "1"], "unrecognized arguments: --ids 1"),
    (["--delay", "21"], "unrecognized arguments: --delay"),
    (["--window", "32"], "unrecognized arguments: --window"),
    (["--scan", "1"], "unrecognized arguments: 1"),
    (["--temperature", "2.5"], "--temperature '2.5': 0 to 2"),
    (["--temperature", "-0.1"], "--temperature '-0.1': 0 to 2"),
    (["--temperature", "nan"], "--temperature 'nan': 0 to 2"),
    (["--temperature", "1e-1"], "--temperature '1e-1': 0 to 2"),
    (["--top-p", "0"], "--top-p '0': above 0, at most 1"),
    (["--top-p", "1.01"], "--top-p '1.01': above 0, at most 1"),
    (["--top-p", ".9"], "--top-p '.9': above 0, at most 1"),
    (["--top-p", "0.99999"], "--top-p '0.99999': above 0, at most 1"),
])
def test_the_gate_refuses_a_warden_rehearsal_it_cant_check(host, bad, why):
    good = dict(zip(REQUEST[1::2], REQUEST[2::2]))
    good.update(dict(zip(bad[::2], bad[1::2])))
    rc, out = gate(host, ["warden-rehearse", *[w for kv in good.items() for w in kv]], started=[])
    assert rc == 3 and why in json.loads(out)["refused"]
    assert not host.certifier.exists()


def test_a_key_that_reads_only_starts_no_warden_rehearsal(host):
    why = f"this key reads only: a {REQUEST[0].removesuffix('-rehearse')} rehearsal"
    for key in (None, "-"):
        rc, out = gate(host, REQUEST, key=key, started=[])
        assert rc == 3 and why in json.loads(out)["refused"]
    assert not host.certifier.exists()


def test_the_least_minutes_are_two_seconds_a_request_three_minutes_and_the_key_more_rounded_up():
    """The key is 2^H one-time keys at `KEYGEN_LEAVES_S` a second: 256 s at the deployed height, the default, and
    nothing that rounds up a minute at 10."""
    assert [A.warden_minutes(n, 10) for n in (4, 60, 61, 100, 400)] == [4, 5, 6, 7, 17]
    assert [A.warden_minutes(n) for n in (4, 60, 61, 100, 400)] == [A.warden_minutes(n, 20) for n in
                                                                    (4, 60, 61, 100, 400)] == [8, 10, 10, 11, 21]
    for h in (10, 20):
        assert all(A.warden_minutes(n + 30, h) == A.warden_minutes(n, h) + 1 for n in range(4, A.WARDEN_REQUESTS[1] - 29))


def test_one_warden_session_takes_the_gates_most_requests_at_height_13_in_an_hour(host):
    """The gate's most requests, 1700, at height 13 (8192 one-time keys) take 60 minutes, a scan rehearsal's on its own
    prompts too: one session of about 57 minutes of arrivals and tail, the whole-node wipe's 745 link-windows and more
    (`wipe_gate.sh`)."""
    assert A.WARDEN_REQUESTS[1] == A.WARDEN_PROMPT_REQUESTS[1] == 1700 and A.warden_minutes(1700, 13) == 60
    for token, argv in (("ab12", []), ("cd34", ["--scan", "--prompts", "--tokens", "24"])):
        rc, out = gate(host, REQUEST + argv + ["--requests", "1700", "--minutes", "60", "--height", "13"],
                       stdin=json.dumps([[1]] * A.WARDEN_PROMPTS[1]).encode(), started=[], token=token)
        assert rc == 0, out
        assert {k: json.loads(out)[k] for k in ("requests", "minutes", "height")} == \
            {"requests": 1700, "minutes": 60, "height": 13}


def test_a_warden_key_signs_five_a_link_window_so_1031_windows_take_height_13_and_the_gate_refuses_12(host):
    """A warden window takes `WARDEN_WINDOW_LEAVES` = 5 one-time signatures (three certificates and two cell heads),
    after one for the session's start, so height 12's 4096 sign the start and 819 windows, exactly, and 1031 (55 minutes
    of the scan's 3.2 s windows) take 13; a load's heads come first, and one head more than fits takes the next
    height. 1559 requests at the gate's rate are `warden_windows` 1031 and fit 55 minutes at height 13, which the gate
    takes; at 12 it refuses them with the height that fits, writing nothing, rather than run a session whose signer
    runs out 819 windows in. The default height, 20, signs anything the gate takes."""
    assert [A.warden_height(w) for w in (1, 204, 205, 819, 820, 1031, 1638, 1639)] == [10, 10, 11, 12, 13, 13, 13, 14]
    assert A.warden_height(1031, heads=733) == 13
    assert A.key_signatures(819) == 4096 and A.warden_height(819) == 12 and A.warden_height(819, heads=1) == 13
    assert (A.warden_windows(1559), A.warden_windows(1700), A.warden_windows(1700, scan=False)) == (1031, 1119, 60)
    assert (A.warden_minutes(1559, 13), A.warden_minutes(1700)) == (55, 64)
    long = REQUEST[:-1] + ["55", "--scan", "--prompts", "--requests", "1559", "--height"]
    rc, out = gate(host, long + ["13"], stdin=b"[[1]]", started=[])
    assert rc == 0 and {k: json.loads(out)[k] for k in ("requests", "minutes", "height")} == \
        {"requests": 1559, "minutes": 55, "height": 13}
    rc, out = gate(host, long + ["12"], stdin=b"[[1]]", started=[], token="cd34")
    assert rc == 3 and json.loads(out)["refused"] == (
        "--height 12: 1559 requests take 1031 link-windows, 5156 one-time signatures (the session's start and 5 a "
        "window), past the key's 4096: give --height 13 or more, or fewer --requests")
    assert not (host.certifier / "c20261024T110000Z-cd34").exists()
    rc, out = gate(host, REQUEST[:-1] + ["60", "--scan", "--requests", "1700", "--height", "12"], started=[],
                   token="ef56")
    assert rc == 3 and "give --height 13 or more" in json.loads(out)["refused"]
    rc, out = gate(host, REQUEST[:-1] + ["64", "--scan", "--requests", "1700"], started=[], token="ab34")
    assert rc == 0 and json.loads(out)["height"] == 20


SALT_C = "5a" * 64
CELL_SALTS = bytes(range(192))
#: the stand-in registrations' link
LINK = "warden"
#: the stand-in deployment less where it reached the endpoint, each key one the gate gives out
#: (`DEPLOYMENT_PUBLIC`): its session and what a published line's forms are read from (`deployment_forms`), its
#: link, the schedule's digest, a cell's bytes and an execution phase's buckets, and its key's parameters
DEPLOYMENT = {"session": "00" * 32, "link": LINK, "digest": "31" * 64, "params": {"cell_bytes": 512, "window": 600},
              "key": {"params": [64, 8, 1], "key_id": "0c" * 16}}
#: the forms the stand-in deployment gives, and a signature's hex digits under its key
GIVEN = {"link": (LINK,), "links": (LINK, f"{LINK}-fails"), "parameters": ("31" * 64,), "cell_bytes": (512,),
         "buckets": (600,), "signature": re.compile("[0-9a-f]{%d}" % (2 * A.lms.Params(64, 8, 1).signature_bytes()))}
SIGNED = 2 * A.lms.Params(64, 8, 1).signature_bytes()
#: the registration of an execution phase's egress cells, with `warden_rows.registration`'s keys
REGISTRATION = {"format": "verity-network-warden/registration/v2", "value": f"{LINK}/egress/0", "link": LINK,
                "direction": "egress", "window": 0, "parameters": "31" * 64, "cell_bytes": 512, "rows": 600,
                "words": 1024, "word_bits": 16, "row_schema": "hm96-sha512/row/v2", "domain": "d0" * 64,
                "wiring": "17" * 64, "root": "ab" * 64,
                "layout": {"buckets": 600, "slots": 4, "connection_slots": 4, "first_bucket": 0},
                "certificate": {"key": "0c" * 16, "digest": "9d" * 64, "signature": "e5" * (SIGNED // 2)}}
#: a record's certificate as `certified.jsonl` holds it, with `certificate.Certificate`'s fields
CERTIFICATE = {"digest": "cd" * 64, "direction": "ingress", "key_id": "0c" * 16, "link": LINK,
               "signature": "ef" * (SIGNED // 2), "window": 0}
#: an execution phase's connection rows' registration, with `connection_rows.registration`'s keys: it names no
#: direction
ROWS_REGISTRATION = {"format": "verity-network-certifier/connection-rows-registration/v1", "link": LINK, "window": 0,
                     "parameters": "31" * 64, "word_bits": 16, "row_schema": "rows/v1",
                     "tables": [{"name": "lengths", "value": f"{LINK}/lengths/0", "rows": 48, "words": 4,
                                 "domain": "d1" * 64, "root": "ac" * 64, "layout": {"first_bucket": 0}}],
                     "certificate": {"key": "0c" * 16, "digest": "9e" * 64, "signature": "e6" * (SIGNED // 2)}}
#: a line's cost: a registration's rows and their bytes, and two measured times
COST = {"rows": 600, "row_bytes": 2048, "commit_ms": 41.375, "sign_ms": 7.125}
#: the network certifier's word that an execution phase has no head, with `connection_rows.no_head`'s keys, and its
#: line's cost, one measured time
NO_HEAD = {"format": "verity-network-certifier/connection-rows-no-head-word/v1", "link": LINK, "window": 2,
           "parameters": "31" * 64,
           "certificate": {"key": "0c" * 16, "digest": "9f" * 64, "signature": "e7" * (SIGNED // 2)}}
WORD_COST = {"sign_ms": 3.625}
#: a stand-in for the witness's `check.json` of a rehearsal that served `REQUEST`'s requests, the gate's default
CHECK = {"accepted": True, "problems": [], "summary": {
    "requests": 100, "ended": 100, "refused": 0, "failed": [], "windows": 2, "certified_windows": 6, "certificates": 6,
    "signer_refused": [], "cell_rows": {"registered": 4, "of": 4, "rows": [600], "row_bytes": [2048],
                                        "commit_ms": {"n": 4, "max": 41.375}, "sign_ms": {"n": 4, "max": 7.125}},
    "cells": {"ingress": 9, "egress": 31},
    "wire_bits": {"ingress": 18432, "egress": 63488}, "cells_in_their_bucket": [31, 31], "tick_lateness_ms": {"n": 9},
    "data_bytes": {"egress": 4242}, "completion_tokens": 77, "prompt_tokens": [16, 99], "arrival_lag_ms": {"n": 31},
    "finish_reasons": {"length": 4}}, "connections": [{"ordinal": 1, "outcome": "end", "text_bytes": 300}]}
#: the rehearsal's last line of output, in the form its check is printed in: it ends with the costs' two times
LAST_LINE = ("ACCEPTED: 100/100 ended, 0 refused, failed []; cell rows 4/4 registered, [600] rows a window, "
             "commit ms {'n': 4, 'max': 41.375}, sign ms {'n': 4, 'max': 7.125}")
#: what the rehearsal says on its stderr: free text
STDERR = "the endpoint closed connection 3 early"
#: a stand-in for `warden_serve.py rehearse`'s files, of a run whose every execution phase was certified: public,
#: then witness, then what it prints
WARDEN_FILES = (
    f'(out / "deployment.json").write_text(json.dumps({DEPLOYMENT!r} | {{"upstream": url}}))\n'
    f'(out / "cells.jsonl").write_text(json.dumps({{"registration": {REGISTRATION!r}, "cost": {COST!r}}}) + "\\n")\n'
    f'(out / "rows.jsonl").write_text(json.dumps({{"registration": {ROWS_REGISTRATION!r}, "cost": {COST!r}}}) + "\\n")\n'
    f'(out / "certified.jsonl").write_text(json.dumps({{"salt_c": {SALT_C!r}, "head": {{"root": "77" * 64}}, '
    f'"certificate": {CERTIFICATE!r}}}) + "\\n")\n'
    f'(out / "cells.salts").write_bytes({CELL_SALTS!r})\n'
    '(out / "requests.records").write_bytes(b"GET / prompt")\n'
    '(out / "client.json").write_text("{}")\n'
    f'(out / "check.json").write_text(json.dumps({CHECK!r}))\n'
    f'print({STDERR!r}, file=sys.stderr)\nprint({LAST_LINE!r})\n')
#: the lines a run writes for an execution phase the signer refused, one in each of its three lines' files
REFUSED_LINES = {"cells.jsonl": {"refused": [LINK, "ingress", 1], "withheld": True, "cost": COST},
                 "rows.jsonl": {"refused": [LINK, "egress", 1], "cost": COST},
                 "certified.jsonl": {"refused": [f"{LINK}-fails", "egress", 1]}}
#: after `WARDEN_FILES`, the same run with one execution phase the signer refused
REFUSED_PHASE = "".join(f'with (out / {name!r}).open("a") as f:\n    f.write({json.dumps(line) + chr(10)!r})\n'
                        for name, line in REFUSED_LINES.items())
#: the check of such a run: it does not accept, and each of its counts of the certified is one fewer
REFUSED_CHECK = CHECK | {"accepted": False, "problems": ["egress window 1's cells aren't registered"],
                         "summary": CHECK["summary"] | {
                             "certified_windows": 5, "certificates": 5, "signer_refused": [[LINK, "ingress", 1]],
                             "cell_rows": CHECK["summary"]["cell_rows"] | {"registered": 3}}}
#: the names of a run's lines' files in `out/`
LINE_FILES = {"certificates.jsonl", "cells-certificates.jsonl", "rows-certificates.jsonl", "cells-registrations.jsonl"}
#: the verdict of an end that is not accepted, of `CHECK`'s schedule: none of the counts a refused execution phase
#: lowers (`auditor.ACCEPTED_COUNTS`, `auditor.ACCEPTED_CELL_ROWS`)
NOT_ACCEPTED = {"accepted": False, "failed": [], "windows": 2, "cell_rows": {"of": 4}}


def rehearsal(source: str, seen: list | None = None):
    """A stand-in for `warden_serve.py rehearse`: Python `source` run with `out` and `url` bound."""
    def command(req, out, url):
        if seen is not None:
            seen.append((req, out, url))
        return [sys.executable, "-c",
                f"import json, pathlib, sys; out, url = pathlib.Path({str(out)!r}), {url!r}\n{source}"]
    return command


#: what a tunnel's `ssh` says when the peer's port refuses a forwarded connection: free text, a line a connection
TUNNEL_LINE = "channel 2: open failed: connect failed: Connection refused"


def speaking_tunnel(said: Path) -> list[str]:
    """A stand-in for a tunnel's `ssh` that says `TUNNEL_LINE` on its stderr and then opens the FIFO `said`."""
    return [sys.executable, "-c", f"import signal, sys\ntry:\n    print({TUNNEL_LINE!r}, file=sys.stderr, flush=True)\n"
                                  f"finally:\n    open({str(said)!r}, 'w').close()\nsignal.pause()"]


def run_warden(host, source=WARDEN_FILES, seen=None, token="ab12") -> tuple[str, dict]:
    """The rehearsal run through a tunnel that has said its line before the rehearsal starts: `up` waits on the
    tunnel's opening of the FIFO."""
    rc, out = gate(host, REQUEST, started=[], token=token)
    assert rc == 0, out
    run = json.loads(out)["run"]
    said = host.peers.parent / f"said-{token}"
    os.mkfifo(said)
    end = A.certifier_run(host, run, pinned=PIN, tunnel_argv=lambda s, l: speaking_tunnel(said),
                          up=lambda t, p: Path(said).read_text() == "", command=rehearsal(source, seen),
                          wait=lambda proc, s: proc.wait())
    return run, end


def tar_names(blob: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        return set(tar.getnames())


def ended_why(host, end: dict) -> str:
    """The sentence a run's end's `ended` word stands for, which the gate writes to the run's `witness/` alone."""
    assert end["ended"] in A.ENDED
    return (host.certifier / end["run"] / A.WITNESS / A.ENDED_WHY).read_text().removesuffix("\n")


def end_text(d) -> str:
    """A run's `end.json` as written, which `status` and a `fetch` without the witness give."""
    return (d / "end.json").read_text()


def test_a_rehearsal_writes_witness_apart_and_publishes_its_deployment_and_certificates(host):
    seen = []
    run, end = run_warden(host, WARDEN_FILES, seen)
    d = host.certifier / run
    (req, work, url), = seen
    assert work == d / A.WITNESS and url.startswith("http://127.0.0.1:") and req["device"] == "warden"
    assert (d / A.WITNESS).stat().st_mode & 0o777 == 0o700
    assert end["device"] == "warden" and end["rc"] == 0 and end["ended"] == "ended"
    assert ended_why(host, end) == "the rehearsal ended"
    assert end["pinned"] == PIN and "runs" not in end
    assert end["verdict"] == {"accepted": True, "failed": [], "windows": 2, "certified_windows": 6, "certificates": 6,
                              "cell_rows": {"registered": 4, "of": 4}} and "load" not in end
    assert CHECK["summary"]["signer_refused"] == [] and "signer_refused" not in end_text(d)
    assert set(end) == {"run", "rc", "ended", "pinned", "device", "verdict", "files"}
    assert "registrations" not in end
    assert {p.name for p in (d / A.WITNESS).iterdir()} == {
        "deployment.json", "cells.jsonl", "rows.jsonl", "certified.jsonl", "cells.salts", "requests.records",
        "client.json", "check.json", "rehearse.log", "tunnel.log", A.ENDED_WHY}
    assert "witness" not in end and not any(n in end_text(d) for n in ("cells.salts", "requests.records",
                                                                         "client.json", "rehearse.log"))
    assert set(end["files"]) == {"deployment.json", "certificates.jsonl", "cells-certificates.jsonl",
                                 "rows-certificates.jsonl", "cells-registrations.jsonl"}
    assert sorted((d / A.WITNESS / "rehearse.log").read_text().splitlines()) == sorted([STDERR, LAST_LINE])
    assert (d / A.WITNESS / "tunnel.log").read_text().splitlines() == [TUNNEL_LINE]
    out = d / "out"
    assert json.loads((d / A.WITNESS / "deployment.json").read_text()) == DEPLOYMENT | {"upstream": url}
    assert json.loads((out / "deployment.json").read_text()) == DEPLOYMENT
    assert A.deployment_forms(d / A.WITNESS / "deployment.json") == GIVEN
    assert [json.loads(x) for x in (out / "certificates.jsonl").read_text().splitlines()] == [
        {"certificate": CERTIFICATE}]
    public = b"".join(p.read_bytes() for p in out.rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    for secret in (SALT_C.encode(), b"77" * 64, CELL_SALTS[:16], CELL_SALTS[:16].hex().encode(), b"prompt",
                   b"completion_tokens", b"data_bytes", b"text_bytes", STDERR.encode(), LAST_LINE.encode(),
                   TUNNEL_LINE.encode(), b"open failed"):
        assert secret not in public


def test_an_accepted_end_served_the_requests_its_request_asked_for(host, tmp_path):
    """The end carries no count of the traffic, and a reader takes an accepted run's requests from its request. The
    gate holds the two equal: a check that accepted and counted another number of requests, or none, is not accepted
    in the end, and the run ends nonzero in one word, the two numbers in the witness alone. A check that refused is
    refused as it stands, and a relay session, whose request asks for no number, is held to none."""
    check = CHECK | {"summary": CHECK["summary"] | {"requests": 4, "ended": 4}}
    run, end = run_warden(host, WARDEN_FILES.replace(repr(CHECK), repr(check)))
    assert json.loads((host.certifier / run / "request.json").read_text())["requests"] == 100
    assert (end["rc"], end["ended"], end["verdict"]["accepted"]) == (2, "miscounted", False)
    assert set(end["files"]) == {"deployment.json"}, "a run that isn't accepted publishes none of its lines"
    assert ended_why(host, end) == "check.json accepts and counts 4 requests where the request asked for 100"
    assert "requests" not in end["verdict"] and "counts 4" not in end_text(host.certifier / run)

    out, witness = tmp_path / "out", tmp_path / "witness"
    out.mkdir()
    witness.mkdir()

    def outcome(summary, requests, accepted=True):
        (witness / "check.json").write_text(json.dumps({"accepted": accepted, "problems": [], "summary": summary}))
        return A.warden_outcome(out, witness, requests)

    assert outcome({"requests": 100}, 100) == {"verdict": {"accepted": True}}
    for counted in ({"requests": 99}, {"requests": 0}, {}, {"requests": "100"}, {"requests": 100.0},
                    {"requests": None}):
        got = outcome(counted, 100)
        assert got["verdict"] == {"accepted": False} and "asked for 100" in got["miscounted"], counted
    assert outcome({"requests": True}, 1)["verdict"] == {"accepted": False}
    assert outcome({"requests": 4}, 100, accepted=False) == {"verdict": {"accepted": False}}
    assert outcome({"requests": 4}, None) == {"verdict": {"accepted": True}}
    assert outcome({}, None, accepted="yes") == {"verdict": {"accepted": False}}


def test_the_ends_failed_list_holds_the_gates_fixed_words_and_none_of_a_clients(host):
    """The end's `failed` says how the connections that neither ended nor were refused failed, in `FAILED`'s words
    alone, whatever the check's summary holds: a client's exception's words, which the check may list, are `error`
    in the end and stay whole in the witness's `check.json`. A key with no name reads the end."""
    said = "ConnectionResetError: the endpoint hung up on the third request"
    check = CHECK | {"accepted": False, "problems": [f"client connection 2: {said}"],
                     "summary": CHECK["summary"] | {"ended": 2, "failed": ["cut", said]}}
    run, end = run_warden(host, WARDEN_FILES.replace(repr(CHECK), repr(check)))
    d = host.certifier / run
    assert end["verdict"]["failed"] == ["cut", "error"] and "problems" not in end["verdict"]
    assert end["verdict"]["accepted"] is False
    public = b"".join(p.read_bytes() for p in (d / "out").rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    assert said.encode() not in public and b"hung up" not in public
    assert json.loads((d / A.WITNESS / "check.json").read_text())["summary"]["failed"] == ["cut", said]
    assert A.failed_words(["fail", "cut", "fail"]) == ["cut", "fail"] and A.failed_words([]) == []
    assert A.failed_words(said) == A.failed_words([None]) == A.failed_words([["cut"]]) == ["error"]


def test_of_a_row_registration_the_rehearsal_publishes_its_certificate_and_nothing_else(host):
    """Each line of `cells.jsonl` and of `rows.jsonl` stays whole in `witness/`. Of an accepted run `out/` holds,
    line for line, its certificate under the link, direction and window its registration names (a connection
    rows' the link's egress); and of `cells.jsonl`, each line as the witness writes it less the
    registration's wiring and the line's cost: the same bytes but for those two, so its root, domain, layout and
    certificate are public. No wiring is in `out/` or the end, nor a line's cost, nor the check's summary of the
    costs, nor the line the rehearsal prints them in, nor a root, domain, layout or value of a connection rows'
    registration."""
    run, end = run_warden(host)
    d = host.certifier / run
    out = d / "out"
    whole = [{"registration": REGISTRATION, "cost": COST}]
    witness = (d / A.WITNESS / "cells.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in witness] == whole
    assert not (out / "cells.jsonl").exists() and not (out / "rows.jsonl").exists()
    assert [json.loads(x) for x in (out / "cells-certificates.jsonl").read_text().splitlines()] == [
        {"certificate": {"key_id": "0c" * 16, "link": LINK, "direction": "egress", "window": 0,
                         "digest": "9d" * 64, "signature": "e5" * (SIGNED // 2)}}]
    assert [json.loads(x) for x in (out / "rows-certificates.jsonl").read_text().splitlines()] == [
        {"certificate": {"key_id": "0c" * 16, "link": LINK, "direction": "egress", "window": 0,
                         "digest": "9e" * 64, "signature": "e6" * (SIGNED // 2)}}]
    published = (out / "cells-registrations.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in published] == [
        {"registration": {k: v for k, v in REGISTRATION.items() if k != "wiring"}}]
    assert [list(json.loads(x)["registration"]) for x in published] == [[k for k in REGISTRATION if k != "wiring"]]
    wiring, cost = f'"wiring": "{REGISTRATION["wiring"]}", ', f', "cost": {json.dumps(COST)}'
    assert witness[0].count(wiring) == witness[0].count(cost) == 1
    assert published == [witness[0].replace(wiring, "").replace(cost, "")]
    assert not (out / "rows-registrations.jsonl").exists()
    public = b"".join(p.read_bytes() for p in out.rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    for shown in (REGISTRATION[k] for k in ("value", "parameters", "domain", "root", "row_schema")):
        assert shown.encode() in public, shown
    body = [REGISTRATION["wiring"], *(t[k] for t in ROWS_REGISTRATION["tables"] for k in ("value", "domain", "root"))]
    for held in (*body, "wiring", "tables", "cost", "row_bytes", "commit_ms", "sign_ms", "withheld",
                 "41.375", "7.125", "commit ms", "sign ms"):
        assert held.encode() not in public, held

    rc, st = A.gate(["--ssh", f"status {run}"], host, certifier_active=lambda r: False)
    assert rc == 0 and json.loads(st)["end"] == end


def _line(registration=REGISTRATION, **more) -> str:
    return json.dumps({"registration": registration, "cost": COST} | more)


def _of(registration=REGISTRATION, **keys) -> str:
    """A line whose registration holds `keys` in place of its own, a `layout_` or `certificate_` key in that object."""
    doc = dict(registration)
    for k, v in keys.items():
        under, _, key = k.partition("_") if k.startswith(("layout_", "certificate_")) else ("", "", k)
        doc = doc | ({under: doc[under] | {key: v}} if under else {key: v})
    return _line(doc)


FORM = "isn't in the form ROW_LINE"


@pytest.mark.parametrize("line, why", [
    (_line(REGISTRATION | {"note": "x"}), "the line's registration holds the key 'note', which ROW_LINE names neither"),
    (_line(REGISTRATION | {"layout": REGISTRATION["layout"] | {"lanes": 2}}),
     "the line's registration's layout holds the key 'lanes'"),
    (_line(note="x"), "the line holds the key 'note'"),
    (_of(root={"of": "ab" * 64}), f"the line's registration's root {FORM}"),
    (_of(rows=600.5), f"the line's registration's rows {FORM}"),
    (_line(REGISTRATION | {"layout": 4}), "the line's registration's layout isn't an object of the keys"),
    (json.dumps({"registration": REGISTRATION, "cost": COST}, separators=(",", ":")),
     "isn't one JSON object as json.dumps writes it by default"),
    (_of(format="verity-network-warden/registration/v3"), f"registration's format {FORM}"),
    (_of(row_schema="cells/v1"), f"registration's row_schema {FORM}"),
    (_of(word_bits=8), f"registration's word_bits {FORM}"),
    (_of(link="another", value="another/egress/0"), f"registration's link {FORM}"),
    (_of(value=f"{LINK}/egress/1"), f"registration's value {FORM}"),
    (_of(direction="both", value=f"{LINK}/both/0"), f"registration's direction {FORM}"),
    (_of(parameters="32" * 64), f"registration's parameters {FORM}"),
    (_of(cell_bytes=1024), f"registration's cell_bytes {FORM}"),
    (_of(layout_buckets=601), f"registration's layout's buckets {FORM}"),
    (_of(window=-1, value=f"{LINK}/egress/-1"), f"registration's window {FORM}"),
    (_of(words=True), f"registration's words {FORM}"),
    (_of(layout_first_bucket=1 << 63), f"registration's layout's first_bucket {FORM}"),
    (_of(layout_slots="4"), f"registration's layout's slots {FORM}"),
    (_of(domain="D0" * 64), f"registration's domain {FORM}"),
    (_of(root="ab" * 65), f"registration's root {FORM}"),
    (_of(certificate_key="0c" * 17), f"registration's certificate's key {FORM}"),
    (_of(certificate_digest="9d" * 32), f"registration's certificate's digest {FORM}"),
    (_of(certificate_signature="e5" * (SIGNED // 2 + 1)), f"registration's certificate's signature {FORM}"),
    (_of(certificate_signature="a word the peer chose"), f"registration's certificate's signature {FORM}"),
    (json.dumps(REFUSED_LINES["cells.jsonl"]), "the line says the signer refused its execution phase"),
    (json.dumps({"registration": REGISTRATION, "refused": [LINK, "ingress", 1], "cost": COST}),
     "the line says the signer refused its execution phase"),
], ids=["a registration's key", "a layout's key", "a line's key", "an object under a key", "a float", "no object",
        "other separators", "another format", "another schema", "other word bits", "another link", "another's name",
        "no direction", "another schedule", "other cell bytes", "other buckets", "a negative", "a boolean", "2^63",
        "a number's text", "upper case", "a longer root", "a longer key id", "a shorter digest", "a longer signature",
        "a sentence for a signature", "a refused execution phase", "a registration that says refused"])
def test_a_registration_line_the_list_does_not_rule_is_not_published(host, line, why):
    """The keys `out/` publishes of a cells' registration are the one list `ROW_LINE`, each in its form. A line with
    a key the list names neither as published nor as left out, at any depth, a published value out of its key's
    form (another constant, a value the run's deployment doesn't give, hex of another length or case, a number
    that is no count), or a line that isn't the writer's own form, which the gate couldn't cut to the same bytes,
    is not published, nor is any line of the run: the rehearsal's end is nonzero, says
    `unruled` and is not accepted, and the line and what to change are the sentence in its `witness/`, of which
    the end and `out/` hold no word. So is a line that says its execution phase was refused, which a check that
    accepted can't hold. The deployment is published as before."""
    with pytest.raises(A.Unruled, match=why):
        A.row_registration(line, GIVEN)
    for torn, said in ((_line()[:-1], "the line isn't JSON"), (json.dumps([_line()]), "the line isn't one JSON object")):
        with pytest.raises(A.Unruled, match=said):
            A.row_registration(torn, GIVEN)
    good = _line()
    assert A.row_registration(good, GIVEN) == good.replace(f'"wiring": "{REGISTRATION["wiring"]}", ', "").replace(
        f', "cost": {json.dumps(COST)}', "")
    with pytest.raises(A.Unruled, match="held to what the run's deployment.json gives, which gives none"):
        A.row_registration(good, {})
    with pytest.raises(A.Unruled, match="the line holds the key 'withheld'"):
        A.row_registration(json.dumps({"withheld": True, "cost": COST}), GIVEN)
    bad = WARDEN_FILES.replace("(out / \"rows.jsonl\")", f"(out / \"cells.jsonl\").write_text({good + chr(10)!r} + "
                               f"{line + chr(10)!r})\n(out / \"rows.jsonl\")", 1)
    assert bad != WARDEN_FILES
    run, end = run_warden(host, bad)
    d = host.certifier / run
    said = ended_why(host, end)
    assert end["rc"] == 2 and end["ended"] == "unruled" and said.startswith("cells.jsonl line 2: ")
    # a value the line's certificate holds too (its link, direction, execution phase and the certificate's own
    # fields) is out of the certificate's form first: the line's certificate is read before its registration
    assert why in said or ("CERTIFICATE_LINE" in said and FORM in why)
    assert said.endswith("; out/ holds none of the run's lines")
    assert "unruled" not in end and end["verdict"] == NOT_ACCEPTED
    public = b"".join(p.read_bytes() for p in (d / "out").rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    assert b"line 2" not in public and b"none of the run" not in public and b"ROW_LINE" not in public
    assert b"a word the peer chose" not in public and b"another" not in public and b"refused" not in public
    assert set(end["files"]) == {"deployment.json"} == {p.name for p in (d / "out").iterdir()}
    if "ROW_LINE names neither" in said or "holds the key" in said:
        assert "Rule whether it is public, and write it there (auditor.py) with its form, or as WITHHELD" in said


@pytest.mark.parametrize("name, line, why", [
    ("certified.jsonl", {"certificate": CERTIFICATE | {"note": "x"}}, "the line's certificate holds the key 'note'"),
    ("certified.jsonl", {"certificate": CERTIFICATE | {"link": "another"}}, "certificate's link isn't in the form"),
    ("certified.jsonl", {"certificate": CERTIFICATE | {"signature": "a word the peer chose"}},
     "certificate's signature isn't in the form CERTIFICATE_LINE"),
    ("certified.jsonl", {"certificate": CERTIFICATE | {"window": "0"}}, "certificate's window isn't in the form"),
    ("certified.jsonl", REFUSED_LINES["certified.jsonl"], "the line says the signer refused its execution phase"),
    ("certified.jsonl", {"salt_c": SALT_C}, "the line holds no certificate"),
    ("rows.jsonl", REFUSED_LINES["rows.jsonl"], "the line says the signer refused its execution phase"),
    ("certified.jsonl", ["a list"], "the line isn't a JSON object"),
    ("rows.jsonl", {"registration": ROWS_REGISTRATION | {"link": "another"}, "cost": COST},
     "certificate's link isn't in the form"),
    ("rows.jsonl", {"registration": {"link": LINK, "window": 0}, "cost": COST},
     "the line's registration doesn't hold a certificate's fields"),
    ("rows.jsonl", {A.ROWS_NO_HEAD: NO_HEAD | {"window": 1.5}, "cost": WORD_COST},
     "certificate's window isn't in the form"),
], ids=["a certificate's key", "another link", "a sentence for a signature", "a number's text",
        "a refused execution phase", "no record's certificate", "a rows' refused execution phase", "no object",
        "a rows' link", "no certificate", "a word's window"])
def test_a_certificates_line_out_of_form_publishes_none_of_the_runs_lines(host, name, line, why):
    """A certificate goes to `out/` only by `CERTIFICATE_LINE`'s keys, each in its form, from a record's line of
    `certified.jsonl` and from a row registration's alike. A line with another key or a value out of form, or one
    that says its execution phase was refused, ends the rehearsal nonzero as `unruled` and not accepted; none of
    the run's lines' files is written, and neither the end nor `out/` holds a word of the line."""
    with pytest.raises(A.Unruled, match=why):
        (A.certified_certificate if name == "certified.jsonl" else A.row_certificate)(line, GIVEN)
    assert A.certified_certificate({"salt_c": SALT_C, "certificate": CERTIFICATE}, GIVEN) == {"certificate": CERTIFICATE}
    bad = WARDEN_FILES + (f'with (out / {name!r}).open("a") as f:\n    f.write({json.dumps(line) + chr(10)!r})\n')
    run, end = run_warden(host, bad)
    d = host.certifier / run
    said = ended_why(host, end)
    assert end["rc"] == 2 and end["ended"] == "unruled" and f"{name} line 2: " in said and why in said
    assert end["verdict"] == NOT_ACCEPTED
    public = b"".join(p.read_bytes() for p in (d / "out").rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    assert b"a word the peer chose" not in public and b"another" not in public and b"line 2" not in public
    assert b"refused" not in public
    assert set(end["files"]) == {"deployment.json"} == {p.name for p in (d / "out").iterdir()}


def test_a_run_that_is_not_accepted_publishes_no_line_and_no_count_a_refused_execution_phase_lowers(host):
    """Which execution phases the signer refused is the traffic's. A run with one is not accepted, and of it the
    gate publishes none of the four lines' files, which would say it by a line or by a certificate that is
    missing, and none of the counts that are one fewer for it: how many execution phases were certified, by how
    many certificates, and how many cells' registrations the check accepted. Its end holds that it was not
    accepted, how its connections failed and what the schedule fixes. The lines stay whole in the witness.

    The same run without the refused execution phase, under a check that accepts, publishes all four files and
    the counts. The verdict decides and not the lines: the certified lines alone under a check that does not accept
    publish nothing either. A refused line under a check that accepts is a witness no run writes, and ends the run
    `unruled` and not accepted, with the same verdict as the first run's."""
    refused = WARDEN_FILES.replace(repr(CHECK), repr(REFUSED_CHECK)) + REFUSED_PHASE
    assert refused.count(repr(REFUSED_CHECK)) == 1
    run, end = run_warden(host, refused, token="a001")
    d = host.certifier / run
    assert (end["rc"], end["ended"]) == (0, "ended") and ended_why(host, end) == "the rehearsal ended"
    assert end["verdict"] == NOT_ACCEPTED
    assert set(end["files"]) == {"deployment.json"} == {p.name for p in (d / "out").iterdir()}
    for name, line in REFUSED_LINES.items():
        assert json.loads((d / A.WITNESS / name).read_text().splitlines()[-1]) == line
    public = b"".join(p.read_bytes() for p in (d / "out").rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    for said in (b"refused", b"certified_windows", b"certificates", b"registered", b"withheld", b"aren't registered"):
        assert said not in public, said
    rc, blob = A.gate(["--ssh", f"fetch {run}"], host)
    assert rc == 0 and not {n.rsplit("/", 1)[-1] for n in tar_names(blob)} & LINE_FILES

    run, end = run_warden(host, token="a002")
    assert end["verdict"] == {"accepted": True, "failed": [], "windows": 2, "certified_windows": 6, "certificates": 6,
                              "cell_rows": {"registered": 4, "of": 4}}
    assert set(end["files"]) == {"deployment.json"} | LINE_FILES

    run, end = run_warden(host, WARDEN_FILES.replace(repr(CHECK), repr(REFUSED_CHECK)), token="a003")
    assert end["verdict"]["accepted"] is False and set(end["files"]) == {"deployment.json"}

    run, end = run_warden(host, WARDEN_FILES + REFUSED_PHASE, token="a004")
    assert (end["rc"], end["ended"], end["verdict"]) == (2, "unruled", NOT_ACCEPTED)
    assert set(end["files"]) == {"deployment.json"}
    assert ended_why(host, end) == f"certified.jsonl line 2: {A.REFUSED_LINE}; out/ holds none of the run's lines"
    assert b"refused" not in (host.certifier / run / "end.json").read_bytes()


def test_an_end_is_accepted_only_when_all_its_lines_are_out(host, tmp_path):
    """A program that reads an end's `accepted` alone takes an accepted end as one that published its lines. So an
    end is accepted only when the check accepted, counted the request's requests, and every line of the run is
    ruled and in `out/`. Where the check accepted and a line was refused, or is neither a certificate nor a
    refusal, or is out of its form, the gate cannot rule: the end says `unruled`, exits 2 and is not accepted, with
    none of the counts a refused execution phase lowers and no line's file, for a rehearsal and for a relay
    session, whose request asks for no number. The same witness with every line ruled is accepted, with all four
    files and the counts."""
    unruled = {
        "a refused line": WARDEN_FILES + REFUSED_PHASE,
        "neither a certificate nor a refusal": WARDEN_FILES + (
            f'with (out / "certified.jsonl").open("a") as f:\n'
            f'    f.write({json.dumps({"salt_c": SALT_C}) + chr(10)!r})\n'),
        "a certificate out of form": WARDEN_FILES + (
            f'with (out / "certified.jsonl").open("a") as f:\n'
            f'    f.write({json.dumps({"certificate": CERTIFICATE | {"link": "another"}}) + chr(10)!r})\n'),
        "a registration out of form": WARDEN_FILES + (
            f'with (out / "cells.jsonl").open("a") as f:\n    f.write({_of(root="ab" * 65) + chr(10)!r})\n'),
    }
    for i, (case, source) in enumerate(unruled.items()):
        run, end = run_warden(host, source, token=f"b00{i}")
        d = host.certifier / run
        witness = d / A.WITNESS
        assert json.loads((witness / "check.json").read_text())["accepted"] is True, case
        assert (end["rc"], end["ended"], end["verdict"]) == (2, "unruled", NOT_ACCEPTED), case
        assert set(end["files"]) == {"deployment.json"} == {p.name for p in (d / "out").iterdir()}, case
        for requests in (100, None):
            out = tmp_path / f"{i}-{requests}"
            out.mkdir()
            got = A.warden_outcome(out, witness, requests)
            assert got["verdict"] == NOT_ACCEPTED and set(got) == {"verdict", "unruled"}, case
            assert not {p.name for p in out.iterdir()} & LINE_FILES, case

    run, end = run_warden(host, token="b00a")
    assert (end["rc"], end["ended"], end["verdict"]["accepted"]) == (0, "ended", True)
    assert LINE_FILES <= set(end["files"]) and set(A.ACCEPTED_COUNTS) <= set(end["verdict"])
    assert set(A.ACCEPTED_CELL_ROWS) <= set(end["verdict"]["cell_rows"])


def test_no_line_is_published_beside_a_deployment_that_doesnt_fix_its_forms(host):
    """The forms of a line's link, schedule and signature are the run's deployment's. A deployment that doesn't
    give them, or gives one out of form, gives no form at all, and no certificate or registration is published
    beside it. A key's parameters are in form exactly where the signer's own bounds take them (`lms.Params`), so
    the gate reads a signature's length of no others."""
    for params in ([64, 1, 1], [64, 2, 13], [64, 4, 20], [64, 8, 30]):
        assert A.in_form(params, A.KEY_PARAMS) and A.lms.Params(*params).signature_bytes() > 0
    for params in ([32, 8, 1], [64, 3, 1], [64, 16, 1], [64, 8, 0], [64, 8, 31], [64, 8, True]):
        assert not A.in_form(params, A.KEY_PARAMS)
        with pytest.raises(ValueError, match="LMS parameters are"):
            A.lms.Params(*params)
    for params in ([64, 8], [64, 8, 1, 1], (64, 8, 1), "64 8 1", None, {"N": 64}, [64.0, 8, 1]):
        assert not A.in_form(params, A.KEY_PARAMS)
    for less in ("link", "digest", "params", "key"):
        doc = {k: v for k, v in DEPLOYMENT.items() if k != less}
        (host.root / "deployment.json").write_text(json.dumps(doc))
        assert A.deployment_forms(host.root / "deployment.json") == {}
    for bad in ({"link": "a link"}, {"digest": "31" * 63}, {"params": {"cell_bytes": 512, "window": -600}},
                {"params": {"cell_bytes": "512", "window": 600}}, {"params": [512, 600]},
                {"key": {"params": [64, 3, 1]}}, {"key": {"params": [64, 8]}}, {"key": [64, 8, 1]}):
        (host.root / "deployment.json").write_text(json.dumps(DEPLOYMENT | bad))
        assert A.deployment_forms(host.root / "deployment.json") == {}
    for text in (json.dumps([DEPLOYMENT]), json.dumps(DEPLOYMENT)[:-1], ""):
        (host.root / "deployment.json").write_text(text)
        assert A.deployment_forms(host.root / "deployment.json") == {}
    assert A.deployment_forms(host.root / "none.json") == {}
    (host.root / "deployment.json").write_text(json.dumps(DEPLOYMENT))
    assert A.deployment_forms(host.root / "deployment.json") == GIVEN
    run, end = run_warden(host, WARDEN_FILES.replace(f"{DEPLOYMENT!r} | ", "", 1))
    assert end["rc"] == 2 and end["ended"] == "unruled" and set(end["files"]) == {"deployment.json"}
    assert end["verdict"] == NOT_ACCEPTED
    assert "certified.jsonl line 1: the line's certificate's link is held to what the run's deployment.json gives" in \
        ended_why(host, end)


@pytest.mark.parametrize("argv, witness", [
    (["--ssh", "fetch {run}"], False),
    (["--ssh", "fetch {run}", "--key", "-"], False),
    (["--ssh", "fetch {run}", "--key", "memory-accounting"], False),
    (["--ssh", "fetch {run}", "--key", LANE], True),
    (["fetch", "{run}"], True),
])
def test_fetch_gives_a_warden_rehearsals_witness_to_its_own_lane_and_vy_admin_alone(host, argv, witness):
    run, _ = run_warden(host)
    rc, blob = A.gate([w.format(run=run) for w in argv], host, by="vy-admin")
    names = tar_names(blob)
    assert rc == 0 and {f"{run}/request.json", f"{run}/end.json", f"{run}/out/cells-certificates.jsonl",
                        f"{run}/out/cells-registrations.jsonl",
                        f"{run}/out/rows-certificates.jsonl", f"{run}/out/certificates.jsonl",
                        f"{run}/out/deployment.json"} <= names
    assert not {f"{run}/out/cells.jsonl", f"{run}/out/rows.jsonl", f"{run}/out/rehearse.log",
                f"{run}/out/tunnel.log"} & names
    hidden = {n for n in names if n == f"{run}/witness" or n.startswith(f"{run}/witness/")}
    assert bool(hidden) == witness
    if witness:
        assert {f"{run}/witness/cells.jsonl", f"{run}/witness/rows.jsonl", f"{run}/witness/cells.salts",
                f"{run}/witness/certified.jsonl", f"{run}/witness/check.json",
                f"{run}/witness/rehearse.log", f"{run}/witness/tunnel.log"} <= hidden


@pytest.mark.parametrize("argv, own", [
    (["--ssh", "{verb} {run}"], False),
    (["--ssh", "{verb} {run}", "--key", "-"], False),
    (["--ssh", "{verb} {run}", "--key", "memory-accounting"], False),
    (["--ssh", "{verb} {run}", "--key", LANE], True),
    (["{verb}", "{run}"], True),
])
def test_a_runs_peer_and_the_lane_that_asked_go_to_its_own_lane_and_vy_admin_alone(host, argv, own):
    """The request names the peer, a machine, and the lane that asked. `status` gives any other key the request
    less the two. A fetch's `request.json` is that form for every key, so what a record releases of a fetch names
    no peer, and the request whole is in the fetch's `witness/`, which only the run's own lane and vy-admin get.
    The gate reads its own copy whole."""
    run, _ = run_warden(host)
    whole = json.loads((host.certifier / run / "request.json").read_text())
    shown = {k: v for k, v in whole.items() if k not in A.REQUEST_WITHHELD}
    assert A.REQUEST_WITHHELD == ("peer", "by") and set(whole) - set(shown) == {"peer", "by"}
    assert A.certifier_request(run, host) == whole and A.public_request(whole) == shown

    def ask(verb: str) -> bytes:
        rc, out = A.gate([w.format(verb=verb, run=run) for w in argv], host, by="root",
                         certifier_active=lambda r: False)
        assert rc == 0
        return out

    assert json.loads(ask("status"))["request"] == (whole if own else shown)
    with tarfile.open(fileobj=io.BytesIO(ask("fetch"))) as tar:
        names = tar.getnames()
        assert names.count(f"{run}/request.json") == 1
        assert json.loads(tar.extractfile(f"{run}/request.json").read()) == shown
        assert (f"{run}/witness/request.json" in names) == own
        if own:
            assert tar.extractfile(f"{run}/witness/request.json").read() == (
                host.certifier / run / "request.json").read_bytes()
        public = b"".join(tar.extractfile(m).read() for m in tar if m.isfile() and "/witness/" not in m.name)
    assert whole["peer"].encode() not in public


def test_of_the_word_that_an_execution_phase_has_no_head_the_rehearsal_publishes_its_certificate(host):
    """A line of `rows.jsonl` that holds, in place of a registration, the network certifier's word that its
    execution phase has no head stays whole in `witness/`. `out/` holds its certificate in the fields a head's has,
    the link's egress in the word's execution phase: each value is one the word's public record holds, and the
    line's cost and the key the line holds the word under are in neither `out/` nor the end."""
    source = WARDEN_FILES + ('with (out / "rows.jsonl").open("a") as f:\n'
                             f'    f.write(json.dumps({{{A.ROWS_NO_HEAD!r}: {NO_HEAD!r}, "cost": {WORD_COST!r}}}) + "\\n")\n')
    run, _ = run_warden(host, source)
    d = host.certifier / run
    out = d / "out"
    line = {A.ROWS_NO_HEAD: NO_HEAD, "cost": WORD_COST}
    assert json.loads((d / A.WITNESS / "rows.jsonl").read_text().splitlines()[1]) == line
    word = NO_HEAD["certificate"]
    published = {"certificate": {"key_id": word["key"], "link": NO_HEAD["link"], "direction": "egress",
                                 "window": NO_HEAD["window"], "digest": word["digest"],
                                 "signature": word["signature"]}}
    assert [json.loads(x) for x in (out / "rows-certificates.jsonl").read_text().splitlines()][1:] == [published]
    assert A.row_certificate(line, GIVEN) == published
    public = b"".join(p.read_bytes() for p in out.rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    for held in (A.ROWS_NO_HEAD, "cost", "sign_ms", "3.625"):
        assert held.encode() not in public, held


def test_a_warden_rehearsal_that_never_ran_publishes_nothing_and_its_end_says_why(host):
    rc, out = gate(host, REQUEST, started=[])
    run = json.loads(out)["run"]
    end = A.certifier_run(host, run, pinned=PIN, tunnel_argv=lambda s, l: TUNNEL,
                          up=lambda t, p: False, command=lambda *a: pytest.fail("ran without its tunnel"))
    assert end["rc"] == 2 and end["ended"] == "tunnel"
    assert "the tunnel to vy-nebius-1's 127.0.0.1:8011 didn't come up" in ended_why(host, end)
    assert REQUEST[2] not in json.dumps(end)
    assert end["verdict"] is None and "registrations" not in end and "witness" not in end
    assert end["files"] == {}

    run, end = run_warden(host, "raise SystemExit(1)", token="cd34")
    assert end["rc"] == 1 and end["ended"] == "exited" and end["verdict"] is None
    assert ended_why(host, end) == "the rehearsal exited 1"


def test_the_warden_rehearsal_runs_warden_serve_on_the_gates_traffic_through_the_tunnels_local_end(tmp_path):
    """A request.json written before the gate took `--height` runs at the deployed key's, 20, as one that gives none."""
    req = {"device": "warden", "requests": 100}
    argv = A.warden_command(req, tmp_path, "http://127.0.0.1:40001", tree=Path("/opt/vy-challenger") / PIN)
    assert argv == [sys.executable, f"/opt/vy-challenger/{PIN}/benchmarks/network_traces/warden_serve.py", "rehearse",
                    "--upstream", "127.0.0.1:40001", "--out", str(tmp_path), "--height", "20", "--requests", "100",
                    "--rate", "0.5", "--seed", "0", "--tail", "60", "--temperature", "0.0", "--top-p", "1.0"]
    assert A.warden_command(req | {"height": 20}, tmp_path, "http://127.0.0.1:40001",
                            tree=Path("/opt/vy-challenger") / PIN) == argv
    assert A.warden_command(req | {"height": 10}, tmp_path, "u")[7:9] == ["--height", "10"]
    assert A.warden_command(req | {"tokens": 64}, tmp_path, "http://127.0.0.1:1")[-2:] == ["--max-tokens", "64"]


def test_a_scan_rehearsal_is_recorded_only_when_asked_and_runs_id_requests_at_the_scans_schedule(host, tmp_path):
    """`--scan` is recorded (`scan: true`) only when given, and its run is the same rehearsal with `--ids` and
    `SCAN_SCHEDULE` added: the same traffic, sampling, height and tokens, so `warden_minutes` bounds it as it does any
    (`test_warden_serve` holds the schedule to `scan_zk.PARAMS`)."""
    rc, out = gate(host, REQUEST + ["--scan", "--tokens", "64"], started=[])
    assert rc == 0, out
    scan = json.loads((host.certifier / json.loads(out)["run"] / "request.json").read_text())
    assert scan["scan"] is True and scan["device"] == "warden" and scan["tokens"] == 64
    rc, out = gate(host, REQUEST, started=[], token="cd34")
    plain = json.loads((host.certifier / json.loads(out)["run"] / "request.json").read_text())
    assert "scan" not in plain
    rc, out = gate(host, REQUEST + ["--scan", "--minutes", "10"], started=[], token="ef56")
    assert rc == 3 and "--minutes 10: 100 requests take 11 minutes" in json.loads(out)["refused"]

    argv = A.warden_command(scan, tmp_path, "http://127.0.0.1:40001")
    without = A.warden_command({k: v for k, v in scan.items() if k != "scan"}, tmp_path, "http://127.0.0.1:40001")
    at = without.index("--max-tokens")
    assert argv == [*without[:at], "--ids", *A.SCAN_SCHEDULE, *without[at:]]
    assert A.warden_command(plain, tmp_path, "u") == A.warden_command(plain | {"scan": False}, tmp_path, "u")
    assert "--ids" not in A.warden_command(plain, tmp_path, "u")


def test_a_scan_rehearsals_tick_is_recorded_only_when_given_and_its_run_and_minutes_are_at_that_tick(host, tmp_path):
    """`--bucket-ms` is recorded (``bucket_ms``) only when given, and the run is the scan's schedule with that tick in
    place of its 100 ms, every other flag the scan's. The gate counts the minutes at it: a sampling block of 32 ticks
    of 4 s is 128 s, past the minute `warden_minutes` counts for a shorter one, so 100 requests take a minute more, and
    one group of 4 prompts of 4 ids out takes its one block, 6 minutes at height 10 where 100 ms takes 4."""
    rc, out = gate(host, REQUEST + ["--scan", "--bucket-ms", "4000"], started=[])
    assert rc == 0, out
    req = json.loads((host.certifier / json.loads(out)["run"] / "request.json").read_text())
    assert (req["scan"], req["bucket_ms"]) == (True, 4000)
    argv = A.warden_command(req, tmp_path, "http://127.0.0.1:40001")
    schedule = argv[argv.index("--ids") + 1:][:len(A.SCAN_SCHEDULE)]
    assert schedule == list(A.scan_schedule(4000)) and schedule[schedule.index("--bucket-ms") + 1] == "4000"
    assert [x for i, x in enumerate(schedule) if schedule[i - 1] != "--bucket-ms"] == \
        [x for i, x in enumerate(A.SCAN_SCHEDULE) if A.SCAN_SCHEDULE[i - 1] != "--bucket-ms"]
    assert A.scan_schedule() == A.scan_schedule(100) == A.SCAN_SCHEDULE

    rc, out = gate(host, REQUEST + ["--scan", "--minutes", "11"], started=[], token="cd34")
    assert rc == 0 and "bucket_ms" not in json.loads(out), out
    rc, out = gate(host, REQUEST + ["--scan", "--bucket-ms", "4000", "--minutes", "11"], started=[], token="ef56")
    assert rc == 3 and "--minutes 11: 100 requests take 12 minutes" in json.loads(out)["refused"]

    group = SCAN + ["--window-per-group", "--tokens", "4"]
    rc, out = gate(host, group + ["--minutes", "4"], stdin=b"[[1],[2],[3],[4]]", started=[], token="0a0b")
    assert rc == 0, out
    rc, out = gate(host, group + ["--bucket-ms", "4000", "--minutes", "5"], stdin=b"[[1],[2],[3],[4]]", started=[],
                   token="0c0d")
    assert rc == 3 and "--minutes 5: 1 groups take 6 minutes" in json.loads(out)["refused"]
    rc, out = gate(host, group + ["--bucket-ms", "4000", "--minutes", "6"], stdin=b"[[1],[2],[3],[4]]", started=[],
                   token="0e0f")
    assert rc == 0, out
    req = json.loads((host.certifier / json.loads(out)["run"] / "request.json").read_text())
    assert (req["bucket_ms"], req["window_per_group"], req["requests"]) == (4000, True, 4)
    argv = A.warden_command(req, tmp_path, "u")
    assert argv[argv.index("--ids") + 1:][:len(A.SCAN_SCHEDULE)] == list(A.scan_schedule(4000))


@pytest.mark.parametrize("argv, why", [
    (["--bucket-ms", "2000"], "--bucket-ms: the scan's tick, with --scan"),
    (["--scan", "--bucket-ms", "99"], "--bucket-ms '99': 100 to 4000"),
    (["--scan", "--bucket-ms", "4001"], "--bucket-ms '4001': 100 to 4000"),
    (["--scan", "--bucket-ms", "0400"], "--bucket-ms '0400': 100 to 4000"),
    (["--scan", "--bucket-ms", "2.5"], "--bucket-ms '2.5': 100 to 4000"),
])
def test_the_gate_refuses_a_tick_without_the_scan_or_past_its_range_and_writes_nothing(host, argv, why):
    rc, out = gate(host, REQUEST + argv, started=[])
    assert rc == 3 and why in json.loads(out)["refused"]
    assert not host.certifier.exists()


def test_a_warden_rehearsals_sampling_is_recorded_and_passed_on_every_request(host, tmp_path):
    """The defaults, temperature 0 and top_p 1, and given values alike are recorded and passed on, so that every request
    of the traffic carries both (`warden_serve.varied_traffic`) and an endpoint's own generation config never applies;
    a request.json written before the gate took them runs at the defaults."""
    for k, (given, recorded) in enumerate((([], (0.0, 1.0)), (["--temperature", "0.6", "--top-p", "0.9"], (0.6, 0.9)),
                                          (["--temperature", "2", "--top-p", "0.0001"], (2.0, 0.0001)))):
        rc, out = gate(host, REQUEST + given, started=[], token=f"{k:04x}")
        assert rc == 0, out
        request = json.loads((host.certifier / json.loads(out)["run"] / "request.json").read_text())
        assert (request["temperature"], request["top_p"]) == recorded
        argv = A.warden_command(request, tmp_path, "http://127.0.0.1:1")
        assert argv[argv.index("--temperature"):][:4] == ["--temperature", str(recorded[0]), "--top-p", str(recorded[1])]
    older = {k: v for k, v in request.items() if k not in ("temperature", "top_p")}
    assert A.warden_command(older, tmp_path, "u")[-4:] == ["--temperature", "0.0", "--top-p", "1.0"]


def test_a_warden_rehearsals_watermark_is_recorded_and_passed_on_only_when_asked(host, tmp_path):
    """`--watermark` is recorded (``watermark: true``) only when given, and its run is the same rehearsal with
    `rehearse --watermark` added: the traffic, schedule and minutes are the same, since the stream changes nothing
    the warden certifies."""
    rc, out = gate(host, REQUEST + ["--watermark", "--tokens", "64"], started=[])
    assert rc == 0, out
    req = json.loads((host.certifier / json.loads(out)["run"] / "request.json").read_text())
    rc, out = gate(host, REQUEST + ["--tokens", "64"], started=[], token="cd34")
    plain = json.loads((host.certifier / json.loads(out)["run"] / "request.json").read_text())
    assert req["watermark"] is True and "watermark" not in plain
    assert {k: v for k, v in req.items() if k not in ("watermark", "run")} == \
        {k: v for k, v in plain.items() if k != "run"}
    argv, without = A.warden_command(req, tmp_path, "u"), A.warden_command(plain, tmp_path, "u")
    at = without.index("--max-tokens")
    assert argv == [*without[:at], "--watermark", *without[at:]]
    assert A.warden_command(plain | {"watermark": False}, tmp_path, "u") == without
    refusals = [gate(host, REQUEST + extra + ["--minutes", "3"], started=[], token="ef56") for extra in
                ([], ["--watermark"])]
    assert refusals[0] == refusals[1] and refusals[0][0] == 3
    assert "--minutes 3: 100 requests take" in json.loads(refusals[0][1])["refused"]


#: a relay session of 20 execution phases at the scan's schedule (64 s of them), its clients' port 8012 on the peer
SESSION = ["relay-session", *REQUEST[1:5], "--client-port", "8012", "--windows", "20", "--minutes", "5", "--height",
           "10"]


def test_a_lane_key_starts_a_relay_session_at_the_scans_schedule_one_at_a_time_with_the_rehearsals(host):
    """`relay-session` records the peer, the endpoint's port, the clients' port and its execution phases, always with
    ``scan`` and never with a rehearsal's traffic or sampling; its tick, class and watermark are recorded only when
    given, as a scan rehearsal's. It shares the rehearsals' unit: one of any of them runs at a time, and a key that
    reads only starts none."""
    started = []
    rc, out = gate(host, SESSION, started=started)
    doc = json.loads(out)
    assert rc == 0 and doc == {"run": "c20261024T110000Z-ab12", "device": "warden", "peer": REQUEST[2], "port": 8011,
                               "client_port": 8012, "windows": 20, "minutes": 5, "height": 10, "scan": True,
                               "by": LANE, "requested_utc": "2026-10-24T11:00:00Z",
                               "unit": f"vy-certifier@{doc['run']}.service"}
    assert started == [doc["run"]] and A.read_ledger(host.ledger) == []
    assert json.loads((host.certifier / doc["run"] / "request.json").read_text()) == \
        {k: v for k, v in doc.items() if k != "unit"}

    for argv in (SESSION, REQUEST):
        rc, out = gate(host, argv, active=lambda r: r == doc["run"], started=started, token="cd34")
        assert rc == 3 and f"{doc['run']} still runs: one at a time" in json.loads(out)["refused"]
    rc, out = gate(host, REQUEST, started=started, token="cd34")
    assert rc == 0
    rc, out = gate(host, SESSION, active=lambda r: r == "c20261024T110000Z-cd34", started=started, token="ef56")
    assert rc == 3 and "c20261024T110000Z-cd34 still runs" in json.loads(out)["refused"]
    assert started == [doc["run"], "c20261024T110000Z-cd34"]

    given = SESSION[:-4] + ["--minutes", "6", "--height", "10", "--bucket-ms", "200", "--scan-class", "w63k4",
                            "--watermark"]
    rc, out = gate(host, given, started=started, token="ef56")
    assert rc == 0 and {k: v for k, v in json.loads(out).items() if k not in doc} == \
        {"bucket_ms": 200, "scan_class": "w63k4", "watermark": True}
    rc, out = gate(host, SESSION[:-2], started=started, token="ab34")
    assert rc == 3 and "--minutes 5: 20 execution phases of 3200 ms take 9 minutes" in json.loads(out)["refused"]

    rc, out = gate(host, SESSION, key=None, started=started, token="ab56")
    assert rc == 3 and json.loads(out)["refused"] == \
        "this key reads only: a relay session is started with a lane's own gate key, or by vy-admin"
    assert len(started) == 3


@pytest.mark.parametrize("argv, why", [
    (SESSION[:5] + SESSION[7:], "relay-session: the following arguments are required: --client-port"),
    (SESSION[:7] + SESSION[9:], "relay-session: the following arguments are required: --windows"),
    (SESSION + ["--client-port", "80"], "--client-port '80': 1024 to 65535"),
    (SESSION + ["--client-port", "8011"], "--client-port 8011: a port of the peer's loopback for the relay's clients, "
                                           "not the endpoint's --port and none of the challenge links' (7811 to 7899)"),
    (SESSION + ["--client-port", "7811"], "--client-port 7811: a port of the peer's loopback"),
    (SESSION + ["--client-port", "7899"], "--client-port 7899: a port of the peer's loopback"),
    (SESSION + ["--windows", "0"], "--windows '0': 1 to 1119"),
    (SESSION + ["--windows", "1120"], "--windows '1120': 1 to 1119"),
    (SESSION + ["--windows", "205", "--minutes", "170"],
     "--height 10: the key's 1024 one-time signatures don't sign 205 execution phases: give --height 11 or more, or "
     "fewer --windows"),
    (SESSION + ["--minutes", "4"], "--minutes 4: 20 execution phases of 3200 ms take 5 minutes (their span, the start, "
                                   "the check and the key at height 10): give --minutes 5 or more, or fewer --windows"),
    (SESSION + ["--bucket-ms", "200"], "--minutes 5: 20 execution phases of 6400 ms take 6 minutes"),
    (SESSION + ["--bucket-ms", "50"], "--bucket-ms '50': 100 to 4000"),
    (SESSION + ["--scan-class", "w1"], "--scan-class 'w1': one of ['w63', 'w63k4']"),
    (SESSION + ["--scan"], "relay-session: unrecognized arguments: --scan"),
    (SESSION + ["--requests", "4"], "relay-session: unrecognized arguments: --requests 4"),
    (SESSION + ["--tokens", "16"], "relay-session: unrecognized arguments: --tokens 16"),
    (SESSION + ["--prompts"], "relay-session: unrecognized arguments: --prompts"),
    (SESSION + ["--window-per-group"], "relay-session: unrecognized arguments: --window-per-group"),
    (SESSION + ["--temperature", "0"], "relay-session: unrecognized arguments: --temperature 0"),
    (SESSION + ["--top-p", "1"], "relay-session: unrecognized arguments: --top-p 1"),
    (SESSION[:2] + ["nobody"] + SESSION[3:], "--peer 'nobody': not one of this device's peers"),
])
def test_the_gate_refuses_a_relay_session_it_cant_check_and_writes_nothing(host, argv, why):
    """Each refusal the verb adds (the clients' port, the execution phases, the key and the minutes they take), each
    flag of a rehearsal's traffic or sampling, which a session has none of, and the scan rehearsal's own refusals of
    what the two share."""
    rc, out = gate(host, argv, started=[])
    assert rc == 3 and why in json.loads(out)["refused"], out
    assert not host.certifier.exists()


def test_the_longest_relay_session_is_an_hour_at_the_scans_tick_and_the_gates_minutes_past_280_ms():
    """The numbers the module docstring gives a runner that asks whether its load fits: a session's execution phases
    last windows · T · τ; `SESSION_WINDOWS`' most are 59 min 40 s at the scan's tick, and from τ = 280 ms the gate's
    most `--minutes` bounds them, to the whole execution phases in 167 minutes less the key's seconds (162 min 44 s
    at the deployed key's height, from 273 ms)."""
    T = A.SCAN["--window"]

    def longest(tick: int, height: int | None = None) -> tuple[int, int, float]:
        """The most execution phases the gate takes at `tick`, their `--minutes` and their span in seconds."""
        w = A.SESSION_WINDOWS[1]
        while A.session_minutes(w, height or A.warden_height(w), T * tick) > A.CERTIFIER_MINUTES[1]:
            w -= 1
        return w, A.session_minutes(w, height or A.warden_height(w), T * tick), w * T * tick / 1000

    assert A.SESSION_WINDOWS == (1, A.warden_windows(A.WARDEN_REQUESTS[1])) == (1, 1119)
    assert A.SCAN_BUCKET_MS == (100, 4000) and T == 32 and A.CERTIFIER_MINUTES[1] == 170
    assert {tick: longest(tick) for tick in (100, 200, 250, 279, 280, 1000, 4000)} == {
        100: (1119, 63, 3580.8), 200: (1119, 123, 7161.6), 250: (1119, 153, 8952.0), 279: (1119, 170, 9990.432),
        280: (1118, 170, 10017.28), 1000: (313, 170, 10016.0), 4000: (78, 170, 9984.0)}
    assert (longest(1000)[2], longest(4000)[2]) == (166 * 60 + 56, 166 * 60 + 24)
    for height, first, most in ((None, 280, 167 * 60), (20, 273, 162 * 60 + 44)):
        assert longest(first - 1, height)[0] == 1119 > longest(first, height)[0]
        for tick in range(first, A.SCAN_BUCKET_MS[1] + 1):
            windows, minutes, span = longest(tick, height)
            key = (1 << (height or A.warden_height(windows + 1))) // 4096
            assert most - (0 if height else key) - T * tick / 1000 - 1 < span <= most, (tick, height)
    assert longest(100, 20)[:2] == (1119, 67)
    assert (A.session_minutes(1119, 13), A.session_minutes(1119, 20)) == (63, 67)
    # a rehearsal's execution phases hold its tail, so the same 1119 as a session's leave the check a minute more
    assert A.warden_minutes(A.WARDEN_REQUESTS[1], 13) == 60 and A.warden_height(1119) == 13
    assert [A.session_minutes(1, h) for h in (10, 20)] == [4, 8]


def session(source: str, seen: list):
    """A stand-in for `warden_serve.py session`, as `rehearsal`: it is given where its relay listens, and `fd`, the
    descriptor it inherits the unit's listener as."""
    def command(req, out, url, listen, listen_fd):
        seen.append((req, out, url, listen))
        return [sys.executable, "-c", f"import json, pathlib, sys; out, url, fd = pathlib.Path({str(out)!r}), {url!r}, "
                                      f"{listen_fd}\n{source}"]
    return command


def test_a_relay_sessions_one_ssh_brings_the_peers_clients_to_its_relay_and_its_status_says_when_it_listens(
        host, tmp_path):
    """The unit's ssh forwards the peer's client port to the relay's listener, a free port here that is not the
    tunnel's own local end, and the unit runs `warden_serve.py session` there for the request's execution phases at
    the scan's schedule. `status` carries ``relay``: null until the relay's `deployment.json` is whole, then where a
    client connects and the deployment; the end is a rehearsal's (`warden_outcome`), saying the session ended."""
    ssh = ["-i", "KEY", "research@peer"]
    assert A.forward_argv(ssh, 40002, 8011, (8012, 40003)) == [
        *A.TUNNEL_SSH, "-L", "127.0.0.1:40002:127.0.0.1:8011", "-R", "127.0.0.1:8012:127.0.0.1:40003", *ssh]
    assert A.forward_argv(ssh, 40002, 8011) == [*A.TUNNEL_SSH, "-L", "127.0.0.1:40002:127.0.0.1:8011", *ssh]
    assert "ExitOnForwardFailure=yes" in A.TUNNEL_SSH

    rc, out = gate(host, SESSION, started=[])
    run = json.loads(out)["run"]
    d = host.certifier / run
    req = json.loads((d / "request.json").read_text())
    argv = A.warden_command(req, tmp_path, "http://127.0.0.1:40001", tree=Path("/opt/gate") / PIN, listen=40003)
    assert argv == [sys.executable, f"/opt/gate/{PIN}/benchmarks/network_traces/warden_serve.py", "session",
                    "--upstream", "127.0.0.1:40001", "--listen", "127.0.0.1:40003", "--out", str(tmp_path),
                    "--height", "10", "--windows", "20", *A.scan_schedule()]
    given = A.warden_command(req | {"bucket_ms": 200, "scan_class": "w63k4", "watermark": True}, tmp_path, "u",
                             listen=1)
    assert given[-len(A.SCAN_SCHEDULE) - 1:] == [*A.scan_schedule(200, "w63k4"), "--watermark"]
    held = A.warden_command(req, tmp_path, "http://127.0.0.1:40001", tree=Path("/opt/gate") / PIN, listen=40003,
                            listen_fd=7)
    assert held == [*argv[:7], "--listen-fd", "7", *argv[7:]] and argv[5:7] == ["--listen", "127.0.0.1:40003"]

    def status() -> dict:
        rc, st = A.gate(["--ssh", f"status {run}"], host, certifier_active=lambda r: True)
        assert rc == 0
        return json.loads(st)

    shown = A.public_request(req)
    assert status() == {"run": run, "request": shown, "end": None, "active": True, "relay": None}
    public = DEPLOYMENT | {"windows": 20, "origin_unix": 1.5}
    deployment = public | {"listen": ["127.0.0.1", 40003], "upstream": ["127.0.0.1", 40004], "origin_raw": 9.25}
    (d / A.WITNESS).mkdir(mode=0o700)
    (d / A.WITNESS / "deployment.json").write_text(json.dumps(deployment)[:-9])
    assert status()["relay"] is None
    (d / A.WITNESS / "deployment.json").write_text(json.dumps(deployment))
    assert status()["relay"] == {"client": "127.0.0.1:8012", "deployment": public}
    (d / A.WITNESS / "deployment.json").unlink()

    seen, forwards = [], []
    files = WARDEN_FILES.replace('(out / "client.json").write_text("{}")\n', "").replace(
        f'{DEPLOYMENT!r} | {{"upstream": url}}', repr(deployment))
    assert repr(deployment) in files and "client.json" not in files
    end = A.certifier_run(host, run, pinned=PIN,
                          tunnel_argv=lambda s, l, back: forwards.append((l, back)) or TUNNEL, up=lambda t, p: True,
                          command=session(files, seen), wait=lambda proc, s: proc.wait())
    ((got, work, url, listen),) = seen
    ((local, back),) = forwards
    assert got == req and work == d / A.WITNESS and url == f"http://127.0.0.1:{local}"
    assert back == (8012, listen) and listen != local
    assert end["rc"] == 0 and end["ended"] == "ended" and end["device"] == "warden"
    assert ended_why(host, end) == "the session ended"
    assert end["verdict"]["accepted"] is True and not {"problems", "requests"} & set(end["verdict"])
    assert {k: end["verdict"][k] for k in ("windows", "certified_windows", "certificates")} == {
        k: CHECK["summary"][k] for k in ("windows", "certified_windows", "certificates")}
    assert not (d / A.WITNESS / "client.json").exists() and "deployment.json" in end["files"]
    assert json.loads((d / A.WITNESS / "deployment.json").read_text()) == deployment
    assert json.loads((d / "out" / "deployment.json").read_text()) == public
    assert status() | {"active": False} == {"run": run, "request": shown, "end": end, "active": False,
                                           "relay": {"client": "127.0.0.1:8012", "deployment": public}}
    rehearsal_status = json.loads(A.gate(["--ssh", f"status {run_warden(host, token='cd34')[0]}"], host,
                                         certifier_active=lambda r: False)[1])
    assert "relay" not in rehearsal_status


def test_a_relay_session_whose_ssh_exits_is_stopped_there_and_its_end_says_the_tunnel_closed(host):
    """A session's clients reach its relay through the unit's ssh alone. When that ssh exits (the peer refused a
    forward, or the link dropped) the unit stops the session's runner rather than let it run out its execution
    phases with no client, and the end is nonzero and says `tunnel`, whatever the check then reads; the sentence,
    which names the peer, is the witness's. A session whose runner fails says so as a rehearsal does."""
    rc, out = gate(host, SESSION, started=[])
    run = json.loads(out)["run"]
    end = A.certifier_run(host, run, pinned=PIN, tunnel_argv=lambda s, l, back: [sys.executable, "-c", ""],
                          up=lambda t, p: True, command=session("import signal; signal.pause()", []),
                          wait=lambda proc, s: proc.wait())
    assert end["rc"] != 0 and end["verdict"] is None
    assert end["ended"] == "tunnel" and REQUEST[2] not in json.dumps(end)
    assert ended_why(host, end) == (f"the tunnel to {REQUEST[2]} closed before the session ended, and with it the "
                                    "clients' way to the relay and the relay's to the endpoint (tunnel.log)")

    rc, out = gate(host, SESSION, started=[], token="cd34")
    end = A.certifier_run(host, json.loads(out)["run"], pinned=PIN,
                          tunnel_argv=lambda s, l, back: TUNNEL, up=lambda t, p: True,
                          command=session("raise SystemExit(1)", []), wait=lambda proc, s: proc.wait())
    assert end["rc"] == 1 and end["ended"] == "exited" and ended_why(host, end) == "the session exited 1"
    rc, out = gate(host, SESSION, started=[], token="ef56")
    end = A.certifier_run(host, json.loads(out)["run"], pinned=PIN,
                          tunnel_argv=lambda s, l, back: TUNNEL, up=lambda t, p: False,
                          command=lambda *a, **k: pytest.fail("ran without its tunnel"))
    assert end["rc"] == 2 and end["ended"] == "tunnel" and "didn't come up (tunnel.log)" in ended_why(host, end)


def test_a_relay_sessions_listener_is_the_units_own_from_before_its_ssh_until_that_ssh_has_exited(
        host, tmp_path, monkeypatch):
    """The `-R` of a session's ssh ends at a port of this host's loopback, and that ssh is up before the relay serves
    and after it stops. The unit binds the port itself before the ssh starts, hands the listening socket to the
    relay's runner and closes it only once the ssh has exited. Here the ssh's stand-in connects to the port as it
    comes up and again as it is stopped, and the test reads the unit's socket: the forward's end is that socket; a
    connection made before the runner serves is the one the runner accepts on the descriptor it inherited; one made
    after the runner stopped arrives at the unit's socket; and the socket still listens when the ssh is stopped and
    is closed after."""
    rc, out = gate(host, SESSION, started=[])
    run = json.loads(out)["run"]
    held, early, mark = [], [], tmp_path / "listened-at-the-sshs-end"
    monkeypatch.setattr(A, "held_listener", lambda make=A.held_listener: held.append(make()) or held[0])

    def sent(conn_or_listener, n: int) -> bytes:
        conn, _ = conn_or_listener.accept()
        with conn:
            return conn.recv(n)

    def tunnel_argv(ssh, local, back):  # before the ssh starts: the forward's end is the unit's socket, listening
        (listener,) = held
        assert listener.getsockname() == ("127.0.0.1", back[1]) and back[1] != local
        return [sys.executable, "-c",
                "import pathlib, signal, socket, sys\n"
                "def stopped(*_):\n"
                f"    socket.create_connection(('127.0.0.1', {back[1]})).close()\n"
                f"    pathlib.Path({str(mark)!r}).write_text('yes')\n"
                "    sys.exit(0)\n"
                "signal.signal(signal.SIGTERM, stopped)\n"
                f"socket.create_connection(('127.0.0.1', {back[1]})).sendall(b'up')\n"
                "signal.pause()\n"]

    def up(tunnel, port):  # the ssh is up and no runner serves yet
        assert sent(held[0], 2) == b"up"
        early.append(socket.create_connection(held[0].getsockname()))
        early[0].sendall(b"early")
        return True

    def wait(proc, seconds):  # the runner has stopped and the ssh is still up
        rc = proc.wait()
        with socket.create_connection(held[0].getsockname()) as late:
            late.sendall(b"late")
            assert sent(held[0], 4) == b"late"
        return rc

    runner = ("import socket\n"
              "conn, _ = socket.socket(fileno=fd).accept()\n"
              '(out / "first").write_bytes(conn.recv(5))\n'
              "conn.sendall(b'served')\n")
    end = A.certifier_run(host, run, pinned=PIN, tunnel_argv=tunnel_argv, up=up, command=session(runner, []),
                          wait=wait)
    d = host.certifier / run
    assert end["rc"] == 0 and end["ended"] == "ended", (d / A.WITNESS / "rehearse.log").read_text()
    assert ended_why(host, end) == "the session ended"
    assert (d / A.WITNESS / "first").read_bytes() == b"early" and early[0].recv(6) == b"served"
    early[0].close()
    assert mark.read_text() == "yes" and held[0].fileno() == -1


PROMPTS = [[70001, 3], [5], [7]]
SCAN = REQUEST + ["--scan", "--prompts", "--height", "10"]


def test_a_scan_rehearsal_takes_its_prompts_off_stdin_into_its_witness_alone_and_runs_rehearse_on_them(host):
    """`--prompts` reads the prompts off the gate's stdin and writes them, canonical, to `witness/prompts.json` (0700,
    the file 0600, both `auditor`'s); the request records their count and ids, never the ids, and `--requests` is their
    count unless given. The run runs `rehearse --prompts` on that file (with `--window-per-group` when asked), which
    stays in `witness/`, beside what the rehearsal writes, and `out/` publishes a grouped run's `groups.json`; `fetch`
    gives the prompts to the run's own lane alone."""
    owned, data = [], json.dumps(PROMPTS, indent=2).encode()
    rc, out = gate(host, SCAN + ["--window-per-group", "--tokens", "2", "--requests", "6", "--minutes", "4"],
                   stdin=data, own=owned.append, started=[])
    assert rc == 0, out
    run = json.loads(out)["run"]
    d, w = host.certifier / run, host.certifier / run / A.WITNESS
    req = json.loads((d / "request.json").read_text())
    assert {k: req[k] for k in ("device", "scan", "prompts", "window_per_group", "requests", "tokens", "height")} == \
        {"device": "warden", "scan": True, "prompts": {"count": 3, "ids": 4}, "window_per_group": True, "requests": 6,
         "tokens": 2, "height": 10}
    assert b"70001" not in out and b"70001" not in (d / "request.json").read_bytes()
    assert (w / A.PROMPTS).read_bytes() == b"[[70001,3],[5],[7]]"
    assert (w.stat().st_mode & 0o777, (w / A.PROMPTS).stat().st_mode & 0o777) == (0o700, 0o600)
    assert owned == [host.certifier, w / A.PROMPTS, w, d]

    rc, out = gate(host, SCAN + ["--minutes", "4"], stdin=data, started=[], token="cd34")
    assert rc == 0 and json.loads(out)["requests"] == 3 and "window_per_group" not in json.loads(out)

    seen = []
    counted = CHECK | {"summary": CHECK["summary"] | {"requests": req["requests"], "ended": req["requests"]}}
    grouped = WARDEN_FILES.replace(repr(CHECK), repr(counted)) + (
        '(out / "groups.json").write_text(json.dumps({"groups": [{"group": 0}]}))\n')
    end = A.certifier_run(host, run, pinned=PIN, tunnel_argv=lambda s, l: TUNNEL, up=lambda t, p: True,
                          command=rehearsal(grouped, seen), wait=lambda proc, s: proc.wait())
    (got, work, url), = seen
    assert got == req and work == w and end["rc"] == 0 and "witness" not in end
    assert (w / A.PROMPTS).read_bytes() == b"[[70001,3],[5],[7]]" and A.PROMPTS not in end_text(d)
    argv = A.warden_command(got, work, url)
    assert argv[argv.index(A.SCAN_SCHEDULE[-2]) + 2:] == ["--prompts", str(w / A.PROMPTS), "--window-per-group",
                                                       "--max-tokens", "2"]
    assert "groups.json" in end["files"] and json.loads((d / "out" / "groups.json").read_text())["groups"]
    public = b"".join(p.read_bytes() for p in (d / "out").rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    assert b"70001" not in public
    rc, blob = A.gate(["--ssh", f"fetch {run}", "--key", "memory-accounting"], host)
    assert rc == 0 and not any(n.startswith(f"{run}/witness") for n in tar_names(blob))
    rc, blob = A.gate(["--ssh", f"fetch {run}", "--key", LANE], host)
    assert rc == 0 and f"{run}/witness/prompts.json" in tar_names(blob)


@pytest.mark.parametrize("argv, stdin, why", [
    (["--prompts"], b"[[1]]", "--prompts: the scan's id requests' prompts, with --scan"),
    (["--scan", "--window-per-group"], b"[[1]]", "--window-per-group: groups of the --prompts"),
    (["--window-per-group"], b"[[1]]", "--window-per-group: groups of the --prompts"),
    (["--scan", "--prompts", "--requests", "0"], b"[[1]]", "--requests '0': 1 to 1700"),
    (["--scan", "--prompts", "--requests", "1701"], b"[[1]]", "--requests '1701': 1 to 1700"),
    (["--scan", "--prompts", "--tokens", "0"], b"[[1]]", "--tokens '0': 1 to 64"),
    (["--scan", "--prompts", "--tokens", "65"], b"[[1]]", "--tokens '65': 1 to 64"),
    *[(["--scan", "--prompts"], bad, "--prompts: stdin is a JSON array of 1 to 400 prompts, each an array of 1 to 1024 "
                                     "token ids below 2^17")
      for bad in (b"", b"x", b"[]", b"[[]]", b"[1]", b"[[1.0]]", b"[[131072]]", b"[[-1]]", b"[[true]]", b'[["1"]]',
                  b'{"p": [[1]]}', b"\xff", json.dumps([[1]] * 401).encode(), json.dumps([[0] * 1025]).encode(),
                  json.dumps([[1], [0] * 2048]).encode())],
    (["--scan", "--prompts"], b" " * A.WARDEN_PROMPTS_BYTES + b"[[1]]", "--prompts: over 1048576 bytes on stdin"),
    (["--scan", "--prompts", "--window-per-group"], json.dumps([[1]] * 5).encode(),
     "--window-per-group: 5 prompts, past a group's 4 (the scan's lanes)"),
    (["--scan", "--scan-class", "w63k4"], b"[[1]]", "--scan-class: the scan's class of the --prompts"),
    (["--scan", "--prompts", "--scan-class", "w99"], b"[[1]]", "--scan-class 'w99': one of ['w63', 'w63k4']"),
    (["--scan", "--prompts", "--scan-class", "w63k4", "--tokens", "5"], b"[[1]]", "--tokens '5': 1 to 4"),
    (["--scan", "--prompts", "--scan-class", "w63k4"], json.dumps([[0] * 33]).encode(),
     "--prompts: stdin is a JSON array of 1 to 400 prompts, each an array of 1 to 32 token ids below 2^17"),
    (["--scan", "--prompts", "--minutes", "3"], b"[[1]]", "--minutes 3: 1 requests take 4 minutes"),
    (["--scan", "--prompts", "--window-per-group", "--requests", "400", "--minutes", "45"], b"[[1]]",
     "--minutes 45: 400 groups take 46 minutes"),
])
def test_the_gate_refuses_prompts_it_cant_check_and_writes_nothing(host, argv, stdin, why):
    rc, out = gate(host, REQUEST + ["--height", "10"] + argv, stdin=stdin, started=[])
    assert rc == 3 and why in json.loads(out)["refused"]
    assert not host.certifier.exists()


def test_a_scan_rehearsals_prompt_may_be_the_scans_ki_ids_and_no_more():
    """A scan rehearsal's prompt is at most the scan's KI = 1024 ids (`WARDEN_PROMPT_IDS`; `test_warden_serve` holds
    it to `scan_zk.CLASS.KI`): the scan refuses a request of more, so the gate takes exactly KI and refuses KI + 1 at
    input. A varied certifier rehearsal's prompts, which no scan reads, keep the varied plan's 2048
    (`CERTIFIER_PROMPTS`)."""
    assert A.prompts_of(json.dumps([[0] * 1024, [1]]).encode()) == [[0] * 1024, [1]]
    with pytest.raises(A.Refused, match="each an array of 1 to 1024 token ids"):
        A.prompts_of(json.dumps([[0] * 1025]).encode())
    assert A.prompts_of(json.dumps([[0] * 2048]).encode(), A.CERTIFIER_PROMPTS) == [[0] * 2048]


def test_a_scan_rehearsal_at_another_class_runs_at_its_egress_cells_with_its_ki_and_tokens(host, tmp_path):
    """`--scan-class w63k4` is recorded (``scan_class``) only when given, takes a prompt of its KI = 32 ids and 4 tokens
    out (its 5 egress cells less the trailer), and runs the network certifier at the scan's schedule with its 5 egress cells, every
    other flag the class of record's (`test_warden_serve` holds each class's schedule to `scan_zk.py schedule --class`).
    A request without it is the class of record's: its bounds are `WARDEN_PROMPT_IDS` and `WARDEN_PROMPT_TOKENS`."""
    rc, out = gate(host, SCAN + ["--scan-class", "w63k4", "--window-per-group", "--tokens", "4", "--minutes", "4"],
                   stdin=json.dumps([[0] * 32] * 4).encode(), started=[])
    assert rc == 0, out
    req = json.loads((host.certifier / json.loads(out)["run"] / "request.json").read_text())
    assert (req["scan_class"], req["tokens"], req["prompts"]) == ("w63k4", 4, {"count": 4, "ids": 128})
    argv = A.warden_command(req, tmp_path, "u")
    schedule = argv[argv.index("--ids") + 1:][:len(A.SCAN_SCHEDULE)]
    assert schedule == list(A.scan_schedule(scan_class="w63k4")) and schedule[schedule.index("--egress-cells") + 1] == "5"
    assert [x for i, x in enumerate(schedule) if schedule[i - 1] != "--egress-cells"] == \
        [x for i, x in enumerate(A.SCAN_SCHEDULE) if A.SCAN_SCHEDULE[i - 1] != "--egress-cells"]
    both = A.scan_schedule(4000, "w63k4")
    assert both[::2] == A.SCAN_SCHEDULE[::2] and dict(zip(both[::2], both[1::2])) == \
        dict(zip(A.SCAN_SCHEDULE[::2], A.SCAN_SCHEDULE[1::2])) | {"--bucket-ms": "4000", "--egress-cells": "5"}
    assert A.scan_bounds("w63k4") == ((1, 32), (1, 4))
    assert A.scan_bounds() == A.scan_bounds(A.SCAN_CLASS) == (A.WARDEN_PROMPT_IDS, A.WARDEN_PROMPT_TOKENS)

    rc, out = gate(host, SCAN + ["--minutes", "4"], stdin=b"[[1]]", started=[], token="cd34")
    assert rc == 0, out
    req = json.loads((host.certifier / json.loads(out)["run"] / "request.json").read_text())
    assert "scan_class" not in req
    assert A.warden_command(req, tmp_path, "u") == A.warden_command(req | {"scan_class": A.SCAN_CLASS}, tmp_path, "u")


def test_a_groups_windows_are_the_scans_delay_and_its_tokens_buckets_and_their_minutes_the_groups_windows():
    """`test_warden_serve` holds `scan_group_windows` to `warden_serve.group_windows` and the warden's schedule."""
    assert [A.scan_group_windows(t) for t in (1, 2, 11, 12, 139, 140)] == [1, 1, 1, 2, 2, 3]
    assert [A.warden_group_minutes(g, 2, 10) for g in (1, 10, 100, 400)] == [4, 4, 9, 25]
    assert A.warden_group_minutes(400, 64, 10) == 46 and A.warden_group_minutes(1, 2) == 8
