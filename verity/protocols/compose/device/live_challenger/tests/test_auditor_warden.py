"""The gate's warden rehearsal (`warden-rehearse`, `auditor.py`), with its unit, the tunnel and the rehearsal injected:
the gate refuses a request it can't check, one whose minutes the traffic doesn't fit in, a read-only key, and a second
rehearsal (a certifier's or a warden's) while one runs; a lane's own key starts one, whose run (as `auditor`) runs the
warden through the tunnel into `witness/`, publishes to `out/` only the warden's deployment, registrations and
certificates, and writes its end, a stop of its unit included; `fetch` gives `witness/` to the run's own lane and
vy-admin alone."""

from __future__ import annotations

import io
import json
import os
import shlex
import signal
import socket
import sys
import tarfile
from datetime import datetime, UTC
from pathlib import Path

import pytest

from verity.protocols.accounting.certifier.verifier import parties as NC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402

T0 = datetime(2026, 10, 24, 11, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 24, 11, 20, 0, tzinfo=UTC)
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
    so height 12's 4096 sign 819 windows and 1031 (55 minutes of the scan's 3.2 s windows) take 13; a load's heads come
    first. 1559 requests at the gate's rate are `warden_windows` 1031 and fit 55 minutes at height 13, which the gate
    takes; at 12 it refuses them with the height that fits, writing nothing, rather than run a session whose signer
    runs out 819 windows in. The default height, 20, signs anything the gate takes."""
    assert [A.warden_height(w) for w in (1, 204, 205, 819, 820, 1031, 1638, 1639)] == [10, 10, 11, 12, 13, 13, 13, 14]
    assert A.warden_height(1031, heads=733) == 13
    assert A.warden_height(819, heads=1) == 12 and A.warden_height(819, heads=2) == 13
    assert (A.warden_windows(1559), A.warden_windows(1700), A.warden_windows(1700, scan=False)) == (1031, 1119, 60)
    assert (A.warden_minutes(1559, 13), A.warden_minutes(1700)) == (55, 64)
    long = REQUEST[:-1] + ["55", "--scan", "--prompts", "--requests", "1559", "--height"]
    rc, out = gate(host, long + ["13"], stdin=b"[[1]]", started=[])
    assert rc == 0 and {k: json.loads(out)[k] for k in ("requests", "minutes", "height")} == \
        {"requests": 1559, "minutes": 55, "height": 13}
    rc, out = gate(host, long + ["12"], stdin=b"[[1]]", started=[], token="cd34")
    assert rc == 3 and json.loads(out)["refused"] == (
        "--height 12: 1559 requests take 1031 link-windows, 5155 one-time signatures (5 a window), past the key's "
        "4096: give --height 13 or more, or fewer --requests")
    assert not (host.certifier / "c20261024T110000Z-cd34").exists()
    rc, out = gate(host, REQUEST[:-1] + ["60", "--scan", "--requests", "1700", "--height", "12"], started=[],
                   token="ef56")
    assert rc == 3 and "give --height 13 or more" in json.loads(out)["refused"]
    rc, out = gate(host, REQUEST[:-1] + ["64", "--scan", "--requests", "1700"], started=[], token="ab34")
    assert rc == 0 and json.loads(out)["height"] == 20


SALT_C = "5a" * 64
CELL_SALTS = bytes(range(192))
REGISTRATION = {"format": "verity-network-warden/registration/v2", "value": "warden/egress/0", "root": "ab" * 64}
CHECK = {"accepted": True, "problems": [], "summary": {
    "requests": 4, "ended": 4, "refused": 0, "failed": [], "windows": 2, "certified_windows": 6, "certificates": 6,
    "signer_refused": [], "cell_rows": {"registered": 4, "of": 4}, "cells": {"ingress": 9, "egress": 31},
    "wire_bits": {"ingress": 18432, "egress": 63488}, "cells_in_their_bucket": [31, 31], "tick_lateness_ms": {"n": 9},
    "data_bytes": {"egress": 4242}, "completion_tokens": 77, "prompt_tokens": [16, 99], "arrival_lag_ms": {"n": 31},
    "finish_reasons": {"length": 4}}, "connections": [{"ordinal": 1, "outcome": "end", "text_bytes": 300}]}
#: a stand-in for `warden_serve.py rehearse`'s files: public, then witness
WARDEN_FILES = (
    f'(out / "deployment.json").write_text(json.dumps({{"session": "00" * 32}}))\n'
    f'(out / "endpoints.json").write_text(json.dumps({{"upstream": url}}))\n'
    f'(out / "cells.jsonl").write_text(json.dumps({{"registration": {REGISTRATION!r}, "cost": {{"rows": 600}}}}) + "\\n"'
    f' + json.dumps({{"refused": ["warden", "ingress", 1], "cost": {{"rows": 600}}}}) + "\\n")\n'
    f'(out / "certified.jsonl").write_text(json.dumps({{"salt_c": {SALT_C!r}, "head": {{"root": "77" * 64}}, '
    f'"certificate": {{"digest": "cd" * 64, "signature": "ef"}}}}) + "\\n" + json.dumps({{"refused": ["warden-fails", '
    f'"egress", 1]}}) + "\\n")\n'
    f'(out / "cells.salts").write_bytes({CELL_SALTS!r})\n'
    '(out / "requests.records").write_bytes(b"GET / prompt")\n'
    '(out / "client.json").write_text("{}")\n'
    f'(out / "check.json").write_text(json.dumps({CHECK!r}))\n')


def rehearsal(source: str, seen: list | None = None):
    """A stand-in for `warden_serve.py rehearse`: Python `source` run with `out` and `url` bound."""
    def command(req, out, url):
        if seen is not None:
            seen.append((req, out, url))
        return [sys.executable, "-c", f"import json, pathlib; out, url = pathlib.Path({str(out)!r}), {url!r}\n{source}"]
    return command


def run_warden(host, source=WARDEN_FILES, seen=None, token="ab12") -> tuple[str, dict]:
    rc, out = gate(host, REQUEST, started=[], token=token)
    assert rc == 0, out
    run = json.loads(out)["run"]
    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l: TUNNEL, up=lambda t, p: True,
                          command=rehearsal(source, seen), wait=lambda proc, s: proc.wait())
    return run, end


def tar_names(blob: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        return set(tar.getnames())


def test_a_warden_rehearsal_writes_witness_apart_and_publishes_its_deployment_registrations_and_certificates(host):
    seen = []
    run, end = run_warden(host, WARDEN_FILES, seen)
    d = host.certifier / run
    (req, work, url), = seen
    assert work == d / A.WITNESS and url.startswith("http://127.0.0.1:") and req["device"] == "warden"
    assert (d / A.WITNESS).stat().st_mode & 0o777 == 0o700
    assert end["device"] == "warden" and end["rc"] == 0 and end["ended"] == "the rehearsal ended"
    assert end["pinned"] == PIN and end["ended_utc"] == "2026-10-24T11:20:00Z" and "runs" not in end
    assert end["verdict"] == {"accepted": True, "problems": 0, **{k: CHECK["summary"][k] for k in A.WARDEN_COUNTS
                                                                   if k != "load"}} and "load" not in end
    assert end["registrations"] == {"warden/egress/0": "ab" * 64}
    assert end["witness"] == {n: (d / A.WITNESS / n).stat().st_size for n in (
        "deployment.json", "endpoints.json", "cells.jsonl", "certified.jsonl", "cells.salts", "requests.records",
        "client.json", "check.json")}
    assert set(end["files"]) == {"deployment.json", "cells.jsonl", "certificates.jsonl", "tunnel.log", "rehearse.log"}
    out = d / "out"
    assert json.loads((d / A.WITNESS / "endpoints.json").read_text())["upstream"] == url
    assert not {"listen", "upstream"} & set(json.loads((out / "deployment.json").read_text()))
    assert [json.loads(x) for x in (out / "certificates.jsonl").read_text().splitlines()] == [
        {"certificate": {"digest": "cd" * 64, "signature": "ef"}}, {"refused": ["warden-fails", "egress", 1]}]
    public = b"".join(p.read_bytes() for p in out.rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    for secret in (SALT_C.encode(), b"77" * 64, CELL_SALTS[:16], CELL_SALTS[:16].hex().encode(), b"prompt",
                   b"completion_tokens", b"data_bytes", b"text_bytes"):
        assert secret not in public

    rc, st = A.gate(["--ssh", f"status {run}"], host, certifier_active=lambda r: False)
    assert rc == 0 and json.loads(st)["end"] == end


#: each file a warden run publishes to `out/` (`WARDEN_PUBLIC`, and `warden_outcome`'s `certificates.jsonl`) by the
#: certifier's message that carries it to the verifier
PUBLISHED = {"deployment.json": NC.DEPLOYMENT, "cells.jsonl": NC.REGISTRATION, "load.json": NC.LOAD,
             "certificates.jsonl": NC.CERTIFICATE, "groups.json": NC.GROUPS}


def test_the_certifiers_manifest_declares_exactly_the_files_a_warden_run_publishes():
    """Every file a warden run publishes is one of the certifier's messages to the verifier, and the verifier's channel
    carries those messages and the out side's record, nothing else."""
    assert set(A.WARDEN_PUBLIC) | {"certificates.jsonl"} == set(PUBLISHED)
    (to_verifier,) = (c.messages for c in NC.MANIFEST.channels if c.receiver == NC.VERIFIER)
    assert sorted(to_verifier) == sorted({*PUBLISHED.values(), NC.OUT_RECORD})


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
    assert rc == 0 and {f"{run}/request.json", f"{run}/end.json", f"{run}/out/cells.jsonl",
                        f"{run}/out/certificates.jsonl", f"{run}/out/deployment.json"} <= names
    hidden = {n for n in names if n == f"{run}/witness" or n.startswith(f"{run}/witness/")}
    assert bool(hidden) == witness
    if witness:
        assert {f"{run}/witness/cells.salts", f"{run}/witness/certified.jsonl", f"{run}/witness/check.json"} <= hidden


def test_a_warden_rehearsal_that_never_ran_publishes_nothing_and_its_end_says_why(host):
    rc, out = gate(host, REQUEST, started=[])
    run = json.loads(out)["run"]
    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l: TUNNEL,
                          up=lambda t, p: False, command=lambda *a: pytest.fail("ran without its tunnel"))
    assert end["rc"] == 2 and "the tunnel to vy-nebius-1's 127.0.0.1:8011 didn't come up" in end["ended"]
    assert end["verdict"] is None and end["registrations"] == {} and end["witness"] == {}
    assert set(end["files"]) == {"tunnel.log"}

    run, end = run_warden(host, "raise SystemExit(1)", token="cd34")
    assert end["rc"] == 1 and end["ended"] == "the rehearsal exited 1" and end["verdict"] is None


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
    (SESSION[:2] + ["nobody"] + SESSION[3:], "--peer 'nobody': one of"),
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

    assert status() == {"run": run, "request": req, "end": None, "active": True, "relay": None}
    deployment = {"windows": 20, "origin_unix": 1.5, "session": "00" * 32}
    (d / A.WITNESS).mkdir(mode=0o700)
    (d / A.WITNESS / "deployment.json").write_text(json.dumps(deployment)[:-9])
    assert status()["relay"] is None
    (d / A.WITNESS / "deployment.json").unlink()

    seen, forwards = [], []
    files = WARDEN_FILES.replace('(out / "client.json").write_text("{}")\n', "").replace(
        '{"session": "00" * 32}', repr(deployment))
    assert repr(deployment) in files and "client.json" not in files
    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1,
                          tunnel_argv=lambda s, l, back: forwards.append((l, back)) or TUNNEL, up=lambda t, p: True,
                          command=session(files, seen), wait=lambda proc, s: proc.wait())
    ((got, work, url, listen),) = seen
    ((local, back),) = forwards
    assert got == req and work == d / A.WITNESS and url == f"http://127.0.0.1:{local}"
    assert back == (8012, listen) and listen != local
    assert end["rc"] == 0 and end["ended"] == "the session ended" and end["device"] == "warden"
    assert (end["verdict"]["accepted"], end["verdict"]["problems"]) == (True, 0)
    assert {k: end["verdict"][k] for k in ("windows", "certified_windows", "certificates")} == {
        k: CHECK["summary"][k] for k in ("windows", "certified_windows", "certificates")}
    assert "client.json" not in end["witness"] and "deployment.json" in end["files"]
    assert status() | {"active": False} == {"run": run, "request": req, "end": end, "active": False,
                                           "relay": {"client": "127.0.0.1:8012", "deployment": deployment}}
    rehearsal_status = json.loads(A.gate(["--ssh", f"status {run_warden(host, token='cd34')[0]}"], host,
                                         certifier_active=lambda r: False)[1])
    assert "relay" not in rehearsal_status


def test_a_relay_session_whose_ssh_exits_is_stopped_there_and_its_end_says_the_tunnel_closed(host):
    """A session's clients reach its relay through the unit's ssh alone. When that ssh exits (the peer refused a
    forward, or the link dropped) the unit stops the session's runner rather than let it run out its execution
    phases with no client, and the end is nonzero and says the tunnel closed, whatever the check then reads; a
    session whose runner fails says so as a rehearsal does."""
    rc, out = gate(host, SESSION, started=[])
    run = json.loads(out)["run"]
    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l, back: [sys.executable, "-c", ""],
                          up=lambda t, p: True, command=session("import signal; signal.pause()", []),
                          wait=lambda proc, s: proc.wait())
    assert end["rc"] != 0 and end["verdict"] is None
    assert end["ended"] == (f"the tunnel to {REQUEST[2]} closed before the session ended, and with it the clients' "
                            "way to the relay and the relay's to the endpoint (tunnel.log)")

    rc, out = gate(host, SESSION, started=[], token="cd34")
    end = A.certifier_run(host, json.loads(out)["run"], pinned=PIN, now=lambda: T1,
                          tunnel_argv=lambda s, l, back: TUNNEL, up=lambda t, p: True,
                          command=session("raise SystemExit(1)", []), wait=lambda proc, s: proc.wait())
    assert end["rc"] == 1 and end["ended"] == "the session exited 1"
    rc, out = gate(host, SESSION, started=[], token="ef56")
    end = A.certifier_run(host, json.loads(out)["run"], pinned=PIN, now=lambda: T1,
                          tunnel_argv=lambda s, l, back: TUNNEL, up=lambda t, p: False,
                          command=lambda *a, **k: pytest.fail("ran without its tunnel"))
    assert end["rc"] == 2 and "didn't come up (tunnel.log)" in end["ended"]


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
    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=tunnel_argv, up=up,
                          command=session(runner, []), wait=wait)
    d = host.certifier / run
    assert end["rc"] == 0 and end["ended"] == "the session ended", (d / "out" / "rehearse.log").read_text()
    assert (d / A.WITNESS / "first").read_bytes() == b"early" and early[0].recv(6) == b"served"
    early[0].close()
    assert mark.read_text() == "yes" and held[0].fileno() == -1


def test_a_relay_session_whose_unit_is_stopped_stops_its_runner_then_its_ssh_then_its_listener_and_writes_its_end(
        host, tmp_path, monkeypatch, stop_held):
    """The unit's stop (`systemctl stop`, its `RuntimeMaxSec`) sends `certifier-run` alone a SIGTERM
    (`KillMode=mixed`). It goes to the runner, which here ends only on a signal; once the runner has exited the ssh is
    still up and the listener still listens when the ssh is stopped, the listener is closed after, and the end says
    the unit stopped the session. The handler is this process's only while the run runs."""
    rc, out = gate(host, SESSION, started=[])
    run = json.loads(out)["run"]
    held, mark = [], tmp_path / "listened-at-the-sshs-end"
    monkeypatch.setattr(A, "held_listener", lambda make=A.held_listener: held.append(make()) or held[0])

    def tunnel_argv(ssh, local, back):
        return [sys.executable, "-c",
                "import pathlib, signal, socket, sys\n"
                "def stopped(*_):\n"
                f"    socket.create_connection(('127.0.0.1', {back[1]})).close()\n"
                f"    pathlib.Path({str(mark)!r}).write_text('yes')\n"
                "    sys.exit(0)\n"
                "signal.signal(signal.SIGTERM, stopped)\n"
                f"socket.create_connection(('127.0.0.1', {back[1]})).sendall(b'up')\n"
                "signal.pause()\n"]

    def up(tunnel, port):  # the ssh's stand-in holds its SIGTERM handler
        conn, _ = held[0].accept()
        with conn:
            return conn.recv(2) == b"up"

    def wait(proc, seconds):
        os.kill(os.getpid(), signal.SIGTERM)
        rc = proc.wait()
        assert not mark.exists() and held[0].fileno() != -1
        return rc

    before = signal.getsignal(signal.SIGTERM)
    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=tunnel_argv, up=up,
                          command=session("import signal; signal.pause()", []), wait=wait)
    assert end["rc"] == -signal.SIGTERM and end["ended"] == A.STOPPED
    assert mark.read_text() == "yes" and held[0].fileno() == -1
    assert json.loads((host.certifier / run / "end.json").read_text()) == end
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_relay_session_whose_unit_is_stopped_fails_though_its_runner_exits_0_on_the_signal(host, stop_held, tmp_path):
    """A stopped session is a failed run whatever its runner returns: a runner that exits 0 on the unit's SIGTERM still
    leaves the run's rc nonzero."""
    rc, out = gate(host, SESSION, started=[])
    ready = tmp_path / "ready"
    os.mkfifo(ready)

    def wait(proc, seconds):
        open(ready).close()
        os.kill(os.getpid(), signal.SIGTERM)
        return proc.wait()

    runner = ("import signal\nsignal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
              f"open({str(ready)!r}, 'w').close()\nsignal.pause()")
    end = A.certifier_run(host, json.loads(out)["run"], pinned=PIN, now=lambda: T1,
                          tunnel_argv=lambda s, l, back: TUNNEL, up=lambda tunnel, port: True,
                          command=session(runner, []), wait=wait)
    assert end["rc"] == 2 and end["ended"] == A.STOPPED


def test_a_relay_session_whose_unit_is_stopped_before_its_tunnel_is_up_runs_nothing_and_writes_its_end(
        host, stop_held):
    rc, out = gate(host, SESSION, started=[])
    run = json.loads(out)["run"]

    def up(tunnel, port):
        os.kill(os.getpid(), signal.SIGTERM)
        pytest.fail("the stop didn't end the wait for the tunnel")

    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l, back: TUNNEL, up=up,
                          command=lambda *a, **k: pytest.fail("ran after its unit was stopped"))
    assert end["rc"] == 2 and end["ended"] == f"{A.STOPPED} before its runner started" and end["verdict"] is None
    assert set(end["files"]) == {"tunnel.log"}


def test_a_relay_session_whose_unit_is_stopped_once_its_tunnel_didnt_come_up_still_writes_its_end(host, stop_held):
    """A stop at the step after the tunnel's check failed, the run holding why it ends, still closes the tunnel and
    writes the end. The stop is raised there through a line tracer on `certifier_run`'s frame, since a SIGTERM's
    handler runs only where the interpreter checks for one, which no signal sent from outside could aim at."""
    rc, out = gate(host, SESSION, started=[])
    run = json.loads(out)["run"]
    code, previous = A.certifier_run.__wrapped__.__code__, sys.gettrace()

    def line(frame, event, arg):
        if event == "line" and frame.f_locals.get("why") is not None:
            signal.raise_signal(signal.SIGTERM)
        return line

    sys.settrace(lambda frame, event, arg: line if frame.f_code is code else None)
    try:
        end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l, back: TUNNEL,
                              up=lambda tunnel, port: False, command=lambda *a, **k: pytest.fail("ran with no tunnel"))
    finally:
        sys.settrace(previous)
    assert end["rc"] == 2 and end["ended"] == f"{A.STOPPED} before its runner started" and end["verdict"] is None
    assert set(end["files"]) == {"tunnel.log"}


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
    grouped = WARDEN_FILES + '(out / "groups.json").write_text(json.dumps({"groups": [{"group": 0}]}))\n'
    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l: TUNNEL, up=lambda t, p: True,
                          command=rehearsal(grouped, seen), wait=lambda proc, s: proc.wait())
    (got, work, url), = seen
    assert got == req and work == w and end["rc"] == 0 and end["witness"][A.PROMPTS] == len(b"[[70001,3],[5],[7]]")
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
