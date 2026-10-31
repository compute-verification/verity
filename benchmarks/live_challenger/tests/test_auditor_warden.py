"""The gate's warden rehearsal (`warden-rehearse`, `auditor.py`), with its unit, the tunnel and the rehearsal injected:
the gate refuses a request it can't check, one whose minutes the traffic doesn't fit in, a read-only key, and a second
rehearsal (a certifier's or a warden's) while one runs; a lane's own key starts one, whose run (as `auditor`) runs the
warden through the tunnel into `witness/`, publishes to `out/` only the warden's deployment, registrations and
certificates, and writes its end; `fetch` gives `witness/` to the run's own lane and vy-admin alone."""

from __future__ import annotations

import io
import json
import shlex
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
    for key in (None, "-"):
        rc, out = gate(host, REQUEST, key=key, started=[])
        assert rc == 3 and "this key reads only: a warden rehearsal" in json.loads(out)["refused"]
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
    f'(out / "deployment.json").write_text(json.dumps({{"upstream": url, "session": "00" * 32}}))\n'
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
        "deployment.json", "cells.jsonl", "certified.jsonl", "cells.salts", "requests.records", "client.json",
        "check.json")}
    assert set(end["files"]) == {"deployment.json", "cells.jsonl", "certificates.jsonl", "tunnel.log", "rehearse.log"}
    out = d / "out"
    assert json.loads((out / "deployment.json").read_text())["upstream"] == url
    assert [json.loads(x) for x in (out / "certificates.jsonl").read_text().splitlines()] == [
        {"certificate": {"digest": "cd" * 64, "signature": "ef"}}, {"refused": ["warden-fails", "egress", 1]}]
    public = b"".join(p.read_bytes() for p in out.rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    for secret in (SALT_C.encode(), b"77" * 64, CELL_SALTS[:16], CELL_SALTS[:16].hex().encode(), b"prompt",
                   b"completion_tokens", b"data_bytes", b"text_bytes"):
        assert secret not in public

    rc, st = A.gate(["--ssh", f"status {run}"], host, certifier_active=lambda r: False)
    assert rc == 0 and json.loads(st)["end"] == end


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


def test_a_groups_windows_are_the_scans_delay_and_its_tokens_buckets_and_their_minutes_the_groups_windows():
    """`test_warden_serve` holds `scan_group_windows` to `warden_serve.group_windows` and the warden's schedule."""
    assert [A.scan_group_windows(t) for t in (1, 2, 11, 12, 139, 140)] == [1, 1, 1, 2, 2, 3]
    assert [A.warden_group_minutes(g, 2, 10) for g in (1, 10, 100, 400)] == [4, 4, 9, 25]
    assert A.warden_group_minutes(400, 64, 10) == 46 and A.warden_group_minutes(1, 2) == 8
