"""The gate's certifier rehearsal (`certifier-rehearse`, `auditor.py`), with its unit, the tunnel and the rehearsal
injected: the gate refuses a request it can't check, a read-only key and a second rehearsal while one runs; a lane's own
key starts one, whose run (as `auditor`) opens the tunnel to the peer's endpoint, runs the rehearsal through it for at
most its minutes, closes the tunnel and writes its end, which `status` and `fetch` return."""

from __future__ import annotations

import io
import json
import shlex
import signal
import sys
import tarfile
from datetime import datetime, timezone

import pytest

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402

T0 = datetime(2026, 10, 23, 11, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 23, 11, 30, 0, tzinfo=timezone.utc)
PIN = "c" * 40
REQUEST = ["certifier-rehearse", "--peer", "vy-nebius-1", "--port", "8011", "--minutes", "30"]
#: a tunnel that stays up until the run closes it
TUNNEL = [sys.executable, "-c", "import signal; signal.pause()"]


@pytest.fixture
def host(tmp_path) -> A.Host:
    (tmp_path / "var").mkdir()
    (tmp_path / "peers").write_text("# peers\nvy-challenger-1 research@10.0.0.1\n"
                                    "vy-nebius-1 -i /var/lib/vy-challenger/.ssh/vy-challenger-tunnel research@10.0.0.2\n")
    return A.Host(root=tmp_path / "var", peers=tmp_path / "peers", pins=tmp_path / "pins.jsonl")


def gate(host, argv, *, by="network-accounting", active=lambda r: False, started=None, token="ab12"):
    return A.gate(argv, host, by=by, now=T0, token=token, own=lambda p: None, certifier_active=active,
                  start_certifier=(started.append if started is not None else lambda r: None))


def test_a_lane_key_starts_a_rehearsal_and_a_second_waits_for_it_to_end(host):
    started = []
    rc, out = gate(host, ["--ssh", shlex.join(REQUEST + ["--tamper", "extra"]), "--key", "network-accounting"],
                   started=started)
    doc = json.loads(out)
    assert rc == 0 and doc == {"run": "c20261023T110000Z-ab12", "peer": "vy-nebius-1", "port": 8011, "minutes": 30,
                               "tamper": "extra", "requests": 400, "profile": "planned", "temperature": 0.0,
                               "top_p": 1.0, "by": "network-accounting",
                               "requested_utc": "2026-10-23T11:00:00Z", "unit": "vy-certifier@c20261023T110000Z-ab12.service"}
    assert started == [doc["run"]]
    request = json.loads((host.certifier / doc["run"] / "request.json").read_text())
    assert request == {k: v for k, v in doc.items() if k != "unit"}
    assert A.read_ledger(host.ledger) == []

    rc, out = gate(host, REQUEST, active=lambda r: r == doc["run"], started=started, token="cd34")
    assert rc == 3 and "c20261023T110000Z-ab12 still runs: one at a time" in json.loads(out)["refused"]
    assert started == [doc["run"]] and A.certifier_runs(host) == [doc["run"]]
    rc, out = gate(host, REQUEST, started=started, token="cd34")
    assert rc == 0 and started == [doc["run"], "c20261023T110000Z-cd34"]
    assert json.loads(out)["tamper"] is None and json.loads(out)["by"] == "network-accounting"
    rc, out = gate(host, REQUEST + ["--profile", "served"], started=started, token="ef56")
    assert rc == 0 and json.loads(out)["profile"] == "served"


#: what both rehearsal verbs refuse of `--temperature` and `--top-p`, (flag, the refusal, values)
A_SAMPLING_REFUSED = [
    ("--temperature", "0 to 2", ["2.01", "2.5", "3", "-0.1", "-1", ".5", "1.", "0.00001", "1e-1", "nan", "inf",
                                 "+1", "0x1", "1,5", " 1", "1\n", ""]),
    ("--top-p", "above 0, at most 1", ["0", "0.0", "0.0000", "1.0001", "1.1", "2", "-0.5", ".9", "nan", "1e0", "9e-1",
                                       "0.99999", ""]),
]


@pytest.mark.parametrize("bad, why", [
    (["--peer", "vy-challenger-1"], "--peer"),
    (["--peer", "vy-nebius-9"], "--peer"),
    (["--port", "80"], "--port '80': 1024 to 65535"),
    (["--port", "65536"], "--port"),
    (["--port", "08011"], "--port"),
    (["--port", "8011\n"], "--port"),
    (["--minutes", "0"], "--minutes '0': 1 to 120"),
    (["--minutes", "121"], "--minutes"),
    (["--requests", "3"], "--requests '3': 4 to 400"),
    (["--requests", "401"], "--requests"),
    (["--in-flight", "0"], "--in-flight '0': 1 to 4"),
    (["--in-flight", "5"], "--in-flight '5': 1 to 4"),
    (["--in-flight", "04"], "--in-flight '04': 1 to 4"),
    (["--in-flight", "2 "], "--in-flight '2 ': 1 to 4"),
    (["--tokens", "15"], "--tokens '15': 16 to 64"),
    (["--tokens", "65"], "--tokens '65': 16 to 64"),
    (["--tokens", "+32"], "--tokens '+32': 16 to 64"),
    (["--tokens", "32\n"], "--tokens '32\\n': 16 to 64"),
    (["--in", "2"], "unrecognized arguments: --in"),
    (["--delta-buckets", "74"], "unrecognized arguments: --delta-buckets"),
    (["--tamper", "burst"], "--tamper 'burst'"),
    (["--profile", "both"], "--profile 'both': one of ['planned', 'served']"),
    (["--profile", "Served"], "--profile 'Served'"),
    (["--upstream", "http://10.0.0.2:8011"], "unrecognized arguments: --upstream"),
    (["--command", "id"], "unrecognized arguments: --command"),
    *[([flag, v], f"--{flag[2:]} {v!r}: {why}") for flag, why, values in A_SAMPLING_REFUSED for v in values],
])
def test_the_gate_refuses_a_rehearsal_it_cant_check(host, bad, why):
    good = dict(zip(REQUEST[1::2], REQUEST[2::2]))
    good.update(dict(zip(bad[::2], bad[1::2])))
    argv = ["--ssh", shlex.join(["certifier-rehearse", *[w for kv in good.items() for w in kv]]), "--key",
            "network-accounting"]
    rc, out = gate(host, argv, started=[])
    assert rc == 3 and why in json.loads(out)["refused"]
    assert not host.certifier.exists()


def test_a_key_that_reads_only_starts_no_rehearsal(host):
    for argv in (["--ssh", shlex.join(REQUEST)], ["--ssh", shlex.join(REQUEST), "--key", "-"]):
        rc, out = gate(host, argv, started=[])
        assert rc == 3 and "this key reads only" in json.loads(out)["refused"]
    assert not host.certifier.exists()


def started_run(host, *extra, token="ab12") -> str:
    rc, out = gate(host, REQUEST + list(extra), started=[], token=token)
    assert rc == 0, out
    return json.loads(out)["run"]


def rehearsal(source: str):
    """A stand-in for `certifier_serve.py rehearse`: Python `source` run with `out` and `url` bound."""
    return lambda req, out, url: [sys.executable, "-c", f"import json, pathlib, sys; out, url = pathlib.Path({str(out)!r}), "
                                                        f"{url!r}\n{source}"]


HONEST = {"verdict": "accepted", "network_verdict": {"accepted": True, "windows": 3}, "audit_accepted": True,
          "salt_accepted": True, "salt_again": "refused: already revealed", "exit": {"serve": 0, "client": 0}}
SUMMARY = (f'(out / "summary.json").write_text(json.dumps({{"upstream": url, "runs": {{"honest": {HONEST!r}}}}}))\n'
           '(out / "honest").mkdir(); (out / "honest" / "check.json").write_text("{}")')


def test_a_rehearsal_runs_through_the_tunnel_and_its_end_is_status_and_fetch(host):
    run = started_run(host)
    tunnels, waited = [], []

    def tunnel_argv(ssh, local):
        tunnels.append((ssh, local))
        return TUNNEL

    def wait(proc, seconds):
        waited.append(seconds)
        return proc.wait()

    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=tunnel_argv, up=lambda t, p: True,
                          command=rehearsal(SUMMARY), wait=wait)
    ssh, local = tunnels[0]
    assert ssh == ["-i", "/var/lib/vy-challenger/.ssh/vy-challenger-tunnel", "research@10.0.0.2"] and waited == [30 * 60]
    assert end["rc"] == 0 and end["ended"] == "the rehearsal ended"
    assert end["runs"] == {"honest": {"verdict": "accepted", "network": True, "audit": True, "salt": True,
                                      "salt_again": "refused: already revealed"}}
    assert end["pinned"] == PIN and end["ended_utc"] == "2026-10-23T11:30:00Z"
    out = host.certifier / run / "out"
    assert json.loads((out / "summary.json").read_text())["upstream"] == f"http://127.0.0.1:{local}"
    assert set(end["files"]) == {"summary.json", "honest/check.json", "tunnel.log", "rehearse.log"}

    rc, st = A.gate(["--ssh", f"status {run}"], host, certifier_active=lambda r: True)
    doc = json.loads(st)
    assert rc == 0 and doc["end"] == end and doc["request"]["port"] == 8011 and not doc["active"]
    rc, tgz = A.gate(["--ssh", f"fetch {run}"], host)
    with tarfile.open(fileobj=io.BytesIO(tgz)) as tar:
        assert {f"{run}/request.json", f"{run}/end.json", f"{run}/out/summary.json",
                f"{run}/out/honest/check.json"} <= set(tar.getnames())
    with pytest.raises(A.Refused, match="has run already"):
        A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l: TUNNEL, up=lambda t, p: True,
                        command=rehearsal(SUMMARY), wait=wait)
    for argv in (["status", "c20261023T110000Z-ffff"], ["fetch", "c20261023T110000Z-ffff"], ["status", "c../../etc"]):
        rc, out = A.gate(argv, host)
        assert rc == 3 and json.loads(out)["refused"]


def test_a_rehearsal_past_its_minutes_is_stopped_and_a_tunnel_that_never_came_up_runs_none(host):
    run = started_run(host, "--tamper", "late")

    def past_its_minutes(proc, seconds):
        proc.send_signal(signal.SIGTERM)
        return proc.wait()

    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l: TUNNEL, up=lambda t, p: True,
                          command=rehearsal("import signal; signal.pause()"), wait=past_its_minutes)
    assert end["rc"] == -signal.SIGTERM and end["ended"] == "stopped at --minutes 30" and end["runs"] == {}

    run = started_run(host, "--minutes", "5", token="cd34")
    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l: TUNNEL, up=lambda t, p: False,
                          command=lambda *a: pytest.fail("ran without its tunnel"))
    assert end["rc"] == 2 and "the tunnel to vy-nebius-1's 127.0.0.1:8011 didn't come up" in end["ended"]
    assert set(end["files"]) == {"tunnel.log"}


def test_the_rehearsal_runs_certifier_serve_on_the_requests_shim_and_profile_through_the_tunnels_local_end(tmp_path):
    """N2's profile is the request's, planned for a request.json written before the gate took `--profile`."""
    req = {"requests": 400, "tamper": None, "profile": "planned"}
    argv = A.certifier_command(req, tmp_path, "http://127.0.0.1:40001", tree=Path("/opt/vy-challenger") / PIN)
    assert argv[1:] == [f"/opt/vy-challenger/{PIN}/benchmarks/network_traces/certifier_serve.py", "rehearse",
                        "--upstream", "http://127.0.0.1:40001", "--out", str(tmp_path), "--requests", "400",
                        "--temperature", "0.0", "--top-p", "1.0", "--profile", "planned", "--strategies", "honest"]
    assert A.certifier_command(req | {"tamper": "late"}, tmp_path, "u")[-2:] == ["--strategies", "late"]
    assert A.certifier_command(req | {"profile": "served"}, tmp_path, "u")[-4:-2] == ["--profile", "served"]
    del req["profile"]
    assert A.certifier_command(req, tmp_path, "u")[-4:-2] == ["--profile", "planned"]
    assert A.certifier_env(Path("/opt/vy-challenger") / PIN)["PYTHONPATH"] == f"/opt/vy-challenger/{PIN}"


@pytest.mark.parametrize("given, recorded, passed", [
    (["--in-flight", "1"], {"in_flight": 1}, ["--in-flight", "1"]),
    (["--in-flight", "4"], {"in_flight": 4}, ["--in-flight", "4"]),
    (["--tokens", "16"], {"tokens": 16}, ["--tokens", "16"]),
    (["--tokens", "64"], {"tokens": 64}, ["--tokens", "64"]),
    (["--tokens", "32", "--in-flight", "2"], {"in_flight": 2, "tokens": 32}, ["--in-flight", "2", "--tokens", "32"]),
])
def test_in_flight_and_tokens_within_their_bounds_are_recorded_and_reach_the_rehearsal(host, tmp_path, given,
                                                                                        recorded, passed):
    started = []
    rc, out = gate(host, REQUEST + given, started=started)
    doc = json.loads(out)
    assert rc == 0 and started == [doc["run"]]
    assert {k: doc[k] for k in ("in_flight", "tokens") if k in doc} == recorded
    request = json.loads((host.certifier / doc["run"] / "request.json").read_text())
    assert request == {k: v for k, v in doc.items() if k != "unit"}
    argv = A.certifier_command(request, tmp_path, "u")
    assert argv[argv.index("--requests"):] == ["--requests", "400", *passed, "--temperature", "0.0", "--top-p", "1.0",
                                               "--profile", "planned", "--strategies", "honest"]


@pytest.mark.parametrize("given, recorded", [
    (["--temperature", "0.6", "--top-p", "0.9"], {"temperature": 0.6, "top_p": 0.9}),
    (["--temperature", "0"], {"temperature": 0.0, "top_p": 1.0}),
    (["--temperature", "2"], {"temperature": 2.0, "top_p": 1.0}),
    (["--temperature", "2.0000"], {"temperature": 2.0, "top_p": 1.0}),
    (["--temperature", "1.2345"], {"temperature": 1.2345, "top_p": 1.0}),
    (["--top-p", "1"], {"temperature": 0.0, "top_p": 1.0}),
    (["--top-p", "1.0"], {"temperature": 0.0, "top_p": 1.0}),
    (["--top-p", "0.0001"], {"temperature": 0.0, "top_p": 0.0001}),
])
def test_a_rehearsals_sampling_is_recorded_and_reaches_every_request(host, tmp_path, given, recorded):
    """Temperature 0 to 2 and top_p above 0 up to 1, the defaults 0 and 1, are recorded and passed on, so every request
    carries both (`certifier_serve.request_body`), and an endpoint's own generation config never applies."""
    rc, out = gate(host, REQUEST + given, started=[])
    doc = json.loads(out)
    assert rc == 0 and {k: doc[k] for k in ("temperature", "top_p")} == recorded
    request = json.loads((host.certifier / doc["run"] / "request.json").read_text())
    argv = A.certifier_command(request, tmp_path, "u")
    assert argv[argv.index("--temperature"):argv.index("--profile")] == [
        "--temperature", str(recorded["temperature"]), "--top-p", str(recorded["top_p"])]
    assert [float(argv[argv.index(f) + 1]) for f in ("--temperature", "--top-p")] == list(recorded.values())


def test_with_neither_in_flight_nor_tokens_the_request_and_command_are_todays(host, tmp_path):
    """A request without `--in-flight` or `--tokens` records neither and runs `certifier_serve.py rehearse` on its own
    4 and 64, exactly as before the gate took them; a request.json written before them reads the same. Its sampling is
    the defaults, temperature 0 and top_p 1, always recorded and passed on."""
    run = started_run(host)
    request = json.loads((host.certifier / run / "request.json").read_text())
    assert "in_flight" not in request and "tokens" not in request
    assert request["temperature"] == 0.0 and request["top_p"] == 1.0
    argv = A.certifier_command(request, tmp_path, "http://127.0.0.1:40001", tree=Path("/opt/vy-challenger") / PIN)
    assert argv == [sys.executable, f"/opt/vy-challenger/{PIN}/benchmarks/network_traces/certifier_serve.py",
                    "rehearse", "--upstream", "http://127.0.0.1:40001", "--out", str(tmp_path), "--requests", "400",
                    "--temperature", "0.0", "--top-p", "1.0", "--profile", "planned", "--strategies", "honest"]
    older = {k: v for k, v in request.items() if k not in ("temperature", "top_p")}
    assert A.certifier_command(older, tmp_path, "http://127.0.0.1:40001", tree=Path("/opt/vy-challenger") / PIN) == argv
