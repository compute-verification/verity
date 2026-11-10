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


def gate(host, argv, *, by="network-accounting", active=lambda r: False, started=None, token="ab12", stdin=None,
         own=None):
    return A.gate(argv, host, by=by, now=T0, token=token, own=own or (lambda p: None), certifier_active=active,
                  start_certifier=(started.append if started is not None else lambda r: None),
                  read=lambda n: pytest.fail("read stdin") if stdin is None else stdin[:n])


def test_a_lane_key_starts_a_rehearsal_and_a_second_waits_for_it_to_end(host):
    started = []
    rc, out = gate(host, ["--ssh", shlex.join(REQUEST + ["--tamper", "extra"]), "--key", "network-accounting"],
                   started=started)
    doc = json.loads(out)
    assert rc == 0 and doc == {"run": "c20261023T110000Z-ab12", "peer": "vy-nebius-1", "port": 8011, "minutes": 30,
                               "tamper": "extra", "requests": 400, "height": 20, "profile": "planned",
                               "temperature": 0.0, "top_p": 1.0, "by": "network-accounting",
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
#: what both rehearsal verbs refuse of `--height`, (value, the refusal)
A_HEIGHT_REFUSED = [(v, "10 to 20") for v in ("9", "1", "0", "21", "30", "010", "20.0", "+20", "20 ", "1e1", "H", "")]


@pytest.mark.parametrize("bad, why", [
    (["--peer", "vy-challenger-1"], "--peer"),
    (["--peer", "vy-nebius-9"], "--peer"),
    (["--port", "80"], "--port '80': 1024 to 65535"),
    (["--port", "65536"], "--port"),
    (["--port", "08011"], "--port"),
    (["--port", "8011\n"], "--port"),
    (["--minutes", "0"], "--minutes '0': 1 to 170"),
    (["--minutes", "171"], "--minutes '171': 1 to 170"),
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
    (["--tamper", "burst"], "--tamper 'burst': one of ['late', 'extra', 'drop', 'token', 'byte'], or none for an honest shim"),
    (["--tamper", "honest"], "--tamper 'honest'"),
    (["--tamper", "Drop"], "--tamper 'Drop'"),
    (["--tamper", "drop,token"], "--tamper 'drop,token'"),
    (["--tamper", ""], "--tamper ''"),
    (["--profile", "both"], "--profile 'both': one of ['planned', 'served']"),
    (["--profile", "Served"], "--profile 'Served'"),
    (["--upstream", "http://10.0.0.2:8011"], "unrecognized arguments: --upstream"),
    (["--command", "id"], "unrecognized arguments: --command"),
    *[([flag, v], f"--{flag[2:]} {v!r}: {why}") for flag, why, values in A_SAMPLING_REFUSED for v in values],
    *[(["--height", v], f"--height {v!r}: {why}") for v, why in A_HEIGHT_REFUSED],
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


@pytest.mark.parametrize("tamper", ["late", "extra", "drop", "token"])
def test_each_tampering_shim_is_recorded_rehearsed_alone_and_its_verdict_ends_the_run(host, tmp_path, tamper):
    """late, extra and drop are runs the rehearsal's check refuses and token one it accepts (only the I/O tie refuses
    it), each the one strategy of its rehearsal, whose shim `certifier_serve.py` derives from the gate's own plan."""
    verdict = "accepted" if tamper == "token" else "refused"
    run = started_run(host, "--tamper", tamper)
    request = json.loads((host.certifier / run / "request.json").read_text())
    assert request["tamper"] == tamper
    assert A.certifier_command(request, tmp_path, "u")[-2:] == ["--strategies", tamper]

    summary = HONEST | {"verdict": verdict, "network_verdict": {"accepted": verdict == "accepted", "windows": 3}}
    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l: TUNNEL, up=lambda t, p: True,
                          command=rehearsal(f'(out / "summary.json").write_text(json.dumps({{"upstream": url, '
                                            f'"runs": {{{tamper!r}: {summary!r}}}}}))'),
                          wait=lambda proc, seconds: proc.wait())
    assert end["rc"] == 0 and end["runs"] == {tamper: {"verdict": verdict, "network": verdict == "accepted",
                                                       "audit": True, "salt": True,
                                                       "salt_again": "refused: already revealed"}}


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
                        "--upstream", "http://127.0.0.1:40001", "--out", str(tmp_path), "--height", "20",
                        "--requests", "400", "--temperature", "0.0", "--top-p", "1.0", "--profile", "planned",
                        "--strategies", "honest"]
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


@pytest.mark.parametrize("given, height", [([], 20), (["--height", "20"], 20), (["--height", "16"], 16),
                                           (["--height", "10"], 10), (["--traffic", "varied"], 20),
                                           (["--traffic", "varied", "--height", "10"], 10)])
def test_a_rehearsals_keys_are_the_deployed_keys_height_unless_its_request_gives_one(host, tmp_path, given, height):
    """The keys' height is the deployed key's (`lms.DEPLOYED`, 20) unless the request gives `--height` (10 to 20),
    recorded and passed on to `certifier_serve.py rehearse`, which makes both keys at it: a run of record never signs
    at a rehearsal's own 10 by default."""
    assert A.REHEARSAL_HEIGHT == (10, A.lms.DEPLOYED.H) and A.lms.DEPLOYED.H == 20
    rc, out = gate(host, REQUEST + given, started=[])
    assert rc == 0, out
    request = json.loads((host.certifier / json.loads(out)["run"] / "request.json").read_text())
    assert request["height"] == height
    argv = A.certifier_command(request, tmp_path, "u")
    assert argv.count("--height") == 1 and argv[argv.index("--height") + 1] == str(height)


def test_with_neither_in_flight_nor_tokens_the_request_and_command_are_todays(host, tmp_path):
    """A request without `--in-flight` or `--tokens` records neither and runs `certifier_serve.py rehearse` on its own
    4 and 64, exactly as before the gate took them; a request.json written before them reads the same. Its sampling is
    the defaults, temperature 0 and top_p 1, and its keys' height the deployed key's, 20, each always recorded and
    passed on, a request.json written before the gate took them running at the same."""
    run = started_run(host)
    request = json.loads((host.certifier / run / "request.json").read_text())
    assert "in_flight" not in request and "tokens" not in request
    assert request["temperature"] == 0.0 and request["top_p"] == 1.0 and request["height"] == 20
    argv = A.certifier_command(request, tmp_path, "http://127.0.0.1:40001", tree=Path("/opt/vy-challenger") / PIN)
    assert argv == [sys.executable, f"/opt/vy-challenger/{PIN}/benchmarks/network_traces/certifier_serve.py",
                    "rehearse", "--upstream", "http://127.0.0.1:40001", "--out", str(tmp_path), "--height", "20",
                    "--requests", "400", "--temperature", "0.0", "--top-p", "1.0", "--profile", "planned",
                    "--strategies", "honest"]
    older = {k: v for k, v in request.items() if k not in ("temperature", "top_p", "height")}
    assert A.certifier_command(older, tmp_path, "http://127.0.0.1:40001", tree=Path("/opt/vy-challenger") / PIN) == argv


VARIED = REQUEST + ["--traffic", "varied"]


@pytest.mark.parametrize("given, recorded, passed", [
    ([], {}, []),
    (["--tokens", "16"], {"tokens": 16}, ["--tokens", "16"]),
    (["--rate", "0.5"], {"rate": 0.5}, ["--rate", "0.5"]),
    (["--rate", "8"], {"rate": 8.0}, ["--rate", "8.0"]),
    (["--prompt-range", "1:2048"], {"prompt_range": "1:2048"}, ["--prompt-range", "1:2048"]),
    (["--prompt-range", "64:64"], {"prompt_range": "64:64"}, ["--prompt-range", "64:64"]),
    (["--seed", "0"], {"seed": 0}, ["--seed", "0"]),
    (["--seed", "4294967295"], {"seed": 4294967295}, ["--seed", "4294967295"]),
    (["--eos-ids", "128009"], {"eos_ids": [128009]}, ["--eos-ids", "128009"]),
    (["--eos-ids", "0,128001,128008,128009"], {"eos_ids": [0, 128001, 128008, 128009]},
     ["--eos-ids", "0,128001,128008,128009"]),
    (["--eos-ids", "128009", "--seed", "7", "--prompt-range", "16:512", "--rate", "1.5", "--tokens", "32"],
     {"tokens": 32, "rate": 1.5, "prompt_range": "16:512", "seed": 7, "eos_ids": [128009]},
     ["--tokens", "32", "--rate", "1.5", "--prompt-range", "16:512", "--seed", "7", "--eos-ids", "128009"]),
    (["--compact", "8"], {"compact": 8}, ["--compact", "8"]),
    (["--compact", "4096"], {"compact": 4096}, ["--compact", "4096"]),
    (["--compact", "16", "--eos-ids", "128009", "--tokens", "32"], {"tokens": 32, "eos_ids": [128009], "compact": 16},
     ["--tokens", "32", "--eos-ids", "128009", "--compact", "16"]),
    (["--margin", "1"], {"margin": 1.0}, ["--margin", "1.0"]),
    (["--margin", "8"], {"margin": 8.0}, ["--margin", "8.0"]),
    (["--margin", "4", "--compact", "8", "--rate", "0.5"], {"rate": 0.5, "compact": 8, "margin": 4.0},
     ["--rate", "0.5", "--compact", "8", "--margin", "4.0"]),
])
def test_a_varied_rehearsal_records_its_flags_when_given_and_runs_the_varied_plan_named_for_its_run(host, tmp_path,
                                                                                                   given, recorded,
                                                                                                   passed):
    """`--traffic varied` is recorded with each varied flag given and no N2 profile (the varied plan's is the advice),
    its sampling greedy, and its command runs `rehearse --traffic varied --name RUN`: every request's `user` is
    `RUN.j`, which no other run's is."""
    started = []
    rc, out = gate(host, VARIED + given + ["--tamper", "token"], started=started)
    doc = json.loads(out)
    assert rc == 0 and started == [doc["run"]]
    assert {k: v for k, v in doc.items() if k not in ("run", "by", "requested_utc", "unit")} == {
        "peer": "vy-nebius-1", "port": 8011, "minutes": 30, "tamper": "token", "requests": 400, "height": 20,
        "traffic": "varied", "temperature": 0.0, "top_p": 1.0} | recorded
    request = json.loads((host.certifier / doc["run"] / "request.json").read_text())
    assert request == {k: v for k, v in doc.items() if k != "unit"}
    argv = A.certifier_command(request, tmp_path, "http://127.0.0.1:40001", tree=Path("/opt/vy-challenger") / PIN)
    assert argv == [sys.executable, f"/opt/vy-challenger/{PIN}/benchmarks/network_traces/certifier_serve.py",
                    "rehearse", "--upstream", "http://127.0.0.1:40001", "--out", str(tmp_path), "--height", "20",
                    "--requests", "400", *passed, "--temperature", "0.0", "--top-p", "1.0", "--traffic", "varied", "--name", doc["run"],
                    "--strategies", "token"]


@pytest.mark.parametrize("bad, why", [
    (["--traffic", "burst"], "--traffic 'burst': one of ['fixed', 'varied']"),
    (["--traffic", "Varied"], "--traffic 'Varied'"),
    (["--traffic", "varied", "--in-flight", "2"], "--in-flight: fixed traffic's"),
    (["--traffic", "varied", "--profile", "planned"], "--profile: fixed traffic's"),
    (["--traffic", "varied", "--temperature", "0.6"], "--temperature, --top-p: fixed traffic's"),
    (["--traffic", "varied", "--top-p", "0.9"], "--temperature, --top-p: fixed traffic's"),
    (["--traffic", "varied", "--tokens", "65"], "--tokens '65': 16 to 64"),
    *[(["--traffic", "varied", "--rate", v], f"--rate {v!r}: above 0, at most 8")
      for v in ("0", "0.0", "8.0001", "9", "10", "-1", ".5", "2.", "1e0", "nan", "")],
    *[(["--traffic", "varied", "--prompt-range", v], f"--prompt-range {v!r}: LO:HI, 1 ≤ LO ≤ HI ≤ 2048")
      for v in ("0:16", "16:0", "64:16", "16:2049", "16", "16:", ":16", "16-512", "016:512", "16:512:1", " 16:512",
                "")],
    *[(["--traffic", "varied", "--seed", v], f"--seed {v!r}: 0 to 4294967295")
      for v in ("-1", "01", "4294967296", "1.5", "+3", "")],
    *[(["--traffic", "varied", "--eos-ids", v], f"--eos-ids {v!r}: 1 to 8 token ids, comma-separated")
      for v in ("", ",", "1,", ",1", "1,,2", "01", "-1", "1 2", "12345678", "1,2,3,4,5,6,7,8,9")],
    (["--traffic", "varied", "--minutes", "1"], "--minutes 1: 400 requests at 2 a second arrive over 200 seconds"),
    (["--traffic", "varied", "--minutes", "1", "--requests", "100", "--rate", "1.5"],
     "--minutes 1: 100 requests at 1.5 a second arrive over 66.6667 seconds"),
    *[(["--traffic", "varied", "--compact", v], f"--compact {v!r}: 8 to 4096")
      for v in ("7", "0", "4097", "65536", "-8", "08", "8.0", "B", "")],
    *[(["--traffic", "varied", "--margin", v], f"--margin {v!r}: 1 to 8, a plain decimal")
      for v in ("0", "0.5", "0.9999", "8.0001", "9", "10", "-1", ".5", "4.", "1e0", "nan", "inf", "")],
    *[([flag, v], f"{flag}: varied traffic's (--traffic varied)")
      for flag, v in (("--rate", "2"), ("--prompt-range", "16:1024"), ("--seed", "0"), ("--eos-ids", "2"),
                      ("--compact", "8"), ("--margin", "4"))],
    *[(["--traffic", "fixed", flag, v], f"{flag}: varied traffic's (--traffic varied)")
      for flag, v in (("--rate", "2"), ("--seed", "0"), ("--compact", "8"), ("--margin", "2"))],
    (["--traffic", "varied", "--name", "x"], "unrecognized arguments: --name"),
])
def test_the_gate_refuses_a_varied_rehearsal_it_cant_check_and_varied_flags_on_fixed_traffic(host, bad, why):
    good = dict(zip(REQUEST[1::2], REQUEST[2::2]))
    good.update(dict(zip(bad[::2], bad[1::2])))
    argv = ["--ssh", shlex.join(["certifier-rehearse", *[w for kv in good.items() for w in kv]]), "--key",
            "network-accounting"]
    rc, out = gate(host, argv, started=[])
    assert rc == 3 and why in json.loads(out)["refused"], out
    assert not host.certifier.exists()


PROMPTS = [[70001, 3], [5], [7]]


def test_a_varied_rehearsal_takes_its_prompts_off_stdin_into_its_witness_and_rehearses_on_that_file(host):
    """`--prompts` on varied traffic reads the prompts off the gate's stdin as a scan rehearsal's are read and writes
    them, canonical, to `witness/prompts.json` (0700, the file 0600, both `auditor`'s); the request records their count
    and ids, never the ids, and keeps its own `--requests`. The run's command passes that file, beside the run's `out/`,
    as `rehearse --prompts-file`, with no `--prompt-range`; `fetch` gives the witness to the run's own lane alone."""
    owned, data = [], json.dumps(PROMPTS, indent=2).encode()
    rc, out = gate(host, VARIED + ["--prompts", "--margin", "4", "--requests", "100", "--rate", "0.2", "--tokens",
                                   "24", "--eos-ids", "0", "--minutes", "45"], stdin=data, own=owned.append,
                   started=[])
    assert rc == 0, out
    run = json.loads(out)["run"]
    d, w = host.certifier / run, host.certifier / run / A.WITNESS
    req = json.loads((d / "request.json").read_text())
    assert {k: req[k] for k in ("traffic", "prompts", "margin", "requests", "rate", "tokens", "eos_ids")} == {
        "traffic": "varied", "prompts": {"count": 3, "ids": 4}, "margin": 4.0, "requests": 100, "rate": 0.2,
        "tokens": 24, "eos_ids": [0]}
    assert "device" not in req and "prompt_range" not in req
    assert b"70001" not in out and b"70001" not in (d / "request.json").read_bytes()
    assert (w / A.PROMPTS).read_bytes() == b"[[70001,3],[5],[7]]"
    assert (w.stat().st_mode & 0o777, (w / A.PROMPTS).stat().st_mode & 0o777) == (0o700, 0o600)
    assert owned == [host.certifier, w / A.PROMPTS, w, d]

    seen = []
    end = A.certifier_run(host, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l: TUNNEL, up=lambda t, p: True,
                          command=lambda req, out, url: seen.append((req, out, url)) or rehearsal(SUMMARY)(req, out,
                                                                                                         url),
                          wait=lambda proc, s: proc.wait())
    (got, work, url), = seen
    assert got == req and work == d / "out" and end["rc"] == 0
    argv = A.certifier_command(got, work, url, tree=Path("/opt/vy-challenger") / PIN)
    assert argv[argv.index("--requests"):argv.index("--temperature")] == [
        "--requests", "100", "--tokens", "24", "--rate", "0.2", "--eos-ids", "0", "--margin", "4.0",
        "--prompts-file", str(w / A.PROMPTS)]
    assert "--prompt-range" not in argv and argv[argv.index("--traffic"):] == [
        "--traffic", "varied", "--name", run, "--strategies", "honest"]
    for key, witness in (("memory-accounting", False), ("network-accounting", True)):
        rc, blob = A.gate(["--ssh", f"fetch {run}", "--key", key], host)
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            names = tar.getnames()
        assert rc == 0 and (f"{run}/witness/prompts.json" in names) == witness
        assert witness or not any(n.startswith(f"{run}/witness") for n in names)


@pytest.mark.parametrize("argv, stdin, why", [
    (["--prompts"], b"[[1]]", "--prompts: varied traffic's (--traffic varied)"),
    (["--traffic", "fixed", "--prompts"], b"[[1]]", "--prompts: varied traffic's (--traffic varied)"),
    (["--traffic", "varied", "--prompts", "--prompt-range", "8:8"], b"[[1]]",
     "--prompt-range: the chat turns' lengths, which --prompts replaces"),
    *[(["--traffic", "varied", "--prompts"], bad, ("--prompts: stdin is a JSON array of 1 to 400 prompts, each an array "
                                                   "of 1 to 2048 token ids below 2^17"))
      for bad in (b"", b"[]", b"[[]]", b"[1]", b"[[1.0]]", b"[[131072]]", b"[[-1]]", b'{"p": [[1]]}',
                  json.dumps([[1]] * 401).encode(), json.dumps([[0] * 2049]).encode())],
    (["--traffic", "varied", "--prompts"], b" " * A.WARDEN_PROMPTS_BYTES + b"[[1]]",
     "--prompts: over 1048576 bytes on stdin"),
])
def test_the_gate_refuses_a_varied_rehearsals_prompts_it_cant_check_and_writes_nothing(host, argv, stdin, why):
    rc, out = gate(host, REQUEST + argv, stdin=stdin, started=[])
    assert rc == 3 and why in json.loads(out)["refused"], out
    assert not host.certifier.exists()


def test_fixed_traffic_given_or_not_is_todays_request_and_command(host, tmp_path):
    """`--traffic fixed`, the default, records no traffic and runs the command a request without it ran, N2's profile
    planned unless given."""
    for extra in ([], ["--traffic", "fixed"]):
        run = started_run(host, *extra, token="ab12" if not extra else "cd34")
        request = json.loads((host.certifier / run / "request.json").read_text())
        assert "traffic" not in request and request["profile"] == "planned"
        argv = A.certifier_command(request, tmp_path, "u")
        assert "--traffic" not in argv and "--name" not in argv and argv[-4:] == ["--profile", "planned",
                                                                                   "--strategies", "honest"]
