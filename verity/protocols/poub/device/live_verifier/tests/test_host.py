"""The verifier host's gate (`host.py`) with its root steps (chown, systemctl) and the clock injected, on the deployed
`verifier.json` with a dummy challenger ssh: a run is held to the whitelist, written as its request and started as its
one unit, and its command line is one verifier.py's own parser reads as the request; the gate refuses any other verb,
any other form of its forced command, a parameter the whitelist doesn't hold, and a read-only key's run; a run is
stopped only by its lane or vy-admin, a replay or a close waits for no unit to run, and a run whose pin moved, or on a
host with no first-parent list of main, doesn't run. The gate's stand-ins are `gate_harness.py`'s; a run through the
gate that needs the remote session's code (a real session, its line of record, a synthetic checkpoint's binding) is
tested beside it (`benchmarks/remote_session/tests/test_live_verifier.py`)."""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from gate_harness import (BASE, BOUND, KEY, LANE, PACED, PACKAGING, PIN, ROLE, ROOT, RUN, SSH, T0, T1, H, Units, V,
                          call, gate, host_at, parsed, refused)
from verity.protocols.poub.verifier.session import challengers

HERE = Path(__file__).resolve().parent
SESSIONS = ["run", "--peer", "gpu-node", "--port", "7958", "--period", "memory-line-1", "--pin", KEY,
            "--sessions", "3", "--windows", "7", "--window-ms", "1000", "--samples", "2773", "--minutes", "30",
            "--jitter-ms", "2.5", "--quantile", "p99", "--lifetime", "4"]


@pytest.fixture
def svc(tmp_path) -> H.Host:
    return host_at(tmp_path)


def test_a_run_is_held_to_the_whitelist_written_as_its_request_and_started_as_its_one_unit(svc):
    units = Units()
    rc, out = call(svc, PACED, units)
    doc = json.loads(out)
    assert rc == 0, doc
    assert (doc["run"], doc["by"], doc["pinned"], doc["peer"], doc["listen"]) == (
        RUN, LANE, PIN, "gpu-node", "127.0.0.1:7951")
    assert units.started == [f"vy-verifier@{RUN}.service"]
    req = json.loads((svc.runs / RUN / "request.json").read_text())
    assert {k: v for k, v in req.items() if k != "argv"} == {
        "run": RUN, "by": LANE, "requested_utc": "2026-10-21T13:00:00Z", "pinned": PIN,
        "pin_log": {"line": 0, "utc": "2026-10-21T13:00:00Z"}, "peer": "gpu-node", "port": 7951,
        "period": "memory-line-1", "pin": KEY, "paced": ["2:8000:40:2", "4:16000:40:2"]}
    argv = req["argv"]
    assert argv == doc["argv"]
    assert argv[:3] == ["/usr/bin/python3", "-I",
                        str(H.TREE / "verity" / "protocols" / "poub" / "verifier" / "remote_session" / "verifier_process.py")]
    i = argv.index("--paced")
    assert dict(zip(argv[3:i:2], argv[4:i:2])) == {
        "--host": "127.0.0.1", "--port": "7951", "--out": str(svc.runs / RUN / "out"), "--registry": str(svc.registry),
        "--transcript": str(svc.scratch / RUN / "transcripts"), "--challenger": "gate",
        "--challenger-ssh": SSH, "--challenger-peer": "auditor-host", "--challenger-pin": KEY,
        "--challenger-period": "memory-line-1", "--challenger-main": str(svc.main), "--host-bytes": "1843153879040",
        "--ingress-bytes-per-s": "4400000000.0"}
    assert argv[i:] == ["--paced", "2:8000:40:2", "4:16000:40:2"]
    assert "run" in refused(call(svc, PACED, units, token="cd34")) and units.started == [f"vy-verifier@{RUN}.service"]
    assert [p.name for p in svc.runs.iterdir()] == [RUN]


def test_a_run_is_refused_before_anything_starts_while_the_host_has_no_first_parent_list_of_main(svc):
    """The verifier refuses every gate session whose challenger pin isn't on main's first-parent line, read from the
    list infra writes at `Host.main`; without that list no run starts, so none ends with every session refused."""
    units = Units()
    svc.main.unlink()
    assert refused(call(svc, PACED, units)) == (
        f"this host has no {svc.main}, main's first-parent history (infra writes `git rev-list --first-parent "
        "origin/main` there, root's, 0644): the verifier refuses every gate session whose challenger pin it can't find "
        "on it, so no run starts without it")
    assert units.started == [] and list(svc.runs.iterdir()) == []


@pytest.mark.parametrize("words, want", [
    (PACED, {"paced": ["2:8000:40:2", "4:16000:40:2"], "port": 7951, "challenger_minutes": 120}),
    (SESSIONS, {"paced": None, "port": 7958, "sessions": 3, "windows": 7, "window_ms": 1000.0,
                "samples": 2773, "challenger_minutes": 30, "jitter_ms": 2.5, "quantile": "p99", "lifetime": 4}),
])
def test_a_runs_command_line_is_one_verifier_py_reads_as_its_request_with_the_hosts_paths_and_capacities(
        svc, monkeypatch, words, want):
    rc, out = call(svc, words)
    assert rc == 0, out
    a, listened = parsed(monkeypatch, json.loads(out)["argv"])
    assert listened == ("127.0.0.1", want["port"])
    assert {k: getattr(a, k) for k in want} == want
    assert (a.model, a.checkpoint, a.pin, a.arena_blocks) == (None, None, None, None)
    assert (a.out, a.registry, a.transcript) == (svc.runs / RUN / "out", svc.registry, svc.scratch / RUN / "transcripts")
    assert (a.challenger, a.challenger_ssh, a.challenger_peer, a.challenger_pin, a.challenger_period,
            a.challenger_main) == ("gate", SSH, "auditor-host", KEY, "memory-line-1", svc.main)
    assert (a.host_bytes, a.ingress_bytes_per_s, a.hbm_bytes, a.lean_exe, a.host) == (
        1843153879040, 4.4e9, V.HBM_BYTES, None, "127.0.0.1")
    for spec in a.paced or []:
        V.pace_of(spec, 17.0)


@pytest.fixture
def tree(svc, tmp_path) -> Path:
    """A pinned tree carrying core's pin (`models.json`) and ROLE's directory in the host's models dir, which is all
    the gate checks of a bound run (`bound_paths`): the store's binding to the pin is verifier.py's, tested on a
    synthetic checkpoint beside the remote session (`test_live_verifier.py`)."""
    (svc.root / "models" / ROLE).mkdir(parents=True)
    t = tmp_path / "tree"
    (t / H.MODEL_PINS).parent.mkdir(parents=True)
    (t / H.MODEL_PINS).write_text(json.dumps({ROLE: {"checkpoint_sha512": "0" * 128}}) + "\n")
    return t


def test_a_bound_run_is_refused_before_anything_starts_without_its_roles_checkpoint_or_cores_pin(svc, tree, tmp_path):
    """The role's directory must be on the host, in the models dir `verifier.json` names (`models`, else
    /var/lib/vy-verifier/models), and the pinned tree must carry core's pin; else nothing is written or started."""
    units = Units()
    lacking = [*PACED, "--model", "QWEN25_32B_BF16", "--arena-blocks", "7"]
    assert refused(call(svc, lacking, units, tree=tree)) == (
        f"--model QWEN25_32B_BF16: this host has no {svc.root / 'models' / 'QWEN25_32B_BF16'}, the role's checkpoint "
        "(infra provisions it, root-held regular files)")
    pins = (tree / H.MODEL_PINS).read_text()
    (tree / H.MODEL_PINS).unlink()
    assert refused(call(svc, BOUND, units, tree=tree)) == (
        f"--model {ROLE}: the pinned tree has no {H.MODEL_PINS}, core's pin of each role, which the verifier binds the "
        "store to")
    (tree / H.MODEL_PINS).write_text(pins)
    settings = json.loads(svc.config.read_text())
    svc.config.write_text(json.dumps({**settings, "models": str(tmp_path / "models")}))
    assert f"this host has no {tmp_path / 'models' / ROLE}," in refused(call(svc, BOUND, units, tree=tree))
    assert units.started == [] and list(svc.runs.iterdir()) == []
    (svc.root / "models").rename(tmp_path / "models")
    rc, out = call(svc, BOUND, units, tree=tree)
    argv = json.loads(out)["argv"]
    assert rc == 0 and argv[argv.index("--checkpoint") + 1] == str(tmp_path / "models" / ROLE)


def test_the_runs_ports_are_the_tunnels_and_not_the_challenger_gates():
    conf = (PACKAGING / "12-vy-verifier.conf").read_text()
    prover = conf.split("Match User prover")[1].split("Match User")[0]
    opened = re.search(r"^\s*PermitOpen (.*)$", prover, re.M).group(1).split()
    assert opened == [f"127.0.0.1:{p}" for p in H.PORTS]
    challenger = conf.split("Match User challenger")[1]
    listened = re.search(r"^\s*PermitListen (.*)$", challenger, re.M).group(1).split()
    assert listened == [f"127.0.0.1:{p}" for p in challengers.GATE_PORTS] and H.GATE_PORTS == challengers.GATE_PORTS
    assert not set(H.PORTS) & set(H.GATE_PORTS)
    tunnel = (PACKAGING / "vy-verifier-tunnel").read_text()
    pattern = re.search(r"\$4 ~ /\^(.*)\$/", tunnel).group(1).replace("\\.", ".")
    assert [p for p in range(7000, 8000) if re.fullmatch(pattern, f"127.0.0.1:{p}")] == list(H.PORTS)


@pytest.mark.parametrize("words, why", [
    # what only the host names: code, a path, a host, a key, the challenger, the node's capacities
    *[([*PACED, flag, value], f"unrecognized arguments: {flag}") for flag, value in [
        ("--host-bytes", "1"), ("--ingress-bytes-per-s", "1"), ("--hbm-bytes", "1"), ("--disk-bytes-per-s", "1"),
        ("--link-bytes-per-s", "1"), ("--challenger", "serve"), ("--challenger-ssh", "research@198.51.100.1"),
        ("--challenger-peer", "gpu-node"), ("--challenger-key", "/tmp/k"), ("--challenger-params", "4,4"),
        ("--challenger-main", "/tmp/main"),
        ("--out", "/tmp/out"), ("--registry", "/tmp/r.json"), ("--transcript", "/tmp/t"), ("--lean-exe", "/bin/sh"),
        ("--host", "0.0.0.0"), ("--check-workers", "1"), ("--delta-log2", "-1"), ("--pings", "1"),
        ("--appends", "9"), ("--memory-of", "/tmp/v.json"), ("--close", "ab"), ("--disk-burst-bytes", "1"),
        ("--checkpoint", "/tmp/m"), ("--layout-workers", "1")]],
    ([*PACED, "-h"], "unrecognized arguments: -h"),
    ([*PACED, "--help"], "unrecognized arguments: --help"),
    ([*BASE, "--sessions", "1", "extra"], "unrecognized arguments: extra"),
    ([*PACED, "extra"], "--paced: 1 to 16 specs"),
    ([*BASE, "--sess", "2"], "unrecognized arguments: --sess"),
    ([*BASE, "--sessions", "2", "--window", "1"], "unrecognized arguments: --window"),
    (PACED[:1] + PACED[3:], "required: --peer"),
    # the peer, the port, the period and the pin
    (["run", "--peer", "cpu-node", *PACED[3:]], "--peer 'cpu-node'"),
    (["run", "--peer", "../etc", *PACED[3:]], "--peer '../etc'"),
    *[([*PACED[:3], "--port", port, *PACED[5:]], f"--port '{port}'")
      for port in ("7811", "7899", "7950", "7959", "07951", "7951.0", "0x1f0f", "22")],
    *[([*PACED[:5], "--period", period, *PACED[7:]], "--period")
      for period in ("$(id)", "p;id", "a" * 65, "", "-x.")],
    ([*PACED[:7], "--pin", KEY.upper(), *PACED[9:]], "--pin"),
    ([*PACED[:7], "--pin", KEY[:-1], *PACED[9:]], "--pin"),
    # the paced specs and the unpaced session's parameters
    *[([*BASE, "--paced", spec], "--paced: 1 to 16 specs")
      for spec in ("2:8000:40", "0:8000:40:2", "2:8000:40:2:1", "2:-1:40:2", "2:1e3:40:2", "1000:1:1:1",
                   "2:8000:0:2", "2:8000:40:0", "2:8000.0001:40:2", "2: 8000:40:2")],
    ([*BASE, "--paced", *["1:4:40:2"] * 17], "--paced: 1 to 16 specs"),
    ([*PACED, "--sessions", "2"], "--paced replaces"),
    ([*PACED, "--window-ms", "1000"], "--paced replaces"),
    (BASE, "--paced SPEC ... or --sessions N"),
    ([*BASE, "--sessions", "2", "--pace-pass-ms", "17"], "--pace-pass-ms is a paced run's"),
    *[([*BASE, "--sessions", n], f"--sessions '{n}'") for n in ("0", "17", "1.5", "0x10", "+1", "01", "-1")],
    ([*BASE, "--sessions", "1", "--windows", "65"], "--windows '65'"),
    ([*BASE, "--sessions", "1", "--samples", "100001"], "--samples '100001'"),
    ([*PACED, "--lifetime", "0"], "--lifetime '0'"),
    ([*PACED, "--minutes", "721"], "--minutes '721'"),
    *[([*BASE, "--sessions", "1", "--window-ms", ms], f"--window-ms '{ms}'")
      for ms in ("1e3", "nan", "inf", "-1", "0.0001", "600001", "0")],
    ([*PACED, "--jitter-ms", "1001"], "--jitter-ms '1001'"),
    ([*PACED, "--pace-pass-ms", "1e1"], "--pace-pass-ms '1e1'"),
    ([*PACED, "--verdict", "lean"], "unrecognized arguments: --verdict"),
    ([*PACED, "--quantile", "p90"], "--quantile 'p90'"),
    # a bound run's role and its store's shape; its checkpoint and pin are the host's
    *[([*PACED, "--model", role, "--arena-blocks", "7"], f"--model {role!r}: a role of core's pin")
      for role in ("qwen25_32b", "A/B", "..", "", "A" * 65, "A-B", "A B", "QWEN.BF16")],
    ([*PACED, "--model", ROLE], "--model needs --arena-blocks A"),
    ([*PACED, "--model", ROLE, "--engines", "8"], "--model needs --arena-blocks A"),
    *[([*PACED, flag, "7"], f"{flag} shapes a bound run's store: it comes with --model ROLE")
      for flag in ("--engines", "--arena-blocks")],
    *[([*PACED, "--model", ROLE, "--engines", n, "--arena-blocks", "7"], f"--engines '{n}': an integer, 1 to 8")
      for n in ("0", "9", "01", "1.0", "+1", "-1", "0x2")],
    *[([*PACED, "--model", ROLE, "--arena-blocks", n], f"--arena-blocks '{n}': an integer, 1 to 33554432")
      for n in ("0", "33554433", "999999999", "1e3", "07", "-1", "7.0")],
    ([*PACED, "--model", ROLE, "--arena-blocks", "7", "--pin", "/tmp/models.json"], "--pin"),
])
def test_a_run_is_refused_unless_every_parameter_is_one_the_whitelist_holds(svc, words, why):
    units = Units()
    assert why in refused(call(svc, words, units))
    assert units.started == [] and list(svc.runs.iterdir()) == []


@pytest.mark.parametrize("argv, why", [
    (["--ssh", "status", "--key", "root"], "gate key name 'root'"),
    (["--ssh", "status", "--key", "auditor"], "gate key name 'auditor'"),
    (["--ssh", "status", "--key", "prover"], "gate key name 'prover'"),
    (["--ssh", "status", "--key", "a b"], "gate key name 'a b'"),
    (["--ssh", "status", "--key", "../x"], "gate key name '../x'"),
    (["--ssh", "status", "--lane", LANE], "--ssh WORDS --key NAME"),
    (["--ssh"], "--ssh WORDS --key NAME"),
    (["--ssh", "status", "--key", LANE, "--key", "-"], "--ssh WORDS --key NAME"),
    (["--ssh", "status " + "x" * 8190, "--key", LANE], "over 8192 bytes"),
    (["--ssh", "run 'unterminated", "--key", LANE], "doesn't parse as shell words"),
])
def test_the_gate_takes_a_request_only_in_its_forced_commands_form(svc, argv, why):
    assert why in refused(gate(svc, argv))


@pytest.mark.parametrize("words", [
    [], ["sh"], ["bash", "-c", "id"], ["ledger"], ["pubkey"], ["publish", RUN], ["execute", RUN], ["Run", *PACED[1:]],
    ["status", RUN, "--all"], ["gate", "status"],
])
def test_the_gate_refuses_any_other_verb(svc, words):
    assert refused(call(svc, words)).startswith("verbs: run --peer NAME")


@pytest.mark.parametrize("words", [PACED, ["replay", RUN], ["stop", RUN], ["close", "ab"]])
def test_a_read_only_key_starts_replays_stops_and_closes_nothing_but_reads(svc, words):
    units = Units()
    assert call(svc, PACED, units)[0] == 0
    assert "this key reads only" in refused(call(svc, words, units, key="-"))
    assert units.started == [f"vy-verifier@{RUN}.service"] and units.stopped == units.closed == []
    rc, out = call(svc, ["status"], units, key="-")
    assert rc == 0 and json.loads(out) == {"pinned": PIN, "busy": [f"vy-verifier@{RUN}.service"],
                                           "pins": [{"line": 0, "commit": PIN, "utc": "2026-10-21T13:00:00Z"}],
                                           "runs": [{"run": RUN, "by": LANE, "rc": None, "active": True}]}


def test_a_run_is_stopped_only_by_its_own_lane_or_vy_admin_and_its_stop_is_recorded(svc):
    units = Units()
    assert call(svc, PACED, units)[0] == 0
    assert refused(call(svc, ["stop", RUN], units, key="network-accounting")) == (
        f"run {RUN} is {LANE}'s, not network-accounting's")
    rc, out = call(svc, ["stop", RUN], units)
    assert rc == 0 and json.loads(out) == {"run": RUN, "stopping": [f"vy-verifier@{RUN}.service"], "by": LANE}
    assert json.loads((svc.runs / RUN / "stop.json").read_text()) == {"by": LANE, "utc": "2026-10-21T13:00:00Z"}
    rc, out = gate(svc, ["stop", RUN], units, by="vy-admin")
    assert rc == 0 and json.loads(out)["by"] == "vy-admin" and units.stopped == [f"vy-verifier@{RUN}.service"] * 2
    units.live.clear()
    assert refused(call(svc, ["stop", RUN], units)) == f"run {RUN} has no active unit"
    for run in ("v20261021T130000Z-ffff", "../runs", "x"):
        assert refused(call(svc, ["stop", run], units)).startswith("no run")


def test_a_replay_and_a_close_wait_for_no_unit_and_a_replay_for_its_runs_end(svc):
    units = Units()
    assert call(svc, PACED, units)[0] == 0
    assert "one unit at a time" in refused(call(svc, ["replay", RUN], units))
    assert "no run is live" in refused(call(svc, ["close", "ab" * 8], units))
    units.live.clear()
    assert refused(call(svc, ["replay", RUN], units)) == f"run {RUN} has no end: it replays once it has ended"
    (svc.runs / RUN / "end.json").write_text(json.dumps({"rc": 0}))
    rc, out = call(svc, ["replay", RUN], units)
    assert rc == 0 and json.loads(out) == {"run": RUN, "replay": "started", "by": LANE}
    assert units.started[-1] == f"vy-verifier-replay@{RUN}.service"
    assert "one unit at a time" in refused(call(svc, ["replay", RUN], units))
    units.live.clear()
    for node in ("AB", "abc", "zz", "ab cd", ""):
        assert refused(call(svc, ["close", node], units)).startswith("close NODE_HEX")
    rc, out = call(svc, ["close", "ab" * 8], units)
    assert rc == 0 and units.closed == ["ab" * 8]
    (line,) = (svc.root / "closes.jsonl").read_text().splitlines()
    assert json.loads(line) == {"node": "ab" * 8, "by": LANE, "utc": "2026-10-21T13:00:00Z",
                                "closed": {"node": "ab" * 8, "closed": True}}


def test_a_run_whose_pin_moved_before_its_unit_started_does_not_run(svc):
    assert call(svc, PACED)[0] == 0
    end = H.execute(svc, RUN, pinned="d" * 40, now=lambda: T1,
                    argv=[sys.executable, "-c", "raise SystemExit('the verifier ran')"])
    assert end["rc"] == 2 and "the pin moved, so the run doesn't run" in end["why"]
    assert not (svc.runs / RUN / "out" / "verifier.log").exists() and end["transcripts"] == []
    with pytest.raises(H.Refused, match="has ended"):
        H.execute(svc, RUN, pinned=PIN, now=lambda: T1)


def test_each_pin_is_a_line_of_the_sealed_log_and_a_run_starts_only_from_its_last(svc, tmp_path):
    """pin.sh takes any commit it is handed, so the host keeps the record. Each pin is a line of a log that is
    sealed append-only before its first line, and a run starts only from the log's last pin and names that line."""
    log, sealed = tmp_path / "log" / "pins.jsonl", []
    log.parent.mkdir()
    seal = lambda p: sealed.append(p.read_text())  # noqa: E731
    assert H.record_pin(log, PIN, T0, seal=seal) == {"commit": PIN, "utc": "2026-10-21T13:00:00Z"}
    assert sealed == [""] and log.stat().st_mode & 0o777 == 0o644
    H.record_pin(log, "d" * 40, T1, seal=seal)
    assert [json.loads(ln) for ln in log.read_text().splitlines()] == [
        {"commit": PIN, "utc": "2026-10-21T13:00:00Z"}, {"commit": "d" * 40, "utc": "2026-10-21T13:30:00Z"}]
    assert len(sealed) == 2
    for bad in ("C" * 40, "c" * 39, "c" * 41, "", "../x"):
        with pytest.raises(H.Refused, match="40 lowercase hex digits"):
            H.record_pin(log, bad, T1, seal=seal)
    assert len(log.read_text().splitlines()) == 2 and len(sealed) == 2

    units = Units()
    H.record_pin(svc.pins, "d" * 40, T1, seal=lambda p: None)
    assert refused(call(svc, PACED, units)) == (
        f"the pin log's last pin is {'d' * 40}, not this tree's {PIN}: a run starts only from the tree pin.sh last "
        "pinned")
    unlogged = H.Host(root=svc.root, config=svc.config, pins=tmp_path / "absent.jsonl")
    assert "records no pin" in refused(call(unlogged, PACED, units))
    assert units.started == [] and list(svc.runs.iterdir()) == []
    H.record_pin(svc.pins, PIN, T1, seal=lambda p: None)
    rc, out = call(svc, PACED, units, now=T1)
    assert rc == 0 and json.loads(out)["pin_log"] == {"line": 2, "utc": "2026-10-21T13:30:00Z"}
    run = json.loads(out)["run"]
    assert json.loads((svc.runs / run / "request.json").read_text())["pin_log"]["line"] == 2
    rc, out = call(svc, ["status"], units, key="-")
    assert [(p["line"], p["commit"]) for p in json.loads(out)["pins"]] == [(0, PIN), (1, "d" * 40), (2, PIN)]


def test_pin_sh_runs_no_unpinned_code_and_logs_each_pin_before_current_names_it():
    """Infra's review: elan is a release, its archive checked against a pinned sha256 before it runs (never a
    script fetched from a branch), and the pin is logged (`host.py pinned`) before `current` moves to it."""
    code = "\n".join(ln for ln in (PACKAGING / "pin.sh").read_text().splitlines() if not ln.lstrip().startswith("#"))
    assert "raw.githubusercontent.com" not in code and not re.search(r"\|\s*(ba)?sh\b", code)
    assert re.search(r"^ELAN=v\d+\.\d+\.\d+ ELAN_SHA256=[0-9a-f]{64}$", code, re.M)
    assert "https://github.com/leanprover/elan/releases/download/$ELAN/" in code
    assert code.index('echo "$ELAN_SHA256  $e/elan.tar.gz" | sha256sum -c') < code.index('"$e/elan-init"')
    assert code.index('host.py" pinned "$c"') < code.index('ln -sfn "$c" "$O/current.tmp"')


def test_fetch_reads_a_run_and_nothing_else(svc):
    for words in (["fetch"], ["fetch", RUN, "--all"], ["fetch", RUN, "--transcripts", "x"]):
        assert refused(call(svc, words, key="-")) == "fetch RUN [--transcripts]"
    for run in ("../../etc", "v20261021T130000Z-ab12/../..", RUN):
        assert refused(call(svc, ["fetch", run], key="-")).startswith("no run")


def test_publish_puts_the_runs_directory_with_its_request_as_meta(svc):
    assert call(svc, PACED)[0] == 0
    (svc.runs / RUN / "end.json").write_text(json.dumps({"rc": 0, "ended_utc": "2026-10-21T13:30:00Z"}))
    got = []
    assert H.publish(svc, RUN, put=lambda argv: got.append(argv) or 0) == 0
    (argv,) = got
    assert argv[:2] == ["put", "--kind"] and argv[2] == "verifier-run/v1" and argv[-3:] == [
        "--tree", str(svc.runs / RUN), "--preserve"]
    assert json.loads(argv[argv.index("--meta") + 1]) == {
        "run": RUN, "by": LANE, "pinned": PIN, "peer": "gpu-node", "period": "memory-line-1",
        "requested_utc": "2026-10-21T13:00:00Z", "end": {"rc": 0, "ended_utc": "2026-10-21T13:30:00Z"}, "replay": None}


@pytest.mark.parametrize("code", [
    "print(host.replay_module().__file__)",
    "sys.path.insert(0, str(host.RESEARCH)); from research.store import cli; print(cli.__file__)",
    "sys.argv = ['verifier.py', '--help']; runpy.run_path(str(host.VERIFIER), run_name='__main__')",
])
def test_the_hosts_code_runs_on_a_python_with_no_site_packages(code, tmp_path):
    """The host runs its units with its own python3 (`-I`), where neither `verity` nor any other package is installed:
    each import the run, the replay and the publish make resolves within the pinned tree."""
    p = subprocess.run([sys.executable, "-I", "-S", "-c", f"import runpy, sys; sys.path.insert(0, {str(HERE.parent)!r}); "
                        f"import host; {code}"], capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "site-packages" not in p.stdout


def party(svc, words, units: Units | None = None, *, key=LANE, out=b'{"listening": 47611}\n{"signed": "x"}\n', rc=0):
    """`words` sent by `key`; the party's run is recorded, and prints `out` and exits `rc`."""
    ran: list[list[str]] = []

    def run(argv, sink, log, minutes):
        ran.append([*argv, f"minutes={minutes}"])
        sink.write(out)
        log.write_bytes(out)
        return rc
    got = call(svc, words, units, key=key, party=run)
    return got, ran


def test_a_party_runs_its_module_from_the_pinned_tree_as_auditor_with_its_own_device_and_port(svc):
    (rc, out), ran = party(svc, ["wipe_party", "--reads", "1024", "--native", "--threads", "16"])
    assert rc == 0 and out == b'{"listening": 47611}\n{"signed": "x"}\n'
    assert ran == [["runuser", "-u", "auditor", "--", "/usr/bin/env", "-i", "PYTHONDONTWRITEBYTECODE=1",
                    "HOME=/var/lib/vy-verifier", "PATH=/usr/bin:/bin", "/usr/bin/python3", "-s", "-E",
                    "-m", "verity.protocols.pous.verifier.wipe_party", "--device", str(svc.parties / "wipe"),
                    "--listen", "0.0.0.0:47611", "--reads", "1024", "--streamer", "build", "--threads", "16", "minutes=60"]]
    reqs = [json.loads(ln) for ln in (svc.parties / "wipe" / "requests.jsonl").read_text().splitlines()]
    assert [(r["verb"], r["by"], r.get("rc")) for r in reqs] == [("wipe_party", LANE, None), ("wipe_party", LANE, 0)]
    assert (svc.parties / "wipe" / "log" / reqs[0]["log"]).read_bytes() == out
    (rc, _), ran = party(svc, ["challenger_party", "--minutes", "5"])
    assert rc == 0 and ran[0][-7:] == ["-m", H.PARTIES["challenger_party"][1], "--device",
                                       str(svc.parties / "pous"), "--listen", "0.0.0.0:47610", "minutes=5"]


def test_bw_party_holds_the_store_the_block_challenger_kept_for_a_pous_session_or_the_named_fixture_s(svc):
    """The bandwidth party's registration is the host's: the block challenger's own record of the PoUS session the lane
    names (its device on this host, by session id), or with `--fixture` the fixture's registration in the pinned tree;
    never a path. A party that names neither is refused before it starts: V3's rerun 58a7 reached the gate through a key
    whose forced command dropped `--pous-session`, so the party held the fixture's store and refused the session's node."""
    sid = "0f" * 32
    (rc, _), ran = party(svc, ["bw_party", "--pous-session", sid, "--minutes", "8"])
    assert rc == 0 and ran[0][-11:] == ["-m", H.PARTIES["bw_party"][1], "--device", str(svc.parties / "bw"),
                                        "--listen", "0.0.0.0:47612", "--pous-record", str(svc.parties / "pous"),
                                        "--pous-session", sid, "minutes=8"]
    (rc, _), ran = party(svc, ["bw_party", "--fixture"])
    assert rc == 0 and ran[0][-9:] == ["-m", H.PARTIES["bw_party"][1], "--device", str(svc.parties / "bw"),
                                       "--listen", "0.0.0.0:47612", *H.PARTY_FIXED["bw_party"], "minutes=60"]
    assert (H.TREE / H.PARTY_FIXED["bw_party"][1]).is_file()
    for words in (["bw_party"], ["bw_party", "--minutes", "10"]):
        got, ran = party(svc, words)
        assert "bw_party names no store" in refused(got) and ran == []


@pytest.mark.parametrize("words, why", [
    (["wipe_party", "--egress-bits", "8"], "got '--egress-bits'"),
    (["wipe_party", "--device", "/tmp/x"], "got '--device'"),
    (["wipe_party", "--listen", "0.0.0.0:1"], "got '--listen'"),
    (["wipe_party", "--reads", "0"], "--reads takes an integer"),
    (["wipe_party", "--reads", "2152", "--reads", "8"], "got '--reads'"),
    (["wipe_party", "--threads", "8"], "give --native too"),
    (["wipe_party", "--streamer", "/tmp/s"], "got '--streamer'"),
    (["challenger_party", "--reads", "8"], "challenger_party takes no flags"),
    (["challenger_party", "--minutes", "181"], "--minutes takes one integer"),
    (["wipe_party", "--minutes"], "--minutes takes one integer"),
    (["bw_party", "--registration", "/tmp/r.json"], "bw_party takes --pous-session SID | --fixture; got '--registration'"),
    (["bw_party", "--pous-record", "/tmp"], "got '--pous-record'"),
    (["bw_party", "--pous-session", "../" + "a" * 61], "64 lowercase hex digits"),
    (["bw_party", "--pous-session", "A" * 64], "64 lowercase hex digits"),
    (["bw_party", "--pous-session", "a" * 64, "--pous-session", "b" * 64], "bw_party holds one store"),
    (["bw_party", "--fixture", "--pous-session", "a" * 64], "bw_party holds one store"),
    (["bw_party", "--pous-session", "a" * 64, "--fixture"], "bw_party holds one store"),
])
def test_a_party_is_refused_any_flag_its_whitelist_lacks_before_it_starts(svc, words, why):
    got, ran = party(svc, words)
    assert why in refused(got) and ran == []


def test_a_read_only_key_starts_no_party_and_a_party_waits_for_every_unit_and_its_own_last_session(svc):
    for verb in ("wipe_party", "challenger_party"):
        got, ran = party(svc, [verb], key="-")
        assert "this key reads only" in refused(got) and ran == []
    units = Units()
    assert call(svc, PACED, units)[0] == 0
    got, ran = party(svc, ["wipe_party"], units)
    assert "a party session waits for no unit" in refused(got) and ran == []
    svc.parties.mkdir(exist_ok=True)
    with open(svc.parties / "wipe.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        got, ran = party(svc, ["wipe_party"])
        assert "serving another session" in refused(got) and ran == []
        assert party(svc, ["challenger_party"])[0][0] == 0


def test_party_key_prints_only_the_public_half_made_on_the_host(svc):
    asked: list[list[str]] = []

    def keygen(argv):
        asked.append(argv)
        return "ab" * 32
    rc, out = call(svc, ["party_key", "pous"], key="-", keygen=keygen)
    assert rc == 0 and json.loads(out) == {"party": "pous", "verb": "challenger_party", "public_key": "ab" * 32}
    assert asked[0][-3:] == ["-c", H.KEYGEN, str(svc.parties / "pous")] and asked[0][:3] == ["runuser", "-u", "auditor"]
    assert "party_key" in refused(call(svc, ["party_key", "/etc"], key="-", keygen=keygen))
    assert "not a public key" in refused(call(svc, ["party_key", "wipe"], key="-", keygen=lambda a: "Traceback"))


def test_the_wipe_party_and_its_key_import_on_a_python_with_no_site_packages():
    """The parties run on the host's python3 (`-s -E`), where no package is installed: the modules resolve in the tree."""
    code = ("import verity.protocols.pous.verifier.wipe_party, verity.protocols.pous.verifier.period, "
            "verity.protocols.pous.device.challenger_party, verity.protocols.poub.verifier.p2_party; "
            "print('ok')")
    p = subprocess.run([sys.executable, "-S", "-s", "-E", "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0 and p.stdout.strip() == "ok", p.stderr[-2000:]


def test_a_party_is_killed_when_its_minutes_run_out_and_its_log_says_so(tmp_path, monkeypatch):
    """`exec_party` with a party that never exits (no prover joined): its timer kills it and its parent, and the log records
    why. Killing the parent alone (runuser, on an auditor host) left the party listening and holding the pipe."""
    armed: list[float] = []

    class Timer:
        """Runs out the moment it starts: the party is killed whatever it was doing."""

        def __init__(self, seconds, fn):
            armed.append(seconds)
            self.fn = fn

        def start(self):
            self.fn()

        def cancel(self):
            pass
    monkeypatch.setattr(H.threading, "Timer", Timer)
    monkeypatch.setattr(H, "TREE", tmp_path)
    sink, log = io.BytesIO(), tmp_path / "log.jsonl"
    child = "import subprocess, sys; subprocess.run([sys.executable, '-c', 'import sys; sys.stdin.read()'])"
    rc = H.exec_party([sys.executable, "-c", child], sink, log, 2)     # a party under a parent, as under runuser
    assert armed == [120] and rc == -9
    assert json.loads(log.read_bytes().splitlines()[-1]) == {"killed": "the session's 2 min ran out"}


def test_party_log_reads_the_sessions_and_each_ones_stdout_as_the_host_kept_it_for_any_key(svc):
    (rc, _), _ = party(svc, ["wipe_party", "--reads", "64"], out=b'{"listening": 47611}\n{"window": {"t0": "a", "t1": "b"}}\n')
    rc, out = call(svc, ["party_log", "wipe"], key="-")
    sessions = json.loads(out)["sessions"]
    assert rc == 0 and [s.get("rc") for s in sessions] == [None, 0]
    rc, out = call(svc, ["party_log", "wipe", sessions[0]["log"]], key="-")
    assert rc == 0 and json.loads(out)["lines"] == [{"listening": 47611}, {"window": {"t0": "a", "t1": "b"}}]
    assert "party_log" in refused(call(svc, ["party_log", "wipe", "../../etc/passwd"], key="-"))
    assert "no session log" in refused(call(svc, ["party_log", "pous", "20261026T000000Z.jsonl"], key="-"))


def test_bw_party_runs_with_the_registration_the_host_names_and_no_lane_flag(svc):
    (rc, _), ran = party(svc, ["bw_party", "--fixture", "--minutes", "10"])
    assert rc == 0 and ran[0][-9:] == ["-m", "verity.protocols.poub.verifier.p2_party", "--device", str(svc.parties / "bw"),
                                       "--listen", "0.0.0.0:47612", "--registration",
                                       "verity/protocols/poub/verifier/bandwidth-mem69-v2.json", "minutes=10"]
    assert "bw_party takes --pous-session SID | --fixture" in refused(party(svc, ["bw_party", "--registration", "/tmp/x"])[0])


# --- the compose gate ---------------------------------------------------------------------------------------------------------

HANDED = {"sampled-proofs-served/window/case.json": b'{"blobs": []}', "private/compose/compose.env": b"K=7\n",
          "h1/log.jsonl": b"{}\n"}


@pytest.fixture
def composer(tmp_path) -> H.Host:
    host = H.Host(root=tmp_path / "compose", config=tmp_path / "compose.json", pins=tmp_path / "compose-pins.jsonl",
                  main=tmp_path / "compose-main.txt", compose=True)
    H.record_pin(host.pins, PIN, T0, on="remote", seal=lambda p: None)
    return host


def raw_tar(members: list[tuple[tarfile.TarInfo | str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for ti, body in members:
            ti = tarfile.TarInfo(ti) if isinstance(ti, str) else ti
            ti.size = len(body)
            t.addfile(ti, io.BytesIO(body))
    return buf.getvalue()


def tar_of(files: dict[str, bytes], *, sums: dict[str, str] | None = None, unlisted: tuple[str, ...] = (),
           more: str = "", extra: tuple[tuple[tarfile.TarInfo, bytes], ...] = ()) -> bytes:
    """A hand-off: `files` and the sums file over them (``sums`` replacing a line, ``unlisted`` left out of it, ``more``
    added), then ``extra``'s members."""
    lines = "".join(f"{(sums or {}).get(f) or hashlib.sha256(b).hexdigest()}  {f}\n" for f, b in files.items()
                    if f not in unlisted) + more
    return raw_tar([*files.items(), (H.COMPOSE_SUMS, lines.encode()), *extra])


def compose(host: H.Host, data: bytes, *, words=("compose",), key="vllm", units: Units | None = None,
            rc: int = 0) -> tuple[tuple[int, bytes], dict]:
    """`words` through the gate on `host` with `data` as the hand-off, its compose `compose_handoff` in-process (no
    runuser) whose run writes a verdict line to $OUT and exits ``rc``: the gate's answer, and what the run saw."""
    seen: dict = {}

    def run(env, log):
        d = Path(env["D"])
        seen.update(env=env, modes={str(p.relative_to(d)): oct(p.stat().st_mode & 0o777) for p in d.rglob("*")
                                    if p.is_file()})
        (Path(env["OUT"]) / "record").mkdir(parents=True)
        (Path(env["OUT"]) / "verdicts.txt").write_text('RECORD {"composed": "accept"}\n')
        return rc

    def exec_(argv, scratch):
        seen.update(argv=argv, scratch=scratch, existed=scratch.is_dir())
        return H.compose_handoff(host, scratch, io.BytesIO(data), pinned=argv[-2], on=argv[-1], run=run)

    return call(host, list(words), units, key=key, compose=exec_), seen


def test_the_compose_gate_composes_a_hand_off_in_a_scratch_of_its_own_and_answers_its_out(composer):
    """`compose`: as `auditor`, the hand-off received into a scratch of its own (0600 files), composed there at the pin
    (its tree's commit and the branch infra pinned it on, every path under the scratch, the tree built at its pin), its
    $OUT answered, the scratch removed and the compose logged with the sums it was held to."""
    (rc, out), seen = compose(composer, tar_of(HANDED))
    doc = json.loads(out)
    scratch = composer.scratch / "compose-20261021T130000Z-ab12"
    assert rc == 0, doc
    assert seen["argv"][:3] == ["runuser", "-u", "auditor"] and "/usr/bin/env" in seen["argv"]
    assert seen["argv"][-6:] == ["-I", str(Path(H.__file__).resolve()), "compose", str(scratch), PIN, "remote"]
    assert seen["existed"] and not scratch.exists() and list(composer.scratch.iterdir()) == []
    assert set(seen["modes"].values()) == {"0o600"} and sorted(seen["modes"]) == sorted([*HANDED, H.COMPOSE_SUMS])
    env = seen["env"]
    assert (env["D"], env["OUT"], env["HOME"]) == (str(scratch / "serve"), str(scratch / "out"), str(composer.root))
    assert (env["RESEARCH_SOURCE_SHA"], env["COMPOSE_TREE_ON"], env["FOLLOW_ANY_TREE"], env["COMPOSE_BUILT"]) == (
        PIN, "remote", "1", "1")
    sums = hashlib.sha512("".join(f"{hashlib.sha256(b).hexdigest()}  {f}\n" for f, b in HANDED.items()).encode()).hexdigest()
    assert doc == {"by": "vllm", "pinned": PIN, "pin": {"line": 0, "utc": "2026-10-21T13:00:00Z", "on": "remote"},
                   "utc": "2026-10-21T13:00:00Z",
                   "compose": {"rc": 0, "pinned": PIN, "on": "remote", "received": 4, "sums_sha512": sums,
                               "files": {"verdicts.txt": 'RECORD {"composed": "accept"}\n'}}}
    logged = [json.loads(x) for x in (composer.root / "composes.jsonl").read_text().splitlines()]
    assert logged == [{"by": "vllm", "utc": "2026-10-21T13:00:00Z", "pinned": PIN, "pin": doc["pin"], "rc": 0,
                       "sums_sha512": sums, "refused": None}]
    (rc, out), _ = compose(composer, tar_of(HANDED), rc=1)
    assert rc == 1 and json.loads(out)["compose"]["rc"] == 1


def test_the_compose_gate_serves_compose_alone_and_the_verifier_gate_never_does(svc, composer):
    """The compose gate and the verifier's share no verb: each is its own tree's service (`host_of`), the compose gate
    refuses every other verb, the verifier's refuses compose, a read-only key composes nothing and compose takes no
    words (its hand-off is its stdin)."""
    assert H.host_of(Path("/opt/vy-compose") / PIN) == H.COMPOSE_HOST and H.COMPOSE_HOST.compose
    assert H.host_of(Path("/opt/vy-verifier") / PIN) == H.Host() and not H.Host().compose
    for words in (["status"], PACED, ["fetch", RUN], ["party_key", "wipe"], ["wipe_party"]):
        assert "the compose gate serves compose alone" in refused(compose(composer, b"", words=words)[0])
    assert "compose runs on the compose gate" in refused(compose(svc, tar_of(HANDED))[0])
    assert "this key reads only" in refused(compose(composer, tar_of(HANDED), key="-")[0])
    assert "compose takes no words" in refused(compose(composer, tar_of(HANDED), words=("compose", "/tmp/x"))[0])
    assert H.compose_handoff(H.Host(), Path("/nonexistent"), io.BytesIO(tar_of(HANDED)), pinned=PIN, on=None)[0] == 3
    assert not (composer.root / "composes.jsonl").exists()


def test_a_compose_waits_for_no_verifier_unit_runs_one_at_a_time_and_only_from_the_last_pin(composer, tmp_path):
    units = Units()
    units.live.append(f"vy-verifier@{RUN}.service")
    assert "a compose waits for no verifier unit" in refused(compose(composer, tar_of(HANDED), units=units)[0])
    composer.root.mkdir()
    with open(composer.root / "compose.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert refused(compose(composer, tar_of(HANDED))[0]) == "another compose runs; one at a time"
    H.record_pin(composer.pins, "d" * 40, T1, seal=lambda p: None)
    assert "not this tree's" in refused(compose(composer, tar_of(HANDED))[0])
    assert not (composer.root / "composes.jsonl").exists()


def test_a_compose_that_answers_nothing_is_refused_logged_and_its_scratch_removed(composer):
    seen = {}

    def exec_(argv, scratch):
        seen["scratch"] = scratch
        raise H.Refused("the compose answered nothing (Traceback)")

    assert refused(call(composer, ["compose"], key="vllm", compose=exec_)) == "the compose answered nothing (Traceback)"
    assert not seen["scratch"].exists()
    logged = [json.loads(x) for x in (composer.root / "composes.jsonl").read_text().splitlines()]
    assert [(x["rc"], x["refused"]) for x in logged] == [(3, "the compose answered nothing (Traceback)")]


def _member(name: str, kind: bytes = tarfile.REGTYPE, link: str = "") -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.type, ti.linkname = kind, link
    return ti


@pytest.mark.parametrize("data, why", [
    (tar_of(HANDED, extra=((_member("private/h1/key"), b"k"),)), "the hand-off carries the served run's witness: "
     "private/h1/key (H1's prover state (its key))"),
    (tar_of({**HANDED, "requests.jsonl": b"{}"}), "the hand-off carries the served run's witness: requests.jsonl"),
    (tar_of({**HANDED, "sampled-proofs-served/window/m0/inst-0.bin": b"x"}), "witness: "
     "sampled-proofs-served/window/m0/inst-0.bin (a prover's instance file (its rows))"),
    (tar_of(HANDED, extra=((_member("h1/w7", tarfile.SYMTYPE, "/etc/passwd"), b""),)), "h1/w7 isn't a regular file"),
    (tar_of(HANDED, extra=((_member("h1/w8", tarfile.LNKTYPE, "h1/log.jsonl"), b""),)), "h1/w8 isn't a regular file"),
    (tar_of(HANDED, extra=((_member("../x"), b"x"),)), "the hand-off's member '../x' isn't a path under the serve dir"),
    (tar_of(HANDED, extra=((_member("/etc/x"), b"x"),)), "the hand-off's member '/etc/x' isn't a path under the serve dir"),
    (tar_of(HANDED, extra=((_member("h1/../../x"), b"x"),)), "isn't a path under the serve dir"),
    (tar_of(HANDED, extra=((_member("h1/log.jsonl"), b"{}\n"),)), "the hand-off carries h1/log.jsonl twice"),
    (tar_of(HANDED, unlisted=("h1/log.jsonl",)), "the hand-off carries h1/log.jsonl, which its sums don't list"),
    (tar_of(HANDED, sums={"h1/log.jsonl": "0" * 64}), "h1/log.jsonl isn't the file the served node hashed"),
    (tar_of(HANDED, extra=((_member(H.COMPOSE_SUMS), b"0" * 64 + b"  h1/gone.json\n"),)),
     f"the hand-off carries {H.COMPOSE_SUMS} twice"),
    (b"not a tar" * 100, "the hand-off isn't a tar"),
])
def test_a_hand_off_is_refused_before_anything_composes_it_unless_it_is_the_served_nodes_files_alone(
        composer, data, why):
    """`receive` refuses a witness by its name before writing its bytes, any member that isn't a regular file at a
    relative path under the serve dir, a second copy, a file the sums don't list or whose SHA-256 isn't its sum, and
    anything that isn't a tar; nothing composes, the scratch is removed and the refusal is logged."""
    (rc, out), seen = compose(composer, data)
    assert why in refused((rc, out)) and "env" not in seen
    assert not seen["scratch"].exists()
    assert json.loads((composer.root / "composes.jsonl").read_text())["rc"] == 3


def test_receive_writes_no_witness_byte_and_holds_the_sums_to_every_file(tmp_path):
    d = tmp_path / "serve"
    with pytest.raises(H.Refused, match="witness"):
        H.receive(io.BytesIO(tar_of(HANDED, extra=((_member("private/h1/key"), b"secret"),))), d)
    assert not (d / "private" / "h1" / "key").exists()
    with pytest.raises(H.Refused, match="has no private/compose/inputs.sha256"):
        H.receive(io.BytesIO(raw_tar([("h1/log.jsonl", b"x")])), tmp_path / "nosums")
    with pytest.raises(H.Refused, match="lacks h1/gone.json, which its sums list"):
        H.receive(io.BytesIO(tar_of(HANDED, more=f"{'0' * 64}  h1/gone.json\n")), tmp_path / "lacks")
    with pytest.raises(H.Refused, match=r"sums list the served run's witness: private/h1/key \(H1's prover state"):
        H.receive(io.BytesIO(tar_of(HANDED, more=f"{'0' * 64}  private/h1/key\n")), tmp_path / "listed")
    with pytest.raises(H.Refused, match="a line that isn't sha256sum's"):
        H.receive(io.BytesIO(tar_of(HANDED, more="nonsense\n")), tmp_path / "line")
    with pytest.raises(H.Refused, match="over 10 bytes"):
        H.receive(io.BytesIO(tar_of(HANDED)), tmp_path / "cap", cap=10)
    got = H.receive(io.BytesIO(tar_of(HANDED)), tmp_path / "ok")
    assert got == {**{f: hashlib.sha256(b).hexdigest() for f, b in HANDED.items()},
                   H.COMPOSE_SUMS: hashlib.sha256((tmp_path / "ok" / H.COMPOSE_SUMS).read_bytes()).hexdigest()}
    assert oct((tmp_path / "ok").stat().st_mode & 0o777) == "0o700"


def test_the_gate_names_each_witness_and_why():
    """The gate's `witness` names each path of a served run's witness with why, and nothing else; that its rule is the
    sender's (`served_verdicts.py witness`) is held beside the sender (`test_served_verdicts.py`)."""
    assert H.witness(["h1/log.jsonl", "", "windows", "a/values/x", "windows/7/rows", "private/h1/key", "h1/x.key"]) == [
        ("a/values/x", "a committed value file"), ("windows/7/rows", "a sealed row dir (the window's rows)"),
        ("private/h1/key", "H1's prover state (its key)"), ("h1/x.key", "a key")]


def test_the_compose_runs_the_pinned_trees_build_and_answers_its_out_within_its_cap(composer, tmp_path):
    tree = tmp_path / "tree"
    (tree / H.SAMPLED_EXE).parent.mkdir(parents=True)
    assert "SAMPLED_EXE" not in H.compose_env(composer, tmp_path / "s", PIN, None, tree)
    (tree / H.SAMPLED_EXE).write_text("")
    env = H.compose_env(composer, tmp_path / "s", PIN, None, tree)
    assert env["SAMPLED_EXE"] == str(tree / H.SAMPLED_EXE) and "COMPOSE_TREE_ON" not in env
    assert env["GIT_CEILING_DIRECTORIES"] == str(tmp_path) and env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    out = tmp_path / "out"
    (out / "record").mkdir(parents=True)
    (out / "verdicts.txt").write_text("a" * 6)
    (out / "record" / "verdicts.json").write_text("b" * 6)
    assert H.answered(out, cap=8) == {"files": {"record/verdicts.json": "b" * 6}, "not_sent": {"verdicts.txt": 6}}
    assert H.answered(tmp_path / "none") == {"files": {}}


def test_a_pin_line_names_the_branch_its_commit_is_on_where_infra_names_it(tmp_path):
    log = tmp_path / "pins.jsonl"
    assert H.record_pin(log, PIN, T0, on="remote", seal=lambda p: None) == {
        "commit": PIN, "utc": "2026-10-21T13:00:00Z", "on": "remote"}
    with pytest.raises(H.Refused, match="a branch"):
        H.record_pin(log, PIN, T0, on="a b", seal=lambda p: None)
    host = H.Host(root=tmp_path, pins=log)
    assert H.pin_of(host, PIN) == {"line": 0, "utc": "2026-10-21T13:00:00Z", "on": "remote"}
    H.record_pin(log, PIN, T1, seal=lambda p: None)
    assert H.pin_of(host, PIN) == {"line": 1, "utc": "2026-10-21T13:30:00Z"}


def test_a_compose_is_killed_with_everything_it_started_when_its_minutes_run_out(tmp_path, monkeypatch):
    """`run_compose` with a compose that never ends (a child blocked on a FIFO): its timer kills the whole process group
    and the exit is 124; one that ends is its exit."""
    armed: list[float] = []

    class Timer:
        """Runs out the moment it starts."""

        def __init__(self, seconds, fn):
            armed.append(seconds)
            self.fn = fn

        def start(self):
            self.fn()

        def cancel(self):
            pass

    script = tmp_path / "benchmarks" / "one_stage" / "served_verdicts.sh"
    script.parent.mkdir(parents=True)
    script.write_text('mkfifo "$D/f"; (read -r x < "$D/f") & wait\n')
    env = {"PATH": "/usr/bin:/bin", "D": str(tmp_path)}
    monkeypatch.setattr(H.threading, "Timer", Timer)
    assert H.run_compose(env, tmp_path / "log", 3, tree=tmp_path) == 124 and armed == [180]
    monkeypatch.undo()
    script.write_text('echo composed; exit 7\n')
    assert H.run_compose(env, tmp_path / "log", tree=tmp_path) == 7
    assert (tmp_path / "log").read_text() == "composed\n"
