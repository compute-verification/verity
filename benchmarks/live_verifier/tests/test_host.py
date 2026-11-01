"""The verifier host's gate (`host.py`) with its root steps (chown, systemctl) and the clock injected, on the deployed
`verifier.json` with a dummy challenger ssh: a run is held to the whitelist, written as its request and started as its
one unit, and its command line is one verifier.py's own parser reads as the request; the gate refuses any other verb,
any other form of its forced command, a parameter the whitelist doesn't hold, and a read-only key's run; a run is
stopped only by its lane or vy-admin, a replay or a close waits for no unit to run, and a run whose pin moved, or on a
host with no first-parent list of main, doesn't run. The slow test runs a run through the stub verifier
(`stub_verifier.py`, a real paced session): its out/ is verifier.py's, which `auditor_line.py` reads, `fetch` ships it
and `replay` replays it through both verifiers."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RS = ROOT / "benchmarks" / "remote_session"
PACKAGING = ROOT / "tools" / "research" / "src" / "research" / "pods" / "nebius" / "verifier"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(RS))

import challengers  # noqa: E402
import host as H  # noqa: E402
import store_layout as SL  # noqa: E402
import verifier as V  # noqa: E402

T0 = datetime(2026, 10, 21, 13, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 21, 13, 30, 0, tzinfo=timezone.utc)
PIN = "c" * 40
KEY = "ab" * 64
LANE = "memory-accounting"
RUN = "v20261021T130000Z-ab12"
SSH = "-i /nonexistent/test-gate-key -o IdentitiesOnly=yes research@192.0.2.1"
BASE = ["run", "--peer", "gpu-node", "--port", "7951", "--period", "memory-line-1", "--pin", KEY]
PACED = [*BASE, "--paced", "2:8000:40:2", "4:16000:40:2"]
SESSIONS = ["run", "--peer", "gpu-node", "--port", "7958", "--period", "memory-line-1", "--pin", KEY,
            "--sessions", "3", "--windows", "7", "--window-ms", "1000", "--samples", "2773", "--minutes", "30",
            "--jitter-ms", "2.5", "--quantile", "p99", "--lifetime", "4"]
ROLE = "SYNTHETIC_BF16"
BOUND = [*PACED, "--model", ROLE, "--engines", "2", "--arena-blocks", "7"]


@pytest.fixture
def svc(tmp_path) -> H.Host:
    settings = json.loads((PACKAGING / "verifier.json").read_text())
    settings["challenger"].update(ssh=SSH, peer="auditor-host")
    settings["peers"] = {"gpu-node": next(iter(settings["peers"].values()))}
    (tmp_path / "verifier.json").write_text(json.dumps(settings))
    (tmp_path / "var" / "runs").mkdir(parents=True)
    host = H.Host(root=tmp_path / "var", config=tmp_path / "verifier.json", pins=tmp_path / "pins.jsonl",
                  main=tmp_path / "main-first-parent.txt")
    H.record_pin(host.pins, PIN, T0, seal=lambda p: None)
    host.main.write_text(f"{PIN}\n{'e' * 40}\n")
    return host


class Units:
    """systemd as the gate sees it: the units it started (live until cleared), stopped and closed."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.live: list[str] = []
        self.closed: list[str] = []

    def busy(self) -> list[str]:
        return list(self.live)

    def start(self, unit: str) -> None:
        self.started.append(unit)
        self.live.append(unit)

    def stop(self, unit: str) -> None:
        self.stopped.append(unit)

    def close(self, node: str) -> dict:
        self.closed.append(node)
        return {"node": node, "closed": True}


def gate(svc, argv, units: Units | None = None, *, token="ab12", now=T0, **hooks) -> tuple[int, bytes]:
    units = units or Units()
    sink = io.BytesIO()
    rc = H.gate(argv, svc, sink, now=now, token=token, pinned=PIN, busy=units.busy, start=units.start,
                stop=units.stop, close=units.close, own=lambda p: None, **hooks)
    return rc, sink.getvalue()


def call(svc, words, units: Units | None = None, *, key=LANE, **kw) -> tuple[int, bytes]:
    """`words` as a gate key registered for `key` sends them (`-`, a read-only key)."""
    return gate(svc, ["--ssh", shlex.join(words), "--key", key], units, **kw)


def refused(got: tuple[int, bytes]) -> str:
    rc, out = got
    assert rc == 3, out
    return json.loads(out)["refused"]


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
    assert argv[:3] == ["/usr/bin/python3", "-I", str(H.TREE / "benchmarks" / "remote_session" / "verifier.py")]
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


def parsed(monkeypatch, argv: list[str]) -> tuple[argparse.Namespace, tuple]:
    """verifier.py's arguments as its own parser reads the gate's command line, and where it then listens: its main up
    to the listen, which nothing reaches."""
    got = {}
    parse = argparse.ArgumentParser.parse_args

    def keep(self, *args, **kw):
        got["a"] = parse(self, *args, **kw)
        return got["a"]

    class Listened(Exception):
        pass

    def listen(address, port):
        raise Listened(address, port)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", keep)
    monkeypatch.setattr(V, "Checks", lambda workers: None)
    monkeypatch.setattr(V.wire, "listen", listen)
    monkeypatch.setattr(sys, "setswitchinterval", lambda s: None)
    monkeypatch.setattr(sys, "argv", argv[2:])
    with pytest.raises(Listened) as e:
        V.main()
    return got["a"], e.value.args


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
    """A pinned tree whose core pin (`models.json`'s shape) pins ROLE as the checkpoint infra provisioned in the host's
    models dir: a small synthetic BF16 one (`store_layout.synthetic`, an arena of 6 weights blocks)."""
    ckpt = svc.root / "models" / ROLE
    pin = SL.synthetic(ckpt, name=ROLE, layers=1, hidden=16, heads=2, kv_heads=1, intermediate=32, vocab=64)
    (ckpt / "pin.json").unlink()
    t = tmp_path / "tree"
    (t / H.MODEL_PINS).parent.mkdir(parents=True)
    (t / H.MODEL_PINS).write_text(json.dumps({ROLE: {
        "checkpoint_sha512": pin["checkpoint_sha512"], "files": pin["checkpoint"], "recipe": "bf16/v1",
        "repo": "example/synthetic-bf16", "revision": "0" * 40}}, indent=1, sort_keys=True) + "\n")
    return t


def test_a_bound_runs_command_line_names_its_roles_checkpoint_and_cores_pin_which_verifier_py_binds_to(
        svc, tree, monkeypatch, capsys, tmp_path):
    """`--model ROLE` names no path: the gate adds the role's directory in the host's models dir as `--checkpoint` and
    the pinned tree's `models.json` as verifier.py's `--pin`, apart from the gate's own `--pin` (the challenger's key,
    `--challenger-pin`); the rest is the unbound run's command line, and verifier.py binds the store to that pin before
    it listens."""
    rc, out = call(svc, BOUND, tree=tree)
    doc = json.loads(out)
    assert rc == 0, doc
    req = json.loads((svc.runs / RUN / "request.json").read_text())
    assert (req["model"], req["engines"], req["arena-blocks"], req["pin"]) == (ROLE, 2, 7, KEY)
    argv = req["argv"]
    assert argv == doc["argv"]
    i = argv.index("--model")
    assert argv[i:] == ["--model", ROLE, "--checkpoint", str(svc.root / "models" / ROLE), "--pin",
                        str(tree / H.MODEL_PINS), "--engines", "2", "--arena-blocks", "7",
                        "--paced", "2:8000:40:2", "4:16000:40:2"]
    assert argv.count("--pin") == 1 and argv[argv.index("--challenger-pin") + 1] == KEY
    unbound = {k: v for k, v in req.items() if k not in ("model", "engines", "arena-blocks", "argv")}
    assert argv[:i] + argv[i + 10:] == H.verifier_argv(svc, RUN, unbound, svc.settings(), tree=tree)

    monkeypatch.setattr(V.os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    a, listened = parsed(monkeypatch, argv)
    assert listened == ("127.0.0.1", 7951)
    assert (a.model, a.checkpoint, a.pin, a.engines, a.arena_blocks, a.challenger_pin, a.layout_workers) == (
        ROLE, svc.root / "models" / ROLE, tree / H.MODEL_PINS, 2, 7, KEY, 1)
    pinned = json.loads((tree / H.MODEL_PINS).read_text())[ROLE]["checkpoint_sha512"]
    assert f"store: {ROLE} pinned ({pinned[:16]}...), " in capsys.readouterr().out


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


def test_a_tunnel_key_gets_a_pty_only_while_the_challengers_pinned_client_forces_one():
    """ssh ends a session whose forced pty (`-tt`) the server refuses (OpenSSH 9.6, checked against a real sshd), so the
    prover's key, whose client is run.sh's `ssh -T`, has no `pty`, and the challenger's keeps it only while serve.py
    forces one."""
    setup = (PACKAGING / "setup.sh").read_text()
    assert "opts=restrict,port-forwarding\n" in setup and '[ "$acct" = prover ] || opts+=,pty\n' in setup
    verifier_host = (RS / "run.sh").read_text().split('[ -n "${VERIFIER_SSH:-}" ]; then')[1].split("\nelif ")[0]
    assert re.search(r"\bssh -T\b", verifier_host) and "-tt" not in verifier_host
    serve = (ROOT / "benchmarks" / "live_challenger" / "serve.py").read_text()
    assert re.search(r'^SSH = \["ssh", "-tt"', serve, re.M), (
        "serve.py no longer forces a pty: drop `pty` from the challenger's tunnel key in verifier/setup.sh, and this check")


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
    "sys.argv = ['auditor_line.py', '--help']; runpy.run_path(str(host.VERIFIER.parent / 'auditor_line.py'), "
    "run_name='__main__')",
])
def test_the_hosts_code_runs_on_a_python_with_no_site_packages(code, tmp_path):
    """The host runs its units with its own python3 (`-I`), where neither `verity` nor any other package is installed:
    each import the run, the replay and the publish make resolves within the pinned tree."""
    p = subprocess.run([sys.executable, "-I", "-S", "-c", f"import runpy, sys; sys.path.insert(0, {str(HERE.parent)!r}); "
                        f"import host; {code}"], capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "site-packages" not in p.stdout


def names(tar: tarfile.TarFile) -> set[str]:
    return {m.name for m in tar.getmembers() if m.isfile()}


@pytest.mark.slow
def test_a_run_through_the_stub_verifier_leaves_verifier_pys_record_which_fetch_ships_and_replay_replays(
        svc, tmp_path):
    """The one test that runs a session (with the Lean verdict, so it builds the Lean verifier and the difftest): what
    the cheaper tests can't show is that the runner's out/ is the one `auditor_line.py` and `replay.py` read, from a
    real transcript, gzipped, summed and shipped."""
    units = Units()
    rc, out = call(svc, [*BASE, "--paced", "2:8:40:2"], units)
    assert rc == 0, out
    req = json.loads((svc.runs / RUN / "request.json").read_text())
    end = H.execute(svc, RUN, pinned=PIN, now=lambda: T1,
                    argv=[sys.executable, str(HERE / "stub_verifier.py"), *req["argv"][3:]])
    d, o = svc.runs / RUN, svc.runs / RUN / "out"
    assert end["rc"] == 0, (o / "verifier.log").read_text()
    assert sorted(p.name for p in o.iterdir()) == [
        "challenger", "rc.txt", "registry.json", "transcripts", "verifier.json", "verifier.log"]
    assert not (svc.scratch / RUN).exists()
    v = json.loads((o / "verifier.json").read_text())
    (s,) = v["sessions"]
    t = s["transcript"]
    name = Path(t["path"]).name
    assert s["accepted"] and Path(t["path"]).parent == svc.scratch / RUN / "transcripts"
    raw = gzip.decompress((o / "transcripts" / f"{name}.gz").read_bytes())
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == (t["bytes"], t["sha256"])
    assert (o / "transcripts" / "SHA256SUMS").read_text() == (
        f"{hashlib.sha256((o / 'transcripts' / f'{name}.gz').read_bytes()).hexdigest()}  ./{name}.gz\n")
    assert end["transcripts"] == [{"name": name, "bytes": t["bytes"]}]
    assert end["files"] == {str(p.relative_to(d)): H.digest_of(p) for p in sorted(o.rglob("*")) if p.is_file()}
    assert json.loads((o / "registry.json").read_text()) == v["registry"] == json.loads(svc.registry.read_text())
    units.live.clear()
    rc, st = call(svc, ["status", RUN], units, key="-")
    st = json.loads(st)
    assert rc == 0 and st["listening"] and not st["active"] and st["end"]["rc"] == 0 and st["replay"] is None

    import auditor_line
    line = auditor_line.analyse(o, host_bytes=1843153879040, pin=None, period=None, heads=None, peer="auditor-host")
    assert line["registration"] == v["registration"]["digest"]
    assert [(r["k"], r["accepted"], r["skipped"]) for r in line["sessions"]] == [
        (0, True, "no challenger log: coins drawn in the verifier's process")]

    rc, shipped = call(svc, ["fetch", RUN], units, key="-")
    assert rc == 0
    with tarfile.open(fileobj=io.BytesIO(shipped), mode="r:gz") as tar:
        light = names(tar)
    assert {f"{RUN}/request.json", f"{RUN}/end.json", f"{RUN}/out/verifier.json", f"{RUN}/out/transcripts/SHA256SUMS",
            f"{RUN}/out/transcripts/transcripts.json"} <= light
    assert not any(n.endswith(".json.gz") for n in light)
    rc, shipped = call(svc, ["fetch", RUN, "--transcripts"], units, key="-")
    assert rc == 0
    with tarfile.open(fileobj=io.BytesIO(shipped), mode="r:") as tar:
        assert names(tar) == light | {f"{RUN}/out/transcripts/{name}.gz"}
        tar.extractall(tmp_path / "fetched", filter="data")
    assert all((tmp_path / "fetched" / n).read_bytes() == (svc.runs / n).read_bytes() for n in light)

    assert call(svc, ["replay", RUN], units)[0] == 0
    replayed = H.replay_run(svc, RUN, now=lambda: T1)
    assert (replayed["rc"], replayed["kind"], replayed["sums_match"], replayed["gz_sums"]) == (
        0, "transcript", True, {f"{name}.gz": True}), (d / "replay" / "replay.log").read_text()
    rows = json.loads((d / "replay" / "replay.json").read_text())
    assert rows and all(r["agree"] and r["held"] and r["recorded"] == "accepted" for r in rows), rows
    assert not (svc.scratch / f"{RUN}-replay").exists()

    gz = o / "transcripts" / f"{name}.gz"
    gz.write_bytes(gzip.compress(raw.replace(b'"events"', b'"events" ', 1)))
    tampered = H.replay_run(svc, RUN, now=lambda: T1)
    assert tampered["rc"] == 1 and tampered["gz_sums"] == {f"{name}.gz": False} and not tampered["sums_match"]


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
                    "-m", "verity.core.protocols.pous.wipe_party", "--device", str(svc.parties / "wipe"),
                    "--listen", "0.0.0.0:47611", "--reads", "1024", "--streamer", "build", "--threads", "16", "minutes=60"]]
    reqs = [json.loads(ln) for ln in (svc.parties / "wipe" / "requests.jsonl").read_text().splitlines()]
    assert [(r["verb"], r["by"], r.get("rc")) for r in reqs] == [("wipe_party", LANE, None), ("wipe_party", LANE, 0)]
    assert (svc.parties / "wipe" / "log" / reqs[0]["log"]).read_bytes() == out
    (rc, _), ran = party(svc, ["challenger_party", "--minutes", "5"])
    assert rc == 0 and ran[0][-7:] == ["-m", H.PARTIES["challenger_party"][1], "--device",
                                       str(svc.parties / "pous"), "--listen", "0.0.0.0:47610", "minutes=5"]


def test_bw_party_holds_the_store_the_block_challenger_kept_for_a_pous_session_else_the_fixture_s(svc):
    """The bandwidth party's registration is the host's: the block challenger's own record of the PoUS session the lane
    names (its device on this host, by session id), or the fixture's registration in the pinned tree; never a path."""
    sid = "0f" * 32
    (rc, _), ran = party(svc, ["bw_party", "--pous-session", sid, "--minutes", "8"])
    assert rc == 0 and ran[0][-11:] == ["-m", H.PARTIES["bw_party"][1], "--device", str(svc.parties / "bw"),
                                        "--listen", "0.0.0.0:47612", "--pous-record", str(svc.parties / "pous"),
                                        "--pous-session", sid, "minutes=8"]
    (rc, _), ran = party(svc, ["bw_party"])
    assert rc == 0 and ran[0][-9:] == ["-m", H.PARTIES["bw_party"][1], "--device", str(svc.parties / "bw"),
                                       "--listen", "0.0.0.0:47612", *H.PARTY_FIXED["bw_party"], "minutes=60"]
    assert (H.TREE / H.PARTY_FIXED["bw_party"][1]).is_file()


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
    (["bw_party", "--registration", "/tmp/r.json"], "bw_party takes [--pous-session SID]; got '--registration'"),
    (["bw_party", "--pous-record", "/tmp"], "got '--pous-record'"),
    (["bw_party", "--pous-session", "../" + "a" * 61], "64 lowercase hex digits"),
    (["bw_party", "--pous-session", "A" * 64], "64 lowercase hex digits"),
    (["bw_party", "--pous-session", "a" * 64, "--pous-session", "b" * 64], "got '--pous-session'"),
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
    code = ("import verity.core.protocols.pous.wipe_party, verity.core.protocols.pous.period, "
            "verity.core.protocols.pous.challenger_party, verity.core.protocols.remote_bandwidth.p2_party; print('ok')")
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
    (rc, _), ran = party(svc, ["bw_party", "--minutes", "10"])
    assert rc == 0 and ran[0][-9:] == ["-m", "verity.core.protocols.remote_bandwidth.p2_party", "--device", str(svc.parties / "bw"),
                                       "--listen", "0.0.0.0:47612", "--registration",
                                       "benchmarks/pous/p2_v1/bandwidth-mem69-v2.json", "minutes=10"]
    assert "bw_party takes [--pous-session SID]" in refused(party(svc, ["bw_party", "--registration", "/tmp/x"])[0])
