"""The verifier host's gate (`host.py`) as its tests drive it: a host on the deployed `verifier.json` with a dummy
challenger ssh, systemd as the gate sees it, a request sent through the gate's forced command, and verifier.py's own
parser on the command line the gate writes. `tests/test_host.py` and the remote session's tests of a run through the
gate (`benchmarks/remote_session/tests/test_live_verifier.py`) share it."""
from __future__ import annotations

import argparse
import io
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[6]
PACKAGING = ROOT / "tools" / "research" / "src" / "research" / "pods" / "nebius" / "verifier"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT))

import host as H  # noqa: E402
from verity.protocols.poub.verifier.session import verifier as V  # noqa: E402
from verity.protocols.poub.verifier.remote_session import verifier_process as VP  # noqa: E402

T0 = datetime(2026, 10, 21, 13, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 21, 13, 30, 0, tzinfo=UTC)
PIN = "c" * 40
KEY = "ab" * 64
LANE = "memory-accounting"
RUN = "v20261021T130000Z-ab12"
SSH = "-i /nonexistent/test-gate-key -o IdentitiesOnly=yes research@192.0.2.1"
BASE = ["run", "--peer", "gpu-node", "--port", "7951", "--period", "memory-line-1", "--pin", KEY]
PACED = [*BASE, "--paced", "2:8000:40:2", "4:16000:40:2"]
ROLE = "SYNTHETIC_BF16"
BOUND = [*PACED, "--model", ROLE, "--engines", "2", "--arena-blocks", "7"]


def host_at(tmp_path: Path) -> H.Host:
    """A host under `tmp_path` on the deployed `verifier.json`, its challenger a dummy ssh and its one peer `gpu-node`,
    PIN its pin log's one line and the head of its first-parent list of main."""
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


def parsed(monkeypatch, argv: list[str]) -> tuple[argparse.Namespace, tuple]:
    """The verifier's arguments as its own parser reads the gate's command line, and where it then listens: its
    process's main (`verifier_process.py`) up to the listen, which nothing reaches."""
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
        VP.main()
    return got["a"], e.value.args
