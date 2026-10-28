"""The gate's load (`weights-write` and `load-relay`, `auditor.py`), with its unit, the tunnels and the rehearsal
injected: `weights-write` needs the store disk and a lane's own key, and its run writes interface 1's stream there once
(`load_relay.py write`, here at the tiny shape) and carries the store's record in its end; `load-relay` is refused
unless the store holds the shape as `weights-write` wrote it, its program, sink port, streams and shape check, and its
minutes cover the load and then the rehearsal; its run opens a tunnel to the sink a stream beside the endpoint's, gives
the rehearsal their local ends and the store, and publishes the load's signed registrations and the relay's counts,
never a salt. `test_warden_serve` holds `warden_command`'s load flags to `warden_serve.py rehearse`'s, and
`test_load_relay` the relay itself."""

from __future__ import annotations

import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402

T0 = datetime(2026, 10, 26, 4, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 26, 4, 30, 0, tzinfo=timezone.utc)
PIN = "c" * 40
LANE = "network-accounting"
PROGRAM = "9e" * 64
TUNNEL = [sys.executable, "-c", "import signal; signal.pause()"]
LOAD = ["load-relay", "--peer", "vy-nebius-1", "--port", "8011", "--sink-port", "8021", "--program", PROGRAM,
        "--shape", "tiny", "--minutes", "40"]
LR = A._load_relay()
UNIT = HERE.parents[2] / "tools/research/src/research/pods/nebius/challenger/vy-certifier@.service"


@pytest.fixture
def host(tmp_path) -> A.Host:
    (tmp_path / "var").mkdir()
    (tmp_path / "peers").write_text("# peers\nvy-challenger-1 research@10.0.0.1\n"
                                    "vy-nebius-1 -i /var/lib/vy-challenger/.ssh/vy-challenger-tunnel research@10.0.0.2\n")
    return A.Host(root=tmp_path / "var", peers=tmp_path / "peers", pins=tmp_path / "pins.jsonl")


@pytest.fixture
def stored(host) -> A.Host:
    host.store.mkdir()
    LR.write_store(host.store, LR.TINY)
    return host


def gate(host, argv, *, key=LANE, active=lambda r: False, token="ab12", stdin=None):
    argv = ["--ssh", shlex.join(argv)] + ([] if key is None else ["--key", key])
    return A.gate(argv, host, now=T0, token=token, own=lambda p: None, certifier_active=active,
                  start_certifier=lambda r: None,
                  read=lambda n: pytest.fail("read stdin") if stdin is None else stdin[:n])


def refused(rc: int, out: bytes) -> str:
    assert rc == 3, out
    return json.loads(out)["refused"]


def test_weights_write_needs_the_store_disk_and_a_lane_key_and_its_run_writes_the_store_once(host):
    """The run is `load_relay.py write` itself, at the tiny shape: written the first time, present (read back to the
    pin) the second, and its end carries the store's record less the digests."""
    write = ["weights-write", "--shape", "tiny"]
    assert f"no store disk at {host.store}: infra mounts it, auditor's" in refused(*gate(host, write))
    host.store.mkdir()
    for key in (None, "-"):
        assert "this key reads only: the store's write is started with a lane's own gate key, or by vy-admin" in \
            refused(*gate(host, write, key=key))
    assert "--shape 'small': one of ['h200', 'tiny']" in refused(*gate(host, ["weights-write", "--shape", "small"]))
    assert "weights-write: unrecognized arguments: --peer" in refused(*gate(host, write + ["--peer", "vy-nebius-1"]))
    assert not host.certifier.exists()

    ends = []
    for token in ("ab12", "cd34"):
        rc, out = gate(host, write, token=token)
        doc = json.loads(out)
        run = doc["run"]
        assert rc == 0 and doc == {"run": run, "device": "weights", "shape": "tiny", "minutes": A.WEIGHTS_MINUTES,
                                   "by": LANE, "requested_utc": "2026-10-26T04:00:00Z",
                                   "unit": f"vy-certifier@{run}.service"}
        assert "still runs: one at a time" in refused(*gate(host, write, active=lambda r: r == run, token="ef56"))
        ends.append(A.certifier_run(host, run, pinned=PIN, now=lambda: T1, wait=lambda proc, s: proc.wait()))
    for end, status in zip(ends, ("written", "present")):
        assert end["device"] == "weights" and end["rc"] == 0 and end["ended"] == "the rehearsal ended", end
        assert end["store"] == {"status": status, "format": LR.STORE_FORMAT, "shape": LR.TINY.name,
                                "model": LR.model_id(LR.TINY), "stream_bytes": 34240,
                                "fold": LR.STORE_DIGESTS[LR.TINY.name]}
        assert set(end["files"]) == {"rehearse.log", "store.json"}
    data, sha = LR.read_store(host.store, LR.TINY)
    assert data.stat().st_size == 34240 and len(sha) == len(LR.manifest(LR.TINY))


def test_the_store_write_runs_load_relay_write_over_every_core_into_the_store():
    store, out = Path("/var/lib/vy-challenger/store"), Path("/var/lib/vy-challenger/certifier/c1/out")
    assert A.weights_command({"shape": "h200"}, out, store, tree=Path("/opt/vy-challenger") / PIN) == [
        sys.executable, f"/opt/vy-challenger/{PIN}/benchmarks/network_traces/load_relay.py", "write", "--store",
        str(store), "--shape", "h200", "--workers", str(A.os.cpu_count() or 1), "--out", str(out / "store.json")]
    assert A.Host().store == store


@pytest.mark.parametrize("bad, why", [
    (["--program", "9E" * 64], "the reading program's digest, 128 lowercase hex digits"),
    (["--program", "9e" * 63], "the reading program's digest, 128 lowercase hex digits"),
    (["--sink-port", "8011"], "--sink-port 8011: the sink's port, not the endpoint's --port"),
    (["--sink-port", "80"], "--sink-port '80': 1024 to 65535"),
    (["--streams", "0"], "--streams '0': 1 to 8"),
    (["--streams", "9"], "--streams '9': 1 to 8"),
    (["--shape", "small"], "--shape 'small': one of ['h200', 'tiny']"),
    (["--shape", "h200"], "the store: no store record"),
    (["--minutes", "16"], "--minutes 16: the tiny load takes 16 minutes"),
    (["--minutes", "26"], "--minutes 26: 100 requests take 11 minutes"),
    (["--minutes", "26"], "at height 20) after the load's 16: give --minutes 27 or more, or fewer --requests"),
    (["--minutes", "171"], "--minutes '171': 1 to 170"),
    (["--tamper", "late"], "load-relay: unrecognized arguments: --tamper"),
    (["--load-store", "/tmp"], "load-relay: unrecognized arguments: --load-store"),
])
def test_the_gate_refuses_a_load_it_cant_check(stored, bad, why):
    good = dict(zip(LOAD[1::2], LOAD[2::2]))
    good.update(dict(zip(bad[::2], bad[1::2])))
    assert why in refused(*gate(stored, ["load-relay", *[w for kv in good.items() for w in kv]]))
    assert not stored.certifier.exists()


def test_a_load_is_refused_unless_the_store_holds_its_shape_as_weights_write_wrote_it(host):
    assert "the store: no store record" in refused(*gate(host, LOAD))
    host.store.mkdir()
    assert "write the store first (weights-write)" in refused(*gate(host, LOAD))
    LR.write_store(host.store, LR.TINY)
    data, rec = LR.store_paths(host.store, LR.TINY)
    doc = json.loads(rec.read_text())
    rec.write_text(json.dumps(doc | {"sha256": doc["sha256"][::-1]}))
    assert f"the store: the store record {rec} isn't interface 1's {LR.TINY.name} stream" in \
        refused(*gate(host, LOAD))
    rec.write_text(json.dumps(doc))
    with open(data, "ab") as f:
        f.write(b"\0")
    assert "isn't its 34,240 bytes: write the store again (weights-write)" in refused(*gate(host, LOAD))
    assert "--program" in refused(*gate(host, [w for w in LOAD if w not in ("--program", PROGRAM)]))
    assert not host.certifier.exists()


def test_a_key_that_reads_only_starts_no_load(stored):
    for key in (None, "-"):
        assert "this key reads only: a load is started with a lane's own gate key, or by vy-admin" in \
            refused(*gate(stored, LOAD, key=key))
    assert not stored.certifier.exists()


def test_a_lane_key_starts_a_load_a_warden_rehearsal_with_its_load_whose_minutes_come_first(stored):
    """The request is a warden rehearsal's with ``load``; a `--prompts` request's least minutes are its prompts' after
    the load's, as `warden_prompts` counts them."""
    rc, out = gate(stored, LOAD)
    doc = json.loads(out)
    assert rc == 0 and doc == {"run": "c20261026T040000Z-ab12", "device": "warden", "peer": "vy-nebius-1",
                               "port": 8011, "minutes": 40, "requests": 100, "height": 20, "temperature": 0.0,
                               "top_p": 1.0, "load": {"shape": "tiny", "program": PROGRAM, "sink_port": 8021,
                                                      "streams": 4, "minutes": 16},
                               "by": LANE, "requested_utc": "2026-10-26T04:00:00Z",
                               "unit": "vy-certifier@c20261026T040000Z-ab12.service"}
    assert "still runs: one at a time" in refused(*gate(stored, ["weights-write"], active=lambda r: r == doc["run"],
                                                        token="cd34"))
    rc, out = gate(stored, LOAD + ["--streams", "1", "--minutes", "27"], token="cd34")
    assert rc == 0 and json.loads(out)["load"]["streams"] == 1 and json.loads(out)["minutes"] == 27

    scan = LOAD + ["--scan", "--prompts", "--height", "10"]
    assert "--minutes 19: 1 requests take 4 minutes (with the check and the key at height 10) after the load's 16: " \
        "give --minutes 20 or more" in refused(*gate(stored, scan + ["--minutes", "19"], stdin=b"[[1]]", token="ef56"))
    rc, out = gate(stored, scan + ["--minutes", "20"], stdin=b"[[1]]", token="ef56")
    assert rc == 0 and json.loads(out)["load"]["minutes"] == 16 and json.loads(out)["prompts"] == {"count": 1, "ids": 1}


def test_a_loads_minutes_are_its_weights_and_witness_at_the_links_rate_and_the_endpoints_wait():
    """H200: 139,736,443,904 B of weights and 192 B of salts a leaf (25,284,864 leaves) at 80 MB/s, and 900 s."""
    assert (A.load_minutes("h200"), A.load_minutes("tiny")) == (46, 16)
    assert set(A.LOAD_SHAPES) == set(LR.SHAPES) and LR.SALT_BYTES == 192


def test_the_longest_minutes_fit_in_the_rehearsal_units_runtime_and_leave_an_h200_load_100_of_serving():
    """The gate's cap and 10 minutes for the tunnels and the outcome are within the unit's RuntimeMaxSec."""
    limit = re.search(r"^RuntimeMaxSec=(\d+)h$", UNIT.read_text(), re.M)
    assert limit and A.CERTIFIER_MINUTES[1] + 10 <= int(limit[1]) * 60
    assert A.CERTIFIER_MINUTES[1] - A.load_minutes("h200") >= 100


SALT = "5a" * 64
LOAD_SUMMARY = {"values": 19, "registered": 19, "relayed": True}
RELAY = {"shape": "tiny", "values": 19, "bytes": 34240, "streams": 4, "sinks": [{"accepted": True}] * 4}
#: a stand-in for `warden_serve.py rehearse --load-store`'s files: public, then witness
LOAD_FILES = (
    '(out / "deployment.json").write_text(json.dumps({"upstream": url, "session": "00" * 32}))\n'
    '(out / "cells.jsonl").write_text("")\n'
    f'(out / "certified.jsonl").write_text(json.dumps({{"salt_c": {SALT!r}}}) + "\\n")\n'
    '(out / "load.json").write_text(json.dumps({"format": "load/v1", "registrations": [{"root": "ab" * 64}]}))\n'
    f'(out / "load-relay.json").write_text(json.dumps({RELAY!r}))\n'
    '(out / "load-heads.json").write_text(json.dumps({"heads": [{"root": "ab" * 64}]}))\n'
    f'(out / "check.json").write_text(json.dumps({{"accepted": True, "problems": [], '
    f'"summary": {{"requests": 4, "load": {LOAD_SUMMARY!r}}}}}))\n')


def test_a_load_runs_through_a_tunnel_to_the_sink_a_stream_and_publishes_its_registrations_and_relay_alone(stored):
    rc, out = gate(stored, LOAD)
    run = json.loads(out)["run"]
    tunnels, sinks, ups, seen = [], [], [], []

    def command(req, work, url, **kw):
        seen.append((req, work, url, kw))
        return [sys.executable, "-c", f"import json, pathlib; out, url = pathlib.Path({str(work)!r}), {url!r}\n"
                                      f"{LOAD_FILES}"]

    end = A.certifier_run(stored, run, pinned=PIN, now=lambda: T1,
                          tunnel_argv=lambda s, l: tunnels.append((s, l)) or TUNNEL,
                          sink_argv=lambda s, l, r: sinks.append((s, l, r)) or TUNNEL,
                          up=lambda t, p: ups.append(p) or True, command=command, wait=lambda proc, s: proc.wait())
    (ssh, local), = tunnels
    assert [r for _, _, r in sinks] == [8021] * 4 and all(s == ssh for s, _, _ in sinks)
    locals_ = [l for _, l, _ in sinks]
    assert len({local, *locals_}) == 5 and ups == [local, *locals_]
    (req, work, url, kw), = seen
    d = stored.certifier / run
    assert req["load"]["shape"] == "tiny" and work == d / A.WITNESS and url == f"http://127.0.0.1:{local}"
    assert kw == {"sinks": [("127.0.0.1", p) for p in locals_], "store": stored.store}
    assert end["device"] == "warden" and end["rc"] == 0 and end["ended"] == "the rehearsal ended"
    assert end["load"] == RELAY and end["verdict"] == {"accepted": True, "problems": 0, "requests": 4,
                                                       "load": LOAD_SUMMARY}
    assert set(end["files"]) == {"deployment.json", "cells.jsonl", "certificates.jsonl", "load.json",
                                 "load-relay.json", "tunnel.log", "rehearse.log"}
    assert json.loads((d / "out" / "load.json").read_text())["registrations"] == [{"root": "ab" * 64}]
    assert "load-heads.json" in end["witness"]
    public = b"".join(p.read_bytes() for p in (d / "out").rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    assert SALT.encode() not in public


def test_a_load_whose_sink_tunnel_doesnt_come_up_runs_nothing_and_its_end_says_why(stored):
    rc, out = gate(stored, LOAD)
    run = json.loads(out)["run"]
    ups = []
    end = A.certifier_run(stored, run, pinned=PIN, now=lambda: T1, tunnel_argv=lambda s, l: TUNNEL,
                          sink_argv=lambda s, l, r: TUNNEL, up=lambda t, p: not ups.append(p) and len(ups) < 3,
                          command=lambda *a, **k: pytest.fail("ran without its sink's tunnels"))
    assert end["rc"] == 2 and end["ended"] == \
        "a tunnel to vy-nebius-1's sink at 127.0.0.1:8021 didn't come up (tunnel.log)"
    assert len(ups) == 3 and end["verdict"] is None and "load" not in end and set(end["files"]) == {"tunnel.log"}


def test_a_loads_rehearsal_is_warden_serves_with_its_load_flags_after_the_rehearsals(tmp_path):
    req = {"device": "warden", "requests": 100, "load": {"shape": "tiny", "program": PROGRAM, "sink_port": 8021,
                                                         "streams": 2, "minutes": 16}}
    store = Path("/var/lib/vy-challenger/store")
    plain = A.warden_command({k: v for k, v in req.items() if k != "load"}, tmp_path, "http://127.0.0.1:40001")
    argv = A.warden_command(req, tmp_path, "http://127.0.0.1:40001", sinks=[("127.0.0.1", 40002), ("127.0.0.1", 40003)],
                            store=store)
    assert argv == [*plain, "--load-store", str(store), "--load-shape", "tiny", "--load-sinks",
                    "127.0.0.1:40002,127.0.0.1:40003", "--load-program", PROGRAM, "--load-ready-s", "900"]
    assert A.forward_argv(["-i", "KEY", "research@10.0.0.2"], 40002, 8021) == [
        *A.TUNNEL_SSH, "-L", "127.0.0.1:40002:127.0.0.1:8021", "-i", "KEY", "research@10.0.0.2"]
