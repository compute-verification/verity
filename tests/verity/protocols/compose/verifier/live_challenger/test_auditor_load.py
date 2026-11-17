"""The gate's load (`weights-write` and `load-relay`, `auditor.py`), with its unit, the tunnels and the rehearsal
injected: `weights-write` needs the store disk and a lane's own key, and its run writes interface 1's stream there once
(`load_relay.py write`, here at the tiny shape) and carries the store's record in its end, and a stop of its unit,
which the unit sends to `attester-run` alone, still ends it with its end; `load-relay` is refused
unless the store holds the shape as `weights-write` wrote it, its program, sink port, streams and shape check, and its
minutes cover the load and then the rehearsal; its run opens a tunnel to the sink a stream beside the endpoint's, gives
the rehearsal their local ends and the store, and publishes the load's signed registrations and the relay's counts,
never a salt. `test_network_attester_serve` holds `network_attester_command`'s load flags to `network_attester_serve.py rehearse`'s, and
`test_load_relay` the relay itself."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import sys
from datetime import datetime, UTC
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[5] / "verity" / "protocols" / "compose" / "verifier" / "live_challenger"))
import auditor as A  # noqa: E402

T0 = datetime(2026, 10, 26, 4, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 26, 4, 30, 0, tzinfo=UTC)
PIN = "c" * 40
LANE = "network-accounting"
PROGRAM = "9e" * 64
TUNNEL = [sys.executable, "-c", "import signal; signal.pause()"]
LOAD = ["load-relay", "--peer", "vy-nebius-1", "--port", "8011", "--sink-port", "8021", "--program", PROGRAM,
        "--shape", "tiny", "--minutes", "40"]
LR = A._load_relay()
UNIT = Path(__file__).resolve().parents[6] / "tools/research/src/research/pods/nebius/challenger/vy-certifier@.service"


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
    return A.gate(argv, host, now=T0, token=token, own=lambda p: None, attester_active=active,
                  start_attester=lambda r: None,
                  read=lambda n: pytest.fail("read stdin") if stdin is None else stdin[:n])


def ended_why(host, end: dict) -> str:
    """The sentence a run's end's `ended` word stands for, which the gate writes to the run's `witness/` alone."""
    assert end["ended"] in A.ENDED
    return (host.attester / end["run"] / A.WITNESS / A.ENDED_WHY).read_text().removesuffix("\n")


def refused(rc: int, out: bytes) -> str:
    assert rc == 3, out
    return json.loads(out)["refused"]


def test_weights_write_needs_the_store_disk_and_a_lane_key_and_its_run_writes_the_store_once(host):
    """The run is `load_relay.py write` itself, at the tiny shape: written the first time, present (read back to the
    pin) the second, and its end carries the store's record less the digests. The record as the write left it
    and what the write printed are the witness's, and no refusal says where the store is."""
    write = ["weights-write", "--shape", "tiny"]
    no_disk = refused(*gate(host, write))
    assert "the store: no-store (one of load_relay.REFUSALS' words)" in no_disk and str(host.store) not in no_disk
    host.store.mkdir()
    for key in (None, "-"):
        assert "this key reads only: the store's write is started with a lane's own gate key, or by vy-admin" in \
            refused(*gate(host, write, key=key))
    assert "--shape 'small': one of ['h200', 'tiny']" in refused(*gate(host, ["weights-write", "--shape", "small"]))
    assert "weights-write: unrecognized arguments: --peer" in refused(*gate(host, write + ["--peer", "vy-nebius-1"]))
    assert not host.attester.exists()

    ends = []
    for token in ("ab12", "cd34"):
        rc, out = gate(host, write, token=token)
        doc = json.loads(out)
        run = doc["run"]
        assert rc == 0 and doc == {"run": run, "device": "weights", "shape": "tiny", "minutes": A.WEIGHTS_MINUTES,
                                   "by": LANE, "requested_utc": "2026-10-26T04:00:00Z",
                                   "unit": f"vy-certifier@{run}.service"}
        assert "still runs: one at a time" in refused(*gate(host, write, active=lambda r: r == run, token="ef56"))
        ends.append(A.certifier_run(host, run, pinned=PIN, wait=lambda proc, s: proc.wait()))
    for end, status in zip(ends, ("written", "present")):
        assert end["device"] == "weights" and end["rc"] == 0 and end["ended"] == "ended", end
        assert ended_why(host, end) == "the rehearsal ended"
        assert end["store"] == {"status": status, "format": LR.STORE_FORMAT, "shape": LR.TINY.name,
                                "model": LR.model_id(LR.TINY), "stream_bytes": 34240,
                                "fold": LR.STORE_DIGESTS[LR.TINY.name]}
        d = host.attester / end["run"]
        assert set(end["files"]) == {"store.json"} and json.loads((d / "out" / "store.json").read_text()) == end["store"]
        assert {p.name for p in (d / A.WITNESS).iterdir()} == {"rehearse.log", "store.json", A.ENDED_WHY}
        assert json.loads((d / A.WITNESS / "store.json").read_text()) == end["store"]
    assert A.STORE_FORMAT == LR.STORE_FORMAT and set(A.LOAD_SHAPES) == set(LR.SHAPES)
    data, sha = LR.read_store(host.store, LR.TINY)
    assert data.stat().st_size == 34240 and len(sha) == len(LR.manifest(LR.TINY))


def test_a_refused_store_write_ends_in_its_word_and_a_record_out_of_form_is_not_published(host):
    """A write the runner refuses ends in the refusal's word alone: the sentence, which names the store's path on
    the auditor's device, is in the witness's log and nowhere in the end or `out/`. A record `STORE_PUBLIC`
    doesn't hold whole and in form is not published: the run ends nonzero as `unruled`."""
    host.store.mkdir()
    write = ["weights-write", "--shape", "tiny"]
    run = json.loads(gate(host, write)[1])["run"]
    host.store.rmdir()
    end = A.certifier_run(host, run, pinned=PIN, wait=lambda proc, s: proc.wait())
    d = host.attester / run
    assert end["rc"] == 1 and end["ended"] == "exited" and end["store"] == {"refused": "no-store"}
    assert json.loads((d / "out" / "store.json").read_text()) == {"refused": "no-store"}
    public = (d / "end.json").read_text() + (d / "out" / "store.json").read_text()
    assert str(host.store) not in public and "no store directory" not in public
    assert f"refused (no-store): no store directory {host.store}" in (d / A.WITNESS / "rehearse.log").read_text()

    good = {"status": "written", "format": A.STORE_FORMAT, "shape": "tiny", "model": "m/tiny/0a",
            "stream_bytes": 34240, "fold": "ab" * 64}
    assert A.store_public(good) == good and A.store_public({"refused": "store-size"}) == {"refused": "store-size"}
    assert A.store_public({"refused": f"no store directory {host.store}"}) == {"refused": "other"}
    for bad in (good | {"sha256": []}, {k: v for k, v in good.items() if k != "fold"}, good | {"status": "kept"},
                good | {"format": "another/v1"}, good | {"shape": "a shape"}, good | {"model": "a model"},
                good | {"stream_bytes": True}, good | {"stream_bytes": 1 << 63}, good | {"fold": "AB" * 64}, [good]):
        with pytest.raises(A.Unruled, match="STORE_PUBLIC"):
            A.store_public(bad)
    host.store.mkdir()
    run = json.loads(gate(host, write, token="cd34")[1])["run"]
    wrote = f'import json; (out / "store.json").write_text(json.dumps({good | {"path": str(host.store)}!r}))'
    end = A.certifier_run(host, run, pinned=PIN, wait=lambda proc, s: proc.wait(),
                          command=lambda req, out, store: [sys.executable, "-c",
                                                           f"import pathlib; out = pathlib.Path({str(out)!r})\n{wrote}"])
    assert (end["rc"], end["ended"], end["store"], end["files"]) == (2, "unruled", None, {})
    assert "holds ['fold', 'format', 'model', 'path'" in ended_why(host, end)
    assert str(host.store) not in (host.attester / run / "end.json").read_text()


def test_the_store_write_runs_load_relay_write_over_every_core_into_the_store():
    store, out = Path("/var/lib/vy-challenger/store"), Path("/var/lib/vy-challenger/certifier/c1/out")
    assert A.weights_command({"shape": "h200"}, out, store, tree=Path("/opt/vy-challenger") / PIN) == [
        sys.executable, f"/opt/vy-challenger/{PIN}/verity/protocols/resource_accounting/network/attester/load_relay.py", "write", "--store",
        str(store), "--shape", "h200", "--workers", str(A.os.cpu_count() or 1), "--out", str(out / "store.json")]
    assert A.Host().store == store


def test_a_store_write_whose_unit_is_stopped_stops_its_writer_and_writes_its_end(host, stop_held):
    """The unit's stop reaches the writer, which here ends only on a signal, and the end says the unit stopped it."""
    host.store.mkdir()
    rc, out = gate(host, ["weights-write", "--shape", "tiny"])

    def wait(proc, seconds):
        os.kill(os.getpid(), signal.SIGTERM)
        return proc.wait()

    end = A.certifier_run(host, json.loads(out)["run"], pinned=PIN, wait=wait,
                          command=lambda req, out, store: [sys.executable, "-c", "import signal; signal.pause()"])
    assert end["device"] == "weights" and end["rc"] == -signal.SIGTERM and end["ended"] == "stopped"
    assert ended_why(host, end) == A.STOPPED
    assert end["store"] is None and end["files"] == {}
    assert (host.attester / end["run"] / A.WITNESS / "rehearse.log").is_file()


def test_a_store_write_whose_unit_is_stopped_fails_though_its_writer_exits_0_on_the_signal(host, stop_held, tmp_path):
    """A stopped run is a failed one whatever its writer returns: a writer that exits 0 on the unit's SIGTERM still
    leaves the run's rc nonzero."""
    host.store.mkdir()
    rc, out = gate(host, ["weights-write", "--shape", "tiny"])
    ready = tmp_path / "ready"
    os.mkfifo(ready)

    def wait(proc, seconds):
        open(ready).close()
        os.kill(os.getpid(), signal.SIGTERM)
        return proc.wait()

    writer = ("import signal, sys\nsignal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
              f"open({str(ready)!r}, 'w').close()\nsignal.pause()")
    end = A.certifier_run(host, json.loads(out)["run"], pinned=PIN, wait=wait,
                          command=lambda req, out, store: [sys.executable, "-c", writer])
    assert end["rc"] == 2 and end["ended"] == "stopped"
    assert ended_why(host, end) == A.STOPPED


@pytest.mark.parametrize("bad, why", [
    (["--program", "9E" * 64], "the reading program's digest, 128 lowercase hex digits"),
    (["--program", "9e" * 63], "the reading program's digest, 128 lowercase hex digits"),
    (["--sink-port", "8011"], "--sink-port 8011: the sink's port, not the endpoint's --port"),
    (["--sink-port", "80"], "--sink-port '80': 1024 to 65535"),
    (["--streams", "0"], "--streams '0': 1 to 8"),
    (["--streams", "9"], "--streams '9': 1 to 8"),
    (["--shape", "small"], "--shape 'small': one of ['h200', 'tiny']"),
    (["--shape", "h200"], "the store: no-record (one of load_relay.REFUSALS' words): weights-write --shape h200"),
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
    assert not stored.attester.exists()


def test_a_load_is_refused_unless_the_store_holds_its_shape_as_weights_write_wrote_it(host):
    def word() -> str:
        why = refused(*gate(host, LOAD))
        assert why.startswith("the store: ") and str(host.store) not in why and "weights-write --shape tiny" in why
        return why.removeprefix("the store: ").split(" ")[0]

    assert word() == "no-record"
    host.store.mkdir()
    assert word() == "no-record"
    LR.write_store(host.store, LR.TINY)
    data, rec = LR.store_paths(host.store, LR.TINY)
    doc = json.loads(rec.read_text())
    rec.write_text(json.dumps(doc | {"sha256": doc["sha256"][::-1]}))
    assert word() == "not-interface-1"
    rec.write_text(json.dumps(doc))
    with open(data, "ab") as f:
        f.write(b"\0")
    assert word() == "store-size"
    assert "--program" in refused(*gate(host, [w for w in LOAD if w not in ("--program", PROGRAM)]))
    assert not host.attester.exists()


def test_a_key_that_reads_only_starts_no_load(stored):
    for key in (None, "-"):
        assert "this key reads only: a load is started with a lane's own gate key, or by vy-admin" in \
            refused(*gate(stored, LOAD, key=key))
    assert not stored.attester.exists()


def test_a_lane_key_starts_a_load_a_network_attester_rehearsal_with_its_load_whose_minutes_come_first(stored):
    """The request is a network attester rehearsal's with ``load``; a `--prompts` request's least minutes are its prompts' after
    the load's, as `network_attester_prompts` counts them."""
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
    assert limit and A.ATTESTER_MINUTES[1] + 10 <= int(limit[1]) * 60
    assert A.ATTESTER_MINUTES[1] - A.load_minutes("h200") >= 100


def test_the_rehearsal_units_stop_signals_attester_run_alone_and_gives_it_time_to_write_the_end():
    """`KillMode=mixed`: a stop's SIGTERM goes to `attester-run` alone, which passes it to the runner, closes the
    tunnels and writes the end (`auditor._Stop`); only what is left at `TimeoutStopSec` is killed."""
    unit = UNIT.read_text()
    assert re.search(r"^KillMode=mixed$", unit, re.M)
    assert re.search(r"^TimeoutStopSec=\d+(s|min)$", unit, re.M)


SALT = "5a" * 64
#: a load value's registration as the stand-in writes it: its root, and its certificate's digest and signature
ROOT, DIGEST, SIGNATURE = "ab" * 64, "9f" * 64, "e7" * 40
LOAD_DOC = {"format": "load/v1", "values": [
    {"value": "w", "root": ROOT, "certificate": {"key": "0c" * 16, "digest": DIGEST, "signature": SIGNATURE}}]}
LOAD_SUMMARY = {"values": 19, "registered": 19, "relayed": True}
#: a relayed load's record with `load_relay.relay_load`'s keys (`test_load_relay` holds the relay to them): the
#: shape's counts, the relay's own seconds, jobs and workers, and each stream's acknowledgement
RELAY = {"shape": "tiny-h32-l1", "model": "verity/seeded-weights/v1/tiny-h32-l1/" + "5e" * 16, "values": 19,
         "rows": 32, "link_bytes": 34240, "witness_bytes": 6144, "seconds": 0.375, "jobs": 3, "workers": 2,
         "streams": [{"weights": 9000, "witness": 1536}, {"weights": 8000, "witness": 2048},
                     {"weights": 9240, "witness": 512}, {"weights": 8000, "witness": 2048}]}
#: a stand-in for `network_attester_serve.py rehearse --load-store`'s files
LOAD_FILES = (
    '(out / "deployment.json").write_text(json.dumps({"session": "00" * 32}))\n'
    '(out / "endpoints.json").write_text(json.dumps({"upstream": url}))\n'
    '(out / "cells.jsonl").write_text("")\n'
    '(out / "certified.jsonl").write_text("")\n'
    f'(out / "cells.salts").write_text({SALT!r})\n'
    f'(out / "load.json").write_text(json.dumps({LOAD_DOC!r}))\n'
    f'(out / "load-relay.json").write_text(json.dumps({RELAY!r}))\n'
    f'(out / "load-heads.json").write_text(json.dumps({{"heads": [{{"root": {ROOT!r}}}]}}))\n'
    f'(out / "check.json").write_text(json.dumps({{"accepted": True, "problems": [], '
    f'"summary": {{"requests": 100, "load": {LOAD_SUMMARY!r}}}}}))\n')


def test_a_load_runs_through_a_tunnel_to_the_sink_a_stream_and_publishes_neither_its_registrations_nor_its_relay(stored):
    rc, out = gate(stored, LOAD)
    run = json.loads(out)["run"]
    tunnels, sinks, ups, seen = [], [], [], []

    def command(req, work, url, **kw):
        seen.append((req, work, url, kw))
        return [sys.executable, "-c", f"import json, pathlib; out, url = pathlib.Path({str(work)!r}), {url!r}\n"
                                      f"{LOAD_FILES}"]

    end = A.certifier_run(stored, run, pinned=PIN,
                          tunnel_argv=lambda s, l: tunnels.append((s, l)) or TUNNEL,
                          sink_argv=lambda s, l, r: sinks.append((s, l, r)) or TUNNEL,
                          up=lambda t, p: ups.append(p) or True, command=command, wait=lambda proc, s: proc.wait())
    (ssh, local), = tunnels
    assert [r for _, _, r in sinks] == [8021] * 4 and all(s == ssh for s, _, _ in sinks)
    locals_ = [l for _, l, _ in sinks]
    assert len({local, *locals_}) == 5 and ups == [local, *locals_]
    (req, work, url, kw), = seen
    d = stored.attester / run
    assert req["load"]["shape"] == "tiny" and work == d / A.WITNESS and url == f"http://127.0.0.1:{local}"
    assert kw == {"sinks": [("127.0.0.1", p) for p in locals_], "store": stored.store}
    assert end["device"] == "warden" and end["rc"] == 0 and end["ended"] == "ended"
    assert ended_why(stored, end) == "the rehearsal ended"
    assert "load" not in end and not (d / "out" / "load-relay.json").exists()
    assert json.loads((d / A.WITNESS / "load-relay.json").read_text()) == RELAY
    assert end["verdict"] == {"accepted": True, "load": LOAD_SUMMARY}
    assert set(end["files"]) == {"deployment.json", "cells-certificates.jsonl", "cells-registrations.jsonl",
                                 "certificates.jsonl"}
    assert (d / "out" / "cells-registrations.jsonl").read_text() == ""
    assert (d / A.WITNESS / "cells.salts").read_text() == SALT
    assert json.loads((d / A.WITNESS / "load.json").read_text()) == LOAD_DOC
    assert {"load.json", "load-relay.json", "load-heads.json", "rehearse.log", "tunnel.log"} <= {
        p.name for p in (d / A.WITNESS).iterdir()} and "witness" not in end
    public = b"".join(p.read_bytes() for p in (d / "out").rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    assert not any(x.encode() in public for x in (SALT, ROOT, DIGEST, SIGNATURE))


def test_a_load_whose_sink_tunnel_doesnt_come_up_runs_nothing_and_its_end_says_why(stored):
    rc, out = gate(stored, LOAD)
    run = json.loads(out)["run"]
    ups = []
    end = A.certifier_run(stored, run, pinned=PIN, tunnel_argv=lambda s, l: TUNNEL,
                          sink_argv=lambda s, l, r: TUNNEL, up=lambda t, p: not ups.append(p) and len(ups) < 3,
                          command=lambda *a, **k: pytest.fail("ran without its sink's tunnels"))
    assert end["rc"] == 2 and end["ended"] == "tunnel" and LOAD[2] not in json.dumps(end)
    assert ended_why(stored, end) == "a tunnel to vy-nebius-1's sink at 127.0.0.1:8021 didn't come up (tunnel.log)"
    assert len(ups) == 3 and end["verdict"] is None and "load" not in end
    assert end["files"] == {} and "witness" not in end


def test_a_refused_load_publishes_nothing_of_its_relay_and_its_sentence_stays_in_the_witness(stored):
    """Of a refused load `out/` and the end hold nothing of the relay: its record (`load-relay.json`, the word it
    was refused in) and the relay's sentence, which may name a value, a path of this host's store or what the
    developer's sink said, stay in the witness, beside the check's `check.json`. The words a refused store's
    write publishes are the relay's (`LOAD_REFUSED`: `load_relay.REFUSALS` and `other`)."""
    assert A.LOAD_REFUSED == (*LR.REFUSALS, "other"), (
        "auditor.LOAD_REFUSED is the relay's words (load_relay.REFUSALS) and `other`: rule a word there first")
    said = "the sink acknowledged 1,024 bytes of the weights and 0 of the witness, of 34,240 and 6,144 sent"
    rc, out = gate(stored, LOAD)
    run = json.loads(out)["run"]
    files = ('(out / "load-relay.json").write_text(json.dumps({"refused": "sink"}))\n'
             f'(out / "check.json").write_text(json.dumps({{"accepted": False, "problems": [{"load: " + said!r}], '
             '"summary": {}}))\nraise SystemExit(1)\n')
    end = A.certifier_run(
        stored, run, pinned=PIN, tunnel_argv=lambda s, l: TUNNEL, sink_argv=lambda s, l, r: TUNNEL,
        up=lambda t, p: True, wait=lambda proc, s: proc.wait(),
        command=lambda req, work, url, **kw: [
            sys.executable, "-c", f"import json, pathlib; out = pathlib.Path({str(work)!r})\n{files}"])
    d = stored.attester / run
    assert (end["rc"], end["ended"]) == (1, "exited") and "load" not in end
    assert end["verdict"] == {"accepted": False} and end["files"] == {}
    assert json.loads((d / A.WITNESS / "load-relay.json").read_text()) == {"refused": "sink"}
    public = b"".join(p.read_bytes() for p in (d / "out").rglob("*") if p.is_file()) + (d / "end.json").read_bytes()
    assert b"refused" not in public and b"acknowledged" not in public and b"34,240" not in public
    assert said in (d / A.WITNESS / "check.json").read_text()


def test_a_loads_rehearsal_is_network_attester_serves_with_its_load_flags_after_the_rehearsals(tmp_path):
    req = {"device": "warden", "requests": 100, "load": {"shape": "tiny", "program": PROGRAM, "sink_port": 8021,
                                                         "streams": 2, "minutes": 16}}
    store = Path("/var/lib/vy-challenger/store")
    plain = A.network_attester_command({k: v for k, v in req.items() if k != "load"}, tmp_path, "http://127.0.0.1:40001")
    argv = A.network_attester_command(req, tmp_path, "http://127.0.0.1:40001", sinks=[("127.0.0.1", 40002), ("127.0.0.1", 40003)],
                            store=store)
    assert argv == [*plain, "--load-store", str(store), "--load-shape", "tiny", "--load-sinks",
                    "127.0.0.1:40002,127.0.0.1:40003", "--load-program", PROGRAM, "--load-ready-s", "900"]
    assert A.forward_argv(["-i", "KEY", "research@10.0.0.2"], 40002, 8021) == [
        *A.TUNNEL_SSH, "-L", "127.0.0.1:40002:127.0.0.1:8021", "-i", "KEY", "research@10.0.0.2"]
