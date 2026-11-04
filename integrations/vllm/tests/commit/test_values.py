"""`verity_vllm.commit.values` against the ZK server's value service, `core-firewall values` (``FC_CORE_FIREWALL``), run as a process
of its own on a state directory and a key file in a temporary directory that no verity_vllm code here opens:

* a stack commits and registers in turn, one session an epoch, each record counting one session: a second commit before the
  register is refused by the service, and by this side before it is sent, which an in-process stand-in for the service at the
  socket shows whatever the callers do (no service needed);
* a window's served rows, under ``FC_FIREWALL_VALUES``, commit through the service at the window's close, a session a window,
  each registered in turn: the record's root, domain, leaf count and schema are its registration's table, the verifier's own
  ``--registered`` port over the record (`served_rows.reads_port`) reads that table, every leaf's ``b ‖ c`` opens to the published
  root, the digests sent are the step trace's rows, and no salt is written;
* a statement staged from the service's ``b ‖ c`` (`UnitStager.stage`'s ``committed``), its ports' tables one session, registered,
  proves through the ZK server under the server's salts with none staged (`flock-circuit --zk --salt-refs`, ``FLOCK_CIRCUIT`` and
  ``FC_FIREWALL_LEAN``), its registration admitted on the record by the session loop holding the server's key; restaged on the
  sample from what was committed, nothing is committed again; staged under salts of its own, it is refused at admission and
  nothing of it reaches the verifier;
* `predict node-registration` under the switch commits every θ value in one session, its log passes nci-check's full check, and
  no `.salts` file is written;
* each call site that still draws its own salts refuses under the switch, naming itself, and node-registration's flags are held
  to the switch."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import re
import struct
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.core.protocols.certifier import served_rows as SRW
from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.circuits.boolean import BIT
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.frame_v3 import BitRowLeaf
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import class_statement as CS
from verity_vllm.commit import values as V
from verity_vllm.config import UsageError
from verity_vllm.program.predict import node_registration as NR

import tests.program.predict.test_node_registration as TNR
from tests.pipeline.test_serve_openai import private, served_run, served_statics
from tests.program.predict.test_node_registration import commit  # noqa: F401  -- fixture

CORE, FLOCK, LEAN = (os.environ.get(k) for k in ("FC_CORE_FIREWALL", "FLOCK_CIRCUIT", "FC_FIREWALL_LEAN"))
needs_service = pytest.mark.skipif(not (CORE and Path(CORE).is_file()),
                                   reason="FC_CORE_FIREWALL names no core-firewall build (the ZK server's value service)")
needs_prover = pytest.mark.skipif(not all(x and Path(x).is_file() for x in (FLOCK, LEAN)),
                                  reason="FLOCK_CIRCUIT and FC_FIREWALL_LEAN name no flock-circuit and flock-firewall builds")

#: the recursive sessions the ZK server's record holds before the service starts, each opened at L0 and its inner session
#: finished, as `verity_flock.firewall` writes them
SESSIONS = ("r0000", "r0001")
OPENED = "".join(f'{{"event": "open", "parts": ["L0"], "seq": {2 * k}, "session": "{s}", "stream": "value-salts"}}\n'
                 f'{{"event": "finish", "part": "inner", "seq": {2 * k + 1}, "session": "{s}"}}\n' for k, s in enumerate(SESSIONS))
UNIT = bind(definition("ValuesSampleUnit", 1, lambda C, a, b, w: [C.AND(a[k], b[k]) ^ C.AND(w[k], w[2047 - k]) for k in range(16)],
                       (("a", Array(16, BIT)), ("b", Array(16, BIT)), ("w", Array(2048, BIT))), Array(16, BIT), register=False))
ROWS = [("ab", V2, 32, ["a", "b"]), ("w", V2, 2048, ["w"])]
N = 5


class Service:
    """A running `core-firewall values` on ``root``'s state directory (its record seeded with `OPENED`) and key file."""

    def __init__(self, root: Path) -> None:
        self.state, self.key, errs = root / "firewall", root / "salt.key", root / "values.err"
        self.state.mkdir()
        (self.state / "record.jsonl").write_text(OPENED)
        with open(errs, "wb") as err:
            self.p = subprocess.Popen([CORE, "values", "--state", str(self.state), "--salt-key", str(self.key)],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        line = self.p.stdout.readline().decode()
        assert line.startswith("VALUES "), (line, errs.read_text())
        self.address = line.split()[1]

    def events(self, *kinds: str) -> list[dict]:
        return [e for e in map(json.loads, (self.state / "record.jsonl").read_text().splitlines()) if e["event"] in kinds]

    def close(self) -> None:
        self.p.stdin.close()
        rc = self.p.wait()
        self.p.stdout.close()
        assert rc == 0


@pytest.fixture
def service(tmp_path_factory, monkeypatch):
    s = Service(tmp_path_factory.mktemp("zk-server"))
    monkeypatch.setenv(V.ENV, s.address)
    yield s
    s.close()


def _no_salts(root: Path) -> None:
    assert not [p for p in root.rglob("*") if p.suffix == ".salts" or p.name == "salts"]


@pytest.fixture(autouse=True)
def _one_stack(monkeypatch):
    monkeypatch.setattr(V, "_pending", None)


TABLE = V.Table(hashlib.sha512(b"values/domain").digest(), V.SCHEMAS[1], b"".join(hashlib.sha512(bytes([i])).digest() for i in range(3)))


@needs_service
def test_a_stack_commits_and_registers_in_turn_one_session_an_epoch(service):
    t = TABLE
    a, b = V.ValueSession(), V.ValueSession()
    with pytest.raises(V.Refused, match="commit first"):
        a.register()
    (ca,) = a.commit([t])
    assert V.pending() is a and ca.n == 3 and len(ca.path(2)) == 2
    with pytest.raises(V.Refused, match="commits once"):
        a.commit([t])
    with pytest.raises(V.Refused, match="not yet registered: a stack commits once per registration"):
        b.commit([t])
    with pytest.raises(V.Refused, match="the current epoch's session is not yet registered"):
        V.ValueSession()._request(f"commit 1 {t.domain.hex()} {t.n} {t.schema}", [t.digests], V._line)   # past this side's guard
    ra = a.register()
    assert V.pending() is None and (ra.number, ra.sessions, ra.refs) == (0, 1, (ca.ref(),))
    with pytest.raises(V.Refused, match="registers once"):
        a.register()
    (cb,) = b.commit([t])
    assert cb.session != ca.session and cb.root != ca.root
    rb = b.register()
    assert (rb.number, rb.sessions, rb.refs) == (1, 1, (cb.ref(),))
    assert [(e["n"], e["record"]) for e in service.events("register")] == [(0, ra.record.hex()), (1, rb.record.hex())]
    assert V.Committed.of(cb.ref(), cb.bcs) == cb
    with pytest.raises(V.Refused, match="not the root"):
        V.Committed.of(cb.ref(), ca.bcs)


class Wire:
    """The value service of record at the socket, in process: each request's first word kept (``lines``), one session an epoch,
    and every commit that arrives while the epoch's session is unregistered counted (``overlaps``). ``refuse`` answers the next
    commit ``err``, opening nothing; ``cut`` ends the next commit's answer after its session line."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.open: tuple[bytes, list[bytes]] | None = None
        self.number = self.overlaps = 0
        self.refuse = self.cut = False

    def connect(self, _address):
        return _Conn(self)

    def answer(self, req: bytes) -> bytes:
        head, _, body = req.partition(b"\n")
        words = head.decode().split()
        self.lines.append(words[0])
        if words[0] == "register":
            if self.open is None or self.open[0].hex() != words[1]:
                return b"err the session is none the current epoch opened\n"
            (s, refs), self.open = self.open, None
            record = struct.pack(">QQI", self.number, 1, len(refs)) + b"".join(refs) + hashlib.sha512(b"tag" + s).digest()
            self.number += 1
            return f"ok 1 {len(record)}\n".encode() + record
        if self.open is not None:
            self.overlaps += 1
            return b"err the current epoch's session is not yet registered\n"
        if self.refuse:
            self.refuse = False
            return b"err a commit body of the wrong length\n"
        s = hashlib.sha512(b"session %d" % len(self.lines)).digest()
        out, refs, at = [f"ok {s.hex()}\n".encode()], [], 0
        for j, (d, n, schema) in enumerate(zip(words[2::3], words[3::3], words[4::3])):
            n, d = int(n), bytes.fromhex(d)
            bcs = [hashlib.sha512(s + body[at + 64 * i:at + 64 * (i + 1)]).digest() * 2 for i in range(n)]
            at += 64 * n
            root, tag = V.tree(d, schema, bcs)[-1][0], hashlib.sha512(b"tag" + s + bytes([j])).digest()
            refs.append(V.Committed(s, j, d, schema, n, root, tag).ref())
            out.append(f"ok {root.hex()} {tag.hex()} {n}\n".encode() + b"".join(bcs))
        self.open = (s, refs)
        if self.cut:
            self.cut = False
            return out[0]
        return b"".join(out)


class _Conn:
    def __init__(self, wire: Wire) -> None:
        self.wire, self.req, self.ans = wire, bytearray(), b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sendall(self, b: bytes) -> None:
        self.req += b

    def shutdown(self, _how) -> None:
        self.ans = self.wire.answer(bytes(self.req))

    def makefile(self, _mode):
        return io.BytesIO(self.ans)


def test_the_client_never_sends_a_second_commit_before_the_register(monkeypatch):
    """Whatever the callers do (another session, the same one again, eight threads at once, a commit the service refused or
    one cut short), what reaches the service is commit, register, commit, register: a second commit before the register is
    refused here and never sent, and a failed commit is never retried."""
    wire = Wire()
    monkeypatch.setattr(V.socket, "create_connection", wire.connect)
    monkeypatch.setenv(V.ENV, "127.0.0.1:1")
    a = V.ValueSession()
    a.commit([TABLE])
    for again in (V.ValueSession(), V.ValueSession(), a):
        with pytest.raises(V.Refused, match="not yet registered|commits once"):
            again.commit([TABLE])
    assert a.register().sessions == 1 and V.pending() is None

    racers, gate = [V.ValueSession() for _ in range(8)], threading.Barrier(8)

    def race(s):
        gate.wait()
        try:
            s.commit([TABLE])
        except V.Refused as e:
            return str(e)
        return None
    with ThreadPoolExecutor(8) as pool:
        why = list(pool.map(race, racers))
    (won,) = [s for s, w in zip(racers, why) if w is None]
    assert all("not yet registered" in w for w in why if w is not None) and V.pending() is won
    won.register()

    wire.refuse = True
    refused = V.ValueSession()
    with pytest.raises(V.Refused, match="the value service refused: a commit body"):
        refused.commit([TABLE])
    assert V.pending() is None
    with pytest.raises(V.Refused, match="commits once: a failed commit is its caller's failure"):
        refused.commit([TABLE])
    nxt = V.ValueSession()
    nxt.commit([TABLE])
    nxt.register()

    wire.cut = True
    cut = V.ValueSession()
    with pytest.raises(V.Refused, match="ended"):
        cut.commit([TABLE])
    assert V.pending() is cut and cut.id is None
    with pytest.raises(V.Refused, match="not yet registered"):
        V.ValueSession().commit([TABLE])
    with pytest.raises(V.Refused, match="commit first"):
        cut.register()
    assert wire.lines == ["commit", "register", "commit", "register", "commit", "commit", "register", "commit"]
    assert wire.overlaps == 0 and wire.number == 3


@needs_service
def test_a_windows_served_rows_commit_through_the_service_as_the_verifier_reads_them(service, tmp_path, monkeypatch):
    sent: list = []
    commit_ = V.ValueSession.commit

    def spy(self, tables, sink=None):
        sent.append(list(tables))
        return commit_(self, tables, sink)
    monkeypatch.setattr(V.ValueSession, "commit", spy)
    statics = served_statics()
    wins, reqs, trace = served_run(tmp_path, statics)
    full = [w for w in wins if not w.get("empty")]
    assert len(full) > 1 and all(w["passed"] for w in wins), [w.get("served_rows") for w in wins]
    regs = service.events("register")
    assert [e["n"] for e in regs] == list(range(len(full))) and len(sent) == len(full)
    arrival = {rid: r["arrival"] for r in reqs for rid in r["runner_ids"]}
    S = SimpleNamespace(**statics)
    for k, w in enumerate(full):
        rec, priv, run_root = w["served_rows"], private(tmp_path, w["window"]), bytes.fromhex(w["run_root"])
        assert "salts" not in priv and priv["values"]["table"] == rec["values"]["table"]
        table = V.Committed.of(bytes.fromhex(rec["values"]["table"]), [bytes.fromhex(x) for x in priv["values"]["bcs"]])
        assert rec["values"]["registration"] == {"number": k, "sessions": 1, "record": regs[k]["record"]}
        assert bytes.fromhex(regs[k]["record"])[V.HEAD:V.HEAD + V.REF] == table.ref()
        port = SRW.reads_port(rec, run_root, list(range(table.n)))
        assert (port["root"], port["domain"], port["leaves"], port[REG.SCHEMA_KEY]) == (table.root.hex(), table.domain.hex(), table.n,
                                                                                       table.schema)
        assert all(SRW.leaf_ok(rec, run_root, i, table.bcs[i], table.path(i)) for i in range(table.n))
        assert not SRW.leaf_ok(rec, run_root, 0, bytes([table.bcs[0][0] ^ 1]) + table.bcs[0][1:], table.path(0))
        by_step = {t["step"]: [tuple(r) for r in t["rows"]] for t in trace if t["window"] == w["window"]}
        ids = {rid: arrival[rid] for rows in by_step.values() for rid, *_ in rows}
        digest = BitRowLeaf(rec["n_bits"]).layout().digest
        rows = [SRW._row(SRW.block_bits(S, win, S.P, 0))[0] for win in SRW.windows(S, by_step, ids, w["steps"])]
        (t,) = sent[k]
        assert (t.domain, t.schema, t.digests) == (table.domain, table.schema, b"".join(digest(r) for r in rows))
    _no_salts(tmp_path)


def _session(service: Service, d: Path, session: str, *extra: str) -> dict:
    """One firewalled CPU session of the statement staged in ``d``: an outer session at L0 of the recursive session ``session``
    on the ZK server's record, through the session loop holding the server's key, so that it admits the statement's
    registrations; its record (`flock-circuit`'s ``NEG`` line)."""
    env = {**os.environ, "FC_FIREWALL": "1", "FC_HOST_WITNESS": "1", "FC_FIREWALL_SALT_KEY": str(service.key)}
    r = subprocess.run([FLOCK, "selftest", "--zk", "--instances", str(d / "inst.bin"), "--circuit", str(d / "circuit.txt"),
                        "--tables", str(d / "tables"), "--only", "zk_worker_session", "--firewall-state", str(service.state),
                        "--session", session, "--part", "L0", *extra], capture_output=True, text=True, env=env, check=False)
    (d / "session.log").write_text(r.stdout + r.stderr)
    neg = [json.loads(ln.split("\t", 1)[1]) for ln in r.stdout.splitlines() if ln.startswith("NEG\t")]
    assert len(neg) == 1, r.stdout[-3000:] + r.stderr[-3000:]
    return neg[0]


@pytest.mark.slow
@needs_service
@needs_prover
def test_a_sampled_block_proves_under_the_servers_salts_admitted_on_its_record(service, tmp_path):
    """Two firewalled CPU sessions of a five-instance unit through the ZK server (its session loop, release side and value
    service, as `flock-circuit`'s `zk_value_salts` runs them, on a statement verity_vllm staged and committed). What it catches
    that a cheaper test can't: a staging whose roots the server's admission doesn't hold, or whose rows the server's salts
    don't open, which only a proof through the server shows."""
    st = CS.UnitStager(UNIT, "unit/values-sample", ROWS, [(V2, (0,))])
    rng = np.random.default_rng(3)
    rows = {"ab": rng.integers(0, 1 << 16, (N, 2), dtype=np.uint16), "w": rng.integers(0, 1 << 16, (N, 128), dtype=np.uint16)}
    s = V.ValueSession()
    staged = st.stage(rows, tmp_path / "committed", None, committed=s.bcs)
    reg = s.register()
    assert staged["salts"] is None and (reg.number, reg.sessions, len(reg.refs)) == (0, 1, len(st.ports))
    st.stage(rows, tmp_path / "sample", None, committed=V.stored(s.tables))
    for f in ("inst.bin", "pub.bin"):
        assert (tmp_path / "sample" / f).read_bytes() == (tmp_path / "committed" / f).read_bytes(), f
    (tmp_path / "refs.bin").write_bytes(V.refs_file(s.tables, V.instance_rows(N, len(st.ports))))
    st.stage(rows, tmp_path / "own", b"k" * 32)
    over = _session(service, tmp_path / "committed", "r0000", "--salt-refs", str(tmp_path / "refs.bin"))
    own = _session(service, tmp_path / "own", "r0001")
    proofs = sum(x[0] == "proof" for x in over["seen"])
    assert over["accepted"] is True and proofs > 0, {k: over.get(k) for k in ("accepted", "why", "prover_stopped", "error")}
    assert own["accepted"] is False and not own["seen"] and "no value commitment of the ZK server's" in (own["why"] or ""), own
    assert [(e["event"], e["n"], e["session"]) for e in service.events("admit", "skip")] == [("admit", 0, "r0000")]
    assert len(service.events("register")) == 1
    _no_salts(tmp_path)


@needs_service
def test_node_registration_commits_theta_in_one_session_and_writes_no_salts(service, commit, tmp_path):
    o = NR.Options(**{**commit["o"].__dict__, "salts": None, "bcs": str(tmp_path / "private" / "bcs"),
                      "checkpoints_manifest": TNR._manifest(commit)})
    assert NR.main(o) == 0
    log, bcs = Path(o.out), Path(o.bcs)
    rec, trusted, values = (json.loads((log / f"{n}.json").read_text()) for n in ("record", "trusted", "values"))
    (reg,) = service.events("register")
    assert values["registration"] == {"number": 0, "sessions": 1, "record": reg["record"]}
    record, by = bytes.fromhex(reg["record"]), {e["value"]: e for e in rec["registered"]}
    assert sorted(by) == sorted(values["tables"]) and [e["value"] for e in rec["registered"]] == list(TNR.NODE.theta)
    for j, (value, ref) in enumerate(values["tables"].items()):
        assert record[V.HEAD + j * V.REF:V.HEAD + (j + 1) * V.REF].hex() == ref
        raw = (bcs / f"{value}.bc").read_bytes()
        t = V.Committed.of(bytes.fromhex(ref), [raw[i:i + V.BC] for i in range(0, len(raw), V.BC)])
        e = by[value]
        assert (t.root.hex(), t.domain.hex(), t.n, t.schema) == (e["root"], e["domain"], e["leaves"], REG.SCHEMA), value
    inst, _computed = TNR._instance(json.loads((log / "provenance.json").read_text()))
    assert TNR.REF.check_baseline(inst, trusted)["complies"] is True
    assert sorted(p.name for p in bcs.iterdir()) == sorted(f"{v}.bc" for v in values["tables"])
    _no_salts(tmp_path)


def _at(module: str):
    return importlib.import_module(f"verity_vllm.{module}")


#: each call site that draws its own salts for a public commitment, by the name it refuses under, and a call that reaches it
REFUSING = {
    "pouw_h1.registered.salts": lambda: _at("protocols.pouw.h1.registered").salts(b"k" * 32, 0, 1),
    "pouw_h1.registered.commit": lambda: _at("protocols.pouw.h1.registered").commit("p", "v", np.zeros((1, 32), np.uint16), b"k" * 32, 0),
    "pouw_h1.requests.hiding_rows": lambda: _at("protocols.pouw.h1.requests").hiding_rows(bytes(64), 0, [b"x"], b"k" * 32, 0),
    "the ranks' H1 salt key": lambda: _at("protocols.pouw.h1.ranks").install(None, rows=1, steps=1, coins=object()),
    "window_rows.index_salt": lambda: _at("program.predict.window_rows").index_salt(b"k" * 32, 0),
    "rows-sha512 step 3's salt key": lambda: _at("commit.native_ranges").NativeRangeOpenings._row_key(None, 3),
    "rows-sha512's weights registration salt key": lambda: _at("commit.committer.weights_rows").key(None),
    "lead_rows.WindowKeying's ks_w and y0": lambda: _at("commit.lead_rows").WindowKeying(1),
    "remote_bandwidth.StoreBackend's hiding salts": lambda: _at("protocols.remote_bandwidth.remote").StoreBackend(None, None),
    "advice_windows.commit_advice": lambda: _at("pipeline.advice_windows").commit_advice(None, b""),
    "registration_log.write": lambda: _at("pipeline.registration_log").write("p", [], [], None, {}),
    "pearl_c.new_salt_key": lambda: _at("protocols.pouw.pearl_c").new_salt_key("x"),
}


@pytest.mark.parametrize("site", list(REFUSING))
def test_a_site_that_draws_its_own_salts_refuses_under_the_switch(site, monkeypatch):
    monkeypatch.setenv(V.ENV, "127.0.0.1:9")
    with pytest.raises(V.Refused, match=re.escape(site) + ".*the ZK server chooses every public value commitment's salts"):
        REFUSING[site]()


def test_node_registrations_flags_are_held_to_the_switch(commit, tmp_path, monkeypatch):
    o = commit["o"].__dict__
    monkeypatch.setenv(V.ENV, "127.0.0.1:9")
    for given in ({}, {"salts": None}, {"bcs": str(tmp_path / "bcs")}):
        with pytest.raises(UsageError, match="no --salts, and --bcs for the b [|][|] c it answers"):
            NR.register(NR.Options(**{**o, **given}))
    with pytest.raises(UsageError, match="the unsalted row digests are staged there"):
        NR.register(NR.Options(**{**o, "salts": None, "bcs": str(Path(o["out"]) / "bcs")}))
    monkeypatch.delenv(V.ENV)
    for given in ({"salts": None}, {"bcs": str(tmp_path / "bcs")}):
        with pytest.raises(UsageError, match="--salts is required, and --bcs is the value service's"):
            NR.register(NR.Options(**{**o, **given}))
    _no_salts(tmp_path)
