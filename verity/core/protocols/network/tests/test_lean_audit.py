"""`lean_audit.Audit`, `audit.Audit` with every verdict the Lean check's (`network-records verdicts`), against `audit.Audit`
itself: the same verdicts, clock, `accepted` and `violations` on `test_network_records_protocol`'s generated units, raw,
committed, certified and mixed, with each egress link checked over a prefix of the windows of its own, so the unit's clock
comes from several links; on `service.py`'s calls; and a refusal, never a Python verdict, outside the Lean's domain."""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
import warnings
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import test_network_check_protocol as V2
import test_network_records_protocol as R
import verity_difftest as D
from verity_difftest import protocol as P
from verity.ml.crypto import lms
from verity.core.protocols.certifier import audit, lean_audit, service
from verity.core.protocols.certifier.certificate import CertifiedRecord, Direction, Issuer, LmsSigner, parameters_digest
from verity.core.protocols.certifier.commitment import commit
from verity.core.protocols.certifier.grid import Certifier, GridParams
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def lean():
    s = lean_audit.Server(D.build(V2.LEAN, "network-records"))
    yield s
    s.close()


def _decoded(case):
    """The unit, records and evidence a well-formed case decodes to, or None."""
    try:
        return V2.params(case["params"]), R.records(P.unhex(case["commitment"]))[0], V2.evidence(P.unhex(case["evidence"]))
    except (D.Reject, KeyError, TypeError):
        return None


def _run(a, u, recs, e, prefixes, keyless):
    out = []
    for (link, (frames, syncs)), k in zip(zip(u.egress, e), prefixes):
        if keyless:
            out += a.check_link(link, recs, frames, syncs[:k], u.sync_set, u.grid)
        else:
            out += a.check_link(link, recs, frames, syncs[:k], u.sync_set, u.grid, u.key)
    for link in u.ingress:
        out += (a.check_ingress_link(link, recs, u.windows) if keyless
                else a.check_ingress_link(link, recs, u.windows, u.key, u.ingress_grid))
    return out


def _same(lean, u, recs, e, prefixes, keyless=False):
    ref, port = audit.Audit(), lean_audit.Audit(server=lean)
    assert _run(port, u, recs, e, prefixes, keyless) == _run(ref, u, recs, e, prefixes, keyless)
    assert (port.verdicts, port.syncs, port.accepted, port.violations) == \
        (ref.verdicts, ref.syncs, ref.accepted, ref.violations)
    assert port.lean == port.reference == ref.verdicts
    return ref.verdicts


@pytest.mark.slow
def test_the_adapter_gives_the_audits_verdicts_on_generated_units(lean):
    """On generated units of every kind and change: identical verdict lists, with each egress link checked over a window
    prefix of its own and, when no record is certified, with no key and no ingress parameters (`active_replay.py`'s
    calls); and a `ValueError` where the Lean check refuses the evidence (a frame of other than B bytes)."""
    kinds = {}

    @settings(derandomize=True, database=None, deadline=None, max_examples=600)
    @given(R.record_cases(), st.data())
    def same(case, data):
        d = _decoded(case)
        if d is None:
            return
        u, recs, e = d
        if len(e) != len(u.egress) or any(len(s) != u.windows for _, s in e):
            return
        prefixes = [data.draw(st.integers(0, u.windows)) for _ in u.egress]
        if any(len(f.payload) != u.grid.frame_bytes for frames, _ in e for f in frames):
            with pytest.raises(ValueError, match="outside the Lean audit's domain"):
                _run(lean_audit.Audit(server=lean), u, recs, e, prefixes, False)
            kinds["refused"] = kinds.get("refused", 0) + 1
            return
        vs = _same(lean, u, recs, e, prefixes)
        if not any(isinstance(r, CertifiedRecord) for r in recs):
            _same(lean, u, recs, e, prefixes, keyless=True)
        for v in vs:
            kinds[v.violation or "accept"] = kinds.get(v.violation or "accept", 0) + 1

    same()
    print(kinds)
    assert {"accept", "no-verdict", "duplicate", "signature", "advice", "mismatch"} <= set(kinds), kinds
    assert any(k.startswith("status:") for k in kinds), kinds


def test_the_clock_is_the_first_link_checked_in_each_window(lean):
    """Links over 2, 4 and 3 windows: each window's sync is the first link's that checked it, as in `audit.Audit`."""
    g = GridParams(2, 2, 1)
    ss = SyncSet((0, 1, 2), (Fraction(1), Fraction(1, 2)))
    s = [ClockSync(d, Fraction(1)) for d in (0, 1, 2, 0)]
    frames = [LogicalFrame(bytes([i]), i, 0) for i in range(6)]
    recs = {n: run_shaper(frames, [0] * 6, sy, Certifier(n, g)) for n, sy in
            (("a", s[:2]), ("b", s[:4]), ("c", [s[0], s[2], s[3]]))}
    for kind in ("raw", "committed"):
        published = [r if kind == "raw" else commit(r, g) for rs in recs.values() for r in rs]
        ref, port = audit.Audit(), lean_audit.Audit(server=lean)
        for a in (ref, port):
            a.check_link("a", published, frames, s[:2], ss, g)
            a.check_link("b", published, frames, s[:4], ss, g)
            a.check_link("c", published, frames, [s[0], s[2], s[3]], ss, g)
            a.check_link("", published, frames, [], ss, g)
        assert port.verdicts == ref.verdicts and port.syncs == ref.syncs
        assert [v.violation for v in ref.verdicts if v.link == "c"] == [None, "advice", "advice"]


SERVICE_FRAMES = [LogicalFrame(bytes([i, i]), 2 * i, 0) for i in range(8)]


def _service_rounds(tmp_path, W, record=lambda w, r: r, certified=lambda w, e: e):
    """`service.py`'s rounds on link "responses": each window's (link-windows, stream, opening, draw, syncs), the egress
    record passed through `record(w, r)`, certified, then through `certified(w, e)`."""
    g, ss = GridParams(3, 2, 2), SyncSet((0, 1), (Fraction(1),))
    syncs = [ClockSync(w % 2, Fraction(1)) for w in range(W)]
    signer = LmsSigner.create(lms.keygen(lms.Params(8, 8, 4, toy=True), bytes(range(8))), tmp_path / "signer-state.json")
    issuer = Issuer(signer)
    lw = service.LinkWindows("responses", g, ss, signer.public_key(), "r-lean-audit-test", proof_deadline=60.0)
    stream = lw.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test", coins=lambda n: b"\x07" * n,
                       clock=lambda: 0.0)
    egress = run_shaper(SERVICE_FRAMES, [0] * len(SERVICE_FRAMES), syncs, Certifier("responses", g))
    ingress, ins = Certifier("responses", g), []
    for _ in range(W):
        for _ in range(g.buckets):
            ingress.tick()
        ins.append(ingress.close_window())
    previous = None
    for w in range(W):
        e = certified(w, issuer.certify(record(w, egress[w]), Direction.EGRESS, g, parameters_digest(g, ss),
                                        deadline=w != 3))
        i = issuer.certify(ins[w], Direction.INGRESS, g, parameters_digest(g))
        opening = lw.commit(w, e, i, salts=(bytes([2 * w + 1]) * 192, bytes([2 * w + 2]) * 192))
        reg = lw.register(stream, lw.registration(opening, previous))
        previous = reg.digest
        draw = stream.select(reg)
        assert stream.deliver(draw, [lw.session(opening, draw)]) is None
        yield lw, stream, opening, draw, syncs


def test_service_verify_is_the_same_under_the_adapter(lean, tmp_path, monkeypatch):
    """`LinkWindows.verify`, with `service.Audit` the adapter, gives `audit.Audit`'s verdict of record on six link-windows
    in order: honest ones, a forged signature, a grid other than the Program's, and an unset deadline bit; the adapter's
    says `by: lean` and carries the reference's accept bit."""
    def other_grid(w, r):
        return dataclasses.replace(r, grid=((bytes([9, 9]), None),) + r.grid[1:]) if w % 3 == 2 else r

    def forged(w, e):
        return CertifiedRecord(e.head, dataclasses.replace(e.certificate, signature=b"")) if w % 3 == 1 else e

    accepted = []
    for lw, stream, opening, draw, syncs in _service_rounds(tmp_path, 6, other_grid, forged):
        got = {}
        for name, impl in (("python", audit.Audit), ("lean", lambda: lean_audit.Audit(server=lean))):
            monkeypatch.setattr(service, "Audit", impl)
            got[name] = lw.verify(opening, draw, SERVICE_FRAMES, syncs)
        assert got["lean"] == got["python"] | {"by": "lean", "reference": {"accepted": got["python"]["accepted"]}}
        accepted.append(got["python"]["accepted"])
        assert lw.outcome(stream, draw, opening, SERVICE_FRAMES, syncs).accepted == accepted[-1]
    assert accepted == [True, False, False, False, False, False]


def test_a_disagreement_with_the_reference_rejects(lean, tmp_path, monkeypatch):
    """With the Python reference made to reject one link-window the Lean check accepts (and, on ingress, the reverse
    direction of the same flip), that link-window is a `disagreement`: the audit rejects, `lean` keeps the Lean's
    verdicts and `reference` the Python's, and `service.py`'s verdict of record rejects an honest link-window while
    carrying the reference's bit beside `by: lean`. A reference that raises where the Lean answers raises."""
    real_egress, real_ingress = audit.Audit.check_link, audit.Audit.check_ingress_link

    def flip(vs, at):
        return [audit.Verdict(v.link, v.window, "mismatch" if v.accepted else None) if v.window == at else v for v in vs]

    monkeypatch.setattr(audit.Audit, "check_link", lambda self, *a, **k: flip(real_egress(self, *a, **k), 0))
    monkeypatch.setattr(service, "Audit", lambda: lean_audit.Audit(server=lean))
    lw, stream, opening, draw, syncs = next(_service_rounds(tmp_path, 1))
    a = lean_audit.Audit(server=lean)
    vs = a.check_link("responses", [opening.records[0]], SERVICE_FRAMES, syncs[:1], lw.sync_set, lw.params, key=lw.key)
    assert vs == [audit.Verdict("responses", 0, "disagreement")] and not a.accepted and a.violations == vs
    assert a.lean == [audit.Verdict("responses", 0, None)] and a.reference == [audit.Verdict("responses", 0, "mismatch")]
    verdict = lw.verify(opening, draw, SERVICE_FRAMES, syncs)
    assert verdict["accepted"] is False and verdict["public"]["verdict"] == 0
    assert verdict["by"] == "lean" and verdict["reference"] == {"accepted": False}
    assert not lw.outcome(stream, draw, opening, SERVICE_FRAMES, syncs).accepted

    monkeypatch.setattr(audit.Audit, "check_link", real_egress)
    monkeypatch.setattr(audit.Audit, "check_ingress_link", lambda self, *a, **k: flip(real_ingress(self, *a, **k), 0))
    a = lean_audit.Audit(server=lean)
    assert a.check_ingress_link("responses", [], 1) == [audit.Verdict("responses", 0, "disagreement")]
    assert a.lean == [audit.Verdict("responses", 0, "no-verdict")] and a.reference == [audit.Verdict("responses", 0, None)]

    def boom(self, *a, **k):
        raise ArithmeticError("reference")

    monkeypatch.setattr(audit.Audit, "check_ingress_link", boom)
    a = lean_audit.Audit(server=lean)
    with pytest.raises(RuntimeError, match="the reference audit raised where the Lean check answered"):
        a.check_ingress_link("responses", [], 1)
    assert a.verdicts == a.lean == a.reference == []


def test_outside_the_domain_it_refuses(lean, tmp_path):
    """A certified record under a key the Lean can't verify, a certified record with no key, a raw grid that isn't tuples,
    a frame of other than B bytes, a missing executable and no `lake` to build one each raise; none gets a verdict computed
    in Python, and the audit keeps no verdict or sync of a call that raised."""
    g = GridParams(1, 1, 1)
    ss = SyncSet((0,), (Fraction(1),))
    sync = [ClockSync(0, Fraction(1))]
    rec = run_shaper([], [], sync, Certifier("a", g))[0]
    certified = Issuer(V2._Signer(V2._key(4))).certify(rec, Direction.EGRESS, g, parameters_digest(g, ss))

    class OtherKey:
        key_id = certified.certificate.key_id

        def verify(self, message, signature):
            return True

    a = lean_audit.Audit(server=lean)
    with pytest.raises(ValueError, match="not an LmsKey"):
        a.check_link("a", [certified], [], sync, ss, g, OtherKey())
    assert a.check_link("b", [certified], [], sync, ss, g, OtherKey()) == [audit.Verdict("b", 0, "no-verdict")]
    with pytest.raises(ValueError, match="pass key"):
        a.check_link("a", [certified], [], sync, ss, g)
    with pytest.raises(ValueError, match="pass params"):
        a.check_ingress_link("a", [dataclasses.replace(certified, head=dataclasses.replace(
            certified.head, direction=Direction.INGRESS))], 1, OtherKey())
    with pytest.raises(ValueError, match="tuple of tuples"):
        a.check_link("a", [dataclasses.replace(rec, grid=list(rec.grid))], [], sync, ss, g)
    with pytest.raises(ValueError, match="a frame of other than 1 bytes"):
        a.check_link("a", [rec], [LogicalFrame(b"xy", 0, 0)], sync, ss, g)
    with pytest.raises(FileNotFoundError, match="lake build network-records"):
        lean_audit.Audit(server=lean_audit.Server(tmp_path / "missing")).check_link("a", [rec], [], sync, ss, g)
    with pytest.MonkeyPatch.context() as m:
        m.delenv("VERITY_NETWORK_RECORDS_EXE", raising=False)
        m.setenv("PATH", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="lake is not on the PATH.*set VERITY_NETWORK_RECORDS_EXE"):
            lean_audit.Audit(server=lean_audit.Server()).check_link("a", [rec], [], sync, ss, g)
    assert a.verdicts == [audit.Verdict("b", 0, "no-verdict")]


def test_with_the_executable_named_it_needs_nothing_else(lean, tmp_path):
    """With `VERITY_NETWORK_RECORDS_EXE` set, a fresh interpreter with no `lake` on its PATH and `verity_difftest`
    unimportable gets the Lean check's verdicts, and its `SERVER` is closed at exit (seen by an exit handler registered
    before the import, which runs after `SERVER.close`)."""
    script = (
        "import atexit, sys; sys.modules['verity_difftest'] = None\n"
        "atexit.register(lambda: print('closed', sys.modules['verity.core.protocols.certifier.lean_audit']"
        ".SERVER._proc is None))\n"
        "from fractions import Fraction\n"
        "from verity.core.protocols.certifier import lean_audit\n"
        "from verity.core.protocols.certifier.grid import Certifier, GridParams\n"
        "from verity.core.protocols.certifier.schedule import ClockSync, SyncSet, run_shaper\n"
        "g, s = GridParams(1, 1, 1), [ClockSync(0, Fraction(1))]\n"
        "a = lean_audit.Audit()\n"
        "print(a.check_link('a', run_shaper([], [], s, Certifier('a', g)), [], s, SyncSet((0,), (Fraction(1),)), g))\n"
        "assert lean_audit.SERVER._proc is not None\n")
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       env={"PATH": str(tmp_path), "VERITY_NETWORK_RECORDS_EXE": str(lean.exe)})
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines() == ["[Verdict(link='a', window=0, violation=None)]", "closed True"]


def test_a_forked_process_starts_its_own_executable(lean):
    """A process forked after the parent's server started gets its own executable at its first request, and the parent's
    keeps answering on its own pipe."""
    g, ss, s = GridParams(1, 1, 1), SyncSet((0,), (Fraction(1),)), [ClockSync(0, Fraction(1))]
    rec = run_shaper([], [], s, Certifier("a", g))
    honest = [audit.Verdict("a", 0, None)]
    assert lean_audit.Audit(server=lean).check_link("a", rec, [], s, ss, g) == honest
    parent = lean._proc.pid
    r, w = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        try:
            ok = lean_audit.Audit(server=lean).check_link("a", rec, [], s, ss, g) == honest
            os.write(w, b"%d %d" % (ok, lean._proc.pid))
            lean.close()
        finally:
            os._exit(0)
    os.close(w)
    with os.fdopen(r) as fh:
        ok, child = map(int, fh.read().split())
    assert os.waitpid(pid, 0)[1] == 0
    assert ok == 1 and child != parent
    assert lean_audit.Audit(server=lean).check_link("a", rec, [], s, ss, g) == honest and lean._proc.pid == parent
