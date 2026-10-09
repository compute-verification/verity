"""`lean_audit.Audit`, `audit.Audit` with every verdict the Lean check's (`network-records verdicts`), against `audit.Audit`
itself: the same verdicts, clock, `accepted` and `violations` on `test_network_records_protocol`'s generated units, raw,
committed, certified and mixed, with each egress link checked over a prefix of the windows of its own, so the unit's clock
comes from several links; on `service.py`'s calls; and a refusal, never a Python verdict, outside the Lean's domain."""

from __future__ import annotations

import dataclasses
import shutil
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


def test_service_verify_is_the_same_under_the_adapter(lean, tmp_path, monkeypatch):
    """`LinkWindows.verify`, with `service.Audit` the adapter, gives `audit.Audit`'s verdict of record on six link-windows
    in order: honest ones, a forged signature, a grid other than the Program's, and an unset deadline bit."""
    W, g = 6, GridParams(3, 2, 2)
    ss = SyncSet((0, 1), (Fraction(1),))
    syncs = [ClockSync(w % 2, Fraction(1)) for w in range(W)]
    frames = [LogicalFrame(bytes([i, i]), 2 * i, 0) for i in range(8)]
    signer = LmsSigner.create(lms.keygen(lms.Params(8, 8, 4, toy=True), bytes(range(8))), tmp_path / "signer-state.json")
    issuer = Issuer(signer)
    lw = service.LinkWindows("responses", g, ss, signer.public_key(), "r-lean-audit-test", proof_deadline=60.0)
    stream = lw.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test", coins=lambda n: b"\x07" * n,
                       clock=lambda: 0.0)
    egress = run_shaper(frames, [0] * len(frames), syncs, Certifier("responses", g))
    ingress, ins = Certifier("responses", g), []
    for _ in range(W):
        for _ in range(g.buckets):
            ingress.tick()
        ins.append(ingress.close_window())
    previous, accepted = None, []
    for w in range(W):
        record = egress[w]
        if w % 3 == 2:
            record = dataclasses.replace(record, grid=((bytes([9, 9]), None),) + record.grid[1:])
        e = issuer.certify(record, Direction.EGRESS, g, parameters_digest(g, ss), deadline=w != 3)
        if w % 3 == 1:
            e = CertifiedRecord(e.head, dataclasses.replace(e.certificate, signature=b""))
        i = issuer.certify(ins[w], Direction.INGRESS, g, parameters_digest(g))
        opening = lw.commit(w, e, i, salts=(bytes([2 * w + 1]) * 192, bytes([2 * w + 2]) * 192))
        reg = lw.register(stream, lw.registration(opening, previous))
        previous = reg.digest
        draw = stream.select(reg)
        assert stream.deliver(draw, [lw.session(opening, draw)]) is None
        got = {}
        for name, impl in (("python", audit.Audit), ("lean", lambda: lean_audit.Audit(server=lean))):
            monkeypatch.setattr(service, "Audit", impl)
            got[name] = lw.verify(opening, draw, frames, syncs)
        assert got["lean"] == got["python"]
        accepted.append(got["python"]["accepted"])
        assert lw.outcome(stream, draw, opening, frames, syncs).accepted == accepted[-1]
    assert accepted == [True, False, False, False, False, False]


def test_outside_the_domain_it_refuses(lean, tmp_path):
    """A certified record under a key the Lean can't verify, a certified record with no key, a raw grid that isn't tuples,
    a frame of other than B bytes and a missing executable each raise; none gets a verdict computed in Python, and the
    audit keeps no verdict or sync of a call that raised."""
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
    assert a.verdicts == [audit.Verdict("b", 0, "no-verdict")]
