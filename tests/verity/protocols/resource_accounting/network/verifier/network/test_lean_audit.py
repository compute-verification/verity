"""`lean_audit.Audit`, `audit.Audit` with every verdict the Lean check's (`network-records verdicts`), against `audit.Audit`
itself: the same verdicts, clock, `accepted` and `violations` on `test_network_records_protocol`'s generated units, raw,
committed, attested and mixed, with each egress link checked over a prefix of the windows of its own, so the unit's clock
comes from several links; on attested records without a grid; and a refusal, never a Python verdict, outside the Lean's
domain. The client's attestation refusals (`certificate_refusals`) are their reference's
(`audit.certificate_refusals`) on crafted and generated inputs, a disagreement refusing, and outside the Lean's domain they raise."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
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
from verity.commitments.crypto import lms
from verity.protocols.resource_accounting.network.verifier import audit, lean_audit
from verity.protocols.resource_accounting.network.verifier.certificate import (CertifiedRecord, Direction, Issuer, LmsKey,
                                                        LmsSigner, parameters_digest)
from verity.protocols.resource_accounting.network.verifier.commitment import commit
from verity.protocols.resource_accounting.network.verifier.grid import Certifier, GridParams
from verity.protocols.resource_accounting.network.verifier.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

#: the changes of an attested record that leave its window without a verdict (`V2.EXPECT`)
UNDECIDED = tuple(m for m, v in V2.EXPECT.items() if v == "no-verdict")


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


def _run(a, u, recs, e, prefixes, keyless, grid=True):
    out = []
    for (link, (frames, syncs)), k in zip(zip(u.egress, e), prefixes):
        if keyless:
            out += a.check_link(link, recs, frames, syncs[:k], u.sync_set, u.grid, grid=grid)
        else:
            out += a.check_link(link, recs, frames, syncs[:k], u.sync_set, u.grid, u.key, u.session, grid)
    for link in u.ingress:
        out += (a.check_ingress_link(link, recs, u.windows) if keyless
                else a.check_ingress_link(link, recs, u.windows, u.key, u.ingress_grid, u.session))
    return out


def _same(lean, u, recs, e, prefixes, keyless=False, grid=True):
    ref, port = audit.Audit(), lean_audit.Audit(server=lean)
    assert _run(port, u, recs, e, prefixes, keyless, grid) == _run(ref, u, recs, e, prefixes, keyless, grid)
    assert (port.verdicts, port.syncs, port.accepted, port.violations) == \
        (ref.verdicts, ref.syncs, ref.accepted, ref.violations)
    assert port.lean == port.reference == ref.verdicts
    return ref.verdicts


@pytest.mark.slow
def test_the_adapter_gives_the_audits_verdicts_on_generated_units(lean):
    """On generated units of every kind and change: identical verdict lists, with each egress link checked over a window
    prefix of its own and, when no record is attested, with no key and no ingress parameters; and a `ValueError` where
    the Lean check refuses the evidence (a frame of other than B bytes). With `grid` False, the same on units whose
    egress records are all attested, with no egress `mismatch` (an ingress one stays), and a `ValueError` from both
    where a raw or committed record names an egress link. Then 20 attested units under the changes that leave a window
    undecided, checked over every window (about a minute more): the only cases that give a grid-free `no-verdict`
    whatever else the suite imports, which no cheaper test of the adapter compares."""
    kinds, no_grid = {}, {}

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
        if any(not isinstance(r, CertifiedRecord) and r.link in u.egress for r in recs):
            for a in (audit.Audit(), lean_audit.Audit(server=lean)):
                with pytest.raises(ValueError, match="only a certified record's grid is held by its head check"):
                    _run(a, u, recs, e, prefixes, False, grid=False)
            no_grid["refused"] = no_grid.get("refused", 0) + 1
            return
        for v in _same(lean, u, recs, e, prefixes, grid=False)[:sum(prefixes)]:
            no_grid[v.violation or "accept"] = no_grid.get(v.violation or "accept", 0) + 1

    same()

    # Hypothesis mixes the literals of every imported local module into what it samples, so the 600 cases above move with
    # unrelated code; a grid-free no-verdict is among them only a few times, and attested units under the changes that
    # leave a window undecided, checked over every window, carry it whatever else is imported.
    @settings(derandomize=True, database=None, deadline=None, max_examples=20)
    @given(R.record_cases(("certified",), UNDECIDED))
    def undecided(case):
        d = _decoded(case)
        if d is None:
            return
        u, recs, e = d
        if (len(e) != len(u.egress) or any(len(s) != u.windows for _, s in e)
                or any(len(f.payload) != u.grid.frame_bytes for frames, _ in e for f in frames)):
            return
        prefixes = [u.windows] * len(u.egress)
        for v in _same(lean, u, recs, e, prefixes, grid=False)[:sum(prefixes)]:
            no_grid[v.violation or "accept"] = no_grid.get(v.violation or "accept", 0) + 1

    undecided()
    print(kinds, no_grid)
    assert {"accept", "no-verdict", "duplicate", "signature", "advice", "mismatch"} <= set(kinds), kinds
    assert any(k.startswith("status:") for k in kinds), kinds
    assert {"refused", "accept", "no-verdict", "signature"} <= set(no_grid) and "mismatch" not in no_grid, no_grid


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
SESSION = bytes(range(lms.SESSION_BYTES))


def _certified(tmp_path, W, record=lambda w, r: r):
    """Link "responses"'s egress records under the shaper, each passed through `record(w, r)` and attested: the
    records, the declared syncs, the grid, Σ_sync and the attester's key."""
    g, ss = GridParams(3, 2, 2), SyncSet((0, 1), (Fraction(1),))
    syncs = [ClockSync(w % 2, Fraction(1)) for w in range(W)]
    signer = LmsSigner.create(lms.keygen(lms.Params(8, 8, 4, toy=True), bytes(range(8))), tmp_path / "signer-state.json")
    issuer = Issuer(signer, SESSION)
    egress = run_shaper(SERVICE_FRAMES, [0] * len(SERVICE_FRAMES), syncs, Certifier("responses", g))
    records = [issuer.certify(record(w, egress[w]), Direction.EGRESS, g, parameters_digest(g, ss)) for w in range(W)]
    return records, syncs, g, ss, signer.public_key()


def test_the_adapters_check_without_a_grid_is_the_leans_verdicts_no_grid(lean, tmp_path):
    """The adapter's `check_link(grid=False)` is the Lean's `verdicts-no-grid`: it accepts a window whose frames differ
    from the Program's in content alone, at its counts, where `verdicts` gives `mismatch`, with the Program's frames or
    with none, and refuses a record that isn't attested, whose grid no head check holds."""
    def content(w, r):
        if w != 1:
            return r
        slots = [list(row) for row in r.grid]
        b, k = next((b, k) for b, row in enumerate(slots) for k, s in enumerate(row) if s is not None)
        slots[b][k] = bytes(x ^ 0xFF for x in slots[b][k])
        return dataclasses.replace(r, grid=tuple(tuple(row) for row in slots))

    egress, syncs, g, ss, key = _certified(tmp_path, 3, content)
    args = (SERVICE_FRAMES, syncs[:2], ss, g, key, SESSION)
    assert [v.violation for v in lean_audit.Audit(server=lean).check_link("responses", [egress[1]], *args)] == \
        ["no-verdict", "mismatch"]
    for frames in (SERVICE_FRAMES, ()):
        a = lean_audit.Audit(server=lean)
        assert [v.violation for v in a.check_link("responses", [egress[1]], frames, *args[1:], grid=False)] == \
            ["no-verdict", None]
        assert a.lean == a.reference
    raw = content(1, run_shaper(SERVICE_FRAMES, [0] * 8, syncs, Certifier("responses", g))[1])
    for records in ([raw], [commit(raw, g)]):
        with pytest.raises(ValueError, match="only a certified record's grid is held by its head check"):
            lean_audit.Audit(server=lean).check_link("responses", records, *args, grid=False)


def test_a_disagreement_with_the_reference_rejects(lean, tmp_path, monkeypatch):
    """With the Python reference made to reject one execution phase the Lean check accepts (and, on ingress, the reverse
    direction of the same flip), that execution phase is a `disagreement`: the audit rejects, `lean` keeps the Lean's
    verdicts and `reference` the Python's. A reference that raises where the Lean answers raises."""
    real_egress, real_ingress = audit.Audit.check_link, audit.Audit.check_ingress_link

    def flip(vs, at):
        return [audit.Verdict(v.link, v.window, "mismatch" if v.accepted else None) if v.window == at else v for v in vs]

    monkeypatch.setattr(audit.Audit, "check_link", lambda self, *a, **k: flip(real_egress(self, *a, **k), 0))
    egress, syncs, g, ss, key = _certified(tmp_path, 1)
    a = lean_audit.Audit(server=lean)
    vs = a.check_link("responses", egress, SERVICE_FRAMES, syncs, ss, g, key=key, session=SESSION)
    assert vs == [audit.Verdict("responses", 0, "disagreement")] and not a.accepted and a.violations == vs
    assert a.lean == [audit.Verdict("responses", 0, None)] and a.reference == [audit.Verdict("responses", 0, "mismatch")]

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
    """An attested record under a key the Lean can't verify, an attested record with no key or no session, a session id
    not 32 bytes, a raw grid that isn't tuples, a frame of other than B bytes, a missing executable and no `lake` to build one each raise; none gets a verdict computed
    in Python, and the audit keeps no verdict or sync of a call that raised."""
    g = GridParams(1, 1, 1)
    ss = SyncSet((0,), (Fraction(1),))
    sync = [ClockSync(0, Fraction(1))]
    rec = run_shaper([], [], sync, Certifier("a", g))[0]
    certified = Issuer(V2._Signer(V2._key(4)), SESSION).certify(rec, Direction.EGRESS, g, parameters_digest(g, ss))

    class OtherKey:
        key_id = certified.certificate.key_id

        def verify(self, message, signature):
            return True

    a = lean_audit.Audit(server=lean)
    with pytest.raises(ValueError, match="not an LmsKey"):
        a.check_link("a", [certified], [], sync, ss, g, OtherKey(), SESSION)
    assert a.check_link("b", [certified], [], sync, ss, g, OtherKey(), SESSION) == [audit.Verdict("b", 0, "no-verdict")]
    with pytest.raises(ValueError, match="pass key"):
        a.check_link("a", [certified], [], sync, ss, g)
    key = LmsKey(V2._key(4).params, lms.public_key(V2._key(4)))
    with pytest.raises(ValueError, match="pass key and session"):
        a.check_link("a", [certified], [], sync, ss, g, key)
    for short in (SESSION[:-1], SESSION + b"\0"):
        with pytest.raises(ValueError, match="outside the Lean audit's domain: .*a certifier session id is 32 bytes"):
            a.check_link("a", [certified], [], sync, ss, g, key, short)
    with pytest.raises(ValueError, match="pass params"):
        a.check_ingress_link("a", [dataclasses.replace(certified, head=dataclasses.replace(
            certified.head, direction=Direction.INGRESS))], 1, OtherKey(), session=SESSION)
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
        "atexit.register(lambda: print('closed', sys.modules['verity.protocols.resource_accounting.network.verifier.lean_audit']"
        ".SERVER._proc is None))\n"
        "from fractions import Fraction\n"
        "from verity.protocols.resource_accounting.network.verifier import lean_audit\n"
        "from verity.protocols.resource_accounting.network.verifier.grid import Certifier, GridParams\n"
        "from verity.protocols.resource_accounting.network.verifier.schedule import ClockSync, SyncSet, run_shaper\n"
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


def _digest(i: int) -> bytes:
    return hashlib.sha512(b"served certificate %d" % i).digest()


@functools.lru_cache(maxsize=None)
def _served():
    """A toy key's attestations as a reveal endpoint serves window 1 of link "n2" in `SESSION`, each named by the
    refusal it gets as the first attestation of its kind: kept (``egress``, ``ingress``, the clock attestation
    ``clock``), another key's, another link's and another window's, an ingress clock attestation, a forged signature,
    one that doesn't decode, an altered digest, and a clock attestation signing ``seen``'s digest."""
    key, other = V2._key(4), V2._key(4, seed=1)
    i, o = Issuer(V2._Signer(key), SESSION), Issuer(V2._Signer(other), SESSION)
    egress = i.sign("n2", Direction.EGRESS, 1, _digest(0))
    certs = {"egress": egress, "ingress": i.sign("n2", Direction.INGRESS, 1, _digest(1)),
             "key": o.sign("n2", Direction.EGRESS, 1, _digest(2)),
             "link": i.sign("m", Direction.EGRESS, 1, _digest(3)), "window": i.sign("n2", Direction.EGRESS, 2, _digest(3)),
             "forged": dataclasses.replace(egress, signature=bytes([egress.signature[0] ^ 1]) + egress.signature[1:]),
             "undecoded": dataclasses.replace(egress, signature=egress.signature[:-1]),
             "altered": dataclasses.replace(egress, digest=_digest(9)),
             "clock": i.sign("n2", Direction.EGRESS, 1, _digest(5)),
             "ingress clock": i.sign("n2", Direction.INGRESS, 1, _digest(6)),
             "seen": i.sign("n2", Direction.EGRESS, 1, _digest(7))}
    return LmsKey(key.params, lms.public_key(key)), certs


def _refusals(lean, certificates, clocks, seen=(), session=SESSION):
    """The Lean's refusals and the reference's of window 1 of "n2" under the toy key, which must be one."""
    key, _ = _served()
    got = lean_audit.certificate_refusals(key, session, "n2", 1, certificates, clocks, seen, server=lean)
    assert got == audit.certificate_refusals(key, session, "n2", 1, certificates, clocks, seen)
    return got


def test_the_clients_certificate_refusals_are_the_references(lean):
    """`lean_audit.certificate_refusals` (`network-records verdicts`' ``certificates``, `NetworkCheck.TransferBlocks`) gives
    `audit.certificate_refusals`'s refusal of each attestation and clock attestation of a served window: another key's,
    another link's or window's, a forged signature, one that doesn't decode and an altered digest, a second of a
    direction, an ingress clock attestation and one signing a digest that `seen` holds, and in another session every
    signature; the honest ones are kept."""
    _, c = _served()
    certs = [c[k] for k in ("key", "link", "window", "forged", "undecoded", "altered", "egress", "clock", "ingress")]
    clocks = [c[k] for k in ("ingress clock", "seen", "clock", "egress")]
    assert _refusals(lean, certs, clocks, [_digest(7)]) == (
        ["key", "link-window", "link-window", "signature", "signature", "signature", None, "second", None],
        ["clock", "reused", None, "second"])
    assert _refusals(lean, certs, clocks) == (
        ["key", "link-window", "link-window", "signature", "signature", "signature", None, "second", None],
        ["clock", None, "second", "second"])
    assert _refusals(lean, certs, clocks, session=bytes(lms.SESSION_BYTES)) == (
        ["key", "link-window", "link-window"] + ["signature"] * 6, ["clock"] + ["signature"] * 3)
    assert _refusals(lean, [], []) == ([], [])


def test_on_generated_answers_the_lean_is_the_reference(lean):
    """On answers sampled from the served attestations in any order and number, with any of their digests seen: the Lean's
    refusals are the reference's, every refusal among them."""
    _, c = _served()
    pool = sorted(c.values(), key=lambda x: (x.link, x.direction.code, x.window, x.digest, x.signature))
    found = set()

    @settings(derandomize=True, database=None, deadline=None, max_examples=200)
    @given(st.lists(st.sampled_from(pool), max_size=6), st.lists(st.sampled_from(pool), max_size=6),
           st.sets(st.sampled_from([_digest(k) for k in range(10)])))
    def same(certificates, clocks, seen):
        a, b = _refusals(lean, certificates, clocks, sorted(seen))
        found.update(a + b)

    same()
    assert found == {None, "key", "link-window", "clock", "signature", "second", "reused"}, found


def test_a_certificate_disagreement_refuses_the_window(lean, monkeypatch):
    """With the reference made to keep everything, every attestation and clock attestation of the window is a
    `disagreement`, those the Lean keeps among them; a reference that raises where the Lean answers raises."""
    key, c = _served()
    monkeypatch.setattr(audit, "certificate_refusals",
                        lambda key, s, link, w, xs, ks, seen=(): ([None] * len(xs), [None] * len(ks)))
    assert lean_audit.certificate_refusals(key, SESSION, "n2", 1, [c["egress"], c["forged"]], [c["clock"]],
                                           server=lean) == (["disagreement"] * 2, ["disagreement"])
    assert lean_audit.certificate_refusals(key, SESSION, "n2", 1, [c["egress"]], [], server=lean) == ([None], [])

    def boom(*a, **k):
        raise ArithmeticError("reference")

    monkeypatch.setattr(audit, "certificate_refusals", boom)
    with pytest.raises(RuntimeError, match="the reference raised where the Lean check answered"):
        lean_audit.certificate_refusals(key, SESSION, "n2", 1, [c["egress"]], [], server=lean)


def test_certificates_outside_the_domain_are_refused(lean):
    """A key other than an `LmsKey`, an attestation of ill-typed fields (a negative window, a link that isn't a string)
    and an LMS key of another length each raise `ValueError`, with no refusal computed in Python."""
    key, c = _served()

    class OtherKey:
        key_id = key.key_id

        def verify(self, message, signature):
            return True

    with pytest.raises(ValueError, match="not an LmsKey"):
        lean_audit.certificate_refusals(OtherKey(), SESSION, "n2", 1, [c["egress"]], [], server=lean)
    for bad in (dataclasses.replace(c["egress"], window=-1), dataclasses.replace(c["egress"], link=b"n2")):
        with pytest.raises(ValueError, match="ill-typed fields"):
            lean_audit.certificate_refusals(key, SESSION, "n2", 1, [bad], [], server=lean)
    with pytest.raises(ValueError, match="outside the Lean audit's domain: .*an LMS public key is"):
        lean_audit.certificate_refusals(LmsKey(key.params, key.public[:-1]), SESSION, "n2", 1, [c["egress"]], [],
                                        server=lean)

