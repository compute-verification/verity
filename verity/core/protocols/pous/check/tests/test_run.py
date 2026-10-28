"""`pous-run` (`PousCheck.Run`) against `run.check_at`, through the core difftest harness: the outcome, its reason word for
word, on runs drawn at reduced points (the verifier's list of memory-challenger keys, a setup's public outputs, the table of
`C*`'s row digests, one session the memory challenger signed bound to the setup's root, the id the auditor designated),
honest and under one edit of the verifier's list, the setup, the table, the point, the session or the designated id, the
session edits among them every one of `mc_cases.MUTATIONS`, and runs whose JSON is not exactly what it decodes to, a
session that carries keys of its own among them. Then `pous-run TRUSTED.json SETUP.json DIGESTS.bin SESSION.json
SESSION_ID`, the deployed check, on a run at the deployed point signed by a key at S1's deployed parameters: both accept
it, and both refuse it once one row of the table is another, the designated id is another or the verifier's list holds
another key."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period, run
from verity.core.protocols.pous import setup as pous_setup
from verity.core.protocols.pous.schemes import p2 as P2
from verity.core.protocols.pous.verifier import VerifierKey

LEAN = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

_spec = importlib.util.spec_from_file_location("mc_cases", ROOT / "verity/core/devices/memory_challenger/lean/tests/mc_cases.py")
M = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


@pytest.fixture(autouse=True, scope="module")
def _test_signer_dir(tmp_path_factory):
    """The test signer's session logs under the suite's basetemp (`pous_test_signer.DIR`)."""
    M.ts.DIR = tmp_path_factory.mktemp("test-signer")
    yield
    M.ts.DIR = None

#: Every scheme `pous-run serve` verifies, by id (`MemoryChallenger.KNOWN_SCHEMES`).
KNOWN = {**mc.SCHEMES, **M.ts.SCHEMES}
#: Reduced widths (`2^w − c0` prime, as the spec's ladder) and parameters `P2Params` refuses.
WIDTHS = [(64, 189), (96, 17), (128, 173)]
INVALID = [(20, 189), (64, 3), (64, 0)]


@dataclass(frozen=True)
class Run:
    point: run.Point
    setup: run.Setup
    session: M.Session
    schemes: tuple[str, ...]
    session_id: bytes
    setup_of: dict | None = None
    """The setup protocol's case that published `setup`, for a run over an honest setup's `C*` (`runs(stores=...)`)."""

    @property
    def digests(self) -> bytes:
        return b"".join(self.session.expected)

    def doc(self) -> dict:
        q = self.point.params
        return {"point": {"params": {"blocks_per_segment": q.blocks_per_segment, "c0": q.c0, "segments": q.segments, "w": q.w},
                          "k": self.point.k, "limit_ns": self.point.limit_ns},
                "schemes": list(self.schemes), "trusted": [t.hex() for t in self.session.trusted],
                "setup": self.setup.to_json(), "digests": self.digests.hex(), "session": session_of(self.session).to_json(),
                "session_id": self.session_id.hex()} | ({"setup_of": self.setup_of} if self.setup_of else {})


def session_of(s: M.Session) -> run.ChallengerRecord:
    """The record of `s`; `s.trusted` is the verifier's list, which the record never carries."""
    return run.ChallengerRecord(s.signed, s.commitment, s.openings)


def rows_of(q: P2.P2Params, tag: int) -> tuple[bytes, ...]:
    return tuple(mc.block_digest(M.block(tag * 1000 + i, q.block_bytes)) for i in range(q.blocks))


@st.composite
def runs(draw, signer: M.AtLeaf | None = None, stores: st.SearchStrategy | None = None) -> Run:
    """An honest run: a setup at a drawn reduced point, the memory challenger's session over its `C*` (some rounds late, some
    answers tampered) at the point's `k`, a limit within the point's, bound to the setup's root, under the designated id.
    With `stores`, a strategy of honest setups `(q, W, salt, C*)`, the setup is one it draws and `C*` its encoding, which
    the setup protocol accepts (the run's `setup_of`); else `C*` is labels of no setup."""
    stored = draw(stores) if stores is not None else None
    if stored is None:
        w, c0 = draw(st.sampled_from(WIDTHS))
        q = P2.P2Params(w, c0, draw(st.integers(1, 4)), draw(st.integers(1, 3)))
    else:
        q = stored[0]
    k, limit = draw(st.integers(1, 4)), draw(st.integers(2, 10 ** 6))
    point = run.Point(q, k, limit)
    tag = draw(st.integers(0, 3))
    if stored is None:
        salt = hashlib.sha512(b"run salt %d" % tag).digest()[:run.SALT_BYTES]
        blocks = tuple(M.block(tag * 1000 + i, q.block_bytes) for i in range(q.blocks))
        commitment, setup_of = hashlib.sha512(b"W %d" % tag).digest(), None
    else:
        _, w_bytes, salt, encoded = stored
        blocks = tuple(b for seg in encoded for b in seg)
        commitment = pous_setup.commit(q, w_bytes)
        setup_of = period.setup_case(q, commitment, salt, encoded)
    rows = tuple(mc.block_digest(b) for b in blocks)
    root = VerifierKey.from_row_digests(q, salt, list(rows)).root
    setup = run.Setup(q.blocks, commitment, root, salt)
    if signer is None:
        height = draw(st.integers(1, 2))
        signer = M.test_signer(draw(st.sampled_from(M.SECRETS)), height, draw(st.integers(0, 2 ** height - 1)))
    session_limit = draw(st.sampled_from([limit, limit - 1, limit // 2 + 1]))
    session_id = hashlib.sha512(b"run session %d" % draw(st.integers(0, 3))).digest()[:32]
    tree = mc.CoinTree.draw(b"run source %d" % tag, session_id, q.blocks, k, key=draw(st.sampled_from(M.COIN_KEYS)),
                            salts=[hashlib.sha512(b"run coin salt %d %d" % (tag, j)).digest() * 3 for j in range(k)])
    spends = draw(st.lists(st.sampled_from([0, 1, session_limit, session_limit + 1]), min_size=k, max_size=k))
    tampered = draw(st.sets(st.integers(0, k - 1), max_size=1))
    clock = M.Clock(draw(st.integers(0, 2 ** 40)))
    c = mc.MemoryChallenger(tree, signer, log=M.ts.session_log(), limit_ns=session_limit, block_bytes=q.block_bytes,
                            binding=root, clock=clock)
    while not c.done:
        j = len(c.rounds)
        i = c.challenge()
        clock.now += spends[j]
        b = blocks[i]
        c.receive(b[:-1] + bytes([b[-1] ^ 1]) if j in tampered else b)
    s = M.Session(c.close(), c.commitment, c.openings(), rows, (signer.public_key,), signer)
    return Run(point, setup, s, draw(st.sampled_from([(signer.scheme,), (signer.scheme, mc.LMS), (mc.LMS,)])), session_id,
               setup_of)


def _published(r: Run, **edits) -> Run:
    """The session's published commitment edited, and the signed one with it, re-signed."""
    c = dataclasses.replace(r.session.commitment, **edits)
    return dataclasses.replace(r, session=dataclasses.replace(r.session.resign(commitment=c), commitment=c))


def _point(r: Run, **edits) -> Run:
    return dataclasses.replace(r, point=dataclasses.replace(r.point, **edits))


def _setup(r: Run, **edits) -> Run:
    return dataclasses.replace(r, setup=dataclasses.replace(r.setup, **edits))


def _rows(r: Run, rows) -> Run:
    return dataclasses.replace(r, session=dataclasses.replace(r.session, expected=tuple(rows)))


def _other_geometry(r: Run, x: int) -> Run:
    """The same labels at another width (the table's leaves and root unchanged), or in another split into segments."""
    q = r.point.params
    w, c0 = WIDTHS[(WIDTHS.index((q.w, q.c0)) + 1 + x % 2) % len(WIDTHS)]
    if x % 3 == 0 and q.blocks > 1:
        return _point(r, params=P2.P2Params(q.w, q.c0, q.blocks, 1) if q.segments > 1 else P2.P2Params(q.w, q.c0, 1, q.blocks))
    return _point(r, params=P2.P2Params(w, c0, q.blocks_per_segment, q.segments))


#: One edit of a run, `(run, x) -> run`, `x` a drawn natural.
RUN_MUTATIONS = [
    lambda r, x: _setup(r, blocks=r.setup.blocks + 1 + x % 3),
    lambda r, x: _setup(r, blocks=0),
    lambda r, x: _setup(r, salt=r.setup.salt + bytes(1 + x % 8)),
    lambda r, x: _setup(r, salt=r.setup.salt[:-1 - x % 8]),
    lambda r, x: _setup(r, root=M._flip(r.setup.root, x)),
    lambda r, x: _setup(r, root=r.setup.root[:-1]),
    lambda r, x: _setup(r, root=VerifierKey.from_row_digests(r.point.params, bytes(run.SALT_BYTES),
                                                             list(r.session.expected)).root),
    lambda r, x: _rows(r, r.session.expected[1:] + r.session.expected[:1]),
    lambda r, x: _rows(r, r.session.expected + r.session.expected[:1]),
    lambda r, x: _rows(r, rows_of(r.point.params, 7)),
    lambda r, x: _rows(r, (r.digests[:-1 - x % 64],)),
    _other_geometry,
    lambda r, x: _point(r, k=r.point.k + 1 + x % 3),
    lambda r, x: _point(r, k=max(r.point.k - 1, 0)),
    lambda r, x: _point(r, limit_ns=r.session.commitment.limit_ns - 1),
    lambda r, x: _point(r, limit_ns=r.session.commitment.limit_ns),
    lambda r, x: _published(r, binding=hashlib.sha512(b"another setup's root %d" % (x % 3)).digest()),
    lambda r, x: _published(r, binding=r.setup.root[:-1]),
    lambda r, x: _published(r, blocks=r.session.commitment.blocks + 1 + x % 3),
    lambda r, x: _published(r, block_bytes=r.session.commitment.block_bytes + 1 + x % 3),
    lambda r, x: _published(r, limit_ns=r.point.limit_ns + 1 + x % 1000),
    lambda r, x: dataclasses.replace(r, session=dataclasses.replace(r.session, trusted=r.session.trusted * 2)),
    lambda r, x: dataclasses.replace(r, schemes=()),
    lambda r, x: dataclasses.replace(r, schemes=tuple(KNOWN)),
    lambda r, x: dataclasses.replace(r, session_id=hashlib.sha512(b"another session %d" % (x % 3)).digest()[:32]),
    lambda r, x: dataclasses.replace(r, session_id=r.session_id[:-1 - x % 32]),
    lambda r, x: _published(r, session=hashlib.sha512(b"another session %d" % (x % 3)).digest()[:32]),
    lambda r, x: dataclasses.replace(r, session=dataclasses.replace(r.session, trusted=())),
    lambda r, x: dataclasses.replace(r, session=dataclasses.replace(
        r.session, trusted=(M.test_signer(M.STRANGER, 1, 0).public_key,))),
]
#: The memory challenger's session edits, each reason printable (`test_verdict.records`).
SESSION_MUTATIONS = [M._printable_scheme if m is M._scheme else m for m in M.MUTATIONS]


def _edit(path: tuple[str, ...], f):
    def go(j, x):
        return f(j, x) if not path else j | {path[0]: _edit(path[1:], f)(j[path[0]], x)}
    return go


#: One edit of a run's JSON: a field extra or missing, hex not lowercase or odd, a number not a natural, a header constant
#: not the protocol's, a scheme id the port doesn't verify, parameters `P2Params` refuses.
JSON_MUTATIONS = [
    _edit(("setup",), lambda s, x: s | {"extra": x}),
    _edit(("setup",), lambda s, x: {k: v for k, v in s.items() if k != sorted(s)[x % len(s)]}),
    _edit(("setup",), lambda s, x: s | {"root": s["root"].upper()}),
    _edit(("setup",), lambda s, x: s | {"blocks": -1 - x}),
    _edit(("setup",), lambda s, x: s | {"blocks": str(s["blocks"])}),
    _edit(("setup",), lambda s, x: s | {"salt": s["salt"] + "0"}),
    _edit((), lambda j, x: j | {"digests": j["digests"].upper() or "A"}),
    _edit((), lambda j, x: j | {"digests": j["digests"] + "0"}),
    _edit((), lambda j, x: {k: v for k, v in j.items() if k != sorted(j)[x % len(j)]}),
    _edit((), lambda j, x: j | {"session_id": j["session_id"].upper()}),
    _edit((), lambda j, x: j | {"session_id": j["session_id"] + "0"}),
    _edit(("session",), lambda s, x: s | {"extra": x}),
    _edit(("session",), lambda s, x: {k: v for k, v in s.items() if k != sorted(s)[x % len(s)]}),
    _edit((), lambda j, x: j | {"trusted": [t.upper() for t in j["trusted"]] or ["A"]}),
    _edit((), lambda j, x: j | {"trusted": [j["trusted"][0] + "0"] if j["trusted"] else [x]}),
    _edit((), lambda j, x: j | {"session": j["session"] | {"trusted": j["trusted"]}}),
    _edit(("session", "commitment"), lambda c, x: c | {"protocol": "verity-pous/memory-challenger/v0"}),
    _edit(("session", "signed", "commitment"), lambda c, x: c | {"clock": "CLOCK_REALTIME ns"}),
    _edit(("session", "openings"), lambda o, x: o + [o[0] | {"j": -1}] if o else [None]),
    _edit(("schemes",), lambda s, x: s + [["none", mc.LMS + " ", M.ts.scheme(11), "test/"][x % 4]]),
    _edit(("schemes",), lambda s, x: s + [x]),
    _edit(("point",), lambda p, x: p | {"k": -1}),
    _edit(("point", "params"), lambda p, x: p | dict(zip(("w", "c0"), INVALID[x % len(INVALID)]))),
    _edit(("point", "params"), lambda p, x: p | {"blocks_per_segment": 2 ** 16 + 1 + x}),
]


def _decode_setup(j: Any) -> run.Setup:
    return run.Setup(M.nat(j, "blocks"), M.hex_(j, "commitment"), M.hex_(j, "root"), M.hex_(j, "salt"))


def _decode_session(j: Any) -> run.ChallengerRecord:
    return run.ChallengerRecord(M.signed_of(M._get(j, "signed")), M.commitment_of(M._get(j, "commitment")),
                       tuple(M.opening_of(o) for o in M.list_(j, "openings")))


def _point_of(j: Any) -> run.Point:
    p = M._get(j, "params")
    try:
        q = P2.P2Params(M.nat(p, "w"), M.nat(p, "c0"), M.nat(p, "blocks_per_segment"), M.nat(p, "segments"))
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return run.Point(q, M.nat(j, "k"), M.nat(j, "limit_ns"))


def reference(j: Any) -> dict:
    """`PousCheck.Run.runFn` in Python: decode as the Lean decodes, then `run.check_at`."""
    point = _point_of(M._get(j, "point"))
    ids = M.list_(j, "schemes")
    if not all(isinstance(i, str) for i in ids) or any(i not in KNOWN for i in ids):
        raise D.Reject("a scheme this port doesn't verify")
    setup = M._exactly(_decode_setup, M._get(j, "setup"))
    digests = M.hex_(j, "digests")
    session_id = M.hex_(j, "session_id")
    trusted = M.hexes(j, "trusted")
    session = M._exactly(_decode_session, M._get(j, "session"))
    try:
        run.check_at(point, {i: KNOWN[i] for i in ids}, trusted, setup, digests, session, session_id)
    except run.Refused as e:
        return {"accepted": False, "reason": str(e)}
    return {"accepted": True}


@dataclass(frozen=True)
class _Port(D.Port):
    """`pous-run serve`: the executable serves under `serve`, and judges four files and an id otherwise."""

    def argv(self) -> list[str]:
        return [*super().argv(), "serve"]


def port(inputs, **kw) -> D.Port:
    return _Port("pous-run", "run", reference, inputs, package=LEAN, **kw)


server = D.server_fixture(port(st.none()))


def _any(edits: list) -> st.SearchStrategy:
    """One of `edits`, about evenly: `sampled_from` draws early entries far more often than late ones."""
    return st.integers(0, 2 ** 32).map(lambda i: edits[i % len(edits)])


def documents(rs: st.SearchStrategy | None = None) -> st.SearchStrategy:
    """Honest runs (from `rs`, `runs()` by default), runs under one run edit or session edit, and runs under one JSON
    edit."""
    xs, rs = st.integers(0, 2 ** 16), runs() if rs is None else rs
    honest = st.builds(Run.doc, rs)
    edited = st.builds(lambda r, m, x: m(r, x).doc(), rs, _any(RUN_MUTATIONS), xs)
    session_edited = st.builds(lambda r, m, x: dataclasses.replace(r, session=m(r.session, x)).doc(), rs,
                               _any(SESSION_MUTATIONS), xs)
    malformed = st.builds(lambda r, m, x: m(r.doc(), x), rs, _any(JSON_MUTATIONS), xs)
    return st.one_of(honest, edited, edited, session_edited, session_edited, malformed)


@pytest.mark.slow
def test_run_agrees(server):
    """The outcome on drawn runs: accepted ones, each refusal of the check's order, verdict reasons, and malformed input."""
    stats = D.run(port(documents()), examples=400, server=server)
    assert stats.outputs > 250 and stats.rejects > 20, stats


def test_named_runs_agree(server):
    """An honest run accepted, and one run under each run edit, through the port: the Lean's outcome is the Python's, and
    each edit is refused for a reason of the check's own before the verdict, or by the verdict; the designated id's edits,
    and a session published under another id, for the id's reason; a verifier's list without the signer's key, for the
    verdict's."""
    p = port(st.none())
    r = _named(M.test_signer(M.SECRETS[0], 1, 0))
    assert D.replay(p, r.doc(), server) == ("output", D.canonical({"accepted": True}))
    reasons = {}
    for i, m in enumerate(RUN_MUTATIONS):
        out = D.replay(p, m(r, i).doc(), server)
        assert out[0] == "output", (i, out)
        reasons[i] = json.loads(out[1]).get("reason")
    accepted = {i for i, why in reasons.items() if why is None}
    assert accepted == {15, 21, 23}, reasons
    assert len(set(reasons.values())) >= 15, reasons
    assert {reasons[i] for i in (24, 25, 26)} == {"the session does not run under the id the auditor designated"}, reasons
    assert {reasons[i] for i in (27, 28)} == {"the session is signed by a key that is not a trusted memory challenger's"}, \
        reasons


def _named(signer: M.AtLeaf) -> Run:
    """A run of 3 rounds over two segments of 4 labels of 8 bytes, every answer on time and right."""
    q = P2.P2Params(64, 189, 4, 2)
    point = run.Point(q, 3, 1_000)
    salt = hashlib.sha512(b"named salt").digest()[:run.SALT_BYTES]
    rows = rows_of(q, 0)
    root = VerifierKey.from_row_digests(q, salt, list(rows)).root
    tree = mc.CoinTree.draw(b"named source", bytes(32), q.blocks, 3, key=M.COIN_KEYS[0], salts=[bytes([j]) * 192 for j in range(3)])
    clock = M.Clock(1_000)
    c = mc.MemoryChallenger(tree, signer, log=M.ts.session_log(), limit_ns=1_000, block_bytes=q.block_bytes, binding=root,
                            clock=clock)
    while not c.done:
        i = c.challenge()
        clock.now += 10
        c.receive(M.block(i, q.block_bytes))
    s = M.Session(c.close(), c.commitment, c.openings(), rows, (signer.public_key,), signer)
    return Run(point, run.Setup(q.blocks, bytes(64), root, salt), s, (signer.scheme,), bytes(32))


def _deployed_run(tmp: Path, tampered: int | None = None
                  ) -> tuple[Path, Path, Path, run.Setup, bytes, run.ChallengerRecord, bytes]:
    """A run at the deployed point signed at leaf 0 of a key at S1's deployed parameters (`mc_cases.deployed_signer`, a
    sparse tree), written as `pous-run` reads it, round `tampered`'s answer not its label. The rows the rounds don't name
    are digests of nothing in particular. Last, the signer's public key."""
    d = run.DEPLOYED
    q = d.params
    salt = hashlib.sha512(b"deployed salt").digest()[:run.SALT_BYTES]
    tree = mc.CoinTree.draw(b"deployed source", hashlib.sha512(b"deployed session").digest()[:32], q.blocks, d.k,
                            key=M.COIN_KEYS[0], salts=[hashlib.sha512(b"deployed coin salt %d" % j).digest() * 3
                                                       for j in range(d.k)])
    rows = [hashlib.sha512(b"row %d" % i).digest() for i in range(q.blocks)]
    for i in tree.indices:
        rows[i] = mc.block_digest(M.block(i, q.block_bytes))
    root = VerifierKey.from_row_digests(q, salt, rows).root
    signer, clock = M.deployed_signer(), M.Clock(10 ** 9)
    c = mc.MemoryChallenger(tree, signer, log=M.ts.session_log(), limit_ns=d.limit_ns, block_bytes=q.block_bytes,
                            binding=root, clock=clock)
    while not c.done:
        j = len(c.rounds)
        i = c.challenge()
        clock.now += 300_000
        b = M.block(i, q.block_bytes)
        c.receive(M._flip(b, 0) if j == tampered else b)
    setup = run.Setup(q.blocks, hashlib.sha512(b"deployed W").digest(), root, salt)
    session = run.ChallengerRecord(c.close(), c.commitment, c.openings())
    digests = b"".join(rows)
    paths = tmp / "SETUP.json", tmp / "DIGESTS.bin", tmp / "SESSION.json"
    paths[0].write_text(D.canonical(setup.to_json()))
    paths[1].write_bytes(digests)
    paths[2].write_text(D.canonical(session.to_json()))
    return *paths, setup, digests, session, signer.public_key


def _trusted_file(path: Path, trusted: tuple[bytes, ...]) -> Path:
    """The verifier's `TRUSTED.json`."""
    path.write_text(D.canonical({"trusted": [t.hex() for t in trusted]}))
    return path


def _python_outcome(trusted: tuple[bytes, ...], setup: run.Setup, digests: bytes, session: run.ChallengerRecord,
                    session_id: bytes) -> dict:
    try:
        run.check(trusted, setup, digests, session, session_id)
    except run.Refused as e:
        return {"accepted": False, "reason": str(e)}
    return {"accepted": True}


@pytest.mark.slow
def test_the_deployed_check_agrees_from_its_files(tmp_path):
    """`pous-run`'s four files and the designated id at the deployed point (`2^19` rows, `k = 105`, S1's LMS at its deployed
    parameters): `PousCheck.Run.check` and `run.check` accept the honest run and refuse, for the same reason, the run one of
    whose answers is not its label, the honest run under another designated id and the honest run against a verifier's list
    that holds another key only; `pous-run` refuses an id that is not lowercase hex and a session that carries keys of its
    own. What `test_run_agrees` can't: that `check` itself, not `checkAt` at a drawn point, is the deployed point under
    `SCHEMES`, in Lean."""
    exe = D.build(LEAN, "pous-run")
    another = hashlib.sha512(b"another deployed session").digest()[:32]
    stranger = (M.test_signer(M.STRANGER, 1, 0).public_key,)
    untrusted = "the session is signed by a key that is not a trusted memory challenger's"
    for tampered in (None, 7):
        sub = tmp_path / str(tampered)
        sub.mkdir()
        setup_path, digests_path, session_path, setup, digests, session, key = _deployed_run(sub, tampered)
        designated = session.commitment.session
        expect = [((key,), designated, "1 answers are not their blocks of C")] if tampered is not None else [
            ((key,), designated, None), ((key,), another, "the session does not run under the id the auditor designated"),
            (stranger, designated, untrusted), ((), designated, untrusted)]
        for n, (trusted, sid, reason) in enumerate(expect):
            python = _python_outcome(trusted, setup, digests, session, sid)
            assert python == ({"accepted": True} if reason is None else {"accepted": False, "reason": reason}), python
            trusted_path = _trusted_file(sub / f"TRUSTED-{n}.json", trusted)
            out = subprocess.run([str(exe), str(trusted_path), str(setup_path), str(digests_path), str(session_path),
                                  sid.hex()], capture_output=True, text=True, check=False)
            assert out.returncode == (0 if python["accepted"] else 1), out.stderr
            assert out.stdout == D.canonical(python) + "\n"
        if tampered is None:
            files = [str(exe), str(_trusted_file(sub / "TRUSTED.json", (key,))), str(setup_path), str(digests_path)]
            out = subprocess.run([*files, str(session_path), designated.hex().upper()], capture_output=True, text=True,
                                 check=False)
            assert (out.returncode, out.stdout) == (1, D.canonical({"accepted": False, "reason": "session id: not lowercase hex"})
                                                    + "\n"), out
            carrying = sub / "SESSION-carrying-keys.json"
            carrying.write_text(D.canonical(session.to_json() | {"trusted": [key.hex()]}))
            out = subprocess.run([*files, str(carrying), designated.hex()], capture_output=True, text=True, check=False)
            assert out.returncode == 1 and json.loads(out.stdout)["reason"].startswith("session: "), out
