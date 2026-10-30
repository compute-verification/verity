"""PoUB-P2's core case (`p2_case.py`) replayed by `remote-audit-p2-protocol serve` (`RemoteAudit.P2Protocol`): the case
`p2_party.session` sends the prover for each session of `test_p2_party.py`'s, its answer the core form of the session's
own (`p2_case.answer`), and every field equal: an honest session accepted with its credited reads, one whose receipts
skip a unit's reads and one whose openings are out of form rejected, and one stopped before its last window is
verified undecided. Edits of a root, a coin or the plan are rejected, and the plan is bound by the commitment's
digest. A session the books veto has no case. And a session over the `C*` of PoUS's
own setup names PoUS's store: onsite's `same_store` takes the two as one store, and the Lean accepts the case."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from test_p2_party import Lying, Prover, exe, request, run, store  # noqa: F401
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period as pous
from verity.core.protocols.pous import run as R
from verity.core.protocols.remote_bandwidth import p2_case, p2_lean, p2_party, receipts

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PACKAGE = Path(__file__).resolve().parents[1]
SIGNER = PACKAGE.parents[1] / "pous" / "tests" / "pous_test_signer.py"
COINS = hashlib.sha512(b"the period's live coins").digest()


@pytest.fixture(scope="module")
def lean() -> Path:
    """`remote-audit-p2-protocol`, built in the package under the difftest's build lock (`p2_lean.BUILD_LOCK`)."""
    p2_lean.BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(p2_lean.BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = subprocess.run(["lake", "build", p2_case.EXE], cwd=PACKAGE, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return PACKAGE / ".lake" / "build" / "bin" / p2_case.EXE


class Keeping(Prover):
    """`test_p2_party.Prover`, keeping the case the verifier sends."""

    case_ = None

    def case(self, doc) -> None:
        self.case_ = doc


def session(exe, store, *, plan=None, lean_of=lambda lean: lean, **kw) -> tuple[dict, dict | None]:
    """One session of `test_p2_party.run` with a `Keeping` prover: the answer and the case it kept."""
    kept = []

    def prover_of(clock):
        kept.append(Keeping(store, clock, **kw))
        return kept[-1]

    proved, answer = run(exe, store, prover_of, request(store), plan, lean_of)
    assert proved == answer
    return answer, kept[0].case_


def replayed(lean, case) -> dict:
    return p2_case.serve(lean, [case])[0]


def _evidence(case) -> dict:
    return json.loads(bytes.fromhex(case["evidence"]))


def _with_log(case, log) -> dict:
    return case | {"evidence": json.dumps({"log": log}, sort_keys=True, separators=(",", ":")).encode().hex()}


def test_an_honest_sessions_case_is_accepted_with_its_credited_reads(exe, lean, store) -> None:
    answer, case = session(exe, store)
    assert answer["verdict"] == "accept" and case is not None
    got = replayed(lean, case)
    assert got == p2_case.answer(case, answer)
    assert got["verdict"] == "accept" and got["protocol"] == p2_case.PROTOCOL
    assert got["public"]["q_bytes"] == answer["q_bytes"] > 0 and got["public"]["span_ns"] == answer["span_ns"]
    assert got["commitment"] == p2_case.digest(case).hex()
    assert sorted(case) == ["commitment", "evidence", "params", "store_root"]
    assert case["store_root"] == store.registration.root.hex() == p2_case.store_root(case).hex()
    assert case["params"]["request"] == request(store)
    ops = [x["op"] for x in _evidence(case)["log"]]
    assert ops[0] == "p2_event" and ops.count("p2_openings") == 2 and ops.count("p2_root") == 4


@pytest.mark.parametrize(("kw", "failure"), [({"skip": 1}, "receipt"), ({"garble": True}, "malformed")])
def test_a_rejected_sessions_case_is_rejected_with_its_failure(exe, lean, store, kw, failure) -> None:
    answer, case = session(exe, store, **kw)
    assert answer["verdict"] == "reject" and answer["failure"] == failure
    got = replayed(lean, case)
    assert got == p2_case.answer(case, answer)
    assert got["verdict"] == "reject" and f"rejected: {failure} at window 0" in got["reason"]


def test_a_case_before_its_last_window_is_verified_is_undecided(exe, lean, store) -> None:
    _, case = session(exe, store)
    log = _evidence(case)["log"]
    last = max(k for k, x in enumerate(log) if x["op"] == "p2_openings")
    got = replayed(lean, _with_log(case, log[:last]))
    assert got["verdict"] == "undecided" and got["public"] is None


def test_edits_of_the_log_are_refused_or_change_the_answer(exe, lean, store) -> None:
    """A root line, a coin, an unknown op and a log out of canonical form."""
    _, case = session(exe, store)
    log = _evidence(case)["log"]
    k = next(k for k, x in enumerate(log) if x["op"] == "p2_root")
    root = json.loads(bytes.fromhex(log[k]["line"]))["root"]
    other = json.dumps({"root": root[:-2] + ("00" if root[-2:] != "00" else "01")}).encode() + b"\n"
    edits = {
        "root": log[:k] + [log[k] | {"line": other.hex()}] + log[k + 1:],
        "coin": [log[0] | {"coin": "00" * 32}] + log[1:],
    }
    for name, edited in edits.items():
        got = replayed(lean, _with_log(case, edited))
        assert got["verdict"] == "reject", (name, got)
    got = replayed(lean, _with_log(case, log + [{"op": "p2_hold"}]))
    assert got["verdict"] == "reject" and got["reason"] == "evidence: log: no op p2_hold"
    got = replayed(lean, case | {"evidence": json.dumps({"log": log}).encode().hex()})
    assert got["verdict"] == "reject" and "not canonical JSON" in got["reason"]


def test_the_commitment_binds_the_plan(exe, lean, store) -> None:
    """A session under a plan of fewer units is accepted with those units' reads alone, and its digest is not the
    whole store's: the same log under the whole store's registration file is another commitment, and rejected."""
    units = store.registration.blocks
    answer, case = session(exe, store, plan=receipts.Plan(((1, units - 1),), units))
    got = replayed(lean, case)
    assert got["verdict"] == "accept" and got["profile"]["plan_units"] == units - 1 == answer["plan_units"]
    whole = p2_party.registration_json(store.registration, receipts.Plan.whole(units))
    swapped = case | {"commitment": json.dumps(whole, sort_keys=True, separators=(",", ":")).encode().hex()}
    assert p2_case.digest(swapped) != p2_case.digest(case)
    assert replayed(lean, swapped)["verdict"] == "reject"


def test_a_session_the_books_veto_has_no_case(exe, store) -> None:
    def accepting(lean):
        return Lying(lean, "p2_openings", lambda req, _got: {"accepted": req["t"]})

    answer, case = session(exe, store, lean_of=accepting, skip=1)
    assert answer["failure"] == p2_party.DISAGREEMENT and case is None


def test_the_written_piece_is_private(exe, store, tmp_path) -> None:
    answer, case = session(exe, store)
    d = p2_case.write(tmp_path / "private" / "bandwidth", case, p2_case.answer(case, answer))
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    for name in (p2_case.CASE, p2_case.ANSWER):
        assert stat.S_IMODE((d / name).stat().st_mode) == 0o600
    assert json.loads((d / p2_case.CASE).read_text()) == case
    assert json.loads((d / p2_case.ANSWER).read_text())["commitment"] == p2_case.digest(case).hex()


def _signer():
    spec = importlib.util.spec_from_file_location("pous_test_signer", SIGNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_session_on_pous_setups_store_names_pous_store_and_is_accepted(exe, lean, store, tmp_path) -> None:
    """PoUS's audit of the committed `C*` (its setup published at that `C*`'s commitment, salt and vk root) and PoUB-P2
    registered on the same setup, an honest session over that `C*`: the case names the root PoUS published, onsite's
    `same_store` takes the two as one store, and the Lean accepts the case."""
    S = _signer()
    reg = store.registration
    q, n = reg.params, reg.params.blocks_per_segment
    encoded = [list(store.blocks[n * s:n * (s + 1)]) for s in range(q.segments)]
    held = pous.Store(R.Point(q, 8, pous.LIMIT_NS), R.Setup(q.blocks, reg.commitment, reg.root, reg.salt), store.blocks,
                      b"".join(mc.block_digest(b) for b in store.blocks),
                      pous.setup_case(q, reg.commitment, reg.salt, encoded))
    space = pous.audit("n0", held, COINS, signer=S.TestSigner(b"the memory challenger", directory=tmp_path),
                       log=S.session_log(tmp_path), schemes=S.SCHEMES, protocol=mc.TEST_PREFIX + pous.PROTOCOL,
                       clock=lambda: 10 ** 9)
    assert R.Setup.from_json(json.loads(bytes.fromhex(space.case["commitment"]))).root == reg.root
    answer, case = session(exe, store)
    assert answer["verdict"] == "accept"
    assert p2_case.store_root(case) == reg.root
    assert p2_case.same_store(case, space.case) is None
    assert replayed(lean, case)["verdict"] == "accept"
