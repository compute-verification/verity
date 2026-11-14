"""PoUB-P2's core case (`p2_case.py`) replayed by `remote-audit-p2-protocol serve` (`RemoteAudit.P2Protocol`): the case
`p2_party.session` sends the prover for each session of `test_p2_party.py`'s held to the auditor's session (`SESSION`),
its answer the core form of the session's own (`p2_case.answer`), and every field equal: an honest session accepted
with its credited reads, one whose receipts skip a unit's reads and one whose openings are out of form rejected, and
one stopped before its last window is verified undecided. A toy party's case, and a case whose request is edited off
`SESSION` in any value, are refused naming each one. The memory challenger's close is checked before the log is
replayed, each refusal by name: a key the params don't trust, a line count or a chain that isn't the case's (its
head, the commitment's digest and the clock floor, then its log), a
signature that doesn't verify, a close replayed under another session, a coin that isn't the challenger's whole coin,
and a PoUS session record or a wipe record the same key signed for the same session put as the close. Edits of a root,
a coin or the plan are rejected, and the plan is bound by the commitment's digest. A session the books veto has no
case. And a session over the `C*` of PoUS's own setup names PoUS's store: onsite's `same_store` takes the two as one
store, and the Lean accepts the case."""

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

from test_p2_party import SESSION_ID, TOY, Lying, Prover, exe, request, run, signer, store  # noqa: F401
from verity.commitments import domains
from verity.commitments.crypto import lms
from verity.commitments.identity import canonical_json_bytes
from verity.protocols.accounting.pous.device import memory_challenger as mc
from verity.protocols.accounting.pous.verifier import period as pous
from verity.protocols.accounting.pous.verifier import run as R
from verity.protocols.accounting.pous.verifier import streamed_wipe as sw
from verity.protocols.accounting.poub.verifier import p2_case, p2_lean, p2_party, receipts

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PACKAGE = Path(__file__).resolve().parents[1]
SIGNER = Path(__file__).resolve().parents[7] / "verity/protocols/accounting/pous/verifier/tests/pous_test_signer.py"
COINS = hashlib.sha512(b"the period's live coins").digest()
OTHER_SESSION = hashlib.sha512(b"p2-case-test/another session").digest()[:32]
UNTRUSTED = "the close is signed by a key that is not a trusted memory challenger's"
UNCHAINED = "the close's chain is not the case's"
UNSIGNED = "the close's signature does not verify on the params' session"


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


def session(exe, store, *, plan=None, lean_of=lambda lean: lean, auditor=p2_party.SESSION,
            **kw) -> tuple[dict, dict | None]:
    """One session of `test_p2_party.run` with a `Keeping` prover, held to the auditor's `auditor` (`SESSION`, whose
    case alone the replay takes): the answer and the case it kept."""
    kept = []

    def prover_of(clock):
        kept.append(Keeping(store, clock, **kw))
        return kept[-1]

    proved, answer = run(exe, store, prover_of, request(store, auditor), plan, lean_of, auditor)
    assert proved == answer
    return answer, kept[0].case_


def replayed(lean, case) -> dict:
    return p2_case.serve(lean, [case])[0]


def _evidence(case) -> dict:
    return json.loads(bytes.fromhex(case["evidence"]))


def _with(case, **edit) -> dict:
    """`case` with its evidence's `log` or `close` replaced."""
    return case | {"evidence": canonical_json_bytes(_evidence(case) | edit).hex()}


def _with_log(case, log, by=None) -> dict:
    """`case` with `log`, its close signed again over it (by `signer()`, or `by`) on the params' session."""
    close = p2_case.close_of(by or signer(), bytes.fromhex(case["params"]["session"]), p2_case.head_of(case), log)
    return _with(case, log=log, close=close)


def _flip(h: str) -> str:
    return h[:-2] + ("00" if h[-2:] != "00" else "01")


def test_an_honest_sessions_case_is_accepted_with_its_credited_reads(exe, lean, store) -> None:
    answer, case = session(exe, store)
    assert answer["verdict"] == "accept" and case is not None
    got = replayed(lean, case)
    assert got == p2_case.answer(case, answer)
    assert got["verdict"] == "accept" and got["protocol"] == p2_case.PROTOCOL
    assert got["public"]["q_bytes"] == answer["q_bytes"] > 0 and got["public"]["span_ns"] == answer["span_ns"]
    assert got["commitment"] == p2_case.digest(case).hex()
    assert sorted(case) == ["commitment", "evidence", "params", "store_root"]
    assert sorted(case["params"]) == ["not_before_ns", "request", "session", "trusted"]
    assert case["params"]["session"] == SESSION_ID.hex() and case["params"]["trusted"] == [signer().public_key.hex()]
    evidence = _evidence(case)
    assert sorted(evidence) == ["close", "log"] and sorted(evidence["close"]) == ["chain", "lines", "public_key",
                                                                                  "signature"]
    assert evidence["close"]["lines"] == len(evidence["log"])
    assert evidence["close"]["chain"] == p2_case.chain(p2_case.head_of(case), evidence["log"]).hex()
    assert p2_party.case_digest(case) == evidence["close"]["chain"]
    coins = [x["coin"] for x in evidence["log"] if x["op"] == "p2_event" and x["coin"] is not None]
    assert coins and all(len(bytes.fromhex(c)) == p2_case.COIN_BYTES == 64 for c in coins)
    assert case["store_root"] == store.registration.root.hex() == p2_case.store_root(case).hex()
    assert case["params"]["request"] == request(store, p2_party.SESSION)
    ops = [x["op"] for x in _evidence(case)["log"]]
    windows, passes = p2_party.SESSION["windows"], p2_party.SESSION["passes"]
    assert ops[0] == "p2_event" and ops.count("p2_openings") == windows and ops.count("p2_root") == windows * passes


def test_a_toy_partys_case_is_refused_naming_each_value_off_the_auditors_session(exe, lean, store) -> None:
    """A session a toy party held to its own parameters (`test_p2_party.TOY`), accepted live: its case's replay holds
    the request to `SESSION` alone, so it is refused, naming every value, before its log; `p2_case.answer`, which takes
    the live verifier's word, answers it as its session was."""
    answer, case = session(exe, store, auditor=TOY)
    assert answer["verdict"] == "accept" and p2_case.answer(case, answer)["verdict"] == "accept"
    got = replayed(lean, case)
    why = "; ".join(f"the request's {k} is not the auditor's ({TOY[k]}, not {v})" for k, v in p2_party.SESSION.items())
    assert got["verdict"] == "reject" and got["reason"] == f"the request is refused: {why}", got
    assert got["public"] is None and got["commitment"] == p2_case.digest(case).hex()


@pytest.fixture(scope="module")
def honest(exe, store) -> dict:
    answer, case = session(exe, store)
    assert answer["verdict"] == "accept"
    return case


@pytest.mark.parametrize("off", [{"deadline_ns": 2 * 10 ** 9}, {"passes": 1}, {"samples": 1}, {"window_ns": 10 ** 9},
                                 {"windows": 1}, {"passes": 1, "windows": 1}], ids=lambda off: "+".join(off))
def test_a_cases_request_edited_off_the_auditors_session_is_refused_by_name(lean, honest, off) -> None:
    """An honest case of `SESSION`'s session, its request edited after (which no digest binds) to a longer allowance,
    fewer passes, windows or openings or a shorter window: refused naming each value, never replayed at the edit's."""
    edited = honest | {"params": honest["params"] | {"request": honest["params"]["request"] | off}}
    got = replayed(lean, edited)
    why = "; ".join(f"the request's {k} is not the auditor's ({off[k]}, not {v})" for k, v in p2_party.SESSION.items()
                    if k in off)
    assert got["verdict"] == "reject" and got["reason"] == f"the request is refused: {why}", got
    assert replayed(lean, honest)["verdict"] == "accept"


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
    """A root line or a coin, each refused by the close as it stands and rejected by the replay under a close signed
    over the edit; an unknown op and a log out of canonical form."""
    _, case = session(exe, store)
    log = _evidence(case)["log"]
    k = next(k for k, x in enumerate(log) if x["op"] == "p2_root")
    root = json.loads(bytes.fromhex(log[k]["line"]))["root"]
    other = json.dumps({"root": _flip(root)}).encode() + b"\n"
    edits = {
        "root": log[:k] + [log[k] | {"line": other.hex()}] + log[k + 1:],
        "coin": [log[0] | {"coin": "00" * 64}] + log[1:],
    }
    for name, edited in edits.items():
        got = replayed(lean, _with(case, log=edited))
        assert got["verdict"] == "reject" and got["reason"] == UNCHAINED, (name, got)
        got = replayed(lean, _with_log(case, edited))
        assert got["verdict"] == "reject" and got["reason"] != UNCHAINED, (name, got)
    got = replayed(lean, _with_log(case, log + [{"op": "p2_hold"}]))
    assert got["verdict"] == "reject" and got["reason"] == "evidence: log: no op p2_hold"
    got = replayed(lean, case | {"evidence": json.dumps(_evidence(case)).encode().hex()})
    assert got["verdict"] == "reject" and "not canonical JSON" in got["reason"]


def _close_edits(case) -> dict:
    ev = _evidence(case)
    log, close, n = ev["log"], ev["close"], len(ev["log"])
    other = signer(b"p2-case-test/another memory challenger")
    params = case["params"]
    return {
        "untrusted": (_with(case, close=p2_case.close_of(other, SESSION_ID, p2_case.head_of(case), log)), UNTRUSTED),
        "trusted-list": (case | {"params": params | {"trusted": [other.public_key.hex()]}}, UNTRUSTED),
        "lines": (_with(case, close=close | {"lines": n + 1}), f"the close counts {n + 1} lines, not the log's {n}"),
        "log-longer": (_with(case, log=log + log[-1:]), f"the close counts {n} lines, not the log's {n + 1}"),
        "chain": (_with(case, close=close | {"chain": _flip(close["chain"])}), UNCHAINED),
        "signature": (_with(case, close=close | {"signature": _flip(close["signature"])}), UNSIGNED),
        "params-session": (case | {"params": params | {"session": OTHER_SESSION.hex()}}, UNSIGNED),
        "signed-on-another-session": (
            _with(case, close=p2_case.close_of(signer(), OTHER_SESSION, p2_case.head_of(case), log)), UNSIGNED),
        "not-before": (case | {"params": params | {"not_before_ns": params["not_before_ns"] + 1}}, UNCHAINED),
    }


@pytest.mark.parametrize("edit", ["untrusted", "trusted-list", "lines", "log-longer", "chain", "signature",
                                  "params-session", "signed-on-another-session", "not-before"])
def test_a_close_that_does_not_sign_the_log_is_refused_by_name_before_the_replay(lean, honest, edit) -> None:
    """The honest case's close by another key (or a params' list that doesn't trust the key), a line count or a chain
    that isn't the case's, a signature edited, a close replayed under another session, on either side, and a clock
    floor edited after the close: each refused, naming why, before any entry of the log is replayed."""
    case, why = _close_edits(honest)[edit]
    got = replayed(lean, case)
    assert got["verdict"] == "reject" and got["reason"] == why, got
    assert got["public"] is None and got["commitment"] == p2_case.digest(honest).hex()


def test_the_check_trusts_the_keys_its_params_name(lean, honest) -> None:
    """The same log, signed by another key that the params trust instead, is accepted: the params' list, never the
    close, says which keys are trusted."""
    other = signer(b"p2-case-test/another memory challenger")
    case = _with_log(honest, _evidence(honest)["log"], by=other)
    case = case | {"params": case["params"] | {"trusted": [other.public_key.hex()]}}
    assert replayed(lean, case)["verdict"] == "accept"


def test_a_coin_that_is_not_the_challengers_whole_coin_is_refused_by_name(lean, honest) -> None:
    log = _evidence(honest)["log"]
    k = next(k for k, x in enumerate(log) if x["op"] == "p2_event" and x["coin"] is not None)
    short = log[:k] + [log[k] | {"coin": log[k]["coin"][:64]}] + log[k + 1:]
    got = replayed(lean, _with_log(honest, short))
    assert got["verdict"] == "reject" and got["reason"] == f"log entry {k}: a coin is the challenger's 64 bytes, not 32"


def _pous_record() -> bytes:
    c = mc.SessionCommitment(SESSION_ID, b"\x01" * 64, b"\x02" * 32, 9, 2, 10 ** 6, 8224, b"\x03" * 64)
    return mc.message(c, "complete", [mc.Round(0, 3, 0, 10, b"\x04" * 64)])


def _wipe_record() -> bytes:
    c = sw.WipeCommitment(b"\x05" * 32, SESSION_ID, sw.WipeParams(4, 2, 1, 10 ** 6, (("hbm", 1024),)))
    return sw.message(c, "complete", b"", 0, 10, [])


@pytest.mark.parametrize("record", [_pous_record, _wipe_record], ids=["pous-session", "wipe"])
def test_another_subprotocols_record_signed_for_the_session_is_refused_as_a_close(lean, honest, record) -> None:
    """The memory challenger's key signs a PoUS session record, or a wipe record, for the case's session id; put in
    the case as its close, it is refused by name: those records are 64-byte digests, and P2's is raw bytes of another
    length (`RemoteAudit.P2Close.closeRecord_ne_pous`, `closeRecord_ne_wipe`)."""
    r = record()
    assert len(r) == 64 != len(p2_case.close_record(0, bytes(64)))
    close = _evidence(honest)["close"] | {"signature": signer().sign(SESSION_ID, r).hex()}
    got = replayed(lean, _with(honest, close=close))
    assert got["verdict"] == "reject" and got["reason"] == UNSIGNED, got


def test_the_close_bytes() -> None:
    """P2's tag, NUL-ended and a prefix of no other separator nor any of them of it; the record, raw bytes of the tag's
    length plus 72; the message, the role's (`lms.memory_challenger_message`)."""
    tag = b"verity/remote-bandwidth/p2/close/v0\0"
    assert p2_case.CLOSE_TAG == tag and len(tag) == 36
    others = [v.encode() for k, v in domains.TABLE.items() if k != "REMOTE_BANDWIDTH_P2_CLOSE"]
    others += [mc.SESSION_TAG, sw.SESSION_TAG]
    assert not [o for o in others if tag.startswith(o) or o.startswith(tag[:-1])]
    chain = bytes([0xAB]) * 64
    record = p2_case.close_record(3, chain)
    assert record == tag + (3).to_bytes(8, "big") + chain and len(record) == 108
    key = signer().key
    assert lms.memory_challenger_message(key.I, SESSION_ID, record) == hashlib.sha512(
        (34).to_bytes(8, "big") + b"verity/memory-challenger-record/v1" + key.I
        + (40).to_bytes(8, "big") + (32).to_bytes(8, "big") + SESSION_ID
        + (108).to_bytes(8, "big") + record).digest()
    head = p2_case.head(b"{}", 7)
    assert head == hashlib.sha512(b"verity/remote-bandwidth/p2/commitment/v0\0{}").digest() + b"7" and len(head) == 65
    c0 = hashlib.sha512((36).to_bytes(8, "big") + tag).digest()
    assert p2_case.chain(head, []) == hashlib.sha512(c0 + (65).to_bytes(8, "big") + head).digest()


def test_the_commitment_binds_the_plan(exe, lean, store) -> None:
    """A session under a plan of fewer units is accepted with those units' reads alone, and its digest is not the
    whole store's: the same log under the whole store's registration file is another commitment, which the close
    refuses, since the case's chain begins with the commitment's digest."""
    units = store.registration.blocks
    answer, case = session(exe, store, plan=receipts.Plan(((1, units - 1),), units))
    got = replayed(lean, case)
    assert got["verdict"] == "accept" and got["profile"]["plan_units"] == units - 1 == answer["plan_units"]
    whole = p2_party.registration_json(store.registration, receipts.Plan.whole(units))
    swapped = case | {"commitment": json.dumps(whole, sort_keys=True, separators=(",", ":")).encode().hex()}
    assert p2_case.digest(swapped) != p2_case.digest(case)
    got = replayed(lean, swapped)
    assert got["verdict"] == "reject" and got["reason"] == UNCHAINED, got


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
    held = pous.EncodedStore(R.Point(q, 8, pous.LIMIT_NS), R.Setup(q.blocks, reg.commitment, reg.root, reg.salt),
                             store.blocks, b"".join(mc.block_digest(b) for b in store.blocks),
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
