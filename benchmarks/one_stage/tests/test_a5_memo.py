"""a5's memo (`a5_memo.memo`), each of the one-stage verifier's pure checks asked once per input. Under it a5's negatives
on the challenger's coins, each its own `service.Stream` on the honest draw's bytes (`a5_coins.Replay`) registering the
same record, reuse the registration's checks and draw, and each outcome is the one it has without the memo, refused at
the Lean run's select (`coins`); a changed registration or copy (a root, a root's domain, the ports bound to another run,
a program of one proof block fewer) is checked again and refused as before; a verdict the Lean refused is never kept, and
a hit is a copy. A toy window of two templates."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R, verifier as V

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5_coins as AC  # noqa: E402
import a5_memo as AM  # noqa: E402

XOR8 = PrimitiveDefinition("A5MemoTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("A5MemoTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
TRIPLE = bind(CompositeDefinition("A5MemoTestTriple", 0, (), lambda S_: ((("x", Array(3, Bits(8))),), Bits(8)),
                                  lambda B, S_, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
WINDOW = (PAIR, TRIPLE)
QUERY = P.template_instances_query(WINDOW)
COUNTS = {P.template_id(PAIR): 1, P.template_id(TRIPLE): 9}
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-a5-memo-test"
ROOT = "ab" * 64
SESSION = "5e" * 64
VALUES = (S.Value("x", SCHEMA, "serve"), S.Value("out", SCHEMA, "serve"))
PUBLIC = tuple(S.Public(i, "structure", r) for i, r in (
    ("population", "N, the draw's range"), ("window", "provenance"), ("leaves:x", "x's leaf count"),
    ("domain:x", "x's served domain"), ("leaves:out", "out's leaf count"), ("domain:out", "out's served domain"),
    *((f"population:{P.template_id(t)}", "a template's stratum") for t in WINDOW)))
SPEC = S.Spec(VALUES, S.Law("stratified:3"),
              S.Check(tuple(P.program_digest(Program(t)) for t in WINDOW), QUERY, proof_deadline=60.0), PUBLIC,
              bytes.fromhex(ROOT))
VERDICTS = {"lean": {"of_record": True, "accepted": True, "digest": "dd" * 64, "sessions": [SESSION]}}
READ = bytes([0x5a]) * 4096
#: each negative's coins' evidence and window, as a5's on the challenger's coins pass them (`a5.coins_outcome`)
NEGATIVES = {"coins-no-evidence": (None, ROOT), "coins-stray-evidence": ({"segment": {"openings": [], "start": 0}}, ROOT),
             "coins-for-another-window": (None, "00" * 64)}


def _program(counts: dict = COUNTS) -> Program:
    return P.mixed_population_program([(t, counts[P.template_id(t)]) for t in WINDOW])


def _ports(run: str = RUN) -> dict:
    part = P.partition(_program(), QUERY)
    n = sum(COUNTS.values())
    return {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, SCHEMA, leaves, run)}
            for p, leaves in (("x", n), ("out", 2 * n))}


def _record() -> dict:
    ports = _ports()
    roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **ports[p]} for i, p in enumerate(ports)]
    return R.record(_program(), QUERY, {"law": "stratified", "k": 3}, S.SCHEME, roots, {"kind": "served", "run": RUN, "row": 0})


def _check(rec: dict | None = None, program: Program | None = None, ports: dict | None = None):
    """R1-R7 by the verifier on the toy window's registration and the verifier's copies, each a new object each call."""
    law = S.selection_law(SPEC.selection, sum(COUNTS.values()))
    return V.check(rec or _record(), program or _program(), QUERY, law, S.SCHEME, ports or _ports())


def _asked(monkeypatch) -> Counter:
    """How many times each piece is asked of the Lean from here on."""
    asked, lean = Counter(), V.lean

    def counted(fn, value):
        asked[fn] += 1
        return lean(fn, value)
    monkeypatch.setattr(V, "lean", counted)
    return asked


def _negatives() -> dict[str, dict]:
    """a5's negatives on the challenger's coins: each its own stream on the honest draw's bytes, registering the same record
    (in its window), drawing, and its outcome with its evidence. -> by name: accepted, the Lean run's side, the record."""
    out = {}
    for name, (coins, window) in NEGATIVES.items():
        s = S.Stream(SPEC, WINDOW, budget=1, delta=0.5, challenger="a5 memo test, challenger", coins=AC.Replay(READ),
                     clock=lambda: 0.0)
        d = s.select(s.register(_record(), _ports(), window_root=window, counts=COUNTS))
        assert s.deliver(d, [SESSION]) is None
        got = s.outcome(d, VERDICTS, {"statement": "stub"}, coins=coins)
        audit = s.records[-1]["audit"]
        out[name] = {"accepted": got.accepted, "lean": AC.lean_side(audit), "audit": audit}
    return out


def test_under_the_memo_the_negatives_reuse_the_registrations_checks_and_are_refused_as_without_it(monkeypatch):
    """Without the memo every negative's stream asks the Lean its registration's public-items rule, R1-R7 and the select
    itself; under it the negatives in one window ask each once, the one in another window its R1-R7 again, while each
    stream's whole run (`run`) is still its own. Every negative's outcome, its Lean side and its audit record are the ones
    it has without the memo: refused, at the Lean run's select (`coins`) where it is given no evidence. Afterwards the
    verifier's checks are its own again."""
    asked = _asked(monkeypatch)
    plain = _negatives()
    without = Counter(asked)
    asked.clear()
    own = {name: getattr(V, name) for name in AM.PURE}
    with AM.memo() as m:
        kept = _negatives()
    assert kept == plain
    assert not any(n["accepted"] for n in kept.values()), kept
    for name in ("coins-no-evidence", "coins-for-another-window"):
        assert (kept[name]["lean"]["step"], kept[name]["lean"]["codes"]) == AC.SELECT_COINS
    n = len(NEGATIVES)
    assert (without["check_public"], without["registration"], without["select"], without["run"]) == (n, n, n, n)
    assert (asked["check_public"], asked["registration"], asked["select"], asked["run"]) == (1, 2, 1, n)
    assert m.counts()["reused"]["check"] == n - 2 and not m.counts()["unkeyed"]
    assert {name: getattr(V, name) for name in AM.PURE} == own and AM.counts() is None


def test_under_the_memo_a_changed_registration_or_copy_is_checked_again(monkeypatch):
    """The memo keys R1-R7 by everything it reads: the same registration and copies, as new objects and a new Program,
    reuse the verdict, a copy; a registration with one root changed is asked of the Lean again, and one with a root's
    domain changed (R3), the verifier's ports bound to another run (R3) and a program of one proof block fewer (R2) are
    asked again and refused."""
    asked = _asked(monkeypatch)
    with AM.memo():
        first = _check()
        assert first.ok and asked["registration"] == 1
        first.detail["changed"] = True
        again = _check()
        assert again.ok and "changed" not in again.detail and asked["registration"] == 1
        rec = _record()
        rec["roots"][0]["root"] = "cd" * 64
        assert _check(rec).ok and asked["registration"] == 2
        rec = _record()
        rec["roots"][0]["domain"] = "cd" * 64
        assert "R3-domain" in _check(rec).codes and asked["registration"] == 3
        assert "R3-domain" in _check(ports=_ports("r-another-run")).codes and asked["registration"] == 4
        fewer = _check(program=_program({**COUNTS, P.template_id(TRIPLE): COUNTS[P.template_id(TRIPLE)] - 1}))
        assert "R2-program" in fewer.codes and asked["registration"] == 5


def test_the_memo_keeps_no_verdict_the_lean_refused(monkeypatch):
    """A check whose Lean side refused (here, the Lean down once) is not kept: the same check again asks the Lean, and
    its answer is kept."""
    asked, lean = Counter(), V.lean

    def down_once(fn, value):
        asked[fn] += 1
        return {"refused": "down"} if fn == "registration" and asked[fn] == 1 else lean(fn, value)
    monkeypatch.setattr(V, "lean", down_once)
    with AM.memo():
        assert _check().codes == (V.DISAGREE,)
        assert _check().ok and asked["registration"] == 2
        assert _check().ok and asked["registration"] == 2
