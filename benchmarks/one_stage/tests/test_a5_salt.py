"""The window's seed salt as a5's harnesses carry it (`a5_coins`), on the fake challenger (`fake_challenger.Challenger`):
the session's open releases it, the rank's stream takes it at its release (`RankCoins.take_salt`), the rank's registration
names it (`salted`), and the run of record reads it from the challenger's log (`seed_salt_evidence`'s form, read by
`Run.seedSaltOf`). An honest salted window is accepted end to end, and so is its case in the challenger's form
(`case_form`) by `replay.python`. With no evidence of the salt the outcome fails `run`, the Lean refusing at its register
(`seed-salt`), and the case is refused as `seedSaltOf` missing; the challenger's negatives (`challenger_negatives`) are
refused. A toy window of two templates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.spot import challenger as Ch, coins as CC, service as S
from verity.core.protocols.spot.one_stage import draw as D, partition as P, registration as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "verity" / "core" / "protocols" / "spot" / "tests"))
import a5_coins as AC  # noqa: E402
import fake_challenger as FC  # noqa: E402

_spec = importlib.util.spec_from_file_location("one_stage_replay", HERE.parent / "replay.py")
RP = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = RP
_spec.loader.exec_module(RP)

XOR8 = PrimitiveDefinition("A5SaltTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("A5SaltTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
TRIPLE = bind(CompositeDefinition("A5SaltTestTriple", 0, (), lambda S_: ((("x", Array(3, Bits(8))),), Bits(8)),
                                  lambda B, S_, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
WINDOW = (PAIR, TRIPLE)
QUERY = P.template_instances_query(WINDOW)
COUNTS = {P.template_id(PAIR): 1, P.template_id(TRIPLE): 9}
N = sum(COUNTS.values())
PROGRAM = P.mixed_population_program([(t, COUNTS[P.template_id(t)]) for t in WINDOW])
LAW = {"law": "stratified", "k": 3}
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-a5-salt-test"
ROOT = "ab" * 64
SESSION = "5e" * 64
VALUES = (S.Value("x", SCHEMA, "serve"), S.Value("out", SCHEMA, "serve"))
PUBLIC = tuple(S.Public(i, "structure", r) for i, r in (
    ("population", "N, the draw's range"), ("window", "provenance"), ("leaves:x", "x's leaf count"),
    ("domain:x", "x's served domain"), ("leaves:out", "out's leaf count"), ("domain:out", "out's served domain"),
    *((f"population:{P.template_id(t)}", "a template's stratum") for t in WINDOW))) + (AC.SEED_ITEM,)
SPEC = S.Spec(VALUES, S.Law("stratified:3"),
              S.Check(tuple(P.program_digest(Program(t)) for t in WINDOW), QUERY, proof_deadline=60.0), PUBLIC,
              bytes.fromhex(ROOT))
VERDICTS = {"lean": {"of_record": True, "accepted": True, "digest": "dd" * 64, "sessions": [SESSION]}}
STATEMENT = {"statement": "stub"}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Draws:
    """`a5_coins.Draws` as one rank reads it, on the fake challenger: the session's open releases the window's seed salt
    (`coins.draw_seed_salt`, on a challenger of the seed-salt family opened for `ROOT`), which `plan` returns with its
    release on the streams' clock, and the sample's coins are a challenger of `a5_coins.FAMILY` whose session opens at
    the registration's digest (`commit`)."""

    def __init__(self, clock: Clock) -> None:
        self.seeds = FC.Challenger("a5-salt/seed-salt", 2, CC.SEED_SALT_FAMILY, window={"root": ROOT})
        self.salt, self.at = CC.draw_seed_salt(self.seeds, ROOT), clock()
        self.coins = FC.Challenger("a5-salt", 64, AC.FAMILY, window={"root": ROOT})

    def plan(self, rank: int, coins: int) -> tuple[str, float]:
        return self.salt.hex(), self.at

    def commit(self, rank: int, after: str) -> None:
        self.coins.commit(bytes.fromhex(after))

    def read(self, rank: int, n: int) -> bytes:
        return self.coins(n)

    def drawn(self, rank: int) -> dict:
        return self.coins.segments[-1].to_json()


def _ports() -> dict:
    part = P.partition(PROGRAM, QUERY)
    return {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, SCHEMA, leaves, RUN)}
            for p, leaves in (("x", N), ("out", 2 * N))}


def _record() -> dict:
    ports = _ports()
    roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **ports[p]} for i, p in enumerate(ports)]
    return R.record(PROGRAM, QUERY, LAW, S.SCHEME, roots, {"kind": "served", "run": RUN, "row": 0})


def stream_of(coins: CC.ChallengerCoins, clock: Clock) -> S.Stream:
    return S.Stream(SPEC, WINDOW, budget=1, delta=0.5, challenger="a5 salt test, challenger", coins=coins, clock=clock)


def register(stream: S.Stream, rec: dict, window: str = ROOT) -> S.WindowCommitmentRecord:
    return stream.register(rec, _ports(), window_root=window, counts=COUNTS)


def decide(stream: S.Stream, draw: S.Draw, ev: dict | None, salt_ev: dict | None) -> tuple[S.Outcome, dict]:
    """The sample's proof delivered and its outcome on the coins' evidence `ev` and the salt's `salt_ev`, as a5 decides
    it (`a5.coins_outcome`): -> (the outcome, its audit record)."""
    assert stream.deliver(draw, [SESSION]) is None
    return stream.outcome(draw, VERDICTS, STATEMENT, coins=ev, seed_salt=salt_ev), stream.records[-1]["audit"]


def pre_coin() -> dict:
    """The coin after the window's root that `Draws` opens before any registration (`Draws.pre`), on a challenger of its
    own."""
    pre = FC.Challenger("a5-salt/pre", 2, AC.FAMILY, window={"root": ROOT})
    pre.commit(bytes.fromhex(ROOT))
    pre(Ch.COIN_BYTES)
    return pre.segments[-1].to_json()


def case_of(w: SimpleNamespace, form: dict) -> RP.Case:
    """The audit as a5 writes its case on the challenger's coins (`a5.write_case`): its params naming the challenger and
    the window's open root, its evidence the coins' and the salt's, as `form` (`case_form`'s) gives them."""
    proved = sorted(D.proved(w.draw.draw))
    p = RP.params(PROGRAM, QUERY, LAW, _ports(), None, "1/2", scheme=S.SCHEME, challenger=form.get("challenger"),
                  window_root=form.get("window_root"))
    e = RP.evidence({"draw": dict(w.draw.draw), "proved_units": proved, "receipt": dict(w.draw.registration.receipt),
                     "statement": STATEMENT, "verdicts": VERDICTS}, instances=len(proved), sessions=[SESSION],
                    session_roots=None, served=None, coins=form.get("coins"), seed_salt=form.get("seed_salt"))
    return RP.Case("", "a5", w.rec["format"], "audit", "audit", PROGRAM, p, w.rec, e, None)


@pytest.fixture
def window() -> SimpleNamespace:
    """A rank's window sampled as a5 samples it (`a5.coins_draw`): its plan opens the session, which releases the salt;
    its stream takes the salt (`take_salt`) and its registration names it (`salted`); the coins commit to the
    registration's digest and the stream samples once. The session's evidence of the coins and of the salt comes after."""
    clock = Clock()
    draws = Draws(clock)
    coins = AC.RankCoins(draws, 0)
    stream = stream_of(coins, clock)
    coins.plan(8)
    rec = AC.salted(_record(), coins.take_salt(stream, ROOT))
    reg = register(stream, rec)
    coins.commit(bytes.fromhex(reg.digest))
    draw = stream.select(reg)
    return SimpleNamespace(clock=clock, coins=coins, stream=stream, rec=rec, draw=draw,
                           ev=draws.coins.evidence(reg.digest), salt_ev=draws.seeds.evidence(ROOT))


def test_an_honest_salted_window_is_accepted_end_to_end(window):
    """The registration names the salt the session released at the window's open; the outcome on the session's evidence
    of the coins and of the salt is accepted, the Lean's whole run agreeing; Python reads both as the run does."""
    w = window
    assert w.rec["window"][R.SEED_SALT] == w.coins.salt.hex() and w.draw.registration.record == w.rec
    got, audit = decide(w.stream, w.draw, w.ev, w.salt_ev)
    assert got.accepted and audit["verifiers"]["agree"] and "run" not in audit["checks"], audit["checks"]
    after = bytes.fromhex(w.draw.registration.digest)
    assert CC.coins_reason(w.ev, ROOT, after, bytes(w.coins.read), AC.lms_verify) is None
    assert AC.seed_salt_reason(w.salt_ev, ROOT, w.coins.salt.hex()) is None


def test_a_run_with_no_evidence_of_the_salt_is_refused_as_missing(window):
    """The same sample decided on the coins' evidence and none of the salt's: the outcome fails `run`, the Lean refusing
    at its register (`seed-salt`: `Run.seedSaltOf` finds the evidence missing), and Python reads it missing too."""
    w = window
    got, audit = decide(w.stream, w.draw, w.ev, None)
    assert not got.accepted and audit["checks"]["run"]["ok"] is False
    side = AC.lean_side(audit)
    assert (side["step"], side["codes"]) == AC.REGISTER_SEED_SALT
    assert AC.seed_salt_reason(None, ROOT, w.coins.salt.hex()) == AC.SEED_SALT_MISSING


def test_the_case_in_the_challengers_form_is_accepted_and_without_the_salt_refused_as_missing(window):
    """`replay.python` (`OneStage.check`'s order) accepts the case in the challenger's form, refuses it as
    `seedSaltOf` missing without the salt's evidence, and refuses the coins read on another window."""
    w = window
    form = AC.case_form(w.ev, w.salt_ev, ROOT)
    assert form["challenger"] == {"key": w.ev["key"], "label": AC.FAMILY} and form["window_root"] == ROOT
    assert "key" not in form["coins"] and "key" not in form["seed_salt"]
    honest = RP.python(case_of(w, form))
    assert honest["verdict"] == "accept", honest["reason"]
    missing = RP.python(case_of(w, {k: v for k, v in form.items() if k != "seed_salt"}))
    assert (missing["verdict"], missing["reason"]) == (
        "reject", "registration: seed salt: missing: the verifier names a challenger and a window root, and the evidence "
                  "carries no seed salt")
    other = RP.python(case_of(w, {**form, "window_root": AC.ANOTHER_WINDOW}))
    assert other["verdict"] == "reject" and other["reason"].startswith("coins: "), other["reason"]


def test_the_challengers_negatives_are_refused(window):
    """Each of the challenger's negatives, its own stream's on the honest sample's bytes: the altered coins' evidence
    refused at the Lean's select (`coins`), the honest evidence for another window and a run with no salt evidence at its
    register (`seed-salt`), and Python refusing each."""
    w = window
    pre = pre_coin()
    neg = AC.challenger_negatives(lambda c: stream_of(c, w.clock), lambda s, root: register(s, w.rec, root), decide,
                                  w.draw, w.coins, w.ev, w.salt_ev, pre, ROOT)
    assert set(neg) == {*AC.negatives(w.ev, pre), "coins-for-another-window", "seed-salt-missing"}
    assert all(n["refused"] for n in neg.values()), {k: v for k, v in neg.items() if not v["refused"]}
    missing = neg["seed-salt-missing"]
    assert missing["python"] == AC.SEED_SALT_MISSING
    assert (missing["lean"]["step"], missing["lean"]["codes"]) == AC.REGISTER_SEED_SALT


def test_a_rank_takes_the_salt_once_its_plan_has_opened_the_session():
    clock = Clock()
    coins = AC.RankCoins(Draws(clock), 0)
    with pytest.raises(Ch.CoinError, match="plans its draw first"):
        coins.take_salt(stream_of(coins, clock), ROOT)


def test_salted_names_the_salt_once_and_the_case_form_needs_both_evidences():
    """`salted` adds the salt to the record's window and leaves a window that names one for R6 to hold; `case_form` is
    the challenger's form only with both the coins' and the salt's evidence."""
    rec = _record()
    s = AC.salted(rec, "aa" * 64)
    assert s["window"] == {**rec["window"], R.SEED_SALT: "aa" * 64} and R.SEED_SALT not in rec["window"]
    assert AC.salted(s, "bb" * 64) == s
    ev = {"key": {"k": 1}, "ledger": [], "main": [], "pins": [], "segment": {}, "session": {}}
    assert AC.case_form(ev, None, ROOT) is None and AC.case_form(None, ev, ROOT) is None
