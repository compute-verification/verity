"""`stream.sh`'s workload: the one-stage verifier's checks (`one_stage.verifier`: the Lean of record,
`sampled-proofs-verifier`, beside the Python) on a5's path for a sample on the verifier's own coins (`a5.audit` with
``--coins verifier``) against the registration's ports, as a served window registers a root per member port. In a5's
order: R1-R7 before the sample, a `service.Stream`'s registration, draw and outcome (`a5.verifier_outcome`, with the
sessions' roots of `SESSION_PORTS` ports and the drawn units served), the
negatives on the verifier's coins (`a5_coins.verifier_negatives`), then R2's and R3's on the copies (a program of one
proof block fewer, the ports bound to another run). A toy window of two templates with N ports; per N, a row with
the checks asked once per input as a5 asks them (`a5_memo.memo`), or with ``--no-memo`` each asked every time: whether
the audit accepted and every negative was refused, and per check (`verifier.*`) and per Lean piece its calls and
seconds.

  python stream.py [--no-memo] N...
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/one_stage"))

import a5_coins as AC  # noqa: E402
import a5_memo as AM  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402
from verity.ml.circuits.types import Array, Value as Bits  # noqa: E402
from verity.core.protocols.sampled_proofs import coins as CC, service as S  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R, verifier as V  # noqa: E402

XOR8 = PrimitiveDefinition("VerifierCostStreamXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("VerifierCostStreamPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
TRIPLE = bind(CompositeDefinition("VerifierCostStreamTriple", 0, (), lambda S_: ((("x", Array(3, Bits(8))),), Bits(8)),
                                  lambda B, S_, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
WINDOW = (PAIR, TRIPLE)
QUERY = P.template_instances_query(WINDOW)
COUNTS = {P.template_id(PAIR): 1, P.template_id(TRIPLE): 9}
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-verifier-cost-stream"
WINDOW_ROOT = "ab" * 64
SESSION = "5e" * 64
VERDICTS = {"lean": {"of_record": True, "accepted": True, "digest": "dd" * 64, "sessions": [SESSION]}}
CHECKS = ("check", "check_public", "select", "check_draw", "check_verdicts", "audit_record", "run", "stream")
#: the ports the sample's sessions report roots for, as a5 gives the outcome its sessions' roots
SESSION_PORTS = 8


def program(counts: dict = COUNTS) -> Program:
    return P.mixed_population_program([(t, counts[P.template_id(t)]) for t in WINDOW])


def ports(n: int, run: str = RUN) -> dict[str, dict]:
    """The verifier's `n` ports, each's leaves and domain under `run`."""
    part = P.partition(program(), QUERY)
    out = {}
    for i in range(n):
        p, leaves = f"p{i:07d}", 1 + i % 64
        out[p] = {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, SCHEMA, leaves, run)}
    return out


def spec(names: list[str]) -> S.Spec:
    public = [("population", "N"), ("window", "provenance"), *((f"population:{P.template_id(t)}", "a stratum") for t in WINDOW),
              *((f"{x}:{p}", "a port's layout") for p in names for x in ("leaves", "domain"))]
    return S.Spec(tuple(S.Value(p, SCHEMA, "serve") for p in names), S.Law("stratified:3"),
                  S.Check(tuple(P.program_digest(Program(t)) for t in WINDOW), QUERY, proof_deadline=60.0),
                  tuple(S.Public(i, "structure", why) for i, why in public), bytes.fromhex(WINDOW_ROOT))


class Timed:
    """Calls and seconds by name."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.s: dict[str, float] = defaultdict(float)

    @contextlib.contextmanager
    def on(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.calls[name] += 1
            self.s[name] += time.perf_counter() - t

    def wrap(self, name: str, fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            with self.on(name):
                return fn(*args, **kwargs)
        return call

    def rows(self) -> dict[str, dict]:
        return {k: {"calls": self.calls[k], "s": round(self.s[k], 3)} for k in sorted(self.calls)}


def row(n: int, memo: bool) -> dict:
    mine = ports(n)
    roots = [{"port": p, "root": f"{i:0128x}", **e} for i, (p, e) in enumerate(mine.items())]
    rec = R.record(program(), QUERY, {"law": "stratified", "k": 3}, S.SCHEME, roots, {"kind": "served", "run": RUN, "row": 0})
    sp, other = spec(sorted(mine)), ports(n, "r-another-run")
    sessions = {r["port"]: r["root"] for r in roots[-SESSION_PORTS:]}
    law = S.selection_law(sp.selection, sum(COUNTS.values()))
    checks, pieces = Timed(), Timed()
    own = {name: getattr(V, name) for name in (*CHECKS, "lean")}
    for name in CHECKS:
        setattr(V, name, checks.wrap(name, own[name]))
    V.lean = lambda fn, value: pieces.wrap(fn, own["lean"])(fn, value)

    def stream_of(coins) -> S.Stream:
        name = "challenger" if isinstance(coins, CC.ChallengerCoins) else "verifier"
        return S.Stream(sp, WINDOW, budget=1, delta=0.5, challenger=name, coins=coins, clock=lambda: 0.0)

    def register(s: S.Stream) -> S.Registration:
        return s.register(rec, mine, window_root=WINDOW_ROOT, counts=COUNTS)

    def outcome(s: S.Stream, d: S.Draw, evidence: dict | None):
        assert s.deliver(d, [SESSION]) is None
        return s.outcome(d, VERDICTS, {"statement": "stub"}, served=sorted(d.units), session_roots=sessions,
                         coins=evidence), s.records[-1]["audit"]

    t = time.perf_counter()
    try:
        with AM.memo() if memo else contextlib.nullcontext():
            before = V.check(rec, program(), QUERY, law, S.SCHEME, mine)
            coins = AC.VerifierCoins(lambda k: b"\x5a" * k)
            s = stream_of(coins)
            d = s.select(register(s))
            got, _ = outcome(s, d, None)
            neg = AC.verifier_negatives(stream_of, register, s, coins, d, outcome, WINDOW_ROOT)
            fewer = V.check(rec, program({**COUNTS, P.template_id(TRIPLE): COUNTS[P.template_id(TRIPLE)] - 1}), QUERY, law,
                            S.SCHEME, mine)
            moved = V.check(rec, program(), QUERY, law, S.SCHEME, other)
    finally:
        for name, fn in own.items():
            setattr(V, name, fn)
    return {"n": n, "memo": memo, "registration_bytes": len(canonical_json(rec)), "s": round(time.perf_counter() - t, 3),
            "accepted": before.ok and got.accepted, "negatives_refused": all(x["refused"] for x in neg.values())
            and not fewer.ok and not moved.ok, "checks": checks.rows(), "lean": pieces.rows()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n", type=int, nargs="+")
    ap.add_argument("--no-memo", action="store_true")
    a = ap.parse_args()
    for n in a.n:
        print(json.dumps(row(n, not a.no_memo), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
