"""`PousCheck.Verdict`, the verdict proof block's Program `PousVerdict_v1{K, M, LIMIT}` as code, against the Definition
`verdict_circuit.PousVerdict`, compiled, through the core difftest harness (`pous-verdict-difftest`).

`Proofs.Pous.Continuous.VerdictCode` proves the code's `prog` is `PousVerdict.verdictProg` gate for gate, so
`PousVerdict.VerdictIs` holds of it; these tests hold the Definition's circuit to `prog` at the statics they run.

* `gates`: the gate count, the output wires and the SHA-512 of the gates, one line each (`input`, `const b`, `not i`,
  `and w…`, `xor w…`), against the Definition's canonical layout (`standalone_call`): at the sidecar's statics
  (K = 132, M = 0 and K = 4, M = 1, LIMIT = 500,000), at drawn statics, and at statics both refuse.
* `eval`: the code's run of `prog` on a session's openings, its round verdicts and its acceptance, against the compiled
  circuit's output (`evaluate_bits`), `verdict_circuit.round_verdict` and `verdict_value`: at the sidecar's statics on
  an honest session, late receipts (one past LIMIT and far past), wrong digests (the first and the last bit), records of
  another round, receipts before their sends, and exactly M and M + 1 misses; on drawn statics and sessions; and on
  malformed sessions, which both refuse.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import verdict_circuit as V
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as EB
from verity.ml.circuits.evaluation.reference import standalone_call
from verity_difftest import protocol as P

LEAN = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

SIDECAR = [(132, 0, 500_000, 318_154), (4, 1, 500_000, 9_554)]
"""`verdict_circuit.circuit-check.py`'s statics, the audit's 132 challenges with no miss and 4 with one, and the proof block's
gate count at each (inputs included)."""
RECORD, DIGEST, TOP = V.RECORD_BITS // 8, V.DIGEST_BITS // 8, (1 << 64) - 1


# The reference: the Definition, compiled.

def _statics(i) -> tuple[int, int, int]:
    if not isinstance(i, dict):
        raise D.Reject("the instance is not an object")
    K, M, LIMIT = (i.get(k) for k in ("K", "M", "LIMIT"))
    try:
        V._statics(K, M, LIMIT)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return K, M, LIMIT


@functools.cache
def _unit(K: int, M: int, LIMIT: int):
    return bind(V.PousVerdict, K=K, M=M, LIMIT=LIMIT)


def _line(g) -> str:
    if g.prim.id == "Input1_v1":
        return "input"
    if BL.is_constant_bit(g.prim):
        return f"const {BL.CONSTANTS.index(g.prim)}"
    return " ".join([BL.gate_kind(g.prim)[0], *map(str, g.operands)])


@functools.cache
def _layout(K: int, M: int, LIMIT: int) -> dict:
    C = standalone_call(_unit(K, M, LIMIT))[0].circuit
    text = "".join(_line(C[w]) + "\n" for w in range(C.size))
    return {"count": C.size, "outs": list(C.root_outputs()), "sha512": hashlib.sha512(text.encode()).hexdigest()}


def gates(i) -> dict:
    return _layout(*_statics(i))


def _byte_strings(i: dict, k: str, K: int, n: int) -> list[bytes]:
    xs = i.get(k)
    if not isinstance(xs, list) or not all(isinstance(x, str) for x in xs):
        raise D.Reject(f"{k} is missing or not a list of strings")
    bs = [P.unhex(x) for x in xs]
    if len(bs) != K or any(len(b) != n for b in bs):
        raise D.Reject(f"{k} is not {K} strings of {n} bytes")
    return bs


def _bits(data: bytes) -> list[int]:
    return [(b >> k) & 1 for b in data for k in range(8)]


def evaluate(i, unit=_unit) -> dict:
    K, M, LIMIT = _statics(i)
    rs, es = _byte_strings(i, "records", K, RECORD), _byte_strings(i, "expected", K, DIGEST)
    recs = np.array([[_bits(r) for r in rs]], dtype=np.uint8)
    exps = np.array([[_bits(e) for e in es]], dtype=np.uint8)
    out = EB.evaluate_bits(unit(K, M, LIMIT), recs, exps).reshape(-1)
    return {"out": [int(v) for v in out],
            "verdicts": [V.round_verdict(LIMIT, j, r, e) for j, (r, e) in enumerate(zip(rs, es))],
            "accepts": V.verdict_value(K, M, LIMIT, rs, es)}


# Sessions.

KINDS = {"hit": "hit", "zero": "hit", "edge": "hit", "wrong": "wrong", "wrong-last": "wrong", "late": "late",
         "late-one": "late", "late-wrong": "late", "early": "reject", "round": "reject", "round-next": "reject"}
"""How a round turns out, and its verdict at a LIMIT well below 2^64: a hit (some time elapsed, none, exactly LIMIT); its
digest's first or last bit flipped; received past LIMIT (far past, one past, and past with a wrong digest); received
before it was sent; a record of another round (one bit of j flipped, or j + 1)."""
MISSES = ("late", "wrong", "round", "early")


def _round(rng: random.Random, j: int, LIMIT: int, kind: str) -> tuple[bytes, bytes]:
    e = rng.randbytes(DIGEST)
    elapsed = {"zero": 0, "edge": LIMIT, "late-one": LIMIT + 1, "span": TOP}.get(kind)
    if elapsed is None:
        elapsed = LIMIT + 1 + rng.randrange(1 << 20) if kind.startswith("late") else rng.randrange(LIMIT + 1)
    elapsed = min(elapsed, TOP)
    if kind == "early":
        send = rng.randrange(1, TOP + 1)
        receipt = send - 1 - rng.randrange(min(send, 1 << 20))
    else:
        send = rng.randrange(TOP - elapsed + 1)
        receipt = send + elapsed
    h = bytearray(e)
    flip = {"wrong": (0, 0), "wrong-last": (DIGEST - 1, 7), "late-wrong": (rng.randrange(DIGEST), rng.randrange(8))}
    if kind in flip:
        h[flip[kind][0]] ^= 1 << flip[kind][1]
    rj = {"round": j ^ 1 << rng.randrange(32), "round-next": (j + 1) % (1 << 32)}.get(kind, j)
    return mc.Round(rj, rng.randrange(1 << 64), send, receipt, bytes(h)).value(), e


def _instance(K: int, M: int, LIMIT: int, kinds, seed: int) -> dict:
    rng = random.Random(seed)
    rounds = [_round(rng, j, LIMIT, k) for j, k in enumerate(kinds)]
    return {"K": K, "M": M, "LIMIT": LIMIT, "records": [r.hex() for r, _ in rounds],
            "expected": [e.hex() for _, e in rounds]}


LIMITS = st.one_of(st.sampled_from([1, 2, 7, 1000, 500_000, 1 << 32, TOP - 1, TOP]), st.integers(1, TOP))


@st.composite
def statics(draw) -> dict:
    K = draw(st.integers(1, 6))
    return {"K": K, "M": draw(st.integers(0, K + 2)), "LIMIT": draw(LIMITS)}


BAD_STATICS = st.sampled_from([
    {"K": 0, "M": 0, "LIMIT": 1}, {"K": 1, "M": -1, "LIMIT": 1}, {"K": 1, "M": 0, "LIMIT": 0},
    {"K": 1, "M": 0, "LIMIT": 1 << 64}, {"K": True, "M": 0, "LIMIT": 1}, {"K": 1, "M": 0, "LIMIT": "1"},
    {"K": 1, "M": 0}, {}, []])


@st.composite
def sessions(draw) -> dict:
    s = draw(statics())
    kinds = draw(st.lists(st.sampled_from(sorted(KINDS) + ["span"]), min_size=s["K"], max_size=s["K"]))
    return _instance(s["K"], s["M"], s["LIMIT"], kinds, draw(st.integers(0, 1 << 32)))


def _malformed(i: dict, how: str) -> dict:
    i = dict(i, records=list(i["records"]), expected=list(i["expected"]))
    if how == "fewer":
        i["records"].pop()
    elif how == "more":
        i["expected"].append(i["expected"][0])
    elif how == "short":
        i["records"][0] = i["records"][0][:-2]
    elif how == "long":
        i["expected"][-1] += "00"
    elif how == "upper":
        i["records"][0] = i["records"][0].upper()
    elif how == "odd":
        i["expected"][0] = i["expected"][0][:-1]
    elif how == "number":
        i["records"][0] = 0
    elif how == "missing":
        del i["expected"]
    elif how == "statics":
        i["LIMIT"] = 0
    return i


MALFORMATIONS = ["fewer", "more", "short", "long", "upper", "odd", "number", "missing", "statics"]
ACCEPTS: collections.Counter = collections.Counter()


def _counted(i) -> dict:
    out = evaluate(i)
    ACCEPTS[out["accepts"]] += 1
    return out


GATES = D.Port("pous-verdict-difftest", "gates", gates, st.one_of(statics(), statics(), BAD_STATICS), package=LEAN)
EVAL = D.Port("pous-verdict-difftest", "eval", _counted,
              st.one_of(sessions(), sessions(), sessions(), st.builds(_malformed, sessions(), st.sampled_from(MALFORMATIONS))),
              package=LEAN)
server = D.server_fixture(GATES)


@pytest.mark.slow
def test_the_gates_agree(server):
    stats = D.run(GATES, examples=150, server=server)
    assert stats.outputs > 50 and stats.rejects > 5, stats


@pytest.mark.parametrize("K,M,LIMIT,count", SIDECAR)
def test_the_gates_agree_at_the_sidecars_statics(server, K, M, LIMIT, count):
    kind, line = D.replay(GATES, {"K": K, "M": M, "LIMIT": LIMIT}, server)
    assert kind == "output" and json.loads(line)["count"] == count


@pytest.mark.slow
def test_the_runs_agree(server):
    ACCEPTS.clear()
    stats = D.run(EVAL, examples=300, server=server)
    assert stats.outputs > 100 and stats.rejects > 20, stats
    assert ACCEPTS[0] > 10 and ACCEPTS[1] > 10, ACCEPTS


def _cases(K: int, M: int) -> dict[str, list[str]]:
    """Named sessions of K rounds: honest ones, one miss of each kind, and exactly M and M + 1 misses of mixed kinds."""
    def misses(n: int) -> list[str]:
        return [MISSES[j % len(MISSES)] if j < n else "hit" for j in range(K)]

    def one(kind: str, at: int) -> list[str]:
        return ["hit"] * at + [kind] + ["hit"] * (K - at - 1)

    cases = {"honest": ["hit"] * K, "honest at the edges": [("zero", "edge", "hit")[j % 3] for j in range(K)],
             "every kind": [sorted(KINDS)[j % len(KINDS)] for j in range(K)], "M misses": misses(M),
             "M + 1 misses": misses(M + 1)}
    cases |= {kind: one(kind, (K - 1) * (n % 2)) for n, kind in enumerate(sorted(KINDS)) if KINDS[kind] != "hit"}
    return cases


@pytest.mark.parametrize("K,M,LIMIT,count", [pytest.param(*SIDECAR[0], marks=pytest.mark.slow), SIDECAR[1]])
def test_the_runs_agree_on_named_sessions(server, K, M, LIMIT, count):
    """Each named session's outcome is what it is named for, and the code's and the circuit's agree on it."""
    for n, (name, kinds) in enumerate(_cases(K, M).items()):
        kind, line = D.replay(EVAL, _instance(K, M, LIMIT, kinds, n), server)
        assert kind == "output", name
        out = json.loads(line)
        want = int(sum(KINDS[k] != "hit" for k in kinds) <= M)
        assert (out["verdicts"], out["accepts"], out["out"]) == ([KINDS[k] for k in kinds], want, [want]), name


def test_a_unit_without_the_deadline_is_caught(server):
    """A Definition that took LIMIT as 2^64 − 1 accepts a late round that the code refuses."""
    K, M, LIMIT, _ = SIDECAR[1]
    i = _instance(K, M, LIMIT, ["late", "late", "hit", "hit"], 0)
    port = D.Port("pous-verdict-difftest", "eval", lambda i: evaluate(i, lambda K, M, _: _unit(K, M, TOP)),
                  EVAL.inputs, package=LEAN)
    with pytest.raises(D.Mismatch):
        D.replay(port, i, server)
