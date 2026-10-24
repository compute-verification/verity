"""`pooled.py`'s pooled bound held to its Lean (`RemoteAudit.Pooled`, served by `remote-audit-difftest`'s `pooled`),
refusals as outputs so that both sides' reasons are compared word for word: drawn records (one to many windows, ties
in receipts a draw, windows opened whole, small and large S, δ as a float's exact value or a rational), gate runs
5–7's records, and each refusal, with malformed inputs rejected by both. The line of record's W is the Lean's (the
verifier process's `line` computes it, held exactly to `line.py`'s, which reads this), so the Lean is of record here;
`audit.pooled` (the verifier's diagnostic line and the sampling plan's) reads the Python."""

from __future__ import annotations

import shutil
from fractions import Fraction

import pytest
from hypothesis import strategies as st

import ra_cases as M
import verity_difftest as D
from verity.core.protocols.remote_bandwidth import pooled

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

GATE_RUNS = {"5/0": (2118, 130023424, 48), "5/2": (96, 130023424, 232), "6/0": (2118, 113770496, 48),
             "6/2": (96, 113770496, 232), "7/0": (2118, 130023424, 48), "7/2": (96, 130023424, 232)}
"""Each accepted session of gate runs 5–7 as `gate_run/k`: its record's windows, all alike (n of them, U receipts
drawn s times each), at δ = 2^-40 (`tests/test_pooled.py`)."""


def port(inputs=st.nothing(), reference=pooled.run_json) -> D.Port:
    return D.Port(M.EXE, "pooled", reference, inputs, package=M.PACKAGE, of_record="lean", rejects=M.REFUSALS)


server = D.server_fixture(port())


def text(q: Fraction) -> str:
    return str(q)


DELTAS = st.one_of(st.sampled_from([2.0 ** -40, 1e-9, 1e-6, 0.5, 0.999]).map(lambda x: text(Fraction(x))),
                   st.fractions(Fraction(1, 10 ** 12), Fraction(999, 1000), max_denominator=10 ** 12).map(text),
                   st.floats(1e-300, 0.99).map(lambda x: text(Fraction(x))))


@st.composite
def inputs(draw) -> dict:
    """A record's windows and δ, as they are or with one fault (most often neither)."""
    shape = draw(st.sampled_from(["small", "alike", "large"]))
    n = draw(st.integers(1, {"small": 6, "alike": 40, "large": 4}[shape]))
    if shape == "alike":
        u = draw(st.integers(1, 10 ** 9))
        ws = [[u, draw(st.integers(1, min(u, 300)))]] * n
    else:
        ws = []
        for _ in range(n):
            u = draw(st.integers(1, 60 if shape == "small" else 10 ** 12))
            ws.append([u, draw(st.one_of(st.integers(1, min(u, 60 if shape == "small" else 10 ** 4)),
                                         st.just(max(1, u // 2)), st.just(u)))])
    v = {"windows": ws, "delta": draw(DELTAS)}
    fault = {4: draw(st.sampled_from(["empty", "none drawn", "over", "delta"])), 5: "malformed", 6: "malformed"}.get(
        draw(st.integers(0, 6)))
    t = draw(st.integers(0, n - 1))
    if fault == "empty":
        v["windows"] = []
    elif fault == "none drawn":
        ws[t] = [ws[t][0], 0]
    elif fault == "over":
        ws[t] = [ws[t][0], ws[t][0] + draw(st.integers(1, 3))]
    elif fault == "delta":
        v["delta"] = draw(st.sampled_from([0, 1, "-1/3", "3/2", "1"]))
    elif fault == "malformed":
        bad = draw(st.sampled_from([-1, "1/0", "1.5", "+3", None, True, [], {}]))
        k = draw(st.sampled_from(["windows", "delta", "window", "count", "drop"]))
        if k in ("windows", "delta"):
            v[k] = bad
        elif k == "window":
            ws[t] = draw(st.sampled_from([[1], [1, 2, 3], {}, "1,1"]))
        elif k == "count":
            ws[t] = draw(st.sampled_from([[bad, 1], [2, bad]]))
        else:
            v.pop(draw(st.sampled_from(["windows", "delta"])))
    return v


def test_pooled(server) -> None:
    told = {"bound": 0, "refused": 0, "capped": 0}

    def reference(v):
        out = pooled.run_json(v)
        told["refused" if "refused" in out else "bound"] += 1
        told["capped"] += "refused" not in out and any(
            Fraction(f) == 1 - Fraction(s, u) and s < u for f, (u, s) in zip(out["floors"], v["windows"]))
        return out

    stats = D.run(port(inputs(), reference), examples=500, server=server)
    assert stats.rejects > 20 and told["bound"] > 250 and told["refused"] > 30 and told["capped"] > 10, (stats, told)


@pytest.mark.parametrize("k", sorted(GATE_RUNS))
def test_gate_runs_records_are_the_leans(server, k) -> None:
    n, u, s = GATE_RUNS[k]
    v = {"windows": [[u, s]] * n, "delta": text(Fraction(2.0 ** -40))}
    kind, body = D.replay(port(), v, server)
    assert kind == "output" and D.canonical(pooled.run_json(v)) == body


def test_ties_and_whole_windows_are_the_leans(server) -> None:
    for ws in ([[4, 2], [2, 1], [6, 3], [9, 9]], [[40, 40]] * 3, [[10 ** 12, 1], [5, 1], [10 ** 12, 1]],
               [[100, 10], [1000, 10], [100, 10], [12, 10]]):
        for d in ("1/1099511627776", "1/3", text(Fraction(1e-9))):
            assert D.replay(port(), {"windows": ws, "delta": d}, server)[0] == "output"


@pytest.mark.parametrize("v,why", [
    ({"windows": [], "delta": "1/3"}, "a session has one or more windows"),
    ({"windows": [[5, 2]], "delta": 0}, "δ is above 0 and below 1"),
    ({"windows": [[5, 2]], "delta": "1"}, "δ is above 0 and below 1"),
    ({"windows": [[5, 2]], "delta": "-1/3"}, "δ is above 0 and below 1"),
    ({"windows": [[5, 2], [3, 0], [1, 2]], "delta": "1/3"}, "window 1 draws no receipt"),
    ({"windows": [[5, 2], [2, 3], [3, 0]], "delta": "1/3"}, "window 1 draws 3 receipts of its 2"),
])
def test_each_refusal_is_the_leans_word_for_word(server, v, why) -> None:
    assert pooled.run_json(v) == {"refused": why}
    assert D.replay(port(), v, server) == ("output", D.canonical({"refused": why}))
