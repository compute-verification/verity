"""`CoreProtocol.Period` against `verity.core.periods`, through the core difftest harness (`verity/core/difftest`).

`core-periods-difftest` (`Core.PeriodDifftest.table`) serves `span` (a span read from its JSON, and its stop), `within`,
`cut` (the parts of a span's cut at its boundaries) and `monitoring` (a monitoring period read from its JSON, its execution
blocks' spans and each block's sampling blocks' spans). Inputs are drawn valid and broken: spans with a key missing or
extra, negative or boolean or string fields, boundaries out of order or past the span, sampling cuts too few or not of
their block's span. Python must refuse exactly what Lean refuses.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.periods import Cut, MonitoringPeriod, PeriodError, TickSpan

LEAN = Path(__file__).resolve().parents[2] / "protocols" / "interface"
EXE = "core-periods-difftest"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

REJECTS = (D.Reject, PeriodError)


def _get(i: Any, k: str) -> Any:
    if not isinstance(i, dict) or k not in i:
        raise D.Reject(f"{k} missing")
    return i[k]


def _spans(ps) -> list[dict[str, int]]:
    return [p.to_json() for p in ps]


def span(i: Any) -> dict:
    p = TickSpan.from_json(i)
    return {"span": p.to_json(), "stop": p.stop}


def within(i: Any) -> dict:
    inner, outer = TickSpan.from_json(_get(i, "inner")), TickSpan.from_json(_get(i, "outer"))
    return {"within": inner.within(outer)}


def cut(i: Any) -> dict:
    p, bs = TickSpan.from_json(_get(i, "span")), _get(i, "boundaries")
    if not isinstance(bs, list) or not all(type(b) is int and b >= 0 for b in bs):
        raise D.Reject("boundaries: not a list of naturals")
    return {"parts": _spans(Cut(p, tuple(bs)).parts)}


def monitoring(i: Any) -> dict:
    m = MonitoringPeriod.from_json(i)
    return {"execution": _spans(m.execution.parts), "sampling": [_spans(c.parts) for c in m.sampling]}


NAT = st.integers(0, 2 ** 70)
NOT_NAT = st.one_of(st.integers(max_value=-1), st.booleans(), st.text(max_size=3), st.none(), st.lists(NAT, max_size=1))


@st.composite
def spans(draw) -> TickSpan:
    return TickSpan(draw(st.integers(0, 2 ** 64)), draw(st.integers(0, 64)))


@st.composite
def boundaries(draw, p: TickSpan) -> list[int]:
    inner = draw(st.lists(st.integers(p.start, p.stop), max_size=4))
    return [p.start, *sorted(inner), p.stop] if p.len or draw(st.booleans()) else [p.start]


def broken_span(j: dict) -> st.SearchStrategy:
    return st.one_of(st.just({k: v for k, v in j.items() if k != "len"}), st.just(j | {"clock": "auditor"}),
                     NOT_NAT.map(lambda v: j | {"start": v}), NOT_NAT.map(lambda v: j | {"len": v}),
                     st.just([j["start"], j["len"]]))


SPAN_JSON = spans().map(TickSpan.to_json)
ANY_SPAN = st.one_of(SPAN_JSON, SPAN_JSON.flatmap(broken_span))


@st.composite
def broken(draw, bs: list[int]) -> list:
    """`bs` with one boundary moved, dropped or duplicated past the end, or made no natural."""
    bs = list(bs)
    k = draw(st.integers(0, len(bs) - 1))
    how = draw(st.sampled_from(["move", "drop", "extend", "nat"]))
    if how == "move":
        bs[k] = draw(st.integers(0, 2 ** 64 + 128))
    elif how == "drop":
        del bs[k]
    elif how == "extend":
        bs.append(bs[-1] + draw(st.integers(0, 3)))
    else:
        bs[k] = draw(NOT_NAT)
    return bs


@st.composite
def cut_inputs(draw) -> dict:
    p = draw(spans())
    bs = draw(boundaries(p))
    return {"span": p.to_json() if draw(st.integers(0, 9)) else draw(broken_span(p.to_json())),
            "boundaries": draw(broken(bs)) if draw(st.booleans()) else bs}


@st.composite
def monitoring_inputs(draw) -> Any:
    p = draw(spans())
    execution = draw(boundaries(p))
    sampling = [draw(boundaries(b)) for b in Cut(p, tuple(execution)).parts]
    j = {"execution": execution, "sampling": sampling, "span": p.to_json()}
    how = draw(st.sampled_from(["valid", "valid", "execution", "sampling", "count", "key", "span"]))
    if how == "execution":
        j["execution"] = draw(broken(execution))
    elif how == "sampling" and sampling:
        k = draw(st.integers(0, len(sampling) - 1))
        j["sampling"] = sampling[:k] + [draw(broken(sampling[k]))] + sampling[k + 1:]
    elif how == "count":
        j["sampling"] = sampling[:-1] if sampling else [[p.start]]
    elif how == "key":
        j = draw(st.sampled_from([{k: v for k, v in j.items() if k != "sampling"}, j | {"blocks": len(sampling)}]))
    elif how == "span":
        j["span"] = draw(st.one_of(broken_span(p.to_json()), spans().map(TickSpan.to_json)))
    return j


PORTS = {
    "span": D.Port(EXE, "span", reference=span, inputs=ANY_SPAN, package=LEAN, rejects=REJECTS),
    "within": D.Port(EXE, "within", reference=within, package=LEAN, rejects=REJECTS,
                     inputs=st.fixed_dictionaries({"inner": ANY_SPAN, "outer": ANY_SPAN})),
    "cut": D.Port(EXE, "cut", reference=cut, inputs=cut_inputs(), package=LEAN, rejects=REJECTS),
    "monitoring": D.Port(EXE, "monitoring", reference=monitoring, inputs=monitoring_inputs(), package=LEAN,
                         rejects=REJECTS),
}
server = D.server_fixture(PORTS["span"])


@pytest.mark.parametrize("fn", sorted(PORTS))
def test_periods_difftest(server, fn):
    D.run(PORTS[fn], examples=300, server=server)
