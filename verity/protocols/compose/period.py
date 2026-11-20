"""The top-level protocol `run`, first version: one monitoring period, its order of stages written once (`run_period`).
It holds the public parameters (`PublicParameters`), the state between windows (`PeriodState`) and compilation's
temporary protocol (`compilation`); a caller supplies each other stage's body (`Stages`), and Lean decides every verdict.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

#: what every result of this first version says beside it: `compilation` accepts every circuit unchecked
COMPILATION_NOT_VERIFIED = "compilation not verified"


@dataclass(frozen=True)
class PublicParameters:
    """`run`'s public parameters, read from the application file once. ``None`` where no key of the file states one
    (the hardware, the policy's limits): no default stands in for it."""

    hardware: Any
    limits: Any
    window: int
    sampling_design: str


@dataclass
class PeriodState:
    """What `run` carries from one stage to the next and between windows: the registration, each window's accounting
    and policy outcomes, and the period's result."""

    registration: Any = None
    windows: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None


@dataclass(frozen=True)
class Stages:
    """Each stage's body but compilation's, called as ``stage(public, workload, state[, window])``: ``windows`` gives
    the period's windows. ``accounting`` is for the switched-on accounting protocols' verified properties of a window's
    blocks, the only thing the policy layer is to read; in this first version it returns each party's outcome, and the
    policy stage reads a file that accounting's steps wrote."""

    registration: Callable[[PublicParameters, Any, PeriodState], Any]
    windows: Callable[[PublicParameters, Any, PeriodState], Iterable[Any]]
    accounting: Callable[[PublicParameters, Any, PeriodState, Any], Any]
    policy: Callable[[PublicParameters, Any, PeriodState, Any, Any], Any]
    result: Callable[[PublicParameters, Any, PeriodState], Any]


def compilation(public: PublicParameters, workload: Any, window: Any) -> dict[str, Any]:
    """The temporary compilation protocol: it accepts every circuit, whoever compiled it and from whatever advice.
    To implement: verify that the registered compiler compiled the window's circuit from its advice."""
    return {"accepted": True, "note": COMPILATION_NOT_VERIFIED}


def accounting(public: PublicParameters, workload: Any, state: PeriodState, window: Any, stages: Stages) -> dict:
    compiled = compilation(public, workload, window)
    return {"compilation": compiled, "outcomes": stages.accounting(public, workload, state, window)}


def run_period(public: PublicParameters, workload: Any, stages: Stages) -> PeriodState:
    """One monitoring period: one registration, then for each window the accounting layer (compilation first) and the
    policy layer on what accounting returned, then the period's result."""
    state = PeriodState()
    state.registration = stages.registration(public, workload, state)
    for window in stages.windows(public, workload, state):
        accounted = accounting(public, workload, state, window, stages)
        state.windows.append({"window": window, "accounting": accounted,
                              "policy": stages.policy(public, workload, state, window, accounted["outcomes"])})
    state.result = stages.result(public, workload, state)
    return state
