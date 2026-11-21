"""The top-level protocol `run`, first version: one monitoring period, its order of stages written once (`run_period`).
It holds the public parameters (`PublicParameters`), the state between windows (`PeriodState`), and the wipe's and
compilation's temporary protocols; a caller supplies each other stage's body (`Stages`). It decides no verdict itself.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

#: what every result of this first version says beside it: compilation accepts every circuit unchecked
COMPILATION_NOT_VERIFIED = "compilation not verified"
#: what every result of this first version says of the memory wipe, which this order calls nothing for
WIPE_BY_LAUNCHER = "the wipe is started by the adapter's launcher today"


@dataclass(frozen=True)
class PublicParameters:
    """`run`'s public parameters, read from the application file once, the window as its class ``S,P,O``. ``None``
    where nothing the file names states one (the hardware, the policy's limits): no default stands in for it."""

    hardware: Any
    limits: Any
    window: str | None
    sampling_design: str


@dataclass
class PeriodState:
    """What `run` carries from one stage to the next and between windows: the wipe, the registration, each window's
    accounting and policy outcomes, and the period's result."""

    wipe: Any = None
    registration: Any = None
    windows: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None


@dataclass(frozen=True)
class Stages:
    """Each stage's body but the wipe's and compilation's, called as ``stage(public, workload, state[, window])``:
    ``windows`` gives the period's windows. The policy stage is given what the accounting stage returned, the only
    thing the policy layer is to read; in this first version that is each party's runner outcome, and the policy stage
    runs the policy's steps on the window's record, which accounting's steps left."""

    registration: Callable[[PublicParameters, Any, PeriodState], Any]
    windows: Callable[[PublicParameters, Any, PeriodState], Iterable[Any]]
    accounting: Callable[[PublicParameters, Any, PeriodState, Any], Any]
    policy: Callable[[PublicParameters, Any, PeriodState, Any, Any], Any]
    result: Callable[[PublicParameters, Any, PeriodState], Any]


def wipe_calling_nothing(public: PublicParameters, workload: Any) -> dict[str, Any]:
    """The accounting layer's opening call, the memory wipe on idle hardware before the engine loads, once a server-day,
    which may refuse the day. To implement: call it here; today the adapter's launcher starts it."""
    return {"ran": False, "note": WIPE_BY_LAUNCHER}


def compilation_accepting_everything(public: PublicParameters, workload: Any, window: Any) -> dict[str, Any]:
    """The temporary compilation protocol: it accepts every circuit, whoever compiled it and from whatever advice.
    To implement: verify that the registered compiler compiled the window's circuit from its advice."""
    return {"accepted": True, "note": COMPILATION_NOT_VERIFIED}


def accounting(public: PublicParameters, workload: Any, state: PeriodState, window: Any, stages: Stages) -> dict:
    compiled = compilation_accepting_everything(public, workload, window)
    return {"compilation": compiled, "outcomes": stages.accounting(public, workload, state, window)}


def run_period(public: PublicParameters, workload: Any, stages: Stages) -> PeriodState:
    """One monitoring period: the wipe, one registration, then for each window the accounting layer (compilation
    first) and the policy layer on what accounting returned, then the period's result."""
    state = PeriodState()
    state.wipe = wipe_calling_nothing(public, workload)
    state.registration = stages.registration(public, workload, state)
    for window in stages.windows(public, workload, state):
        accounted = accounting(public, workload, state, window, stages)
        state.windows.append({"window": window, "accounting": accounted,
                              "policy": stages.policy(public, workload, state, window, accounted["outcomes"])})
    state.result = stages.result(public, workload, state)
    return state
