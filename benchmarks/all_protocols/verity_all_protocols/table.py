"""The protocols every period is audited by, and the one interface their legs share.

A leg is ``benchmarks/all_protocols/legs/<component>.py`` (``-`` as ``_``) defining

    leg(period_dir: Path, node: str, out: Path) -> dict

``period_dir`` is the period's loop records: the window `verity_torch_loop.serve` wrote (``registration.json``,
``circuit.json``, ``window.json``, ``index.json``, the value files, ``serve.json``), with ``period.json`` (`PERIOD`: the
period's index, the circuit the verifier registered before the loop ran, the run root it published, the law and member
cap, the built tools, the prover's environment) and ``answers/<component>.json``, the answers of the legs that ran before
it (in `TABLE`'s order: integrity first, which PoUW's evidence reads). ``node`` is the machine the loop ran on. ``out`` is
the leg's own directory, kept in the run's ``$OUT``.

It returns its protocol's answer in the core interface's canonical form (`Answer.canonical`,
verity/core/protocols/interface/README.md): ``{"commitment", "profile", "protocol", "public", "reason", "verdict"}``.
What its acceptance certifies is its ``public`` (W, S, Q, K, the bound on wrong units), and its δ its profile's. It may
write ``out/<component>.case.json``, ``{"commitment": hex, "evidence": hex, "params": JSON}``, the case its port's
``serve`` reads, which onsite's evidence carries for it once the port replaces onsite's stub, and ``out/runs.json``, the
ids of runs it launched. A leg that can't decide raises: the runner reports ``error``, never a verdict. A protocol whose
leg file doesn't exist yet reports ``{"verdict": "missing", "owner": LANE}``.

A lane plugs its protocol in by adding its file, and touches nothing else.
"""
from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

LEGS_DIR = Path(__file__).resolve().parents[1] / "legs"
PERIOD = "period.json"
ANSWERS = "answers"

LegFn = Callable[[Path, str, Path], dict]


@dataclass(frozen=True)
class Entry:
    """One protocol: its name in the report, the component its leg is named by (onsite's, when ``onsite``) and the lane
    that owns its leg."""

    name: str
    component: str
    owner: str
    onsite: bool = True


TABLE: tuple[Entry, ...] = (
    Entry("integrity", "sampled-proofs", "compute-accounting"),
    Entry("work", "pouw", "compute-accounting"),
    Entry("space", "pous", "memory-accounting"),
    Entry("bandwidth", "bandwidth", "memory-accounting", onsite=False),
    Entry("communication", "certifier", "network-accounting"),
)


def load(component: str, legs_dir: Path = LEGS_DIR) -> ModuleType | None:
    """The module ``legs/<component>.py``, or None while that file doesn't exist."""
    path = legs_dir / (component.replace("-", "_") + ".py")
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"all_protocols_leg_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def leg_of(component: str, legs_dir: Path = LEGS_DIR) -> LegFn | None:
    """``legs/<component>.py``'s ``leg``, or None while that file doesn't exist."""
    mod = load(component, legs_dir)
    return None if mod is None else mod.leg


def period_of(period_dir: Path) -> dict[str, Any]:
    """The period's ``period.json``."""
    return json.loads((period_dir / PERIOD).read_text())


def check_answer(a: Any) -> dict[str, Any]:
    """``a`` when it is an answer in the shapes `Answer.ofJson` reads (`pouw.core_check.answer_of`, the Python mirror of
    the core interface's decoder): an accept has no reason and both its public outputs and its profile, a refusal a
    reason and neither. Anything else is a ValueError, which the runner reports as the leg's error."""
    from verity.core.protocols.pouw import core_check as CC

    try:
        CC.answer_of(a)
    except CC.Refused as e:
        raise ValueError(f"not a core answer: {e}") from None
    return dict(a)


__all__ = ["ANSWERS", "Entry", "LEGS_DIR", "LegFn", "PERIOD", "TABLE", "check_answer", "leg_of", "load", "period_of"]
