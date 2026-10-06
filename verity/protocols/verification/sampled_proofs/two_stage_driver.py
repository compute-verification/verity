"""The two-stage driver: sampled proofs' two-stage law (`two_stage`) run on the proof service's `Stream`s.

A protocol states one `service.Spec` whose selection rule is `TwoStage(replay=Law(...), check="all")`. Its values are its
replay units' boundaries, committed by serve time (phase `epoch` or `serve`, and registered values), and their interiors
(phase `interior`). Each replay unit (RU) is one instance of the check's unit. A `TwoStageStream` runs four steps per audit:

1. **Boundaries** (`register`): one-stage's registration record over the N RUs, with the boundary values' roots, and its
   receipt.
2. **The draw** (`select`), after that receipt: RUs by the first stage's law, from the verifier's own randomness, as the
   Lean verifier's `Flock.Draw.drawWith` draws them (`two_stage.draw_first`), a work law with the verifier's closure.
3. **Interiors** (`register_interiors`): the proved RUs' interiors, committed after the draw and registered before any
   proof, as one-stage's record over those RUs in ascending order. Unit i of the interior registration is RU
   `ReplayDraw.proved[i]`, and all of its units are proved.
4. **Proving** (`deliver`, `outcome`): one proof over the proved RUs, delivered once, and the audit's final outcome.

Stage 2 checks every proof unit of a drawn RU, so there is no second draw, and Lean's `late_interior_insecure` (a
proof-unit draw before the interiors are committed) can't arise. The profile is the first stage's law over the RUs (Lean
`effEscape_full`): `TwoStageLaw.profile` of `Stage(law)`, which is one-stage's profile of the boundary registration.

**On the service's Streams**, until `Stream` runs `TwoStage` itself:
* The boundary registration goes through one `Stream` under the first stage's law, whose `sampler` is `draw_first`.
* The interiors go through a second `Stream` of their own per audit, under `All`, because a Stream refuses a second
  registration before the first has an outcome. That Stream has no proof deadline. The deadline is the boundary
  Stream's, from the RU draw, and covers the replay, the interiors and the proof.
* The interior outcome is a verdict of the boundary outcome (`INTERIOR`, not of record), so the boundary Stream accepts
  an audit exactly when both accept, and its failures, budget and symbols are the driver's.
* The driver's record (`AUDIT`) binds both registrations, both receipts, the draw with its closure, and both Streams'
  outcome records.

No retries: the RU draw is made once, the interiors are registered once, a proof is delivered once, and an outcome is
final. A refused call changes nothing.
"""
from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from verity.primitives.circuits.codec import canonical_json
from verity.primitives.circuits.defs import SpecializedDefinition

from .one_stage import draw as D
from .one_stage.verdict import Verdict
from .service import LATE, All, Draw, Law, Outcome, Refused, Registration, Spec, Stream, TwoStage, Value
from .two_stage import OwnRandomness, draw_first

STREAM = "verity/proof-service/two-stage/stream/v0"
AUDIT = "verity/proof-service/two-stage/audit/v0"
INTERIOR = "two-stage/interior"
"""The boundary outcome's verdict that the interior outcome accepted; its `outcome` is that outcome's record."""


def _sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


def _refuse(code: str, detail: Any) -> Refused:
    return Refused(Verdict(False, (code,), {code: detail}))


def _half(spec: Spec, values: tuple[Value, ...], selection: All | Law, check=None) -> Spec:
    names = {v.name for v in values}
    public = tuple(p for p in spec.public if p.kind != "commitment" or p.item in names)
    return Spec(values, selection, check or spec.check, public, spec.context)


@dataclass(frozen=True)
class ReplayDraw:
    """The draw of RUs: `draw`, the boundary Stream's (one-stage's draw object, which carries no closure), and `replay`,
    the first stage's whole draw (`draw_first`'s, with a work law's `closure`)."""

    draw: Draw
    replay: Mapping[str, Any]

    @property
    def units(self) -> list[int]:
        return list(self.replay["units"])

    @property
    def proved(self) -> list[int]:
        """The RUs whose interiors are registered and proved: the drawn ones and their closure, ascending."""
        return D.proved(self.replay)


@dataclass
class _Audit:
    registration: Registration
    draw: ReplayDraw | None = None
    interiors: Stream | None = None
    interior: Registration | None = None
    interior_draw: Draw | None = None
    interior_outcome: Outcome | None = None
    sessions: list[str] | None = None
    outcome: Outcome | None = None


class TwoStageStream:
    """One protocol's two-stage stream for one `Spec`, held by the challenger.

    `unit` is the challenger's copy of the check's Definition, of which each RU is one instance. `budget`, `delta`,
    `challenger` and `clock` are the boundary `Stream`'s. `randomness` is the verifier's own (`OwnRandomness`, the
    operating system's bytes unless a test injects them).
    """

    def __init__(self, spec: Spec, unit: SpecializedDefinition, *, budget: int, delta: float, challenger: str,
                 randomness: OwnRandomness | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(spec.selection, TwoStage):
            raise ValueError(f"the two-stage driver runs a TwoStage selection, not {spec.selection!r}")
        if spec.selection.check != "all":
            raise ValueError("the two-stage driver checks every proof unit of a drawn RU (check='all'): an RU here is one "
                             "instance of the check's unit, and a count needs RUs of several proof units")
        if any(v.phase == "in-deadline" for v in spec.values):
            raise ValueError("an RU's values are committed by serve time or after the draw, never inside a deadline")
        interior = tuple(v for v in spec.values if v.phase == "interior")
        boundary = tuple(v for v in spec.values if v.phase != "interior")
        if not interior or not boundary:
            raise ValueError("a two-stage spec commits its RUs' boundaries by serve time and their interiors "
                             "(phase 'interior') after the draw")
        if any(v.registered for v in interior):
            raise ValueError("a registered value is committed once per deployment, never as an RU's interior")
        self.spec, self.unit, self.delta, self.challenger = spec, unit, delta, challenger
        self.boundary_spec = _half(spec, boundary, spec.selection.replay)
        self.interior_spec = _half(spec, interior, All(), dataclasses.replace(spec.check, proof_deadline=None))
        self.randomness = randomness or OwnRandomness()
        self._clock = clock
        self._interior_names = {v.name for v in interior}
        self._closure: Mapping[int, Sequence[int]] | None = None
        self._drawn: dict[str, Any] | None = None
        self._boundary = Stream(self.boundary_spec, unit, budget=budget, delta=delta, challenger=challenger, clock=clock,
                                sampler=self._sample)
        self.digest = _sha512({"format": STREAM, "spec": spec.digest(), "budget": budget, "delta": delta,
                               "challenger": challenger})
        self._audits: list[_Audit] = []
        self._records: list[dict[str, Any]] = []

    @property
    def failures(self) -> int:
        return self._boundary.failures

    @property
    def stopped(self) -> bool:
        return self._boundary.stopped

    @property
    def symbols(self) -> tuple[str, ...]:
        """One symbol per final outcome, in order: `accept`, `reject` or `late`."""
        return self._boundary.symbols

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """The audits' records, in order; the outcome's `record` is each one's SHA-512."""
        return tuple(self._records)

    @property
    def boundary_records(self) -> tuple[dict[str, Any], ...]:
        """The boundary Stream's outcome records, one per audit."""
        return self._boundary.records

    def _sample(self, law: Mapping[str, Any], n: int) -> list[int]:
        self._drawn = draw_first(self.randomness, law, n, self._closure)
        return self._drawn["units"]

    def _audit(self, registration: Registration) -> _Audit:
        o = registration.order
        if not (type(o) is int and 0 <= o < len(self._audits)) or self._audits[o].registration != registration:
            raise _refuse("unknown", {"order": o})
        return self._audits[o]

    def _drawn_audit(self, draw: ReplayDraw) -> _Audit:
        a = self._audit(draw.draw.registration)
        if a.draw != draw:
            raise _refuse("not-drawn", {"order": draw.draw.registration.order})
        if a.outcome is not None:
            raise _refuse("final", {"order": draw.draw.registration.order})
        return a

    def register(self, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]], **kw: Any) -> Registration:
        """Step 1: the boundary registration, held by the boundary Stream (`Stream.register`; `kw` are its
        per-registration copies). Raises `Refused` as it does, including while an earlier audit has no outcome."""
        reg = self._boundary.register(record, ports, **kw)
        self._audits.append(_Audit(reg))
        return reg

    def select(self, registration: Registration, closure: Mapping[int, Sequence[int]] | None = None) -> ReplayDraw:
        """Step 2: the RU draw, once, after the boundary receipt, from the verifier's own randomness. `closure` is the
        verifier's closure map (`{RU: [RUs]}`), for a work law only. Raises `Refused` for a second draw."""
        a = self._audit(registration)
        self._closure, self._drawn = closure, None
        try:
            draw = self._boundary.select(registration)
        finally:
            self._closure = None
        assert self._drawn is not None and self._drawn["units"] == draw.units
        a.draw = ReplayDraw(draw, self._drawn)
        return a.draw

    def register_interiors(self, draw: ReplayDraw, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]],
                           **kw: Any) -> Registration:
        """Step 3: the proved RUs' interiors, registered once, after the draw and before any proof. `record` is one-stage's
        record over `len(draw.proved)` units, the RUs in `draw.proved`'s order, under `All`. It is held to the
        interior values by its own Stream (`Stream.register`). Raises `Refused`: `interior-registered` for a second
        one, `interior-population` when its population is not the proved RUs'."""
        a = self._drawn_audit(draw)
        if a.interior is not None:
            raise _refuse("interior-registered", {"order": a.registration.order})
        n = record.get("population") if isinstance(record, Mapping) else None
        if n != len(draw.proved):
            raise _refuse("interior-population", {"registered": n, "proved": len(draw.proved)})
        interiors = Stream(self.interior_spec, self.unit, budget=1, delta=self.delta, challenger=self.challenger,
                           clock=self._clock, sampler=lambda law, n: range(n))
        reg = interiors.register(record, ports, **kw)
        a.interiors, a.interior, a.interior_draw = interiors, reg, interiors.select(reg)
        return reg

    def deliver(self, draw: ReplayDraw, sessions: Sequence[str]) -> Outcome | None:
        """Step 4: the proof, delivered once to both registrations, after the interiors are registered (`interior`
        otherwise). A proof after the boundary's proof deadline is `late`, returned at once; otherwise None."""
        a = self._drawn_audit(draw)
        if a.interior is None:
            raise _refuse("interior", "the interiors are registered before any proof")
        late = self._boundary.deliver(draw.draw, sessions)
        a.sessions = list(sessions)
        if late is not None:
            return self._finish(a, late)
        assert a.interiors is not None and a.interior_draw is not None
        a.interiors.deliver(a.interior_draw, sessions)
        return None

    def outcome(self, draw: ReplayDraw, verdicts: Mapping[str, Mapping[str, Any]], statement: Mapping[str, Any], *,
                session_roots: Mapping[str, str] | None = None,
                session_registered: Mapping[str, str] | None = None) -> Outcome:
        """The audit's final outcome, once. With no proof delivered, `late` once the boundary's deadline has passed and
        `Pending` before. With one, the interior outcome (`Stream.outcome` over the interior registration), then the
        boundary outcome with the interior's as its `INTERIOR` verdict. `session_roots` (`{port: root}` from the
        sessions' statements) is split between them by value; `session_registered` is the boundary's."""
        a = self._drawn_audit(draw)
        if INTERIOR in verdicts:
            raise _refuse("verdicts", f"{INTERIOR!r} is the driver's own verdict")
        if a.sessions is None:
            return self._finish(a, self._boundary.outcome(draw.draw, verdicts, statement))
        roots_i = roots_b = None
        if session_roots is not None:
            roots_i = {p: r for p, r in session_roots.items() if p in self._interior_names}
            roots_b = {p: r for p, r in session_roots.items() if p not in self._interior_names}
        assert a.interiors is not None and a.interior_draw is not None
        if a.interior_outcome is None:
            a.interior_outcome = a.interiors.outcome(a.interior_draw, verdicts, statement, session_roots=roots_i)
        interior = {"of_record": False, "accepted": a.interior_outcome.accepted, "sessions": list(a.sessions),
                    "outcome": a.interior_outcome.record}
        boundary = self._boundary.outcome(draw.draw, {**verdicts, INTERIOR: interior}, statement, session_roots=roots_b,
                                          session_registered=session_registered)
        return self._finish(a, boundary)

    def _finish(self, a: _Audit, boundary: Outcome) -> Outcome:
        assert a.draw is not None
        reg, inner = a.registration, a.interior
        body = {"format": AUDIT, "stream": self.digest, "order": reg.order,
                "boundary": {"registration": dict(reg.record), "receipt": dict(reg.receipt), "outcome": boundary.record},
                "draw": dict(a.draw.replay), "replay_units": a.draw.proved,
                "interior": None if inner is None else {
                    "registration": dict(inner.record), "receipt": dict(inner.receipt),
                    "outcome": None if a.interior_outcome is None else a.interior_outcome.record},
                "sessions": a.sessions or [], "symbol": self._boundary.symbols[-1]}
        assert (body["symbol"] == LATE) == (boundary.verdict.codes == (LATE,))
        a.outcome = dataclasses.replace(boundary, record=_sha512(body))
        self._records.append(body)
        return a.outcome


__all__ = ["AUDIT", "INTERIOR", "STREAM", "ReplayDraw", "TwoStageStream"]
