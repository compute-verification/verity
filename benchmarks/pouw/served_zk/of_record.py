"""Lean is the served window's verifier of record (`verity.core.protocols.pouw.verifier`). Each of PoUW's checks on the
challenger's path runs its Lean and its Python reference on one input:

* ``registered``: PoUW's rule over the registration (`pouw-check-difftest registered`, `core_check.registered`);
* ``registration``: one-stage's R1-R6 against the challenger's own copies (`sampled-proofs-difftest registration`,
  `registration.check`); R7 (registered values) has no Lean form yet, and a PoUW window registers none;
* ``public``: the service's public-items rule over the Stream's Spec (`sampled-proofs-difftest check_public`,
  `service.check_public`);
* ``pass``: the pass's epoch under the challenger's coins (`pouw-check-difftest pass`, `core_check.pass_epoch`);
* ``draw``: the drawn units are the draw coin's, under the challenger key the window's parameters pin when it is
  certified (`pouw-check-difftest draw`, `core_check.draw`);
* ``opening``: a drawn unit's opening (`pouw-check-difftest opening`, `core_check.opening`), with ``prelude``,
  `served_commit.check`'s reading of the registration before it, which may refuse only what the opening refuses;
* ``answer``: PoUW's check over the stream's outcome (`pouw-check serve`, `core_check.case`).

Lean's verdict is the record's, and Python's goes beside it in the witness (`VerdictPair.witness`, one entry per check). The
two agree when their outputs are the same canonical JSON (an `answer`'s reason included), or when both refuse the input.
A disagreement raises `Disagree`, which the challenger turns into the refusal ``lean-python-disagree: <check>``; a failure
is visible and nothing retries it. `VerdictPair()` takes the side of record from `verifier.RECORD_VERDICT`; with it ``"python"``,
or with `VerdictPair(lean=False)`, the references run alone (the witness then holds no Lean verdict), which is how the switch is
undone.
"""
from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping

import verity_difftest as D

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import verifier as V
from verity.ml.circuits import codec

REPO = Path(__file__).resolve().parents[3]
POUW = REPO / "verity/core/protocols/pouw/lean/check"
PIECES = REPO / "verity/core/service/sampled_proofs"
DISAGREE = "lean-python-disagree"
CHECKS = ("registered", "registration", "public", "pass", "draw", "opening", "prelude", "answer")


class Disagree(Exception):
    def __init__(self, check: str, lean: Any, python: Any) -> None:
        super().__init__(f"{DISAGREE}: {check}")
        self.check, self.lean, self.python = check, lean, python

    @property
    def code(self) -> str:
        return f"{DISAGREE}: {self.check}"


def hexed(j: Any) -> str:
    return D.canonical(j).encode().hex()


def _python(f: Callable[[Any], Any], x: Any) -> D.Outcome:
    try:
        return "output", D.canonical(f(copy.deepcopy(x)))
    except (ValueError, TypeError, KeyError) as e:
        return "reject", f"{type(e).__name__}: {e}"


def _same(lean: D.Outcome, py: D.Outcome) -> bool:
    return lean == py if lean[0] == "output" else py[0] == "reject"


def _value(o: D.Outcome) -> Any:
    return json.loads(o[1]) if o[0] == "output" else {"refused": o[1]}


# ---- the inputs, as the challenger holds them -----------------------------------------------------------------------

def core_params(params: Mapping[str, Any], check: Mapping[str, Any], query: str) -> dict:
    """`core_check`'s parameters of a served window: the check item and query digest, and the window's own numbers."""
    return {"check": dict(check), "context": params["context"], "delta": params["delta"], "eps": params["eps"],
            "grid_bits": params["grid_bits"], "length": dict(params["length"]), "query": query,
            "range": list(params["range"])}


def registration_params(program: Any, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str,
                        ports: Mapping[str, Any], log: list[dict], delta: str) -> dict:
    """`sampled-proofs-difftest registration`'s parameters for `Stream.register`'s `registration.check`: no leaf layer,
    anchors, partition, work table or registered values, which a PoUW window registers none of."""
    return {"anchors": [], "delta": delta, "law": dict(law), "leaf_layer": None, "log": [dict(r) for r in log],
            "partition": None, "ports": {p: dict(v) for p, v in ports.items()},
            "program": json.loads(codec.canonical_json(codec.encode_program(program))), "query": dict(query),
            "registered": None, "scheme": scheme, "work_table": None}


def public_params(spec: Any, record: Mapping[str, Any], scheme: str) -> dict:
    """`sampled-proofs-difftest check_public`'s input for `service.check_public(spec, record)`: the record, the service's
    scheme, the check's query, the Spec's values (each its name and whether it is registered) and its declared items."""
    return {"record": dict(record), "scheme": scheme, "query": dict(spec.check.query),
            "values": [{"name": v.name, "registered": v.registered} for v in spec.values],
            "public": [i.item for i in spec.public]}


def service_answer(accepted: bool, codes: list[str], profile: Mapping[str, Any] | None, registration: str,
                   draw: Mapping[str, Any] | None, delta: str) -> dict:
    """The service's answer (`Core.Answer`) of a stream's outcome: on acceptance its profile (δ as the window's ratio when
    the float is it) and its draw."""
    base = {"protocol": CC.ONE_STAGE, "commitment": registration}
    if accepted:
        d = profile["delta"]
        d = d if isinstance(d, str) else (delta if float(Fraction(delta)) == d else str(Fraction(d).limit_denominator(10**9)))
        return {**base, "verdict": "accept", "reason": None, "public": {"draw": dict(draw)}, "profile": {**profile, "delta": d}}
    return {**base, "verdict": "reject", "reason": ", ".join(codes or ["reject"]), "public": None, "profile": None}


# ---- the references ------------------------------------------------------------------------------------------------

def _params_commitment(x: dict) -> tuple[dict, dict]:
    if not ("params" in x and isinstance(x.get("commitment"), str)):
        raise CC.Refused("the input's params or commitment is malformed")
    return CC.params(x["params"]), CC.commitment(bytes.fromhex(x["commitment"]))


def py_registered(x: dict) -> dict:
    return {"reason": CC.registered(*_params_commitment(x))}


def py_opening(x: dict) -> str | None:
    p, c = _params_commitment(x)
    salt = None if x.get("salt") is None else bytes.fromhex(x["salt"])
    return CC.opening(p, c, x["units"], x["unit"], bytes.fromhex(x["pub"]), x["opening"], salt)


def py_pass(x: dict) -> dict:
    raw = None if x["raw"] is None else bytes.fromhex(x["raw"])
    certified = (None if x.get(k) is None else bytes.fromhex(x[k]) for k in ("reveal", "committed", "stream", "prover"))
    how, refused = CC.pass_epoch(x["coins"], raw, x["context"], x["recorded"], *certified)
    return {"how": how} if refused is None else {"refused": list(refused)}


def py_draw(x: dict) -> dict | None:
    refused = CC.draw(x["coins"], x["recorded"], x["opening"], x["stream"], x["registration"], x["receipt"], x["k"],
                      x["population"], x["units"], x.get("key"))
    return None if refused is None else {"refused": list(refused)}


class VerdictPair:
    """Both verdicts of each check, Lean's of record. `lean` False runs the references alone; by default it is whether
    `verifier.RECORD_VERDICT` is ``"lean"``. The servers start at their first check and build their executables (`lake
    build`) then, unless `exes` names them (``pouw-check``, ``pouw-check-difftest``, ``sampled-proofs-difftest``)."""

    def __init__(self, lean: bool | None = None, exes: Mapping[str, str | Path] | None = None) -> None:
        self.lean = V.RECORD_VERDICT == "lean" if lean is None else lean
        self.exes = dict(exes or {})
        self.witness: list[dict] = []
        self._servers: dict[str, D.Server] = {}

    def _server(self, exe: str, package: Path, *args: str) -> D.Server:
        if exe not in self._servers:
            path = self.exes.get(exe) or D.build(package, exe)
            self._servers[exe] = D.Server([str(path), *args])
        return self._servers[exe]

    def close(self) -> None:
        for s in self._servers.values():
            s.close()
        self._servers.clear()

    def _note(self, check: str, py: Any, lean: Any, same: bool | None) -> None:
        self.witness.append({"check": check, "python": py, "lean": lean, "same": same, "of_record": "lean" if self.lean
                             else "python"})

    def _piece(self, check: str, exe: str, package: Path, fn: str, x: Any, reference: Callable[[Any], Any]) -> Any:
        """Lean's output on `x` (`{"refused": why}` for an input it refuses), once Python's agrees."""
        py = _python(reference, x)
        if not self.lean:
            self._note(check, _value(py), None, None)
            return _value(py)
        lean = self._server(exe, package).call(fn, x)
        same = _same(lean, py)
        self._note(check, _value(py), _value(lean), same)
        if not same:
            raise Disagree(check, _value(lean), _value(py))
        return _value(lean)

    def registered(self, x: dict) -> dict:
        """``{"reason"}`` (null when PoUW's rule holds), or ``{"refused"}`` for parameters or a registration it can't read."""
        return self._piece("registered", "pouw-check-difftest", POUW, "registered", x, py_registered)

    def registration(self, params: dict, record: Any, python: Callable[[], list[str]]) -> list[str]:
        """R1-R6's codes for `record` against `params` (`registration_params`); `python` is the reference's call."""
        out = self._piece("registration", "sampled-proofs-difftest", PIECES, "registration",
                          {"params": params, "record": record}, lambda _: {"codes": list(python())})
        return out["codes"] if "codes" in out else ["registration"]

    def public(self, x: dict, python: Callable[[], dict]) -> dict:
        """The service's public-items rule's ``{"ok", "codes", "detail"}`` on `x` (`public_params`), or ``{"refused"}``
        for an input it can't read; `python` is the reference's call (`service.check_public(...).to_json()`)."""
        return self._piece("public", "sampled-proofs-difftest", PIECES, "check_public", x, lambda _: python())

    def prelude(self, python: str | None, lean: str | None) -> str | None:
        """`served_commit.check`'s own reading of the registration before a unit's opening (`python`: its level in the
        window's range, its window domain the served one) beside the opening's reason of record (`lean`), which reads the
        domain itself, as PoUW's rule read the level at receipt.  They disagree only when the reading refuses an opening
        Lean accepts; the reason of record is `lean`'s."""
        same = python is None or lean is not None
        self._note("prelude", {"reason": python}, {"reason": lean} if self.lean else None, same if self.lean else None)
        if self.lean and not same:
            raise Disagree("prelude", {"reason": lean}, {"reason": python})
        return lean if self.lean else python or lean

    def pass_epoch(self, x: dict) -> dict:
        return self._piece("pass", "pouw-check-difftest", POUW, "pass", x, py_pass)

    def draw(self, x: dict) -> dict | None:
        return self._piece("draw", "pouw-check-difftest", POUW, "draw", x, py_draw)

    def opening(self, x: dict) -> Any:
        return self._piece("opening", "pouw-check-difftest", POUW, "opening", x, py_opening)

    def answer(self, case: dict) -> dict:
        """PoUW's outcome (`Core.Answer`) of a `serve` case, the whole outcome compared, reason included."""
        py = CC.case(copy.deepcopy(case))
        if not self.lean:
            self._note("answer", py, None, None)
            return py
        lean = self._server("pouw-check", POUW, "serve").exchange(case)
        same = D.canonical(lean) == D.canonical(py)
        self._note("answer", py, lean, same)
        if not same:
            raise Disagree("answer", lean, py)
        return lean


__all__ = ["CHECKS", "DISAGREE", "Disagree", "VerdictPair", "core_params", "hexed", "public_params", "registration_params",
           "service_answer"]
