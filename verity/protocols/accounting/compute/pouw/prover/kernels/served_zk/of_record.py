"""Lean is the served window's verifier (`verity.protocols.accounting.compute.pouw.verifier.verifier`). Each of PoUW's checks on the
challenger's path is its Lean's alone, with no Python answer beside it:

* ``registered``: PoUW's rule over the registration (`pouw-verifier-difftest registered`);
* ``registration``: one-stage's R1-R8 against the challenger's own copies (`spot-difftest registration`), with the
  fan-out and tied values its verdict publishes; a PoUW window registers no values;
* ``caller``: the caller's codes (`spot-flock-caller caller`, `VStar.Recursive.Caller.check`);
* ``public``: the service's public-items rule over the Stream's Spec (`spot-difftest check_public`);
* ``pass``: the pass's epoch under the challenger's coins (`pouw-verifier-difftest pass`);
* ``draw``: the sampled proof blocks are the sampling coin's, under the challenger key the window's parameters pin when it is
  certified (`pouw-verifier-difftest draw`);
* ``opening``: a sampled proof block's opening (`pouw-verifier-difftest opening`);
* ``answer``: PoUW's check over the stream's outcome (`pouw-verifier serve`).

Each verdict goes in the witness (`Record.witness`, one entry per check). The Python twins in `core_check`
(`registered_twin`, `opening_twin`, `pass_epoch_twin`, `draw_twin`, `case_twin`) answer the difftests only.

Verity imports neither Lean's executables nor the canonical JSON the checks' inputs are written in (`verity_difftest`): a
`Record`'s `canonical` and `start` raise here, and `benchmarks/pouw/served_zk/of_record_lean.py`'s `Record` supplies
them.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.circuits import codec
from verity.protocols.accounting.compute.pouw.verifier import window as Wd

REPO = Path(__file__).resolve().parents[8]
POUW = REPO / "verity/protocols/accounting/compute/pouw/verifier/lean/check"
PIECES = REPO / "verity/protocols/proof/spot/verifier/lean"
CALLER = REPO / "verity/protocols/proof/spot_flock/verifier/lean"
NEEDS_DIFFTEST = ("Record.{} needs verity_difftest, which verity does not import (compute-accounting/pouw-package-holds-kernels): "
                  "benchmarks/pouw/served_zk/of_record_lean.py's Record supplies it")
#: a check's outcome on one input: ``("output", its canonical JSON)`` or ``("reject", why)``
Outcome = tuple[str, str]
CHECKS = ("registered", "registration", "caller", "public", "pass", "draw", "opening", "answer")


def _value(o: Outcome) -> Any:
    return json.loads(o[1]) if o[0] == "output" else {"refused": o[1]}


# ---- the inputs, as the challenger holds them -----------------------------------------------------------------------

def core_params(params: Mapping[str, Any], check: Mapping[str, Any], query: str) -> dict:
    """`core_check`'s parameters of a served window: the check item and query digest, and the window's own numbers."""
    return {"check": dict(check), "context": params["context"], "delta": params["delta"], "eps": params["eps"],
            "grid_bits": params["grid_bits"], "length": dict(params["length"]), "query": query,
            "range": list(params["range"])}


def registration_params(program: Any, query: Mapping[str, Any], sampling_design: Mapping[str, Any], scheme: str,
                        ports: Mapping[str, Any], log: list[dict], delta: str) -> dict:
    """`spot-difftest registration`'s parameters for a window `Stream.register` receives: no leaf layer,
    anchors, partition, work table or registered values, which a PoUW window registers none of."""
    return {"anchors": [], "delta": delta, "law": dict(sampling_design), "leaf_layer": None, "log": [dict(r) for r in log],
            "partition": None, "ports": {p: dict(v) for p, v in ports.items()},
            "program": json.loads(codec.canonical_json(codec.encode_program(program))), "query": dict(query),
            "registered": None, "scheme": scheme, "work_table": None}


def public_params(spec: Any, record: Mapping[str, Any], scheme: str) -> dict:
    """`spot-difftest check_public`'s input for `service.check_public(spec, record)`: the record, the service's
    scheme, the check's query, the Spec's values (each its name and whether it is registered) and its declared items."""
    return {"record": dict(record), "scheme": scheme, "query": dict(spec.check.query),
            "values": [{"name": v.name, "registered": v.registered} for v in spec.values],
            "public": [i.item for i in spec.public]}


def service_answer(accepted: bool, codes: list[str], profile: Mapping[str, Any] | None, registration: str,
                   draw: Mapping[str, Any] | None, delta: str) -> dict:
    """The service's answer (`Core.Answer`) of a stream's outcome: on acceptance its profile (δ as the window's ratio when
    the float is it) and its sample."""
    base = {"protocol": Wd.ONE_STAGE, "commitment": registration}
    if accepted:
        d = profile["delta"]
        d = d if isinstance(d, str) else (delta if float(Fraction(delta)) == d else str(Fraction(d).limit_denominator(10**9)))
        return {**base, "verdict": "accept", "reason": None, "public": {"draw": dict(draw)}, "profile": {**profile, "delta": d}}
    return {**base, "verdict": "reject", "reason": ", ".join(codes or ["reject"]), "public": None, "profile": None}


class Record:
    """Each check's verdict, Lean's. The servers start at their first check and build their executables (`lake build`)
    then, unless `exes` names them (``pouw-verifier``, ``pouw-verifier-difftest``, ``spot-difftest``,
    ``spot-flock-caller``)."""

    def __init__(self, exes: Mapping[str, str | Path] | None = None) -> None:
        self.exes = dict(exes or {})
        self.witness: list[dict] = []
        self._servers: dict[str, Any] = {}

    def canonical(self, j: Any) -> str:
        """The canonical JSON text of `j` (`verity_difftest.canonical`, as `Flock.canon` writes it); `TypeError` outside
        its domain."""
        raise NotImplementedError(NEEDS_DIFFTEST.format("canonical"))

    def start(self, exe: str, package: Path, *args: str) -> Any:
        """A server of Lean's executable `exe` in the Lake package `package`, with `args` (`verity_difftest.Server` on
        `self.exes`' path for it, or its `build`)."""
        raise NotImplementedError(NEEDS_DIFFTEST.format("start"))

    def hexed(self, j: Any) -> str:
        return self.canonical(j).encode().hex()

    def _server(self, exe: str, package: Path, *args: str) -> Any:
        if exe not in self._servers:
            self._servers[exe] = self.start(exe, package, *args)
        return self._servers[exe]

    def close(self) -> None:
        for s in self._servers.values():
            s.close()
        self._servers.clear()

    def _piece(self, check: str, exe: str, package: Path, fn: str, x: Any) -> Any:
        """Lean's output on `x` (`{"refused": why}` for an input it refuses)."""
        lean = _value(self._server(exe, package).call(fn, x))
        self.witness.append({"check": check, "lean": lean})
        return lean

    def registered(self, x: dict) -> dict:
        """``{"reason"}`` (null when PoUW's rule holds), or ``{"refused"}`` for parameters or a registration it can't read."""
        return self._piece("registered", "pouw-verifier-difftest", POUW, "registered", x)

    def registration(self, params: dict, record: Any) -> list[str]:
        """R1-R8's codes for `record` against `params` (`registration_params`), Lean's (`spot-difftest registration`,
        with the fan-out and tied values its answer carries in the witness)."""
        out = self._piece("registration", "spot-difftest", PIECES, "registration", {"params": params, "record": record})
        return out.get("codes", ["registration"])

    def caller(self, params: dict, record: Any) -> list[str]:
        """The caller's codes (`VStar.Recursive.Caller.check`) for `record` against `params`, the registration's,
        Lean's (`spot-flock-caller caller`)."""
        out = self._piece("caller", "spot-flock-caller", CALLER, "caller", {"params": params, "record": record})
        return out.get("codes", ["caller"])

    def public(self, x: dict) -> dict:
        """The service's public-items rule's ``{"ok", "codes", "detail"}`` on `x` (`public_params`), or ``{"refused"}``
        for an input it can't read."""
        return self._piece("public", "spot-difftest", PIECES, "check_public", x)

    def pass_epoch(self, x: dict) -> dict:
        return self._piece("pass", "pouw-verifier-difftest", POUW, "pass", x)

    def draw(self, x: dict) -> dict | None:
        return self._piece("draw", "pouw-verifier-difftest", POUW, "draw", x)

    def opening(self, x: dict) -> Any:
        return self._piece("opening", "pouw-verifier-difftest", POUW, "opening", x)

    def answer(self, case: dict) -> dict:
        """PoUW's outcome (`Core.Answer`) of a `serve` case."""
        lean = self._server("pouw-verifier", POUW, "serve").exchange(case)
        self.witness.append({"check": "answer", "lean": lean})
        return lean


__all__ = ["CHECKS", "Outcome", "Record", "core_params", "public_params", "registration_params",
           "service_answer"]
