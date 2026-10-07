"""The NCI policy's check of a declared division, straight from the definitions: the reference `NciCheck.check` is held to.

    check(instance) -> {"gates", "ius", "complies", "failure"} | {"error": code}

`Nci.Complies Γ α` (`Security/Definitions/Nci.lean`): the computed gates of `D` divide into isolation units of at most `Γ`
gates, each with `novelReads R = width(into(R) \\ θ) ≤ α · size R`, where `into(R)` is the gates outside `R` with a wire
into `R` (`Security/Definitions/Core/Circuit.lean`). The instance and the refusal codes are the ones `NciCheck.Check`'s
docstring gives: the declaration's circuit is the canonical layout of a descriptor v1 Program (`verity.ml.circuits`),
gate `g` of width `Value<w>` bits, an input gate when its primitive is an input (`program.is_input`, with no operands),
reading the gates its operands name; θ is a list of input gate ids; the division is one owner per gate, `"input"` or an IU index.
A primitive is taken as the descriptor declares it, as C-Flock's decoder takes it, so no registry is needed.
"""
from __future__ import annotations

import json
from fractions import Fraction

from verity.ml.circuits import codec
from verity.ml.circuits.defs import PrimitiveDefinition, Registry
from verity.ml.circuits.program import input_id, is_input
from verity.ml.circuits.types import type_from_json

FIELDS = ("program", "theta", "owner", "gamma", "alpha")


class Malformed(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _nat(x) -> bool:
    return type(x) is int and x >= 0


def _unevaluated(*_a):
    raise NotImplementedError("the NCI check evaluates no gate")


def declared(desc: dict) -> Registry:
    """A registry of the descriptor's primitives as it declares them (inputs aside, which the decoder makes itself)."""
    reg = Registry()
    for fid, d in desc["definitions"].items():
        if d.get("form") == "primitive" and input_id(fid) is None:
            name, _, version = fid.rpartition("_v")
            reg.add(PrimitiveDefinition(name, int(version), tuple((n, type_from_json(t)) for n, t in d["params"]),
                                        type_from_json(d["ret"]), _unevaluated, register=False))
    return reg


def circuit(program: str) -> tuple[list[int], list[bool], list[set[int]]]:
    """Each gate's width, whether it is an input gate, and the gates it reads."""
    try:
        desc = json.loads(program)
        C = codec.decode_program(desc, declared(desc)).circuit
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise Malformed("program") from e
    w, inp, reads = [], [], []
    for g in range(C.size):
        gate = C.gate(g)
        if is_input(gate.prim) and gate.operands:
            raise Malformed("program")
        if any(a >= g for a in gate.operands):
            raise Malformed("not-topological")
        w.append(gate.prim.ret.w)
        inp.append(is_input(gate.prim))
        reads.append(set(gate.operands))
    return w, inp, reads


def evaluate(inst) -> dict:
    if not (isinstance(inst, dict) and all(k in inst for k in FIELDS) and isinstance(inst["program"], str)
            and isinstance(inst["theta"], list) and isinstance(inst["owner"], list)):
        raise Malformed("instance")
    gamma, alpha = inst["gamma"], inst["alpha"]
    if not (_nat(gamma) and isinstance(alpha, list) and len(alpha) == 2 and all(map(_nat, alpha)) and alpha[1] > 0):
        raise Malformed("params")
    alpha = Fraction(*alpha)
    w, inp, reads = circuit(inst["program"])
    N = len(w)
    theta: set[int] = set()
    for t in inst["theta"]:
        if not (_nat(t) and t < N):
            raise Malformed("theta-unknown")
        if not inp[t]:
            raise Malformed("theta-not-input")
        if t in theta:
            raise Malformed("theta-duplicate")
        theta.add(t)
    owner = inst["owner"]
    if len(owner) != N:
        raise Malformed("owner-length")
    for g, o in enumerate(owner):
        if not (o == "input" or _nat(o)):
            raise Malformed("owner-entry")
        if inp[g] and o != "input":
            raise Malformed("input-in-iu")
        if not inp[g] and o == "input":
            raise Malformed("gate-in-no-iu")
    used = {o for o in owner if o != "input"}
    if used != set(range(len(used))):
        raise Malformed("empty-iu")
    ius = []
    for u in range(len(used)):
        R = {g for g in range(N) if owner[g] == u}
        into = {a for g in R for a in reads[g] if a not in R}
        ius.append([len(R), sum(w[a] for a in into - theta)])
    failure = None
    for u, (size, novel) in enumerate(ius):
        if size > gamma:
            failure = [u, "size"]
        elif novel > alpha * size:
            failure = [u, "nci"]
        if failure:
            break
    return {"gates": N, "ius": ius, "complies": failure is None, "failure": failure}


def check(inst) -> dict:
    try:
        return evaluate(inst)
    except Malformed as e:
        return {"error": e.code}
