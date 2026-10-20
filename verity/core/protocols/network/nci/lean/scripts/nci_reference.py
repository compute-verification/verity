"""The NCI policy's check of a declared division, straight from the definitions: the reference `NciCheck.check` is held to.

    check(instance) -> {"gates", "ius", "complies", "failure"} | {"error": code}

`Nci.Complies Γ α` (`Security/Definitions/Nci.lean`): the computed gates of `D` divide into isolation units of at most `Γ`
gates, each with `novelReads R = width(into(R) \\ θ) ≤ α · size R`, where `into(R)` is the gates outside `R` with a wire
into `R` (`Security/Definitions/Core/Circuit.lean`). The instance and the refusal codes are the ones `NciCheck.Check`'s
docstring gives: the declaration's circuit is the canonical layout of a descriptor v1 Program (`verity.ml.circuits`),
gate `g` of width `Value<w>` bits, an input gate when its primitive is an input (`program.is_input`, with no operands),
reading the gates its operands name; θ is a list of ids of input gates bound at registration (`program.is_registered`;
one bound per run is refused, `theta-not-registered`); the division is one owner per gate, `"input"` or an IU index.
A primitive is taken as the descriptor declares it, as C-Flock's decoder takes it, so no registry is needed.

    check_desc(instance) -> the same, for `NciCheck.Desc`'s instance: θ and the owners per root node

`check_desc` refuses as `NciCheck.Desc` does, in its order, and otherwise is `check` on the instance expanded gate by gate
(`expand`), so the two checks are held to one reference.
"""
from __future__ import annotations

import json
from fractions import Fraction

from verity.ml.circuits import codec
from verity.ml.circuits.defs import PrimitiveDefinition, Registry
from verity.ml.circuits.program import input_id, is_input, is_registered
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


def circuit(program: str) -> tuple[list[int], list[bool], list[bool], list[set[int]]]:
    """Each gate's width, whether it is an input gate, whether it is one bound at registration, and the gates it
    reads."""
    try:
        desc = json.loads(program)
        C = codec.decode_program(desc, declared(desc)).circuit
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise Malformed("program") from e
    w, inp, reg, reads = [], [], [], []
    for g in range(C.size):
        gate = C.gate(g)
        if is_input(gate.prim) and gate.operands:
            raise Malformed("program")
        if any(a >= g for a in gate.operands):
            raise Malformed("not-topological")
        w.append(gate.prim.ret.w)
        inp.append(is_input(gate.prim))
        reg.append(is_registered(gate.prim))
        reads.append(set(gate.operands))
    return w, inp, reg, reads


def evaluate(inst) -> dict:
    if not (isinstance(inst, dict) and all(k in inst for k in FIELDS) and isinstance(inst["program"], str)
            and isinstance(inst["theta"], list) and isinstance(inst["owner"], list)):
        raise Malformed("instance")
    gamma, alpha = inst["gamma"], inst["alpha"]
    if not (_nat(gamma) and isinstance(alpha, list) and len(alpha) == 2 and all(map(_nat, alpha)) and alpha[1] > 0):
        raise Malformed("params")
    alpha = Fraction(*alpha)
    w, inp, reg, reads = circuit(inst["program"])
    N = len(w)
    theta: set[int] = set()
    for t in inst["theta"]:
        if not (_nat(t) and t < N):
            raise Malformed("theta-unknown")
        if not inp[t]:
            raise Malformed("theta-not-input")
        if not reg[t]:
            raise Malformed("theta-not-registered")
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


def root_nodes(program: str) -> tuple[list[bool], list[bool], list[tuple[int, int]]]:
    """Each root node: whether it is an input node, whether it is one bound at registration, and its gates `[lo, hi)`
    of the canonical layout."""
    desc = json.loads(program)
    root = codec.decode_program(desc, declared(desc)).root
    offs = root.body.offsets
    nodes = root.body.nodes
    return ([is_input(n.fn) for n in nodes], [is_registered(n.fn) for n in nodes],
            [(offs[k], offs[k + 1]) for k in range(len(nodes))])


def expand(inst) -> dict:
    """An instance of `NciCheck.Desc` (θ and the owners per root node) as `check` reads one: θ every gate of its nodes,
    and each gate the owner of its root node. The instance is well-formed as `NciCheck.Desc` reads it."""
    _, _, spans = root_nodes(inst["program"])
    theta = [g for k in inst["theta"] for g in range(*spans[k])]
    owner = [inst["owner"][k] for k, (lo, hi) in enumerate(spans) for _ in range(lo, hi)]
    return inst | {"theta": theta, "owner": owner}


def evaluate_desc(inst) -> dict:
    """`NciCheck.Desc.evaluate`: its refusals in its order, then `evaluate` on the expansion."""
    if not (isinstance(inst, dict) and all(k in inst for k in FIELDS) and isinstance(inst["program"], str)
            and isinstance(inst["theta"], list) and isinstance(inst["owner"], list)):
        raise Malformed("instance")
    gamma, alpha = inst["gamma"], inst["alpha"]
    if not (_nat(gamma) and isinstance(alpha, list) and len(alpha) == 2 and all(map(_nat, alpha)) and alpha[1] > 0):
        raise Malformed("params")
    try:
        desc = json.loads(inst["program"])
        program = codec.decode_program(desc, declared(desc))
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise Malformed("program") from e
    from verity.ml.circuits.partition_object import order_violation
    if order_violation(program) is not None:
        raise Malformed("not-topological")
    inp, reg, spans = root_nodes(inst["program"])
    K = len(inp)
    seen: set[int] = set()
    for t in inst["theta"]:
        if not (_nat(t) and t < K):
            raise Malformed("theta-unknown")
        if not inp[t]:
            raise Malformed("theta-not-input")
        if not reg[t]:
            raise Malformed("theta-not-registered")
        if t in seen:
            raise Malformed("theta-duplicate")
        seen.add(t)
    owner = inst["owner"]
    if len(owner) != K:
        raise Malformed("owner-length")
    for k, o in enumerate(owner):
        if not (o == "input" or _nat(o)):
            raise Malformed("owner-entry")
        if spans[k][1] > spans[k][0]:
            if inp[k] and o != "input":
                raise Malformed("input-in-iu")
            if not inp[k] and o == "input":
                raise Malformed("gate-in-no-iu")
    used = {o for k, o in enumerate(owner) if o != "input" and spans[k][1] > spans[k][0]}
    if used != set(range(len(used))):
        raise Malformed("empty-iu")
    return evaluate(expand(inst))


def check_desc(inst) -> dict:
    try:
        return evaluate_desc(inst)
    except Malformed as e:
        return {"error": e.code}
