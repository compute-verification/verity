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

    check_baseline(instance, trusted) -> the same, then θ's provenance (`NciCheck.Baseline`)

`check_baseline` is `check_desc` followed by the check of each θ entry's `provenance` against the trusted input (the
deployment's registration log, the run's own record and the program function's parameters), with `NciCheck.Baseline`'s
codes in its order; `check_baseline_unrolled` is `check` followed by the same, on an instance expanded by
`expand_baseline`. θ holds registered public data only: an entry of any kind but `registered`, an earlier run's `output`
among them, is refused `theta-provenance`. One log is one deployment, and every entry cites its first record, the first
run's own included; any other record is refused `theta-not-at-deployment-start`.
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


#: `Flock.Steps.FUEL`: the most steps `Prog.gate`'s walk to an operand takes
FUEL = 2 ** 62


def walk_bound(program) -> int:
    """`NciCheck.Desc.walkBound`: `(A + 1) · (H + 2)` for the root's activations `A` (its own and each composite node
    member's, once per Definition and multiplied through batches and scans) and the depth `H` they nest to."""
    from verity.ml.circuits.parts import n_members
    memo: dict[int, tuple[int, int]] = {}

    def of(fn) -> tuple[int, int]:
        if not hasattr(fn, "body"):
            return 0, 0
        if id(fn) not in memo:
            subs = [(n_members(n), *of(n.fn)) for n in fn.body.nodes]
            memo[id(fn)] = (1 + sum(m * a for m, a, _ in subs), 1 + max((h for _, _, h in subs), default=0))
        return memo[id(fn)]
    acts, depth = of(program.root)
    return (acts + 1) * (depth + 2)


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
    if walk_bound(program) > FUEL // 2:
        raise Malformed("program")
    if any(n.form == "scan" and n.n == 0 and n.out.leaves > 0
           for fn in program.defs.values() if hasattr(fn, "body") for n in fn.body.nodes):
        raise Malformed("program")
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


# θ's provenance (`NciCheck.Baseline`): after the policy check, each θ entry against the trusted input

REGISTRATION_V1 = "verity/registration/v1"
REGISTRATION_V2 = "verity/registration/v2"
TRUSTED_KEYS = {"log", "run", "params"}
REGISTERED_KEYS = {"kind", "registration", "value", "root"}


def record_digest(rec: dict) -> str:
    """`registration.digest`: SHA-512 of the record's own format (v1 or v2), a NUL and its canonical JSON."""
    import hashlib
    tag = REGISTRATION_V2 if rec.get("format") == REGISTRATION_V2 else REGISTRATION_V1
    return hashlib.sha512(tag.encode() + b"\0" + codec.canonical_json(rec)).hexdigest()


def lower(t) -> list[tuple[int, int]]:
    """§5.1: the input nodes a parameter of type `t` lowers to, (width, n) each (`SampledProofs.Registered.lower`)."""
    from verity.ml.circuits.types import Array, Record, Tuple
    ws = t.width_set()
    if len(ws) == 1:
        return [(next(iter(ws)), t.leaves)]
    if isinstance(t, Array):
        return lower(t.elem) * t.n
    if isinstance(t, Tuple):
        return [x for f in t.fields for x in lower(f)]
    if isinstance(t, Record):
        return [x for _, f in t.fields for x in lower(f)]
    return []


def owners(program, params) -> list[int | None] | None:
    """Per root node, the index of the parameter whose input nodes it is one of, when the parameters lower onto the root
    body's input nodes exactly; None otherwise."""
    nodes = program.root.body.nodes
    flat = [(i, wn) for i, (_, t) in enumerate(params) for wn in lower(t)]
    own: list[int | None] = [None] * len(nodes)
    at = 0
    for k, n in enumerate(nodes):
        if not is_input(n.fn):
            continue
        members = n.n if n.form in ("batch", "scan") else 1
        if at >= len(flat) or flat[at][1] != (n.fn.ret.w, members):
            return None
        own[k] = flat[at][0]
        at += 1
    return own if at == len(flat) else None


def trusted_of(tr) -> dict:
    """The trusted input: the log with each record's digest (R6's chain), the run's position, the program function's
    parameters; `trusted` otherwise."""
    if not (isinstance(tr, dict) and set(tr) == TRUSTED_KEYS and isinstance(tr["log"], list)):
        raise Malformed("trusted")
    digests: list[str] = []
    for rec in tr["log"]:
        if not isinstance(rec, dict) or rec.get("previous") != (digests[-1] if digests else None):
            raise Malformed("trusted")
        try:
            digests.append(record_digest(rec))
        except (ValueError, TypeError) as e:
            raise Malformed("trusted") from e
    run, ps = tr["run"], tr["params"]
    if not (isinstance(run, str) and run in digests and isinstance(ps, list)):
        raise Malformed("trusted")
    params = []
    for p in ps:
        if not (isinstance(p, list) and len(p) == 2 and isinstance(p[0], str)):
            raise Malformed("trusted")
        try:
            params.append((p[0], type_from_json(p[1])))
        except (ValueError, TypeError, KeyError, IndexError) as e:
            raise Malformed("trusted") from e
    if len({n for n, _ in params}) != len(params):
        raise Malformed("trusted")
    return {"log": tr["log"], "digests": digests, "run": digests.index(run), "params": params}


def last_with(rec: dict, field: str, key: str, value: str):
    es = rec.get(field)
    hits = [e for e in es if isinstance(e, dict) and e.get(key) == value] if isinstance(es, list) else []
    return hits[-1] if hits else None


def dims(e: dict):
    r, w, b = e.get("leaves"), e.get("words"), e.get("word_bits", 16)
    return (r, w, b) if all(map(_nat, (r, w, b))) else None


def provenance(inst, tr, node) -> None:
    """θ's provenance on an instance the policy check gave a verdict: θ entry t lies in root node node(offsets, θ[t])."""
    prov = inst.get("provenance")
    if not (isinstance(prov, list) and len(prov) == len(inst["theta"])):
        raise Malformed("theta-provenance")
    for p in prov:
        if not (isinstance(p, dict) and p.get("kind") == "registered" and set(p) == REGISTERED_KEYS
                and all(isinstance(v, str) for v in p.values())):
            raise Malformed("theta-provenance")
    T = trusted_of(tr)
    desc = json.loads(inst["program"])
    program = codec.decode_program(desc, declared(desc))
    offs = program.root.body.offsets
    nodes = program.root.body.nodes
    own = owners(program, T["params"]) or [None] * len(nodes)
    counts = [0] * len(T["params"])
    for k, i in enumerate(own):
        if i is not None:
            counts[i] += offs[k + 1] - offs[k]
    digest = codec.program_digest(desc)
    run = T["log"][T["run"]]
    for t, p in zip(inst["theta"], prov, strict=True):
        k = node(offs, t)
        D = p["registration"]
        if D not in T["digests"]:
            raise Malformed("theta-record-unknown")
        if T["digests"].index(D) != 0:
            raise Malformed("theta-not-at-deployment-start")
        e = last_with(T["log"][0], "registered", "value", p["value"])
        if e is None or e.get("root") != p["root"]:
            raise Malformed("theta-root")
        if run.get("program") != digest or own[k] is None:
            raise Malformed("theta-unbound")
        i = own[k]
        mine = last_with(run, "registered", "value", T["params"][i][0])
        if mine is None or mine.get("root") != p["root"]:
            raise Malformed("theta-unbound")
        bits = 16 * -(-nodes[k].fn.ret.w // 16)
        m = dims(e)
        if not (m is not None and m[0] * m[1] == counts[i] and m[2] == bits):
            raise Malformed("theta-width")


def _node_of(offs: list[int], g: int) -> int:
    """The root node holding gate g (`NciCheck.Desc.nodeOf`)."""
    import bisect
    return bisect.bisect_right(offs, g) - 1


def check_baseline(inst, tr) -> dict:
    """`NciCheck.Baseline.check`: `check_desc`, then θ's provenance; the verdict is the policy check's."""
    out = check_desc(inst)
    if "error" in out:
        return out
    try:
        provenance(inst, tr, lambda _offs, k: k)
    except Malformed as e:
        return {"error": e.code}
    return out


def check_baseline_unrolled(inst, tr) -> dict:
    """`NciCheck.Baseline.checkUnrolled`: `check`, then θ's provenance, gate g of θ in the root node holding it."""
    out = check(inst)
    if "error" in out:
        return out
    try:
        provenance(inst, tr, _node_of)
    except Malformed as e:
        return {"error": e.code}
    return out


def expand_baseline(inst) -> dict:
    """`expand`, with θ's provenance repeated for every gate of its node."""
    _, _, spans = root_nodes(inst["program"])
    prov = [p for k, p in zip(inst["theta"], inst["provenance"], strict=True) for _ in range(*spans[k])]
    return expand(inst) | {"provenance": prov}
