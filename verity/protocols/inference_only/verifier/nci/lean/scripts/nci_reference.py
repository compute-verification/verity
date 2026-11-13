"""The inference-only policy's check of a declared division, straight from the definitions: the reference `NciCheck.check` is held to.

    check(instance) -> {"gates", "ius", "complies", "failure"} | {"error": code}

`Nci.CompliesRate Γ α` (`security/Definitions/Nci.lean`): the computed gates of `D` divide into isolation units of at
most `Γ` gates, each with `novelReads R = width(into(R) \\ θ) ≤ α · size R`, where `into(R)` is the gates outside `R`
with a wire into `R` (`security/Definitions/Core/Circuit.lean`). The instance and the refusal codes are the ones
`NciCheck.Check`'s docstring gives: the declaration's circuit is the canonical layout of a descriptor v1 Program
(`verity.circuits`), gate `g` of width `Value<w>` bits, an input gate when its primitive is an input
(`program.is_input`, with no operands), reading the gates its operands name; θ is a list of ids of input gates bound at
registration (`program.is_registered`; one bound per run is refused, `theta-not-registered`); the division is one owner
per gate, `"input"` or an IU index. A primitive is taken as the descriptor declares it, as Flock's decoder takes it, so
no registry is needed.

    check_desc(instance) -> the same, for `NciCheck.Desc`'s instance: θ and the owners per root node

`check_desc` refuses as `NciCheck.Desc` does, in its order, and otherwise is `check` on the instance expanded gate by gate
(`expand`), so the two checks are held to one reference.

    check_count(instance) -> {"complies"} | {"error": code}
    check_count_detail(instance) -> {"blocks", "total", "complies", "failure"} | {"error": code}

`check_count_detail` is `NciCheck.Count`'s count of `(c, k)`-inference blocks, with its refusals in its order,
activation by activation where `NciCheck.Count` reads one summary per Definition: from the root down through the
Definitions outside every block, each activation of a block Definition is a block (`walk_blocks`), and a block
Definition's Boolean gates and fresh bits are read off one activation of it on its own, gate by gate (`block_row`): the
table's gates of each of its primitive gates, and the widths of the parameter leaves its gates read (each reference
resolved up to the activation's parameters, `source`) over the parameters that are not θ at every one of its blocks. It
is the private record's, never published; `check_count` is what the count publishes, the judgment against n alone.

    check_baseline(instance, trusted) -> the same, then θ's provenance (`NciCheck.Baseline`)

`check_baseline` is `check_desc` followed by the check of each θ entry's `provenance` against the trusted input (the
deployment's registration log, the run's own record and the program function's parameters), with `NciCheck.Baseline`'s
codes in its order; `check_baseline_unrolled` is `check` followed by the same, on an instance expanded by
`expand_baseline`. The run's own record names the Program checked (its `program` is the descriptor's digest), whatever θ
is, and is refused `run-program` otherwise. θ holds registered public data only: an entry of any kind but `registered`,
an earlier run's `output` among them, is refused `theta-provenance`. One log is one deployment, and every entry cites its
first record, the first run's own included; any other record is refused `theta-not-at-deployment-start`.
"""
from __future__ import annotations

import json
from fractions import Fraction

from verity.circuits import codec
from verity.circuits.defs import PrimitiveDefinition, DefinitionTable
from verity.circuits.program import input_id, is_input, is_registered
from verity.circuits.types import type_from_json

FIELDS = ("program", "theta", "owner", "gamma", "alpha")


class Malformed(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _nat(x) -> bool:
    return type(x) is int and x >= 0


def _unevaluated(*_a):
    raise NotImplementedError("the NCI check evaluates no gate")


def declared(desc: dict) -> DefinitionTable:
    """A registry of the descriptor's primitives as it declares them (inputs aside, which the decoder makes itself)."""
    reg = DefinitionTable()
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
    from verity.circuits.parts import n_members
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


def desc_program(text: str):
    """The Program `NciCheck.Desc` reads, with its refusals in its order: `program` where it doesn't decode,
    `not-topological`, and `program` past `walkBound`'s limit or for a scan of no iterations with output leaves."""
    try:
        desc = json.loads(text)
        program = codec.decode_program(desc, declared(desc))
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise Malformed("program") from e
    from verity.circuits.partition_object import order_violation
    if order_violation(program) is not None:
        raise Malformed("not-topological")
    if walk_bound(program) > FUEL // 2:
        raise Malformed("program")
    if any(n.form == "scan" and n.n == 0 and n.out.leaves > 0
           for fn in program.defs.values() if hasattr(fn, "body") for n in fn.body.nodes):
        raise Malformed("program")
    return program


def theta_nodes(theta: list, inp: list[bool], reg: list[bool]) -> set[int]:
    """θ as root nodes, entry by entry as `NciCheck.Desc.thetaStep` reads it."""
    seen: set[int] = set()
    for t in theta:
        if not (_nat(t) and t < len(inp)):
            raise Malformed("theta-unknown")
        if not inp[t]:
            raise Malformed("theta-not-input")
        if not reg[t]:
            raise Malformed("theta-not-registered")
        if t in seen:
            raise Malformed("theta-duplicate")
        seen.add(t)
    return seen


def evaluate_desc(inst) -> dict:
    """`NciCheck.Desc.evaluate`: its refusals in its order, then `evaluate` on the expansion."""
    if not (isinstance(inst, dict) and all(k in inst for k in FIELDS) and isinstance(inst["program"], str)
            and isinstance(inst["theta"], list) and isinstance(inst["owner"], list)):
        raise Malformed("instance")
    gamma, alpha = inst["gamma"], inst["alpha"]
    if not (_nat(gamma) and isinstance(alpha, list) and len(alpha) == 2 and all(map(_nat, alpha)) and alpha[1] > 0):
        raise Malformed("params")
    desc_program(inst["program"])
    inp, reg, spans = root_nodes(inst["program"])
    K = len(inp)
    theta_nodes(inst["theta"], inp, reg)
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


# The count of (c, k)-inference blocks (`NciCheck.Count`), activation by activation

COUNT_FIELDS = ("program", "theta", "blocks", "table", "c", "k", "n")


def member_refs(node, m: int) -> list:
    """The references member `m` of a batch of a primitive reads, in operand order (`layout._prim_member_args`)."""
    out: list = []
    for a, ax in zip(node.args, node.axes):
        if ax is None:
            out.extend(a.refs)
        elif ax == 0:
            L = a.type.elem.leaves
            out.extend(a.refs[m * L + j] for j in range(L))
        else:
            out.extend(a.take(ax, m).refs)
    return out


def activation_gates(scope):
    """Each gate of the activation `scope`: its primitive, the activation whose body holds it, and the references it
    reads there."""
    from verity.circuits.parts import n_members
    for k, node in enumerate(scope.spec.body.nodes):
        if node.form == "primitive":
            yield node.fn, scope, [r for a in node.args for r in a.refs]
        elif node.form == "batch" and isinstance(node.fn, PrimitiveDefinition):
            for m in range(node.n):
                yield node.fn, scope, member_refs(node, m)
        else:
            for m in range(n_members(node)):
                yield from activation_gates(scope.child(k, m))


def source(scope, ref, top):
    """Where `ref`, read in the activation `scope` inside the activation `top`, leads: `(i, j)` for leaf `j` of `top`'s
    parameter `i`, else `None` (a gate of `top`). `layout.resolve_gate`'s walk, stopped at `top`'s parameters."""
    from verity.circuits.layout import Scope
    space, idx, leaf = ref
    while True:
        if space == "p":
            if scope is top:
                return idx, leaf
            parent = scope.parent
            node = parent.spec.body.nodes[scope.node_idx]
            if node.form == "call":
                space, idx, leaf = node.args[idx].refs[leaf]
            elif node.form == "batch":
                ax, a = node.axes[idx], node.args[idx]
                if ax is None:
                    space, idx, leaf = a.refs[leaf]
                elif ax == 0:
                    space, idx, leaf = a.refs[scope.member * a.type.elem.leaves + leaf]
                else:
                    space, idx, leaf = a.take(ax, scope.member).refs[leaf]
            elif idx == 0:
                if scope.member == 0:
                    space, idx, leaf = node.args[0].refs[leaf]
                else:
                    scope = Scope(scope.spec, scope.offset - scope.spec.gates, parent, scope.node_idx, scope.member - 1)
                    space, idx, leaf = scope.spec.body.ret[leaf]
                    continue
            elif idx <= node.n_xs:
                a = node.args[idx]
                space, idx, leaf = a.refs[scope.member * a.type.elem.leaves + leaf]
            else:
                space, idx, leaf = node.args[idx].refs[leaf]
            scope = parent
            continue
        node = scope.spec.body.nodes[idx]
        fn = node.fn
        if node.form == "primitive" or (node.form == "batch" and isinstance(fn, PrimitiveDefinition)):
            return None
        if node.form == "call":
            m, j = 0, leaf
        elif node.form == "batch":
            m, j = divmod(leaf, fn.out_leaves)
        else:
            c = node.out.fields[0].leaves
            if leaf < c:
                m, j = node.n - 1, leaf
            else:
                m, r = divmod(leaf - c, node.out.fields[1].elem.leaves)
                j = c + r
        scope = scope.child(idx, m)
        space, idx, leaf = fn.body.ret[j]


def params_of(fn) -> tuple:
    return fn.params if isinstance(fn, PrimitiveDefinition) else fn.body.params


def block_row(fn, table: dict, theta: list[bool]) -> tuple[int, int]:
    """A block Definition's Boolean gates and fresh bits, from one activation of it on its own: its gates gate by gate
    from the table (refused, `gate-table`, where the table lacks one), and the widths of the parameter leaves its gates
    read over the parameters not θ (`theta`), every leaf of a primitive's."""
    from verity.circuits.layout import Scope
    widths = [t.leaf_widths() for _, t in params_of(fn)]
    if isinstance(fn, PrimitiveDefinition):
        gates = [fn]
        used = [set(range(len(w))) for w in widths]
    else:
        top = Scope(fn, 0, None, 0, 0)
        gates, used = [], [set() for _ in widths]
        for prim, scope, refs in activation_gates(top):
            gates.append(prim)
            for r in refs:
                if (s := source(scope, r, top)) is not None:
                    used[s[0]].add(s[1])
    if any(codec.descriptor_id(p) not in table for p in gates):
        raise Malformed("gate-table")
    fresh = sum(widths[i][j] for i in range(len(widths)) if not theta[i] for j in used[i])
    return sum(table[codec.descriptor_id(p)] for p in gates), fresh


def reached(program) -> dict:
    """The Definitions the root reaches, by descriptor key."""
    out, todo = {}, [program.root]
    while todo:
        fn = todo.pop()
        key = codec.descriptor_id(fn)
        if key in out:
            continue
        out[key] = fn
        if not isinstance(fn, PrimitiveDefinition):
            todo += [n.fn for n in fn.body.nodes]
    return out


def walk_blocks(program, blocks: set[str], theta: set[int]):
    """From the root down, activation by activation, through the Definitions outside every block: per block Definition
    its blocks and, per parameter, whether it is θ at all of them; and each block's site (the activation, node and
    member). A parameter is θ at a block when every reference of its argument names θ: in the root's body a root node
    of θ, in another a parameter θ at that activation; a scan's carry never is. A gate in no block is refused
    (`gate-in-no-block`)."""
    from verity.circuits.layout import Scope
    from verity.circuits.parts import n_members
    count = dict.fromkeys(blocks, 0)
    flags: dict[str, list[bool]] = {}
    sites: list[tuple[str, object, int, int]] = []

    def names(r, at_root: bool, th) -> bool:
        space, idx, _ = r
        return space == "n" and idx in theta if at_root else space == "p" and th[idx]

    def visit(scope, th) -> None:
        at_root = scope.parent is None
        for k, node in enumerate(scope.spec.body.nodes):
            if is_input(node.fn):
                continue
            key = codec.descriptor_id(node.fn)
            args = [not (node.form == "scan" and i == 0) and all(names(r, at_root, th) for r in a.refs)
                    for i, a in enumerate(node.args)]
            for m in range(n_members(node)):
                if key in blocks:
                    count[key] += 1
                    flags[key] = [x and y for x, y in zip(flags.get(key, args), args, strict=True)]
                    sites.append((key, scope, k, m))
                elif isinstance(node.fn, PrimitiveDefinition):
                    raise Malformed("gate-in-no-block")
                else:
                    visit(scope.child(k, m), args)
    visit(Scope(program.root, 0, None, 0, 0), None)
    return count, flags, sites


def evaluate_count(inst) -> dict:
    """`NciCheck.Count.evaluate`: its refusals in its order, then the count activation by activation (`walk_blocks`,
    `block_row`)."""
    if not (isinstance(inst, dict) and all(k in inst for k in COUNT_FIELDS) and isinstance(inst["program"], str)
            and isinstance(inst["theta"], list) and isinstance(inst["blocks"], list)
            and isinstance(inst["table"], dict)):
        raise Malformed("instance")
    c, k, n, blocks, table = inst["c"], inst["k"], inst["n"], inst["blocks"], inst["table"]
    if not (_nat(c) and _nat(k) and _nat(n) and all(isinstance(b, str) for b in blocks)
            and all(map(_nat, table.values()))):
        raise Malformed("params")
    program = desc_program(inst["program"])
    inp, reg, _ = root_nodes(inst["program"])
    theta = theta_nodes(inst["theta"], inp, reg)
    defs = reached(program)
    root = codec.descriptor_id(program.root)
    for i, b in enumerate(blocks):
        if b not in defs:
            raise Malformed("block-unknown")
        if b == root:
            raise Malformed("block-root")
        if is_input(defs[b]):
            raise Malformed("block-input")
        if b in blocks[:i]:
            raise Malformed("block-duplicate")
    count, flags, _ = walk_blocks(program, set(blocks), theta)
    rows = []
    for b in blocks:
        gates, fresh = block_row(defs[b], table, flags.get(b, [False] * len(params_of(defs[b]))))
        rows.append([b, gates, fresh, count[b]])
    total = sum(r[3] for r in rows)
    failure = next(([i, "gates" if g > c else "fresh"] for i, (_, g, x, m) in enumerate(rows)
                    if m > 0 and (g > c or x > k)), None)
    if failure is None and total > n:
        failure = [None, "count"]
    return {"blocks": rows, "total": total, "complies": failure is None, "failure": failure}


def check_count_detail(inst) -> dict:
    """`NciCheck.Count.checkDetail`: the whole verdict, or the refusal, for the private record only, never published."""
    try:
        return evaluate_count(inst)
    except Malformed as e:
        return {"error": e.code}


def check_count(inst) -> dict:
    """`NciCheck.Count.check`, what the count publishes: the judgment against n, or the refusal."""
    res = check_count_detail(inst)
    return res if "error" in res else {"complies": res["complies"]}


# θ's provenance (`NciCheck.Baseline`): after the policy check, each θ entry against the trusted input

TRUSTED_KEYS = {"log", "run", "params"}
REGISTERED_KEYS = {"kind", "registration", "value", "root"}


def record_digest(rec: dict) -> str:
    """The window record's digest, `registration.digest`: SHA-512 of the record's own format (v1 or v2), a NUL and its
    canonical JSON."""
    from verity.protocols.spot.verifier.one_stage.registration import digest
    return digest(rec)


def lower(t) -> list[tuple[int, int]]:
    """§5.1: the input nodes a parameter of type `t` lowers to, (width, n) each (`Spot.Registered.lower`)."""
    from verity.circuits.types import Array, RecordType, Tuple
    ws = t.width_set()
    if len(ws) == 1:
        return [(next(iter(ws)), t.leaves)]
    if isinstance(t, Array):
        return lower(t.elem) * t.n
    if isinstance(t, Tuple):
        return [x for f in t.fields for x in lower(f)]
    if isinstance(t, RecordType):
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
    if run.get("program") != digest:
        raise Malformed("run-program")
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
        if own[k] is None:
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
