"""The vector scripts' made-up primitives (`NciOp*`, `NciSum*`, `NciConst*`, `RecOp*`, `RecConst*`) and their Boolean bodies.

These are test stand-ins. A Program must give each word primitive one Boolean Definition (PROTOCOL.md §4.9). The
inference-only check reads only a Program's wiring and widths, so a stand-in's body is one of its bit signature whose every
output bit is the constant 0: it computes nothing the primitive computes.

Within `bodies` (and `program`, one build within it), a Program takes the catalog's lookup (`versions.boolean_version`) and,
for a made-up primitive alone (`made_up`: its id is one of the scripts' names and its evaluator is no module of `verity` or an
adapter), its stand-in body. The lookup is registered only inside it and the one before put back after, as
`restrict.program` does, so it never stays registered and never reaches a served description. Any other primitive with no
catalog body is refused (`E_NO_BODY`), as it would be anywhere.

`v2` gives the same stand-in bodies as descriptor JSON, for the scripts that write descriptors by hand, whose word primitives
are all made up; it imports nothing of `verity`.
"""
from __future__ import annotations

import contextlib
import re

SCHEMA = "verity-ir/descriptor/v2"
MADE_UP = re.compile(r"(NciOp|NciSum|NciConst|RecOp|RecConst)[0-9a-z]*_v1")
CONST0 = "Const1[0x0]_v1"
INPUT = re.compile(r"(Registered)?Input[1-9][0-9]*_v1")
#: PROTOCOL.md §4.9's ids that are no word primitive: the gates, the constant bits and the inputs
NOT_WORD = re.compile(r"(And|Xor)([2-9]|[1-9][0-9]+)_v1|Not_v1|Const1\[0x[01]\]_v1|(Registered)?Input[1-9][0-9]*_v1")

_BODIES: dict[str, object] = {}


def _bits(t):
    if t[0] == "Value":
        return t if t[1] == 1 else ["Array", t[1], ["Value", 1]]
    if t[0] == "Array":
        return ["Array", t[1], _bits(t[2])]
    if t[0] == "Record":
        return ["Record", [[n, _bits(f)] for n, f in t[1]]]
    return ["Tuple", [_bits(f) for f in t[1]]]


def v2(defs: dict, callees: list, types: list, root: str) -> dict:
    """A hand-written descriptor's `schema`, `definitions`, `callees`, `types`, `boolean` and `inputs` under schema v2: its
    own tables, the definitions `root` reaches, a stand-in body for each word primitive among them, in key order, and one
    parameter per input node leading `root`'s body (PROTOCOL.md §4.1, §4.6, §4.9, §5.1)."""
    reached, todo = set(), [root]
    while todo:
        k = todo.pop()
        if k not in reached and k in defs:
            reached.add(k)
            todo += [callees[n["fn"]] for n in (defs[k].get("body") or {}).get("nodes", [])]
    defs, callees, types, boolean = {k: d for k, d in defs.items() if k in reached}, list(callees), list(types), {}

    def intern(table: list, x) -> int:
        if x not in table:
            table.append(x)
        return table.index(x)

    for key, d in sorted(dict(defs).items()):
        if d["form"] != "primitive" or NOT_WORD.fullmatch(key):
            continue
        r = d["ret"][1]
        name = f"StandIn[{key.removesuffix('_v1')}]_v1"
        defs.setdefault(CONST0, {"form": "primitive", "id": CONST0, "params": [], "ret": ["Value", 1]})
        c = intern(callees, CONST0)
        node = ({"args": [], "fn": c, "form": "primitive", "out": intern(types, ["Value", 1])} if r == 1 else
                {"args": [], "axes": [], "fn": c, "form": "batch", "n": r, "out": intern(types, _bits(d["ret"]))})
        defs[name] = {"form": "composite", "id": name, "bindings": {}, "body": {
            "params": [[n, _bits(t)] for n, t in d["params"]], "ret_type": _bits(d["ret"]), "nodes": [node],
            "ret": {"affine": ["n", 0, 0, r]}}}
        boolean[key] = name
    nodes = defs[root]["body"]["nodes"]
    lead = next((k for k, n in enumerate(nodes) if not INPUT.fullmatch(callees[n["fn"]])), len(nodes))
    return {"schema": SCHEMA, "definitions": defs, "callees": callees, "types": types, "boolean": boolean,
            "inputs": [[f"x{k}", 1] for k in range(lead)]}


def made_up(p) -> bool:
    """`p` is one of the scripts' made-up primitives."""
    module = getattr(p.evaluate, "__module__", "") or ""
    return MADE_UP.fullmatch(p.id) is not None and module.split(".")[0] not in ("verity", "verity_vllm", "verity_torch")


def body(p):
    """The stand-in Boolean body of `p`: its bit signature, every output bit the constant 0."""
    from verity.circuits import boolean as BL
    from verity.circuits.defs import CompositeDefinition, bind
    if p.id not in _BODIES:
        params, ret = p.signature()

        def build(B, S, *_args):
            return B.call(BL.constant(0)) if ret.w == 1 else B.batch(BL.constant(0), axes=(), n=ret.w)

        sig = (tuple((n, BL.bit_type(t)) for n, t in params), BL.bit_type(ret))
        _BODIES[p.id] = bind(CompositeDefinition(f"StandIn[{p.name}]", 1, (), lambda S: sig, build, register=False))
    return _BODIES[p.id]


def _lookup(p):
    from verity.circuits.ops.boolean import versions as V
    return body(p) if made_up(p) else V.boolean_version(p)


@contextlib.contextmanager
def bodies():
    """Within it, a Program built gives its word primitives the catalog's bodies and the made-up ones their stand-ins."""
    from verity.circuits import program as PG
    from verity.circuits.ops.boolean import versions as _  # noqa: F401  registers the catalog's lookup, the one put back
    saved = list(PG._BODY_LOOKUP)
    PG.register_boolean_bodies(_lookup)
    try:
        yield
    finally:
        PG._BODY_LOOKUP[:] = saved


def program(fn, **kw):
    """`Program(fn, **kw)` within `bodies`."""
    from verity.circuits.program import Program
    with bodies():
        return Program(fn, **kw)
