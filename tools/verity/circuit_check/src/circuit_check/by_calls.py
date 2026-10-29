"""circuit-check's by-calls mode, for a Program root too large to build whole: a binding file lists the root in its
``_by_calls_roots`` (`targets.BY_CALLS_ROOTS`), and only a root listed there is checked so; it is never a default.

Such a root's body is calls and wiring only, and it is checked without building any callee's body (`unlaid`: the root's own
nodes, each callee read by its signature alone):

    calls only        no node calls a primitive (a primitive's call, or a batch of one): ``by-calls/primitive-node``
    ports             its declared ports bind to its signature (`SpecializedDefinition.port_types`): ``by-calls/ports``
    wiring            each node's arguments match its callee's signature (arity, types) and give its declared output, and the
                      widths each reads are the ones it is typed at (`Builder`); its descriptor entry (`codec.definition_entry`,
                      callees by descriptor id) decodes node by node as `codec.decode_program` decodes one, each reference naming
                      a leaf of a parameter or of an earlier node: ``by-calls/wiring``
    round trip        that decoded body encodes to the same entry: ``by-calls/round-trip``
    stable            two builds of it give one entry: ``by-calls/unstable``
    its callees       each distinct callee specialization relies on one target of its family's sidecar at that sidecar's own
                      small bindings (`targets.sidecar_targets`): the one that agrees with it in every static that is not a
                      number, differing in the fewest numbers, then of the fewest gates; none: ``by-calls/unbound``, which
                      names a binding that agrees but `targets.select` drops (over `targets.MAX_GATES` and not its family's
                      smallest there).  A static-free callee is its own target, a catalog root.

The entry says ``"mode": "by-calls"`` and has no gate count.  Its ``by_calls`` lists each callee with the target it relies on,
and ``entry_sha512``, the digest of the root's own entry, which is not the whole Program's `codec.program_digest`: that needs
every callee's body.  The report (`checks.report`, `joined`) adds each relied-on target's verdict to the root's: a target the
report doesn't hold is ``by-calls/callee-unchecked``, one that failed ``by-calls/callee-failed``.  A named run checks those
targets beside the root (`cli._tasks`); ``--all`` holds them when its suite does.

The rest of circuit-check never builds such a root's body: the catalog doesn't descend into it (`targets.reachable`),
`targets.select` keeps it whatever its size, its key is its own entry (`cache.descriptor`, `entry`), and its catalog record
says so (`units`)."""
from __future__ import annotations

import hashlib
from typing import Any

from verity.ml.circuits import codec as C
from verity.ml.circuits import ports as P
from verity.ml.circuits.defs import Builder, Node, PrimitiveDefinition, SpecializedDefinition
from verity.ml.circuits.refs import Affine, Coll

from . import cache as CC
from . import sidecars as SC
from . import targets as TG

CHECK = "by-calls"


class _Unlaid(Builder):
    """A `Builder` that lays out no gates: a node's interval is its callee's gate count, which builds the callee's body."""

    def _emit(self, node: Node) -> Coll:
        self.nodes.append(node)
        self.offsets.append(0)
        return Coll(node.out, Affine("n", len(self.nodes) - 1, 0, node.out.leaves))


def unlaid(fn: SpecializedDefinition) -> Any:
    """``fn``'s body as its builder writes it, each callee read by its signature alone (`_Unlaid`): its nodes and wiring, no
    gate count."""
    params, ret = fn.signature()
    B = _Unlaid(params, ret)
    args = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    return B.finish(fn.defn.body(B, fn.S(), *args))


def _shell(fn: SpecializedDefinition, body: Any) -> SpecializedDefinition:
    """A stand-in for ``fn`` holding ``body``, so ``fn`` itself never holds a body without gates."""
    s = SpecializedDefinition(fn.defn, fn.bindings, fn.key)
    s.set_body(body)
    return s


_ENTRIES: dict[int, tuple[Any, dict]] = {}


def entry(fn: SpecializedDefinition) -> dict:
    """``fn``'s own descriptor entry (`codec.definition_entry` of its `unlaid` body), made once a process.  What made it is held
    as ``fn``'s build (`cache.attributed` reads it as a built body's)."""
    got = _ENTRIES.get(id(fn))
    if got is None or got[0] is not fn:
        with CC.recording(merge=False) as fr:
            e = C.definition_entry(_shell(fn, unlaid(fn)))
        fn._circuit_check_built = fr
        got = _ENTRIES[id(fn)] = (fn, e)
    return got[1]


def _digest(e: dict) -> str:
    return hashlib.sha512(C.canonical_json(e)).hexdigest()


def _decoded(fn: SpecializedDefinition, e: dict, callees: dict[str, Any]) -> tuple[Any, Any]:
    """``fn``'s entry ``e`` decoded as `codec.decode_program` decodes one definition, each callee the one ``callees`` holds under
    its descriptor id, a node reading only parameters and earlier nodes: (its body, its ports); ValueError names what it refuses."""
    fid = C.descriptor_id(fn)
    if (why := C.jcs_domain_error(e)) is not None:
        raise ValueError(f"{fid}: {why}")
    statics = tuple(fn.defn.statics)
    if e.get("form") != "composite" or e.get("id") != fn.base_id or tuple(e.get("bindings", {})) != statics:
        raise ValueError(f"{fid}: its entry is {e.get('form')!r} {e.get('id')!r} with statics {list(e.get('bindings', {}))}")
    key = e["id"] + ("{" + ",".join(f"{k}={C.canonical_json(e['bindings'][k]).decode()}" for k in statics) + "}" if statics else "")
    if key != fid:
        raise ValueError(f"{fid}: its entry's id and binding record are {key!r}")
    b = e["body"]
    params = tuple((n, C._type(t)) for n, t in b["params"])
    ret_type = C._type(b["ret_type"])
    if (params, ret_type) != tuple(fn.signature()):
        raise ValueError(f"{fid}: its entry's signature is not its own")
    B = _Unlaid(params, ret_type)
    outs = [C._type(nd.get("out")) for nd in b["nodes"]]

    def leaves(below: int):
        def at(space: str, idx: int) -> int | None:
            if space == "p":
                return params[idx][1].leaves if idx < len(params) else None
            return outs[idx].leaves if idx < below else None
        return at

    for nd in b["nodes"]:
        k = len(B.nodes)
        c = callees.get(nd.get("fn")) if isinstance(nd.get("fn"), str) else None
        if c is None:
            raise ValueError(f"{fid} node {k}: calls {nd.get('fn')!r}, which its body calls nowhere")
        C._check_node(fid, k, nd, c)
        try:
            args = tuple(Coll(C._type(a["type"]), C._decode_refs(a["refs"], B.nodes, C.SCHEMA_V0, leaves(k))) for a in nd["args"])
        except (ValueError, TypeError, KeyError, IndexError) as ex:
            raise ValueError(f"{fid} node {k}: {ex}") from ex
        C._build_node(B, fid, k, nd, c, args, outs[k])
    try:
        body = B.finish(Coll(ret_type, C._decode_refs(b["ret"], B.nodes, C.SCHEMA_V0, leaves(len(outs)))))
    except (ValueError, TypeError, KeyError, IndexError) as ex:
        raise ValueError(f"{fid} ret: {ex}") from ex
    return body, (P.PortTypes.from_json(fid, e["ports"], params, ret_type) if "ports" in e else None)


# --- the targets its callees rely on ------------------------------------------------------------------------------------------

_ABSENT = object()


def _leaves(v: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(v, dict):
        for k in sorted(v):
            yield from _leaves(v[k], (*path, str(k)))
    else:
        yield path, v


def _number(v: Any) -> bool:
    if isinstance(v, (list, tuple)):
        return bool(v) and all(map(_number, v))
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def differs(a: dict, b: dict) -> tuple[list[str], list[str]]:
    """The statics in which binding records ``a`` and ``b`` differ, a dict-valued one key by key (``ROUTE.SPLIT_K``):
    (switches, sizes), a size being a number, or a tuple of numbers, on both sides."""
    la, lb = dict(_leaves(a)), dict(_leaves(b))
    switches, sizes = [], []
    for p in sorted(la.keys() | lb.keys()):
        x, y = la.get(p, _ABSENT), lb.get(p, _ABSENT)
        if x is not _ABSENT and y is not _ABSENT and C.canonical_json(C._static_json(x)) == C.canonical_json(C._static_json(y)):
            continue
        (sizes if _number(x) and _number(y) else switches).append(".".join(p))
    return switches, sizes


def _family_targets(c: Any) -> tuple[str | None, dict[str, Any]]:
    """The binding file of the module that defines ``c``'s family (`sidecars.owner`), relative to the repository, and its targets
    (`targets.sidecar_targets`); (None, {}) when that module has none."""
    p = SC.discover().get(SC.owner(c) or "", {}).get("py")
    if p is None:
        return None, {}
    p.read_bytes()      # read whole, so a check that relies on one of its targets is keyed by all of it (`cache.attributed`)
    return SC.rel(p), TG.sidecar_targets(str(p))[0]


def _family_dropped(c: Any) -> dict[str, Any]:
    """What the binding file of ``c``'s module reaches of that module that its targets don't keep (`targets.sidecar_dropped`)."""
    p = SC.discover().get(SC.owner(c) or "", {}).get("py")
    return {} if p is None else TG.sidecar_dropped(str(p))


def relied(c: Any) -> dict:
    """The target callee ``c`` relies on (the module's docstring): ``binding`` (its id, or None), ``sidecar``, ``sizes`` (the
    statics in which it differs from ``c``), or ``why`` none is."""
    if not c.defn.statics:
        return {"binding": c.id, "sidecar": None, "sizes": []}
    sidecar, found = _family_targets(c)
    if sidecar is None:
        return {"binding": None, "why": f"its module {SC.owner(c)} has no circuit-check sidecar"}
    fam, best, near = TG.family(c), None, None
    for f in found.values():
        if not isinstance(f, SpecializedDefinition) or TG.family(f) != fam or TG.is_by_calls(f):
            continue
        switches, sizes = differs(c.bindings, f.bindings)
        if switches:
            near = min(near or (len(switches), f.id, switches), (len(switches), f.id, switches))
            continue
        if best is None or (len(sizes), f.gates, f.id) < best[0]:
            best = ((len(sizes), f.gates, f.id), f.id, sizes)
    if best is None:
        why = f"{sidecar} binds no target of {fam}" + (f" that agrees with it in {', '.join(near[2])} (nearest: {near[1]})"
                                                        if near else "")
        dropped = sorted(f.id for f in _family_dropped(c).values()
                         if isinstance(f, SpecializedDefinition) and TG.family(f) == fam and not differs(c.bindings, f.bindings)[0])
        if dropped:
            why += (f"; it binds {', '.join(dropped[:3])}, which `targets.select` drops: over MAX_GATES ({TG.MAX_GATES}) and "
                    f"not the smallest {fam} it binds. Bind one of at most MAX_GATES gates")
        return {"binding": None, "sidecar": sidecar, "why": why}
    return {"binding": best[1], "sidecar": sidecar, "sizes": best[2]}


def callees(body: Any) -> list[tuple[Any, int]]:
    """Each distinct callee of ``body``, in the order its nodes first call it, with how many of its nodes call it."""
    seen: dict[str, list] = {}
    for n in body.nodes:
        seen.setdefault(n.fn.id, [n.fn, 0])[1] += 1
    return [(fn, k) for fn, k in seen.values()]


# --- the check --------------------------------------------------------------------------------------------------------------

def check(fn: SpecializedDefinition) -> dict:
    """``fn``'s entry in the by-calls mode (the module's docstring), with its failures; its callees' verdicts are the report's
    (`joined`)."""
    from . import checks as CK

    F = CK.Findings()
    out: dict[str, Any] = {"definition": fn.id, "form": "composite", "mode": CHECK, "gates": None}
    with CK.spent(CHECK):
        try:
            body = unlaid(fn)
        except Exception as ex:  # noqa: BLE001  (a body that doesn't build from its callees' signatures fails by name)
            F.fail(CHECK, "wiring", fn.id, f"its body doesn't build from its callees' signatures: {type(ex).__name__}: {str(ex)[:300]}")
            out["failures"], out["warnings"] = F.failures, F.warnings
            return out
        prims = sorted({n.fn.id for n in body.nodes if isinstance(n.fn, PrimitiveDefinition)})
        if prims:
            F.fail(CHECK, "primitive-node", fn.id, f"its body calls {len(prims)} primitive(s) itself ({', '.join(prims[:4])}): a "
                   "by-calls root is calls and wiring only")
        record: dict[str, Any] = {"nodes": len(body.nodes)}
        try:
            _ = fn.port_types
            own = entry(fn)
        except Exception as ex:  # noqa: BLE001  (ports that don't bind fail by name)
            F.fail(CHECK, "ports", fn.id, f"{type(ex).__name__}: {str(ex)[:300]}")
            own = None
        if own is not None:
            record["entry_sha512"] = _digest(own)
            record["entry_of"] = ("the root's own descriptor entry, its callees by descriptor id: not the whole Program's "
                                  "program_digest, which needs every callee's body")
            try:
                back, ports = _decoded(fn, own, {C.descriptor_id(c): c for c, _k in callees(body)})
            except (ValueError, TypeError, KeyError, IndexError) as ex:
                F.fail(CHECK, "wiring", fn.id, f"its entry doesn't decode: {str(ex)[:400]}")
            else:
                again = _shell(fn, back)
                again.set_port_types(ports)
                if C.canonical_json(C.definition_entry(again)) != C.canonical_json(own):
                    F.fail(CHECK, "round-trip", fn.id, "its entry decodes to a body that encodes to another entry")
            if _digest(C.definition_entry(_shell(fn, unlaid(fn)))) != record["entry_sha512"]:
                F.fail(CHECK, "unstable", fn.id, "two builds of it give two entries")
        rows = []
        for c, k in callees(body):
            row = {"callee": c.id, "family": TG.family(c), "nodes": k}
            if not isinstance(c, PrimitiveDefinition):
                row.update(relied(c))
                if row["binding"] is None:
                    F.fail(CHECK, "unbound", fn.id, f"{c.id}: {row['why']}", callee=c.id)
            rows.append(row)
        record["callees"] = rows
    out["by_calls"] = record
    out["failures"], out["warnings"] = F.failures, F.warnings
    return out


def joined(entries: list[dict]) -> list[dict]:
    """``entries`` with each by-calls root's relied-on targets' verdicts joined to its own: a target none of ``entries`` is
    fails it with ``by-calls/callee-unchecked``, one that failed with ``by-calls/callee-failed`` (naming its failures); each
    callee's row says ``verdict``.  Entries are copied, never changed."""
    by: dict[str, dict] = {}
    for e in entries:
        for k in (e.get("definition"), e.get("target")):
            if isinstance(k, str):
                by.setdefault(k, e)
    out = []
    for e in entries:
        record = e.get("by_calls")
        if e.get("mode") != CHECK or not isinstance(record, dict):
            out.append(e)
            continue
        root, fails, rows = e.get("definition") or e.get("target"), list(e.get("failures", [])), []
        for row in record.get("callees", []):
            b = row.get("binding")
            got = by.get(b) if b is not None else None
            if b is None:
                rows.append(row)
                continue
            if got is None:
                verdict, what = "unchecked", f"{row['callee']}: its target {b} is not among this report's targets"
            elif got.get("failures"):
                names = [f.get("name", "") for f in got["failures"]]
                verdict, what = "failed", f"{row['callee']}: its target {b} fails: {'; '.join(names[:3])}"
            else:
                verdict, what = "passed", None
            if what is not None:
                fails.append({"check": CHECK, "code": f"callee-{verdict}", "definition": root, "callee": row["callee"],
                              "name": f"{CHECK}/callee-{verdict} {root} {what}"})
            rows.append({**row, "verdict": verdict})
        out.append({**e, "failures": fails, "by_calls": {**record, "callees": rows}})
    return out


def relied_targets(fn: SpecializedDefinition) -> list[Any]:
    """The targets a by-calls root's callees rely on (`relied`), made: what a named run of it checks beside it."""
    out: dict[str, Any] = {}
    for c, _k in callees(unlaid(fn)):
        if isinstance(c, PrimitiveDefinition):
            continue
        if not c.defn.statics:
            out.setdefault(c.id, c)
            continue
        b = relied(c).get("binding")
        if b is not None:
            out.setdefault(b, _family_targets(c)[1][b])
    return list(out.values())


__all__ = ["CHECK", "callees", "check", "differs", "entry", "joined", "relied", "relied_targets", "unlaid"]
