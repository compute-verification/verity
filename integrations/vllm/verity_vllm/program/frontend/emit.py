"""The frontend's emitter of runtime correspondence (`verity_vllm.correspondence.runtime`, schema `runtime-correspondence/v1`).

Runs once, after the flat body of a derivation is finished, over the derivation context: every Call the lowering rules emitted is
mapped to the fx node it was lowered from (the source / runtime occurrence), and the Call's return Value and argument Values are
located inside that occurrence's output / input tensors as leaf ranges.  Everything here is read off the exported graph's metadata
(`nn_module_stack`, `meta["val"]` fake tensors), the context's alias / mutation bookkeeping and the finished body's reference
sequences -- nothing is asserted by rule name, and no model or family name appears.  Rules stay untouched: a rule that emits
through `Ctx.call` is covered by the explicit fx node it passes; a rule that emits through the Builder directly is covered by the
node the translation loop was at when the Builder grew (`Ctx.note_emitted`).

What the frontend can NOT populate here (and where it would come from) is not guessed at: kernel
launch identity / grid (the runtime observation), physical addresses (the acquisition), the peer rank's own Call ids (the peer
Program's correspondence).
"""

from __future__ import annotations

import bisect
import re
from typing import Any

import torch

from verity.ml.circuits.refs import Affine, Coll, Concat, Strided, prefix_runs
from verity.ml.circuits.types import Record
from verity_vllm.program.frontend.value_types import flat_types
from verity_vllm.correspondence.runtime import NAMESPACE, SCHEMA

_ROW_RE = re.compile(r"\[(\d+)\]")
_COLLECTIVE_KINDS = {"all_reduce2": "all_reduce", "all_gather2": "all_gather",    # op name -> collective kind (operands in rank order);
                     "all_reduce": "all_reduce", "all_gather": "all_gather"}      # the unsuffixed ops take the world's `parts` as one list
MAX_RUNS = 1 << 16   # a reference sequence with more contiguous runs than this is not indexed (a leaf-strided view of a cache-sized
                     # tensor has one run per leaf); its slots stay unlocated and say so -- the emitter is O(runs), never O(leaves)
MAX_SPANS = 1 << 9   # an argument read from several fx inputs is resolved run by run only up to this many runs; past it, one whose
                     # every run an input holds stays unlocated and says so.  An attention row whose keys alternate two inputs
                     # (`k_nope ++ k_pe` per key) is 2 x keys runs, so spans for every row of every head would be O(rows^2) records a layer


def _runs(refs, max_runs: int = MAX_RUNS) -> list[tuple[str, int, int, int]] | None:
    """`verity.ml.circuits.refs.runs` with a budget: the maximal contiguous runs `(space, idx, lo, hi)` of `refs` in sequence order, or None
    once more than `max_runs` runs would be produced (the walk stops there instead of enumerating leaves)."""
    out: list[tuple[str, int, int, int]] = []

    def push(space, idx, lo, hi) -> bool:
        if out and out[-1][0] == space and out[-1][1] == idx and out[-1][3] == lo:
            out[-1] = (space, idx, out[-1][2], hi)
            return True
        if len(out) >= max_runs:
            return False
        out.append((space, idx, lo, hi))
        return True

    def walk(r) -> bool:
        if isinstance(r, Affine):
            return push(r.space, r.idx, r.base, r.base + r.count) if r.count else True
        if isinstance(r, Strided):
            if r.inner_stride == 1 or r.inner == 1:
                rows = r.count // r.inner
                if rows > max_runs:                     # every row is its own run at best
                    return False
                for i in range(rows):
                    b = r.base + i * r.outer_stride
                    if not push(r.space, r.idx, b, b + r.inner):
                        return False
                return True
            if r.count > max_runs:                      # one run per leaf
                return False
            for space, idx, leaf in r:
                if not push(space, idx, leaf, leaf + 1):
                    return False
            return True
        if isinstance(r, Concat):
            return all(walk(p) for p in r.parts)
        if len(r) > max_runs:
            return False
        for space, idx, leaf in r:
            if not push(space, idx, leaf, leaf + 1):
                return False
        return True

    return out if walk(refs) else None


# ---------------------------------------------------------------------------------------------------------------------------
# tensor views
# ---------------------------------------------------------------------------------------------------------------------------


def _view(val: Any) -> dict | None:
    if not isinstance(val, torch.Tensor):
        return None
    d: dict[str, Any] = {"shape": [int(s) for s in val.shape], "dtype": str(val.dtype).removeprefix("torch.")}
    try:
        d["strides"] = [int(s) for s in val.stride()]
        d["contiguous"] = bool(val.is_contiguous())
        d["storage_offset"] = int(val.storage_offset())
    except Exception:  # noqa: BLE001  (a fake tensor without stride metadata)
        d["strides"], d["contiguous"], d["storage_offset"] = None, None, None
    return d


def _output_views(node) -> list[dict | None]:
    val = node.meta.get("val")
    if isinstance(val, (list, tuple)):
        return [_view(v) for v in val]
    return [_view(val)]


# ---------------------------------------------------------------------------------------------------------------------------
# reference-sequence indexing: where a run of leaves sits inside a collection's flat leaf order
# ---------------------------------------------------------------------------------------------------------------------------


def _interval_index(refs) -> dict[tuple[str, int], list[tuple[int, int, int]]] | None:
    """(space, idx) -> [(lo, hi, offset)] sorted by lo: the runs of `refs` with the flat leaf offset each run starts at; None when the
    sequence has more than MAX_RUNS runs (not indexed)."""
    rs = _runs(refs)
    if rs is None:
        return None
    idx: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
    off = 0
    for space, i, lo, hi in rs:
        idx.setdefault((space, i), []).append((lo, hi, off))
        off += hi - lo
    for ivs in idx.values():
        ivs.sort()
    return idx


def _locate(index: dict | None, space: str, i: int, lo: int, hi: int) -> tuple[int, int] | None:
    """The flat leaf range [a, b) of `index`'s collection that holds leaves [lo, hi) of (space, i), if one run holds them all."""
    if not index:
        return None
    ivs = index.get((space, i))
    if not ivs:
        return None
    p = bisect.bisect_right(ivs, (lo, float("inf"), float("inf")))       # candidates: runs starting at or before `lo`
    for j in range(p - 1, -1, -1):
        rlo, rhi, off = ivs[j]
        if rlo <= lo and hi <= rhi:
            return off + (lo - rlo), off + (hi - rlo)
    return None


def _arg_runs(refs) -> tuple[list[tuple[str, int, int, int]] | None, Any, int]:
    """`(_runs(refs), log, s)`: for a `Concat` of `Affine`s cut from a `PartLog`, the first `s` runs are the log's shared runs
    (`prefix_runs`); otherwise `log` is None."""
    if isinstance(refs, Concat) and refs.prefix is not None:
        log, j = refs.prefix
        if log.affine_parts(j) and all(isinstance(p, Affine) for p in refs.parts[j:]):
            rs, log, s = prefix_runs(refs)
            return (rs, log, s) if len(rs) <= MAX_RUNS else (None, None, 0)
    return _runs(refs), None, 0


def _locate_shared(index: dict, rs: list[tuple[str, int, int, int]], log, s: int, memo: dict) -> tuple[int, int] | None | bool:
    """`_locate_runs(index, rs)` when the first `s` runs of `rs` are `log`'s shared runs: their spans are located once per
    (index, log) and kept in `memo`, so a longer cut of the same log locates only its own runs."""
    st = memo.get((id(index), id(log)))
    if st is None or st[0] is not log or st[1] is not index:
        st = memo[(id(index), id(log))] = [log, index, [], None, None]   # log, index, spans, first unlocated run, first gap
    spans = st[2]
    if st[3] is None and len(spans) < s:
        for space, i, lo, hi in log.shared_runs(len(spans), s):
            r = _locate(index, space, i, lo, hi)
            if r is None:
                st[3] = len(spans)
                break
            if spans and st[4] is None and r[0] != spans[-1][1]:
                st[4] = len(spans)
            spans.append(r)
    if st[3] is not None and st[3] < s:
        return False
    rest: list[tuple[int, int]] = []
    for space, i, lo, hi in rs[s:]:
        r = _locate(index, space, i, lo, hi)
        if r is None:
            return False
        rest.append(r)
    if not s and not rest:
        return False
    if st[4] is not None and st[4] < s:
        return None
    a, b = spans[0] if s else rest[0]
    if s:
        b = spans[s - 1][1]
    for lo, hi in (rest if s else rest[1:]):
        if lo != b:
            return None
        b = hi
    return a, b


def _all_located(inputs: list, rs: list[tuple[str, int, int, int]], log, s: int, memo: dict) -> bool:
    """Every run of `rs` lies in one of the occurrence's `inputs` (what resolving it run by run needs).  The first `s` runs, `log`'s
    shared runs, are checked once per (inputs, log) and kept in `memo`, so a longer cut of the same log checks only its own runs."""
    def held(space, i, lo, hi) -> bool:
        return any(ix and _locate(ix, space, i, lo, hi) is not None for _a, _v, ix in inputs)

    if log is None or not s:
        return all(held(*r) for r in rs)
    st = memo.get((id(inputs), id(log)))
    if st is None or st[0] is not log or st[1] is not inputs:
        st = memo[(id(inputs), id(log))] = [log, inputs, 0, None]   # log, inputs, shared runs checked, the first one not held
    if st[3] is None and st[2] < s:
        for r in log.shared_runs(st[2], s):
            if not held(*r):
                st[3] = st[2]
                break
            st[2] += 1
    return (st[3] is None or st[3] >= s) and all(held(*r) for r in rs[s:])


def _locate_runs(index: dict, rs: list[tuple[str, int, int, int]]) -> tuple[int, int] | None | bool:
    """Every run of `rs` inside one collection: its contiguous flat range, `None` if covered but not contiguous, False if not covered."""
    spans: list[tuple[int, int]] = []
    for space, i, lo, hi in rs:
        r = _locate(index, space, i, lo, hi)
        if r is None:
            return False
        spans.append(r)
    if not spans:
        return False
    a, b = spans[0]
    for lo, hi in spans[1:]:
        if lo != b:
            return None
        b = hi
    return a, b


# ---------------------------------------------------------------------------------------------------------------------------
# naming hints -> member / row (the frontend's own convention, parsed where it was produced)
# ---------------------------------------------------------------------------------------------------------------------------


def _member_row(name: str | None, base: str, node_name: str) -> tuple[str | None, int | None, str | None]:
    """Call name -> (member, row, unparsed hint).  The frontend names a Call `<hint>` or `<hint>[row]` or `<hint>[row]/<member>`, where
    `<hint>` is `hint_name(fx)` (`<module>/<fx>`) or, for a mid-module site, `<module>.<fx>/<fx>`; the row-less forms drop `[row]`.
    Anything with a bracket that is not `[<int>]` (`[t=7]`) is returned raw as `hint` and NOT parsed."""
    if not name:
        return None, None, None
    m = re.search(re.escape(node_name) + r"\[(\d+)\]", name)      # the LAST `<fx>[row]` -- a site name repeats `<fx>` before it
    if m:
        for m2 in re.finditer(re.escape(node_name) + r"\[(\d+)\]", name):
            m = m2
        row: int | None = int(m.group(1))
        rest = name[m.end():]
    elif name.startswith(base):
        row, rest = None, name[len(base):]
    else:
        return None, None, name
    if "[" in rest or "]" in rest:
        return None, row, name
    parts = [p for p in rest.split("/") if p]
    return ("/".join(parts) or None), row, None


# ---------------------------------------------------------------------------------------------------------------------------
# the emitter
# ---------------------------------------------------------------------------------------------------------------------------


def _rank_coordinate(rank: tuple[int, int] | None) -> dict[str, int] | None:
    """The document's TP rank coordinate."""
    if rank is None:
        return None
    return {"world": int(rank[0]), "rank": int(rank[1])}


def _collective_record(ctx, node, kind: str, occurrence: int) -> dict:
    """The collective of `node` (`verity_tp::<op>`): its operands in rank order, the one this rank computed (`own_operand`: not a
    Program input) and its peers' inputs.  The two-rank ops name one peer (`peer_operand` / `peer_input`); the world's `parts` list
    names every peer in rank order (`peer_operands` / `peer_inputs`), and its `peer_operand` / `peer_input` are None."""
    if isinstance(node.args[0], (list, tuple)):
        operands = [a for a in node.args[0] if isinstance(a, torch.fx.Node)]
    else:
        operands = [a for a in node.args if isinstance(a, torch.fx.Node)][:2]
    own = [i for i, a in enumerate(operands) if ctx.is_input_storage(a) is None]
    own_i = own[0] if len(own) == 1 else None
    rec = {"kind": kind, "occurrence": occurrence, "operand_ranks": list(range(len(operands))), "own_operand": own_i}
    if not isinstance(node.args[0], (list, tuple)):
        peer_i = (1 - own_i) if own_i is not None and len(operands) == 2 else None
        return rec | {"peer_operand": peer_i, "peer_input": ctx.is_input_storage(operands[peer_i]) if peer_i is not None else None}
    peers = [i for i in range(len(operands)) if i != own_i] if own_i is not None else []
    return rec | {"peer_operand": None, "peer_input": None, "peer_operands": peers,
                  "peer_inputs": [ctx.is_input_storage(operands[i]) for i in peers]}


def emit_correspondence(ctx, body, *, rank: tuple[int, int] | None = None) -> dict:
    """The `runtime-correspondence/v1` document of the finished flat `body` derived under `ctx` (see module docstring)."""
    params = list(body.params)
    n_user = len(ctx.user_inputs)
    input_names = [n for n, _ in params[:n_user]]
    weights_index = len(params) - 1 if ctx.weights_coll is not None else None
    field_names: list[str] = []
    field_starts: list[int] = []
    if weights_index is not None and isinstance(params[weights_index][1], Record):
        off = 0
        for fname, ft in params[weights_index][1].fields:
            field_names.append(fname)
            field_starts.append(off)
            off += ft.leaves
    from verity_vllm.program.frontend.derive import _mutates, fx_arg_nodes  # local: derive imports this module

    def field_of(leaf: int) -> str | None:
        j = bisect.bisect_right(field_starts, leaf) - 1
        return field_names[j] if 0 <= j < len(field_names) else None

    # (which Calls / inputs / weight fields an argument's leaves come from is the Program's own data -- the body's reference sequences --
    #  and is NOT repeated here: the record only locates the argument inside the occurrence's input tensors)

    def source_of(fx_arg, v) -> tuple[str, str | None]:
        """Source class of an fx input from its binding: read off the FIRST and LAST reference only (O(1); a cache-sized tensor is
        never walked here) -- `call` (another lowering's output), `input`, `weight` (+ the field), `constant`, `other` (mixed)."""
        if isinstance(v, Coll):
            n = len(v.refs)
            if n == 0:
                return "other", None
            (s0, i0, l0), (s1, i1, _l1) = v.refs[0], v.refs[n - 1]
            if s0 == "n" and s1 == "n":
                return "call", None
            if s0 == "p" and s1 == "p" and i0 == i1 and i0 < n_user:
                return "input", input_names[i0]
            if s0 == "p" and s1 == "p" and i0 == i1 and i0 == weights_index:
                return "weight", field_of(l0)
            return "other", None
        if type(v).__name__ in ("_LiftedConstant", "StaticIndex"):
            return "constant", getattr(v, "name", None)
        return "other", None

    # -- occurrences: one per fx node that some Call was lowered from, in graph order ------------------------------------------
    node_fx: dict[int, Any] = dict(getattr(ctx, "node_fx", {}))
    by_node: dict[Any, list[int]] = {}
    for k in range(len(body.nodes)):
        fx = node_fx.get(k)
        if fx is not None:
            by_node.setdefault(fx, []).append(k)
    order = {n: i for i, n in enumerate(ctx.ep.graph.nodes)}
    fx_nodes = sorted(by_node, key=lambda n: order.get(n, len(order)))
    rule_of = {a["node"]: a["rule"] for a in ctx.applications}
    collectives = 0
    occurrences: list[dict] = []
    occ_index: dict[Any, int] = {}
    occ_inputs: list[list[tuple[Any, dict]]] = []      # per occurrence: (fx arg node, interval index of its value AS READ)
    occ_outputs: list[dict] = []                        # per occurrence: interval index over its produced / written collections, tagged by port
    bindings: dict[Any, dict] = getattr(ctx, "occurrence_bindings", {})
    index_memo: dict[int, tuple[Any, dict | None]] = {}   # id(refs) -> (refs kept alive, its interval index): a cache read by many
                                                          # occurrences is indexed once

    def indexed(refs) -> dict | None:
        hit = index_memo.get(id(refs))
        if hit is None or hit[0] is not refs:
            hit = index_memo[id(refs)] = (refs, _interval_index(refs))
        return hit[1]

    for node in fx_nodes:
        stack = node.meta.get("nn_module_stack") or {}
        module = ctx.stack_path(node) if stack else ""
        mcls = ctx.module_class(node) if stack else None
        t = node.target
        op = str(t) if not isinstance(t, torch._ops.OpOverload) else f"{t.namespace}.{t.__name__}"
        frozen = bindings.get(node)                     # bindings frozen at lowering; the end-of-translation env is the fallback
        # fx inputs: the fx-node arguments (flattened: positional, then keyword), each with its source class and view as READ
        args_flat = fx_arg_nodes(node)
        n_positional = sum(1 for a in node.args if isinstance(a, torch.fx.Node)) + sum(
            sum(1 for x in a if isinstance(x, torch.fx.Node)) for a in node.args if isinstance(a, (list, tuple)))
        kw_names: list[str] = []
        for kname, a in node.kwargs.items():
            if isinstance(a, torch.fx.Node):
                kw_names.append(kname)
            elif isinstance(a, (list, tuple)):
                kw_names.extend(kname for x in a if isinstance(x, torch.fx.Node))
        inputs: list[dict] = []
        idxs: list[tuple[Any, Any, dict | None]] = []
        for j, a in args_flat:
            v = frozen["in"].get(j) if frozen is not None else ctx.env.get(a)
            src, nm = source_of(a, v)
            index = indexed(v.refs) if isinstance(v, Coll) else {}
            inp = {"node": a.name, "arg": j if j < n_positional else None, "kwarg": kw_names[j - n_positional] if j >= n_positional else None,
                   "source": src, "name": nm, "view": _view(a.meta.get("val"))}
            if index is None:
                inp["unindexed"] = f"runs>{MAX_RUNS}"
            inputs.append(inp)
            idxs.append((a, v, index))
        # outputs: the node's value(s) as PRODUCED (fx output ports), then the arguments an in-place lowering WROTE (written ports)
        val = frozen["out"] if frozen is not None else ctx.env.get(node)
        outs = val if isinstance(val, (list, tuple)) else [val]
        out_index: dict[tuple[str, int], list[tuple[int, int, int, Any]]] = {}
        unindexed: list[Any] = []
        for port, c in enumerate(outs):
            if isinstance(c, Coll):
                ix = indexed(c.refs)
                if ix is None:
                    unindexed.append(port)
                    continue
                for key, ivs in ix.items():
                    out_index.setdefault(key, []).extend((lo, hi, off, port) for lo, hi, off in ivs)
        written: list[dict] = []
        for j, c in sorted((frozen or {}).get("written", {}).items()):
            a = dict(args_flat)[j]
            wd = {"arg": j, "node": a.name, "view": _view(a.meta.get("val"))}
            if isinstance(c, Coll):
                ix = indexed(c.refs)
                if ix is None:
                    wd["unindexed"] = f"runs>{MAX_RUNS}"
                else:
                    for key, ivs in ix.items():
                        out_index.setdefault(key, []).extend((lo, hi, off, ("arg", j)) for lo, hi, off in ivs)
            written.append(wd)
        base = ctx.storage_base(node)
        w = ctx.is_weight_storage(node)
        lifetime = {"in_place": bool(_mutates(t)), "view_of": base.name if base is not node and isinstance(base, torch.fx.Node) else None,
                    "aliases_input": ctx.is_input_storage(node), "aliases_weight": w.name if w is not None else None,
                    "rebound": node in ctx.written_allocs, "stale": node in ctx.stale}
        collective = None
        if isinstance(t, torch._ops.OpOverload) and t.namespace == "verity_tp" and t.__name__.split(".")[0] in _COLLECTIVE_KINDS:
            nm = t.__name__.split(".")[0]
            if nm.endswith("2") and rank is not None and int(rank[0]) != 2:
                raise NotImplementedError(f"tensor-parallel world {int(rank[0])} refused: verity_tp::{nm} is the two-rank op "
                                          f"(two operands, the peer is the other one); a world-N rank calls verity_tp::{nm[:-1]}")
            collective = _collective_record(ctx, node, _COLLECTIVE_KINDS[nm], collectives)
            collectives += 1
        occ_index[node] = len(occurrences)
        occ = {"node": node.name, "module": module, "module_class": mcls, "op": op, "rule": rule_of.get(node.name),
               "outputs": _output_views(node), "written": written, "inputs": inputs, "lifetime": lifetime, "collective": collective}
        if unindexed:
            occ["unindexed_outputs"] = unindexed
        occurrences.append(occ)
        occ_inputs.append(idxs)
        occ_outputs.append(out_index)

    # -- calls: Call k -> occurrence + return slots + argument slots --------------------------------------------------------------
    calls, located = [], {}   # located: (id(index), id(PartLog)) -> the log's shared runs located in that index (`_locate_shared`)
    held: dict = {}           # (id(inputs), id(PartLog)) -> how far the log's shared runs lie in the occurrence's inputs (`_all_located`)
    for k, nd in enumerate(body.nodes):
        fx = node_fx.get(k)
        if fx is None:
            calls.append({"occ": None, "member": None, "row": None, "hint": nd.name, "returns": [], "args": []})
            continue
        o = occ_index[fx]
        member, row, hint = _member_row(nd.name, ctx.hint_name(fx), fx.name)
        rets: list[dict] = []
        off = 0
        for port, ct in enumerate(flat_types(nd.out)):
            lo, hi = off, off + ct.leaves
            off = hi
            slot: dict[str, Any] = {"port": port, "output": None, "leaves": None}
            for rlo, rhi, roff, oport in occ_outputs[o].get(("n", k), ()):
                if rlo <= lo and hi <= rhi:
                    if isinstance(oport, tuple):        # the Value lands in the storage of a WRITTEN fx argument (an in-place lowering)
                        slot["written_arg"] = oport[1]
                    else:
                        slot["output"] = oport
                    slot["leaves"] = [roff + (lo - rlo), roff + (hi - rlo)]
                    break
            rets.append(slot)
        args: list[dict] = []
        for a in nd.args:
            found: dict[str, Any] = {"input": None, "leaves": None}
            # fast path: the argument IS an fx input's whole binding (a weight, a pass-through): no walk at all
            whole = next((j for j, (_fx_arg, v, _ix) in enumerate(occ_inputs[o]) if isinstance(v, Coll) and v.refs is a.refs), None)
            if whole is not None:
                found["input"], found["leaves"] = whole, [0, int(a.type.leaves)]
                args.append(found)
                continue
            rs, log, s = _arg_runs(a.refs)
            if rs is None:
                found["unlocated"] = f"runs>{MAX_RUNS}"
                args.append(found)
                continue
            for j, (_fx_arg, _v, index) in enumerate(occ_inputs[o]):
                if not index:
                    continue
                r = _locate_runs(index, rs) if log is None else _locate_shared(index, rs, log, s, located)
                if r is False:
                    continue
                found["input"] = j
                found["leaves"] = [int(r[0]), int(r[1])] if r is not None else None
                break
            if found["input"] is None and len(occ_inputs[o]) > 1 and len(rs) > MAX_SPANS:
                if _all_located(occ_inputs[o], rs, log, s, held):
                    found["unlocated"] = f"spans>{MAX_SPANS}"
            elif found["input"] is None and len(occ_inputs[o]) > 1:
                # the Call's argument spans SEVERAL fx inputs (a collective's operand is `own ++ peer`): resolve run by run
                spans: list[dict] = []
                for space, i, lo, hi in rs:
                    hit = None
                    for j, (_fx_arg, _v, index) in enumerate(occ_inputs[o]):
                        r = _locate(index, space, i, lo, hi) if index else None
                        if r is not None:
                            hit = {"input": j, "leaves": [int(r[0]), int(r[1])]}
                            break
                    if hit is None:
                        spans = []
                        break
                    if spans and spans[-1]["input"] == hit["input"] and spans[-1]["leaves"][1] == hit["leaves"][0]:
                        spans[-1]["leaves"][1] = hit["leaves"][1]
                    else:
                        spans.append(hit)
                if spans:
                    found["spans"] = spans
            args.append(found)
        calls.append({"occ": o, "member": member, "row": row, "hint": hint, "returns": rets, "args": args})

    return {"schema": SCHEMA, "body": "program_fn", "inputs": input_names, "weights": field_names or None,
            "rank": _rank_coordinate(rank),
            "occurrences": occurrences, "calls": calls}


def attach(program, doc: dict) -> None:
    """Write `doc` under the one namespace of `program.annotations` (annotations are excluded from `program_digest`)."""
    program.annotations[NAMESPACE] = doc


__all__ = ["emit_correspondence", "attach", "NAMESPACE"]
