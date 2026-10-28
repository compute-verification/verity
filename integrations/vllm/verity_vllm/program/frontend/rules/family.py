"""Program families: claiming a REGISTERED program identity for a derived flat program (design §4.4, §43.4).

A derivation produces a flat program over leaf kinds (one call per kernel invocation). A `RegisteredFamily`
rule claims that this flat program *is* a registered composite program (`Serve_v1{C,LP,STEPS}`, a decoder
layer, ...) — the identity the certificate of record pins — and the claim is accepted only after verification:

1. the registered program's body is expanded to the same leaf granularity (composites whose base id the flat
   program calls are leaves; every other composite call is inlined; `batch`/`scan` above leaf level = no claim);
2. a node-for-node bijection between expanded leaves and flat nodes is constructed (same callee id, identical
   reference sequences for every argument once the flat side's node indices are substituted), with the flat
   program's weights record leaves bound to the family's weight roles as a side effect (a run of the family's
   `weights` param that a leaf consumes must correspond to a run of the module's parameter of equal length,
   consistently and injectively across all uses — so a tied parameter used twice binds one role twice, and two
   distinct module tensors can never bind one family role);
3. every flat node is matched (no extra computation — a source that computes more, even dead work, is a
   different program), and the return references agree.

Only then does the derivation return the registered program (`Program(bind(family, **statics))`), whose digest
is by construction the registered one; provenance records the verified isomorphism and the weights binding
(module fqn -> family role). Names (`nn_module_stack`, node hints) play no part.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import Body, CompositeDefinition, SpecializedDefinition
from verity.ml.circuits.defs import bind as ir_bind
from verity.ml.circuits.refs import Affine, Refs, concat
from verity.ml.circuits.refs import runs as _runs  # pure (was `..derive._runs`): this module must import without torch (capture fold M4 comparison)
from verity.ml.circuits.types import Array, Record, Tuple as TupleT, Type
from verity_vllm.program.frontend.rules.base import FamilyRule, RuleInfo

Run = tuple[str, int, int, int]  # (space, idx, lo, hi)


class _NoMatch(Exception):
    pass


@dataclass
class _Leaf:
    id: str
    args: list[Refs]
    out_leaves: int
    path: str


def _translate(refs: Refs, env: dict[tuple[str, int], Refs]) -> Refs:
    parts = []
    for space, idx, lo, hi in _runs(refs):
        src = env.get((space, idx))
        if src is None:
            raise _NoMatch(f"reference into unknown space {(space, idx)}")
        parts.append(src.slice(lo, hi))
    return concat(parts) if parts else refs


def node_leaf_id(node) -> str:
    """The matching id of a leaf node: the callee's descriptor id, and for `batch` nodes also the axes and the count."""
    if node.form == "batch":
        return f"batch[{descriptor_id(node.fn)}|axes={','.join('-' if a is None else str(a) for a in node.axes)}|n={node.n}]"
    return descriptor_id(node.fn)


def expand_leaves(spec: SpecializedDefinition, leaf_bases: set[str]) -> tuple[list[_Leaf], Refs, tuple[tuple[str, Type], ...], Type]:
    """Inline `spec`'s body down to calls of definitions whose base id is in `leaf_bases`. Params live in space
    "q"; leaf outputs in space "x" (index = leaf number)."""
    leaves: list[_Leaf] = []
    params, ret_t = spec.signature()

    def expand(sp: SpecializedDefinition, arg_refs: list[Refs], path: str) -> Refs:
        body: Body = sp.body
        env: dict[tuple[str, int], Refs] = {("p", i): r for i, r in enumerate(arg_refs)}
        for k, node in enumerate(body.nodes):
            targs = [_translate(a.refs, env) for a in node.args]
            fn = node.fn
            if node.form == "call" and isinstance(fn, SpecializedDefinition) and fn.base_id not in leaf_bases:
                env[("n", k)] = expand(fn, targs, f"{path}/{node.name or fn.id}")
            elif node.form in ("call", "primitive", "batch"):
                # a leaf: a call of a leaf-granularity composite, a primitive gate (Const / I32Eq / Select...), or a `batch`
                # node (its id carries the batched definition, axes and count: the same three things the derived node must have)
                j = len(leaves)
                leaves.append(_Leaf(node_leaf_id(node), targs, node.out.leaves, f"{path}/{node.name or k}"))
                env[("n", k)] = Affine("x", j, 0, node.out.leaves)
            else:
                raise _NoMatch(f"the family body has a {node.form} node of {descriptor_id(fn)} above leaf granularity at {path}")
        return _translate(body.ret, env)

    root_args = [Affine("q", i, 0, t.leaves) for i, (_, t) in enumerate(params)]
    ret = expand(spec, root_args, spec.id)
    return leaves, ret, tuple(params), ret_t


class _IntervalMap:
    """Injective piecewise-translation map between two leaf spaces (pattern weights -> flat weights)."""

    def __init__(self) -> None:
        self.fwd: list[tuple[int, int, int]] = []  # (p_lo, p_hi, f_lo) sorted by p_lo, non-overlapping
        self.rev: list[tuple[int, int, int]] = []  # (f_lo, f_hi, p_lo)

    def translate(self, lo: int, hi: int) -> list[tuple[int, int]] | None:
        """Flat runs covering [lo, hi) or None if any part is unmapped."""
        out = []
        pos = lo
        for p_lo, p_hi, f_lo in self.fwd:
            if p_hi <= pos:
                continue
            if p_lo > pos:
                return None
            take = min(hi, p_hi)
            out.append((f_lo + (pos - p_lo), f_lo + (take - p_lo)))
            pos = take
            if pos >= hi:
                break
        return out if pos >= hi else None

    def compatible(self, p_lo: int, p_hi: int, f_lo: int) -> bool:
        """Adding [p_lo,p_hi) -> f_lo is consistent (same delta where overlapping, injective elsewhere)."""
        delta = f_lo - p_lo
        for a, b, c in self.fwd:
            if a < p_hi and p_lo < b and c - a != delta:
                return False
        f_hi = f_lo + (p_hi - p_lo)
        for a, b, c in self.rev:
            if a < f_hi and f_lo < b and c - a != -delta:
                return False
        return True

    def add(self, p_lo: int, p_hi: int, f_lo: int) -> None:
        if not self.compatible(p_lo, p_hi, f_lo):
            raise _NoMatch(f"weights binding conflict for pattern leaves [{p_lo},{p_hi}) -> {f_lo}")
        self.fwd.append((p_lo, p_hi, f_lo))
        self.fwd.sort()
        self.rev.append((f_lo, f_lo + (p_hi - p_lo), p_lo))
        self.rev.sort()
        self._coalesce()

    def _coalesce(self) -> None:
        for lst, key in ((self.fwd, 0), (self.rev, 0)):
            i = 0
            while i + 1 < len(lst):
                a_lo, a_hi, a_to = lst[i]
                b_lo, b_hi, b_to = lst[i + 1]
                if b_lo <= a_hi and b_to - b_lo == a_to - a_lo:
                    lst[i] = (a_lo, max(a_hi, b_hi), a_to)
                    del lst[i + 1]
                else:
                    i += 1


def _field_intervals(t: Type, prefix: str, base: int = 0) -> list[tuple[int, int, str]]:
    """Leaf-typed fields of a (nested) record type as (lo, hi, name)."""
    if isinstance(t, Record):
        out, off = [], base
        for k, ft in t.fields:
            out += _field_intervals(ft, f"{prefix}.{k}" if prefix else k, off)
            off += ft.leaves
        return out
    if isinstance(t, TupleT):
        out, off = [], base
        for i, ft in enumerate(t.fields):
            out += _field_intervals(ft, f"{prefix}.{i}", off)
            off += ft.leaves
        return out
    if isinstance(t, Array) and isinstance(t.elem, (Record, TupleT)):
        out, off = [], base
        for i in range(t.n):
            out += _field_intervals(t.elem, f"{prefix}[{i}]", off)
            off += t.elem.leaves
        return out
    return [(base, base + t.leaves, prefix)]


def _align(pr: list[Run], fr: list[Run]):
    """Walk two run lists in lockstep by leaf offset, yielding aligned pieces (p_piece, f_piece)."""
    i = j = 0
    pi = pj = 0  # consumed within current runs
    total_p = sum(hi - lo for _, _, lo, hi in pr)
    total_f = sum(hi - lo for _, _, lo, hi in fr)
    if total_p != total_f:
        raise _NoMatch(f"argument leaf counts differ ({total_p} vs {total_f})")
    while i < len(pr) and j < len(fr):
        ps, pidx, plo, phi = pr[i]
        fs, fidx, flo, fhi = fr[j]
        rem_p, rem_f = (phi - plo) - pi, (fhi - flo) - pj
        take = min(rem_p, rem_f)
        yield (ps, pidx, plo + pi, plo + pi + take), (fs, fidx, flo + pj, flo + pj + take)
        pi += take
        pj += take
        if pi == phi - plo:
            i += 1
            pi = 0
        if pj == fhi - flo:
            j += 1
            pj = 0


def match_program(flat: Body, spec: SpecializedDefinition, *, flat_weights_param: int | None, allow_extra: bool = False,
                  with_map: bool = False) -> dict:
    """Verified isomorphism between a flat derived body and a registered program's expanded body. Returns a report
    dict with `matched` and, when matched, the weights binding (family field -> flat field).

    `allow_extra`: the derived body may contain nodes the family does not (a documented superset such as vLLM's
    row policy); the family's every leaf must still have its isomorphic derived node and the returns must agree.
    The report then has `matched=False`, `superset=True` and lists the `extra_nodes`. `with_map` adds `node_map`:
    `[derived node index, family leaf path]` for every matched leaf (an address map for correspondence)."""
    t0 = time.time()
    rep: dict = {"family": spec.id, "matched": False, "flat_nodes": len(flat.nodes)}
    try:
        forms = {n.form for n in flat.nodes}
        if forms - {"call", "primitive", "batch"}:
            raise _NoMatch(f"the flat program has {sorted(forms - {'call', 'primitive', 'batch'})} nodes; families match call / batch leaves only")
        leaf_bases = {n.fn.base_id for n in flat.nodes if isinstance(n.fn, SpecializedDefinition)}
        leaves, pret, pparams, pret_t = expand_leaves(spec, leaf_bases)
        rep["family_leaves"] = len(leaves)
        if len(pparams) != len(flat.params):
            raise _NoMatch(f"parameter counts differ: family {len(pparams)} vs derived {len(flat.params)}")
        pw = len(pparams) - 1 if flat_weights_param is not None else None
        for i, ((_, pt), (_, ft)) in enumerate(zip(pparams, flat.params)):
            if i == pw:
                continue
            if pt != ft:
                raise _NoMatch(f"input {i} type differs: family {pt!r} vs derived {ft!r}")
        if pret_t != flat.ret_type:
            raise _NoMatch(f"return type differs: family {pret_t!r} vs derived {flat.ret_type!r}")
        if len(leaves) != len(flat.nodes) and not (allow_extra and len(flat.nodes) > len(leaves)):
            rep["family_leaf_ids"] = _histogram(l.id for l in leaves)
            rep["flat_node_ids"] = _histogram(descriptor_id(n.fn) for n in flat.nodes)
            raise _NoMatch(f"leaf counts differ: family {len(leaves)} vs derived {len(flat.nodes)}")

        flat_runs = [[_runs(a.refs) for a in n.args] for n in flat.nodes]
        flat_ids = [node_leaf_id(n) for n in flat.nodes]
        flat_out = [n.out.leaves for n in flat.nodes]
        by_id: dict[str, list[int]] = {}
        for k, fid in enumerate(flat_ids):
            by_id.setdefault(fid, []).append(k)
        node_map: dict[int, int] = {}
        used: set[int] = set()
        wmap = _IntervalMap()

        def translate_known(runs: list[Run]) -> list[Run] | None:
            out: list[Run] = []
            for space, idx, lo, hi in runs:
                if space == "x":
                    k = node_map.get(idx)
                    if k is None:
                        return None
                    out.append(("n", k, lo, hi))
                elif space == "q" and idx == pw:
                    tr = wmap.translate(lo, hi)
                    if tr is None:
                        return None
                    out += [("p", flat_weights_param, a, b) for a, b in tr]
                elif space == "q":
                    out.append(("p", idx, lo, hi))
                else:
                    raise _NoMatch(f"unexpected space {space}")
            return _merge(out)

        def try_candidate(leaf: _Leaf, k: int) -> tuple[list[tuple[int, int, int]], dict[int, int]] | None:
            """Bindings (p_lo, p_hi, f_lo) the match would add plus the zero-arg leaves it binds through this
            consumer (family leaf -> derived node), or None if k does not match."""
            new: list[tuple[int, int, int]] = []
            zero: dict[int, int] = {}
            if len(leaf.args) != len(flat_runs[k]):
                return None
            for pr, fr in zip(leaf.args, flat_runs[k]):
                prs = _runs(pr)
                try:
                    pieces = list(_align(prs, fr))
                except _NoMatch:
                    return None
                for (ps, pidx, plo, phi), (fs, fidx, flo, fhi) in pieces:
                    if ps == "x":
                        kk = node_map.get(pidx, zero.get(pidx))
                        if kk is None and pidx in deferred:
                            # a zero-argument leaf (a constant gate) is identified by who reads it: bind it here if the
                            # derived node read is an unused gate of the same kind, covering the whole output
                            if (fs == "n" and fidx not in used and fidx not in zero.values() and flat_ids[fidx] == leaves[pidx].id
                                    and plo == 0 and flo == 0 and phi == leaves[pidx].out_leaves and fhi == flat_out[fidx]):
                                zero[pidx] = fidx
                                continue
                            return None
                        if kk is None or fs != "n" or fidx != kk or flo != plo:
                            return None
                    elif ps == "q" and pidx == pw:
                        if fs != "p" or fidx != flat_weights_param:
                            return None
                        tr = wmap.translate(plo, phi)
                        if tr is not None:
                            if tr != [(flo, fhi)]:
                                return None
                        else:
                            if not wmap.compatible(plo, phi, flo):
                                return None
                            for a, b, c in new:  # consistency among this candidate's own new bindings
                                if a < phi and plo < b and c - a != flo - plo:
                                    return None
                            new.append((plo, phi, flo))
                    elif ps == "q":
                        if fs != "p" or fidx != pidx or flo != plo:
                            return None
                    else:
                        return None
            return new, zero

        deferred: set[int] = set()  # zero-argument leaves with several candidates: bound through their first reader
        for j, leaf in enumerate(leaves):
            cands = [k for k in by_id.get(leaf.id, ()) if k not in used]
            if not cands:
                raise _NoMatch(f"no derived node for family leaf {j} {leaf.id} at {leaf.path}")
            if not leaf.args and len(cands) > 1:
                deferred.add(j)
                continue
            # narrow by the first fully known argument
            if len(cands) > 1:
                for ai, a in enumerate(leaf.args):
                    tr = translate_known(_runs(a))
                    if tr is not None:
                        cands = [k for k in cands if _merge(flat_runs[k][ai]) == tr]
                        break
            chosen = None
            for k in cands:
                res = try_candidate(leaf, k)
                if res is not None:
                    chosen = (k, res)
                    break
            if chosen is None:
                raise _NoMatch(f"family leaf {j} {leaf.id} at {leaf.path} has no isomorphic derived node "
                               f"({len(by_id.get(leaf.id, ()))} derived calls of that kind, {len(cands)} unused candidates checked)")
            k, (new, zero) = chosen
            for p_lo, p_hi, f_lo in new:
                wmap.add(p_lo, p_hi, f_lo)
            for pj, fk in zero.items():
                node_map[pj] = fk
                used.add(fk)
                deferred.discard(pj)
            node_map[j] = k
            used.add(k)
        if deferred:
            # constant gates nobody reads (dead in the family too): pair the remaining ones by kind
            for j in sorted(deferred):
                cands = [k for k in by_id.get(leaves[j].id, ()) if k not in used]
                if not cands:
                    raise _NoMatch(f"no derived node for family leaf {j} {leaves[j].id} at {leaves[j].path}")
                node_map[j] = cands[0]
                used.add(cands[0])
        extra_idx = [k for k in range(len(flat.nodes)) if k not in used]
        if extra_idx and not allow_extra:
            extra = [f"{k}: {flat_ids[k]}" for k in extra_idx[:8]]
            raise _NoMatch(f"{len(flat.nodes) - len(used)} derived nodes have no family counterpart (extra computation): {extra}")
        if extra_idx:
            rep["extra_nodes"] = [{"index": k, "id": flat_ids[k], "name": flat.nodes[k].name,
                                   "args": [[list(r) for r in _merge(rs)] for rs in flat_runs[k]]} for k in extra_idx]
            rep["superset"] = True
        if with_map:
            rep["node_map"] = [[node_map[j], leaves[j].path] for j in range(len(leaves))]
        tr = translate_known(_runs(pret))
        if tr is None:
            raise _NoMatch("the family's return references cover unbound weights")
        if tr != _merge(_runs(flat.ret)):
            raise _NoMatch("return reference sequences differ")

        # weights binding at field granularity
        binding: dict[str, str] = {}
        partial: list[str] = []
        if pw is not None:
            pfields = _field_intervals(pparams[pw][1], "")
            ffields = _field_intervals(flat.params[flat_weights_param][1], "")
            fby = {name: (lo, hi) for lo, hi, name in ffields}
            for lo, hi, name in pfields:
                segs = [(a, b, c) for a, b, c in wmap.fwd if a < hi and lo < b]
                if not segs:
                    partial.append(f"{name}: unreferenced by any leaf")
                    continue
                deltas = {c - a for a, b, c in segs}
                if len(deltas) != 1:
                    raise _NoMatch(f"family field {name} binds to non-contiguous module leaves")
                d = next(iter(deltas))
                f_lo, f_hi = lo + d, hi + d
                owner = [fn for fn, (a, b) in fby.items() if a <= f_lo and f_hi <= b]
                if len(owner) != 1 or fby[owner[0]] != (f_lo, f_hi):
                    raise _NoMatch(f"family field {name} does not bind a whole module tensor (leaves [{f_lo},{f_hi}))")
                binding[name] = owner[0]
            bound_flat = set(binding.values())
            rep["unused_module_weights"] = [name for _, _, name in ffields if name not in bound_flat]
            rep["unreferenced_family_fields"] = partial
        rep.update(matched=not extra_idx, weights_binding=binding, node_bijection=len(node_map), seconds=round(time.time() - t0, 3))
        if extra_idx:
            rep["reason"] = (f"{len(extra_idx)} derived nodes have no family counterpart (extra computation); every family leaf is "
                            f"matched and the returns agree — a documented superset, not the family's identity")
    except _NoMatch as e:
        rep.update(matched=False, reason=str(e), seconds=round(time.time() - t0, 3))
    return rep


def _merge(runs: list[Run]) -> list[Run]:
    out: list[Run] = []
    for r in runs:
        if out and out[-1][0] == r[0] and out[-1][1] == r[1] and out[-1][3] == r[2]:
            out[-1] = (r[0], r[1], out[-1][2], r[3])
        else:
            out.append(r)
    return out


def _histogram(ids) -> dict[str, int]:
    h: dict[str, int] = {}
    for i in ids:
        h[i] = h.get(i, 0) + 1
    return dict(sorted(h.items()))


# ---------------------------------------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------------------------------------


class RegisteredFamily(FamilyRule):
    """Claim `defn{**binder(flat, ctx)}` for a derived flat program after verified isomorphism."""

    def __init__(self, rule_id: str, version: int, defn: CompositeDefinition, binder: Callable, doc: str, modes=frozenset({"profile"})) -> None:
        self.defn, self.binder, self.modes = defn, binder, modes
        self.info = RuleInfo(rule_id, version, "program-family", applies_to=f"derived flat programs whose leaf kinds are those of {defn.name}_v{defn.version}",
                             preconditions=("the binder resolves the family's statics from the derivation's config / input shapes",
                                            "expanded family body and derived body are node-for-node isomorphic (callee ids, reference sequences)",
                                            "weights bind injectively: one module tensor per family role, tied tensors reuse one role",
                                            "every derived node is matched (no extra computation) and the returns agree"),
                             evidence_domain=doc)

    def applies_program(self, flat_sd: SpecializedDefinition, ctx) -> bool | str:
        try:
            statics = self.binder(flat_sd, ctx)
        except Exception as e:  # noqa: BLE001
            return f"binder failed: {type(e).__name__}: {e}"
        if statics is None:
            return "binder: not applicable to this derivation"
        self._statics = statics
        return True

    def claim(self, flat_sd: SpecializedDefinition, ctx) -> dict | None:
        try:
            spec = ir_bind(self.defn, **self._statics)
            spec.signature()
        except Exception as e:  # noqa: BLE001
            return {"family": f"{self.defn.name}_v{self.defn.version}", "matched": False, "reason": f"cannot bind statics: {type(e).__name__}: {e}"}
        wp = len(flat_sd.body.params) - 1 if ctx.weights_coll is not None else None
        rep = match_program(flat_sd.body, spec, flat_weights_param=wp)
        rep["rule"] = self.rule_id
        if rep["matched"]:
            rep["program"] = spec
            rep["module_weights_by_role"] = {role: _module_name(ctx, fld) for role, fld in rep["weights_binding"].items()}
        return rep


def _module_name(ctx, field_name: str) -> str:
    for w in ctx.weights:
        if w.name == field_name:
            return w.name + (f" (tied: {', '.join(w.aliases)})" if w.aliases else "")
    return field_name


# ---------------------------------------------------------------------------------------------------------
# the registered families of the vLLM eager profile
# ---------------------------------------------------------------------------------------------------------

from verity.ml.definitions.registry import b1 as _b1  # noqa: E402


def _serve_binder(flat_sd, ctx):
    C = ctx.config
    if not all(k in C for k in ("H", "NL", "NH", "KVH", "D", "I", "V", "QKV", "GU", "EPS", "MAX_POS")):
        return None
    params, ret = flat_sd.signature()
    if len(params) != 2 or not isinstance(ret, Array):
        return None
    LP = params[0][1].n if isinstance(params[0][1], Array) else None
    if LP is None:
        return None
    return {"C": C, "LP": LP, "STEPS": ret.n - 1}


SERVE_FAMILY = RegisteredFamily("Family[Serve_v1]", 1, _b1.Serve, _serve_binder,
                                "structural: verified isomorphism to Serve_v1{C,LP,STEPS} (the certificate of record's program identity)")


def _serve_v3_binder(flat_sd, ctx):
    """`Serve_v3{C3, LP, D_MAX, GEMM, ATTENTION, NORM}` (serve3.py): the padded circuit. `C3` is the derivation's config
    with `EOS_ID` (spec §10: bound at the binding site, never into a v1/v2 dict); the guards the source computed must
    all test that id. The implementation records come from the vocabulary (a v2-skeleton profile)."""
    from verity_vllm.program.frontend.rules.vocab import profile_records
    C = ctx.config
    if not all(k in C for k in ("H", "NL", "NH", "KVH", "D", "I", "V", "QKV", "GU", "EPS", "MAX_POS", "EOS_ID")):
        return None
    recs = profile_records(ctx.vocab)
    if recs is None:
        return None
    params, ret = flat_sd.signature()
    if len(params) != 2 or not isinstance(ret, TupleT) or len(ret.fields) != 2:
        return None
    LP = params[0][1].n if isinstance(params[0][1], Array) else None
    out_t = ret.fields[0]
    if LP is None or not isinstance(out_t, Array):
        return None
    eos = {a["preconditions_checked"].get("literal") for a in ctx.applications if a["rule"].startswith("EqScalar[pad]")}
    if eos != {C["EOS_ID"]}:
        return None  # the guards test a different id (or none): not this program
    return {"C": C, "LP": LP, "D_MAX": out_t.n - 1, "GEMM": recs["GEMM"], "ATTENTION": recs["ATTENTION"], "NORM": recs["NORM"](C)}


from verity.ml.definitions.registry import serve3 as _serve3  # noqa: E402

SERVE_V3_FAMILY = RegisteredFamily("Family[Serve_v3]", 1, _serve3.ServeV3, _serve_v3_binder,
                                   "structural: verified isomorphism to Serve_v3{C3,LP,D_MAX,GEMM,ATTENTION,NORM} (the padded circuit, serve3-spec)")


def _layer_weights_with_table(C: dict) -> Record:
    lw = _b1.layer_weights_type(C)
    from verity.ml.circuits.types import Value
    return Record(*(list(lw.fields) + [("cos_sin", Array(C["MAX_POS"], Array(C["D"], Value(16))))]))


def _decoder_layer_sig(S):
    C = S.C
    H = C["H"]
    from verity.ml.circuits.types import Value
    BF = Value(16)
    rows = Array(S.T, Array(H, BF))
    params = [("h", rows)] + ([] if S.L0 else [("res", rows)]) + [("w", _layer_weights_with_table(C))]
    return (tuple(params), TupleT(rows, rows))


def _decoder_layer_body(B, S, h, *rest):
    """TA1's layer at positions T0..T0+T-1 (one layer's prefill): LayerPre x T, Attention x T over this call's
    K/V rows, LayerPost x T. Layer 0 (`L0`) takes no residual (the residual is `h`)."""
    C, T, T0 = S.C, S.T, S.T0
    res, w = (h, rest[0]) if S.L0 else (rest[0], rest[1])
    from verity.ml.circuits.refs import Coll as _Coll, array_of as _array_of, tuple_of
    lw = _Coll(_b1.layer_weights_type(C), w.refs.slice(0, _b1.layer_weights_type(C).leaves))
    cs = w["cos_sin"]
    pre = [B.call(ir_bind(_b1.LayerPre, C=C, L0=S.L0), h[i], res[i], lw, cs[T0 + i], name=f"t={T0 + i}/pre") for i in range(T)]
    kc = _Coll(_b1.kv_type(C, T), concat([p[2].refs for p in pre]))
    vc = _Coll(_b1.kv_type(C, T), concat([p[3].refs for p in pre]))
    outs, ress = [], []
    for i in range(T):
        attn = B.call(ir_bind(_b1.Attention, T=i + 1, NH=C["NH"], KVH=C["KVH"], D=C["D"]), pre[i][1], kc[0:i + 1], vc[0:i + 1], name=f"t={T0 + i}/attn")
        post = B.call(ir_bind(_b1.LayerPost, C=C), attn, pre[i][0], lw, name=f"t={T0 + i}/post")
        outs.append(post[0])
        ress.append(post[1])
    return tuple_of(_array_of(outs), _array_of(ress))


#: The hand-built decoder layer at T positions (TA1's LayerPre/Attention/LayerPost) — the S2 structural oracle.
DecoderLayerT = CompositeDefinition("DeriveOracleDecoderLayer", 1, ("C", "L0", "T", "T0"), _decoder_layer_sig, _decoder_layer_body,
                                    doc="TA1 decoder layer at T positions (prefill of one layer): LayerPre x T, Attention x T, LayerPost x T",
                                    conformance="structural oracle (composed of exact-model-tested / model-untested TA1 composites)", register=False)


def _layer_binder(flat_sd, ctx):
    C = ctx.config
    if not all(k in C for k in ("H", "NL", "NH", "KVH", "D", "I", "QKV", "GU", "EPS", "MAX_POS")):
        return None
    params, ret = flat_sd.signature()
    if len(params) not in (2, 3):
        return None
    h_t = params[0][1]
    if not (isinstance(h_t, Array) and isinstance(h_t.elem, Array)):
        return None
    ids = {n.fn.base_id for n in flat_sd.body.nodes if isinstance(n.fn, SpecializedDefinition)}
    L0 = "RMSNormTriton_v1" in ids
    if L0 != (len(params) == 2):
        return None
    T0 = 0
    for a in ctx.applications:
        pos = a.get("preconditions_checked", {}).get("positions")
        if a["rule"].startswith("RotaryEmbedding") and pos:
            T0 = pos[0]
            break
    return {"C": C, "L0": L0, "T": h_t.n, "T0": T0}


LAYER_FAMILY = RegisteredFamily("Family[DecoderLayerT]", 1, DecoderLayerT, _layer_binder,
                                "structural: verified isomorphism to the hand-built TA1 layer at T positions (S2 oracle)")

DEFAULT_FAMILIES = [SERVE_FAMILY, SERVE_V3_FAMILY, LAYER_FAMILY]

__all__ = ["RegisteredFamily", "match_program", "expand_leaves", "node_leaf_id", "SERVE_FAMILY", "SERVE_V3_FAMILY", "LAYER_FAMILY", "DecoderLayerT", "DEFAULT_FAMILIES"]
