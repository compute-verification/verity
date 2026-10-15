"""The Program rows sampled replay reads, indexed per request.

`ProgramIndex` holds a request Program's rows with the manifest address of each output slice (op_path, member, request-local
step and row), the weights struct a row binds (`parse_struct`), and the MoE plane a row's module is committed under
(`moe_plane_kind`).  Nothing here reads committed words.
"""
from __future__ import annotations

import bisect
import re
from typing import Any

import numpy as np

from verity_vllm.query.manifest import format as RM
from verity_vllm.query.manifest.format import moe_plane_kind
from verity_vllm.program.kernels.rows import SAMPLED_MEMBER


SAMPLER_OP = "runner.sampler"
IDS_MEMBER = "in"
SEED_PARAM = "seed"                     # the stochastic Program's seed input (Param name; committed as runner.sampler/seed per (step, request))
VALUES_PARAM = "values"                 # a `registered_values` Program's registered values (the norms' eps, the attention scale)
_RUN_PARAM = re.compile(r"^run(_\d+)?$")  # its run values, one parameter per request (the sampler's temperature and top_p, a Philox position)

_CONST_ROW = re.compile(r"^Const(32|64)\[0x([0-9a-fA-F]+)\]_v\d+$")


def const_word(row: dict | None) -> np.ndarray | None:
    """A `Const32[0x..]_v1` / `Const64[0x..]_v1` Program row -> its one word (u32 / u64), else None.  The constant IS the Program's
    text (top_p, temperature, pos of the stochastic selector): read from the spec, no committed identity to look up."""
    m = _CONST_ROW.match(str((row or {}).get("spec") or ""))
    if m is None:
        return None
    bits, hexv = int(m.group(1)), int(m.group(2), 16)
    if hexv >= (1 << bits):
        return None
    return np.asarray([hexv], dtype=np.uint32 if bits == 32 else np.uint64)


def parse_struct(type_str: str) -> list[tuple[str, int, list[int], int]]:
    """`{a: Array<3,Array<2,Value<16>>>, b: Array<4,Value<16>>}` -> [(name, element count, shape, value bits)] in declaration order."""
    s = type_str.strip()
    if not (s.startswith("{") and s.endswith("}")):
        raise ValueError("weights parameter is not a struct type")
    body, fields, depth, buf = s[1:-1], [], 0, ""
    for ch in body:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            fields.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        fields.append(buf)
    out = []
    for f in fields:
        name, _, ty = f.strip().partition(":")
        ty = ty.strip()
        shape, rest, bits = [], ty, 0
        while rest.startswith("Array<"):
            n, _, rest = rest[len("Array<"):].partition(",")
            shape.append(int(n))
            rest = rest.strip()
            if rest.endswith(">"):
                rest = rest[:-1]
        if rest.startswith("Value<"):
            bits = int(rest[len("Value<"):].rstrip(">").rstrip(">"))
        count = 1
        for n in shape:
            count *= n
        out.append((name.strip(), count, shape, bits))
    return out


#: the query engine whose population addresses a Call's committed output (`addresses.rule` on the record)
V2_QUERY = "v2-query"
#: the query a `ProgramIndex` addresses under when its caller names none (None: the query of record)
DEFAULT_QUERY: str | None = None


class ProgramIndex:
    """The Program's rows with their manifest addresses: (op_path, member, request-local step, request-local row) per output slice.

    A row's members come from the v2 query's population of the SAME Program (`verity_vllm.query.required.Population.members_of`: the
    required Values a Call produces under `query` -- the manifest's; the query of record by default -- + the policy, the rows the manifest
    counts), read with the correspondence of its Build dir (`program_dir`; without one, the rows' recorded names)."""

    def __init__(self, inst: dict[str, Any], *, prefix: str | None = None, attention_paths: list[str] | None = None, tp: dict[str, Any] | None = None,
                 program_dir: str | None = None, query: str | None = None, norm_scales: bool = False):
        self.rows: list[dict] = list(inst["rows"])
        # [tp TP-08d] a RANK Program (derive_step --tp W --rank r; result.json `tp` = {"world": W, "rank": r}): its population carries the
        # TP-04 policy (the rank-local partials `part_rank<r>` / `shard_rank<r>` beside every collective output), the one the rank's
        # manifest was built with.  `tp` = {"rank": r, "world": W}.
        self.tp: dict[str, Any] | None = (dict(rank=int(tp.get("rank", 0)), world=int(tp.get("world", 1))) if tp and int(tp.get("world", 1) or 1) > 1 else None)
        # [v2 flip, P2] Call id -> [(member, width, dtype)] of the required Values it produces, and the Program path they are spelt under
        # (`Population.program_path`: its owning module, a collective site's site path)
        self._v2: dict[int, list[tuple[str, int, str]]] = {}
        self._v2_ports: dict[int, dict[str, str]] = {}
        self._v2_module: dict[int, str] = {}
        self._v2_fault: dict[int, str] = {}
        from verity_vllm.correspondence.reader_for_query import Correspondence
        from verity_vllm.query.program_view import from_instances
        from verity_vllm.query import norm_scales as NS
        from verity_vllm.query.required import population
        # a manifest built under the norm-scale policy spells an ATen norm's rsqrt as `<norm module>/norm_scale`, not a Call boundary
        population = NS.population if norm_scales else population
        P = from_instances(inst)
        corr = Correspondence.of(P, program_dir, implementation_paths=attention_paths or ())
        pop = population(P, corr, tp=self.tp, **({"query": q} if (q := query or DEFAULT_QUERY) else {}))
        for cid in pop.producers():
            mem = pop.members_of(cid)
            if mem:
                self._v2[cid] = [(m.member, m.width, m.dtype) for m in mem]
                self._v2_ports[cid] = dict(reversed([(m.port, m.member) for m in mem]))      # a port's own member, before a partial on it
                self._v2_module[cid] = pop.program_path(cid)
        self.correspondence: dict[str, Any] | None = corr.provenance()
        del P, corr, pop
        self.params: list[list] = [list(p) for p in inst.get("params", [])]
        self.prefix = RM.wrapper_prefix_of(self.rows, attention_paths or []) if prefix is None else prefix
        self.segs = RM.segment_steps(self.rows)
        self._seg_lo = [lo for lo, _hi in self.segs]
        self.by_i = {int(r["i"]): r for r in self.rows}
        self.pnames = [p[0] for p in self.params]
        # the supplied inputs' layout as the build states it (instances header `supplied`): parameter index -> (binding, {leaf offset: field})
        self.supplied: dict[int, tuple[str, dict[int, dict]]] = {}
        sup = inst.get("supplied") or {}
        for binding, ents in (("registration", [sup["registration"]] if sup.get("registration") else []), ("run", sup.get("run") or [])):
            for ent in ents:
                a = int(ent["param"])
                if a >= len(self.pnames) or self.pnames[a] != ent["name"]:
                    raise ValueError(f"supplied {binding} parameter {ent['name']!r} at index {a} is not the Program's parameter "
                                     f"{self.pnames[a] if a < len(self.pnames) else None!r}")
                self.supplied[a] = (binding, {int(f["offset"]): dict(f, bits=int(f["bits"], 16)) for f in ent.get("fields") or []})
        for a, n in enumerate(self.pnames):
            if (n == VALUES_PARAM or _RUN_PARAM.match(n)) and a not in self.supplied:
                raise ValueError(f"parameter {n!r} is a supplied input but the instances header states no layout for it")
        self.fields: list[tuple[str, int, list[int], int]] = []
        self.foff: list[int] = []
        wt = next((p[1] for p in self.params if p[0] == "weights"), None)
        if wt:
            self.fields = parse_struct(wt)
            off = 0
            for _n, c, _s, _b in self.fields:
                self.foff.append(off)
                off += c
        self._desc: dict[int, dict | None] = {}
        # [MAN-06f v3, moe R16] the fused-MoE block rows of each step: per MoE module its geometry (M routed tokens, TOPK, E, H, I, N_up --
        # `manifest_format.moe_geometry`, the same numbers the manifest sizes the `moe_block_stream` planes with) and, per row, its
        # PAIR / TOKEN ordinal among the rows of its plane kind in Program order (router / sum rows: token t; up / act / down rows: pair
        # p = t*TOPK + k).  `describe` addresses these rows at their PLANE (op_path `moe/L<k>`, member = the plane).
        self._moe_geo: list[dict[str, dict[str, Any]]] = [RM.moe_geometry(self.rows[lo: hi + 1]) for lo, hi in self.segs]
        self._moe_ord: dict[int, int] = {}
        self._moe_kind: dict[int, str] = {}
        mcounts: dict[tuple, int] = {}
        for r in self.rows:
            mod0 = RM.split_name(r.get("name") or "")[0]
            kind = moe_plane_kind(RM.spec_family(r["spec"]), mod0)
            if kind == "quant":
                # the activation's group quant reads an `act` row of its runner; the input's reads the block's input
                first = ((r.get("args") or [[]])[0] or [None])[0]
                src = self.by_i.get(int(first[1])) if first and first[0] == "n" else None
                prod = RM.split_name(src.get("name") or "")[0] if src else None
                kind = "act_quant" if prod == mod0 and moe_plane_kind(RM.spec_family(src["spec"]), prod) == "act" else "in_quant"
            if kind is not None:
                key = (mod0, kind, self.step_of(int(r["i"])))
                self._moe_kind[int(r["i"])] = kind
                self._moe_ord[int(r["i"])] = mcounts.get(key, 0)
                mcounts[key] = self._moe_ord[int(r["i"])] + 1
        self._ordinal: dict[int, int] = {}
        counts: dict[tuple, int] = {}
        for r in self.rows:
            i = int(r["i"])
            s = self.step_of(i)
            # the request-local ROW of a Call = its ordinal among the Calls producing the same (module, member) in its step, Program order --
            # exactly the rows `required.request_manifest` counts per identity; a Call whose members would sit at different rows is not one
            # addressable row (named, never read at a guessed row)
            mem2 = self._v2.get(i)
            if mem2:
                ords = []
                for m in mem2:
                    k2 = (self._v2_module[i], m[0], s)
                    ords.append(counts.get(k2, 0))
                    counts[k2] = ords[-1] + 1
                if len(set(ords)) == 1:
                    self._ordinal[i] = ords[0]
                else:
                    self._v2_fault[i] = f"members {[m[0] for m in mem2]} sit at request-local rows {ords}: not one row"

    @property
    def n_steps(self) -> int:
        return len(self.segs)

    def ids_params(self) -> set[int]:
        """The Program parameter(s) carrying the declared prompt ids: every parameter that is not the registered `weights` record, not a
        sampler input (`seed`, `splits`) and not a supplied input (`values`, `run_<i>`) -- the same reading `_gather_inputs` resolves a
        `['p', a, k, k+1]` slice by (the committed `input_token_ids` row k of the arrival step)."""
        return {a for a, n in enumerate(self.pnames) if n not in ("weights", SEED_PARAM, RM.SPLITS_PARAM) and a not in self.supplied}

    def embed_op_path(self) -> str | None:
        """The op_path of the Program's Embedding rows (the `input_token_ids` identity lives under it as member `in`): the first row whose
        arguments READ the prompt-ids parameter (`ids_params`) -- the Program's own dataflow names the consumer of the declared ids; the
        row's family name never enters (Draft 3.1 §12.4)."""
        if not hasattr(self, "_embed_op"):
            self._embed_op = None
            ids = self.ids_params()
            for r in self.rows:
                if any(sl and sl[0] == "p" and int(sl[1]) in ids for g in (r.get("args") or []) for sl in g):
                    d = self.describe(r)
                    self._embed_op = d["op_path"] if d else None
                    break
        return self._embed_op

    def step_of(self, i: int) -> int:
        return max(0, bisect.bisect_right(self._seg_lo, int(i)) - 1)

    def member_of_port(self, i: int) -> dict[str, str]:
        """Row i's output port -> the member it is committed as (`CallMember.port`); {} for a row whose outputs are interior."""
        return dict(self._v2_ports.get(int(i), {}))

    def describe(self, r: dict) -> dict | None:
        """Family, statics, module -> op_path, members (name, width, dtype) and request-local row of a Program row; None if unmodelled."""
        i = int(r["i"])
        if i in self._desc:
            return self._desc[i]
        fam = RM.spec_family(r["spec"])
        st = RM.spec_statics(r["spec"])
        mod, node, hint, row = RM.split_name(r.get("name") or "")
        d: dict | None
        kind = self._moe_kind.get(i)
        # [F-r19-int-24] FusedMoE.forward returns the MoeSum: a boundary Value (`<experts>/out`) beside its plane copy (MOE-TP-02 below)
        tp_sum = kind == "sum" and bool(self._v2.get(i))
        if tp_sum:
            kind = None
        if RM.is_token_select(fam):
            d = {"family": fam, "statics": st, "module": "", "op_path": SAMPLER_OP, "hint": hint, "row": 0,
                 "members": [(SAMPLED_MEMBER, 1, "i64")], "step": self.step_of(i)}
        elif kind is not None:
            d = self._describe_plane(i, fam, st, mod, hint, row, kind)
        else:
            fault = self._v2_fault.get(i)
            mod2 = self._v2_module.get(i, mod)
            d = None if fault else {"family": fam, "statics": st, "module": mod2, "op_path": RM.collector_path(mod2, self.prefix), "hint": hint,
                                    "row": int(self._ordinal.get(i, 0)), "name_row": row, "members": list(self._v2.get(i, ())), "step": self.step_of(i)}
        if tp_sum and d is not None and d["members"]:
            # [MOE-TP-02 / moe, tp M-1528] on a TP rank the MoeSum row's ONE output is committed under TWO required names: the module's
            # `part_rank<r>` (tp_rank_partials, TP-04/TP-06: the all-reduce input) AND the MOE-01 plane `moe/L<k>/sum` (moe_block_stream,
            # staged by MoeBlockSource on the rank exactly as at TP1).  The manifest requires both (TP-06); the
            # replay addressed the row at the partial only, so F-r17b-38's reconciliation read every `moe/L<k>/sum` identity as
            # `identities_without_rows` (OLMoE TP2 B8 int-6: 6,944 per rank, COMMIT FAIL).  `copies` = the plane address of the same
            # output: recorded in the population under the plane identity AND read back and compared against the same evaluated words
            # (two committed copies, one relation, both checked).  The plane mapping is the checked one (_describe_plane: geometry,
            # ordinal, dataflow); a mapping failure is NAMED here and the plane identity stays without rows -- fail closed, never aliased.
            # The same holds at TP1: the MoeSum is the module's returned tensor `<experts>/out` (F-r19-int-24).
            pd = self._describe_plane(i, fam, st, mod, hint, row, "sum")
            if pd.get("plane_error") or not pd["members"]:
                d["copy_error"] = pd.get("plane_error") or "sum plane has no member"
            else:
                pm = pd["members"][0]
                why = "F-r19-int-24: the MoeSum output = the MoE module's returned tensor (a boundary Value) = its moe_block_stream sum plane"
                d["copies"] = [{"op_path": pd["op_path"], "member": pm[0], "width": int(pm[1]), "dtype": str(pm[2]), "row": int(pd["row"]),
                                "of": d["members"][0][0], "why": why}]
        self._desc[i] = d
        return d

    def _describe_plane(self, i: int, fam: str, st: dict[str, Any], mod: str, hint: str, name_row: int | None, kind: str) -> dict:
        """[MAN-06f v3] a fused-MoE block row -> its `moe_block_stream` PLANE address: op_path `moe/L<layer>`, members = the plane(s) the row
        produces (router: `router_w` f32[TOPK] + `router_ids` i32[TOPK] per token; up: `up` bf16[N_up] per pair; act (the SiluMul under the
        MoE runner): `act` bf16[I] per pair; in_quant: `in_q` e4m3[H] + `in_scale` f32[H/G] per token; act_quant: `act_q` e4m3[I] +
        `act_scale` f32[I/G] per pair; down: `down` bf16[H] per pair; sum: `sum` bf16[H] per token), row = the row's token / pair
        ordinal in Program order (pair-major p = t*TOPK + k: MOE-03 source order, `moe_geometry` docstring).  The plane's request-local
        element range is the binding's per-request split of the batched plane (binding.py: rows == tokens -> the request's token rows, x
        TOPK for `act`), so `store.words(rid, step, "moe/L<k>", plane, row*width + lo, ...)` reads exactly this row's words.  The mapping
        is CHECKED against the Program, never guessed: the module's geometry must be consistent (expert-GEMM rows == M*TOPK, the rule
        build_manifest sizes the planes with), the ordinal must lie inside the plane (M tokens / M*TOPK pairs) and must equal the row's own
        `[k]` name index when the Program names one; a violation yields members [] + `plane_error` -> counted by NAME as a mapping failure
        (population), never a silent read of another pair's words."""
        s = self.step_of(i)
        g = self._moe_geo[s].get(mod) or {}
        layer = RM.layer_of(mod)
        M, TOPK = int(g.get("M", 0) or 0), int(g.get("TOPK", 0) or 0)
        ord_ = int(self._moe_ord.get(i, -1))
        err: str | None = None
        if layer < 0:
            err = f"MoE module {mod!r} has no `.layers.<k>` index (plane name moe/L<k> undefined)"
        elif M <= 0 or TOPK <= 0:
            err = f"MoE module {mod!r} step {s}: no router rows / TOPK in the Program (M={M}, TOPK={TOPK})"
        elif int(g.get("expert_gemm_rows", 0)) != M * TOPK:
            err = f"MoE module {mod!r} step {s}: {g.get('expert_gemm_rows')} expert-GEMM rows != M*TOPK = {M}*{TOPK}"
        if kind == "router":
            mem = [("router_w", TOPK, "f32"), ("router_ids", TOPK, "i32")]
            n_rows = M
        elif kind == "up":
            mem = [("up", int(st.get("N", g.get("N_up", 0))), "bf16")]
            n_rows = M * TOPK
        elif kind == "act":
            mem = [("act", int(st.get("I", g.get("I", 0))), "bf16")]
            n_rows = M * TOPK
        elif kind in ("in_quant", "act_quant"):
            K, G = int(st.get("K", st.get("I", 0))), int(st.get("G", 0) or 1)
            pre = "in" if kind == "in_quant" else "act"
            mem = [(f"{pre}_q", K, "w8"), (f"{pre}_scale", K // G, "f32")]
            n_rows = M if kind == "in_quant" else M * TOPK
        elif kind == "down":
            mem = [("down", int(st.get("N", g.get("H", 0))), "bf16")]
            n_rows = M * TOPK
        else:                                                                    # "sum"
            mem = [("sum", int(st.get("H", g.get("H", 0))), "bf16")]
            n_rows = M
        if err is None and not (0 <= ord_ < n_rows):
            err = f"{fam} row ordinal {ord_} outside the {kind} plane of {mod!r} step {s} ({n_rows} rows = {'M' if n_rows == M else 'M*TOPK'})"
        if err is None and name_row is not None and int(name_row) != ord_:
            err = f"{fam} row ordinal {ord_} != its Program name index [{name_row}] ({mod!r} step {s}, {kind} plane)"
        if err is None and any(int(m[1]) <= 0 for m in mem):
            err = f"{fam} {kind} plane width 0 ({mod!r} step {s}: statics {st}, geometry {g})"
        if err is None:
            err = self._plane_dataflow_error(i, kind, ord_, TOPK, mod, fp8=fam.startswith("MoeExpertFp8Gemm"))
        # `plane` = the plane the row PRODUCES: the first member's name (the router row produces two: router_w + router_ids)
        d = {"family": fam, "statics": st, "module": mod, "op_path": f"moe/L{layer}", "hint": hint, "row": ord_, "name_row": name_row,
             "members": ([] if err else mem), "step": s, "plane": mem[0][0], "plane_kind": kind, "moe_module": mod}
        if err:
            d["plane_error"] = err
        return d

    def _plane_dataflow_error(self, i: int, kind: str, p: int, TOPK: int, mod: str, fp8: bool = False) -> str | None:
        """The pair-major reading of a plane row's ordinal (p = t*TOPK + k) cross-checked against the Program's OWN dataflow: the up row p
        reads router row t's id slot k (`['n', router, TOPK + k, TOPK + k + 1]`), the act row p reads up row p, the down row p reads act
        row p + router row t's id slot k + weight slot k, the sum row t reads down rows t*TOPK .. t*TOPK + TOPK - 1 in slot order.  Block-FP8
        expert GEMMs read a group quant first, (xq, sx): the up row p reads the input quant row t (when the runner quantizes its input) and
        the id slot as its third operand, the down row p reads the activation quant row p and the id and weight slots as its third and fourth;
        the activation quant row p reads act row p (up row p when fused with the SiLU).  A Program whose MoE rows are not in that order (or not
        of that shape) is named here -- the replay never reads a plane word of another pair.
        (Measured on the OLMoE B1 256/32 request Program: 119,392 MoE rows, 0 inconsistencies.)"""
        r = self.by_i[i]
        args = r.get("args") or []

        def prod(gi: int, si: int = 0) -> tuple[str | None, int, int, int, int] | None:
            if gi >= len(args) or si >= len(args[gi]) or args[gi][si][0] != "n":
                return None
            j, lo, hi = int(args[gi][si][1]), int(args[gi][si][2]), int(args[gi][si][3])
            pr = self.by_i.get(j)
            if pr is None:
                return None
            pm = RM.split_name(pr.get("name") or "")[0]
            return (self._moe_kind.get(j) if pm == mod else None), int(self._moe_ord.get(j, -1)), lo, hi, j

        def quant_pair(qkind: str, row: int) -> str | None:
            q, s = prod(0), prod(1)
            st = RM.spec_statics(self.by_i[q[4]]["spec"]) if q else {}
            width = int(st.get("K", st.get("I", -1)))
            if q is None or q[0] != qkind or q[1] != row or q[2] != 0 or s is None or s[4] != q[4] or s[2] != width:
                return f"{kind} row {p} does not read {qkind} row {row}'s (q, scale) (Program dataflow {q}, {s}): not pair-major"
            return None

        t, k = (p // TOPK, p % TOPK) if TOPK > 0 else (p, 0)
        if kind == "up":
            x = prod(0)
            if fp8 and x is not None and x[0] is not None and (err := quant_pair("in_quant", t)):
                return err
            e = prod(2 if fp8 else 1)
            if e is None or e[0] != "router" or e[1] != t or (e[2], e[3]) != (TOPK + k, TOPK + k + 1):
                return f"up row {p} does not read router row {t} id slot {k} (Program dataflow {e}): not pair-major"
        elif kind == "act":
            u = prod(0)
            if u is None or u[0] != "up" or u[1] != p or u[2] != 0:
                return f"act row {p} does not read up row {p} (Program dataflow {u}): not pair-major"
        elif kind == "act_quant":
            u = prod(0)
            want = "up" if RM.spec_family(r["spec"]).startswith(RM.FUSED_ACT_QUANT) else "act"
            if u is None or u[0] != want or u[1] != p or u[2] != 0:
                return f"act_quant row {p} does not read {want} row {p} (Program dataflow {u}): not pair-major"
        elif kind == "down":
            if fp8 and (err := quant_pair("act_quant", p)):
                return err
            a, e, g = (None, prod(2), prod(3)) if fp8 else (prod(0), prod(1), prod(2))
            if not fp8 and (a is None or a[0] != "act" or a[1] != p or a[2] != 0):
                return f"down row {p} does not read act row {p} (Program dataflow {a}): not pair-major"
            if e is None or e[0] != "router" or e[1] != t or (e[2], e[3]) != (TOPK + k, TOPK + k + 1):
                return f"down row {p} does not read router row {t} id slot {k} (Program dataflow {e}): not pair-major"
            if g is None or g[0] != "router" or g[1] != t or (g[2], g[3]) != (k, k + 1):
                return f"down row {p} does not read router row {t} weight slot {k} (Program dataflow {g}): not pair-major"
        elif kind == "sum":
            sl = args[0] if args else []
            if len(sl) != TOPK:
                return f"sum row {p} reads {len(sl)} slices, not TOPK = {TOPK}"
            for kk in range(TOPK):
                y = prod(0, kk)
                if y is None or y[0] != "down" or y[1] != p * TOPK + kk or y[2] != 0:
                    return f"sum row {p} slot {kk} does not read down row {p * TOPK + kk} (Program dataflow {y}): not pair-major"
        return None

    def member_at(self, j: int, lo: int, hi: int) -> tuple[dict, str, int, int, str] | None:
        """Producer row j's output slice [lo, hi) -> (desc, member, member-local lo, hi, dtype); None if it is not inside one member."""
        pr = self.by_i.get(int(j))
        if pr is None:
            return None
        d = self.describe(pr)
        if not d:
            return None
        off = 0
        for m in d["members"]:
            name, width, dtype = m[0], int(m[1]), str(m[2])
            if off <= lo and hi <= off + width:
                return d, name, lo - off, hi - off, dtype
            off += width
        return None

    def weight_slice(self, lo: int, hi: int) -> tuple[str, int, int, list[int]] | None:
        """Element slice of the `weights` struct -> (field name, field-local lo, hi, field shape); None if it spans fields."""
        if not self.fields:
            return None
        k = bisect.bisect_right(self.foff, lo) - 1
        if k < 0:
            return None
        name, count, shape, _bits = self.fields[k]
        base = self.foff[k]
        if lo < base or hi > base + count:
            return None
        return name, lo - base, hi - base, shape

    def weight_bits(self, field: str) -> int:
        """[R16 fp8] the leaf width the Program declares for a `weights` field (`Value<16>` bf16, `Value<8>` e4m3, `Value<32>` f32 block scales)."""
        for name, _count, _shape, bits in self.fields:
            if name == field:
                return int(bits) or 16
        return 16


#: the `moe_block_stream` address schema (the manifest / `binding.py` write `moe/L<layer>` as the op_path of the six
#: planes): parsed here to label the (request, step, layer) triple -- the family's own address vocabulary, not a module-path rule
_MOE_LAYER_RE = re.compile(r"^moe/L(\d+)$")
