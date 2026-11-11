"""A MoE layer's expert GEMMs cut per expert in a served window (`serving_templates.EXPERTS`, ``FUSED``, ``SLOTS`` and ``READ``):
a (token, slot) pair's GEMM over its expert's rows alone, a block of n rows at a time, each block's rows and its G-block's scale
row read at the pair's expert from a table registered one expert a row, so no proof block holds another expert's weights.  Four
layouts:

* ``slots``: a router row cut into slots (`moe.MoeRouterSigmoidBiasSlots`), a unit a token: ``in0`` its logits and ``in1`` its
  bias -> ``o<k>`` slot k's weight and ``o<TOPK + k>`` its expert id, the row's committed members;
* ``read``: a table read (`gates.embedding.EmbeddingRead_v1{V=E, H}`), a unit a block of a pair: ``in0`` the pair's expert (the
  read's key), ``in1`` and ``in2`` the table's root and domain id, ``in3``, ``in4`` and ``in5`` the table's row at the expert, its
  salt and its path -> ``o0`` the row and ``o1`` its verdict;
* ``expert``: a block GEMM (`fp8_moe.MoeExpertFp8BlockGemm(W)`), a unit a block of a pair: ``xq``, ``sx``, (``g``,) then ``w`` and
  ``ws``, the block's rows of the pair's expert and their G-block's scale row -> ``o0`` the pair's output words of those rows;
* ``fused``: a block GEMM that reads its block's rows itself (`boolean_fp8_moe.MoeExpertFp8ReadGemm(W)_v1`), a unit a block of a
  pair: a read's six ports (``in0`` .. ``in5``), then ``xq``, ``sx``, (``g``,) and ``ws`` -> ``o0`` the pair's output words of
  the block's rows and ``o1`` the read's verdict.  It stands for a block GEMM and the read of its block's rows; a pair row's
  blocks are fused or apart, never both.

A table is one block of a pair row's weight stack, one row an expert (`table_words`): row e of block b is expert e's rows
[b n, b n + n) as their e4m3 bytes in u16 words (a read of rows, ``H = n K / 2``), or its scale row b's f32 words (a read of scale
rows, ``H = 2 K / G``).  Its parameter is ``weights/<label>@<H>/b<b>/table`` (`table_param`), a registered parameter of its own
that the record declares as a table (`interface.Table`), with its root and domain id table reads of ``in1`` and ``in2`` and its
row, salt and path an opening at the key ``in0`` that the verdict ``o1`` holds (`declarations`).  The committer commits each
table the window reads from the weights it serves, under its key at the program and nonce its registration names
(`records.expert_tables`, `committer`, `commit_tables`), so the window's root of a table is the registered one exactly when it
serves the registered rows, and its openings are of that root.

A pair is a pair row; its layer is the op_path of the router row its expert id reads, its slot k the id's place among the
router's ids, and its token that router row.  Units go layer by layer, the layers in the order they first appear: a read's
table-major, ((l B + b) TOPK + k) T + t, so a table's declarations are three and a member's key reads one router slot; a block
GEMM's slot-major, ((l TOPK + k) T + t) B + b, so a pair's blocks are consecutive and a member's routed weight reads one router
slot.  A member holds one table's units of one slot (a read) or one layer's of one slot (a block GEMM), at most its chunk cap
(`cut`).  The wires (`wire`): a read's ``in0`` is its token's router unit's ``o<TOPK + k>``, a block GEMM's ``g`` its ``o<k>``,
its ``w`` the ``o0`` of the read of its block's rows and its ``ws`` that of the read of its G-block's scale row.  A fused block's
units go as a read's and are cut as a read's; it is wired as a read (its key) and as a block GEMM (``g`` and ``ws``), and it
declares its table as a read does.  A wire whose writer's template is not in the window is not made, so its port is free.  A
block GEMM's output is held to the served words by the twin of the pair row (`fp8_moe_rows.moe_expert_fp8_block_row`)."""
from __future__ import annotations

import bisect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from verity.commitments.windows import interface as IF
from verity.commitments.windows import rows as CS
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline import slice_rows as SL
from verity_vllm.pipeline.serving_templates import EXPERTS, template
from verity_vllm.program.compact import runs_from_file_items
from verity_vllm.records import expert_tables as XT

Row = tuple[str, Any, dict, dict]
_LAYER = re.compile(r"\.layers\.(\d+)\.")
#: the per-expert cut's layouts (`serving_templates.template`)
LAYOUTS = ("slots", "read", "expert", "fused")
#: a pair row family's arguments: its expert id, its routed weight (None: unweighted), its weight stack and its scale stack
ARGS = {"MoeExpertFp8Gemm_v1": (2, None, 3, 4), "MoeExpertFp8GemmW_v1": (2, 3, 4, 5)}
#: the parts of the cut that read a table of a block's rows, the rest of a read's of scale rows
ROWS = ("w", "fused")


def table_param(label: str, words: int, block: int) -> str:
    """The registered parameter of block ``block`` of weight ``label`` as a table of rows of ``words`` u16 words, one an expert:
    ``weights/<label>@<words>/b<block>/table``."""
    return f"{WR.weight_param(label, words)}/b{int(block)}/table"


def table_words(stack: np.ndarray, what: str, E: int, N: int, K: int, G: int, n: int, block: int) -> np.ndarray:
    """Block ``block``'s table of a pair row's stack ``stack`` (its elements as read: the e4m3 bytes of ``(E, N, K)`` weights for
    ``what`` "w", the f32 words of ``(E, ceil(N / G), K / G)`` scales for "ws"), as u16 words ``(E, H)``: row e expert e's rows
    [block n, block n + n), or its scale row ``block``.  The elements are taken as their bytes, never converted."""
    want = np.dtype(np.uint8 if what == "w" else "<u4")
    a = np.ascontiguousarray(stack)
    if a.dtype.itemsize != want.itemsize:
        raise ValueError(f"a {what} table's elements are {want.itemsize}-byte words, not {a.dtype}")
    a = a.view(want)
    a = a.reshape(E, N, K)[:, block * n:(block + 1) * n, :] if what == "w" else a.reshape(E, -(-N // G), K // G)[:, block, :]
    return np.ascontiguousarray(a).reshape(E, -1).view("<u2")


def kind(t: str, statics: Mapping[str, Any]) -> tuple[str, int]:
    """Template ``t``'s part of the cut on a pair row of ``statics``: ``("slots", 1)``, ``("gemm", n)`` a block GEMM of n rows,
    ``("fused", n)`` a fused block of n rows, ``("w", n)`` the read of a block of n rows, or ``("ws", G)`` the read of a
    G-block's scale row."""
    from verity_vllm.query.program_view import spec_statics

    tp = template(t)
    if tp.layout not in LAYOUTS:
        raise ValueError(f"{t}: a {tp.layout} template is not of the per-expert cut")
    if tp.layout == "slots":
        return "slots", 1
    S = spec_statics(tp.word)
    if tp.layout in ("expert", "fused"):
        return "gemm" if tp.layout == "expert" else "fused", int(S["N"])
    K, G, H = int(statics["K"]), int(statics["G"]), int(S["H"])
    if H == 2 * K // G:
        return "ws", G
    if 2 * H % K == 0 and G % (2 * H // K) == 0:
        return "w", 2 * H // K
    raise ValueError(f"{t}: {H} words a row are neither a block of rows of K = {K} nor a scale row of K / G = {K // G}")


def blocks(t: str, statics: Mapping[str, Any]) -> int:
    """Template ``t``'s units in a row of ``statics``: one a router row, and one a block of a pair (N / n for a block GEMM or the
    read of its rows, ceil(N / G) for the read of its scale rows)."""
    what, n = kind(t, statics)
    return 1 if what == "slots" else -(-int(statics["N"]) // n)


def _base(spec: str) -> str:
    from verity_vllm.query.program_view import spec_base

    return spec_base(spec)


def _route(x: Row) -> tuple[str, int, int]:
    """Pair row ``x``'s router: the router row's op_path, the slot and the router row's index, read off the arguments its expert id
    (and routed weight) read; refused unless one router row's slot k gives both."""
    _rid, pi, r, _d = x
    e, g, _w, _ws = ARGS[_base(r["spec"])]
    got = []
    for j in (e,) if g is None else (e, g):
        runs = list(runs_from_file_items(r["args"][j]))
        if len(runs) != 1 or runs[0][0] != "n" or int(runs[0][3]) - int(runs[0][2]) != 1:
            raise ValueError(f"row {r['i']} ({_base(r['spec'])}): its argument {j} is not one element of a row's output")
        m = pi.member_at(int(runs[0][1]), int(runs[0][2]), int(runs[0][3]))
        if m is None:
            raise ValueError(f"row {r['i']}: its argument {j} is not inside one committed member of its router row")
        got.append((m[0]["op_path"], int(m[2]), int(runs[0][1])))
    if len(set(got)) != 1:
        raise ValueError(f"row {r['i']}: its expert id and routed weight are {got}, not one router row's slot")
    return got[0]


def _label(x: Row, g: int) -> str:
    """The weight field pair row ``x``'s argument ``g`` reads whole."""
    from verity_vllm.pipeline import weight_rows as WT

    got = WT._weight(x[1], x[2]["args"][g])
    if got is None or got[1] != 0 or got[2] != got[3]:
        raise ValueError(f"row {x[2]['i']}: its argument {g} is not one whole weight of the Program's weights")
    return got[0]


class Plan:
    """The per-expert cut over a window's templates ``tmpls`` and their ``rows`` (`serving_rows.population`'s): each template's
    units in the module's order (``units``: each ``(row, block)``), where its members are cut (``runs``), each unit's place by its
    pair row and block (``at``), each router row's unit (``router``) and each pair row's layer, slot and token (``pair``)."""

    def __init__(self, tmpls: Sequence[str], rows: Sequence[list[Row]]) -> None:
        cut = [template(t).layout in LAYOUTS for t in tmpls]
        self.of = {t: rs for t, rs, c in zip(tmpls, rows, cut, strict=True) if c} if any(cut) else {}
        req: dict[str, int] = {}
        for rs in rows:
            for x in rs:
                req.setdefault(x[0], len(req))
        slots = [t for t in self.of if template(t).layout == "slots"]
        if len(slots) > 1:
            raise ValueError(f"router slot templates {slots}: one template's units are a window's router rows")
        self.slots: str | None = slots[0] if slots else None
        self.layers: dict[str, int] = {}
        self.router: dict[tuple[str, int], int] = {}
        for u, (rid, _pi, r, d) in enumerate(self.of.get(self.slots, ()) if self.slots else ()):
            self.router[(rid, int(r["i"]))] = u
            self.layers.setdefault(d["op_path"], len(self.layers))
        self.what: dict[str, tuple[str, int]] = {}
        self.family: dict[str, str] = {}
        self.pair: dict[tuple[str, int], tuple[int, int, tuple[int, int]]] = {}
        held: dict[tuple[tuple[str, int], str], str] = {}
        for t, rs in self.of.items():
            if t == self.slots:
                continue
            fams = {_base(x[2]["spec"]) for x in rs}
            if len(fams) != 1 or not fams <= set(EXPERTS):
                raise ValueError(f"{t}: rows of {sorted(fams)}; a template of the per-expert cut reads rows of one family of "
                                 f"{sorted(EXPERTS)} (bind its Definition at statics no other family's rows share)")
            self.family[t], self.what[t] = fams.pop(), kind(t, rs[0][3]["statics"])
            for x in rs:
                key = (x[0], int(x[2]["i"]))
                if (other := held.setdefault((key, self.what[t][0]), t)) != t:
                    raise ValueError(f"row {key[1]} of request {key[0]} holds units of {other} and {t}: one template of each "
                                     "part of the cut reads a pair row")
                if key not in self.pair:
                    op, k, j = _route(x)
                    self.pair[key] = (self.layers.setdefault(op, len(self.layers)), k, (req[x[0]], j))
        for (key, what), t in held.items():
            if what == "fused" and (other := held.get((key, "gemm")) or held.get((key, "w"))):
                raise ValueError(f"row {key[1]} of request {key[0]} is cut both fused ({t}) and apart ({other}): a pair row's blocks "
                                 "are fused blocks, or block GEMMs and the reads of their rows, never both")
        self.units: dict[str, list[tuple[int, int]]] = {}
        self.runs: dict[str, list[int]] = {}
        self.at: dict[str, dict[tuple[tuple[str, int], int], int]] = {}
        for t in self.what:
            self._order(t)

    def _order(self, t: str) -> None:
        rs, gemm = self.of[t], self.what[t][0] == "gemm"
        keys, seen = [], set()
        for p, x in enumerate(rs):
            layer, k, tok = self.pair[(x[0], int(x[2]["i"]))]
            if (layer, k, tok) in seen:
                raise ValueError(f"{t}: two pair rows of slot {k} of one token's router row {tok}")
            seen.add((layer, k, tok))
            keys += [((layer, k, tok, b) if gemm else (layer, b, k, tok), p, b) for b in range(blocks(t, x[3]["statics"]))]
        keys.sort()
        self.units[t] = [(p, b) for _key, p, b in keys]
        run = [key[:2] if gemm else key[:3] for key, _p, _b in keys]
        self.runs[t] = [u for u in range(1, len(run)) if run[u] != run[u - 1]]
        self.at[t] = {((rs[p][0], int(rs[p][2]["i"])), b): u for u, (p, b) in enumerate(self.units[t])}

    def key(self, t: str, u: int) -> tuple[str, int]:
        """Template ``t``'s unit ``u``'s pair row, ``(request, row index)``."""
        x = self.of[t][self.units[t][u][0]]
        return x[0], int(x[2]["i"])

    def reads(self, t: str) -> tuple[str | None, str | None]:
        """Block GEMM ``t``'s read templates in the window, of its blocks' rows and of their scale rows (None where absent); a
        fused block reads only the latter."""
        n, fam = self.what[t][1], self.family[t]
        got = {w: [s for s, (what, m) in self.what.items() if self.family[s] == fam and what == w and (w == "ws" or m == n)]
               for w in ("w", "ws")}
        return got["w"][0] if got["w"] else None, got["ws"][0] if got["ws"] else None

    def tables(self, t: str) -> list[tuple[int, int, int, str, int]]:
        """Read or fused template ``t``'s tables in its units' order, each ``(first unit, units, block, parameter, leaves)``: one
        a layer and block of its rows, its units every slot's and token's."""
        rs, (what, n) = self.of[t], self.what[t]
        S = rs[0][3]["statics"]
        K, G, E = int(S["K"]), int(S["G"]), int(S["E"])
        H = n * K // 2 if what in ROWS else 2 * K // G
        arg = ARGS[self.family[t]][2 if what in ROWS else 3]
        labels: dict[int, str] = {}
        out: list[tuple[int, int, int, str, int]] = []
        for u, (p, b) in enumerate(self.units[t]):
            layer = self.pair[self.key(t, u)][0]
            if layer not in labels:
                labels[layer] = _label(rs[p], arg)
            if not out or (out[-1][2], out[-1][3]) != (b, table_param(labels[layer], H, b)):
                out.append((u, 0, b, table_param(labels[layer], H, b), H * E))
            out[-1] = (out[-1][0], out[-1][1] + 1, *out[-1][2:])
        if len({o[3] for o in out}) != len(out):
            raise ValueError(f"{t}: two layers' rows read one weight")
        return out


def _bases(ms: Sequence[CS.Member]) -> dict[str, int]:
    base: dict[str, int] = {}
    for m in ms:
        base[m.template] = min(base.get(m.template, m.base), m.base)
    return base


def cut(ms: Sequence[CS.Member], plan: Plan) -> list[CS.Member]:
    """``ms`` (`serving_rows.members`') with each template of the cut but the router's cut again where its units' table or slot
    changes (a read: one table's units of one slot; a block GEMM: one layer's units of one slot) and at its chunk cap (its first
    chunk's size), named ``<descriptor id>/c<j>`` in order."""
    out: list[CS.Member] = []
    for t in dict.fromkeys(m.template for m in ms):
        mine = [m for m in ms if m.template == t]
        if t not in plan.units:
            out += mine
            continue
        first, n, lo, pieces = mine[0], sum(m.n for m in mine), 0, []
        for hi in [*plan.runs[t], n]:
            pieces += [(a, min(a + first.n, hi)) for a in range(lo, hi, first.n)]
            lo = hi
        out += [replace(first, base=first.base + a, n=b - a, name=f"{t}/c{j}") for j, (a, b) in enumerate(pieces)]
    return out


def _port(t: str, name: str) -> str:
    return f"in{[nm for nm, _ty in template(t).fn.signature()[0]].index(name)}"


def wire(ms: Sequence[CS.Member], plan: Plan) -> list[CS.Member]:
    """``ms`` (`cut`'s) with each read's key wired to its token's router unit's slot id, and each block GEMM's routed weight to its
    slot weight and its ``w`` and ``ws`` to the reads of its block's rows and scale row; a fused block's key, routed weight and
    ``ws`` as theirs (module docstring).  A wire whose writer's template is not in the window is not made."""
    from verity_vllm.query.program_view import spec_statics

    base = _bases(ms)
    topk = int(spec_statics(template(plan.slots).word)["TOPK"]) if plan.slots else 0
    out: list[CS.Member] = []
    for m in ms:
        t = m.template
        if t not in plan.units:
            out.append(m)
            continue
        lo = m.base - base[t]
        keys = [plan.key(t, u) for u in range(lo, lo + m.n)]
        wired, (what, n) = list(m.wired), plan.what[t]
        k = plan.pair[keys[0]][1]
        if plan.slots is not None:
            if gone := [key for key in keys if (key[0], plan.pair[key][2][1]) not in plan.router]:
                raise ValueError(f"{t}: row {gone[0][1]} of request {gone[0][0]} is routed by a row outside {plan.slots}'s units: "
                                 "a window cuts a token's router row and its pair rows together")
            routers = tuple(base[plan.slots] + plan.router[(key[0], plan.pair[key][2][1])] for key in keys)
            if what in ("w", "ws", "fused"):
                wired.append(("in0", f"o{topk + k}", routers))
            if what in ("gemm", "fused") and ARGS[plan.family[t]][1] is not None:
                wired.append((_port(t, "g"), f"o{k}", routers))
        if what in ("gemm", "fused"):
            G = int(plan.of[t][0][3]["statics"]["G"])
            bs = [b for _p, b in plan.units[t][lo:lo + m.n]]
            for port, src, at in zip(("w", "ws"), plan.reads(t), (bs, [b * n // G for b in bs]), strict=True):
                if src is not None and (what == "gemm" or port == "ws"):
                    wired.append((_port(t, port), "o0", tuple(base[src] + plan.at[src][(key, a)] for key, a in zip(keys, at, strict=True))))
        out.append(replace(m, wired=tuple(wired)))
    return out


def declarations(plan: Plan) -> IF.Registered | None:
    """The window's tables, table reads and openings (a listed record's ``registered``, `interface.listed_reads`): each read or
    fused template's tables, each with its root and domain id read into ``in1`` and ``in2`` and its row, salt and path (``in3``,
    ``in4``, ``in5``) opened at the key ``in0`` by the verdict ``o1``, over its units.  A table two templates read (one
    projection's fused blocks on two kernels) is declared once, each template's reads and openings of it its own.  None where
    the window reads none."""
    params: dict[str, int] = {}
    tables, roots, openings = [], [], []
    for t, (what, _n) in plan.what.items():
        if what == "gemm":
            continue
        E = int(plan.of[t][0][3]["statics"]["E"])
        for at, n, _b, param, leaves in plan.tables(t):
            if param not in params:
                params[param] = leaves
                tables.append(IF.Table(param, param, 0, leaves // E, E))
            elif params[param] != leaves:
                raise ValueError(f"{t}: table {param} of {leaves} leaves, another template's of {params[param]}")
            roots += [IF.TableRead(t, "in1", param, "root", at, n), IF.TableRead(t, "in2", param, "domain", at, n)]
            openings.append(IF.Opening(t, param, "in0", ("in3", "in4", "in5"), "o1", at, n))
    return IF.Registered(params, tuple(tables), (), tuple(roots), tuple(openings)) if tables else None


def _stack(slices: Callable, x: Row, store: Any, weights: Callable, arrive: int, g: int, bits: int) -> np.ndarray:
    _rid, pi, r, _d = x
    return SL.elements(slices(pi, store, weights, x[0], arrive, r["args"][g]), bits, f"row {r['i']} argument {g}")


def commit_tables(plan: Plan, store: Any, weights: Callable, arrivals: Mapping[str, int] | None, slices: Callable,
                  committer: Mapping[str, Any]) -> dict[str, Any]:
    """Each table the window's reads read, committed from the weights served (each layer's stack as ``slices`` reads it,
    `serving_rows._slices`; `table_words`) as core's ``registered.Registered`` of ``hm96-sha512/row-seg/v1`` rows with its rows
    held, under ``committer``'s secret ``key`` at the ``program`` and ``nonces`` its registration names (`committer`), by
    parameter.  Refused for a table the registration names no nonce for."""
    from verity.protocols.spot.verifier.one_stage import registered as G
    from verity_vllm.protocols.pouw.h1 import registered as H1R

    out: dict[str, Any] = {}
    nonces, key = committer["nonces"], bytes(committer["key"])
    for t, (what, n) in plan.what.items():
        if what == "gemm":
            continue
        rs, of = plan.of[t], "w" if what in ROWS else "ws"
        S = rs[0][3]["statics"]
        E, N, K, G_ = (int(S[x]) for x in ("E", "N", "K", "G"))
        arg, held = ARGS[plan.family[t]][2 if of == "w" else 3], None
        for at, _n, b, param, _leaves in plan.tables(t):
            if param in out:
                continue
            if param not in nonces:
                raise ValueError(f"{param}: the committer's registration names no nonce for the table (records.expert_tables)")
            x = rs[plan.units[t][at][0]]
            g = json.dumps(x[2]["args"][arg])
            if held is None or held[0] != g:
                held = (g, _stack(slices, x, store, weights, int((arrivals or {}).get(x[0], 0)), arg, 8 if of == "w" else 32))
            out[param] = H1R.commit(committer["program"], param, table_words(held[1], of, E, N, K, G_, n, b), key,
                                    int(nonces[param]), keep_rows=True, schema=G.SCHEMA_SEG)
    return out


def committer(reg: XT.ExpertTables, key: bytes) -> dict[str, Any]:
    """A window's committer's tables (`serving_rows.after_commit`'s ``tables``) from its registration ``reg``: its secret
    ``key``, the rank's program and each table's nonce, as the registration names them."""
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("a table's salts are drawn under the committer's 32-byte key")
    return {"key": key, "program": reg.program, "nonces": dict(reg.nonces)}


def of_request(spec: Mapping[str, Any] | None, rank: int = 0) -> dict[str, Any] | None:
    """`committer` of a serving-rows request's ``tables``, ``{"key": FILE, "registration": FILE}`` (``{rank}`` in the
    registration's path its rank's), or None without one."""
    if not spec:
        return None
    if not spec.get("key"):
        raise ValueError("a tables registration is read under its committer's key file (--tables-key)")
    where = str(spec["registration"]).format(rank=rank)
    reg = XT.ExpertTables.of(json.loads(Path(where).read_text()), where)
    if reg.rank != int(rank):
        raise ValueError(f"{where}: the tables registration of rank {reg.rank}, not of rank {rank}")
    return committer(reg, Path(spec["key"]).read_bytes())


def _digest_words(b: bytes) -> np.ndarray:
    """Bytes, a whole number of 64-bit words, as a Boolean port's u16 words (`interface.digest_words`, each 8 bytes big-endian)."""
    return np.frombuffer(bytes(b), ">u8").astype("<u8").view("<u2")


def _entry(lo: int, hi: int, x: Row, arrive: int, members: str) -> list[Any]:
    _rid, _pi, r, d = x
    layer = _LAYER.search(d["op_path"])
    return [lo, hi, x[0], arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"], members, int(d["row"]),
            int(r["i"])]


class Reader:
    """The words and index entries of the cut's members (`read`), each pair row's operands read once, a layer's weight stacks
    while its rows are read; ``slices`` and ``packed`` are `serving_rows._slices` and ``_packed``, ``tables`` `commit_tables`'."""

    def __init__(self, plan: Plan, store: Any, weights: Callable, arrivals: Mapping[str, int] | None, slices: Callable,
                 packed: Callable, tables: Mapping[str, Any] | None) -> None:
        self.plan, self.store, self.weights, self.arrivals = plan, store, weights, arrivals or {}
        self.slices, self.packed, self.tables = slices, packed, tables or {}
        self.stacks: dict[str, np.ndarray] = {}
        self.pairs: dict[tuple[str, int], dict[str, Any]] = {}

    def _elements(self, x: Row, g: int, bits: int) -> np.ndarray:
        return _stack(self.slices, x, self.store, self.weights, int(self.arrivals.get(x[0], 0)), g, bits)

    def _held(self, x: Row, g: int, bits: int) -> np.ndarray:
        key = json.dumps(x[2]["args"][g])
        if key not in self.stacks:
            if len(self.stacks) >= 4:
                self.stacks.clear()
            self.stacks[key] = self._elements(x, g, bits)
        return self.stacks[key]

    def expert(self, x: Row) -> int:
        """Pair row ``x``'s routed expert, its expert id's committed word."""
        E = int(x[3]["statics"]["E"])
        e = int(self._elements(x, ARGS[_base(x[2]["spec"])][0], 32).view("<i4")[0])
        if not 0 <= e < E:
            raise ValueError(f"row {x[2]['i']} ({x[3]['op_path']}): routed expert {e} is not one of its {E}")
        return e

    def pair(self, x: Row) -> dict[str, Any]:
        """Pair row ``x``'s operands: its activation, scales, routed weight, expert, that expert's weight and scale rows and the
        row's served words, which the twin of the row on them must be."""
        from verity.definitions.numerics.fp8_moe_rows import TWIN_DOT_OF, moe_expert_fp8_block_row
        from verity_vllm.query.program_view import spec_statics

        rid, _pi, r, d = x
        if (rid, int(r["i"])) in self.pairs:
            return self.pairs[(rid, int(r["i"]))]
        arrive, what = int(self.arrivals.get(rid, 0)), f"row {r['i']} ({d['op_path']})"
        S, T = d["statics"], spec_statics(r["spec"])
        E, N, K, G = (int(S[k]) for k in ("E", "N", "K", "G"))
        _e, ig, iw, iws = ARGS[_base(r["spec"])]
        e = self.expert(x)
        W, WS = self._held(x, iw, 8).reshape(E, N, K), self._held(x, iws, 32).reshape(E, -(-N // G), K // G)
        xq, sx = self._elements(x, 0, 8), self._elements(x, 1, 32)
        g = None if ig is None else int(self._elements(x, ig, 32)[0])
        (member, width, _dt), = d["members"]
        y, why = self.store.words(rid, arrive + int(d["step"]), d["op_path"], member, int(d["row"]) * int(width),
                                  (int(d["row"]) + 1) * int(width))
        if y is None:
            raise LookupError(f"row {r['i']} output {member}: {why}")
        y = SL.elements([np.asarray(y)], 16, f"{what} output {member}")
        twin, _nonfinite = moe_expert_fp8_block_row(xq, sx, W[e], WS[e], G=G, KB=int(T["KB"]), swap=T["SWAP"] is True, g=g,
                                                    dot=TWIN_DOT_OF[str(T["DOT"])])
        if y.size != N or not np.array_equal(twin, y):
            raise ValueError(f"{what}: the twin of the row on its expert {e}'s rows is not the row's served {member} words")
        self.pairs = {(rid, int(r["i"])): {"xq": xq, "sx": sx, "g": g, "e": e, "w": W[e], "ws": WS[e], "y": y, "arrive": arrive,
                                            "member": str(member)}}
        return self.pairs[(rid, int(r["i"]))]

    def read(self, ms: Sequence[CS.Member]) -> list[tuple[dict[str, Any], list[list[Any]]]]:
        """Members ``ms``, every member of one template of the cut, each its words (every port its elements' bytes in u16 words)
        and index entries: a router row's and a pair's units one entry each, a read's or a fused block's unit its own.  A read's
        or a fused block's opening is of its table among ``tables``, and its root and domain id the table's, which
        `weight_rows.by_reference` gives by reference; refused for a read whose table is missing, and for a fused block whose
        table's row at the expert is not its pair row's block of rows."""
        from verity_vllm.query.program_view import spec_statics

        t, plan = ms[0].template, self.plan
        rs, names, n, b0 = plan.of[t], [q for q, _ in ms[0].rows + ms[0].outs], sum(m.n for m in ms), ms[0].base
        out = {q: np.empty((n, w), dtype=np.uint16) for q, w in ms[0].rows + ms[0].outs}
        entries: list[list[Any]] = []

        def put(u: int, words: Sequence[np.ndarray], what: str) -> None:
            for q, a in zip(names, words, strict=True):
                a = np.asarray(a, dtype=np.uint16).reshape(-1)
                if a.size != out[q].shape[1]:
                    raise ValueError(f"{t} {what}: its {q} is {a.size} words, the member's {out[q].shape[1]}")
                out[q][u] = a

        if t == plan.slots:
            TOPK = int(spec_statics(template(t).word)["TOPK"])
            for u, x in enumerate(rs):
                rid, pi, r, d = x
                arrive = int(self.arrivals.get(rid, 0))
                ins = [self.packed(self.slices(pi, self.store, self.weights, rid, arrive, r["args"][g], bits=32), 32,
                                   f"row {r['i']} argument {g}") for g in range(2)]
                outs = []
                for member, width, dt in d["members"]:
                    w, why = self.store.words(rid, arrive + int(d["step"]), d["op_path"], member, int(d["row"]) * int(width),
                                              (int(d["row"]) + 1) * int(width))
                    if w is None:
                        raise LookupError(f"row {r['i']} output {member}: {why}")
                    outs.append(self.packed([w], 32, f"row {r['i']} output {member}", dt).reshape(-1, 2))
                words = np.concatenate(outs)
                if len(words) != 2 * TOPK:
                    raise ValueError(f"row {r['i']}: {len(words)} outputs, not its router's {TOPK} weights and {TOPK} ids")
                put(u, [*ins, *words], f"row {r['i']}")
                entries.append(_entry(b0 + u, b0 + u + 1, x, arrive, ",".join(str(m[0]) for m in d["members"])))
        elif plan.what[t][0] == "gemm":
            nb, B = plan.what[t][1], blocks(t, rs[0][3]["statics"])
            G = int(rs[0][3]["statics"]["G"])
            routed = ARGS[plan.family[t]][1] is not None
            for u, (p, b) in enumerate(plan.units[t]):
                got = self.pair(rs[p])
                if b == 0:
                    entries.append(_entry(b0 + u, b0 + u + B, rs[p], got["arrive"], got["member"]))
                put(u, [got["xq"].view("<u2"), got["sx"].astype("<u4").view("<u2"),
                        *([np.array([got["g"]], dtype="<u4").view("<u2")] if routed else []),
                        np.ascontiguousarray(got["w"][b * nb:(b + 1) * nb]).reshape(-1).view("<u2"),
                        np.ascontiguousarray(got["ws"][b * nb // G]).astype("<u4").view("<u2"), got["y"][b * nb:(b + 1) * nb]],
                    f"unit {u}")
        else:
            (what, nb), G = plan.what[t], int(rs[0][3]["statics"]["G"])
            routed = ARGS[plan.family[t]][1] is not None
            starts = {at: param for at, _n, _b, param, _l in plan.tables(t)}
            table, layer, pairs = None, None, {}
            for u, (p, b) in enumerate(plan.units[t]):
                if u in starts:
                    table = self.tables.get(starts[u])
                    if table is None:
                        raise ValueError(f"{t}: its table {starts[u]} is not among the committer's tables (commit_tables)")
                    root, dom = _digest_words(table.root), _digest_words(table.domain.domain_id)
                x = rs[p]
                if what != "fused":
                    e = self.expert(x)
                    row = table.rows.at([e])[0]
                    put(u, [np.array([e], dtype="<u4").view("<u2"), root, dom, row, _digest_words(table.salts[e]),
                            _digest_words(table.path(e)), row, np.ones(1, dtype=np.uint16)], f"unit {u}")
                    entries.append(_entry(b0 + u, b0 + u + 1, x, int(self.arrivals.get(x[0], 0)), str(x[3]["members"][0][0])))
                    continue
                if plan.pair[plan.key(t, u)][0] != layer:     # a layer's pairs at a time: a pair's blocks are a table apart
                    layer, pairs = plan.pair[plan.key(t, u)][0], {}
                got = pairs.get(p)
                if got is None:
                    got = pairs[p] = self.pair(x)
                e, row = got["e"], table.rows.at([got["e"]])[0]
                if not np.array_equal(row, np.ascontiguousarray(got["w"][b * nb:(b + 1) * nb]).reshape(-1).view("<u2")):
                    raise ValueError(f"{t} unit {u}: its table's row at expert {e} is not row {x[2]['i']}'s block {b} of rows")
                put(u, [np.array([e], dtype="<u4").view("<u2"), root, dom, row, _digest_words(table.salts[e]),
                        _digest_words(table.path(e)), got["xq"].view("<u2"), got["sx"].astype("<u4").view("<u2"),
                        *([np.array([got["g"]], dtype="<u4").view("<u2")] if routed else []),
                        np.ascontiguousarray(got["ws"][b * nb // G]).astype("<u4").view("<u2"), got["y"][b * nb:(b + 1) * nb],
                        np.ones(1, dtype=np.uint16)], f"unit {u}")
                entries.append(_entry(b0 + u, b0 + u + 1, x, got["arrive"], got["member"]))
        ends = [e[1] for e in entries]
        res = []
        for m in ms:
            a, j, mine = m.base - b0, bisect.bisect_right(ends, m.base), []
            while j < len(entries) and entries[j][0] < m.base + m.n:
                mine.append(entries[j])
                j += 1
            res.append(({part: {q: out[q][a:a + m.n] for q, _w in ports} for part, ports in (("rows", m.rows), ("outs", m.outs))},
                        mine))
        return res
