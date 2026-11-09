"""A request Program by part: its prologue, each layer and its epilogue, each written so it doesn't depend on where the layer sits.

`aot` compares a request Program's digest with its trace's. A model too deep to trace whole at a long window is compared by part:
the whole model's compiled Program against the compiled Programs of windows of it (layers a..a+k as a model of their own,
`window_config`), part by part, each window's Program against its trace by digest as `aot` compares any request.

`pieces(program, world)` puts each node of the program function's body in a part, 0 the prologue, L + 1 layer L, NL + 1 the
epilogue:

- a node that reads a weight or a peer Input is in the latest part among them: a weight `model.layers.{L}.*` is layer L's, the
  embedding the prologue's and the final norm and `lm_head` the epilogue's; a collective's peer Inputs are in its part (the
  embedding's all-reduce the prologue, layer L's attention and MLP all-reduces layer L, the logits gather the epilogue); a part
  earlier than the node before it starts the next launch;
- any other node that reads a node is in the part of the node before it (a layer's residual add after the next layer's fused
  all-reduce and norm is the next layer's);
- a node that reads neither (a constant, or one computed from constants and the request's Inputs alone) is shared, named by its
  content.

Each part is then the run of its nodes in each launch, in order; the launches are the passes the peer Inputs are laid out by
(refused otherwise). A node is written with references of its part's own: a node of its part by its index in the part, a node
of an earlier layer by how many layers back and its index there, the prologue's or epilogue's by its index there, a shared node by
its content, a weight by its name less `model.layers.{L}.` and how many layers back, a peer Input by its pass, its collective, how
many layers back and its peer, a request Input by name. Writing the launches' runs part by part in order and resolving each
reference inverts this, so two Programs with the same header (`Pieces.header`: the program function's id and parameters), the
same callees and equal parts are equal: comparing parts decomposes digest equality, nothing weaker.

A part of a window and the whole model's part of the same layer are equal when the layer reads the same: a window's first layer
reads the window's own embedding (the whole model's layer reads the layer before it through the fused all-reduce and norm), and its
last layer, like the model's, drops the rows no sample reads (`prune.dead_free_form`), so `compare` matches a window's layers
after its first and before its last to the model's interior ones, its last to the model's last, its prologue to the model's when the
window starts at layer 0 and its epilogue to the model's when it ends at the model's last layer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from verity.ml.circuits.codec import definition_digest, definition_entry, definitions, descriptor_id
from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided
from verity.ml.circuits.types import RecordType, field_starts

PEERS = re.compile(r"^peers_(\d+)$")
LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
PROLOGUE_WEIGHTS = ("model.embed_tokens.",)
EPILOGUE_WEIGHTS = ("model.norm.", "lm_head.")
NORM = "model.norm.weight"
WEIGHTS = "weights"


@dataclass
class Pieces:
    """A request Program by part (module docstring). `parts`: each part's digest by name (`prologue`, the layer's index,
    `epilogue`); `sizes` its nodes; `launches` the launches its body splits into; `shared` the shared nodes; `callees` each
    definition its nodes reach, descriptor id to definition digest; `header` the digest of the program function's id and parameters."""

    layers: int
    launches: int
    parts: dict[str, str]
    sizes: dict[str, int]
    shared: int
    callees: dict[str, str] = field(repr=False)
    header: str = ""

    def classes(self) -> list[list[str]]:
        """The layers grouped by equal parts, in order of first layer."""
        by: dict[str, list[str]] = {}
        for k in [str(L) for L in range(self.layers)]:
            by.setdefault(self.parts[k], []).append(k)
        return list(by.values())


def _part_name(p: int, NL: int) -> str:
    return "prologue" if p == 0 else "epilogue" if p == NL + 1 else str(p - 1)


def _flat(r):
    if not isinstance(r, Concat):
        yield r
        return
    stack = [r]
    while stack:
        x = stack.pop()
        if isinstance(x, Concat):
            stack.extend(reversed(x.parts))
        else:
            yield x


def _peer_table(params, NL: int, world: int, H: int) -> dict[int, tuple[int, str, int, int]]:
    """Parameter index of each peer Input -> (pass, collective, its part, peer), from `build_program`'s order: per pass, world - 1
    peers of each collective, the embedding's all-reduce, each layer's attention and MLP all-reduces, then the logits gather on a
    pass that samples. Every collective of a pass has its rows times H leaves; the gather's leaves are the last collective's (the
    last pass samples), which no multiple of H may be, or a gather couldn't be told from the next pass's embedding."""
    idx = [i for i, (n, _) in enumerate(params) if PEERS.match(n)]
    if not idx:
        return {}
    if world < 2 or len(idx) % (world - 1):
        raise ValueError(f"{len(idx)} peer Inputs at a world of {world}")
    groups = [idx[g:g + world - 1] for g in range(0, len(idx), world - 1)]
    leaves = [params[g[0]][1].leaves for g in groups]
    if any(params[i][1].leaves != leaves[k] for k, g in enumerate(groups) for i in g):
        raise ValueError("a collective's peer Inputs differ in size")
    gather = leaves[-1]
    if gather % H == 0:
        raise ValueError(f"the logits gather's {gather} leaves are a multiple of H={H}: a gather can't be told from an embedding")
    out: dict[int, tuple[int, str, int, int]] = {}
    g, r = 0, 0
    while g < len(groups):
        rows = leaves[g]
        if rows % H or g + 1 + 2 * NL > len(groups) or any(x != rows for x in leaves[g:g + 1 + 2 * NL]):
            raise ValueError(f"pass {r}: its collectives aren't an embedding's and {NL} layers' all-reduces of one size")
        kinds = [("embedding", 0)] + [(k, L + 1) for L in range(NL) for k in ("attention", "mlp")]
        if g + 1 + 2 * NL < len(groups) and leaves[g + 1 + 2 * NL] == gather:
            kinds.append(("gather", NL + 1))
        for (kind, part), grp in zip(kinds, groups[g:g + len(kinds)]):
            for q, i in enumerate(grp):
                out[i] = (r, kind, part, q)
        g, r = g + len(kinds), r + 1
    return out


def _weight_parts(t: RecordType) -> tuple[list[int], list[tuple[int | None, str]]]:
    """The weights record's field starts, and each field's (layer or None, name less `model.layers.{L}.`)."""
    NLs = []
    for name, _ in t.fields:
        m = LAYER.match(name)
        NLs.append((int(m[1]), m[2]) if m else (None, name))
    return list(field_starts(t)), NLs


def pieces(program, world: int) -> Pieces:
    """`program` (a request Program, `build.build_program`'s) by part, at a tensor-parallel world of `world` (module docstring)."""
    body = program.fn.body
    params = body.params
    wi = next((i for i, (n, _) in enumerate(params) if n == WEIGHTS), None)
    if wi is None or not isinstance(params[wi][1], RecordType):
        raise ValueError("the program has no weights record")
    starts, fields = _weight_parts(params[wi][1])
    layers = [L for L, _ in fields if L is not None]
    NL = max(layers) + 1 if layers else 0
    if sorted(set(layers)) != list(range(NL)):
        raise ValueError("the weights don't name layers 0..NL-1")
    E = NL + 1
    names = [n for n, _ in params[wi][1].fields]
    if NORM not in names:
        raise ValueError(f"the weights have no {NORM}")
    H = params[wi][1].fields[names.index(NORM)][1].leaves
    fpart = [L + 1 if L is not None else 0 if n.startswith(PROLOGUE_WEIGHTS) else E if n.startswith(EPILOGUE_WEIGHTS) else None
             for (L, _), n in zip(fields, names)]
    peers = _peer_table(params, NL, world, H)

    nodes = body.nodes
    lab: list[int | None] = [None] * len(nodes)
    local: list[int] = [0] * len(nodes)
    skey: dict[int, str] = {}
    count = [0] * (E + 1)
    streams = [hashlib.sha512() for _ in range(E + 1)]
    fn_key: dict[int, str] = {}
    ty_key: dict[int, str] = {}
    keep: list = []

    def fkey(fn) -> str:
        k = fn_key.get(id(fn))
        if k is None:
            keep.append(fn)
            k = fn_key[id(fn)] = descriptor_id(fn)
        return k

    def tkey(t) -> str:
        k = ty_key.get(id(t))
        if k is None:
            keep.append(t)
            k = ty_key[id(t)] = repr(t.to_json())
        return k

    def field_of(leaf: int) -> int:
        lo, hi = 0, len(starts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if starts[mid] <= leaf:
                lo = mid
            else:
                hi = mid
        return lo

    def contributes(p) -> list[int]:
        """The parts a parameter part `p` reads put its reader in."""
        if p.space != "p":
            return []
        if p.idx == wi:
            if isinstance(p, Explicit):
                leaves = [leaf for _s, _i, leaf in p.items]
            elif isinstance(p, Affine):
                leaves = [p.base, p.base + p.count - 1] if p.count else []
            else:
                leaves = [p[0][2], p[p.count - 1][2]] if p.count else []
            return [x for x in {fpart[f] for leaf in leaves for f in range(field_of(min(leaves)), field_of(max(leaves)) + 1)}
                    if x is not None] if leaves else []
        if p.idx in peers:
            return [peers[p.idx][2]]
        return []

    def port(space: str, idx: int, leaf: int, reader: int | None):
        """A reference to leaf `leaf` of node or parameter `idx`, as `reader`'s part writes it."""
        if space == "n":
            t = lab[idx]
            if t is None:
                return ("S", skey[idx], leaf)
            if t == 0:
                return ("P", local[idx], leaf)
            if t == E:
                return ("E", local[idx], leaf)
            if reader is None or reader == 0:
                return ("L@", t - 1, local[idx], leaf)
            return ("L", reader - t, local[idx], leaf)
        if idx == wi:
            f = field_of(leaf)
            L, role = fields[f]
            if L is None:
                return ("W@", role, leaf - starts[f])
            if reader is None or reader in (0, E):
                return ("W@L", L, role, leaf - starts[f])
            return ("W", reader - (L + 1), role, leaf - starts[f])
        if idx in peers:
            r, kind, part, q = peers[idx]
            return ("Q", r, kind, (reader - part) if reader is not None else None, q, leaf)
        return ("I", params[idx][0], leaf)

    def part_key(p, reader: int | None):
        """One flat reference part, as `reader`'s part writes it: a run within one node or field keeps its shape."""
        if isinstance(p, Explicit):
            return ("e", tuple(port(s, i, leaf, reader) for s, i, leaf in p.items))
        if p.count == 0:
            return ("0",)
        first, last = p[0], p[p.count - 1]
        if p.space == "p" and p.idx == wi and field_of(min(first[2], last[2])) != field_of(max(first[2], last[2])):
            return ("e", tuple(port(s, i, leaf, reader) for s, i, leaf in p))
        head = port(p.space, p.idx, p.base, reader)
        if isinstance(p, Affine):
            return ("a", head, p.count)
        assert isinstance(p, Strided)
        return ("s", head, p.inner, p.outer_stride, p.inner_stride, p.count)

    def entry(n, reader: int | None) -> tuple:
        return (n.form, fkey(n.fn), tuple((tkey(a.type), tuple(part_key(p, reader) for p in _flat(a.refs))) for a in n.args),
                tkey(n.out), tuple(n.axes), n.n, n.n_xs)

    pending: list[str] = []
    seen_shared: dict[str, int] = {}
    launch, cur, started = 0, 0, set()
    for k, n in enumerate(nodes):
        parts = [p for a in n.args for p in _flat(a.refs)]
        mine = [x for p in parts for x in contributes(p)]
        if mine:
            p = max(mine)
            if p < cur:
                launch += 1
        elif any(lab[p.idx] is not None for p in parts if p.space == "n" and not isinstance(p, Explicit)) or any(
                s == "n" and lab[i] is not None for p in parts if isinstance(p, Explicit) for s, i, _l in p.items):
            p = cur
        else:
            base = hashlib.sha512(repr(entry(n, None)).encode()).hexdigest()[:32]
            seen = seen_shared.get(base, 0)
            seen_shared[base] = seen + 1
            skey[k] = key = base if not seen else f"{base}#{seen}"
            pending.append(key)
            continue
        cur = p
        lab[k], local[k] = p, count[p]
        count[p] += 1
        s = streams[p]
        if (p, launch) not in started:
            started.add((p, launch))
            s.update(repr(("run", launch)).encode() + b"\n")
        for key in pending:
            s.update(repr(("shared", key)).encode() + b"\n")
        pending = []
        s.update(repr(entry(n, p)).encode() + b"\n")
    passes = len({r for r, _kind, _part, _q in peers.values()})
    if peers and launch + 1 != passes:
        raise ValueError(f"the body splits into {launch + 1} launches, its peer Inputs into {passes} passes")
    for key in pending:
        streams[E].update(repr(("shared", key)).encode() + b"\n")
    streams[E].update(repr(("ret", tuple(part_key(p, E) for p in _flat(body.ret)))).encode() + b"\n")

    root_id = descriptor_id(program.root)
    callees = {fid: definition_digest(definition_entry(fn)) for fid, fn in definitions(program).items() if fid != root_id}
    header = hashlib.sha512(repr((descriptor_id(program.fn), tuple((n, tkey(t)) for n, t in params))).encode()).hexdigest()
    return Pieces(layers=NL, launches=launch + 1, parts={_part_name(p, NL): s.hexdigest() for p, s in enumerate(streams)},
                  sizes={_part_name(p, NL): count[p] for p in range(E + 1)}, shared=len(skey), callees=callees, header=header)


def compare(model: Pieces, window: Pieces, at: int) -> dict:
    """`window`'s parts (a window of `model` starting at layer `at`, `window_config`) against the model's parts of the same layers
    (module docstring): each comparable part's outcome, and whether the callees both reach agree."""
    k = window.layers
    if at < 0 or at + k > model.layers:
        raise ValueError(f"a window of {k} layers at {at} isn't inside {model.layers}")
    rows = []
    if at == 0:
        rows.append(("prologue", "prologue"))
    for i in range(0 if at == 0 else 1, k - 1):
        rows.append((str(i), str(at + i)))
    if at + k == model.layers:
        rows += [(str(k - 1), str(at + k - 1)), ("epilogue", "epilogue")]
    out = [{"window": w, "model": m, "equal": window.parts[w] == model.parts[m]} for w, m in rows]
    callees = sorted(f for f in set(window.callees) & set(model.callees) if window.callees[f] != model.callees[f])
    return {"at": at, "layers": k, "parts": out, "equal": sum(r["equal"] for r in out), "different": sum(not r["equal"] for r in out),
            "callees_differ": callees}


def window_config(hf: dict, at: int, k: int) -> dict:
    """GLM-5.3's (or any sparse-attention checkpoint's) config of layers at..at+k as a model of their own: `indexer_types` and
    `mlp_layer_types` sliced, the dense layers that remain first (`first_k_dense_replace`), and the top-k offset that makes vLLM's
    rule (`skip_topk = max(layer - index_skip_topk_offset + 1, 0) % index_topk_freq != 0`) give the sliced `indexer_types`. A window
    whose first layer has no indexer of its own, or whose pattern no offset gives, is refused."""
    NL = int(hf["num_hidden_layers"])
    if at < 0 or k < 1 or at + k > NL:
        raise ValueError(f"layers {at}..{at + k} aren't inside {NL}")
    kinds = list(hf["indexer_types"][at:at + k])
    if kinds[0] != "full":
        raise ValueError(f"layer {at} shares another layer's top-k: a window starts at a layer with an indexer of its own")
    freq = int(hf.get("index_topk_freq", 1))
    offsets = [o for o in range(-freq, freq + 2)
               if all((max(i - o + 1, 0) % freq == 0) == (kinds[i] == "full") for i in range(k))]
    if not offsets:
        raise ValueError(f"no index_skip_topk_offset gives layers {at}..{at + k}'s indexers {kinds}")
    off = int(hf["index_skip_topk_offset"]) if int(hf.get("index_skip_topk_offset", 0)) in offsets else offsets[-1]
    dense = max(0, int(hf.get("first_k_dense_replace", 0)) - at)
    mlp = list(hf["mlp_layer_types"][at:at + k])
    if mlp != ["dense"] * min(dense, k) + ["sparse"] * (k - min(dense, k)):
        raise ValueError(f"layers {at}..{at + k}'s MLPs {mlp} aren't first_k_dense_replace={dense} dense, then sparse")
    return {**hf, "num_hidden_layers": k, "indexer_types": kinds, "mlp_layer_types": mlp, "first_k_dense_replace": dense,
            "index_skip_topk_offset": off}
