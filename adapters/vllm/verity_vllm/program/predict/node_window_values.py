"""The value check of a node window (`node_window.node_window`): every lowered template's units, recomputed from a made-up
request's values through what the emitted circuit declares, give the words the request's Program gives.

The Program is evaluated node by node on made-up inputs (`made_up`, `Values`): each node by `evaluate_batch`'s fastest kernel,
an unlowered node whose reference evaluator can't take its operand by its NumPy transcription (`NATIVE`, held to the
reference on a small binding before it is used). A unit's inputs come from the circuit alone: its wires (the writer row's
words), its registered reads (the parameter's words at the read's source), its table reads (the committed table's root and
domain id, `interface.digest_words`), its openings (the committed table's row, salt and path at the unit's key) and, for a
port the circuit leaves free, the Program's own words where the frame's ``stand_ins`` window (`NodeWindow.stood`) says it
reads them (a peer's row, an unlowered node's row), or the free port's Definition read (`FREE`). A row no source names is
undeclared: zeros, counted, and the unit's mismatch counted apart. A unit's expected words are the Program's at the frame's
pieces (`NodeWindow.outs`), a gather's the concatenation of the rows it holds (`NodeWindow.wholes`, ``chunks``), an
opening's verdict 1; a writer row with none is recomputed from its own inputs. Tables are committed by
`h1.registered.commit` from the made-up weights as `pipeline.expert_rows.table_words` cuts them, under a run-local key.

This is completeness evidence: the check is of what a verifier would read (the emitted circuit), not of the lowering's
own lists, and it samples (`check`)."""
from __future__ import annotations

import bisect
import dataclasses
import hashlib
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from verity.circuits.defs import bind
from verity.circuits.evaluation.batch import evaluate_batch
from verity.circuits.refs import runs
from verity.commitments import randomness as RND
from verity.commitments.windows import interface as IF

from .node_window import PEERS, STAND_IN, WEIGHTS, NodeWindow, _family

PROMPT = "prompt"
SAMPLE = 1024
SAMPLE_DOMAIN = "verity/node-window/value-check/v1"
_DT = {8: np.uint8, 16: np.uint16, 32: np.dtype("<u4"), 64: np.dtype("<u8")}


def _bf16(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _field(label: str, shape: tuple[int, ...], bits: int, rng: np.random.Generator) -> np.ndarray:
    """Made-up elements of weight ``label`` as their ``bits``-bit words: bf16 matrices (the rotary cache too) at
    1/sqrt(fan-in), norms near 1; e4m3 bytes of magnitude 2^-3 .. 2^2 (never NaN); f32 block scales powers of two near the
    fan-in's inverse square root, biases small, other f32 scales near 1."""
    n = math.prod(shape)
    fan = shape[-1] if shape else 1
    if bits == 16:
        if "norm" in label:
            x = 1 + rng.normal(0, 0.05, n)
        else:
            x = rng.normal(0, fan ** -0.5, n)
        return _bf16(x)
    if bits == 8:
        s = rng.integers(0, 2, n, dtype=np.uint8)
        e = rng.integers(4, 10, n, dtype=np.uint8)
        m = rng.integers(0, 8, n, dtype=np.uint8)
        return (s << 7) | (e << 3) | m
    if bits == 32:
        if "scale_inv" in label:
            x = np.exp2(rng.integers(-8, -5, n)).astype(np.float32)
        elif label.endswith("bias"):
            x = rng.normal(0, 0.05, n).astype(np.float32)
        else:
            x = (1 + rng.normal(0, 0.05, n)).astype(np.float32)
        return x.view("<u4")
    raise ValueError(f"weight {label}: {bits}-bit leaves have no made-up values")


def _shape(t: Any) -> tuple[int, ...]:
    out = []
    while hasattr(t, "elem"):
        out.append(t.n)
        t = t.elem
    return tuple(out)


def made_up(fn: Any, prompt: Sequence[int], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Program ``fn``'s inputs, made up from the caller's ``rng`` (test data, not a challenge): the prompt's ids as given, each
    peer's partials bf16 of N(0, 1/4), the weights record as its u16 words, field by field (`_field`); refused for any other
    input."""
    out: dict[str, np.ndarray] = {}
    for name, t in fn.signature()[0]:
        if name == PROMPT:
            if t.leaves != len(prompt):
                raise ValueError(f"the Program reads {t.leaves} prompt ids, not {len(prompt)}")
            out[name] = np.asarray(prompt, np.int64).astype(np.uint32)
        elif name.startswith(PEERS):
            if t.width_set() != {16}:
                raise ValueError(f"{name}: {sorted(t.width_set())}-bit leaves, not bf16 partials")
            out[name] = _bf16(rng.normal(0, 0.5, t.leaves))
        elif name == WEIGHTS:
            parts = []
            for label, ft in t.fields:
                (bits,) = ft.width_set()
                parts.append(np.ascontiguousarray(_field(label, _shape(ft), bits, rng)).view("<u2"))
            out[name] = np.concatenate(parts)
        else:
            raise ValueError(f"input {name}: no made-up values for it")
    return out


# ---- the Program's values

def _embedding_shard(v: "Values", node: Any, t: int) -> np.ndarray:
    S = node.fn.bindings
    VS, H, START = int(S["VS"]), int(S["H"]), int(S["START"])
    tok = int(v.leaves(v.arg(node, 0, t))[0])
    tok = tok - (1 << 32) if tok >= 1 << 31 else tok
    out = np.zeros(H, np.uint16)
    if START <= tok < START + VS:
        rs = runs(v.arg(node, 1, t))
        if len(rs) != 1:
            return np.asarray(evaluate_batch(node.fn, *(v.leaves(v.arg(node, k, t))[None] for k in range(2)))).reshape(-1)
        space, idx, lo, _hi = rs[0]
        a = lo + (tok - START) * H
        out[:] = v.leaves([(space, idx, a, a + H)])
    return out


def _embedding_shard_small() -> tuple[Any, list[np.ndarray], np.ndarray]:
    from verity.definitions.registry import b1_tp2 as B1T

    fn = bind(B1T.EmbeddingShard, VS=4, H=3, START=2)
    table = np.arange(1, 13, dtype=np.uint16) * 0x0101
    return fn, [np.array([t & 0xFFFFFFFF], np.uint32) for t in range(-1, 8)], table


#: an unlowered family's NumPy transcription, a member at a time, for one whose reference evaluator can't take its operands
NATIVE: dict[str, Callable[["Values", Any, int], np.ndarray]] = {"EmbeddingShard": _embedding_shard}


def native_selftest() -> dict[str, Any]:
    """Each `NATIVE` transcription against the reference evaluator on a small binding; refused where they differ."""
    fn, toks, table = _embedding_shard_small()
    got, want = [], []
    for tok in toks:
        want.append(np.asarray(evaluate_batch(fn, tok[None], table[None], kernel="reference")).reshape(-1))
        t = int(tok[0])
        t = t - (1 << 32) if t >= 1 << 31 else t
        got.append(table.reshape(4, 3)[t - 2] if 2 <= t < 6 else np.zeros(3, np.uint16))
    ok = all(np.array_equal(a.astype(np.uint64), b.astype(np.uint64)) for a, b in zip(got, want))
    if not ok:
        raise AssertionError("EmbeddingShard's NumPy transcription differs from its reference on VS=4, H=3, START=2")
    return {"EmbeddingShard": f"agrees with the reference on {len(toks)} ids at VS=4, H=3, START=2"}


class Values:
    """The Program ``fn``'s values on ``inputs`` (`made_up`): each node's output leaves, member-major, and each member's words
    as the frame ``w`` lays them out (`NodeWindow.meta`)."""

    def __init__(self, fn: Any, inputs: Mapping[str, np.ndarray], w: NodeWindow, log: Callable[[str], None] = print):
        self.fn, self.inputs, self.w, self.log = fn, inputs, w, log
        self.names = [p for p, _ in fn.signature()[0]]
        self.nodes: list[np.ndarray] = []
        self.times: dict[str, float] = {}
        self.native: Counter = Counter()
        self._words: dict[int, np.ndarray] = {}
        self._fields = w.fields
        self._firsts = [f[0] for f in w.fields]

    def arg(self, node: Any, k: int, t: int) -> Any:
        refs = node.args[k].refs
        if node.form != "batch" or node.axes[k] is None:
            return refs
        L = len(refs) // node.n
        return refs.slice(t * L, (t + 1) * L)

    def _weight(self, lo: int, hi: int) -> np.ndarray:
        k = bisect.bisect_right(self._firsts, lo) - 1
        first, word, bits = self._fields[k]
        end = self._fields[k + 1][0]
        if hi > end:
            raise ValueError(f"a read of weight leaves [{lo}, {hi}) spans two weights")
        a = self.inputs[WEIGHTS][word:word + (end - first) * bits // 16].view(_DT[bits])
        return a[lo - first:hi - first]

    def leaves(self, refs: Any) -> np.ndarray:
        """``refs``' leaves (refs, or ``(space, idx, lo, hi)`` runs), each in its own width, widened to the widest."""
        parts = []
        for space, idx, lo, hi in (refs if isinstance(refs, list) else runs(refs)):
            if space == "n":
                parts.append(self.nodes[idx][lo:hi])
            elif self.names[idx] == WEIGHTS:
                parts.append(self._weight(lo, hi))
            else:
                parts.append(self.inputs[self.names[idx]][lo:hi])
        if len(parts) == 1:
            return parts[0]
        dt = max((p.dtype for p in parts), key=lambda d: d.itemsize)
        return np.concatenate([p.astype(dt) for p in parts])

    def evaluate(self) -> Values:
        for j, node in enumerate(self.fn.body.nodes):
            t0 = time.perf_counter()
            self.nodes.append(np.ascontiguousarray(self._node(node)).reshape(-1))
            fam = _family(node)
            self.times[fam] = self.times.get(fam, 0.0) + time.perf_counter() - t0
            if node.out_leaves != self.nodes[-1].size:
                raise ValueError(f"node {j} ({node.fn.id}): {self.nodes[-1].size} leaves, not {node.out_leaves}")
        return self

    def _node(self, node: Any) -> np.ndarray:
        fam = _family(node)
        n = node.n if node.form == "batch" else 1
        if node.form not in ("primitive", "call", "batch"):
            raise ValueError(f"{node.fn.id}: a {node.form} node is not evaluated here")
        if fam in NATIVE:
            self.native[fam] += n
            return np.concatenate([NATIVE[fam](self, node, t) for t in range(n)])
        if node.form != "batch":
            return np.asarray(evaluate_batch(node.fn, *(self.leaves(a.refs)[None] for a in node.args))).reshape(-1)
        if any(x not in (0, None) for x in node.axes):
            raise ValueError(f"{node.fn.id}: a batch on axes {node.axes}, not 0 or none")
        shared = sum(len(a.refs) for a, x in zip(node.args, node.axes) if x is None)
        if shared * n <= 1 << 22:
            args = [self.leaves(a.refs).reshape(n, -1) if x == 0 else np.broadcast_to(self.leaves(a.refs), (n, len(a.refs)))
                    for a, x in zip(node.args, node.axes)]
            return np.asarray(evaluate_batch(node.fn, *args)).reshape(-1)
        return np.concatenate([np.asarray(evaluate_batch(node.fn, *(self.leaves(self.arg(node, k, t))[None]
                                                                     for k in range(len(node.args))))).reshape(-1)
                               for t in range(n)])

    def words(self, j: int) -> np.ndarray:
        """Node ``j``'s members' words ``(members, words)``, each return field's leaves packed into u16 words."""
        got = self._words.get(j)
        if got is None:
            n, L, _W, spans = self.w.meta[j]
            if spans is None:
                raise ValueError(f"node {j}: its return isn't whole u16 words")
            leaves = self.nodes[j].reshape(n, L)
            cols = [np.ascontiguousarray(leaves[:, f0:f1].astype(_DT[bits])).view("<u2").reshape(n, -1)
                    for (f0, _w0, bits), (f1, _w1, _b) in zip(spans, spans[1:])]
            got = self._words[j] = np.concatenate(cols, axis=1) if cols else np.zeros((n, 0), np.uint16)
        return got

    def input_words(self, name: str) -> np.ndarray:
        """Input ``name``'s leaves as u16 words."""
        a = self.inputs[name]
        return a if a.dtype == np.uint16 else np.ascontiguousarray(a).view("<u2")


# ---- the tables

_TABLE = re.compile(r"^weights/(?P<label>.+)@(?P<words>\d+)/b(?P<b>\d+)/(?P<kind>table|vocab)$")


class Tables:
    """Each table the circuit declares, committed on first use from the made-up weights (an expert table's rows by
    `expert_rows.table_words`, an embedding shard's column block's by `node_window_embed.table_words`) under a run-local key,
    by parameter."""

    def __init__(self, circuit: IF.Circuit, values: Values, w: NodeWindow, program: str):
        self.by = {t.name: t for t in circuit.registered.tables}
        self.values, self.program = values, program
        self.fields = {label: (first, words) for label, first, words in w.b.weights}
        self.types = dict(dict(values.fn.signature()[0])[WEIGHTS].fields)
        self.key = hashlib.sha512(b"node-window values check: a test committer's key").digest()[:32]
        self.held: dict[str, Any] = {}

    def __call__(self, name: str) -> Any:
        got = self.held.get(name)
        if got is not None:
            return got
        from verity.protocols.proofs.spot.verifier.one_stage import registered as G
        from verity_vllm.pipeline import expert_rows as XR
        from verity_vllm.protocols.pouw.h1 import registered as H1R

        t = self.by[name]
        m = _TABLE.match(t.param)
        if m is None:
            raise ValueError(f"table {name}: parameter {t.param} is not an expert table's (`expert_rows.table_param`) or an "
                             "embedding shard's (`node_window_embed.shard_param`)")
        label, H, b = m["label"], int(m["words"]), int(m["b"])
        first, words = self.fields[label]
        stack = self.values.inputs[WEIGHTS][first:first + words]
        if m["kind"] == "vocab":
            rows = self._vocab(name, stack, H, b, t.rows)
        else:
            ft = self.types[label]
            (bits,) = ft.width_set()
            E, A, Bd = _shape(ft)
            if bits == 8:
                n = 2 * H // Bd
                rows = XR.table_words(stack.view(np.uint8), "w", E, A, Bd, 1, n, b)
            else:
                rows = XR.table_words(stack.view("<u4"), "ws", E, A, Bd, 1, 0, b)
        if rows.shape != (t.rows, t.words):
            raise ValueError(f"table {name}: {rows.shape} words, not ({t.rows}, {t.words})")
        got = self.held[name] = H1R.commit(self.program, t.param, rows, self.key, len(self.held), keep_rows=True,
                                           schema=G.SCHEMA_SEG)
        return got

    def _vocab(self, name: str, stack: np.ndarray, width: int, b: int, V: int) -> np.ndarray:
        """Column block ``b`` of the embedding shard's rows, padded to ``V`` rows at the shard's START: the shard whose
        table is this field, by its VS×H words."""
        from . import node_window_embed as NE

        shards = {tuple(int(n.fn.bindings[k]) for k in ("VS", "H", "START")) for n in self.values.fn.body.nodes
                  if _family(n) in NE.FAMILIES}
        hit = [S for S in shards if S[0] * S[1] == len(stack)]
        if len(hit) != 1:
            raise ValueError(f"table {name}: {len(hit)} embedding shards of {len(stack)} words, not one")
        ((VS, H, START),) = hit
        return NE.table_words(stack.reshape(VS, H)[:, b * width:(b + 1) * width], START, V)


# ---- the check

def _be_words(b: bytes) -> np.ndarray:
    return np.frombuffer(bytes(b), ">u8").astype("<u8").view("<u2")


#: a free port's Definition read the frame leaves free with no stand-in (an explicit `_Free`), by family: the Program's words of
#: each unit's row, from the unit's node (`NodeWindow.node_of`), its first unit and its offset
FREE: dict[tuple[str, str], Callable[["Check", Any, int, int, int], np.ndarray]] = {}


def _batched_gemm_w(ck: "Check", node: Any, a: int, u: int, words: int) -> np.ndarray:
    """`BatchedGemm` (`node_window_sparse`): unit (b, n) of member t, at a + (b N + n) M + t, reads w[b, n, :] of K words."""
    K = int(node.fn.bindings["K"])
    M = node.n if node.form == "batch" else 1
    i = u - a
    t, bn = i % M, i // M
    w = ck.values.leaves(ck.values.arg(node, 1, t).slice(bn * K, (bn + 1) * K))
    if words != K:
        raise ValueError(f"BatchedGemm w: a row of {words} words, not K = {K}")
    return w.astype(np.uint16)


FREE[("BatchedGemm", "w")] = _batched_gemm_w


class Check:
    """The value check of ``circuit`` (the frame ``w`` built it with ``stand_ins``) against ``values``."""

    def __init__(self, circuit: IF.Circuit, w: NodeWindow, values: Values, tables: Tables):
        self.c, self.w, self.values, self.tables = circuit, w, values, tables
        self.fns = dict(zip(circuit.ids, (f for f, _n in circuit.templates)))
        self.counts = dict(zip(circuit.ids, (n for _f, n in circuit.templates)))
        self.lay = {t: IF.layout(f) for t, f in self.fns.items()}
        self.wires: dict[tuple[str, str], list[IF.Wire]] = {}
        for x in circuit.wires:
            self.wires.setdefault((x.reader, x.port), []).append(x)
        reg = circuit.registered
        self.reads: dict[tuple[str, str], list[Any]] = {}
        for x in reg.reads:
            self.reads.setdefault((x.reader, x.port), []).append(x)
        self.roots: dict[tuple[str, str], list[Any]] = {}
        for x in reg.roots:
            self.roots.setdefault((x.reader, x.port), []).append(x)
        self.openings: dict[str, list[Any]] = {}
        for x in reg.openings:
            self.openings.setdefault(x.reader, []).append(x)
        self.verdicts = {(o.reader, o.verdict) for o in reg.openings}
        self.stood: dict[tuple[str, str], list[Any]] = {}
        for x in w.stood:
            self.stood.setdefault((x[0], x[1]), []).append(x)
        self.free: dict[tuple[str, str], list[Any]] = {}
        for f in w.free:
            self.free.setdefault((f.tid, f.port), []).append(f)
        self.params = self._params()
        #: rows two pieces hold, ``(tid, out, unit, node, the first holder's node)``; the first holder's words are expected
        self.conflicts: list[tuple[str, str, int, int, int]] = []
        self.anchor = self._anchors()
        self.gathered: dict[tuple[str, int, str], tuple[Any, ...]] = {}
        for rows, r in w.wholes.items():
            self.gathered[(r.tid, r.unit, r.out)] = rows
        for rows, r in w.chunks.values():
            self.gathered[(r.tid, r.unit, r.out)] = rows
        self.memo: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self.sources: dict[str, Counter] = {}
        self.fed: dict[str, Counter] = {}
        self.unknown: dict[tuple[str, int], list[str]] = {}

    def _params(self) -> dict[str, np.ndarray]:
        """Each registered parameter a read names, as its u16 words: a weight's (`window_builder.weight_param`) or an input's."""
        from .window_builder import weight_param

        out: dict[str, np.ndarray] = {}
        by = {weight_param(label, 0).rsplit("@", 1)[0]: (first, words) for label, first, words in self.w.b.weights}
        for x in self.c.registered.reads:
            if x.param in out:
                continue
            base = x.param.rsplit("@", 1)[0]
            if base in by:
                first, words = by[base]
                out[x.param] = self.values.inputs[WEIGHTS][first:first + words]
            elif x.param in self.values.inputs:
                out[x.param] = self.values.input_words(x.param)
            else:
                raise ValueError(f"a read of {x.param}: no such weight or input")
        return out

    def _anchors(self) -> dict[tuple[str, str], np.ndarray]:
        """Each (template, output)'s rows the frame's pieces hold: per unit ``(node, member, first word)``, -1 where none."""
        out: dict[tuple[str, str], np.ndarray] = {}
        for j, o in self.w.outs.items():
            if isinstance(o, str):
                continue
            for p in o.pieces:
                r = p.row
                key = (r.tid, r.out)
                a = out.get(key)
                if a is None:
                    a = out[key] = np.full((self.counts[r.tid], 3), -1, np.int64)
                k = (p.hi - p.lo) // r.words
                for m in range(o.members):
                    us = r.unit + m * p.member + np.arange(k) * p.step
                    held = a[us, 0] >= 0
                    if held.any():
                        self.conflicts.append((r.tid, r.out, int(us[held][0]), j, int(a[us[held][0], 0])))
                    us, lo = us[~held], (p.lo + np.arange(k) * r.words)[~held]
                    a[us, 0], a[us, 1], a[us, 2] = j, m, lo
        return out

    # -- expected words

    def width(self, tid: str, out: str) -> int:
        return dict(self.lay[tid].outs)[out]

    def want(self, tid: str, out: str, u: int) -> np.ndarray | None:
        """The Program's words row (tid, u, out) holds, by the frame's pieces, gathers and openings; None where none says."""
        a = self.anchor.get((tid, out))
        if a is not None and a[u, 0] >= 0:
            j, m, lo = (int(x) for x in a[u])
            return self.values.words(j)[m, lo:lo + self.width(tid, out)]
        g = self.gathered.get((tid, u, out))
        if g is not None:
            parts = [self.want(r.tid, r.out, r.unit) for r in g]
            return None if any(p is None for p in parts) else np.concatenate(parts)
        if (tid, out) in self.verdicts:
            return np.ones(self.width(tid, out), np.uint16)
        return None

    def anchored(self, tid: str) -> bool:
        if not hasattr(self, "_anchored"):
            self._anchored = {k[0] for k in self.anchor} | {k[0] for k in self.gathered} | {v[0] for v in self.verdicts}
        return tid in self._anchored

    # -- a unit's inputs

    def row(self, tid: str, out: str, u: int) -> np.ndarray:
        """Row (tid, u, out) as a reader gets it: the Program's words where the frame says them, else recomputed."""
        got = self.want(tid, out, u)
        return got if got is not None else self.outputs(tid, [u])[out][0]

    def _stand(self, r: Any) -> np.ndarray:
        if r.tid == f"{STAND_IN}cat":
            return np.concatenate([self._stand(x) for x in self.w.cats[r.unit]])
        if not r.tid.startswith(STAND_IN):
            return self.row(r.tid, r.out, r.unit)
        lo, hi = (int(x) for x in r.out.split(":"))
        kind, name = r.tid[len(STAND_IN)], r.tid[len(STAND_IN) + 2:]
        if kind == "n":
            return self.values.words(int(name))[r.unit, lo:hi]
        return self.values.input_words(name)[lo:hi]

    def inputs(self, tid: str, units: Sequence[int]) -> list[np.ndarray]:
        """Units ``units``' port rows, from the circuit's declarations (module docstring)."""
        lay, units = self.lay[tid], np.asarray(units, np.int64)
        cuts = self.c.cuts.get(tid, {})
        src = self.sources.setdefault(tid, Counter())
        fed = self.fed.setdefault(tid, Counter())
        ports: dict[str, np.ndarray] = {}
        done: dict[str, np.ndarray] = {}
        order = sorted(lay.rows, key=lambda pr: any(pr[0] in o.ports for o in self.openings.get(tid, ())))
        for port, words in order:
            k = cuts.get(port, 1)
            rw = words // k
            arr = np.zeros((len(units), words), np.uint16)
            got = np.zeros((len(units), k), bool)

            def put(sel: np.ndarray, row: int, vals: np.ndarray, how: str) -> None:
                if got[sel, row].any():
                    src[f"{port}: a row two sources fill"] += int(got[sel, row].sum())
                arr[sel, row * rw:(row + 1) * rw] = vals
                got[sel, row] = True
                src[how] += int(sel.sum())

            for x in self.wires.get((tid, port), ()):
                C, c0 = x.cut[1], x.cut[0]
                sel = (units >= x.at) & (units < x.at + x.reader_units)
                if not sel.any():
                    continue
                for r in range(C):
                    srcs = [int(s) for s in x.source((units[sel] - x.at) * C + r)]
                    need = sorted({s for s in srcs if (x.writer, s) not in self.memo
                                   and self.want(x.writer, x.out, s) is None})
                    if need:
                        self.outputs(x.writer, need)
                    vals = np.stack([self.row(x.writer, x.out, s) for s in srcs])
                    put(sel, c0 + r, vals, "wire")
            for x in self.reads.get((tid, port), ()):
                C, c0 = x.cut[1], x.cut[0]
                sel = (units >= x.at) & (units < x.at + x.reader_units)
                if not sel.any():
                    continue
                pw = self.params[x.param]
                for r in range(C):
                    s = np.asarray(x.source((units[sel] - x.at) * C + r), np.int64)
                    put(sel, c0 + r, pw[s[:, None] + np.arange(rw)], "read")
            for x in self.roots.get((tid, port), ()):
                sel = (units >= x.at) & (units < x.at + x.units)
                if sel.any():
                    e = self.tables(x.table).entry()
                    put(sel, 0, np.broadcast_to(IF.digest_words(bytes.fromhex(e[x.of])), (int(sel.sum()), rw)), f"table {x.of}")
            for o in self.openings.get(tid, ()):
                if port not in o.ports:
                    continue
                sel = (units >= o.at) & (units < o.at + o.units)
                if not sel.any():
                    continue
                tb = self.tables(o.table)
                keys = done[o.key][sel].astype(np.int64)
                key = keys[:, 0] if keys.shape[1] == 1 else keys[:, 0] | (keys[:, 1] << 16)
                key = np.where(key >= 1 << 31, key - (1 << 32), key) if keys.shape[1] == 2 else key
                which = o.ports.index(port)
                vals = []
                for t in key:
                    if not 0 <= t < tb.leaves:
                        vals.append(np.zeros(rw, np.uint16))
                    elif which == 0:
                        vals.append(np.asarray(tb.rows[t], np.uint16))
                    elif which == 1:
                        vals.append(_be_words(tb.salts[t]))
                    else:
                        vals.append(_be_words(tb.path(t)))
                put(sel, 0, np.stack(vals), "opening")
            for (_t, _p, at, count, (first, rows, of), grid) in self.stood.get((tid, port), ()):
                sel = (units >= at) & (units < at + count)
                if not sel.any():
                    continue
                if of != k:
                    raise ValueError(f"{tid} {port}: stood at cut {of}, the circuit cuts it {k} ways")
                idx = np.nonzero(sel)[0]
                vals = np.stack([np.concatenate([self._stand(r) for r in grid[int(units[i]) - at]]) for i in idx])
                for r in range(rows):
                    free = ~got[idx, first + r]
                    if (~free).any():
                        src[f"{port}: a free row also declared"] += int((~free).sum())
                    arr[idx[free], (first + r) * rw:(first + r + 1) * rw] = vals[free, r * rw:(r + 1) * rw]
                    got[idx[free], first + r] = True
                fams = sorted({s[1] for g in grid for s in self.w._sources(g)})
                fed[f"{port} rows {first}..{first + rows - 1} of {k}: {', '.join(fams)}"] += int(sel.sum())
            fam = _family_of(self.fns[tid])
            if (fam, port) in FREE and not got.all():
                for i in np.nonzero(~got.all(axis=1))[0]:
                    u = int(units[i])
                    a = self._alloc(tid, u)
                    node = self.values.fn.body.nodes[self.w.node_of[(tid, a)]]
                    arr[i] = FREE[(fam, port)](self, node, a, u, words)
                    got[i] = True
                fed[f"{port}: its Definition's read ({fam})"] += len(units)
            if not got.all():
                src[f"{port}: undeclared rows"] += int((~got).sum())
                for i in np.nonzero(~got.all(axis=1))[0]:
                    self.unknown.setdefault((tid, int(units[i])), []).append(port)
            ports[port] = arr
            done[port] = arr
        return [ports[p] for p, _w in lay.rows]

    def _alloc(self, tid: str, u: int) -> int:
        allocs = self.w.allocs[tid]
        return allocs[bisect.bisect_right([a[0] for a in allocs], u) - 1][0]

    def evaluate(self, tid: str, rows: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
        fn, lay = self.fns[tid], self.lay[tid]
        n = rows[0].shape[0] if rows else 0
        params = fn.signature()[0]
        leaves = [IF.leaves_of_rows(r, ty) for r, (_n, ty) in zip(rows, params)]
        got = np.asarray(evaluate_batch(fn, *leaves)).reshape(n, -1)
        ret = fn.signature()[1]
        fields = [f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,))]
        out, at = {}, 0
        for (name, words), ty in zip(lay.outs, fields):
            out[name] = IF.rows_of_leaves(got[:, at:at + ty.leaves], ty, words)
            at += ty.leaves
        return out

    def outputs(self, tid: str, units: Sequence[int]) -> dict[str, np.ndarray]:
        need = [u for u in units if (tid, u) not in self.memo]
        if need:
            got = self.evaluate(tid, self.inputs(tid, need))
            for i, u in enumerate(need):
                self.memo[(tid, u)] = {k: v[i] for k, v in got.items()}
        return {k: np.stack([self.memo[(tid, u)][k] for u in units]) for k in self.memo[(tid, units[0])]}

    def check(self, tid: str, units: Sequence[int]) -> dict[str, Any]:
        """Units ``units`` of ``tid`` recomputed, each output the frame says compared with the Program's words."""
        units = [int(u) for u in units]
        t0 = time.perf_counter()
        got = self.outputs(tid, units)
        checked, bad, first, unknown_bad = 0, 0, None, 0
        for i, u in enumerate(units):
            hit, miss = False, False
            for out, _w in self.lay[tid].outs:
                want = self.want(tid, out, u)
                if want is None:
                    continue
                hit = True
                if not np.array_equal(want, got[out][i]):
                    miss = True
                    if first is None:
                        d = np.nonzero(want != got[out][i])[0]
                        first = {"unit": u, "port": out, "word": int(d[0]), "words": len(want),
                                 "expected": [int(x) for x in want[d[0]:d[0] + 8]],
                                 "got": [int(x) for x in got[out][i][d[0]:d[0] + 8]], "differ": int(d.size)}
            checked += hit
            bad += miss
            unknown_bad += miss and (tid, u) in self.unknown
        return {"checked": checked, "mismatches": bad, "mismatches_with_undeclared_inputs": unknown_bad, "first": first,
                "seconds": round(time.perf_counter() - t0, 2)}

    def readers(self, tid: str, units: Sequence[int]) -> dict[str, set[int]]:
        """The units that read rows of ``units`` of ``tid`` through a wire, by template."""
        want = set(int(u) for u in units)
        out: dict[str, set[int]] = {}
        for x in self.c.wires:
            if x.writer != tid:
                continue
            n = x.reader_units * x.cut[1]
            i = np.arange(n, dtype=np.int64)
            s = np.asarray(x.source(i), np.int64)
            hit = np.isin(s, list(want))
            if hit.any():
                out.setdefault(x.reader, set()).update((x.at + i[hit] // x.cut[1]).tolist())
        return out


def _family_of(fn: Any) -> str:
    return fn.defn.name if hasattr(fn, "defn") else fn.id.split("_v")[0]


def sample_key(seed: int) -> RND.Key:
    """The value check's key under ``seed``, from the one challenge derivation."""
    return RND.derive(str(seed).encode(), SAMPLE_DOMAIN, {})


def sample(n: int, key: RND.Key, index: str, most: int = SAMPLE) -> list[int]:
    """Up to ``most`` of ``n`` units, the sample at ``index`` of ``key`` (all of a smaller template)."""
    return list(range(n)) if n <= most else list(key.subset(most, n, index))


def run(ck: Check, seed: int, log: Callable[[str], None] = print, most: int = SAMPLE,
        only: Sequence[str] | None = None) -> dict[str, Any]:
    """Every template of ``ck``'s circuit: up to ``most`` units sampled under ``seed`` (`sample_key`, each template's sample at its
    id; all of a smaller one); a template whose outputs the frame states is checked at its sample, any other through its
    sample's readers that are (a reader's units added to its own check, up to ``most`` more)."""
    key = sample_key(seed)
    picked = {tid: sample(ck.counts[tid], key, tid, most) for tid in ck.c.ids}
    out: dict[str, Any] = {}
    extra: dict[str, set[int]] = {}
    through: dict[str, dict[str, int]] = {}
    for tid in ck.c.ids:
        if ck.anchored(tid):
            continue
        frontier, depth = {tid: set(picked[tid])}, 0
        while frontier and depth < 4:
            nxt: dict[str, set[int]] = {}
            for t, us in frontier.items():
                for r, rus in ck.readers(t, us).items():
                    rus = set(sorted(rus)[:most])
                    (extra.setdefault(r, set()) if ck.anchored(r) else nxt.setdefault(r, set())).update(rus)
                    if ck.anchored(r):
                        through.setdefault(tid, {})[r] = len(rus)
            frontier, depth = nxt, depth + 1
    for tid in ck.c.ids:
        if only is not None and tid not in only:
            continue
        t0 = time.perf_counter()
        units = sorted(set(picked[tid]) | set(sorted(extra.get(tid, ()))[:most])) if ck.anchored(tid) else []
        rec: dict[str, Any] = {"units": ck.counts[tid], "sampled": len(picked[tid]), "family": _family_of(ck.fns[tid])}
        if units:
            rec.update(ck.check(tid, units))
            rec["checked_units"] = len(units)
        else:
            rec.update({"checked": 0, "mismatches": 0, "first": None, "through": through.get(tid, {})})
        out[tid] = rec
        log(f"{tid[:90]}: {rec.get('checked', 0)} checked of {rec['units']}, {rec['mismatches']} mismatches "
            f"({time.perf_counter() - t0:.1f} s)")
    for tid in ck.c.ids:
        if tid in out and not ck.anchored(tid):
            out[tid]["evaluated"] = sum(1 for (t, _u) in ck.memo if t == tid)
            out[tid]["evaluated_of_sampled"] = sum(1 for u in picked[tid] if (tid, u) in ck.memo)
    for tid, rec in out.items():
        rec["sources"] = dict(ck.sources.get(tid, {}))
        rec["fed_from_program"] = dict(ck.fed.get(tid, {}))
    return out


# ---- planted errors

def plant_wire_row(c: IF.Circuit, family: str, port: str, row: int, keep: bool = False) -> tuple[IF.Circuit, str, dict[str, Any]]:
    """``c`` with the first wire into ``family``'s ``port`` at cut row 0 moved to cut row ``row``; with ``keep``, copied there
    beside the original, so row 0 stays wired and only row ``row`` is wrong."""
    for i, x in enumerate(c.wires):
        if x.reader.startswith(family) and x.port == port and tuple(x.cut) == (0, 1):
            moved = IF.Wire(x.reader, x.port, x.writer, x.out, x.at, x.radix, x.base, x.step, (row, 1))
            ws = list(c.wires)
            ws[i:i + 1] = [x, moved] if keep else [moved]
            return dataclasses.replace(c, wires=tuple(ws)), x.reader, {"wire": i, "reader": x.reader, "port": port,
                                                                      "from": list(x.cut), "to": list(moved.cut),
                                                                      "kept": keep, "units": x.reader_units}
    raise ValueError(f"no wire into {family} {port} at cut row 0")


def plant_read_shift(c: IF.Circuit, family: str, port: str | None = None) -> tuple[IF.Circuit, str, dict[str, Any]]:
    """``c`` with the first registered read into ``family``'s ``port`` (any port by default) moved by one row of its own
    width."""
    reg = c.registered
    for i, x in enumerate(reg.reads):
        if x.reader.startswith(family) and port in (None, x.port):
            port = x.port
            fn = dict(zip(c.ids, (f for f, _n in c.templates)))[x.reader]
            rw = dict(IF.layout(fn).rows)[port] // c.cuts.get(x.reader, {}).get(port, 1)
            moved = IF.Read(x.reader, x.port, x.param, x.at, x.radix, x.base + rw, x.step, x.cut)
            rs = list(reg.reads)
            rs[i] = moved
            return (dataclasses.replace(c, registered=dataclasses.replace(reg, reads=tuple(rs))), x.reader,
                    {"read": i, "reader": x.reader, "port": port, "param": x.param, "from": x.base, "to": moved.base,
                     "row_words": rw})
    raise ValueError(f"no registered read into {family} {port}")


__all__ = ["Check", "FREE", "NATIVE", "SAMPLE", "Tables", "Values", "made_up", "native_selftest", "plant_read_shift",
           "plant_wire_row", "run", "sample", "sample_key"]
