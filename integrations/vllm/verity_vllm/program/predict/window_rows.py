"""A window's rows: every unit of a class circuit (`window_circuit.lower`) given its output rows from a served window's
values, ready for `interface.Window.record`.

A node of a slot's class Program has one value, its return fields: the served value where the engine computed it (its
request's tapped words in the sealed record, `Served`), else the class Program's own, its family's row kernel
(`numerics.rows.ROWS`) on the node's operands. The engine never computes the positions past a request's last output, nor,
at a prompt position before the last, the nodes prefill prunes (the last layer from its query on, the final norm, the LM
head and the token select), so those are computed. An embedding's value is the table's row at its position's token, and a
token select's the first maximum of its logits, whichever source gives it; a served one that isn't is refused by name.

A unit its node's lowering filled holds its rows of that value (`_Program.outs`). Every other unit holds its Definition's
function of the window's values:
- a position's `TokenFeed_v1` its token and the slot it carries (`io_shell.feed_reference`), the first position's
  `SlotRow_v1` the slot's row opened, and each token select's `SlotMask_v1` the token or `io_shell.EMPTY`;
- each token select's chunks the first maximum above -inf of their logits and its index (`chunk_best`), and its ``x0``
  unit ``logits[0]``;
- each embedding read's verdict 1, its opening the table's row, salt and path at the token (the table's registered value,
  core's ``one_stage.registered.Registered`` of ``row-seg`` rows or one like it);
- each `AllGather_v1` the concatenation of its parts, read through the circuit's wires once their writers hold them.

`Rows.record` gives a `Window` each template's units in runs of one step and one source (`SOURCES`: served, computed, or
evaluated by its Definition), every input by reference but an opening's ports.
"""
from __future__ import annotations

import bisect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.refs import runs
from verity.ml.commitments.gates import embedding as E
from verity.ml.definitions.numerics.rows import ROWS
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.definitions.registry.b1 import NEG_INF_BF16
from verity.ml.windows import interface as IF
from verity_vllm.protocols.pouw.h1 import registered as H1R
from verity_vllm.query.manifest import format as RM

from .window_builder import WEIGHTS, table_param, weight_param
from .window_circuit import SLOT_ROWS, Lowered, WindowClass, _spans

#: a unit's source, its runs' label: its node's served value, the class Program's value at a node the engine didn't compute,
#: or its Definition's function of the window's values
SOURCES = ("served", "computed", "evaluated")
SERVED, COMPUTED, EVALUATED = range(3)

#: a weight's elements [lo, hi) in its leaves' unsigned width: ``weights(label, lo, hi)``
Weights = Callable[[str, int, int], np.ndarray]


def positions(fn: Any) -> list[int]:
    """Each node's position: a position begins at each embedding."""
    out, p = [], -1
    for n in fn.body.nodes:
        p += n.fn.name == "Embedding"
        out.append(p)
    return out


def _fields(fn: Any) -> list[Any]:
    ret = fn.signature()[1]
    return [f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,))]


def _cast(a: Any, ty: Any, what: str) -> np.ndarray:
    """``a`` as field ``ty``'s elements in their unsigned width: its bits where it is that wide, else its values, refused
    where they don't fit."""
    a = np.asarray(a).reshape(-1)
    (bits,) = ty.width_set()
    dt = np.dtype(f"<u{bits // 8}")
    if a.size != ty.leaves:
        raise ValueError(f"{what}: {a.size} elements, not its field's {ty.leaves}")
    if a.dtype.itemsize == dt.itemsize:
        return np.ascontiguousarray(a).view(dt)
    if a.dtype.kind not in "iu" or (a.size and (int(a.min()) < 0 or int(a.max()) >> bits)):
        raise ValueError(f"{what}: {a.dtype} values that are not {bits}-bit")
    return a.astype(dt)


def _words(fields: Sequence[np.ndarray]) -> np.ndarray:
    """A value's fields as its u16 words."""
    return np.concatenate([np.ascontiguousarray(f).view("<u2") for f in fields])


def _i32(xs: Sequence[int]) -> np.ndarray:
    """i32s (as their 32-bit encodings) as one row of u16 words."""
    return np.array([int(x) & 0xFFFFFFFF for x in xs], dtype="<u4").view("<u2").reshape(1, -1)


def _be_words(b: bytes) -> np.ndarray:
    """Bytes as a Boolean port's 64-bit words (`gates.rowleaf.digest_bits`: each 8 bytes big-endian) in u16 words."""
    return np.frombuffer(bytes(b), ">u8").astype("<u8").view("<u2")


def chunk_best(x: np.ndarray) -> tuple[int, int]:
    """`TokenSelectChunk_v2{C}` on bf16 words ``x``: (best, its index), the first maximum above -inf, a NaN never taken, and
    (-inf, 0) when there is none."""
    x = np.asarray(x, dtype=np.uint16)
    f = (x.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    f = np.where(np.isnan(f), -np.inf, f)
    if not (f > -np.inf).any():
        return NEG_INF_BF16, 0
    i = int(np.argmax(f))
    return int(x[i]), i


class Slot:
    """Slot ``row``'s class Program ``fn`` of class ``cls`` evaluated (module docstring): ``row`` its `SLOT_ROW` row
    (occupancy, prompt length, ids), ``weights`` the weights record's fields and ``served`` its request's served values,
    None for an empty slot. Each node's value (its return fields) and source, each position's token and, below the class's
    prompt cap, the slot its `TokenFeed_v1` reads and the one it carries on."""

    def __init__(self, fn: Any, cls: WindowClass, row: Sequence[int], weights: Weights, served: Served | None = None) -> None:
        self.fn, self.cls, self.weights = fn, cls, weights
        self.row = [int(x) & 0xFFFFFFFF for x in row]
        if len(self.row) != cls.prompt + 2:
            raise ValueError(f"a slot's row is (occupancy, prompt length, {cls.prompt} ids), not {len(self.row)} words")
        if served is not None and served.LP != self.row[1]:
            raise ValueError(f"the served request's {served.LP} prompt ids, its slot row's {self.row[1]}")
        sig = dict(fn.signature()[0])
        self.params = [n for n, _ in fn.signature()[0]]
        self.labels = [label for label, _ in sig[WEIGHTS].fields]
        self.starts = [s[0] for s in _spans("weights", sig[WEIGHTS].fields)]
        self.position = positions(fn)
        first: dict[int, int] = {}
        for j, p in enumerate(self.position):
            first.setdefault(p, j)
        self.values: list[list[np.ndarray]] = []
        self.source: list[int] = []
        self.tokens: list[int] = []
        self.slots: list[list[int] | None] = []
        self.carried: list[list[int] | None] = []
        for j, node in enumerate(fn.body.nodes):
            self._node(j, node, None if served is None else served(self.position[j], j - first[self.position[j]], node))

    def _node(self, j: int, node: Any, got: list[np.ndarray] | None) -> None:
        fam, fields, what = node.fn.name, _fields(node.fn), f"node {j} ({node.fn.id}) at position {self.position[j]}"
        if fam == "Embedding":
            self._token(self.position[j], node.args[0])
        want = self._evaluate(node, fields, what) if got is None or fam in ("Embedding", "TokenSelect") else None
        if got is not None:
            if len(got) != len(fields):
                raise ValueError(f"{what}: served as {len(got)} members, not its {len(fields)} return fields")
            got = [_cast(a, ty, f"{what}, served") for a, ty in zip(got, fields)]
            if want is not None and not all(np.array_equal(a, b) for a, b in zip(got, want)):
                raise ValueError(f"{what}: served as {got[0][:4].tolist()}..., not "
                                 + ("the table's row at its token" if fam == "Embedding" else "the first maximum of its logits"))
        self.values.append(got if got is not None else want)
        self.source.append(SERVED if got is not None else COMPUTED)

    def _token(self, p: int, tok: Any) -> None:
        """Position ``p``'s token (`window_circuit._Program._token`): the first position's its slot row's first id, below the
        prompt cap `TokenFeed_v1`'s of the slot carried and the previous select, past it the previous select."""
        NP = self.cls.prompt
        if p == 0:
            slot, sel = self.row[1:], None
        else:
            sel = int(self.operand(tok)[0])
            slot = self.carried[-1]
        if p < NP:
            t, nxt = IO.feed_reference(slot, sel, NP)
        else:
            t, slot, nxt = sel, None, None
        self.tokens.append(int(t))
        self.slots.append(slot)
        self.carried.append(nxt)

    def _evaluate(self, node: Any, fields: list[Any], what: str) -> list[np.ndarray]:
        if node.fn.name == "Embedding":
            V, H = node.fn.bindings["V"], node.fn.bindings["H"]
            label, lo, _hi = self._weight(node.args[1], V * H)
            t = self.tokens[-1]
            row = np.full(H, E.FILL, dtype=np.uint16) if t >= V else self.weights(label, lo + t * H, lo + (t + 1) * H)
            return [_cast(row, fields[0], what)]
        spec = descriptor_id(node.fn)
        k = ROWS.get(RM.spec_family(spec))
        if k is None or len(k.returns) != len(fields):
            raise ValueError(f"{what}: no row kernel of its {len(fields)} return fields")
        out = k.fn(RM.spec_statics(spec), [self.operand(a) for a in node.args])
        return [_cast(out[m], ty, f"{what}, its row kernel's {m}") for m, ty in zip(k.returns, fields)]

    def _weight(self, arg: Any, n: int) -> tuple[str, int, int]:
        """The weight ``arg`` reads as one run of ``n`` leaves: its label and the run's elements in it."""
        segs = runs(arg.refs)
        if len(segs) != 1 or segs[0][0] != "p" or self.params[segs[0][1]] != WEIGHTS or segs[0][3] - segs[0][2] != n:
            raise ValueError(f"{self.fn.id}: a table read of {segs}, not one run of {n} weight leaves")
        return self._field(segs[0][2], segs[0][3])

    def _field(self, lo: int, hi: int) -> tuple[str, int, int]:
        k = bisect.bisect_right(self.starts, lo) - 1
        if k < 0 or k >= len(self.labels) or hi > self.starts[k + 1]:
            raise ValueError(f"a weight read of leaves [{lo}, {hi}) spans two weights")
        return self.labels[k], lo - self.starts[k], hi - self.starts[k]

    def operand(self, arg: Any) -> np.ndarray:
        """The elements ``arg`` reads: earlier nodes' values and the weights', in their unsigned widths."""
        parts = []
        for space, idx, lo, hi in runs(arg.refs):
            if space == "n":
                off = 0
                for f in self.values[idx]:
                    if off <= lo and hi <= off + f.size:
                        parts.append(f[lo - off:hi - off])
                        break
                    off += f.size
                else:
                    raise ValueError(f"{self.fn.id}: leaves [{lo}, {hi}) of node {idx} span two of its fields")
            elif self.params[idx] == WEIGHTS:
                parts.append(np.asarray(self.weights(*self._field(lo, hi))))
            else:
                raise ValueError(f"{self.fn.id}: a node reads parameter {self.params[idx]!r}, which only the first token does")
        if len({p.dtype.itemsize for p in parts}) != 1:
            raise ValueError(f"{self.fn.id}: an operand of {sorted({p.dtype.str for p in parts})} elements")
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


@dataclass
class Served:
    """One request's served values by class node: its Program's rows as `ProgramIndex.describe` gives them, each with its
    row's ``spec`` (``{spec, step, op_path, members, row}``, its step the request's), and ``read(step, op_path, member, lo,
    hi)``, a member's elements [lo, hi) at a request step (`CommittedStore.words` at the request's arrival plus it).

    The request's prefill is its step 0: each op (its op path and members) a row a prompt position, in the class Program's
    node order within a position, or one row, its last position's, where prefill prunes it; step s ≥ 1 is position
    LP − 1 + s, its rows in node order. A row whose spec isn't its class node's descriptor id is refused by name."""
    rows: Sequence[Mapping[str, Any]]
    read: Callable[[int, str, str, int, int], tuple[np.ndarray | None, str]]
    prefill: list[list[Mapping[str, Any]]] = field(init=False)
    decode: list[list[Mapping[str, Any]]] = field(init=False)
    LP: int = field(init=False)

    def __post_init__(self) -> None:
        steps: dict[int, list[Mapping[str, Any]]] = {}
        for d in self.rows:
            steps.setdefault(int(d["step"]), []).append(d)
        if sorted(steps) != list(range(len(steps))):
            raise ValueError(f"a request's served steps are 0, 1, ...; not {sorted(steps)}")
        ops: dict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]] = {}
        for d in steps.get(0, ()):
            ops.setdefault((str(d["op_path"]), tuple(str(m[0]) for m in d["members"])), []).append(d)
        self.prefill = [sorted(v, key=lambda d: int(d["row"])) for v in ops.values()]
        self.LP = max((len(v) for v in self.prefill), default=0)
        for (op, _), v in zip(ops, self.prefill):
            if [int(d["row"]) for d in v] not in (list(range(self.LP)), [0]):
                raise ValueError(f"{op}: prefill rows {[d['row'] for d in v]}, not one a prompt position nor the last one's")
        self.decode = [steps[s] for s in range(1, len(steps))]
        for s, rs in enumerate(self.decode, 1):
            got = [(str(d["op_path"]), tuple(str(m[0]) for m in d["members"])) for d in rs]
            if got != list(ops):
                raise ValueError(f"step {s}'s {len(rs)} rows are not the prefill's {len(ops)} ops in its order")

    @classmethod
    def of_record(cls, pi: Any, store: Any, rid: str, arrive: int) -> Served:
        """Request ``rid``'s served values from its `ProgramIndex` and the record's `CommittedStore`, arrived at engine step
        ``arrive``."""
        rows = []
        for r in pi.rows:
            d = pi.describe(r)
            if d is None or not d["members"]:
                raise ValueError(f"{rid}: row {r['i']} ({r['spec']}) has no committed member")
            rows.append({**d, "spec": r["spec"]})
        return cls(rows, lambda s, op, m, lo, hi: store.words(rid, arrive + s, op, m, lo, hi))

    def __call__(self, p: int, k: int, node: Any) -> list[np.ndarray] | None:
        """Position ``p``'s node ``k`` (``node``, its index within its position) as served, None where the engine didn't
        compute it."""
        if k >= len(self.prefill):
            raise ValueError(f"node {k} of a position, past the served request's {len(self.prefill)} ops a position")
        if p < self.LP:
            v = self.prefill[k]
            d = v[p] if len(v) == self.LP else v[0] if p == self.LP - 1 else None
        else:
            d = self.decode[p - self.LP][k] if p - self.LP < len(self.decode) else None
        if d is None:
            return None
        if d["spec"] != descriptor_id(node.fn):
            raise ValueError(f"position {p}'s node {k} is {descriptor_id(node.fn)}, served as {d['spec']} ({d['op_path']})")
        members = list(d["members"])
        if len(members) > 1:
            if not all(str(m[0]).isdigit() for m in members):
                raise ValueError(f"{d['op_path']}: members {[m[0] for m in members]}, not its return fields by index")
            members.sort(key=lambda m: int(m[0]))
        out = []
        for m, width, _dtype in members:
            a, why = self.read(int(d["step"]), str(d["op_path"]), str(m), int(d["row"]) * int(width), (int(d["row"]) + 1) * int(width))
            if a is None:
                raise ValueError(f"position {p}'s node {k}: {why}")
            out.append(a)
        return out


class Rows:
    """Every unit's output rows of ``lowered``'s circuit (module docstring) as u16 words by template and output, with each
    unit's source and an embedding read's opening ports, from each slot's `Slot` (in slot order) and the circuit's
    registered values by parameter (`weight_values` and `slot_rows_value`), of which an embedding read opens its table's
    (its ``rows``, ``salts`` and ``path``). Refused unless every unit's every row is filled once."""

    def __init__(self, lowered: Lowered, slots: Sequence[Slot], registered: Mapping[str, Any]) -> None:
        if len(slots) != len(lowered.programs):
            raise ValueError(f"{len(slots)} slots' values for {len(lowered.programs)} class Programs")
        c = self.circuit = lowered.circuit
        self.lowered = lowered
        tables = {t.name: t.param for t in (c.registered.tables if c.registered else ())}
        self.tables: dict[str, Any] = {}
        for o in c.registered.openings if c.registered else ():
            if tables[o.table] not in registered:
                raise ValueError(f"{o.reader} opens table {o.table}: give its parameter {tables[o.table]}'s registered value")
            if self.tables.setdefault(o.reader, registered[tables[o.table]]) is not registered[tables[o.table]]:
                raise ValueError(f"{o.reader}: units opening two tables")
        self.fns = dict(zip(c.ids, (fn for fn, _ in c.templates)))
        self.lays = {tid: IF.layout(fn) for tid, fn in self.fns.items()}
        n = self.n = dict(zip(c.ids, (k for _, k in c.templates)))
        self.outs = {t: [np.zeros((n[t], w), dtype=np.uint16) for _, w in lay.outs] for t, lay in self.lays.items()}
        self.held = {t: np.zeros((n[t], len(lay.outs)), dtype=bool) for t, lay in self.lays.items()}
        self.source = {t: np.full(n[t], EVALUATED, dtype=np.int8) for t in self.lays}
        ports = {(o.reader, p) for o in (c.registered.openings if c.registered else ()) for p in o.ports}
        self.inputs = {t: {p: np.zeros((n[t], w), dtype=np.uint16) for p, w in lay.rows if (t, p) in ports}
                       for t, lay in self.lays.items()}
        for P, S in zip(lowered.programs, slots):
            self._slot(P, S)
        self._gathers()
        if bad := {t: int((~h).any(axis=1).sum()) for t, h in self.held.items() if not h.all()}:
            raise ValueError(f"units with rows no node or Definition fills, by template: {bad}")

    def _put(self, tid: str, unit: int, out: str, rows: np.ndarray, src: int) -> None:
        o, k = int(out[1:]), len(rows)
        if self.held[tid][unit:unit + k, o].any():
            raise ValueError(f"{tid} units [{unit}, {unit + k}) {out}: filled twice")
        self.outs[tid][o][unit:unit + k] = rows
        self.held[tid][unit:unit + k, o] = True
        self.source[tid][unit:unit + k] = np.minimum(self.source[tid][unit:unit + k], src)

    def _slot(self, P: Any, S: Slot) -> None:
        nodes = self.lowered.fn.body.nodes
        for j, pieces in P.outs.items():
            w = _words(S.values[j])
            for lo, hi, r in pieces:
                self._put(r.tid, r.unit, r.out, w[lo:hi].reshape(-1, r.words), S.source[j])
        emb = [j for j, node in enumerate(nodes) if node.fn.name == "Embedding"]
        for p, (j, (feed, row)) in enumerate(zip(emb, P.tokens)):
            r, t = P.outs[j][0][2], S.tokens[p]
            self._put(r.tid, r.unit, "o1", np.ones((1, 1), dtype=np.uint16), EVALUATED)
            if t < nodes[j].fn.bindings["V"]:
                ins, table = self.inputs[r.tid], self.tables[r.tid]
                ins["row"][r.unit] = np.asarray(table.rows[t], dtype=np.uint16)
                ins["salt"][r.unit] = _be_words(table.salts[t])
                ins["path"][r.unit] = _be_words(table.path(t))
            if row is not None:
                self._put(*row, "o0", _i32(S.row[:1]), EVALUATED)
                self._put(*row, "o1", _i32(S.row[1:]), EVALUATED)
            if feed is not None:
                self._put(*feed, "o0", _i32([t]), EVALUATED)
                self._put(*feed, "o1", _i32(S.carried[p]), EVALUATED)
        for j, (chunks, x0, mask) in P.selects.items():
            logits = S.operand(nodes[j].args[0])
            if logits.dtype.itemsize != 2:
                raise ValueError(f"node {j}: logits of {logits.dtype}, not bf16 words")
            lo = 0
            for ct, cu in chunks:
                C = self.fns[ct].bindings["C"]
                best, i = chunk_best(logits[lo:lo + C])
                self._put(ct, cu, "o0", np.array([[best]], dtype=np.uint16), EVALUATED)
                self._put(ct, cu, "o1", _i32([i]), EVALUATED)
                lo += C
            if lo != len(logits):
                raise ValueError(f"node {j}: chunks of {lo} logits, not its {len(logits)}")
            self._put(*x0, "o0", logits[:1].astype(np.uint16).reshape(1, 1), EVALUATED)
            self._put(*mask, "o0", _i32([IO.mask_reference(S.row[0], int(S.values[j][0][0]))]), EVALUATED)

    def _gathers(self) -> None:
        """Each `AllGather_v1` unit's output, its parts' rows in order, once every part's writer holds its row: the gathers
        of gathers a pass later than the gathers they read."""
        pending = {t for t, fn in self.fns.items() if fn.name == "AllGather"}
        wires: dict[str, list[IF.Wire]] = {}
        for w in self.circuit.wires:
            if w.reader in pending:
                wires.setdefault(w.reader, []).append(w)
        while pending:
            progress = False
            for tid in sorted(pending):
                W, N, n = self.fns[tid].bindings["WORLD"], self.fns[tid].bindings["N"], self.n[tid]
                out, have = np.zeros((n, W * N), dtype=np.uint16), np.zeros((n, W), dtype=bool)
                wired = np.zeros((n, W), dtype=bool)
                for w in wires.get(tid, ()):
                    i = np.arange(w.units, dtype=np.int64)
                    u, r, s, o = w.at + i // w.cut[1], w.cut[0] + i % w.cut[1], w.source(i), int(w.out[1:])
                    out[u[:, None], r[:, None] * N + np.arange(N)] = self.outs[w.writer][o][s]
                    have[u, r], wired[u, r] = self.held[w.writer][s, o], True
                if not wired.all():
                    raise ValueError(f"{tid}: parts of {int((~wired).any(axis=1).sum())} units no wire reads")
                new = have.all(axis=1) & ~self.held[tid][:, 0]
                if new.any():
                    self.outs[tid][0][new] = out[new]
                    self.held[tid][new, 0] = True
                    progress = True
                if self.held[tid][:, 0].all():
                    pending.discard(tid)
            if pending and not progress:
                raise ValueError(f"gathers {sorted(pending)}: parts whose writers no node or gather fills")

    def steps(self, tid: str) -> np.ndarray:
        """Each unit of ``tid``'s step (its lowering's label)."""
        out = np.full(self.n[tid], -1, dtype=np.int64)
        for at, count, _slot, step in self.lowered.spans.get(tid, ()):
            out[at:at + count] = step
        return out

    def record(self, win: IF.Window) -> None:
        """Every unit into ``win``, template by template in unit order, in runs of one step and one source, each labelled
        ``{"step", "source"}``: each input by reference but an opening's ports, each port in its element width."""
        for tid, fn in self.fns.items():
            lay, steps, src = self.lays[tid], self.steps(tid), self.source[tid]
            key = steps * len(SOURCES) + src
            ends = [0, *(np.flatnonzero(np.diff(key)) + 1).tolist(), self.n[tid]]
            ins_w, outs_w = lay.widths[:len(lay.rows)], lay.widths[len(lay.rows):]
            for a, b in zip(ends, ends[1:]):
                ins = [None if p not in self.inputs[tid] else _elements(self.inputs[tid][p][a:b], w)
                       for (p, _), w in zip(lay.rows, ins_w)]
                outs = [_elements(o[a:b], w) for o, w in zip(self.outs[tid], outs_w)]
                win.record(fn, ins, outs, {"step": int(steps[a]), "source": SOURCES[int(src[a])]})


def _elements(words: np.ndarray, width: int) -> np.ndarray:
    """``(n, words)`` u16 words as ``width``-bit elements, the same little-endian bytes (`interface.words`' inverse)."""
    return np.ascontiguousarray(words, dtype=np.uint16).view(f"<u{width // 8}")


def weight_values(circuit: IF.Circuit, program: str, fields: Callable[[str], np.ndarray], key: bytes,
                  nonce: int = 0) -> dict[str, G.Registered]:
    """Every weight parameter ``circuit`` reads registered, committed once before a period's first window's coins and held
    across its windows: a weight's parameter (`window_builder.weight_param`) its words in rows of the reading port's
    (``row/v2``), a table's (`window_builder.table_param`) its rows (``row-seg``), each under the committer's secret
    ``key`` (`pouw.h1.registered.commit`), the i-th parameter in name order at salt nonce ``nonce + i``, so the verifier
    learns only each weight's size and hiding root. ``fields(label)``: a weight's elements in their unsigned width."""
    out = {}
    params = circuit.registered.params if circuit.registered else {}
    for i, name in enumerate(sorted(n for n in params if n != SLOT_ROWS)):
        body = name[len(WEIGHTS) + 1:] if name.startswith(f"{WEIGHTS}/") else ""
        label, _, words = body.removesuffix("/table").rpartition("@")
        table = body.endswith("/table")
        if not words.isdigit() or name != (table_param if table else weight_param)(label, int(words)):
            raise ValueError(f"{name}: no weight's registered parameter nor the slot rows")
        w = np.ascontiguousarray(fields(label)).view("<u2")
        if w.size != params[name]:
            raise ValueError(f"{name}: weight {label} of {w.size} words, its parameter's {params[name]}")
        out[name] = H1R.commit(program, name, w.reshape(-1, int(words)), key, nonce + i, keep_rows=True,
                               schema=G.SCHEMA_SEG if table else G.SCHEMA_V2)
    return out


def slot_rows_value(program: str, slot_rows: Sequence[Sequence[int]], key: bytes, nonce: int, *,
                    window: str) -> G.Registered:
    """`SLOT_ROWS`: the slots' rows (``row/v2``), a per-window value under the window's root ``window``, committed under
    the committer's secret ``key`` at salt nonce ``nonce``, which no other value under ``key`` takes."""
    return H1R.commit(program, SLOT_ROWS, np.concatenate([_i32(r) for r in slot_rows]), key, nonce, keep_rows=True,
                      window=window)


__all__ = ["COMPUTED", "EVALUATED", "Rows", "SERVED", "SOURCES", "Served", "Slot", "positions", "slot_rows_value",
           "weight_values"]
