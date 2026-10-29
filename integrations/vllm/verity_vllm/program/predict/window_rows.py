"""A window's rows: every unit of a class circuit (`window_circuit.lower`) given its output rows from a served window's
values, ready for `interface.Window.record`.

A node of a slot's class Program has one value, its return fields: the served value where the engine computed it (its
request's tapped words in the sealed record, `Served`), else the class Program's own, its family's row kernel
(`numerics.rows.ROWS`) on the node's operands. The engine never computes the positions past a request's last output, nor,
at a prompt position before the last, the nodes prefill prunes (the last layer from its query on, the final norm, the LM
head and the token select), so those are computed. An embedding's value is the table's row at its position's token, and a
token select's the first maximum of its logits, whichever source gives it; a served one that isn't is refused by name. So
is a node the engine computed under other statics of its Definition than the class node's whose value isn't the class
node's on its operands: FA3's key block follows its launch's query rows (`targets.fa3_kblock_n`), so a prefill's
attention isn't the class Program's one-row decode's, though at as few keys as one block of either holds the two agree.

A unit its node's lowering filled holds its rows of that value (`_Program.outs`). Every other unit holds its Definition's
function of the window's values:
- a slot's `ScanSlots_v2` every lane's first token and slot, occupancy word and head of the scan state row at its anchor
  (`io_shell.scan_reference`; its lane's row is the slot's, `scan_slots`), every later position's `TokenFeed_v1` below the
  prompt cap its token and the slot it carries (`io_shell.feed_reference`), each token select's `OutCollect_v1` the select
  collected on from the lane's head (`io_shell.collect_reference`), and the last position's `SlotEntry_v1` the slot's entry
  (`io_shell.entry_reference`);
- each token select's group units each group's first maximum above -inf of its logits and its index in the group
  (`chunk_best`), and its ``x0`` unit ``logits[0]``;
- each embedding read's verdict 1, its opening the table's row, salt and path at the token (the table's registered value,
  core's ``one_stage.registered.Registered`` of ``row-seg`` rows or one like it);
- each attention cut by key block (`window_circuit.CUTS`), every Call of a head but its last (the node's output) its
  Definition on the node's operands (`evaluate_batch`): the head's query, its KV group's keys and values in the Call's
  range, and the earlier Calls' running state.

`Rows.record` gives a `Window` each template's units in runs of one step, one slot and one form (`FORMS`: a node's value,
or its Definition's function of the window's values), every input by reference but an opening's ports. The lowering
fixes all three, so a label is class structure and never a value: which slots are occupied, at what lengths, and where a
node's value came from (`SOURCES`) are the scan state's and the rows' own, committed hiding, and the labels show none of
them. The window takes `FORMS` as its ``forms`` (`interface.Window.record`). The index they make is the prover's, bound
by its hiding leaf under `index_salt`.
"""
from __future__ import annotations

import bisect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verity.core.protocols.certifier import ingress_scan as IS
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.refs import runs
from verity.ml.commitments.gates import embedding as E
from verity.ml.definitions.numerics.rows import ROWS
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.definitions.registry.b1 import NEG_INF_BF16
from verity.ml.windows import interface as IF
from verity_vllm.protocols.pouw.h1 import registered as H1R
from verity_vllm.query.manifest import format as RM

from . import scan_state as SS
from . import window_weights as WW
from .window_builder import WEIGHTS
from .window_circuit import Lowered, WindowClass, _spans

#: a unit's source: its node's served value, the class Program's value at a node the engine didn't compute, or its
#: Definition's function of the window's values. It follows occupancy and lengths, so it is the committer's own bookkeeping
#: (`Rows.source`) and never a label
SOURCES = ("served", "computed", "evaluated")
SERVED, COMPUTED, EVALUATED = range(3)
#: a unit's form, its runs' label beside its step and slot: a unit a node's lowering filled (`_Program.outs`) or one its
#: Definition fills; the lowering fixes it
FORMS = ("node", "evaluated")

#: a weight's elements [lo, hi) in its leaves' unsigned width: ``weights(label, lo, hi)``
Weights = Callable[[str, int, int], np.ndarray]
#: the salt nonce of window 0's stand-in scan state under the committer's key (`standin_state`), each next window's the
#: next: above every weight parameter's (`window_weights.weight_nonces`, from 2^62) and below 2^63
STATE_NONCE = WW.WEIGHTS_NONCE + (1 << 32)
#: the salt nonce of window 0's index (`index_salt`), each next window's the next: above every window's scan state's
INDEX_NONCE = STATE_NONCE + (1 << 32)


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


def _pinned(family: str) -> str | None:
    """What the circuit pins a node of Definition ``family``'s value to, which a served value is held to: an embedding's
    its opening of the table, a token select's its groups' scan; None where a served value is taken as served."""
    if family == "Embedding":
        return "the table's row at its token"
    if family == "TokenSelect":
        return "the first maximum of its logits"
    return None


class Slot:
    """Slot ``row``'s class Program ``fn`` of class ``cls`` evaluated (module docstring): ``row`` its lane's row of the scan
    state row at its anchor (occupancy, prompt length, ids; `scan_slots`), ``weights`` the weights record's
    fields and ``served`` its request's served values,
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

    def _node(self, j: int, node: Any, served: tuple[str, list[np.ndarray]] | None) -> None:
        fam, fields, what = node.fn.name, _fields(node.fn), f"node {j} ({node.fn.id}) at position {self.position[j]}"
        if fam == "Embedding":
            self._token(self.position[j], node.args[0])
        pin = _pinned(fam)
        if served is not None and served[0] != descriptor_id(node.fn):
            pin = f"its Definition's value on its operands (served as {served[0]})"
        got = None if served is None else served[1]
        want = self._evaluate(node, fields, what) if got is None or pin else None
        if got is not None:
            if len(got) != len(fields):
                raise ValueError(f"{what}: served as {len(got)} members, not its {len(fields)} return fields")
            got = [_cast(a, ty, f"{what}, served") for a, ty in zip(got, fields)]
            if want is not None and not all(np.array_equal(a, b) for a, b in zip(got, want)):
                raise ValueError(f"{what}: served as {got[0][:4].tolist()}..., not {pin}")
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


def _op(d: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """A served row's op: its op path and its members' names."""
    return str(d["op_path"]), tuple(str(m[0]) for m in d["members"])


def _sites(ops: Sequence[tuple[str, tuple[str, ...]]]) -> dict[tuple[str, tuple[str, ...]], tuple[str, tuple[str, ...], int]]:
    """Each of a step's ops, in Program order, -> its site: its op path less its last ``/`` part (a call boundary's fx node,
    which counts on over the request's steps, `query.manifest.bind`), its members' names, and its ordinal among the step's
    ops there."""
    seen: dict[tuple[str, tuple[str, ...]], int] = {}
    out = {}
    for op, members in ops:
        at = (op.rsplit("/", 1)[0], members)
        out[(op, members)] = (*at, seen.get(at, 0))
        seen[at] = seen.get(at, 0) + 1
    return out


@dataclass
class Served:
    """One request's served values by class node: its Program's rows as `ProgramIndex.describe` gives them, each with its
    row's ``spec`` (``{spec, step, op_path, members, row}``, its step the request's), ``read(step, op_path, member, lo,
    hi)``, a member's elements [lo, hi) at a request step (`CommittedStore.words` at the request's arrival plus it), and
    ``bound(step, op_path, member)``, the elements the record binds that member at.

    The request's prefill is its step 0: each op (its op path and members) a row a prompt position, or one row, its last
    position's, where prefill prunes it; the ops in the order of their last prompt position's rows, the one position that
    computes them all, which is the class Program's node order within a position however the prefill interleaves the
    others. A pruned op's one row is its member's last of the prompt's where the record binds the member at a row a prompt
    position (the engine computes the last layer at them all; only the logits gather a row a request), else its only one.
    Step s ≥ 1 is position LP − 1 + s, a row each of those ops, matched to them by site (`_sites`), since a call
    boundary's op path names an fx node numbered over the request's steps. A row of another Definition than its class
    node's is refused by name; one of its Definition at other statics is `Slot`'s to hold to the class node's value."""
    rows: Sequence[Mapping[str, Any]]
    read: Callable[[int, str, str, int, int], tuple[np.ndarray | None, str]]
    bound: Callable[[int, str, str], int]
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
        last: dict[tuple[str, tuple[str, ...]], int] = {}
        for i, d in enumerate(steps.get(0, ())):
            ops.setdefault(_op(d), []).append(d)
            last[_op(d)] = i
        order = sorted(ops, key=last.__getitem__)
        self.prefill = [sorted(ops[o], key=lambda d: int(d["row"])) for o in order]
        self.LP = max((len(v) for v in self.prefill), default=0)
        for (op, _), v in zip(order, self.prefill):
            if [int(d["row"]) for d in v] not in (list(range(self.LP)), [0]):
                raise ValueError(f"{op}: prefill rows {[d['row'] for d in v]}, not one a prompt position nor the last one's")
        site = _sites(order)
        sites = set(site.values())
        self.decode = []
        for s in range(1, len(steps)):
            by: dict[tuple[str, tuple[str, ...]], Mapping[str, Any]] = {}
            for d in steps[s]:
                if by.setdefault(_op(d), d) is not d:
                    raise ValueError(f"step {s}: op {_op(d)} served at two rows")
            at = {v: by[o] for o, v in _sites(list(by)).items()}
            if missing := [o for o in order if site[o] not in at]:
                raise ValueError(f"step {s}: none of the prefill's op {missing[0]} ({len(missing)} ops)")
            if extra := [_op(d) for v, d in at.items() if v not in sites]:
                raise ValueError(f"step {s}: op {extra[0]}, which the prefill never computes ({len(extra)} ops)")
            self.decode.append([at[site[o]] for o in order])

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

        def bound(s: int, op: str, m: str) -> int:
            ends = [e["identity"].get("element_range", (0, None))[1] for e in store.entries(rid, arrive + s, op, m)[0]]
            return max((int(x) for x in ends if x is not None), default=0)

        return cls(rows, lambda s, op, m, lo, hi: store.words(rid, arrive + s, op, m, lo, hi), bound)

    def __call__(self, p: int, k: int, node: Any) -> tuple[str, list[np.ndarray]] | None:
        """Position ``p``'s node ``k`` (``node``, its index within its position) as served, its row's spec and its values;
        None where the engine didn't compute it."""
        if k >= len(self.prefill):
            raise ValueError(f"node {k} of a position, past the served request's {len(self.prefill)} ops a position")
        pruned = False
        if p < self.LP:
            v = self.prefill[k]
            pruned = len(v) != self.LP
            d = v[p] if not pruned else v[0] if p == self.LP - 1 else None
        else:
            d = self.decode[p - self.LP][k] if p - self.LP < len(self.decode) else None
        if d is None:
            return None
        if RM.spec_family(d["spec"]) != RM.spec_family(descriptor_id(node.fn)):
            raise ValueError(f"position {p}'s node {k} is {descriptor_id(node.fn)}, served as {d['spec']} ({d['op_path']})")
        members = list(d["members"])
        if len(members) > 1:
            if not all(str(m[0]).isdigit() for m in members):
                raise ValueError(f"{d['op_path']}: members {[m[0] for m in members]}, not its return fields by index")
            members.sort(key=lambda m: int(m[0]))
        step, op = int(d["step"]), str(d["op_path"])
        row = int(d["row"])
        if pruned:
            n = {self.bound(step, op, str(m)) / int(width) for m, width, _dtype in members}
            if n not in ({self.LP}, {1}):
                raise ValueError(f"{op}: its one prefill row's members bound at {sorted(n)} rows, not a row a prompt position "
                                 f"({self.LP}) nor one")
            row = self.LP - 1 if n == {self.LP} else 0
        out = []
        for m, width, _dtype in members:
            a, why = self.read(step, op, str(m), row * int(width), (row + 1) * int(width))
            if a is None:
                raise ValueError(f"position {p}'s node {k}: {why}")
            out.append(a)
        return str(d["spec"]), out


class Rows:
    """Every unit's output rows of ``lowered``'s circuit (module docstring) as u16 words by template and output, with each
    unit's source and an embedding read's opening ports, from each slot's `Slot` (in slot order) and the circuit's
    registered values by parameter (`window_weights.weight_values` with their rows kept, and the scan state's,
    `scan_state.SCAN_STATE`, with its rows: the network certifier's, a row a bucket of the window), of which an embedding
    read opens its table's (its ``rows``, ``salts`` and ``path``) and a slot's `ScanSlots_v2` reads the scan state's row at
    its anchor. Refused unless every unit's every row is filled once."""

    def __init__(self, lowered: Lowered, slots: Sequence[Slot], registered: Mapping[str, Any]) -> None:
        if len(slots) != len(lowered.programs):
            raise ValueError(f"{len(slots)} slots' values for {len(lowered.programs)} class Programs")
        c = self.circuit = lowered.circuit
        self.lowered = lowered
        if getattr(registered.get(SS.SCAN_STATE), "rows", None) is None:
            raise ValueError(f"give the scan state's registered value, {SS.SCAN_STATE}, with its rows")
        self.state = registered[SS.SCAN_STATE].rows
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
        self.node = {t: np.zeros(n[t], dtype=bool) for t in self.lays}
        ports = {(o.reader, p) for o in (c.registered.openings if c.registered else ()) for p in o.ports}
        self.inputs = {t: {p: np.zeros((n[t], w), dtype=np.uint16) for p, w in lay.rows if (t, p) in ports}
                       for t, lay in self.lays.items()}
        for P, S in zip(lowered.programs, slots):
            self._slot(P, S)
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
                self.node[r.tid][r.unit:r.unit + (hi - lo) // r.words] = True
        emb = [j for j, node in enumerate(nodes) if node.fn.name == "Embedding"]
        for p, (j, feed) in enumerate(zip(emb, P.tokens)):
            r, t = P.outs[j][0][2], S.tokens[p]
            self._put(r.tid, r.unit, "o1", np.ones((1, 1), dtype=np.uint16), EVALUATED)
            if t < nodes[j].fn.bindings["V"]:
                ins, table = self.inputs[r.tid], self.tables[r.tid]
                ins["row"][r.unit] = np.asarray(table.rows[t], dtype=np.uint16)
                ins["salt"][r.unit] = _be_words(table.salts[t])
                ins["path"][r.unit] = _be_words(table.path(t))
            if feed is not None:
                self._put(*feed, "o0", _i32([t]), EVALUATED)
                self._put(*feed, "o1", _i32(S.carried[p]), EVALUATED)
        picks = []
        for j, (groups, x0) in P.selects.items():
            logits = S.operand(nodes[j].args[0])
            if logits.dtype.itemsize != 2:
                raise ValueError(f"node {j}: logits of {logits.dtype}, not bf16 words")
            lo = 0
            for ct, cu in groups:
                C, G = self.fns[ct].bindings["C"], self.fns[ct].bindings["G"]
                parts = [chunk_best(logits[lo + a:lo + min(a + G, C)]) for a in range(0, C, G)]
                self._put(ct, cu, "o0", np.array([[b for b, _ in parts]], dtype=np.uint16), EVALUATED)
                self._put(ct, cu, "o1", _i32([i for _, i in parts]), EVALUATED)
                lo += C
            if lo != len(logits):
                raise ValueError(f"node {j}: group units of {lo} logits, not its {len(logits)}")
            self._put(*x0, "o0", logits[:1].astype(np.uint16).reshape(1, 1), EVALUATED)
            picks.append(int(S.values[j][0][0]))
        self._scan(P, S, picks)
        for j, (plan, calls) in P.heads.items():
            self._heads(nodes[j], plan, calls, S)

    def _scan(self, P: Any, S: Slot, picks: Sequence[int]) -> None:
        """The slot's scan units: its `ScanSlots_v2` every lane's first token and slot, occupancy word and head of the scan
        state row at its anchor (its lane's row the slot's, refused by name where it isn't), each position's `OutCollect_v1`
        its select collected on from the lane's head, and its `SlotEntry_v1` its entry of the scan's Programs' row."""
        tid, u = P.scan_unit
        for k, lane in enumerate(_lanes(P, self.state, IO.scan_reference)):
            for i, v in enumerate(lane):
                self._put(tid, u, f"o{4 * k + i}", _i32(v if isinstance(v, list) else [v]), EVALUATED)
        row, a = _lanes(P, self.state)[P.lane]
        if row != S.row:
            raise ValueError(f"{P.name}: its slot's row isn't lane {P.lane}'s of the scan state row at bucket {P.anchor}")
        KO = P.cls.outputs
        for p, ((ctid, cu), tok) in enumerate(zip(P.collects, picks, strict=True)):
            a = IO.collect_reference(a, tok, KO, p == 0)
            self._put(ctid, cu, "o0", _i32(a), EVALUATED)
        e = P.entry.bindings
        entry = IO.entry_reference(S.row[0], a, KO, e["IDB"], e["NP"], e["STOP"])
        self._put(*P.entry_unit, "o0", np.array([entry], dtype=np.uint16), EVALUATED)

    def _heads(self, node: Any, plan: Any, calls: Sequence[tuple[str, int]], S: Slot) -> None:
        """An attention cut by key block: every Call but its last (whose output is the node's) its Definition on the node's
        operands, a unit a head: the head's query, its KV group's keys and values in the Call's range, and earlier Calls'
        rows."""
        s = node.fn.bindings
        T, NH, KVH, D = s["T"], s["NH"], s["KVH"], s["D"]
        q, kc, vc = (S.operand(a) for a in node.args)
        g = np.arange(NH) // (NH // KVH)
        ops = {"q": q.reshape(NH, D), "k": kc.reshape(T, KVH, D), "v": vc.reshape(T, KVH, D)}
        got: list[dict[str, np.ndarray]] = []
        for (fn, ports), (tid, u) in zip(plan[:-1], calls):
            args = []
            for _port, (kind, x) in ports:
                args.append(ops["q"] if kind == "q" else got[x[0]][x[1]] if kind == "state"
                            else ops[kind][x[0]:x[1]][:, g].transpose(1, 0, 2).reshape(NH, -1))
            flat, at, outs = np.asarray(evaluate_batch(fn, *args)).reshape(NH, -1), 0, {}
            lay = self.lays[tid]
            for (o, words), w in zip(lay.outs, lay.widths[len(lay.rows):]):
                k = words * 16 // w
                outs[o] = flat[:, at:at + k].astype(f"<u{w // 8}")
                self._put(tid, u, o, outs[o].view("<u2").reshape(NH, words), EVALUATED)
                at += k
            got.append(outs)

    def _span(self, tid: str, k: int) -> np.ndarray:
        out = np.full(self.n[tid], -1, dtype=np.int64)
        for span in self.lowered.spans.get(tid, ()):
            out[span[0]:span[0] + span[1]] = span[k]
        return out

    def steps(self, tid: str) -> np.ndarray:
        """Each unit of ``tid``'s step (its lowering's label)."""
        return self._span(tid, 3)

    def slots(self, tid: str) -> np.ndarray:
        """Each unit of ``tid``'s slot (its lowering's), -1 for a unit of no slot's."""
        return self._span(tid, 2)

    def record(self, win: IF.Window) -> None:
        """Every unit into ``win``, template by template in unit order, in runs of one step, one slot and one form, each
        labelled ``{"step", "slot", "form"}`` (module docstring): each input by reference but an opening's ports, each port
        in its element width."""
        for tid, fn in self.fns.items():
            lay, steps, slots, node = self.lays[tid], self.steps(tid), self.slots(tid), self.node[tid]
            cut = (np.diff(steps) != 0) | (np.diff(slots) != 0) | (np.diff(node) != 0)
            ends = [0, *(np.flatnonzero(cut) + 1).tolist(), self.n[tid]]
            ins_w, outs_w = lay.widths[:len(lay.rows)], lay.widths[len(lay.rows):]
            for a, b in zip(ends, ends[1:]):
                ins = [None if p not in self.inputs[tid] else _elements(self.inputs[tid][p][a:b], w)
                       for (p, _), w in zip(lay.rows, ins_w)]
                outs = [_elements(o[a:b], w) for o, w in zip(self.outs[tid], outs_w)]
                win.record(fn, ins, outs, {"step": int(steps[a]), "slot": int(slots[a]), "form": FORMS[0 if node[a] else 1]})


def _elements(words: np.ndarray, width: int) -> np.ndarray:
    """``(n, words)`` u16 words as ``width``-bit elements, the same little-endian bytes (`interface.words`' inverse)."""
    return np.ascontiguousarray(words, dtype=np.uint16).view(f"<u{width // 8}")


def _lanes(P: Any, state: Sequence[Any], of: Callable[..., list] = IO.slots_reference) -> list[Any]:
    """Every lane of ``state``'s row at class Program ``P``'s anchor: its row and head (`io_shell.slots_reference`), or
    ``of``'s (`io_shell.scan_reference`: `ScanSlots_v2`'s outputs)."""
    b = P.scan.bindings
    row = np.asarray(state[P.anchor], dtype=np.uint16).reshape(-1)
    return of(row.tolist(), b["NP"], b["IDB"], b["K0"], b["K"], b["OCC"], b["LP"], b["MAX"], b["IDS"])


def scan_slots(lowered: Lowered, state: Sequence[Any]) -> list[tuple[list[int], list[int]]]:
    """Each class Program's slot as the scan state's rows ``state`` (`scan_state.SCAN_STATE`'s, a row a bucket of the
    window) give it, in slot order: its lane's row (occupancy, prompt length, ids; a `Slot`'s ``row``) and head (prompt
    length, ``max_tokens``) of the row at its anchor."""
    return [_lanes(P, state)[P.lane] for P in lowered.programs]


def state_value(program: str, cls: WindowClass, rows: Sequence[Sequence[int]], key: bytes, index: int, *,
                window: str) -> G.Registered:
    """`scan_state.SCAN_STATE`'s value: ``rows``, the scan's state rows a window of ``cls`` reads (row a the state before
    bucket a, `ingress_scan.window_state`'s bits), as ``row/v2`` rows committed under the committer's secret ``key`` at salt
    nonce `STATE_NONCE` plus the window's ``index``, a per-window value under the window's root ``window``. It is the
    prover's side of the tree (its rows, salts and levels) for the window's prover and the scan's alike: the scan's state
    ports open their rows in it, by its root and domain."""
    if not 0 <= index < 1 << 32:
        raise ValueError(f"window {index}: a window's index is a u32")
    bits = np.asarray(rows, dtype=np.uint16)
    if bits.shape != (cls.buckets, IS.state_bits(cls.scan)) or bits.max(initial=0) > 1:
        raise ValueError(f"a window of {cls.buckets} buckets reads {cls.buckets} state rows of {IS.state_bits(cls.scan)} "
                         f"bits, not {bits.shape}")
    words = (bits.reshape(cls.buckets, -1, 16) << np.arange(16, dtype=np.uint16)).sum(axis=2, dtype=np.uint16)
    return H1R.commit(program, SS.SCAN_STATE, words, key, STATE_NONCE + index, keep_rows=True, window=window)


def standin_state(program: str, cls: WindowClass, slots: Sequence[tuple[Sequence[int], int]], key: bytes, index: int, *,
                  window: str) -> G.Registered:
    """`state_value` where no scan ran: the rows that give slot k its row (occupancy, prompt length, ids) and
    ``max_tokens``, ``slots[k]``, at its anchor and lane (`scan_slots`), every other field zero."""
    scan = cls.scan
    states = [IS.empty(scan) for _ in range(cls.buckets)]
    for k, (row, max_tokens) in enumerate(slots):
        s = states[cls.anchor(k)].slots[cls.lane(k)]
        s.occ, s.fields, s.max_tokens = int(row[0]), int(row[1]), int(max_tokens)
        for p, v in enumerate(row[2:]):
            s.ids[IS.position(scan, p)] = int(v)
    return state_value(program, cls, [IS.encode(scan, st) for st in states], key, index, window=window)


def index_salt(key: bytes, index: int) -> bytes:
    """Window ``index``'s index salt (`interface.Window.write`'s ``index_salt``, `rows.index_leaf`): drawn under the
    committer's secret ``key`` at salt nonce `INDEX_NONCE` plus the window's ``index``, which no other value under ``key``
    takes, so the committer draws it again from its key alone."""
    if not 0 <= index < 1 << 32:
        raise ValueError(f"window {index}: a window's index is a u32")
    return H1R.salts(key, INDEX_NONCE + index, 1)


__all__ = ["COMPUTED", "EVALUATED", "FORMS", "INDEX_NONCE", "Rows", "SERVED", "SOURCES", "STATE_NONCE", "Served",
           "Slot", "index_salt", "positions", "scan_slots", "standin_state", "state_value"]
