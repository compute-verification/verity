"""A window's rows: every unit of a window's circuit (`window_circuit.lower`, a Program a served request, of its own
positions) given its output rows from the window's served values, ready for `interface.Window.record`.

A node of a request's Program has one value, its return fields: the served value where the engine computed it (its
request's tapped words in the sealed record, `Served`), else the Program's own, its family's row kernel
(`numerics.rows.ROWS`) on the node's operands. The engine never computes, at a prompt position before the last, the nodes
prefill prunes (the last layer from its query on, the final norm, the LM head and the token select), so those are
computed. An embedding's value is the table's row at its position's token, and a
token select's the first maximum of its logits, whichever source gives it; a served one that isn't is refused by name. So
is a node the engine computed under other statics of its Definition than the circuit node's whose value isn't the circuit
node's on its operands: FA3's key block follows its launch's query rows (`targets.fa3_kblock_n`), so a prefill's
attention isn't the Program's one-row decode's, though at as few keys as one block of either holds the two agree.

A unit its node's lowering filled holds its rows of that value (`_Program.outs`), and so does each add part of a fused norm
cut at its residual add (`_Program.norms`): part AT its words of the node's second field, the sum.
Every other unit holds its Definition's function of the window's values:
- a lane's `ScanSlots_v5` its first token and slot, occupancy word and head of the scan state row at its anchor
  (`io_shell.scan_reference`; a slot's lane's row is the slot's, `scan_slots`), every later position's `TokenFeed_v1` below the
  prompt bound its token and the slot it carries (`io_shell.feed_reference`), each token select's `SelectCollect_v1` its
  second row, the select collected on from the lane's head (`io_shell.collect_reference`), and the last position's
  `SelectEntry_v1` the slot's entry (`io_shell.entry_reference`);
- each token select's group units each group's first maximum above -inf of its logits and its index in the group
  (`chunk_best`), and its ``x0`` unit ``logits[0]``;
- each embedding read's verdict word, 1 below V and 0 at or past it (`EmbeddingRead_v3`), its opening the table's row, salt
  and path at the token (the table's registered value, ``registered_value.Registered`` of ``row-seg`` rows or one like it);
- each attention cut by key block (`window_circuit.CUTS`), every Call of a head but its last (the node's output) its
  Definition on the node's operands (`evaluate_batch`): the head's query, its KV group's keys and values in the Call's
  range, and the earlier Calls' running state.

`Rows.record` gives a `Window` each template's units in runs of one step, one slot and one form (`FORMS`: a node's value,
or its Definition's function of the window's values), every input by reference but an opening's ports. The lowering
fixes all three, so a label is class structure and never a value: which slots are occupied, at what lengths, and where a
node's value came from (`SOURCES`) are the scan state's and the rows' own, committed hiding, and the labels show none of
them. The window takes `FORMS` as its ``forms`` (`interface.Window.record`). The index they make is the prover's, bound
by its hiding leaf under the salt `Window.write` draws from the OS and keeps in ``index-salt.bin``.
"""
from __future__ import annotations

import bisect
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.refs import runs
from verity.commitments import randomness as RND
from verity.commitments import registered_value as RV
from verity.commitments.gates import embedding as E
from verity.commitments.periods import interface as IF
from verity.commitments.periods import rows as CS
from verity.commitments.periods.rows import registered as registered_commit

from verity_vllm.check.capture_plan import NODE, captured
from verity_vllm.definitions.numerics.rows import ROWS
from verity_vllm.definitions.numerics.token_select_rows import token_select_chunk_bests
from verity_vllm.definitions.registry import io_shell as IO
from verity_vllm.query.manifest import format as RM

from . import scan_state as SS
from .window_builder import WEIGHTS, _spans
from .window_circuit import Lowered, WindowClass
from .window_lowering import node_family

#: a unit's source: its node's served value, the Program's value at a node the engine didn't compute, or its
#: Definition's function of the window's values. It follows occupancy and lengths, so it is the committer's own bookkeeping
#: (`Rows.source`) and never a label
SOURCES = ("served", "computed", "evaluated")
SERVED, COMPUTED, EVALUATED = range(3)
#: a unit's form, its runs' label beside its step and slot: a unit a node's lowering filled (`_Program.outs`) or one its
#: Definition fills; the lowering fixes it
FORMS = ("node", "evaluated")

#: a weight's elements [lo, hi) in its leaves' unsigned width: ``weights(label, lo, hi)``
Weights = Callable[[str, int, int], np.ndarray]


def positions(fn: Any) -> list[int]:
    """Each node's position: a position begins at each embedding (`window_lowering.node_family`)."""
    out, p = [], -1
    for n in fn.body.nodes:
        p += node_family(n.fn) == "Embedding"
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
    """`TokenSelectChunk_v2{C}` on one chunk of bf16 words (`numerics.token_select_rows`)."""
    best, i = token_select_chunk_bests(x)
    return int(best), int(i)


def _pinned(family: str) -> str | None:
    """What the circuit pins a node of Definition ``family``'s value to, which a served value is held to: an embedding's
    its opening of the table, a token select's its groups' scan; None where a served value is taken as served."""
    if family == "Embedding":
        return "the table's row at its token"
    if family == "TokenSelect":
        return "the first maximum of its logits"
    return None


class Slot:
    """A served request's Program ``fn`` in a window of class ``cls`` evaluated (module docstring): ``row`` its lane's row of
    the scan state row at its anchor (occupancy, prompt length, ids; `scan_slots`), ``weights`` the weights record's
    fields and ``served`` its served values, refused unless ``fn`` is of its served positions. Each node's value (its
    return fields) and source, each
    position's token and, below the class's prompt bound, the slot its `TokenFeed_v1` reads and the one it carries on."""

    def __init__(self, fn: Any, cls: WindowClass, row: Sequence[int], weights: Weights, served: Served) -> None:
        self.fn, self.cls, self.weights, self.reader = fn, cls, weights, getattr(served, "reader", None)
        self.row = [int(x) & 0xFFFFFFFF for x in row]
        if len(self.row) != cls.prompt + 2:
            raise ValueError(f"a slot's row is (occupancy, prompt length, {cls.prompt} ids), not {len(self.row)} words")
        if self.row[1] != served.LP:
            raise ValueError(f"the served request's {served.LP} prompt ids, its slot row's {self.row[1]}")
        sig = dict(fn.signature()[0])
        self.params = [n for n, _ in fn.signature()[0]]
        self.labels = [label for label, _ in sig[WEIGHTS].fields]
        self.starts = [s[0] for s in _spans("weights", sig[WEIGHTS].fields)]
        self.position = positions(fn)
        if self.position[-1] + 1 != served.positions:
            raise ValueError(f"a Program of {self.position[-1] + 1} positions for a request served at {served.positions}")
        first: dict[int, int] = {}
        for j, p in enumerate(self.position):
            first.setdefault(p, j)
        self.values: list[list[np.ndarray]] = []
        self.source: list[int] = []
        self.tokens: list[int] = []
        self.slots: list[list[int] | None] = []
        self.carried: list[list[int] | None] = []
        for j, node in enumerate(fn.body.nodes):
            self._node(j, node, served(self.position[j], j - first[self.position[j]], node))

    def _node(self, j: int, node: Any, served: tuple[str, list[np.ndarray]] | None) -> None:
        fam, fields, what = node_family(node.fn), _fields(node.fn), f"node {j} ({node.fn.id}) at position {self.position[j]}"
        if fam == "Embedding":
            self._token(self.position[j], node.args[0])
        pin = _pinned(fam)
        if served is not None and served[0] != descriptor_id(node.fn):
            pin = f"its Definition's value on its operands (served as {served[0]})"
        got = None if served is None else served[1]
        if got is None:
            self.values.append(self._evaluate(node, fields, what))
            self.source.append(COMPUTED)
            return
        want = self._evaluate(node, fields, what) if pin else None
        if len(got) != len(fields):
            raise ValueError(f"{what}: served as {len(got)} members, not its {len(fields)} return fields")
        got = [_cast(a, ty, f"{what}, served") for a, ty in zip(got, fields)]
        if want is not None and not all(np.array_equal(a, b) for a, b in zip(got, want)):
            raise ValueError(f"{what}: served as {got[0][:4].tolist()}..., not {pin}")
        self.values.append(got)
        self.source.append(SERVED)

    def _token(self, p: int, tok: Any) -> None:
        """Position ``p``'s token (`window_lowering._Program._token`): the first position's its slot row's first id, below the
        prompt bound `TokenFeed_v1`'s of the slot carried and the previous select, past it the previous select."""
        NP = self.cls.prompt
        sel = None if p == 0 else int(self.operand(tok)[0])
        slot = self.row[1:] if p == 0 else self.carried[-1]
        nxt: list[int] | None = None
        if p < NP:
            assert slot is not None, f"position {p} is one of the {NP} prompt positions, and position {p - 1} carried no slot"
            t, nxt = IO.feed_reference(slot, sel, NP)
        else:
            assert sel is not None, "position 0 is past the prompt positions, and no select comes before it"
            t, slot = sel, None
        self.tokens.append(int(t))
        self.slots.append(slot)
        self.carried.append(nxt)

    def _evaluate(self, node: Any, fields: list[Any], what: str) -> list[np.ndarray]:
        if node_family(node.fn) == "Embedding":
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




def _respelt(rows: Sequence[Mapping[str, Any]], store: Any, rid: str, arrive: int) -> dict[tuple[int, str], str]:
    """Each (step, op path) of request ``rid``'s served rows, arrived at engine step ``arrive`` -> the op path the record
    binds it under at engine step ``arrive`` plus its step: its own where the record binds each of its members there; else,
    for a call boundary alone at its module under its op at its step, the record's one op path of that module and op for
    the request there, where that binds each of its members. The serve commits a batch's call boundary under one fx node,
    counted over its earliest arrival's steps, so a later arrival's is another count than its Program's; any other op path
    keeps its own, and its read is refused by name."""
    want: dict[tuple[int, str], set[str]] = {}
    for d in rows:
        want.setdefault((int(d["step"]), str(d["op_path"])), set()).update(str(m[0]) for m in d["members"])
    node = {key: x.groups() for key in want if (x := NODE.fullmatch(key[1]))}
    alone = Counter((s, g) for (s, _op), g in node.items())
    out = {}
    for (s, op), members in want.items():
        out[(s, op)] = op
        e, g = arrive + s, node.get((s, op))
        if g is None or alone[(s, g)] != 1 or store.resolve(op) != (op, 0) or all(store.has(rid, e, op, m) for m in members):
            continue
        same = [o for o in store.ops(rid, e) if (x := NODE.fullmatch(o)) and x.groups() == g]
        if len(same) == 1 and store.resolve(same[0]) == (same[0], 0) and all(store.has(rid, e, same[0], m) for m in members):
            out[(s, op)] = same[0]
    return out


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


#: a narrowed linear's member (`RM.NARROWED_MEMBER`): its widest rows' member and its width
_NARROWED = re.compile(RM.NARROWED_MEMBER.format(member="(.+)", width=r"(\d+)"))


def _fold_narrowed(ops: dict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]],
                   last: dict[tuple[str, tuple[str, ...]], int]) -> list[tuple[str, tuple[str, ...]]]:
    """Fold each prefill op of a narrowed linear's rows (members `RM.NARROWED_MEMBER`: the last layer's fused qkv on the
    prompt rows before the last, its keys' and values' columns) into the op of its widest rows at its op path, whose one
    row is the last position's: one op, a row a prompt position, each its widest row's spec and members at that position.
    -> the ops folded into. A narrowed op with no such op beside it stays, for the prefill's check to name."""
    out = []
    for o in [o for o in ops if o[1] and all(_NARROWED.fullmatch(m) for m in o[1])]:
        w = (o[0], tuple(x[1] for x in map(_NARROWED.fullmatch, o[1]) if x))
        if len(ops.get(w, ())) != 1:
            continue
        rows, end = sorted(ops.pop(o), key=lambda d: int(d["row"])), ops[w][0]
        del last[o]
        ops[w] = [{**end, "row": int(d["row"])} for d in rows] + [{**end, "row": len(rows)}]
        out.append(w)
    return out


@dataclass
class Served:
    """One request's served values by node: its Program's rows as `ProgramIndex.describe` gives them, each with its
    row's ``spec`` (``{spec, step, op_path, members, row}``, its step the request's), ``read(step, op_path, member, lo,
    hi)``, a member's elements [lo, hi) at a request step (`CommittedStore.words` at the request's arrival plus it),
    ``bound(step, op_path, member)``, the elements the record binds that member at, ``reader`` the byte source those
    reads verify through (`commit.opened.OpenedReader`), whose counts `Rows.counts` reports, and ``locate``, ``read``'s
    arguments to where the committer holds those elements (`opening.CommittedStore.locate`), which `window_plan.Plan` reads.

    The request's prefill is its step 0: each op (its op path and members) a row a prompt position, or one row, its last
    position's, where prefill prunes it; the ops in the order of their last prompt position's rows, the one position that
    computes them all, which is the Program's node order within a position however the prefill interleaves the
    others. A pruned op's one row is its member's last of the prompt's where the record binds the member at a row a prompt
    position (the engine computes the last layer at them all; only the logits gather a row a request), else its only one.
    A narrowed linear's rows (the last layer's fused qkv, keys' and values' columns only on the prompt rows before the
    last) are its node's there, beside its widest rows' one row, the last position's (`_fold_narrowed`): the node is read
    a row a prompt position, each its widest member's whole row, which the record must bind at every prompt position.
    Step s ≥ 1 is position LP − 1 + s, a row each of those ops, matched to them by site (`_sites`), since a call
    boundary's op path names an fx node numbered over the request's steps. A row of another Definition than its class
    node's is refused by name; one of its Definition at other statics is `Slot`'s to hold to the circuit node's value."""
    rows: Sequence[Mapping[str, Any]]
    read: Callable[[int, str, str, int, int], tuple[np.ndarray | None, str]]
    bound: Callable[[int, str, str], int]
    reader: Any = None
    locate: Callable[[int, str, str, int, int], tuple[Any, str]] | None = None
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
        folded = _fold_narrowed(ops, last)
        order = sorted(ops, key=last.__getitem__)
        self.prefill = [sorted(ops[o], key=lambda d: int(d["row"])) for o in order]
        self.LP = max((len(v) for v in self.prefill), default=0)
        for (op, _), v in zip(order, self.prefill):
            if [int(d["row"]) for d in v] not in (list(range(self.LP)), [0]):
                raise ValueError(f"{op}: prefill rows {[d['row'] for d in v]}, not one a prompt position nor the last one's")
        for o in folded:
            n = sorted({self.bound(0, o[0], str(m)) / int(width) for m, width, _dtype in ops[o][0]["members"]})
            if n != [self.LP]:
                raise ValueError(f"{o[0]}: a narrowed linear's node, read whole at every prompt position, its widest members "
                                 f"bound at {n} rows, not a row a prompt position ({self.LP})")
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

    @property
    def positions(self) -> int:
        """Its Program's positions: its prompt's, then a decode step each."""
        return self.LP + len(self.decode)

    @classmethod
    def of_record(cls, pi: Any, store: Any, rid: str, arrive: int) -> Served:
        """Request ``rid``'s served values from its `ProgramIndex` and the record's `CommittedStore`, arrived at engine step
        ``arrive``, each op read under the op path the record binds it under (`_respelt`). A check of its Program (its
        instances header's ``checks``: the length rule, the seed rule) computes the Program's verdict from committed
        values; the engine never runs it and the record binds no member of it, so it is no served op and is left out. Any
        other row with no committed member, or with a member whose value family is not served or is unknown, is refused by
        name (`capture_plan.captured`)."""
        rows = captured(pi, rid)
        at = _respelt(rows, store, rid, arrive)

        def bound(s: int, op: str, m: str) -> int:
            es = store.entries(rid, arrive + s, at.get((s, op), op), m)[0]
            ends = [e["identity"].get("element_range", (0, None))[1] for e in es]
            return max((int(x) for x in ends if x is not None), default=0)

        return cls(rows, lambda s, op, m, lo, hi: store.words(rid, arrive + s, at.get((s, op), op), m, lo, hi), bound,
                   getattr(store, "committed", None),
                   lambda s, op, m, lo, hi: store.locate(rid, arrive + s, at.get((s, op), op), m, lo, hi))

    def __call__(self, p: int, k: int, node: Any) -> tuple[str, list[np.ndarray]] | None:
        """Position ``p``'s node ``k`` (``node``, its index within its position) as served, its row's spec and its values;
        None where the engine didn't compute it."""
        return self._members(p, k, node, self.read)

    def where(self, p: int, k: int, node: Any) -> tuple[str, list[Any]] | None:
        """`__call__`'s node as where the committer holds it, a ``locate`` answer a return field, and no value read."""
        if self.locate is None:
            raise ValueError("a served request's values located: give its `locate` (`of_record`'s)")
        return self._members(p, k, node, self.locate)

    def _members(self, p: int, k: int, node: Any, read: Callable[[int, str, str, int, int], tuple[Any, str]]
                 ) -> tuple[str, list[Any]] | None:
        if k >= len(self.prefill):
            raise ValueError(f"node {k} of a position, past the served request's {len(self.prefill)} ops a position")
        pruned = False
        if p < self.LP:
            v = self.prefill[k]
            pruned = len(v) != self.LP
            d = v[p] if not pruned else v[0] if p == self.LP - 1 else None
        else:
            d = self.decode[p - self.LP][k]
        if d is None:
            return None
        if RM.spec_family(d["spec"]) != RM.spec_family(descriptor_id(node.fn)):
            raise ValueError(f"position {p}'s node {k} is {descriptor_id(node.fn)}, served as {d['spec']} ({d['op_path']})")
        members = list(d["members"])
        if len(members) > 1:
            if not all(str(m[0]).isdecimal() for m in members):
                raise ValueError(f"{d['op_path']}: members {[m[0] for m in members]}, not its return fields by index")
            members.sort(key=lambda m: int(m[0]))
        step, op = int(d["step"]), str(d["op_path"])
        row = int(d["row"])
        if pruned:
            n = {self.bound(step, op, str(m)) / int(width) for m, width, _dtype in members}
            if n not in ({self.LP}, {1}):
                raise ValueError(f"{op}: its one prefill row's members bound at {sorted(n)} rows, not a row a prompt position "
                                 f"({self.LP}) nor one")
            row = p if n == {self.LP} else 0
        out = []
        for m, width, _dtype in members:
            a, why = read(step, op, str(m), row * int(width), (row + 1) * int(width))
            if a is None:
                raise ValueError(f"position {p}'s node {k}: {why}")
            out.append(a)
        return str(d["spec"]), out


class Rows:
    """Every unit's output rows of ``lowered``'s circuit (module docstring) as u16 words by template and output, with each
    unit's source and an embedding read's opening ports, from each slot's `Slot` (in slot order) and the circuit's
    registered values by parameter (`window_weights.weight_values` with their rows kept, and the scan state's,
    `scan_state.SCAN_STATE`, with its rows: the network attester's, a row a bucket of the window), of which an embedding
    read opens its table's (its ``rows``, ``salts`` and ``path``) and a lane's `ScanSlots_v5` reads the scan state's row at
    its anchor. Refused unless every unit's every row is filled once."""

    def __init__(self, lowered: Lowered, slots: Sequence[Slot], registered: Mapping[str, Any]) -> None:
        if lowered.requests is None:
            raise ValueError("a class circuit's rows: the committer fills a served window's circuit (`window_circuit.lower` "
                             "of its requests)")
        if len(slots) != len(lowered.programs):
            raise ValueError(f"{len(slots)} slots' values for {len(lowered.programs)} Programs")
        if bad := [k for k, (P, S) in enumerate(zip(lowered.programs, slots)) if S.fn is not P.fn]:
            raise ValueError(f"slot {bad[0]}'s values are of another Program than the circuit's slot {bad[0]}")
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
        for a, _j, (tid, u) in lowered.scans:
            for i, v in enumerate(_lane(self.fns[tid], self.state, a, IO.scan_reference)):
                self._put(tid, u, f"o{i}", _i32(v if isinstance(v, list) else [v]), EVALUATED)
        units = {(a, j): unit for a, j, unit in lowered.scans}
        for s, k, c, e in lowered.empty:
            P = lowered.programs[k]
            self._empty(P, units[(P.cls.anchor(s), P.cls.lane(s))], s, c, e)
        if bad := {t: int((~h).any(axis=1).sum()) for t, h in self.held.items() if not h.all()}:
            raise ValueError(f"units with rows no node or Definition fills, by template: {bad}")
        self.readers = list({id(r): r for S in slots if (r := getattr(S, "reader", None)) is not None}.values())
        print("WINDOW-ROWS " + json.dumps(self.counts(), sort_keys=True), file=sys.stderr)

    def counts(self) -> dict[str, Any]:
        """The window's units by source (`SOURCES`) and its served reads' counts, summed over its slots' readers (``reads``,
        ``failed``: an opening that didn't verify, ``not_retained``: a member the committer no longer holds); ``opened`` is
        None where no slot read through a reader with counts."""
        units = {s: int(sum((x == i).sum() for x in self.source.values())) for i, s in enumerate(SOURCES)}
        stats = [r.stats for r in self.readers if isinstance(getattr(r, "stats", None), Counter)]
        opened = {k: int(sum(c[k] for c in stats)) for k in ("reads", "failed", "not_retained")} if stats else None
        return {"units": units, "opened": opened}

    def _put(self, tid: str, unit: int, out: str, rows: np.ndarray, src: int) -> None:
        o, k = int(out[1:]), len(rows)
        if self.held[tid][unit:unit + k, o].any():
            raise ValueError(f"{tid} units [{unit}, {unit + k}) {out}: filled twice")
        self.outs[tid][o][unit:unit + k] = rows
        self.held[tid][unit:unit + k, o] = True
        self.source[tid][unit:unit + k] = np.minimum(self.source[tid][unit:unit + k], src)

    def _slot(self, P: Any, S: Slot) -> None:
        nodes = P.fn.body.nodes
        for j, pieces in P.outs.items():
            w = _words(S.values[j])
            parts = P.norms.get(j, ())
            sums = [(len(w) // 2 + at * r.words, r) for at, r in enumerate(parts)]
            for lo, hi, r in [*pieces, *((lo, lo + r.words, r) for lo, r in sums)]:
                self._put(r.tid, r.unit, r.out, w[lo:hi].reshape(-1, r.words), S.source[j])
                self.node[r.tid][r.unit:r.unit + (hi - lo) // r.words] = True
        emb = [j for j, node in enumerate(nodes) if node_family(node.fn) == "Embedding"]
        for p, (j, feed) in enumerate(zip(emb, P.tokens)):
            r, t = P.outs[j][0][2], S.tokens[p]
            inside = t < nodes[j].fn.bindings["V"]
            self._put(r.tid, r.unit, "o1", np.full((1, 1), int(inside), dtype=np.uint16), EVALUATED)
            if inside:
                ins, table = self.inputs[r.tid], self.tables[r.tid]
                ins["row"][r.unit] = np.asarray(table.rows[t], dtype=np.uint16)
                ins["salt"][r.unit] = _be_words(table.salts[t])
                ins["path"][r.unit] = _be_words(table.path(t))
            if feed is not None:
                (tid, unit), nxt = feed, S.carried[p]
                assert nxt is not None, f"{P.name}: position {p} has a token feed unit, and its slot carries nothing on"
                self._put(tid, unit, "o0", _i32([t]), EVALUATED)
                self._put(tid, unit, "o1", _i32(nxt), EVALUATED)
                self._pad(tid, unit)
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
            xt, xu = x0
            self._put(xt, xu, "o0", logits[:1].astype(np.uint16).reshape(1, 1), EVALUATED)
            picks.append(int(S.values[j][0][0]))
        self._scan(P, S, picks)
        for j, (plan, calls) in P.heads.items():
            self._heads(nodes[j], plan, calls, S)

    def _scan(self, P: Any, S: Slot, picks: Sequence[int]) -> None:
        """The slot's scan units, its lane's row of the scan state row at its anchor the slot's (refused by name where it
        isn't): its collectors on from the lane's head, a prompt position's before the last (`OutCollect_v1`) on its token, a select's (`SelectCollect_v1`, its
        ``o1``; its select ``o0`` is the node's) on its pick and KO - O pads on the last pick; and its `SlotEntry_v1` entry
        of the scan's Programs' row."""
        row, a = _lane(P.scan, self.state, P.anchor)
        if row != S.row:
            raise ValueError(f"{P.name}: its slot's row isn't lane {P.lane}'s of the scan state row at bucket {P.anchor}")
        KO = P.cls.outputs
        toks = [*S.tokens[:P.L - 1], *picks, *[picks[-1]] * (KO - len(picks))]
        if len(P.collects) != len(toks):
            raise ValueError(f"{P.name}: {len(P.collects)} collectors, not one a prompt position, one a select and one a pad "
                             f"to KO ({len(toks)})")
        for k, ((ct, cu), tok) in enumerate(zip(P.collects, toks)):
            a = IO.collect_reference(a, tok, KO, k == 0)
            self._put(ct, cu, "o0" if self.fns[ct].name == "OutCollect" else "o1", _i32(a), EVALUATED)
            self._pad(ct, cu)
        e = P.entry.bindings
        entry = IO.entry_reference(S.row[0], a, KO, e["IDB"], e["NP"], e["STOP"])
        et, eu = P.entry_unit
        self._put(et, eu, "o0", np.array([entry], dtype=np.uint16), EVALUATED)
        self._pad(et, eu)

    def _empty(self, P: Any, scan: tuple[str, int], s: int, collect: tuple[str, int], entry: tuple[str, int]) -> None:
        """Empty slot ``s``'s collector and entry (`_Program.place_empty`), on its lane's ``scan`` unit's read of the scan
        state row at its anchor."""
        fn, a = self.fns[scan[0]], P.cls.anchor(s)
        (row, head), tok = _lane(fn, self.state, a), _lane(fn, self.state, a, IO.scan_reference)[0]
        KO, e = P.cls.outputs, P.entry.bindings
        a = IO.collect_reference(head, tok, KO, True)
        self._put(*collect, "o0", _i32(a), EVALUATED)
        self._put(*entry, "o0", np.array([IO.entry_reference(row[0], a, KO, e["IDB"], e["NP"], e["STOP"])], dtype=np.uint16),
                  EVALUATED)
        self._pad(*collect)
        self._pad(*entry)

    def _pad(self, tid: str, unit: int) -> None:
        if self.fns[tid].name in ("OutCollect", "SlotEntry", "TokenFeed") and self.fns[tid].version == 2:
            self._put(tid, unit, f"o{len(self.lays[tid].outs) - 1}", np.zeros((1, IO.PAD_WORDS), dtype=np.uint16), EVALUATED)

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


def leaf_strings(win: IF.Window, key: RND.Key) -> dict[tuple[str, str], tuple[bytes, bytes]]:
    """Each of ``win``'s `leaf_rows` ports, its rows' salts and ``b || c`` (`rows.row_strings`), the strings `Window.close`
    takes: unit u's row of port q of template t (`interface.LeafRows`) is salted with ``key``'s stream at index (t, q), leaf u."""
    return {k: (ys, CS.row_strings(lr.rows, ys, lr.schema))
            for k, lr in win.leaf_rows().items()
            for s in [key.stream((lr.template, lr.port), CS.SALT_BYTES * (lr.units[-1] + 1))]
            for ys in [b"".join(s[u * CS.SALT_BYTES:(u + 1) * CS.SALT_BYTES] for u in lr.units)]}


def _elements(words: np.ndarray, width: int) -> np.ndarray:
    """``(n, words)`` u16 words as ``width``-bit elements, the same little-endian bytes (`interface.words`' inverse)."""
    return np.ascontiguousarray(words, dtype=np.uint16).view(f"<u{width // 8}")


def _lane(scan: Any, state: Sequence[Any], anchor: int, of: Callable[..., Any] = IO.slots_reference) -> Any:
    """The lane of `ScanSlots_v5` binding ``scan`` in ``state``'s row ``anchor``: its row and head
    (`io_shell.slots_reference`), or ``of``'s (`io_shell.scan_reference`: the binding's outputs)."""
    b = scan.bindings
    row = np.asarray(state[anchor], dtype=np.uint16).reshape(-1)
    return of(row.tolist(), b["NP"], b["IDB"], b["K0"], b["K"], b["OCC"], b["LP"], b["MAX"], b["IDS"])


def scan_slots(lowered: Lowered, state: Sequence[Any]) -> list[tuple[list[int], list[int]]]:
    """Each Program's slot as the scan state's rows ``state`` (`scan_state.SCAN_STATE`'s, a row a bucket of the
    window) give it, in slot order: its lane's row (occupancy, prompt length, ids; a `Slot`'s ``row``) and head (prompt
    length, ``max_tokens``) of the row at its anchor."""
    return [_lane(P.scan, state, P.anchor) for P in lowered.programs]


def state_value(program: str, cls: WindowClass, rows: Sequence[Sequence[int]], *, window: str) -> RV.Registered:
    """`scan_state.SCAN_STATE`'s value: ``rows``, the scan's state rows a window of ``cls`` reads (row a the state before
    bucket a, `ingress_scan.window_state`'s bits), as ``row/v2`` rows, each under 192 fresh bytes from the OS, a
    per-window value under ``program``, its circuit's `inference_blocks.program_digest`, and the execution phase's open
    root ``window``. It is the prover's side of the tree (its rows, salts and levels) for the window's prover and the
    scan's alike: the scan's state ports open their rows in it, by its root and domain, and its salts exist only in it and
    in the files written from it (the window's ``v<j>-private.bin``, `scan_zk.SCAN_STATE_FILE`)."""
    bits = np.asarray(rows, dtype=np.uint16)
    if bits.shape != (cls.buckets, 16 * SS.words(cls.scan)) or bits.max(initial=0) > 1:
        raise ValueError(f"a window of {cls.buckets} buckets reads {cls.buckets} state rows of {16 * SS.words(cls.scan)} "
                         f"bits, not {bits.shape}")
    words = (bits.reshape(cls.buckets, -1, 16) << np.arange(16, dtype=np.uint16)).sum(axis=2, dtype=np.uint16)
    return registered_commit(program, SS.SCAN_STATE, words, os.urandom(cls.buckets * CS.SALT_BYTES), keep_rows=True,
                             window=window)


__all__ = ["COMPUTED", "EVALUATED", "FORMS", "Rows", "SERVED", "SOURCES", "Served", "Slot", "leaf_strings", "positions", "scan_slots",
           "state_value"]
