"""A `verity_vllm.query.units.Program` view of a derived Program.

Two readers of ONE flat body (`program.fn.body.nodes` of the derived request Program, in body order):

* :func:`from_descriptor` -- the Program descriptor (`descriptor.json.gz`) as `verity.ml.circuits.codec.decode_program` decodes it: the
  program function's body nodes give every Call's Definition, output type and operand references; the annotations' `node_hints` its name.
* :func:`from_instances` -- the Build's instance sequence (`instances.json.gz`, `derive_step._instance_sequence`): the same nodes
  written one row per Call (`i`, `spec`, `form`, `name`, `args` = operand runs) -- a projection of the descriptor body plus the
  name annotation, produced by the same derivation on meta tensors (nothing executed).  The output type is not in a row: it is
  the registered Definition's signature bound to the row's statics (`verity.ml.definitions.registry`), read once per distinct spec.

Both yield the same :class:`ProgramView`: Calls with ports, reads and names; prescribed inputs as Values of negative ids
(`-(k + 1)` for parameter `k`; a record parameter has one port per field, anything else the port `out`); outputs as the reader
knows them (a descriptor's `ret`; an instance sequence carries none).

Ports.  A Definition returning one collection (`Array<N, Value<w>>`, a `Value<w>`) has the one port `out`; a tuple has the ports
`"0"`, `"1"`, ...; a record its field names.  Every port knows its element count and word width (None when the leaves are not
one width) and its leaf interval in the Call's output, so an operand run `("n", j, lo, hi)` resolves to the ports of Call `j` it
overlaps -- the Value(s) read.

Representation.  A request Program's volume is in its READS, not its Calls: at LP=4096 every attention row reads one
K and one V run of every earlier token (nearly all of a Program's operand runs are in its attention rows), so anything held per
read as a Python object (the decoded JSON run lists, a `ValueRef` tuple, a `CallView.reads` tuple) is tens of GB.  Both readers
therefore build ONE columnar body (:class:`_Columns`): flat `array` columns of Call ids and port-name indices with per-Call offsets
for the reads (6 bytes per read), per-Call indices into small tables for the spec / form / ports, the names as one list of strings.
`ProgramView.calls` is a lazy `Sequence[CallView]` (:class:`_Calls`) that materializes each `CallView` once, when it is first indexed
or iterated (a Program's many passes over its Calls reuse it, at about 150 bytes per Call), and a `CallView`'s `reads` from that body
is a lazy `Sequence[ValueRef]` (:class:`_Reads`) that materializes `ValueRef`s when iterated -- a consumer holds a Call's reads only
while it looks at them.  `from_instances` streams the rows of `instances.json.gz` one at a
time (never the `rows` list).  A `ProgramView` built by hand from a tuple of `CallView`s (tests, synthetic Programs) is unchanged.
"""

from __future__ import annotations

import bisect
import gzip
import json
import operator
from array import array
from collections.abc import Sequence as _SequenceABC
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from verity.ml.definitions.call_scope import spec_base, spec_statics
from verity_vllm.query.type_repr import parse_type_repr
from verity_vllm.query.units import ValueRef

__all__ = ["Port", "CallView", "ProgramView", "InputView", "from_instances", "from_descriptor", "definition_ports", "parse_type_repr",
           "spec_base", "spec_statics", "ports_of_type", "param_reads", "reads_any_of", "iter_instance_rows"]


@dataclass(frozen=True, slots=True)
class Port:
    """One output collection of a Call: name, leaf interval `[lo, hi)` in the Call's output, element count, word width."""

    name: str
    lo: int
    hi: int
    elems: int
    width: int | None


@dataclass(frozen=True, slots=True)
class CallView:
    id: int
    definition: str                  # the full spec id `Name_vN{k=v,..}`
    family: str                      # `Name_vN`
    form: str                        # call | batch | scan
    name: str                        # the recorded naming hint (`<module>/<fx node>[/<member>][row]`), "" if none
    port_table: tuple[Port, ...]
    reads: Sequence[ValueRef]        # a tuple, or the lazy `_Reads` of a columnar body (materialized only while iterated)
    bits: str | None = None          # a Boolean Program's Call (its rows' `bits`): the spec id of the Boolean Definition it is of

    @property
    def ports(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.port_table)

    def port(self, name: str) -> Port:
        for p in self.port_table:
            if p.name == name:
                return p
        raise KeyError(f"Call {self.id} ({self.definition}) has no port {name!r}")

    @property
    def statics(self) -> dict[str, Any]:
        return spec_statics(self.definition)


@dataclass(frozen=True, slots=True)
class InputView:
    """One prescribed input (a parameter of the Program function): its id (negative), name, type and ports."""

    id: int
    name: str
    type_repr: str
    port_table: tuple[Port, ...]

    @property
    def ports(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.port_table)


@dataclass(frozen=True)
class ProgramView:
    calls: Sequence[CallView]        # a tuple, or the lazy `_Calls` over a columnar body (both readers)
    params: tuple[InputView, ...]
    outputs: tuple[ValueRef, ...]
    digest: str | None
    source: str                      # "descriptor" | "instances"

    @property
    def inputs(self) -> tuple[ValueRef, ...]:
        return tuple(ValueRef(p.id, port.name) for p in self.params for port in p.port_table)

    def call(self, cid: int) -> CallView:
        c = self.calls[cid]
        if c.id != cid:                                  # ids are body positions for both readers; keep the lookup honest
            for cc in self.calls:
                if cc.id == cid:
                    return cc
            raise KeyError(cid)
        return c

    def input(self, iid: int) -> InputView:
        for p in self.params:
            if p.id == iid:
                return p
        raise KeyError(iid)

    @property
    def columns(self) -> "_Columns | None":
        """The columnar body when the view was read from a Build (None for a hand-built tuple of CallViews)."""
        return self.calls._cols if isinstance(self.calls, _Calls) else None


def param_reads(call: CallView) -> Iterable[ValueRef]:
    """The reads of `call` that are prescribed-input Values (negative producer ids), in operand order -- O(#such reads) on a columnar
    body (a separate index; an attention row's thousands of Call reads are not scanned), a filter of `reads` otherwise."""
    r = call.reads
    if isinstance(r, _Reads):
        return r.params()
    return (v for v in r if v.call < 0)


def reads_any_of(call: CallView, producers: frozenset[int] | set[int]) -> bool:
    """Does `call` read a Value produced by one of `producers` (Call ids)?  On a columnar body a scan of the Call-id column (no
    ValueRef is built), else over `reads`."""
    r = call.reads
    if isinstance(r, _Reads):
        return r.any_producer_in(producers)
    return any(v.call in producers for v in r)


# ---------------------------------------------------------------------------------------------------------------------------------
# the columnar body
# ---------------------------------------------------------------------------------------------------------------------------------

class _Columns:
    """One flat body: per Call an index into the spec table (spec id, family, ports) and the form table, its name; the reads as
    `[read_off[k], read_off[k + 1])` of two parallel columns -- the producer id (`array('i')`: a Call id, or a negative input id) and
    the port-name index (`array('H')` into `port_names`) -- in operand order, each Value once; and the same for the reads of
    prescribed inputs alone (`pread_*`), so a pass over the weights a body reads never touches the Call reads."""

    __slots__ = ("n", "specs", "families", "bits", "port_tables", "spec_idx", "forms", "form_idx", "names", "port_names",
                 "read_off", "read_call", "read_port", "pread_off", "pread_call", "pread_port", "views")

    def __init__(self) -> None:
        self.n = 0
        self.specs: list[str] = []
        self.families: list[str] = []
        self.bits: list[str | None] = []
        self.port_tables: list[tuple[Port, ...]] = []
        self.spec_idx = array("i")
        self.forms: list[str] = []
        self.form_idx = array("B")
        self.names: list[str] = []
        self.port_names: list[str] = []
        self.read_off = array("q", [0])
        self.read_call = array("i")
        self.read_port = array("H")
        self.pread_off = array("q", [0])
        self.pread_call = array("i")
        self.pread_port = array("H")
        self.views: list[CallView | None] = []

    def call(self, k: int) -> "CallView":
        views = self.views
        if len(views) < self.n:
            views.extend([None] * (self.n - len(views)))
        v = views[k]
        if v is None:
            s = self.spec_idx[k]
            v = views[k] = CallView(k, self.specs[s], self.families[s], self.forms[self.form_idx[k]], self.names[k], self.port_tables[s],
                                    _Reads(self, k), self.bits[s])
        return v

    def family_of(self, k: int) -> str:
        return self.families[self.spec_idx[k]]

    def reads_of(self, k: int) -> Iterator[ValueRef]:
        lo, hi = self.read_off[k], self.read_off[k + 1]
        return map(ValueRef._make, zip(self.read_call[lo:hi], map(self.port_names.__getitem__, self.read_port[lo:hi])))

    def param_reads_of(self, k: int) -> Iterator[ValueRef]:
        lo, hi = self.pread_off[k], self.pread_off[k + 1]
        return map(ValueRef._make, zip(self.pread_call[lo:hi], map(self.port_names.__getitem__, self.pread_port[lo:hi])))

    def nbytes(self) -> int:
        """The bytes of the read columns (the part proportional to the expanded Program)."""
        return sum(a.itemsize * len(a) for a in (self.read_off, self.read_call, self.read_port, self.pread_off, self.pread_call, self.pread_port, self.spec_idx, self.form_idx))


class _Reads(_SequenceABC):
    """`CallView.reads` over a columnar body: a `Sequence[ValueRef]` that builds each ValueRef when asked for it."""

    __slots__ = ("_cols", "_k")

    def __init__(self, cols: _Columns, k: int) -> None:
        self._cols, self._k = cols, k

    def __len__(self) -> int:
        return self._cols.read_off[self._k + 1] - self._cols.read_off[self._k]

    def __iter__(self) -> Iterator[ValueRef]:
        return self._cols.reads_of(self._k)

    def __getitem__(self, i):
        c = self._cols
        lo, hi = c.read_off[self._k], c.read_off[self._k + 1]
        if isinstance(i, slice):
            return tuple(ValueRef(c.read_call[lo + j], c.port_names[c.read_port[lo + j]]) for j in range(*i.indices(hi - lo)))
        if i < 0:
            i += hi - lo
        if not 0 <= i < hi - lo:
            raise IndexError(i)
        return ValueRef(c.read_call[lo + i], c.port_names[c.read_port[lo + i]])

    def params(self) -> Iterator[ValueRef]:
        return self._cols.param_reads_of(self._k)

    def any_producer_in(self, producers) -> bool:
        c = self._cols
        lo, hi = c.read_off[self._k], c.read_off[self._k + 1]
        return any(j in producers for j in c.read_call[lo:hi])

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (tuple, list, _Reads)):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __repr__(self) -> str:
        return f"_Reads(call={self._k}, n={len(self)})"


class _Calls(_SequenceABC):
    """`ProgramView.calls` over a columnar body: `Sequence[CallView]`, one CallView built per index / iteration step."""

    __slots__ = ("_cols",)

    def __init__(self, cols: _Columns) -> None:
        self._cols = cols

    def __len__(self) -> int:
        return self._cols.n

    def __iter__(self) -> Iterator[CallView]:
        return map(self._cols.call, range(self._cols.n))

    def __getitem__(self, i):
        n = self._cols.n
        if isinstance(i, slice):
            return tuple(self._cols.call(k) for k in range(*i.indices(n)))
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        return self._cols.call(i)

    def __add__(self, other):
        return tuple(self) + tuple(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (tuple, list, _Calls)):
            return len(self) == len(other) and all(a == b for a, b in zip(self, other))
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __repr__(self) -> str:
        return f"_Calls(n={len(self)})"


def _disjoint_los(table: Sequence[Port]) -> list[int] | None:
    """The ports' `lo` in table order when they are disjoint and ascending (then a run's overlapping ports are found by bisection, in
    table order), else None."""
    los = [p.lo for p in table]
    return los if all(table[k].hi <= los[k + 1] for k in range(len(table) - 1)) else None


class _OutOfOrder(Exception):
    """The streamed rows are not the body in order (row i is not at position i, or a read names a later Call)."""


#: `_Builder(params, memo=True)` resolves a parameter-only operand of at least `RESOLVE_MEMO_MIN` runs once, and a later operand of
#: the same run objects (a document's equal items are one list: `instances_file.read_instances`) takes its ports: every token's Call
#: reads its layer's weights again.  At most `RESOLVE_MEMO_RUNS` runs are kept, and `_MEMO_WAYS` operands per first run; an operand's
#: runs must not change once added.
RESOLVE_MEMO_MIN = 8
RESOLVE_MEMO_RUNS = 1 << 25
_MEMO_WAYS = 4


class _Builder:
    """Fills a `_Columns` one Call at a time, in body order.  `add(i, spec, form, name, args)`: `args` = the operand runs
    `("n"|"p", idx, lo, hi)`, resolved against the ports of the Calls added so far (a run naming a later Call is `_OutOfOrder`)."""

    def __init__(self, params: tuple[InputView, ...], memo: bool = False) -> None:
        self.c = _Columns()
        self._memo: dict[int, list[tuple[tuple, list[tuple[int, int]]]]] | None = {} if memo else None
        self._memo_runs = 0
        self._spec_of: dict[str, int] = {}
        self._form_of: dict[str, int] = {}
        self._pname: dict[str, int] = {}
        self._spec_pidx: list[tuple[int, ...]] = []          # per spec: the port-name index of each port
        self._params = params
        self._param_pidx = [tuple(self._port_index(p.name) for p in pv.port_table) for pv in params]
        self._param_lo = [_disjoint_los(pv.port_table) for pv in params]
        self._param_hi = [None if los is None else [p.hi for p in pv.port_table] for pv, los in zip(params, self._param_lo)]

    def _port_index(self, name: str) -> int:
        k = self._pname.get(name)
        if k is None:
            k = self._pname[name] = len(self.c.port_names)
            self.c.port_names.append(name)
            if k > 0xFFFF:
                raise ValueError("more than 65535 distinct port names")
        return k

    def add(self, i: int, spec: str, form: str, name: str, port_table: tuple[Port, ...], args: Iterable[Iterable[Sequence[Any]]],
            bits: str | None = None) -> None:
        c = self.c
        if i != c.n:
            raise _OutOfOrder(f"row {i} at body position {c.n}")
        s = self._spec_of.get(spec)
        if s is None:
            s = self._spec_of[spec] = len(c.specs)
            c.specs.append(spec)
            c.families.append(spec_base(spec))
            c.bits.append(bits)
            c.port_tables.append(port_table)
            self._spec_pidx.append(tuple(self._port_index(p.name) for p in port_table))
        fi = self._form_of.get(form)
        if fi is None:
            fi = self._form_of[form] = len(c.forms)
            c.forms.append(form)
        c.spec_idx.append(s)
        c.form_idx.append(fi)
        c.names.append(name)
        self._resolve(args, set(), c.read_call, c.read_port, c.pread_call, c.pread_port, self._memo)
        c.n += 1
        c.read_off.append(len(c.read_call))
        c.pread_off.append(len(c.pread_call))

    def _memoized(self, operand: list, memo: dict) -> list[tuple[int, int]] | None:
        """The (parameter id, port index) pairs a parameter-only operand reads, in first-touch order, when the memo has or takes it."""
        ways = memo.get(id(operand[0]), ())
        for runs, ports in ways:
            if len(runs) == len(operand) and all(map(operator.is_, runs, operand)):
                return ports
        if len(ways) >= _MEMO_WAYS or self._memo_runs + len(operand) > RESOLVE_MEMO_RUNS or any(r[0] not in ("P", "p") for r in operand):
            return None
        rc: list[int] = []
        rp: list[int] = []
        self._resolve((operand,), set(), rc, rp, [], [], None)
        memo.setdefault(id(operand[0]), []).append((tuple(operand), ports := list(zip(rc, rp))))
        self._memo_runs += len(operand)
        return ports

    def _resolve(self, args: Iterable[Iterable[Sequence[Any]]], seen: set, rc, rp, prc, prp, memo: dict | None) -> None:
        """Operand runs -> (producer id, port index) appended in operand order, each Value once (`seen` keys: the producer id when it has one
        port, else (id, port index)).  An operand's Values are its own first-touch order less those `seen` before it, so a memoized one's
        are filtered by `seen`."""
        c = self.c
        spec_idx, port_tables, spec_pidx, forms = c.spec_idx, c.port_tables, self._spec_pidx, c.forms
        params, param_pidx = self._params, self._param_pidx
        n = c.n

        def node(idx: int, lo: int, hi: int) -> None:
            if idx >= n:
                raise _OutOfOrder(f"Call {n} reads Call {idx}, not yet read")
            s = spec_idx[idx]
            pidx = spec_pidx[s]
            if len(pidx) == 1:
                if idx not in seen:
                    seen.add(idx)
                    rc.append(idx); rp.append(pidx[0])
                return
            hit = False
            for p, pi in zip(port_tables[s], pidx):
                if p.lo < hi and lo < p.hi:
                    hit = True
                    if (idx, pi) not in seen:
                        seen.add((idx, pi))
                        rc.append(idx); rp.append(pi)
            if not hit and forms[c.form_idx[idx]] == "batch":  # a batch member beyond the single-instance signature: one port
                pi = pidx[0]
                if (idx, pi) not in seen:
                    seen.add((idx, pi))
                    rc.append(idx); rp.append(pi)

        def param(idx: int, lo: int, hi: int, step: int = 0, count: int = 1) -> None:
            """The ports of `count` runs `[lo + k*step, hi + k*step)` of parameter `idx`, in first-touch order (ascending for a
            disjoint ascending table and step > 0, the order the runs one by one give)."""
            pv, pidx, los = params[idx], param_pidx[idx], self._param_lo[idx]
            if len(pidx) == 1:
                sel = pidx
            elif los is None:
                sel = [pi for p, pi in zip(pv.port_table, pidx) if p.lo < hi and lo < p.hi]
            else:                       # a weights struct has thousands of ports: the overlapping ones by bisection
                table, k, sel, end = pv.port_table, max(bisect.bisect_right(los, lo) - 1, 0), [], hi + (count - 1) * step
                while k < len(los) and los[k] < end:
                    j = 0 if count == 1 else max(0, (los[k] - hi) // step + 1)   # the first run ending past the port's start
                    if j < count and lo + j * step < table[k].hi:
                        sel.append(pidx[k])
                    k += 1
            for pi in sel:
                if (pv.id, pi) not in seen:
                    seen.add((pv.id, pi))
                    rc.append(pv.id); rp.append(pi)
                    prc.append(pv.id); prp.append(pi)

        param_lo, param_hi = self._param_lo, self._param_hi
        p_idx = p_lo = p_hi = -1                 # [p_lo, p_hi): the port of parameter p_idx the last one-port progression read
        for operand in args:
            if memo is not None and type(operand) is list and len(operand) >= RESOLVE_MEMO_MIN and operand[0][0] in ("P", "p"):
                ports = self._memoized(operand, memo)
                if ports is not None:
                    for key in ports:
                        if key not in seen:
                            seen.add(key)
                            rc.append(key[0]); rp.append(key[1]); prc.append(key[0]); prp.append(key[1])
                    continue
            for run in operand:
                space = run[0]
                if len(run) == 8 and space in ("N", "P"):                 # a stored progression, resolved without its runs
                    _, idx, lo, hi, dI, dL, dH, cnt = run
                    if space == "P" and dI == 0 and dL == dH and (cnt == 1 or dL > 0 and cnt > 1) and param_lo[idx] is not None:
                        # a weight read is progressions (1.4e8 in a GLM-5.3 rank Program) that each lie inside one field: that port
                        # alone, mostly the one the progression before it read
                        end = hi + (cnt - 1) * dL
                        if idx == p_idx and p_lo <= lo < hi and end <= p_hi:
                            continue
                        los = param_lo[idx]
                        k = bisect.bisect_right(los, lo) - 1
                        if k >= 0 and lo < hi and end <= param_hi[idx][k]:
                            p_idx, p_lo, p_hi = idx, los[k], param_hi[idx][k]
                            pid, pi = params[idx].id, param_pidx[idx][k]
                            if (pid, pi) not in seen:
                                seen.add((pid, pi))
                                rc.append(pid); rp.append(pi)
                                prc.append(pid); prp.append(pi)
                            continue
                        param(idx, lo, hi, dL, cnt)
                        continue
                    one = node if space == "N" else param
                    for k in range(cnt):
                        one(idx + k * dI, lo + k * dL, hi + k * dH)
                elif space == "n":
                    node(int(run[1]), int(run[2]), int(run[3]))
                elif space == "p":
                    param(int(run[1]), int(run[2]), int(run[3]))

    def outputs(self, runs_: Iterable[Sequence[Any]]) -> tuple[ValueRef, ...]:
        """The Values a `ret` reference sequence names (the body's outputs), by the same resolution."""
        c = self.c
        out: list[ValueRef] = []
        seen: set = set()
        for run in runs_:
            space, idx, lo, hi = run[0], int(run[1]), int(run[2]), int(run[3])
            if space == "n":
                table = c.port_tables[c.spec_idx[idx]]
                vs = (ValueRef(idx, table[0].name),) if len(table) == 1 else tuple(ValueRef(idx, p.name) for p in table if p.lo < hi and lo < p.hi)
                if not vs and c.forms[c.form_idx[idx]] == "batch":
                    vs = (ValueRef(idx, table[0].name),)
            elif space == "p":
                pv = self._params[idx]
                vs = (ValueRef(pv.id, pv.port_table[0].name),) if len(pv.port_table) == 1 else tuple(ValueRef(pv.id, p.name) for p in pv.port_table if p.lo < hi and lo < p.hi)
            else:
                continue
            for v in vs:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
        return tuple(out)

    def view(self, outputs: tuple[ValueRef, ...], digest: str | None, source: str) -> ProgramView:
        return ProgramView(_Calls(self.c), self._params, outputs, digest, source)


# ---------------------------------------------------------------------------------------------------------------------------------
# spec ids and types
# ---------------------------------------------------------------------------------------------------------------------------------

def ports_of_type(t) -> tuple[Port, ...]:
    """The output ports of a `verity.ml.circuits.types.Type`: one `out` for a Value / Array, `"0"`.. for a Tuple, field names for a Record."""
    from verity.ml.circuits.types import Record, Tuple as TupleT

    def one(name: str, lo: int, ty) -> Port:
        n = ty.leaves
        return Port(name, lo, lo + n, n, ty.homogeneous_width())

    if isinstance(t, TupleT):
        out, lo = [], 0
        for i, f in enumerate(t.fields):
            out.append(one(str(i), lo, f))
            lo += f.leaves
        return tuple(out)
    if isinstance(t, Record):
        out, lo = [], 0
        for k, f in t.fields:
            out.append(one(k, lo, f))
            lo += f.leaves
        return tuple(out)
    return (one("out", 0, t),)


def _registry():
    """(REGISTRY, import errors) of the integration's registry, every module loaded (`verity.ml.definitions.registry.catalog.load`); a module that
    fails to import is reported with the lookup that needed it, never treated as "not registered"."""
    try:
        from verity.ml.circuits.defs import REGISTRY  # noqa: F401
    except Exception as e:  # the core package missing is an ENVIRONMENT fault, named (never a "not registered" answer)
        from verity_vllm.query.manifest.format import RegistryUnavailable
        raise RegistryUnavailable(e) from e
    from verity.ml.definitions.registry.catalog import load
    return load()


_PORTS_CACHE: dict[str, tuple[Port, ...]] = {}


def definition_ports(spec: str) -> tuple[Port, ...]:
    """The output ports of the Definition a spec id names, bound to its statics.  Raises `KeyError` (not registered) or `TypeError`
    (statics do not bind) -- a Program row whose Definition the registry does not carry is a Program the registry did not build."""
    hit = _PORTS_CACHE.get(spec)
    if hit is not None:
        return hit
    from verity.ml.circuits.codec import lookup_primitive
    reg, errors = _registry()
    base = spec_base(spec)
    d = reg.defs.get(base)
    if d is None:
        try:
            d = lookup_primitive(reg, base)
        except KeyError:
            raise KeyError(f"{base} is not a registered Definition" + (f" (registry import errors: {'; '.join(errors)})" if errors else "")) from None
    if hasattr(d, "bind"):
        sp = d.bind(**spec_statics(spec))
        if not sp.complete:
            missing = [k for k in d.statics if k not in sp.bindings]
            raise TypeError(f"{base}: statics {missing} are not in the spec {spec!r}")
        _params, ret = sp.signature()
    else:
        ret = d.ret
    ports = ports_of_type(ret)
    _PORTS_CACHE[spec] = ports
    return ports


# ---------------------------------------------------------------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------------------------------------------------------------

def _params_view(params: Iterable[tuple[str, Any]]) -> tuple[InputView, ...]:
    out = []
    for k, (name, t) in enumerate(params):
        ty = parse_type_repr(t) if isinstance(t, str) else t
        out.append(InputView(-(k + 1), str(name), repr(ty), ports_of_type(ty)))
    return tuple(out)


_CHUNK = 1 << 22


def _open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def _read_instance_header(path: str) -> dict[str, Any]:
    """`instances.json.gz` minus its rows (schema, digest, params): the text before `"rows"`, closed -- the keys before the array."""
    with _open_text(path) as f:
        buf = f.read(1 << 16)
        k = buf.find('"rows"')
        while k < 0:
            more = f.read(1 << 16)
            if not more:
                break
            buf += more
            k = buf.find('"rows"')
    if k < 0:
        raise ValueError(f"{path}: instance sequence without a rows array")
    return json.loads(buf[:k].rstrip().rstrip(",") + "}")


def iter_instance_rows(path: str, *, expand: bool = True) -> Iterator[dict[str, Any]]:
    """The objects of the top-level `"rows": [...]` array of an instance-sequence file, one at a time (the array is never built).
    The document's key order (`schema`, `digest`, `params`, `rows`) is the writer's (`derive_step._instance_sequence`).  A v2 row's
    stored argument progressions are expanded to the v1 per-leaf run lists, bit-identically (`compact.expanded_file_args`), so every
    reader of this iterator sees the record form whatever the file carries."""
    from verity_vllm.program.compact import expanded_file_args
    dec = json.JSONDecoder()
    with _open_text(path) as f:
        buf = f.read(_CHUNK)
        k = buf.find('"rows"')
        while k < 0:
            more = f.read(_CHUNK)
            if not more:
                raise ValueError(f"{path}: instance sequence without a rows array")
            buf = buf[max(0, len(buf) - 8):] + more
            k = buf.find('"rows"')
        pos = k + len('"rows"')
        while True:
            k = buf.find("[", pos)
            if k >= 0:
                break
            more = f.read(_CHUNK)
            if not more:
                raise ValueError(f"{path}: rows array not opened")
            buf, pos = more, 0
        pos = k + 1
        ws = " \n\r\t,"
        while True:
            while True:
                while pos < len(buf) and buf[pos] in ws:
                    pos += 1
                if pos < len(buf):
                    break
                more = f.read(_CHUNK)
                if not more:
                    raise ValueError(f"{path}: rows array not closed")
                buf, pos = more, 0
            if buf[pos] == "]":
                return
            while True:
                try:
                    obj, end = dec.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    more = f.read(max(_CHUNK, len(buf) - pos))     # a row longer than a chunk: double, so it is parsed O(1) times
                    if not more:
                        raise
                    buf = buf[pos:] + more
                    pos = 0
            a = obj.get("args")
            if a and expand:
                obj["args"] = expanded_file_args(a)
            yield obj
            pos = end
            if pos > _CHUNK:                               # compact: keep the buffer bounded by the chunk size plus one row
                buf, pos = buf[pos:], 0


def _instances_columns(params: tuple[InputView, ...], rows: Iterable[dict[str, Any]], memo: bool = False) -> _Builder:
    """Body order in, columns out; `_OutOfOrder` when the rows are not body positions in order.  A v2 row's stored argument
    progressions are resolved as progressions (`_Builder._resolve`): the same Values, in the same order, as their per-leaf runs.
    `memo` for rows held as a document (streamed rows are decoded afresh, so the memo would only keep them)."""
    b = _Builder(params, memo)
    for r in rows:
        spec = r["spec"]
        b.add(int(r["i"]), spec, str(r.get("form") or "call"), str(r.get("name") or ""), definition_ports(spec), r.get("args") or (),
              r.get("bits"))
    return b


def from_instances(path_or_doc: str | dict[str, Any]) -> ProgramView:
    """`instances.json.gz` (or its loaded document) -> ProgramView.  Rows are body positions; the Definition's ports come from the
    registry (`definition_ports`, once per distinct spec); a row of a `batch` form carries no `n`, so its output is read as the
    single-instance signature (the reference request Programs are `call` rows throughout).  A Boolean Program's rows are its word
    view (`pipeline.build_instances`): each Call is the word Definition its Boolean Definition computes, with that Boolean spec as
    `CallView.bits`, so its families, ports and reads are the word Program's.

    A path is STREAMED: the rows are decoded one at a time into the columnar body (`iter_instance_rows`), the file read once.  A file
    whose rows are not in body order (never written by `derive_step`) is read again as a document; a document's rows are sorted."""
    if isinstance(path_or_doc, str):
        hdr = _read_instance_header(path_or_doc)
        params = _params_view(hdr.get("params") or [])
        try:
            b = _instances_columns(params, iter_instance_rows(path_or_doc, expand=False))
        except _OutOfOrder:
            with _open_text(path_or_doc) as f:
                doc = json.load(f)
            return from_instances(doc)
        return b.view((), hdr.get("digest"), "instances")
    doc = path_or_doc
    params = _params_view(doc.get("params") or [])
    rows = sorted(doc["rows"], key=lambda r: int(r["i"]))
    try:
        b = _instances_columns(params, rows, memo=True)
    except _OutOfOrder as e:
        raise ValueError(f"instance rows are not a body in order: {e}") from None
    return b.view((), doc.get("digest"), "instances")


def from_descriptor(desc: dict[str, Any] | str, registry=None) -> ProgramView:
    """A Program descriptor (dict, or a path to `descriptor.json[.gz]`), decoded by `codec.decode_program` -> ProgramView of its
    program function's body, or of the root's body when the program function is inlined into the root (a request Build's
    descriptor).  Names from the decoded annotations' `node_hints` (node index -> hint); ports from each node's own output type;
    outputs from the body's `ret`."""
    if isinstance(desc, str):
        with gzip.open(desc, "rt") if desc.endswith(".gz") else open(desc) as f:
            desc = json.load(f)
    from verity.ml.circuits.codec import decode_program, descriptor_id
    from verity.ml.circuits.refs import runs
    if registry is None:
        registry, _errors = _registry()
    from verity.ml.circuits.program import is_input
    decoded = decode_program(desc, registry)
    hints = {int(k): str(v) for k, v in decoded.annotations.get("node_hints", {}).items()}
    fn = decoded.defs.get(decoded.program_fn_id)
    if fn is not None:
        body = fn.body
        params = _params_view((n, t) for n, t in body.params)
        rows, remap = list(enumerate(body.nodes)), (lambda r: r)
    else:
        # the program function is inlined into the root (`codec.encode_program`): its Input nodes are the view's parameters
        body = decoded.root.body
        inputs = {k: j for j, k in enumerate(k for k, n in enumerate(body.nodes) if is_input(n.fn))}
        params = _params_view((hints.get(k) or getattr(body.nodes[k], "name", "") or f"input{j}", body.nodes[k].out)
                              for k, j in inputs.items())
        calls = [k for k in range(len(body.nodes)) if k not in inputs]
        pos = {k: i for i, k in enumerate(calls)}
        rows = [(k, body.nodes[k]) for k in calls]
        remap = lambda r: (("p", inputs[r[1]], *r[2:]) if r[1] in inputs else ("n", pos[r[1]], *r[2:])) if r[0] == "n" else r  # noqa: E731
    b = _Builder(params)
    for i, (k, n) in enumerate(rows):
        b.add(i, descriptor_id(n.fn), n.form, hints.get(k, getattr(n, "name", "") or ""), ports_of_type(n.out),
              ([remap(r) for r in runs(a.refs)] for a in n.args))
    outs = b.outputs([remap(r) for r in runs(body.ret)])
    from verity.ml.circuits.codec import program_digest
    return b.view(outs, program_digest(desc), "descriptor")
