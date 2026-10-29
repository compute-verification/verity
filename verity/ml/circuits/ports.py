"""Port types (PROTOCOL.md §12): what a Definition declares about the values it takes and returns beyond their widths, and the
rules a Program is held to when it is built (`program.Program`) and when its descriptor is decoded (`codec.decode_program`).

A Definition declares `Ports(params={name: tree}, ret=tree)`. A tree is an `In` (on a parameter) or an `Out` (on the return),
which types every leaf below it; a tuple of trees, one per field of a `Tuple` or `Record`; a dict of trees by `Record` field
name; `Leaves`, ports by leaf range; or None. A composite may declare its ports as a function of its statics. Undeclared is the
default (no interval, not a sink, kind value, no word type), and a descriptor carries a `ports` key only for a Definition that
declares one, so an undeclared Definition's entry, id and digests are what they were.

* **Intervals.** `Out(interval=(lo, hi))`: every leaf the Definition returns there is in [lo, hi] (a quantizer's clamp range).
  `In(bound=(lo, hi))`: the Definition assumes every operand leaf there is. An instance whose operand's producer interval is not
  within the bound is refused (`E_PORT_BOUND`); a producer without one spans its width, and a constant's is its value. An
  interval reads a `Value<w>` as signed when lo < 0 and as unsigned when hi >= 2^(w-1) (either, when both fit), and one is
  within another when every encoding it holds is one the other holds. A composite's declared interval must hold of what its
  body returns (`E_PORT_INTERVAL`); a primitive's is a fact about its evaluator, which circuit-check tests. A primitive whose
  return's interval follows from its operands' (an exact step's sum) declares `Ports(transfer=f)` instead: `f(*operands)`
  maps each operand's interval (None when it has none; `as_reading` reads it signed or unsigned) to its return's, or None,
  and the Build gives each instance's return that interval. So a composite over such steps declares, and is held to, the
  interval its operands' bounds prove (the s8 chain's words, which a limb split's bound reads). A bound is never taken on
  trust: it holds by its operand's width, its producer's interval, or a range-check Definition's declared one, so a program
  function's parameter (a run input, registered, a coin or a weight) declares no bound narrower than its width
  (`E_PORT_BOUND`): an input narrower than its type is read through a range check whose `Out(interval)` is the bound. Only
  a `harness` takes the bounds of the Definition it evaluates on its inputs, as preconditions of the run it evaluates.
* **Sinks.** `Out(sink=True)` (a receipt fold's output) reaches only a receipt port (`In(receipt=True)`) or the program's
  outputs. A primitive reading a sink at any other port is refused (`E_PORT_SINK`). A sink passed to a composite is a sink in
  its body, and a composite that returns one declares it. A Definition with a receipt port returns only sinks.
* **Kinds.** `In(kind="coin")` takes only a live verifier coin: a run input of a program-function parameter declared kind coin,
  passed through untouched. `In(kind="static")` takes only a constant (the output of a primitive without operands).
  `In(kind="position")` takes only a unit's position: a run input declared kind position, passed through untouched. Anything
  else is refused (`E_PORT_KIND`), so a noise derivation whose parameters are coins reads coins and its statics only, and no
  unit computes its position. A coin is never a registered input.
* **Model weights.** A Program marks the parameters that are the model's weights (`program.Program(fn, theta=names)`), and
  their input leaves have origin ``theta`` rather than ``input`` or ``registered``: the census and the stacking property tell
  the weights from a request's inputs by it. The marking is `annotations["theta"]`, the root's input nodes, outside every
  digest, so it changes no id or digest, and it changes one verdict: a coin is never a weight (`E_PORT_KIND`). It binds
  nothing, so a consumer that treats a leaf differently for being a weight checks it against the weights' commitment.
* **Words.** `Out(word="s32")` fixes the type of each word its producer writes; `In(word="s8x4")` reads each word as that type.
  A word has one type: a word computed at run time is read only as the type its producer fixed, a typed input (one whose port
  declares a word) only as its port's, and only an untyped input or constant takes the type its reader declares
  (`E_PORT_WORD`). Splitting a runtime word into limbs is a registered `LimbSplit` Definition of its own, which reads the word
  as its producer's type and computes the limbs and their carries. A composite keeps the type of each word it returns (it
  declares it) and never fixes one. Below the word types, every leaf is read at the width it was committed with
  (`defs.Builder.finish`, `E_REINTERPRET`).
* **Coin flow.** A primitive with a coin port is a noise primitive when its return isn't all sinks (it derives noise, or
  adds it to an operand it takes) and a receipt fold when it is (the coin is its key). A Definition, primitive or composite,
  with a coin port and a position port whose return holds no sink is a seed: it derives a unit's seed from the coin and the
  unit's position, and no rule follows the coin into its body. A coin reaches only such a coin port: a primitive reading a
  coin at any other port is refused, and so is a Definition returning one, the root included (`E_COIN_FLOW`), since past a
  return it would be a computed value that no rule follows.
* **One use.** A coin a noise primitive reads is read once in the flattened Program, by that instance alone: no second
  noise and no receipt key from it. Each leaf a noise primitive returns, its noise, is read at most once (by one operand
  of one instance, or once by the outputs): no two copies, and no noise shared across rows, matmuls, batch members or
  scan steps (`E_COIN_REUSE`). A coin only receipt folds read may key any number of them. With coin flow, each noise's
  coins reach nothing but its one reader's forward cone (`Accounting.Guarantees.CoinFlow` in `Security/`). A coin seeds read
  is read by seeds alone, any number of them, and no position leaf is read by two seeds, so each (coin, position) seeds
  one unit (`E_COIN_REUSE`).
* **Seeds.** A seed's output and every value computed from it (the unit's noise) are values no rule here follows: they may
  be read any number of times, and where they reach is the unit's own check. PoUW's compile pass holds each unit's z to
  read none of them (``noise-in-z``), or, under a one-stream scheme, to be made from the words its receipt folds by the
  Definitions that peel the noise back out (``stream``). A dataflow rule can't tell a peeled value from a noised one:
  that the peel removes the noise is its arithmetic.

Facts flow down every call and up through declarations. A parameter's facts, in a Definition's body, are the join over all its
instances of its operands' facts, so a bound sees its operand's producer through any number of calls. A Definition's output
carries what its `Out` declares (a primitive's with a transfer, the interval it gives from that instance's operands; a scanned
one's, none), and the check holds a composite's body to it. `facts(root)` gives every leaf's facts (its interval: the
producer's, or an input's declared bound (its width's, but in a harness), read at the leaf's width by `reading`; its origin;
whether it is a sink; its word type) to the census and the property.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from .refs import Affine, Coll, Concat, Explicit, Refs, runs as ref_runs
from .types import Record, Tuple as TupleT, Type, range_widths, remove_axis

KINDS = ("value", "coin", "static", "position")
WORD = re.compile(r"[a-z][a-z0-9_]*")
MAX_SAFE_INT = 2**53 - 1  # the descriptor's integers (`codec.MAX_SAFE_INT`)

#: where a leaf's value comes from: a gate that computes it, a constant (a primitive without operands), a run input, a coin
#: (a run input of kind coin), a registered input (a pre-seed word), a model weight (an input of a parameter the
#: Program marks theta, whatever its binding time) or a position (a run input of kind position)
ORIGINS = ("gate", "static", "input", "coin", "registered", "theta", "position")
MIXED = "*"  # the word type of a leaf whose instances carry different ones


class PortError(ValueError):
    """A refusal by the port rules. `code` is its name: `E_PORT_DECL` (a malformed declaration), `E_PORT_BOUND`,
    `E_PORT_INTERVAL`, `E_PORT_SINK`, `E_PORT_KIND`, `E_PORT_WORD`, `E_COIN_FLOW` or `E_COIN_REUSE`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _decl(message: str) -> PortError:
    return PortError("E_PORT_DECL", message)


def _pair(v: Any, what: str) -> tuple[int, int] | None:
    if v is None:
        return None
    if not (isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
            and v[0] <= v[1]):
        raise _decl(f"{what} is a pair of integers lo <= hi, not {v!r}")
    return int(v[0]), int(v[1])


def _word(w: Any) -> None:
    if w is not None and not (isinstance(w, str) and WORD.fullmatch(w)):
        raise _decl(f"a word type is a name matching {WORD.pattern}, not {w!r}")


@dataclass(frozen=True)
class In:
    """A parameter's port. `bound`: the interval the Definition assumes of every operand leaf. `kind`: ``value``, ``coin`` (a
    live verifier coin), ``static`` (a constant) or ``position`` (a unit's position, a run input). `receipt`: the port takes
    sinks. `word`: the type it reads each word as."""

    bound: tuple[int, int] | None = None
    kind: str = "value"
    receipt: bool = False
    word: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bound", _pair(self.bound, "a bound"))
        if self.kind not in KINDS:
            raise _decl(f"a kind is one of {KINDS}, not {self.kind!r}")
        if not isinstance(self.receipt, bool):
            raise _decl(f"receipt is a boolean, not {self.receipt!r}")
        _word(self.word)

    @property
    def interval(self) -> tuple[int, int] | None:
        return self.bound

    def to_json(self) -> dict:
        j: dict[str, Any] = {}
        if self.bound is not None:
            j["bound"] = list(self.bound)
        if self.kind != "value":
            j["kind"] = self.kind
        if self.receipt:
            j["receipt"] = True
        if self.word is not None:
            j["word"] = self.word
        return j


@dataclass(frozen=True)
class Out:
    """The return's port. `interval`: every leaf returned there is in it. `sink`: it reaches only a receipt port or the
    program's outputs. `word`: the type of each word, fixed here by a primitive and kept by a composite."""

    interval: tuple[int, int] | None = None
    sink: bool = False
    word: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "interval", _pair(self.interval, "an interval"))
        if not isinstance(self.sink, bool):
            raise _decl(f"sink is a boolean, not {self.sink!r}")
        _word(self.word)

    def to_json(self) -> dict:
        j: dict[str, Any] = {}
        if self.interval is not None:
            j["interval"] = list(self.interval)
        if self.sink:
            j["sink"] = True
        if self.word is not None:
            j["word"] = self.word
        return j


_KEYS = {In: ("bound", "kind", "receipt", "word"), Out: ("interval", "sink", "word")}


def _port_from_json(cls: type, j: Any, where: str) -> In | Out:
    if not (isinstance(j, dict) and j and set(j) <= set(_KEYS[cls])):
        raise _decl(f"{where}: a port is a non-empty object with keys from {list(_KEYS[cls])}, not {j!r}")
    p = cls(**j)
    if p.to_json() != j:
        raise _decl(f"{where}: port {j!r} is not canonical (a default is omitted)")
    return p


@dataclass(frozen=True)
class Leaves:
    """Ports by leaf range: `runs` are `(lo, hi, port)`, each typing leaves [lo, hi) of the tree's type."""

    runs: tuple = ()


@dataclass(frozen=True, eq=False)
class Ports:
    """A Definition's declaration: `params` maps a parameter's name to its tree, `ret` is the return's tree. `transfer`, a
    primitive's only: its return's interval from its operands' (`f(*intervals) -> (lo, hi) | None`), a fact about its
    evaluator like a declared interval, which circuit-check tests."""

    params: Mapping[str, Any] = field(default_factory=dict)
    ret: Any = None
    transfer: Callable[..., tuple[int, int] | None] | None = None


#: one run of a bound declaration: (lo, hi, port, w), typing leaves [lo, hi), all of width w (None when they differ)
Run = tuple


@dataclass(frozen=True)
class PortTypes:
    """A declaration bound to a signature: per parameter and for the return, its typed runs of leaves, ascending, maximal and
    without defaults. A root's `inputs` are its input nodes' runs, `(node, lo, hi, port, w)`. `transfer`: a primitive's
    (`Ports.transfer`)."""

    params: tuple[tuple[Run, ...], ...]
    ret: tuple[Run, ...]
    inputs: tuple[tuple, ...] = ()
    transfer: Callable[..., tuple[int, int] | None] | None = None

    def to_json(self) -> dict:
        """The descriptor's `ports` (PROTOCOL.md §12): only what is declared, in leaf order."""
        j: dict[str, Any] = {}
        ps = [[i, lo, hi, p.to_json()] for i, rs in enumerate(self.params) for lo, hi, p, _w in rs]
        if ps:
            j["params"] = ps
        if self.ret:
            j["ret"] = [[lo, hi, o.to_json()] for lo, hi, o, _w in self.ret]
        if self.inputs:
            j["inputs"] = [[k, lo, hi, p.to_json()] for k, lo, hi, p, _w in self.inputs]
        if self.transfer is not None:
            j["transfer"] = True
        return j

    @staticmethod
    def from_json(owner: str, j: Any, params: Sequence[tuple[str, Type]], ret: Type,
                  nodes: Sequence[Type] | None = None) -> "PortTypes":
        """The descriptor's `ports` of `owner`, refused unless well formed and canonical. `nodes`: the root's node output types,
        for its input nodes' runs (only a root has them)."""
        if isinstance(j, dict) and "transfer" in j:
            raise _decl(f"{owner}: only a primitive declares a transfer; a composite's intervals are what its body returns")
        if not (isinstance(j, dict) and j and set(j) <= {"params", "ret", "inputs"}):
            raise _decl(f"{owner}: ports is a non-empty object with keys from params, ret and inputs, not {sorted(j) if isinstance(j, dict) else j!r}")
        if "inputs" in j and nodes is None:
            raise _decl(f"{owner}: only the root declares its input nodes' ports")

        def rows(key: str, n: int) -> list:
            v = j.get(key, [])
            if not isinstance(v, list) or not all(isinstance(r, list) and len(r) == n + 1 and all(
                    isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in r[:n]) for r in v):
                raise _decl(f"{owner}: ports.{key} is a list of [{', '.join(['index', 'lo', 'hi'][3 - n:])}, port] rows")
            return v

        per: list[list] = [[] for _ in params]
        for i, lo, hi, pj in rows("params", 3):
            if i >= len(params):
                raise _decl(f"{owner}: ports name parameter {i}, of {len(params)}")
            per[i].append((lo, hi, _port_from_json(In, pj, f"{owner} parameter {i}")))
        ps = tuple(_merged(owner, f"parameter {name!r}", t, per[i]) for i, (name, t) in enumerate(params))
        rs = _merged(owner, "its return", ret, [(lo, hi, _port_from_json(Out, oj, f"{owner} return")) for lo, hi, oj in rows("ret", 2)])
        by_node: dict[int, list] = {}
        for k, lo, hi, pj in rows("inputs", 3):
            if nodes is None or k >= len(nodes):
                raise _decl(f"{owner}: ports name input node {k}, of {0 if nodes is None else len(nodes)} nodes")
            by_node.setdefault(k, []).append((lo, hi, _port_from_json(In, pj, f"{owner} input node {k}")))
        ins = tuple((k, *r) for k in sorted(by_node) for r in _merged(owner, f"input node {k}", nodes[k], by_node[k]))
        pt = _bound(owner, ps, rs, ret, ins)
        if pt is None or pt.to_json() != j:
            raise _decl(f"{owner}: ports are not canonical (ascending, maximal runs; what the writer gives is "
                        f"{None if pt is None else pt.to_json()})")
        return pt


def full(iv: tuple[int, int], w: int) -> bool:
    """Whether an interval of `Value<w>` holds every encoding (it assumes nothing)."""
    return iv in ((-(1 << (w - 1)), (1 << (w - 1)) - 1), (0, (1 << w) - 1))


def reading(iv: tuple[int, int] | None, w: int | None) -> tuple[int, int] | None:
    """A fact's interval as an interval of a `Value<w>`: None when there is none, or when it is no interval of that width (a
    hull of a signed and an unsigned reading: lo < 0 and hi >= 2^(w-1)), which holds nothing."""
    if iv is None or w is None or (iv[0] < 0 and iv[1] >= 1 << (w - 1)) or iv[0] < -(1 << (w - 1)) or iv[1] > (1 << w) - 1:
        return None
    return iv


def _codes(iv: tuple[int, int] | None, w: int | None) -> tuple[tuple[int, int], ...] | None:
    """The encodings an interval of a `Value<w>` holds, as ascending ranges; None when it is none (`reading`)."""
    r = reading(iv, w)
    if r is None or w is None:
        return None
    lo, hi = r
    m = 1 << w
    if lo >= 0:
        return ((lo, hi),)
    if hi < 0:
        return ((lo + m, hi + m),)
    return ((0, hi), (lo + m, m - 1))


def as_reading(iv: tuple[int, int] | None, w: int, signed: bool = True) -> tuple[int, int] | None:
    """The encodings interval `iv` of a `Value<w>` holds, read as signed (or unsigned) integers: one interval, or None when
    `iv` holds nothing (`reading`) or they are not one interval in that reading. A transfer reads its operands through it."""
    cs = _codes(iv, w)
    if cs is None:
        return None
    m, h = 1 << w, 1 << (w - 1)
    parts: list[tuple[int, int]] = []
    for a, b in cs:
        if signed and a < h <= b:
            parts += [(a, h - 1), (h - m, b - m)]
        else:
            parts.append((a - m, b - m) if signed and a >= h else (a, b))
    parts.sort()
    lo, hi = parts[0]
    for a, b in parts[1:]:
        if a > hi + 1:
            return None
        hi = max(hi, b)
    return lo, hi


def holds(iv: tuple[int, int], w: int, e: int) -> bool:
    """Whether encoding `e` of a `Value<w>` lies in interval `iv`, read as `iv` reads it (signed when lo < 0)."""
    lo, hi = iv
    v = e - (1 << w) if lo < 0 and e >= 1 << (w - 1) else e
    return lo <= v <= hi


def _check_interval(owner: str, where: str, iv: tuple[int, int], w: int | None) -> None:
    if w is None:
        raise _decl(f"{owner}: {where}: an interval types leaves of one width")
    lo, hi = iv
    if lo < -(1 << (w - 1)) or hi > (1 << w) - 1 or (lo < 0 and hi >= 1 << (w - 1)):
        raise _decl(f"{owner}: {where}: [{lo}, {hi}] is no interval of a Value<{w}>: it lies within signed "
                    f"[{-(1 << (w - 1))}, {(1 << (w - 1)) - 1}] or unsigned [0, {(1 << w) - 1}]")
    if max(-lo, hi) > MAX_SAFE_INT:
        raise _decl(f"{owner}: {where}: [{lo}, {hi}] leaves the descriptor's integers (magnitude at most 2^53 - 1)")


def _merged(owner: str, where: str, t: Type, runs: list) -> tuple[Run, ...]:
    """`runs` (lo, hi, port) over the leaves of `t` as maximal ascending runs without defaults, each with its width."""
    runs = sorted((r for r in runs if r[2].to_json()), key=lambda r: r[0])
    out: list[list] = []
    for lo, hi, p in runs:
        if not (0 <= lo < hi <= t.leaves):
            raise _decl(f"{owner}: {where}: leaves [{lo}, {hi}) of a type with {t.leaves}")
        if out and lo < out[-1][1]:
            raise _decl(f"{owner}: {where}: two ports type leaf {lo}")
        if out and lo == out[-1][1] and p == out[-1][2]:
            out[-1][1] = hi
        else:
            out.append([lo, hi, p])
    typed = []
    for lo, hi, p in out:
        ws = range_widths(t, lo, hi)
        w = next(iter(ws)) if len(ws) == 1 else None
        if p.interval is not None:
            _check_interval(owner, where, p.interval, w)
        typed.append((lo, hi, p, w))
    return tuple(typed)


def _flatten(owner: str, where: str, t: Type, tree: Any, cls: type, base: int, out: list) -> None:
    if tree is None:
        return
    if isinstance(tree, (In, Out)):
        if not isinstance(tree, cls):
            raise _decl(f"{owner}: {where} takes {'an In' if cls is In else 'an Out'}, not {tree!r}")
        if t.leaves:
            out.append((base, base + t.leaves, tree))
        return
    if isinstance(tree, Leaves):
        for r in tree.runs:
            if not (isinstance(r, tuple) and len(r) == 3 and isinstance(r[2], cls)):
                raise _decl(f"{owner}: {where}: Leaves holds (lo, hi, {cls.__name__}) runs, not {r!r}")
            out.append((base + r[0], base + r[1], r[2]))
        return
    if isinstance(tree, dict):
        if not isinstance(t, Record):
            raise _decl(f"{owner}: {where}: ports by field name type a Record, not {t!r}")
        names = [k for k, _ in t.fields]
        for name, sub in tree.items():
            if name not in names:
                raise _decl(f"{owner}: {where} has no field {name!r} (it has {names})")
            i = names.index(name)
            _flatten(owner, f"{where}.{name}", t.fields[i][1], sub, cls, base + t.field_offset(i), out)
        return
    if isinstance(tree, (tuple, list)):
        if not isinstance(t, (TupleT, Record)) or len(tree) != len(t.fields):
            raise _decl(f"{owner}: {where}: {len(tree)} field ports for {t!r}")
        for i, sub in enumerate(tree):
            ft = t.fields[i][1] if isinstance(t, Record) else t.fields[i]
            _flatten(owner, f"{where}[{i}]", ft, sub, cls, base + t.field_offset(i), out)
        return
    raise _decl(f"{owner}: {where}: a port tree is an In or Out, a tuple, a dict, Leaves or None, not {tree!r}")


def _bound(owner: str, params: tuple, ret: tuple[Run, ...], ret_type: Type, inputs: tuple = (),
           transfer: Callable | None = None) -> PortTypes | None:
    if not (any(params) or ret or inputs or transfer):
        return None
    if any(p.receipt for rs in params for _lo, _hi, p, _w in rs) and sum(hi - lo for lo, hi, o, _w in ret if o.sink) != ret_type.leaves:
        raise _decl(f"{owner}: a Definition with a receipt port returns only sinks: declare Out(sink=True) on its whole return")
    return PortTypes(tuple(params), ret, inputs, transfer)


def bind_ports(owner: str, params: Sequence[tuple[str, Type]], ret: Type, decl: Any,
               primitive: bool = False) -> PortTypes | None:
    """`decl` (a `Ports`, or None) bound to the signature `params -> ret` of `owner` (a primitive's when `primitive`); None
    when it declares nothing."""
    if decl is None:
        return None
    if not isinstance(decl, Ports):
        raise _decl(f"{owner}: ports are declared as Ports(params=..., ret=...), not {decl!r}")
    if decl.transfer is not None:
        if not primitive:
            raise _decl(f"{owner}: only a primitive declares a transfer; a composite's intervals are what its body returns")
        if not callable(decl.transfer):
            raise _decl(f"{owner}: a transfer is a function of the operands' intervals, not {decl.transfer!r}")
    names = [n for n, _ in params]
    if not isinstance(decl.params, Mapping):
        raise _decl(f"{owner}: Ports.params maps parameter names to port trees, not {decl.params!r}")
    unknown = sorted(set(decl.params) - set(names))
    if unknown:
        raise _decl(f"{owner} has no parameter {unknown} (it has {names})")
    ps = []
    for name, t in params:
        out: list = []
        _flatten(owner, f"parameter {name!r}", t, decl.params.get(name), In, 0, out)
        ps.append(_merged(owner, f"parameter {name!r}", t, out))
    out = []
    _flatten(owner, "its return", ret, decl.ret, Out, 0, out)
    rs = _merged(owner, "its return", ret, out)
    if decl.transfer is not None and any(o.interval is not None for _lo, _hi, o, _w in rs):
        raise _decl(f"{owner}: its transfer gives its return's interval: declare no Out(interval) beside it")
    return _bound(owner, tuple(ps), rs, ret, transfer=decl.transfer)


def port_types(fn: Any) -> PortTypes | None:
    """The bound port types of a Definition (a primitive, a specialization or a root), None when it declares none."""
    return getattr(fn, "port_types", None)


def declared(fn: Any) -> dict | None:
    """What a Definition's descriptor entry carries as `ports` (the census's class `ports`), None when it declares none."""
    pt = port_types(fn)
    return None if pt is None else pt.to_json()


def forward(fn: Any) -> Ports | None:
    """The declaration that gives a wrapper of `fn` (`fn`'s parameters, returning `fn`'s output) `fn`'s ports, so that a one-Call
    Program of `fn` (`evaluation.standalone_call`, `cut.call_scope`) takes its inputs at `fn`'s bounds, kinds and words."""
    pt = port_types(fn)
    if pt is None:
        return None
    params, _ = fn.signature()
    return Ports(params={name: Leaves(tuple((lo, hi, p) for lo, hi, p, _w in rs)) for (name, _t), rs in zip(params, pt.params) if rs},
                 ret=Leaves(tuple((lo, hi, o) for lo, hi, o, _w in pt.ret)) if pt.ret else None)


def harness(defn: Any) -> Any:
    """Mark `defn` (a program function's `CompositeDefinition`) a harness: the wrapper of a one-Call Program that evaluates a
    Definition on arbitrary operands (`evaluation.standalone_call`, `cut.call_scope`). A harness's inputs may feed a kind=static
    port, each taken as a constant of the run it evaluates, and declare a bound narrower than their width, each a precondition
    of that run; no other Program's (nor a decoded one's) may."""
    defn.harness = True
    return defn


def _is_harness(fn: Any) -> bool:
    return getattr(getattr(fn, "defn", None), "harness", False) is True


def root_ports(fn: Any, input_colls: Sequence[Coll], registered: Sequence[bool], nodes: Sequence[Type]) -> PortTypes | None:
    """A root's port types: its input nodes (of output types `nodes`, referenced by `input_colls` per parameter) take the
    program function's parameter ports, its return the function's return ports. A coin is flipped per run, so a coin parameter is
    never registered; a parameter is an input, so never static and never bounded narrower than its width (but in a
    `harness`)."""
    pt = port_types(fn)
    if pt is None:
        return None
    params, ret = fn.signature()
    runs: dict[int, list] = {}
    for (name, _t), rs, coll, reg in zip(params, pt.params, input_colls, registered):
        for lo, hi, p, w in rs:
            if p.kind == "static" and not _is_harness(fn):
                raise PortError("E_PORT_KIND", f"program function {fn.id}: parameter {name!r} is kind=static, but a program "
                                f"function's parameters are inputs and a static is a constant: make it a static of the "
                                f"Definition that reads it, or a constant gate")
            if p.kind == "coin" and reg:
                raise PortError("E_PORT_KIND", f"program function {fn.id}: parameter {name!r} is kind=coin, so it is drawn "
                                f"per run and never registered")
            if p.bound is not None and not full(p.bound, w) and not _is_harness(fn):
                raise PortError("E_PORT_BOUND", f"program function {fn.id}: parameter {name!r} leaves [{lo}, {hi}) declare the "
                                f"bound [{p.bound[0]}, {p.bound[1]}], narrower than Value<{w}>, but a run input's bound is "
                                f"never taken on trust: read the input through a range-check Definition (a primitive whose "
                                f"declared Out(interval) is the bound, which circuit-check tests) and declare the bound on "
                                f"the readers of its output, or drop it")
            for _s, k, a, b in ref_runs(coll.refs.slice(lo, hi)):
                runs.setdefault(k, []).append((a, b, p))
    root = f"Root[{fn.id}]"
    ins = tuple((k, *r) for k in sorted(runs) for r in _merged(root, f"input node {k}", nodes[k], runs[k]))
    return _bound(root, (), pt.ret, ret, ins)


# --- facts ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Fact:
    """What is known of one leaf: `interval` (its producer's, or an input's declared bound; None: none, it spans its width),
    `origin` (a subset of `ORIGINS`: more than one when its instances differ), `sink`, and `word` (its type, `MIXED` when its
    instances differ, None when none is fixed)."""

    interval: tuple[int, int] | None
    origin: frozenset
    sink: bool
    word: str | None

    @property
    def kind(self) -> str:
        """``coin`` for a coin passed through untouched, ``static`` for a constant, ``position`` for a position passed through
        untouched, else ``value``."""
        return ("coin" if self.origin == _COIN else "static" if self.origin == _STATIC else
                "position" if self.origin == _POSITION else "value")

    def to_json(self) -> dict:
        return {"interval": None if self.interval is None else list(self.interval), "origin": sorted(self.origin),
                "sink": self.sink, "word": self.word}


_FACTS: dict[tuple, Fact] = {}
_JOINS: dict[tuple[int, int], Fact] = {}


def fact(interval: tuple[int, int] | None = None, origin: Any = ("gate",), sink: bool = False, word: str | None = None) -> Fact:
    """The (interned) fact: equal facts are one object."""
    key = (interval, frozenset(origin), sink, word)
    f = _FACTS.get(key)
    if f is None:
        f = _FACTS[key] = Fact(*key)
    return f


GATE = fact()
_COIN, _STATIC, _POSITION = frozenset({"coin"}), frozenset({"static"}), frozenset({"position"})


def join(a: Fact, b: Fact) -> Fact:
    """The least fact both are: the hull of the intervals (None absorbs), the union of origins, a sink if either is, the word
    type if they share it."""
    if a is b:
        return a
    key = (id(a), id(b))
    f = _JOINS.get(key)
    if f is None:
        iv = None if a.interval is None or b.interval is None else (min(a.interval[0], b.interval[0]), max(a.interval[1], b.interval[1]))
        f = _JOINS[key] = fact(iv, a.origin | b.origin, a.sink or b.sink, a.word if a.word == b.word else MIXED)
    return f


class Runs:
    """The facts of a collection's leaves, as maximal runs: run i covers leaves [ends[i - 1], ends[i]) (from 0), with `facts[i]`."""

    __slots__ = ("ends", "facts")

    def __init__(self, ends: list[int], facts: list[Fact]) -> None:
        self.ends, self.facts = ends, facts

    @staticmethod
    def uniform(n: int, f: Fact) -> "Runs":
        return Runs([n], [f]) if n else Runs([], [])

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Runs) and self.ends == other.ends and all(a is b for a, b in zip(self.facts, other.facts))

    @property
    def one(self) -> Fact | None:
        """The fact of every leaf, when they share one."""
        return self.facts[0] if len(self.facts) == 1 else None

    @property
    def has_sink(self) -> bool:
        return any(f.sink for f in self.facts)

    def at(self, x: int) -> Fact:
        return self.facts[bisect.bisect_right(self.ends, x)]

    def segments(self, lo: int = 0, hi: int | None = None) -> Iterator[tuple[int, int, Fact]]:
        """`(a, b, fact)` over leaves [lo, hi), ascending."""
        hi = len(self) if hi is None else hi
        i, a = bisect.bisect_right(self.ends, lo), lo
        while a < hi:
            b = min(self.ends[i], hi)
            yield a, b, self.facts[i]
            a, i = b, i + 1

    def to_json(self) -> list:
        return [[a, b, f.to_json()] for a, b, f in self.segments()]

    def __repr__(self) -> str:
        return f"Runs({list(self.segments())})"


class _Build:
    __slots__ = ("ends", "facts")

    def __init__(self) -> None:
        self.ends: list[int] = []
        self.facts: list[Fact] = []

    def add(self, n: int, f: Fact) -> None:
        if n <= 0:
            return
        if self.facts and self.facts[-1] is f:
            self.ends[-1] += n
        else:
            self.ends.append((self.ends[-1] if self.ends else 0) + n)
            self.facts.append(f)

    def extend(self, r: Runs, lo: int = 0, hi: int | None = None) -> None:
        for a, b, f in r.segments(lo, hi):
            self.add(b - a, f)

    def done(self) -> Runs:
        return Runs(self.ends, self.facts)


def _cat(*rs: Runs) -> Runs:
    B = _Build()
    for r in rs:
        B.extend(r)
    return B.done()


def _slice(r: Runs, lo: int, hi: int) -> Runs:
    f = r.one
    if f is not None:
        return Runs.uniform(hi - lo, f)
    B = _Build()
    B.extend(r, lo, hi)
    return B.done()


def _repeat(r: Runs, n: int) -> Runs:
    f = r.one
    if f is not None or not len(r):
        return Runs.uniform(len(r) * n, f) if f is not None else Runs([], [])
    B = _Build()
    for _ in range(n):
        B.extend(r)
    return B.done()


def join_runs(a: Runs | None, b: Runs) -> Runs:
    """Leafwise `join` of two collections' facts (`a` None: `b`)."""
    if a is None or a == b:
        return b
    fa, fb = a.one, b.one
    if fa is not None and fb is not None:
        return Runs.uniform(len(a), join(fa, fb))
    B = _Build()
    i = j = pos = 0
    while i < len(a.ends):
        e = min(a.ends[i], b.ends[j])
        B.add(e - pos, join(a.facts[i], b.facts[j]))
        pos = e
        i += a.ends[i] == e
        j += b.ends[j] == e
    return B.done()


def _gather_into(B: _Build, r: Refs, env: Mapping[tuple[str, int], Runs]) -> None:
    if isinstance(r, Concat):
        for p in r.parts:
            _gather_into(B, p, env)
    elif isinstance(r, Affine):
        if r.count:
            B.extend(env[(r.space, r.idx)], r.base, r.base + r.count)
    elif isinstance(r, Explicit):
        for s, i, leaf in r.items:
            B.add(1, env[(s, i)].at(leaf))
    elif r.count:
        src = env[(r.space, r.idx)]  # type: ignore[attr-defined]
        f = src.one
        if f is not None:
            B.add(r.count, f)
        else:
            for _s, _i, lo, hi in ref_runs(r):
                B.extend(src, lo, hi)


def gather(refs: Refs, env: Mapping[tuple[str, int], Runs]) -> Runs:
    """The facts of the leaves `refs` names, in its order, from the facts `env` holds per collection (`(space, idx)`)."""
    B = _Build()
    _gather_into(B, refs, env)
    return B.done()


def _members(F: Runs, t: Type, ax: int, n: int) -> Runs:
    """The join over the `n` members along axis `ax` of an argument of type `t` with facts `F`: what every member's parameter is."""
    _, mt = remove_axis(t, ax)
    f = F.one
    if f is not None or n == 0:
        return Runs.uniform(mt.leaves, f if f is not None else GATE)
    acc: Runs | None = None
    if ax == 0:
        L = mt.leaves
        for m in range(n):
            acc = join_runs(acc, _slice(F, m * L, (m + 1) * L))
    else:
        pos, env = Coll(t, Affine("a", 0, 0, t.leaves)), {("a", 0): F}
        for m in range(n):
            acc = join_runs(acc, gather(pos.take(ax, m).refs, env))
    assert acc is not None
    return acc


def _covered(runs: Sequence[tuple[int, int]], a: int, b: int) -> int | None:
    """The first leaf of [a, b) outside the union of `runs` (ascending, disjoint), None when they cover it."""
    for lo, hi in runs:
        if hi <= a:
            continue
        if lo > a:
            return a
        a = hi
        if a >= b:
            return None
    return a if a < b else None


def _within(iv: tuple[int, int] | None, bound: tuple[int, int], w: int | None) -> bool:
    """Whether every encoding `iv` holds is one `bound` holds, each read as it reads itself."""
    if w is not None and full(bound, w):
        return True
    have, allowed = _codes(iv, w), _codes(bound, w)
    return have is not None and allowed is not None and all(any(a <= lo and hi <= b for a, b in allowed) for lo, hi in have)


def _producer(f: Fact, w: int | None) -> str:
    if f.interval is None:
        return f"its producer's interval: it declares none, so the value spans Value<{w}>"
    what = "an input's declared bound" if "gate" not in f.origin and "static" not in f.origin else "its producer's interval"
    return f"{what} [{f.interval[0]}, {f.interval[1]}]"


def _origin(f: Fact) -> str:
    names = {"gate": "a computed value", "static": "a constant", "input": "a run input", "coin": "a coin",
             "registered": "a registered input", "theta": "a model weight", "position": "a position"}
    return " or ".join(names[o] for o in ORIGINS if o in f.origin)


def _has_coin(r: Runs) -> bool:
    return any("coin" in f.origin for f in r.facts)


def _check_operand(at: Callable[[], str], pname: str, runs: Sequence[Run], F: Runs, primitive: bool) -> None:
    """Rules 2-5 and coin flow on one operand: its leaves' facts `F` against the parameter's typed runs."""
    if primitive and _has_coin(F):
        coin_ports = [(lo, hi) for lo, hi, p, _w in runs if p.kind == "coin"]
        for a, b, f in F.segments():
            leaf = _covered(coin_ports, a, b) if "coin" in f.origin else None
            if leaf is not None:
                raise PortError("E_COIN_FLOW", f"{at()} parameter {pname!r} leaf {leaf}: reads a coin, which reaches only "
                                f"the coin port (In(kind='coin')) of a noise primitive or a receipt fold; any other reader "
                                f"could re-derive the noise and cancel it")
    if primitive and F.has_sink:
        receipts = [(lo, hi) for lo, hi, p, _w in runs if p.receipt]
        for a, b, f in F.segments():
            leaf = _covered(receipts, a, b) if f.sink else None
            if leaf is not None:
                raise PortError("E_PORT_SINK", f"{at()} parameter {pname!r} leaf {leaf}: reads a sink (a value declared "
                                f"Out(sink=True), such as a receipt fold's output); a sink reaches only a receipt port "
                                f"(In(receipt=True)) or the program's outputs")
    for lo, hi, p, w in runs:
        if p.bound is None and p.kind == "value" and p.word is None:
            continue
        for a, _b, f in F.segments(lo, hi):
            if p.bound is not None and not _within(f.interval, p.bound, w):
                raise PortError("E_PORT_BOUND", f"{at()} parameter {pname!r} leaf {a}: its declared bound [{p.bound[0]}, "
                                f"{p.bound[1]}] does not contain {_producer(f, w)}")
            if p.kind == "coin" and f.origin != _COIN:
                raise PortError("E_PORT_KIND", f"{at()} parameter {pname!r} leaf {a}: takes kind=coin, a live verifier coin "
                                f"(a program input declared In(kind='coin')) passed through untouched, but reads {_origin(f)}")
            if p.kind == "static" and f.origin != _STATIC:
                raise PortError("E_PORT_KIND", f"{at()} parameter {pname!r} leaf {a}: takes kind=static, a constant, but "
                                f"reads {_origin(f)}")
            if p.kind == "position" and f.origin != _POSITION:
                raise PortError("E_PORT_KIND", f"{at()} parameter {pname!r} leaf {a}: takes kind=position, a unit's "
                                f"position (a program input declared In(kind='position')) passed through untouched, but "
                                f"reads {_origin(f)}: a unit never computes its position")
            if p.word is not None and f.word != p.word and ("gate" in f.origin or f.word is not None):
                got = "untyped" if f.word is None else "mixed-type" if f.word == MIXED else f"{f.word}"
                raise PortError("E_PORT_WORD", f"{at()} parameter {pname!r} leaf {a}: reads a {p.word} word, but its value "
                                f"is a {'runtime ' if 'gate' in f.origin else ''}{got} word. A word has one type: a runtime "
                                f"word's is fixed at its producer and a typed input's at its port, and only an untyped input "
                                f"or constant takes its reader's. Split a runtime word into limbs with a registered LimbSplit "
                                f"Definition, which reads it as its producer's type and computes the limbs and their carries")


def _check_ret(sp: Any, pt: PortTypes | None, R: Runs, root: bool) -> None:
    """A composite's return against its declared `Out`: declared intervals and word types hold of what its body returns, and
    every sink and word type it returns is declared (a root returns them freely: its outputs are the receipt)."""
    fid = sp.id
    for a, _b, f in R.segments():
        if "coin" in f.origin:
            where = "the program's outputs" if root else "its return, past which it is a computed value"
            raise PortError("E_COIN_FLOW", f"{fid} returns a coin at return leaf {a}: a coin reaches only the coin port "
                            f"(In(kind='coin')) of a noise primitive or a receipt fold, never {where}")
    outs = pt.ret if pt is not None else ()
    for lo, hi, o, w in outs:
        for a, _b, f in R.segments(lo, hi):
            if o.interval is not None and not _within(f.interval, o.interval, w):
                raise PortError("E_PORT_INTERVAL", f"{fid} declares Out(interval=({o.interval[0]}, {o.interval[1]})) at "
                                f"return leaf {a}, but its body returns {_producer(f, w)}")
            if o.word is not None and f.word != o.word:
                raise PortError("E_PORT_WORD", f"{fid} declares Out(word={o.word!r}) at return leaf {a}, but its body returns "
                                f"a {'untyped' if f.word is None else f.word} word: a composite keeps the type its words' "
                                f"producer fixed, and never fixes one")
    if root:
        return
    sinks = [(lo, hi) for lo, hi, o, _w in outs if o.sink]
    for a, b, f in R.segments():
        leaf = _covered(sinks, a, b) if f.sink else None
        if leaf is not None:
            raise PortError("E_PORT_SINK", f"{fid} returns a sink at return leaf {leaf} without declaring it: declare "
                            f"Out(sink=True) there")
        if f.word == MIXED:
            raise PortError("E_PORT_WORD", f"{fid} returns words of different types at return leaf {a} (they differ between "
                            f"its instances), and a composite keeps one type per word it returns")
        leaf = _covered([(lo, hi) for lo, hi, o, _w in outs if o.word == f.word], a, b) if f.word is not None else None
        if leaf is not None:
            raise PortError("E_PORT_WORD", f"{fid} returns a {f.word} word at return leaf {leaf} without declaring it: "
                            f"declare Out(word={f.word!r}) there")


class Facts:
    """A Program's facts (`facts`): its root body's per node (`node`) and returned (`ret`), and each Definition's parameters'
    (`params`), joined over its instances. `typed`: some Definition in it declares ports."""

    def __init__(self, root: Any, typed: bool, env: dict, ret: Runs, params: dict[int, tuple[Any, tuple[Runs, ...]]]) -> None:
        self.root, self.typed = root, typed
        self._env, self._ret, self._params = env, ret, params

    def node(self, k: int) -> Runs:
        """The facts of root node `k`'s output leaves."""
        return self._env[("n", k)]

    def ret(self) -> Runs:
        """The facts of the leaves the root returns."""
        return self._ret

    def params(self, fn: Any) -> tuple[Runs, ...]:
        """The facts of each parameter of a Definition the root instantiates (by object), joined over its instances: its
        operands' producers' facts."""
        return self._params[id(fn)][1]

    def definitions(self) -> Iterator[tuple[Any, tuple[Runs, ...]]]:
        """Every specialization the root instantiates, with its parameters' facts."""
        return iter(self._params.values())


def _reachable(root: Any) -> tuple[list, bool]:
    """The specializations `root` instantiates, callers before callees, and whether any reachable Definition declares ports."""
    from .defs import SpecializedDefinition

    typed = port_types(root) is not None
    post: list = []
    seen = {id(root)}
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        sp, i = stack.pop()
        nodes = sp.body.nodes
        while i < len(nodes):
            fn = nodes[i].fn
            i += 1
            if not typed and port_types(fn) is not None:
                typed = True
            if isinstance(fn, SpecializedDefinition) and id(fn) not in seen:
                seen.add(id(fn))
                stack.append((sp, i))
                stack.append((fn, 0))
                break
        else:
            post.append(sp)
    post.reverse()
    return post, typed


def _value(fn: Any) -> tuple[int, int] | None:
    """A constant's interval: its value, its encoding read as signed when its top bit is set (and w > 1); None for one that
    reads a table."""
    if fn.tables:
        return None
    e, w = int(fn.evaluate()), fn.ret.w
    v = e - (1 << w) if w > 1 and e >> (w - 1) else e
    return v, v


def _declared_out(fn: Any, cache: dict) -> Runs:
    hit = cache.get(id(fn))
    if hit is not None and hit[0] is fn:
        return hit[1]
    from .defs import PrimitiveDefinition

    _, ret = fn.signature()
    base = fact(_value(fn), ("static",)) if isinstance(fn, PrimitiveDefinition) and not fn.params else GATE
    pt = port_types(fn)
    B, pos = _Build(), 0
    for lo, hi, o, _w in (pt.ret if pt is not None else ()):
        B.add(lo - pos, base)
        B.add(hi - lo, fact(o.interval, base.origin, o.sink, o.word))
        pos = hi
    B.add(ret.leaves - pos, base)
    r = B.done()
    cache[id(fn)] = (fn, r)
    return r


def _width(t: Type) -> int | None:
    ws = t.width_set()
    return next(iter(ws)) if len(ws) == 1 else None


def _transferred(fn: Any, pt: PortTypes, params: Sequence[tuple[str, Type]], ret: Type, ops: Sequence[Runs], out: Runs) -> Runs:
    """A primitive instance's return facts `out` with the interval its transfer gives from its operands' facts `ops` (each
    operand's interval the hull of its leaves', None when one has none)."""
    ivs: list[tuple[int, int] | None] = []
    for (_name, t), F in zip(params, ops):
        w = _width(t)
        iv: tuple[int, int] | None = None
        for _a, _b, f in F.segments():
            r = reading(f.interval, w)
            if r is None:
                iv = None
                break
            iv = r if iv is None else (min(iv[0], r[0]), max(iv[1], r[1]))
        ivs.append(reading(iv, w))
    transfer = pt.transfer
    assert transfer is not None
    r = transfer(*ivs)
    if r is None:
        return out
    iv = _pair(r, f"{fn.id}'s transfer on {ivs} gives {r!r}: an interval")
    assert iv is not None
    _check_interval(fn.id, f"its transfer on {ivs}", iv, _width(ret))
    B = _Build()
    for a, b, f in out.segments():
        B.add(b - a, fact(iv, f.origin, f.sink, f.word))
    return B.done()


def _receive(r: Runs, runs: Sequence[Run], seed: bool = False) -> Runs:
    """A body's view of a parameter: a receipt port's leaves are no sinks inside (what the Definition returns is), and a seed's
    coin port's are values its gates read, which no coin rule follows."""
    rec = [(lo, hi, p) for lo, hi, p, _w in runs if p.receipt or (seed and p.kind == "coin")]
    if not rec or not (r.has_sink or _has_coin(r)):
        return r
    B, pos = _Build(), 0
    for lo, hi, p in rec:
        B.extend(r, pos, lo)
        for a, b, f in r.segments(lo, hi):
            if p.receipt and f.sink:
                f = fact(f.interval, f.origin, False, f.word)
            if seed and p.kind == "coin" and "coin" in f.origin:
                f = fact(f.interval, (f.origin - _COIN) | {"gate"}, f.sink, f.word)
            B.add(b - a, f)
        pos = hi
    B.extend(r, pos, len(r))
    return B.done()


def _input_facts(root: Any, k: int, node: Any, runs: list, harness: bool, theta: bool = False) -> Runs:
    from .program import is_registered

    reg = is_registered(node.fn)
    base = fact(origin=("theta",) if theta else ("registered",) if reg else ("input",))
    B, pos = _Build(), 0
    for lo, hi, p, w in runs:
        if p.kind == "static" and not harness:
            raise PortError("E_PORT_KIND", f"{root.id} input node {k}: an input is never kind=static (only a harness's)")
        if p.kind == "coin" and reg:
            raise PortError("E_PORT_KIND", f"{root.id} input node {k}: a coin is drawn per run, never registered "
                            f"({node.fn.id})")
        if p.kind == "coin" and theta:
            raise PortError("E_PORT_KIND", f"{root.id} input node {k}: a coin is the verifier's, never a model weight: "
                            f"take it in a parameter the Program doesn't mark theta")
        if hi > node.out.leaves:
            raise _decl(f"{root.id} input node {k}: ports type leaves [{lo}, {hi}) of {node.out.leaves}")
        if p.bound is not None and not full(p.bound, w) and not harness:
            raise PortError("E_PORT_BOUND", f"{root.id} input node {k}: leaves [{lo}, {hi}) declare the bound [{p.bound[0]}, "
                            f"{p.bound[1]}], narrower than Value<{w}>, but an input's bound is never taken on trust (only a "
                            f"harness's): a range-check Definition narrows it")
        B.add(lo - pos, base)
        B.add(hi - lo, fact(p.bound, (p.kind,) if p.kind != "value" else base.origin, False, p.word))
        pos = hi
    B.add(node.out.leaves - pos, base)
    return B.done()


def _analyse(root: Any, order: list, typed: bool) -> Facts:
    from .defs import PrimitiveDefinition
    from .program import is_input

    joined: dict[int, list] = {}
    params_out: dict[int, tuple[Any, tuple[Runs, ...]]] = {}
    outs: dict = {}
    root_env: dict = {}
    root_ret = Runs([], [])
    rpt = port_types(root)
    harness_ = _is_harness(getattr(root, "program_fn", None))
    inputs: dict[int, list] = {}
    for k, lo, hi, p, w in (rpt.inputs if rpt is not None else ()):
        inputs.setdefault(k, []).append((lo, hi, p, w))
    for k in inputs:
        if not (k < len(root.body.nodes) and is_input(root.body.nodes[k].fn)):
            raise _decl(f"{root.id}: ports type the inputs of node {k}, which is no input node")
    theta = getattr(root, "theta", ())
    if not (isinstance(theta, (tuple, frozenset)) and all(
            isinstance(k, int) and not isinstance(k, bool) and 0 <= k < len(root.body.nodes) and is_input(root.body.nodes[k].fn)
            for k in theta)):
        raise _decl(f"{root.id}: annotations.theta lists input nodes of the root, not {list(theta) if isinstance(theta, tuple) else theta!r}")
    theta = frozenset(theta)
    for sp in order:
        body = sp.body
        pt = port_types(sp)
        seed = _coin_role(sp) == _SEED
        got = joined.pop(id(sp), None)
        env: dict[tuple[str, int], Runs] = {}
        exposed = []
        for i, (_name, t) in enumerate(body.params):
            r = got[i] if got is not None and got[i] is not None else Runs.uniform(t.leaves, GATE)
            exposed.append(r)
            env[("p", i)] = _receive(r, pt.params[i], seed) if pt is not None else r
        params_out[id(sp)] = (sp, tuple(exposed))
        sinks = any(r.has_sink for r in env.values())
        coins = any(_has_coin(r) for r in env.values())
        for k, node in enumerate(body.nodes):
            fn = node.fn
            prim = isinstance(fn, PrimitiveDefinition)
            if prim and is_input(fn):
                env[("n", k)] = _input_facts(sp, k, node, inputs.get(k, []), harness_, sp is root and k in theta)
                coins = coins or _has_coin(env[("n", k)])
                continue
            fpt = port_types(fn)
            fparams, fret = fn.signature()
            transfer = prim and fpt is not None and fpt.transfer is not None and node.form != "scan"
            need = not prim or sinks or coins or transfer or (fpt is not None and any(fpt.params))
            ops: list[Runs] = []
            if need:
                if node.form == "scan":
                    init = gather(node.args[0].refs, env)
                    carry = _slice(_declared_out(fn, outs), 0, len(init))
                    ops.append(join_runs(init, carry) if node.n else init)
                    for x in node.args[1:1 + node.n_xs]:
                        ops.append(_members(gather(x.refs, env), x.type, 0, node.n))
                    ops.extend(gather(s.refs, env) for s in node.args[1 + node.n_xs:])
                elif node.form == "batch":
                    for a, ax in zip(node.args, node.axes):
                        F = gather(a.refs, env)
                        ops.append(F if ax is None else _members(F, a.type, ax, node.n))
                else:
                    ops = [gather(a.refs, env) for a in node.args]

                def at(sp=sp, k=k, fn=fn, node=node) -> str:
                    return f"{sp.id} node {k} ({node.form} {fn.id})"

                for j, (pname, _pt) in enumerate(fparams):
                    _check_operand(at, pname, fpt.params[j] if fpt is not None else (), ops[j], prim)
                if not prim:
                    slot = joined.setdefault(id(fn), [None] * len(fparams))
                    for j, F in enumerate(ops):
                        slot[j] = join_runs(slot[j], F)
            out = _declared_out(fn, outs)
            if transfer:
                out = _transferred(fn, fpt, fparams, fret, ops, out)
            if node.form == "batch":
                out = _repeat(out, node.n)
            elif node.form == "scan":
                c = fret.fields[0].leaves
                carry = (gather(node.args[0].refs, env) if not node.n else _slice(out, 0, c))
                out = _cat(carry, _repeat(_slice(out, c, len(out)), node.n))
            env[("n", k)] = out
            sinks = sinks or out.has_sink
            coins = coins or _has_coin(out)
        R = gather(body.ret, env)
        _check_ret(sp, pt, R, sp is root)
        if sp is root:
            root_env, root_ret = env, R
    if any(p.kind == "coin" for _k, _lo, _hi, p, _w in (rpt.inputs if rpt is not None else ())):
        _Uses(root, rpt).check()
    return Facts(root, typed, root_env, root_ret, params_out)


# --- one use and seeds ---------------------------------------------------------------------------------------------------------

_NOISE, _FOLD, _SEED, _OTHER = 0, 1, 2, 3
_ROLES = (_NOISE, _FOLD, _SEED, _OTHER)


def _coin_role(fn: Any) -> int | None:
    """`_SEED` for a seed (a primitive or a composite with a coin port and a position port, a return without sinks), `_NOISE`
    for a noise primitive (a coin port, a return not all sinks), `_FOLD` for a receipt fold (a coin port, a return of sinks),
    None for anything else."""
    from .defs import PrimitiveDefinition

    pt = port_types(fn)
    kinds = {p.kind for rs in pt.params for _lo, _hi, p, _w in rs} if pt is not None else ()
    if "coin" not in kinds:
        return None
    sinks = [(lo, hi) for lo, hi, o, _w in pt.ret if o.sink]
    if "position" in kinds and not sinks:
        return _SEED
    if not isinstance(fn, PrimitiveDefinition):
        return None
    _, ret = fn.signature()
    return _FOLD if _covered(sinks, 0, ret.leaves) is None else _NOISE


def _spaces(r: Refs, out: set) -> None:
    if isinstance(r, Concat):
        for p in r.parts:
            _spaces(p, out)
    elif isinstance(r, Explicit):
        out.update((s, i) for s, i, _leaf in r.items)
    elif r.count:
        out.add((r.space, r.idx))  # type: ignore[attr-defined]


def _push(out: list, a: int, b: int, src: tuple, off: int) -> None:
    """Append segment (a, b, src, off) to `out`, joined to the last one when it continues it."""
    if out:
        pa, pb, ps, po = out[-1]
        if pb == a and ps == src and po + pb - pa == off:
            out[-1] = (pa, b, ps, po)
            return
    out.append((a, b, src, off))


class _Map:
    """A collection's tracked leaves: segments (a, b, src, off), ascending and disjoint, its leaves [a, b) being leaves
    [off, off + b - a) of source `src`."""

    __slots__ = ("segs", "starts")

    def __init__(self, segs: list) -> None:
        self.segs, self.starts = segs, [s[0] for s in segs]

    def over(self, lo: int, hi: int) -> Iterator[tuple[int, int, tuple, int]]:
        """(u, w, src, off) for the tracked leaves [u, w) within [lo, hi), leaf u being leaf off of `src`."""
        segs = self.segs
        j = max(bisect.bisect_right(self.starts, lo) - 1, 0)
        while j < len(segs):
            a, b, src, off = segs[j]
            if a >= hi:
                return
            if b > lo:
                u = max(a, lo)
                yield u, min(b, hi), src, off + u - a
            j += 1

    def cut(self, lo: int, hi: int, shift: int, out: list) -> None:
        """Append the segments over leaves [lo, hi) to `out`, leaf x at x + shift."""
        for u, w, src, off in self.over(lo, hi):
            _push(out, u + shift, w + shift, src, off)

    def pattern(self) -> tuple:
        """The tracked leaves, as maximal intervals."""
        out: list = []
        for a, b, _s, _o in self.segs:
            if out and out[-1][1] == a:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
        return tuple(out)


def _merge(ivs: Iterator | Sequence) -> tuple:
    out: list = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return tuple(out)


def _folded(T: _Map, L: int) -> tuple:
    """The leaves some member has tracked, `T` split into consecutive members of `L` leaves (a batch's axis 0, a scan's xs)."""
    ivs = []
    for a, b, _s, _o in T.segs:
        if b - a >= L:
            return ((0, L),)
        m = a // L * L
        ivs += [(a - m, b - m)] if b - m <= L else [(a - m, L), (0, b - m - L)]
    return _merge(ivs)


def _touched(T: _Map, L: int) -> list[int]:
    """The members of `L` leaves holding a tracked leaf of `T`, ascending."""
    out: list[int] = []
    for a, b, _s, _o in T.segs:
        out.extend(range(max(a // L, out[-1] + 1 if out else 0), (b - 1) // L + 1))
    return out


# A read state: how often a noise primitive, a receipt fold, a seed and any other primitive read a leaf, and, when the check
# explains a refusal, the first two readers of the leaf and of each role. A reader is its path of frames (Definition, node,
# member or step), outermost first; a frame with no node is the root's outputs.

def _one(role: int, explain: bool) -> tuple:
    counts = tuple(int(r == role) for r in _ROLES)
    return counts + (((),),) + tuple(((),) if r == role else () for r in _ROLES) if explain else counts


def _plus(x: tuple, y: tuple) -> tuple:
    """Reads `x`, then reads `y`."""
    counts = (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3])
    return counts if len(x) == 4 else counts + tuple((p + q)[:2] for p, q in zip(x[4:], y[4:]))


def _framed(x: tuple, frame: tuple) -> tuple:
    """A callee's reads `x`, read through `frame`."""
    return x if len(x) == 4 else x[:4] + tuple(tuple((frame,) + r for r in rs) for rs in x[4:])


def _total(x: tuple) -> int:
    return x[0] + x[1] + x[2] + x[3]


def _frame(f: tuple) -> str:
    sp, k, idx = f
    if k is None:
        return f"{sp.id}'s outputs"
    node = sp.body.nodes[k]
    at = f"{sp.id} node {k} ({node.form} {node.fn.id})"
    return at if idx is None else f"{at} {'member' if node.form == 'batch' else 'step'} {idx}"


def _who(readers: Sequence[tuple]) -> list[str]:
    return [": ".join(_frame(f) for f in r) for r in readers]


def _made(rec: tuple | None, leaf: int) -> tuple[str, int]:
    """Where leaf `leaf` of a noise source with record `rec` (`_Uses._nids`) is made: the frames from the body holding the
    record down to the noise primitive's instance, and that instance's return leaf."""
    frames = []
    while rec is not None:
        sp, k, unit, rec = rec
        frames.append((sp, k, leaf // unit if sp.body.nodes[k].form in ("batch", "scan") else None))
        leaf %= unit
    return ": ".join(_frame(f) for f in frames), leaf


class _Acc:
    """How each leaf of one source is read: runs [lo[j], hi[j]) of read state st[j], ascending and disjoint. A read added
    later comes after those before it (`_plus`), so the first readers stay first."""

    __slots__ = ("lo", "hi", "st")

    def __init__(self) -> None:
        self.lo: list[int] = []
        self.hi: list[int] = []
        self.st: list[tuple] = []

    def add(self, a: int, b: int, x: tuple) -> None:
        lo, hi, st = self.lo, self.hi, self.st
        if not lo or a >= hi[-1]:
            if lo and a == hi[-1] and st[-1] == x:
                hi[-1] = b
            else:
                lo.append(a)
                hi.append(b)
                st.append(x)
            return
        i = bisect.bisect_right(hi, a)
        j = bisect.bisect_left(lo, b, i)
        pieces = [(lo[i], a, st[i])] if i < j and lo[i] < a else []
        pos = a
        for t in range(i, j):
            if pos < lo[t]:
                pieces.append((pos, lo[t], x))
            pos = min(hi[t], b)
            pieces.append((max(lo[t], a), pos, _plus(st[t], x)))
            if hi[t] > b:
                pieces.append((b, hi[t], st[t]))
        if pos < b:
            pieces.append((pos, b, x))
        if i and hi[i - 1] == pieces[0][0] and st[i - 1] == pieces[0][2]:
            i -= 1
            pieces[0] = (lo[i], pieces[0][1], st[i])
        if j < len(lo) and lo[j] == pieces[-1][1] and st[j] == pieces[-1][2]:
            pieces[-1] = (pieces[-1][0], hi[j], st[j])
            j += 1
        out: list = []
        for p in pieces:
            if out and out[-1][1] == p[0] and out[-1][2] == p[2]:
                out[-1] = (out[-1][0], p[1], p[2])
            else:
                out.append(p)
        lo[i:j], hi[i:j], st[i:j] = [p[0] for p in out], [p[1] for p in out], [p[2] for p in out]

    def runs(self) -> list[tuple[int, int, tuple]]:
        return list(zip(self.lo, self.hi, self.st))


class _Summary:
    """The one-use view of every activation of a Definition whose parameters' tracked leaves are one pattern. `reads[i]`:
    runs (lo, hi, state) of parameter i's leaves its body reads. `ret`: segments (a, b, src, off) of its return's tracked
    leaves, return leaves [a, b) being leaves [off, off + b - a) of ``("p", i)``, a parameter passed through, or of
    ``("c", e)``, noise made inside. `noise[e]`: that noise's record (`_Uses._nids`; None for a noise primitive's own), its
    leaves, and runs of the reads its leaves already had inside."""

    __slots__ = ("reads", "ret", "noise")

    def __init__(self, reads: tuple, ret: tuple, noise: tuple) -> None:
        self.reads, self.ret, self.noise = reads, ret, noise


class _Recount(Exception):
    """A refusal found while counting, which the check counts again with readers to name them."""


class _Uses:
    """The one-use and seed rules over the flattened Program (every call, batch member and scan step counted), at the size of
    its description. A tracked leaf is a root coin's (source ``("coin", node)``), a root position's (``("pos", node)``), a
    parameter's (``("p", i)``) or noise made in the body (``("c", n)``); it is followed as runs of leaves through the
    references (`_Map`), and its reads are counted by role in runs (`_Acc`). A Definition is simulated once per pattern of
    tracked parameter leaves (`_Summary`), its internal reads checked there; what crosses its boundary is a count per tracked
    parameter leaf, the parameter leaves and noise its return carries, and how often that noise was read inside. A primitive
    or a seed is its own summary. A batch of n members applies one summary to its members, at once where they read alike (a
    shared argument n times, or every leaf of their own), else member by member, never simulating again; a scan applies it
    step by step, or with no tracked carry as a batch of its steps. Counting keeps no readers: a refusal is counted again
    with them (`explain`), and its message names its first two readers by Definition, node, member and step."""

    def __init__(self, root: Any, rpt: PortTypes, explain: bool = False) -> None:
        self.root, self.rpt, self.explain = root, rpt, explain
        self.memo: dict[tuple, tuple[Any, _Summary]] = {}
        self.roles: dict[int, tuple[Any, int | None]] = {}
        self.one = tuple(_one(r, explain) for r in _ROLES)
        self.simulations = 0

    def _role(self, fn: Any) -> int | None:
        hit = self.roles.get(id(fn))
        if hit is None or hit[0] is not fn:
            hit = self.roles[id(fn)] = (fn, _coin_role(fn))
        return hit[1]

    def check(self) -> None:
        try:
            self._simulate(self.root, ())
        except _Recount as r:
            _Uses(self.root, self.rpt, explain=True).check()
            raise PortError("E_COIN_REUSE", str(r)) from None

    def _refuse(self, message: str) -> None:
        if self.explain:
            raise PortError("E_COIN_REUSE", message)
        raise _Recount(message)

    def _readers(self, x: tuple, rule: str) -> str:
        """The first two reads of a leaf with reads `x` that `rule` refuses, one of them of the role it refuses."""
        if not self.explain:
            return "readers not recorded"
        first, roles = x[4], x[5:]
        if rule == "noise-coin":
            noisy = (roles[_NOISE] or roles[_OTHER])[0]
            pair = first if noisy in first else (first[0], noisy)
        elif rule == "seed-fold":
            pair = (first[0], roles[_FOLD][0] if first[0] in roles[_SEED] else roles[_SEED][0])
        else:
            pair = roles[_SEED] if rule == "position" else first
        a, b = _who(pair[:2])
        return f"first by {a}; then by {b}"

    def _root_rules(self, src: tuple, A: _Acc) -> None:
        kind, k = src
        for lo, _hi, x in A.runs():
            at = f"{self.root.id} input node {k} leaf {lo}"
            if kind == "pos":
                if x[_SEED] > 1:
                    self._refuse(f"{at}: a position seeds one unit, but {x[_SEED]} seeds read it "
                                 f"({self._readers(x, 'position')}): two seeds of one coin at one position are one seed, so "
                                 f"the noise they derive is common to both. Give each seed a position of its own")
                continue
            if (x[_NOISE] or x[_OTHER]) and _total(x) > 1:
                self._refuse(f"{at}: a coin that a noise primitive reads is read once, by that instance alone, but it is "
                             f"read {_total(x)} times ({self._readers(x, 'noise-coin')}): a second noise or a receipt key "
                             f"from a noise bit's coin puts the bit in the cheater's view. Draw a coin of its own for each "
                             f"noise and for the receipt keys")
            if x[_SEED] and x[_FOLD]:
                self._refuse(f"{at}: a coin that a seed reads is read by seeds alone, but a receipt fold reads it too "
                             f"({self._readers(x, 'seed-fold')}): only a seed binds the coin to a unit's position, so a "
                             f"fold keyed by it computes a value no position separates. Key the folds from the unit's seed, "
                             f"or from a coin of their own")

    def _noise_rule(self, rec: tuple, A: _Acc) -> None:
        for lo, _hi, x in A.runs():
            if _total(x) > 1:
                where, leaf = _made(rec, lo)
                self._refuse(f"{where} return leaf {leaf}: noise is the noise of one operand of one instance and is read "
                             f"at most once, but it is read {_total(x)} times ({self._readers(x, 'noise')}): two copies of "
                             f"one noise cancel, and noise shared across rows, matmuls, batch members or scan steps is "
                             f"fresh for none of them. Draw fresh coins for each use")

    def _gather(self, refs: Refs, env: dict) -> _Map | None:
        spaces: set = set()
        _spaces(refs, spaces)
        if not any(s in env for s in spaces):
            return None
        out: list = []
        pos = 0
        for s, i, lo, hi in ref_runs(refs):
            T = env.get((s, i))
            if T is not None:
                T.cut(lo, hi, pos - lo, out)
            pos += hi - lo
        return _Map(out) if out else None

    @staticmethod
    def _add(acc: dict, src: tuple, lo: int, hi: int, x: tuple) -> None:
        A = acc.get(src)
        if A is None:
            A = acc[src] = _Acc()
        A.add(lo, hi, x)

    def _read(self, acc: dict, T: _Map, lo: int, hi: int, x: tuple) -> None:
        """Reads `x` of leaves [lo, hi) of a collection whose tracked leaves are `T`, counted on their sources."""
        for u, w, src, off in T.over(lo, hi):
            self._add(acc, src, off, off + w - u, x)

    def _nids(self, sp: Any, k: int, S: _Summary, n: int, noise: dict) -> list[tuple[tuple, int]]:
        """Sources for the noise each of node k's `n` instances of `S` returns: one per noise of `S`, instance m's leaves at
        m * size; its record (sp, k, size, the noise's record inside) names where a leaf of it is made (`_made`)."""
        out = []
        for rec, size, _runs in S.noise:
            src = ("c", len(noise))
            noise[src] = ((sp, k, size, rec), n * size)
            out.append((src, size))
        return out

    def _summary(self, fn: Any, pattern: tuple) -> _Summary:
        """`fn`'s summary for tracked parameter leaves `pattern` (per parameter, its tracked leaves as intervals)."""
        hit = self.memo.get((id(fn), pattern))
        if hit is not None and hit[0] is fn:
            return hit[1]
        from .defs import PrimitiveDefinition

        role = self._role(fn)
        if role == _SEED or isinstance(fn, PrimitiveDefinition):
            x = self.one[_OTHER if role is None else role]
            reads = tuple(tuple((a, b, x) for a, b in iv) for iv in pattern)
            if role == _NOISE:
                R = fn.signature()[1].leaves
                S = _Summary(reads, ((0, R, ("c", 0), 0),), ((None, R, ()),))
            else:
                S = _Summary(reads, (), ())
        else:
            S = self._simulate(fn, pattern)
        self.memo[(id(fn), pattern)] = (fn, S)
        return S

    def _member(self, sp: Any, k: int, idx: int | None, m: int, S: _Summary, views: list, nids: list, acc: dict,
                outs: tuple) -> None:
        """Instance m of node k (member or step `idx`), its parameter i's tracked leaves `views[i]` (a map and the offset of
        the instance's leaf 0 in it, or None): count its reads and the reads its noise had inside, and put its return
        leaves [lo, hi) at lo + shift of `out`, for each (lo, hi, out, shift) of `outs`."""
        frame = (sp, k, idx)
        for v, rs in zip(views, S.reads):
            if v is not None:
                T, d = v
                for lo, hi, x in rs:
                    self._read(acc, T, lo + d, hi + d, _framed(x, frame))
        for (src, size), (_rec, _size, runs) in zip(nids, S.noise):
            for lo, hi, x in runs:
                self._add(acc, src, m * size + lo, m * size + hi, _framed(x, frame))
        for a, b, src, o in S.ret:
            for lo, hi, out, shift in outs:
                u, w = max(a, lo), min(b, hi)
                if u >= w:
                    continue
                p = o + u - a
                if src[0] == "c":
                    made, size = nids[src[1]]
                    _push(out, u + shift, w + shift, made, m * size + p)
                elif views[src[1]] is not None:
                    T, d = views[src[1]]
                    T.cut(p + d, p + d + w - u, u + shift - p - d, out)

    def _batch(self, sp: Any, k: int, node: Any, tracked: list, acc: dict, noise: dict, out: list) -> None:
        n = node.n
        if not n:
            return
        lay: list = []
        pattern: list = []
        for T, a, ax in zip(tracked, node.args, node.axes):
            if T is None:
                lay.append(None)
                pattern.append(())
            elif ax is None:
                lay.append((T, None, 0, None))
                pattern.append(T.pattern())
            elif ax == 0:
                L = a.type.leaves // n
                lay.append((T, 0, L, None))
                pattern.append(_folded(T, L))
            else:
                M = _split(T, a.type, ax, n)
                lay.append((T, ax, 0, M))
                pattern.append(_merge([iv for U in M.values() for iv in U.pattern()]))
        S = self._summary(node.fn, tuple(pattern))
        R = node.fn.signature()[1].leaves
        self._members(sp, k, S, lay, n, R, 0, acc, noise, out)

    def _members(self, sp: Any, k: int, S: _Summary, lay: list, n: int, stride: int, lo: int, acc: dict, noise: dict,
                 out: list) -> None:
        """Node k's `n` instances of `S`, parameter i laid out by `lay[i]`: None (untracked), (T, None, 0, None) (each
        instance reads T), (T, 0, L, None) (instance m reads leaves [m L, (m + 1) L) of T) or (T, ax, 0, {m: map}); instance
        m's return leaves [lo, lo + stride) go to lo + m stride of `out`. Instances whose parameters are untracked are
        skipped; where every instance reads alike, the counts are added for all of them at once."""
        nids = self._nids(sp, k, S, n, noise)
        if any(v is not None and v[1] is None for v in lay):
            members: Sequence[int] = range(n)
        else:
            got: set = set()
            for v in lay:
                if v is not None:
                    got.update(_touched(v[0], v[2]) if v[1] == 0 else v[3])
            members = sorted(got)
        if not self.explain and (len(members) == n or not S.noise) and self._alike(S, lay, stride, lo):
            for v, rs in zip(lay, S.reads):
                for a, b, x in rs if v is not None else ():
                    if v[1] is None:
                        self._read(acc, v[0], a, b, tuple(c * n for c in x))
                    else:
                        self._read(acc, v[0], 0, n * v[2], x)
            for (src, size), (_rec, _size, runs) in zip(nids, S.noise):
                for _a, _b, x in runs:
                    self._add(acc, src, 0, n * size, x)
            for _a, _b, src, _o in S.ret:
                _push(out, lo, lo + n * stride, nids[src[1]][0], 0)
            return
        for m in members:
            views = [None if v is None else (v[0], 0) if v[1] is None else (v[0], m * v[2]) if v[1] == 0
                     else (v[3][m], 0) if m in v[3] else None for v in lay]
            self._member(sp, k, m, m, S, views, nids, acc, ((lo, lo + stride, out, m * stride),))

    @staticmethod
    def _alike(S: _Summary, lay: list, stride: int, lo: int) -> bool:
        """Whether every instance reads and returns alike: each shared argument read whole or not at all by each, each split
        one read along axis 0 whole and uniformly, its noise read inside whole and uniformly, and its return nothing tracked
        or exactly one noise of `stride` leaves."""
        for v, rs in zip(lay, S.reads):
            if v is None or not rs or v[1] is None:
                continue
            if v[1] != 0 or len(rs) != 1 or rs[0][:2] != (0, v[2]):
                return False
        if any(runs and (len(runs) != 1 or runs[0][:2] != (0, size)) for _rec, size, runs in S.noise):
            return False
        if not S.ret:
            return True
        if len(S.ret) != 1:
            return False
        a, b, src, o = S.ret[0]
        return src[0] == "c" and (a, b, o) == (lo, lo + stride, 0) and S.noise[src[1]][1] == stride

    def _scan(self, sp: Any, k: int, node: Any, tracked: list, acc: dict, noise: dict, out: list) -> None:
        n, nx = node.n, node.n_xs
        fret = node.fn.signature()[1]
        R, c = fret.leaves, fret.fields[0].leaves
        init = tracked[0]
        if not n:
            if init is not None:
                init.cut(0, c, 0, out)
            return
        xs = [None if T is None else (T, a.type.leaves // n) for T, a in zip(tracked[1:1 + nx], node.args[1:1 + nx])]
        shared = tracked[1 + nx:]
        rest = tuple(() if v is None else _folded(*v) for v in xs) + tuple(() if T is None else T.pattern() for T in shared)
        pc = () if init is None else init.pattern()
        while True:
            S = self._summary(node.fn, (pc,) + rest)
            grown = _merge(pc + tuple((a, min(b, c)) for a, b, _s, _o in S.ret if a < c))
            if grown == pc:
                break
            pc = grown
        if not pc:
            lay = ([None] + [None if v is None else (v[0], 0, v[1], None) for v in xs]
                   + [None if T is None else (T, None, 0, None) for T in shared])
            self._members(sp, k, S, lay, n, R - c, c, acc, noise, out)
            return
        nids = self._nids(sp, k, S, n, noise)
        live = any(T is not None for T in shared)
        carry, ys = init, []
        for i in range(n):
            views = [None if carry is None else (carry, 0)]
            views += [None if v is None or next(v[0].over(i * v[1], (i + 1) * v[1]), None) is None else (v[0], i * v[1])
                      for v in xs]
            views += [None if T is None else (T, 0) for T in shared]
            if not live and all(v is None for v in views):
                carry = None
                continue
            nxt: list = []
            self._member(sp, k, i, i, S, views, nids, acc, ((0, c, nxt, 0), (c, R, ys, i * (R - c))))
            carry = _Map(nxt) if nxt else None
        if carry is not None:
            carry.cut(0, c, 0, out)
        out.extend(ys)

    def _simulate(self, sp: Any, pattern: tuple) -> _Summary | None:
        """Count the reads in `sp`'s body, its parameter i's tracked leaves `pattern[i]`, refusing what breaks a rule there;
        its summary, or None for the root."""
        from .program import is_input

        self.simulations += 1
        root = sp is self.root
        env: dict[tuple[str, int], _Map] = {("p", i): _Map([(a, b, ("p", i), a) for a, b in iv])
                                            for i, iv in enumerate(pattern) if iv}
        if root:
            segs: dict[int, list] = {}
            for k, lo, hi, p, _w in self.rpt.inputs:
                if p.kind in ("coin", "position"):
                    segs.setdefault(k, []).append((lo, hi, ("coin" if p.kind == "coin" else "pos", k), lo))
            for k, ss in segs.items():
                joined: list = []
                for s in sorted(ss):
                    _push(joined, *s)
                env[("n", k)] = _Map(joined)
        acc: dict[tuple, _Acc] = {}
        noise: dict[tuple, tuple] = {}
        for k, node in enumerate(sp.body.nodes):
            if is_input(node.fn):
                continue
            tracked = [self._gather(a.refs, env) for a in node.args]
            if all(T is None for T in tracked):
                continue
            out: list = []
            if node.form == "batch":
                self._batch(sp, k, node, tracked, acc, noise, out)
            elif node.form == "scan":
                self._scan(sp, k, node, tracked, acc, noise, out)
            else:
                S = self._summary(node.fn, tuple(() if T is None else T.pattern() for T in tracked))
                views = [None if T is None else (T, 0) for T in tracked]
                R = node.fn.signature()[1].leaves
                self._member(sp, k, None, 0, S, views, self._nids(sp, k, S, 1, noise), acc, ((0, R, out, 0),))
            if out:
                env[("n", k)] = _Map(out)
        ret = self._gather(sp.body.ret, env)
        if root and ret is not None:
            x = _framed(self.one[_OTHER], (sp, None, None))
            for a, b, src, off in ret.segs:
                self._add(acc, src, off, off + b - a, x)
        for src, (rec, _size) in noise.items():
            if src in acc:
                self._noise_rule(rec, acc[src])
        if root:
            for src in sorted((s for s in acc if s[0] in ("coin", "pos")), key=lambda s: (s[1], s[0])):
                self._root_rules(src, acc[src])
            return None
        reads = tuple(tuple(acc[("p", i)].runs()) if ("p", i) in acc else () for i in range(len(pattern)))
        segs_out: list = []
        renum: dict[tuple, int] = {}
        kept: list = []
        for a, b, src, off in ret.segs if ret is not None else ():
            if src[0] == "c":
                e = renum.get(src)
                if e is None:
                    e = renum[src] = len(kept)
                    rec, size = noise[src]
                    kept.append((rec, size, tuple(acc[src].runs()) if src in acc else ()))
                src = ("c", e)
            segs_out.append((a, b, src, off))
        return _Summary(reads, tuple(segs_out), tuple(kept))


def _split(T: _Map, t: Type, ax: int, n: int) -> dict[int, _Map]:
    """The tracked leaves `T` of a collection of type `t`, split into its `n` members along axis `ax` (members with none
    left out)."""
    pos = Coll(t, Affine("a", 0, 0, t.leaves))
    out: dict[int, _Map] = {}
    for m in range(n):
        segs: list = []
        q = 0
        for _s, _i, lo, hi in ref_runs(pos.take(ax, m).refs):
            T.cut(lo, hi, q - lo, segs)
            q += hi - lo
        if segs:
            out[m] = _Map(segs)
    return out


def check(root: Any) -> Facts | None:
    """Hold the Program rooted at `root` to the port rules, refusing it with a `PortError`; its facts, or None when no
    Definition in it declares ports (then nothing is refused, and nothing is computed)."""
    order, typed = _reachable(root)
    if not typed:
        return None
    return _analyse(root, order, typed)


def facts(root: Any) -> Facts:
    """Every leaf's facts in the Program rooted at `root` (a `Program`'s or a `DecodedProgram`'s `root`), checking the rules
    as `check` does."""
    order, typed = _reachable(root)
    return _analyse(root, order, typed)


__all__ = ["In", "Out", "Ports", "Leaves", "PortTypes", "PortError", "Fact", "Facts", "Runs", "KINDS", "ORIGINS", "MIXED",
           "GATE", "as_reading", "bind_ports", "check", "declared", "fact", "facts", "forward", "full", "gather", "harness",
           "holds", "join", "join_runs", "port_types", "reading", "root_ports"]
