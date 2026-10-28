"""Shape schemas in an authored function's annotations (`Arr["N", "K", BF16]`, `Tup[...]`, `Dep(...)`, a `Value<w>`), and
their unification with argument types, which solves the statics a shape determines."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable

from verity.ml.circuits.types import Array, Tuple as TupleT, Type, Value
from verity_vllm.program.frontend.errors import FrontendError

@dataclass(frozen=True)
class _ArrSchema:
    dims: tuple[int | str, ...]
    width: int


@dataclass(frozen=True)
class _TupSchema:
    fields: tuple[Any, ...]


class Arr:
    """`Arr["K", BF16]`, `Arr["N", "K", BF16]`, `Arr[16, BF16]`, `Arr["2*I", BF16]`: dims are ints, static names
    (solved from the argument shape) or `<int>*NAME` / expressions over statics (checked)."""

    def __class_getitem__(cls, item):
        items = item if isinstance(item, tuple) else (item,)
        *dims, last = items
        if isinstance(last, Value):
            return _ArrSchema(tuple(dims), last.w)
        if isinstance(last, _ArrSchema):
            return _ArrSchema(tuple(dims) + last.dims, last.width)
        raise TypeError("Arr[...] must end in a Value<w> (e.g. BF16) or a nested Arr")


class Tup:
    def __class_getitem__(cls, item):
        items = item if isinstance(item, tuple) else (item,)
        return _TupSchema(tuple(_schema(i) for i in items))


class Dep:
    """`w: Dep(lambda S: layer_weights_type(S["C"]))`: a parameter (or return) whose *type* is a function of the
    statics — records and other structured collections a static configuration determines. The argument is
    checked against the computed type once the statics are known; `specialize(**statics)` computes it."""

    def __init__(self, fn: Callable[[dict], Type]) -> None:
        self.fn = fn

    def __repr__(self) -> str:
        return f"Dep({getattr(self.fn, '__name__', 'λ')})"


def _schema(a):
    if a is inspect.Parameter.empty:
        return None
    if isinstance(a, Value):
        return _ArrSchema((), a.w)
    if isinstance(a, (_ArrSchema, _TupSchema, Dep)):
        return a
    raise FrontendError("E_SHAPE", f"unsupported annotation {a!r} (use Arr[...], Tup[...], Dep(...) or a Value<w>)")


_LIN = re.compile(r"^(\d+)\*([A-Za-z_]\w*)$")


def _unify(sch, T: Type, statics: dict, deferred: list, where: str) -> None:
    if sch is None:
        return
    if isinstance(sch, Dep):
        deferred.append((sch, T))  # checked once every static is known
        return
    if isinstance(sch, _TupSchema):
        if not isinstance(T, TupleT) or len(T.fields) != len(sch.fields):
            raise FrontendError("E_SHAPE", f"expected a {len(sch.fields)}-tuple, got {T!r}", where)
        for s, f in zip(sch.fields, T.fields):
            _unify(s, f, statics, deferred, where)
        return
    for d in sch.dims:
        if not isinstance(T, Array):
            raise FrontendError("E_SHAPE", f"schema {sch} needs another array axis, got {T!r}", where)
        if isinstance(d, int):
            if T.n != d:
                raise FrontendError("E_SHAPE", f"axis {d} expected, got {T.n}", where)
        elif d.isidentifier():
            if d in statics:
                if statics[d] != T.n:
                    raise FrontendError("E_SHAPE", f"static {d}={statics[d]} but the argument has axis {T.n}", where)
            else:
                statics[d] = T.n
        else:
            m = _LIN.match(d.replace(" ", ""))
            if m and m.group(2) not in statics:
                a, name = int(m.group(1)), m.group(2)
                if T.n % a:
                    raise FrontendError("E_SHAPE", f"axis {T.n} is not a multiple of {a} ({d})", where)
                statics[name] = T.n // a
            else:
                deferred.append((d, T.n))
        T = T.elem
    if not isinstance(T, Value) or T.w != sch.width:
        code = "E_WIDTH" if isinstance(T, Value) else "E_SHAPE"
        raise FrontendError(code, f"expected Value<{sch.width}>, got {T!r}", where)


def _check_deferred(deferred: list, statics: dict, where: str) -> None:
    """Dimension expressions and `Dep` types that could only be checked once every static was known."""
    for expr, n in deferred:
        if isinstance(expr, Dep):
            want = _dep_type(expr, statics, where)
            if want != n:
                raise FrontendError("E_SHAPE", f"{expr!r} computes {want!r} but the argument has type {n!r}", where)
        elif _eval_dim(expr, statics, where) != n:
            raise FrontendError("E_SHAPE", f"dimension {expr!r} = {_eval_dim(expr, statics, where)} but the argument has {n}", where)


def _dep_type(sch: Dep, statics: dict, where: str) -> Type:
    try:
        t = sch.fn(dict(statics))
    except Exception as e:  # noqa: BLE001
        raise FrontendError("E_STATIC", f"{sch!r} failed on statics {sorted(statics)}: {type(e).__name__}: {e}", where) from None
    if not isinstance(t, Type):
        raise FrontendError("E_STATIC", f"{sch!r} did not return a Type", where)
    return t


def _eval_dim(expr: str, statics: dict, where: str) -> int:
    try:
        v = eval(expr, {"__builtins__": {}}, {k: v for k, v in statics.items() if isinstance(v, int)})
    except Exception as e:  # noqa: BLE001
        raise FrontendError("E_STATIC", f"cannot evaluate dimension {expr!r}: {e}", where) from None
    if not isinstance(v, int):
        raise FrontendError("E_STATIC", f"dimension {expr!r} is not an int", where)
    return v


def _type_of_schema(sch, statics: dict, where: str) -> Type:
    if isinstance(sch, Dep):
        return _dep_type(sch, statics, where)
    if isinstance(sch, _TupSchema):
        return TupleT(*[_type_of_schema(f, statics, where) for f in sch.fields])
    T: Type = Value(sch.width)
    for d in reversed(sch.dims):
        n = d if isinstance(d, int) else _eval_dim(d, statics, where)
        T = Array(n, T)
    return T
