"""What `query.word` and `query.cross_call` share about one Call's circuit: wiring (`WIRING`), the Definition a spec names
(`specialization`), a Definition as a one-Call Program (`_call_scope`), a batch member's references (`_member_refs`), and the
violation both raise (`QueryRuleViolation`)."""
from __future__ import annotations

import json
from typing import Any

#: primitives that only rearrange bits (each output bit a constant or a copy of one input bit; `tests/query/test_word.py` checks it):
#: wiring, like constants, is computed in no unit, may be read by any, and is never committed
WIRING = frozenset({"Bf16ToF32_v1", "F32BitsShl23_v1", "F32Fabs_v1"})


class QueryRuleViolation(ValueError):
    """A Program whose units cannot satisfy `Q_word_v1{X,W}`: named per Definition and module, never widened."""


def specialization(fam: str, statics: dict[str, Any]):
    """The registered Definition `fam` bound to `statics` (a primitive as it is); function-valued statics (`DOT`, rendered
    `{"fn": id}` or as the bare id) resolve to the registered Definition they name."""
    from verity.ir.codec import lookup_primitive
    from verity_vllm.query.program_view import _registry
    reg, _errors = _registry()
    d = reg.defs.get(fam)
    if d is None:
        return lookup_primitive(reg, fam)

    def fn_value(v: Any) -> Any:
        fid = v.get("fn") if isinstance(v, dict) else (json.loads(v).get("fn") if isinstance(v, str) and v[:5] == '{"fn"' else v)
        if not isinstance(fid, str):
            return v
        try:
            return lookup_primitive(reg, fid)
        except KeyError:
            return reg.defs[fid] if fid in reg.defs else v

    return d.bind(**{k: fn_value(v) for k, v in statics.items()}) if hasattr(d, "bind") else d


def _call_scope(fn: Any) -> tuple[Any, Any]:
    """(circuit, activation of `fn`) in a one-Call Program: the Call's parameters are Input gates, its returned leaves designated
    outputs.  A returned leaf a real Program does not read only shrinks out(S), so widths here bound those in any Program."""
    from verity.ir.defs import CompositeDefinition
    from verity.ir.program import Program
    params, ret = fn.signature()
    shell = CompositeDefinition(f"Word[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *a: B.call(fn, *a), register=False)
    c = Program(shell).circuit
    rs = c.root_scope
    k = next(k for k, n in enumerate(rs.spec.body.nodes) if n.fn is fn)
    return c, rs.child(k)


def _member_refs(a: Any, ax: int | None, m: int) -> Any:
    if ax is None:
        return a.refs
    if ax == 0:
        L = a.type.elem.leaves
        return a.refs.slice(m * L, (m + 1) * L)
    return a.take(ax, m).refs


__all__ = ["WIRING", "QueryRuleViolation", "specialization"]
