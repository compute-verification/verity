"""What `query.word` and `query.cross_call` share beyond core's `verity.ml.circuits.cut`: the registered Definition a spec names
(`specialization`) and the violation both raise (`QueryRuleViolation`)."""
from __future__ import annotations

import json
from typing import Any

class QueryRuleViolation(ValueError):
    """A Program whose units cannot satisfy `Q_word_v1{X,W}`: named per Definition and module, never widened."""


def specialization(fam: str, statics: dict[str, Any]):
    """The registered Definition `fam` bound to `statics` (a primitive as it is); function-valued statics (`DOT`, rendered
    `{"fn": id}` or as the bare id) resolve to the registered Definition they name."""
    from verity.ml.circuits.codec import lookup_primitive
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


__all__ = ["QueryRuleViolation", "specialization"]
