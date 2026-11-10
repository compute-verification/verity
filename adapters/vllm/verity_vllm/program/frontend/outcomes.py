"""The outcomes of a derivation that is refused (design §4.3): `Unsupported` with one of `REASONS`, and `refuse`, which carries it
out of the translation as `Refuse`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NoReturn

import torch

REASONS = ("mutation", "decomposition", "custom_op_without_binding", "side_input", "dynamic_control_flow",
           "numerics_unregistered", "hidden_state", "unsupported_view", "shape", "export", "identity")

_CODE_TO_REASON = {"E_MUTATION": "mutation", "E_HIDDEN_STATE": "hidden_state", "E_DYNAMIC_CONTROL_FLOW": "dynamic_control_flow",
                   "E_UNSUPPORTED_VIEW": "unsupported_view", "E_SHAPE": "shape", "E_WIDTH": "shape", "E_EXPORT": "export",
                   "E_IDENTITY": "identity", "E_UNREGISTERED_OP": "decomposition", "E_OPAQUE_HOST": "side_input",
                   "E_STATIC": "shape", "E_ARITY": "shape", "E_RETURN_TYPE": "shape", "E_DEAD_GATES": "shape"}


@dataclass
class Unsupported:
    """The refusal outcome (design §4.3). `reason` is one of REASONS; `node` the graph node / op; `module_path`
    the `nn_module_stack` hint; `hint` what would make it derivable; `detail` structured evidence."""
    reason: str
    node: str | None = None
    module_path: str | None = None
    hint: str = ""
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reason not in REASONS:
            raise ValueError(f"unknown Unsupported reason {self.reason!r}")

    def to_json(self) -> dict:
        return {"reason": self.reason, "node": self.node, "module_path": self.module_path, "hint": self.hint,
                "detail": _jsonable(self.detail)}

    def __str__(self) -> str:
        where = f" @ {self.module_path}" if self.module_path else ""
        node = f" [{self.node}]" if self.node else ""
        return f"Unsupported({self.reason}){node}{where}: {self.hint}"


class Refuse(Exception):
    """Internal: carries an `Unsupported` out of the translation. `derive` returns it, never raises it."""

    def __init__(self, u: Unsupported) -> None:
        self.u = u
        super().__init__(str(u))


def refuse(reason: str, node=None, module_path: str | None = None, hint: str = "", **detail) -> NoReturn:
    raise Refuse(Unsupported(reason, _node_label(node), module_path, hint, detail))


def _node_label(node) -> str | None:
    if node is None:
        return None
    if isinstance(node, torch.fx.Node):
        return f"{node.name}: {node.target}" if node.op == "call_function" else f"{node.name}: {node.op}"
    return str(node)


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)
