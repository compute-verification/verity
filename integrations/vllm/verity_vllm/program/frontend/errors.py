"""The torch frontend's fail-closed contract: every refusal is a `FrontendError` with one of `CODES`."""

from __future__ import annotations

CODES = {
    "E_EXPORT": "torch.export failed or the pre-pass raised (message attached)",
    "E_UNREGISTERED_OP": "a node has no Verity translation: ATen op, foreign custom op, Python callable",
    "E_MUTATION": "an in-place / mutating op, or an input/buffer mutation recorded by export",
    "E_HIDDEN_STATE": "lifted parameter, buffer or tensor constant not declared as a prescribed input",
    "E_OPAQUE_HOST": "data-dependent host escape (.item(), Python control on values, host callback)",
    "E_DYNAMIC_CONTROL_FLOW": "cond / while_loop / map or another higher-order op; subgraph attributes",
    "E_UNSUPPORTED_VIEW": "a view outside the affine/strided whitelist (inner-dim cat, expand, step != 1, ...)",
    "E_WIDTH": "tensor width does not match the declared Value<w> of the registered function",
    "E_SHAPE": "shape does not conform to the declared schema / static dimensions",
    "E_STATIC": "a static parameter could not be resolved, or a static value is not a finite binding",
    "E_ARITY": "wrong number of arguments for a registered function",
    "E_RETURN_TYPE": "returned pytree does not match the return annotation / scan step contract",
    "E_IDENTITY": "two different bodies claim one fully qualified specialisation (name + version + statics)",
    "E_DEAD_GATES": "the exported program has dead gates and export was asked to be strict about them",
}


class DeadGateWarning(UserWarning):
    """Authoring-mode warning: the exported program has gates no gate or root output consumes (§8.6)."""


class FrontendError(Exception):
    def __init__(self, code: str, message: str, where: str | None = None) -> None:
        assert code in CODES, code
        self.code, self.message, self.where = code, message, where
        super().__init__(f"{code}{' @ ' + where if where else ''}: {message}")
