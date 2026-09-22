"""Torch-free pieces of the compiled-launch capture (CONTRACT §16.1): the launch-argument spec of a
generated Inductor kernel and the row-count model of its integer arguments.

A launch is recorded as an ordered list of *slots*, one per launch argument::

    {"index": i, "name": "out_ptr0", "kind": "out", "port": 0, "dtype": "bfloat16", "row_shape": [576]}
    {"index": i, "name": "in_ptr1", "kind": "tensor", "role": "in"|"weight"|"token"|"positions", ...}
    {"index": i, "name": "xnumel", "kind": "int", "model": {"model": "linear", "a": 576, "b": 0}}
    {"index": i, "name": "HAS_BIAS", "kind": "const", "value": False}

Integer arguments (``xnumel``, ``M``, strides ...) are functions of the batch's row count; they are
observed across the capture's steps and fitted as ``a * rows + b`` (or a constant). The relation
evaluates them at ``rows = 1``. An observation that contradicts the fitted model is a violation
(counted; the node is then not row-independent in this argument and the relation errors).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

OUT_ARG_RE = re.compile(r"^(out_ptr\d+|in_out_ptr\d+)$")
# user-defined (non-generated) Triton kernels that Inductor inlines via triton_kernel_wrapper_mutation:
# their output argument names (Inductor's out_ptr* convention does not apply)
USER_KERNEL_OUTPUTS = {"matmul_kernel_persistent": ["c_ptr"]}
GEMM_FN = {"matmul_kernel_persistent"}


def module_of_param(param: str) -> str:
    """``layers.3.mlp.gate_up_proj.weight`` -> ``layers.3.mlp.gate_up_proj``."""
    for suf in (".weight", ".bias", ".cos_sin_cache"):
        if param.endswith(suf):
            return param[: -len(suf)]
    return param.rsplit(".", 1)[0] if "." in param else param


@dataclass
class IntArg:
    """value = a * rows + b, fitted from (rows, value) observations."""

    name: str
    obs: list[tuple[int, int]] = field(default_factory=list)
    violations: int = 0

    def add(self, rows: int, value: int) -> None:
        m = self.model()
        if m is not None and m.get("model") in ("linear", "const"):
            pred = m["a"] * rows + m["b"] if m["model"] == "linear" else m["value"]
            if pred != value:
                self.violations += 1
        if len(self.obs) < 8 and (rows, value) not in self.obs:
            self.obs.append((rows, value))

    def model(self) -> dict[str, Any] | None:
        if not self.obs:
            return None
        vals = {v for _, v in self.obs}
        if len(vals) == 1:
            if len({r for r, _ in self.obs}) >= 2:
                return {"model": "const", "value": self.obs[0][1]}
            return {"model": "underdetermined", "obs": list(self.obs)}
        (r1, v1) = self.obs[0]
        other = next((o for o in self.obs if o[0] != r1), None)
        if other is None:
            return {"model": "inconsistent", "obs": list(self.obs)}  # same rows, different values
        (r2, v2) = other
        if (v2 - v1) % (r2 - r1):
            return {"model": "underdetermined", "obs": list(self.obs)}
        a = (v2 - v1) // (r2 - r1)
        b = v1 - a * r1
        if any(a * r + b != v for r, v in self.obs):
            return {"model": "inconsistent", "obs": list(self.obs)}
        return {"model": "linear", "a": a, "b": b}


def int_value(slot: dict[str, Any], rows: int) -> int:
    """Evaluate a recorded integer launch argument for ``rows`` rows."""
    m = slot.get("model") or {}
    kind = m.get("model")
    if kind == "linear":
        return int(m["a"]) * rows + int(m["b"])
    if kind == "const":
        return int(m["value"])
    raise ValueError(f"integer launch arg {slot.get('name')!r} has no usable model ({kind}: {m.get('obs')})")


def classify_name(fn_name: str, kernel_name: str, weight_modules: list[str], producer_layers: list[str]) -> tuple[str, str, str | None, str]:
    """(family, op_type, explicit node name or None, module path) for one launch, from the kernel's
    name and the identities of its weight / producer arguments.

    * a GEMM (vLLM's persistent Triton matmul inlined by Inductor) is named after its weight's module;
    * a kernel that reads a ``*norm.weight`` is that norm's rmsnorm (fused with whatever else it does:
      the residual add before it, the embedding lookup, the quantisation cast after it);
    * a kernel that reads a ``cos_sin_cache`` (or is named rope/rotary) is the layer's rope;
    * a ``silu`` kernel is the layer's act_mul;
    * anything else is a generic fused node of the layer its inputs come from.
    """
    norm_w = [w for w in weight_modules if w.endswith("norm.weight")]
    other_w = [w for w in weight_modules if not w.endswith("norm.weight") and not w.endswith("cos_sin_cache") and not w.startswith("embed_tokens")]
    rope_w = [w for w in weight_modules if w.endswith("cos_sin_cache")]
    layer = producer_layers[0] if producer_layers else None
    in_layer = layer not in (None, "embed", "norm", "lm_head")
    if fn_name in GEMM_FN:
        w = other_w[0] if other_w else (weight_modules[0] if weight_modules else None)
        module = module_of_param(w) if w else (f"layers.{layer}.gemm_unknown" if in_layer else "top")
        return "gemm", "gemm", None, module
    if norm_w:
        return "rmsnorm", "rmsnorm", None, module_of_param(norm_w[-1])
    if rope_w or "rope" in kernel_name or "rotary" in kernel_name:
        return "rope", "rope", None, (f"layers.{layer}.self_attn.rotary_emb" if in_layer else "rotary_emb")
    if "silu" in kernel_name:
        return "act_mul", "silu_mul", None, (f"layers.{layer}.mlp.act_fn" if in_layer else "act_fn")
    if "embedding" in kernel_name:
        return "embedding", "embedding", None, "embed_tokens"
    module = f"layers.{layer}" if in_layer else (layer or "top")
    return "fused", "other", f"{module}.fused.{fn_name}", module


# --- W1cF F8: production's autotune choice, carried in the description node -----------------------

_BEST_CONFIG_META = ("num_warps", "num_stages", "configs_hash", "found_by_coordesc")


def launch_config_from_best_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Inductor's ``.best_config`` JSON -> the ``{"kwargs", "num_warps", "num_stages"}`` shape the
    fused-kernel relation pins its standalone launch to (``compiled.CompiledCapture._config_of``)."""
    return {
        "kwargs": {k: v for k, v in cfg.items() if k not in _BEST_CONFIG_META},
        "num_warps": int(cfg.get("num_warps", -1)),
        "num_stages": int(cfg.get("num_stages", -1)),
    }


def production_launch_config(autotune_entries: list[dict[str, Any]] | None, source_sha256: str) -> dict[str, Any] | None:
    """The launch config production's autotuner chose for the kernel whose module source hashes to
    ``source_sha256`` (``kernels.json.autotune_configs`` entries, ``kernel_identity.autotune_configs``:
    Inductor writes the standalone kernel module with the same content as the wrapper-embedded source,
    so ``kernel_module_sha256`` is the description's ``kernel_source_sha256``). ``None`` when production
    recorded no choice for it (single-config kernels never get a ``.best_config``)."""
    for e in autotune_entries or []:
        if e.get("kernel_module_sha256") == source_sha256 and isinstance(e.get("config"), dict):
            return launch_config_from_best_config(e["config"])
    return None
