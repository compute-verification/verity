"""Op vocabulary and argument binding: the parts of a dispatcher op name, a Triton launch's kernel name, and an event's
arguments by parameter name."""

from __future__ import annotations

from typing import Any

from verity_vllm.observe.fold.tree import TreeNode


def op_parts(op: str) -> tuple[str, str, str]:
    """`'aten.slice.Tensor' -> ('aten', 'slice', 'Tensor')`; `'_C.silu_and_mul.default' -> ('_C', 'silu_and_mul',
    'default')`; a Triton function path `'a.b.matmul_kernel_persistent' -> ('a.b', 'matmul_kernel_persistent', '')`
    is *not* what this is for (use `triton_name`)."""
    parts = op.split(".")
    if len(parts) >= 3:
        return parts[0], parts[1], ".".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return "", op, ""


def base_op(op: str) -> str:
    """`'aten.slice.Tensor' -> 'aten.slice'` (namespace.name without the overload)."""
    ns, name, _ = op_parts(op)
    return f"{ns}.{name}" if ns else name


def op_name(op: str) -> str:
    """The op's name without namespace and overload: `'_C_cache_ops.reshape_and_cache_flash.default' ->
    'reshape_and_cache_flash'`."""
    return op_parts(op)[1]


def triton_name(node: TreeNode) -> str:
    """Last component of a Triton launch's `fn` (`'...batch_invariant.matmul_kernel_persistent' ->
    'matmul_kernel_persistent'`); '' for dispatcher nodes."""
    if not node.is_triton:
        return ""
    return node.enter.fn.rsplit(".", 1)[-1]


def schema_param_names(schema: str) -> list[str]:
    """Parameter names of a dispatcher schema string, in order (`'aten::embedding(Tensor weight, Tensor indices,
    SymInt padding_idx=-1, ...) -> Tensor'` -> `['weight', 'indices', 'padding_idx', ...]`).  Tolerates
    `Tensor(a!)`, `int[]`, `Tensor?`, defaults and the `*` keyword-only marker; returns [] when it cannot parse."""
    if not schema or "(" not in schema:
        return []
    start = schema.index("(")
    depth = 0
    end = None
    for i in range(start, len(schema)):
        c = schema[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return []
    body = schema[start + 1:end]
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for c in body:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if cur:
        parts.append("".join(cur))
    names: list[str] = []
    for p in parts:
        p = p.strip()
        if not p or p == "*":
            continue
        lhs = p.split("=", 1)[0].strip()
        toks = lhs.split()
        if len(toks) < 2:
            continue
        names.append(toks[-1])
    return names


def bound_args(node: TreeNode) -> dict[str, Any]:
    """Arguments by name: Triton launches are recorded by name already; dispatcher calls are bound through their
    schema (positional args -> schema parameter names, then kwargs).  Unnamed positions become `arg<i>`."""
    if node.is_triton:
        return dict(node.enter.args)
    names = schema_param_names(node.enter.schema)
    out: dict[str, Any] = {}
    for i, a in enumerate(node.enter.args):
        out[names[i] if i < len(names) else f"arg{i}"] = a
    for k, v in node.enter.kwargs.items():
        out[k] = v
    return out


def first(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def as_float(x: Any) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None
