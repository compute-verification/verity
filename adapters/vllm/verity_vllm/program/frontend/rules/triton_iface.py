"""The opaque node a DIRECT Triton launch becomes under export (CB-A). Pure torch: defines the op and its fake
impl; the interception itself (patching `triton.runtime.jit.JITFunction.run` while exporting) lives in
`verity_vllm/program/frontend/triton_capture.py`, where triton is importable.

    verity_cba::triton_launch(str kernel, Tensor(a!)[] tensors, str meta_json) -> ()

`kernel` is the launched JIT function's qualified name (`<module>.<name>`); `tensors` the tensor arguments in the kernel
signature's order; `meta_json` = {"kernel", "grid", "scalars": {arg name: value (constexprs included)}, "tensor_args": [arg
names of `tensors`], "constexprs": [indices]}. A rule binds one *kernel* (name + pinned constexprs) to one registered kind.
"""

from __future__ import annotations

import json

import torch

_LIB_NAME = "verity_cba"
_lib = torch.library.Library(_LIB_NAME, "FRAGMENT")  # noqa: TOR901
try:
    _lib.define("triton_launch(str kernel, Tensor(a!)[] tensors, str meta_json) -> ()")
    torch.library.register_fake(f"{_LIB_NAME}::triton_launch")(lambda kernel, tensors, meta_json: None)
except RuntimeError:  # already defined in this process
    pass

triton_launch = torch.ops.verity_cba.triton_launch


def launch_meta(node) -> dict:
    """The decoded `meta_json` of a `triton_launch` graph node."""
    return json.loads(node.args[2])


def launch_tensor(node, meta: dict, arg_name: str):
    """The fx node passed for kernel argument `arg_name` (None if that argument was not a tensor)."""
    names = meta.get("tensor_args", [])
    if arg_name not in names:
        return None
    return node.args[1][names.index(arg_name)]


__all__ = ["triton_launch", "launch_meta", "launch_tensor"]
