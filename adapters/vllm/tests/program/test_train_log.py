"""`rules.training.Log`: at a compute capability torch.log is CUDA's logf, libdevice's ``__nv_logf`` without flush-to-zero
(`NvLogf_v4`, on bits `NvLogf_v5`), a bf16 result `F2fpBf16_v1`, bit for bit; without one the rule refuses."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.program.test_train_step import _bits, _data, _outputs, _program, _same_bits
from verity.definitions.registry import cublas_bf16 as CB
from verity.definitions.registry import prims as P
from verity.definitions.registry import sampling as SM
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.rules.training import training_rules

#: subnormals of both signs (where the FTZ build gives -inf), zeros, ±1, FLT_MIN, infinities and a NaN
LOG_EDGES = torch.from_numpy(np.array([0x00000001, 0x80000001, 0x007FFFFF, 0x00400000, 0, 0x80000000, 0x3F800000, 0xBF800000,
                                       0x00800000, 0x7F800000, 0xFF800000, 0x7FC00000], dtype=np.uint32).view(np.float32).copy())


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_gpu_log_is_logf_without_flush_to_zero(dt: torch.dtype) -> None:
    x = torch.cat([_data(19, seed=40), LOG_EDGES]).to(dt)
    prog = _program(lambda v: (torch.log(v),), x, rules=training_rules(cc=CB.TRAIN_CC))
    ids = {nd.fn.id.split("{")[0] for nd in prog.root.body.nodes}
    assert "NvLogf_v4" in ids, ids
    words = [P.Bf16ToF32.evaluate(v) for v in _bits(x)] if dt == torch.bfloat16 else _bits(x)
    want = [SM.nv_logf_gpu(v, ftz=False) for v in words]
    if dt == torch.bfloat16:
        want = [P.F2fpBf16.evaluate(w) << 16 for w in want]
    _same_bits(_outputs(prog, x), torch.from_numpy(np.array(want, dtype=np.uint32).view(np.float32).copy()))


def test_cpu_log_is_refused() -> None:
    x = _data(7, seed=41)
    gm = make_fx(lambda v: (torch.log(v),), tracing_mode="fake")(x)
    u = derive_graph(gm, (x,), mode="reference", rules=training_rules(), name="T").unsupported
    assert isinstance(u, Unsupported) and "CUDA's logf" in u.hint, u
