import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kernels as K


@pytest.mark.parametrize("tw,s,warps", [(86, 96, 12), (43, 192, 8), (43, 192, 12), (128, 64, 16)])
def test_tensor_configs_fit_the_fused_kernel(tw, s, warps):
    """Every config the tensor arm launches passes `wt_fused`'s bounds and its shared memory at M = 32
    (`fused_smem_bytes<4, 0>`), for p768, p1536 (8 and 12 warps) and p512."""
    tlib = object.__new__(K.TensorLib)
    tlib.sb, tlib.task_words, tlib.fused_warps, tlib.fused_span = tw * s, tw, warps, (tw + 30) // 32 + 1
    cfgs = tlib.configs()
    assert cfgs
    for d, st in cfgs:
        assert d >= tlib.fused_span and d + 2 <= warps and st >= -(-32 * d // tw)
        assert st * tlib.sb + 4 * 8 * 528 + 16 * ((5 * st + 1) // 2) + 16 <= K.SMEM_PER_BLOCK
