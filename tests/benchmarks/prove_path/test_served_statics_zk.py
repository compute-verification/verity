"""`served_statics_zk`'s cost: at the served class the largest block that fits Flock's 2^27 is 64 steps."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from verity_vllm.compiler import served_statics as SS

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "benchmarks" / "prove_path")]
import served_statics_zk as Z  # noqa: E402


def test_the_served_class_fits_at_64_steps_a_block():
    c = Z.CLASSES["served"]
    fits = {G: Z.zk_block(SS.rows(SimpleNamespace(**SS.statics(SimpleNamespace(**c), G=G))), SS.size(**SS.statics(
        SimpleNamespace(**c), G=G))["ands"])["fits"] for G in (16, 64, 128)}
    assert fits == {16: True, 64: True, 128: False}
