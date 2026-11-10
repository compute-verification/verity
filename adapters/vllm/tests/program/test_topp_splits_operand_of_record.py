"""The split count S as an OPERAND of the top-p Definitions (`test_topp_splits_operand`), on the B8 record's rows
(`fstoch01_rows.npz`, the kernel-consumed bf16 logits at S = 16, V = 128256, over a tracked file's limit):

  1. S is not decorative: the same record row masks differently at S = 16 and S = 32;
  2. the record rows through the operand form: masked row bit-exact to the kernel (the fixed model behind the operand).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.program.test_topp_splits_operand import _bits, _eval, _u32, _u32s
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import sampling as SM
from verity.ml.definitions.registry import topp_split as TS

f32 = np.float32
RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "topp_split_fixture", "fstoch01_rows.npz")


def test_the_operand_changes_the_result_where_the_geometry_matters():
    """S is not decorative: the same row masks differently under two split counts (the tie quota is per split)."""
    p = 0.95
    z = np.load(RECORD)
    meta = json.load(open(RECORD + ".json"))
    key = "step37_row4"
    bf = (z[key + "_bf16_logits_bits"].astype(np.uint32) << 16).view(f32)
    x = TS.tl_div(bf, f32(meta[key]["device"]["temperature"]), TS.Cfg())
    k16 = TS.topp_keep_row(x, p, 16)
    k32 = TS.topp_keep_row(x, p, 32)
    assert k16.tolist() != k32.tolist(), "the record row masks the same at S = 16 and 32 -- not a geometry-sensitive row"


@pytest.mark.slow
@pytest.mark.parametrize("key", ["step37_row4", "step82_row3", "step92_row3"])
def test_fstoch01_record_rows_are_bit_exact_through_the_operand_form(key):
    z = np.load(RECORD)
    meta = json.load(open(RECORD + ".json"))
    d = meta[key]["device"]
    S, V = int(meta[key]["S"]), int(meta[key]["V"])
    assert S == 16 and V == 128256
    bf = (z[key + "_bf16_logits_bits"].astype(np.uint32) << 16).view(f32)
    x_in = TS.tl_div(bf, f32(d["temperature"]), TS.Cfg())
    out = _u32s(_eval(bind(SM.TopPMask, V=V), [x_in.view(np.uint32), _u32(_bits(d["top_p"])), _u32(S)]))
    obs = z[key + "_masked_f32_bits"]
    diff = np.nonzero(out != obs)[0]
    assert diff.size == 0, f"{key}: {diff.size} lanes differ from the kernel-consumed masked row at S = 16 (operand): {diff[:8].tolist()}"
