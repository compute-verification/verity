"""The admission planner reads a row id through `config` (its fields): the row grammar, and the model key of a row."""
from __future__ import annotations

import re

from verity_vllm import config
from verity_vllm.pipeline.telemetry import admission as A

ROW = "phi3-mini__bf16__rtxpro6000__tp1__b16__i512__o64__mixed__t1__eager"


def test_row_id_pattern_matches_the_row_grammar():
    assert re.fullmatch(config.ROW_ID_PATTERN, ROW)
    assert str(config.RowId.parse(ROW)) == ROW
    for bad in ("foo__bar", "phi3__bf16__h100__tp1__b16__i512__o64__mixed__t1", "phi3__bf16__h100__w1__b16__i512__o64__mixed__t1__eager"):
        assert not re.fullmatch(config.ROW_ID_PATTERN, bad), bad
    assert config.row_tokens(ROW)[0] == "phi3-mini" and len(config.row_tokens(ROW)) == len(config.ROW_FIELDS)
    assert config.row_tokens("x") == ["x"] and config.row_tokens("") == [""]


def test_model_key_for_row():
    assert [A.model_key_for_row(r) for r in ("llama32-1b-padded__bf16__h100__tp1__b8__i1024__o128__mixed__greedy__eager", "phi3__x",
                                             "nounderscore", "")] == ["llama32-1b", "phi3", "nounderscore", ""]
