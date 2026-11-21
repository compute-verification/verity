"""The served window's row name (`served_table.row_id`): a rule over the model, the precision and the slots, handed to the
runner as ROW_ID. It reads no traced row, no manifest and no predicted Build: the same three give the same name, and a change
in any one changes it."""
from __future__ import annotations

import sys

import pytest

from verity.channel import Refused
from verity_cli import run as R

R.registry()
T = sys.modules["served_table"]
BASE = ("verity-int8/Llama-3.2-1B-INT8", "verity_int8", 4)


def test_the_same_model_precision_and_slots_give_the_same_name():
    assert T.row_id(*BASE) == T.row_id(*BASE) == "verity-int8-llama-3-2-1b-int8__verity-int8__s4"


@pytest.mark.parametrize("change", [("verity-int8/Llama-3.2-3B-INT8", "verity_int8", 4), ("verity-int8/Llama-3.2-1B-INT8", "bfloat16", 4),
                                    ("verity-int8/Llama-3.2-1B-INT8", "verity_int8", 5)], ids=["model", "precision", "slots"])
def test_a_change_in_any_one_changes_it(change):
    assert T.row_id(*change) != T.row_id(*BASE)


def test_the_runner_gets_it_as_ROW_ID_from_the_workload_alone():
    w = {**T.SERVED.default_params, "load": "varied"}
    cfg = {"workload": {"params": w}, "service": {"law": "x"}}
    got = T._serve(cfg, {}, {}, {})
    assert got["ROW_ID"] == T.row_id(w["model"], w["precision"], w["slots"])
    assert T._serve({**cfg, "workload": {"params": {**w, "slots": 5}}}, {}, {}, {})["ROW_ID"] != got["ROW_ID"]


@pytest.mark.parametrize("args", [("///", "verity_int8", 4), ("m", "", 4), ("m", "p", 0), ("m", "p", True)])
def test_a_name_that_names_nothing_is_refused(args):
    with pytest.raises(Refused, match="a row is named by"):
        T.row_id(*args)
