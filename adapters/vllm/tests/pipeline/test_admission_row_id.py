"""The admission planner reads a row id through `config` (its pattern in a log line, its fields), and plans the same row
config it always did: the outputs here are pinned from the planner's own parse."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

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


ATTEMPTS = {
    "log": {"commit.log": f"x  --workload workloads/{ROW}.json --pairs 2\n"},
    "preflight": {"preflight.json": json.dumps({"row": "llama32-1b__bf16__h100__tp2__b4__i256__o32__static__greedy__eager"})},
    "row.log": {"row.log": f"running workloads/{ROW}.json\nworkloads/not_a_row.json\n"},
    "none": {"launcher.log": "workloads/foo__bar.json\n"},
}


@pytest.mark.parametrize(("name", "model", "requests", "max_context"),
                         [("log", "phi3-mini", 16, 576), ("preflight", "llama32-1b", 4, 288), ("row.log", "phi3-mini", 16, 576),
                          ("none", "llama32-1b", 1, 1152)])
def test_row_config_from_attempt_reads_the_row(tmp_path: Path, name: str, model: str, requests: int, max_context: int):
    for f, text in ATTEMPTS[name].items():
        (tmp_path / f).write_text(text)
    cfg = A.row_config_from_attempt(str(tmp_path))
    assert (cfg.model, cfg.requests, cfg.max_context) == (model, requests, max_context)


def test_row_config_from_attempt_is_pinned(tmp_path_factory):
    out = {}
    for name, files in ATTEMPTS.items():
        d = tmp_path_factory.mktemp(name.replace(".", "_"))
        for f, text in files.items():
            (d / f).write_text(text)
        cfg = A.row_config_from_attempt(str(d))
        out[name] = {k: v for k, v in dataclasses.asdict(cfg).items() if k != "notes"} | {"notes": cfg.notes}
    assert hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()[:16] == "02bada87b8b14031"
