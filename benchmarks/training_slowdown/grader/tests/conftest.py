"""The suite's step: the lf model's tiny training step (`integrations/vllm/tests/program/train_step_lf.py`, loaded by its path)
with its log-free loss, derived by the training-step frontend at the step's batch shape and at the structure's, and handed
over as a lane hands it over (`grader.write_handover`)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import grader as G
from verity_vllm.program.frontend.train_step import AdamW, export_step, step_report

LF_PATH = Path(__file__).resolve().parents[4] / "integrations" / "vllm" / "tests" / "program" / "train_step_lf.py"
#: the step's batch and the structure Program's, as rows of tokens
STEP, STRUCTURE = (2, 17), (3, 5)


def _lf():
    spec = importlib.util.spec_from_file_location("grader_tests_lf", LF_PATH)
    mod = importlib.util.module_from_spec(spec)
    # registered before it runs: dataclasses resolve the module's string annotations through sys.modules
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


LF = _lf()


def _derive(dtype, rows: int, tokens: int):
    model, batch = LF.meta_step("tiny", rows, ctx=tokens - 1, vocab=64, matmul_dtype=dtype)
    opt = AdamW(**LF.ADAMW)
    ex = export_step(model, LF.square_loss, opt, batch)
    rep = step_report(model, LF.square_loss, opt, batch, exported=ex)
    assert rep.program is not None, rep.unsupported
    return rep.program, ex[0]


@pytest.fixture(scope="session")
def steps(tmp_path_factory):
    """`steps(dtype)`: the tiny step with its linears' operands in `dtype` (None: f32), derived once a session: its hand-over
    (`path`), its step Program, and the step's and the structure's ATen graphs (`gm`, `gm2`)."""
    made: dict = {}

    def get(dtype=None) -> SimpleNamespace:
        if dtype not in made:
            (prog, gm), (prog2, gm2) = _derive(dtype, *STEP), _derive(dtype, *STRUCTURE)
            path = G.write_handover(tmp_path_factory.mktemp("handover"), prog, prog2, run="tiny", checkpoints=[("start", 1)],
                                    frontend="0" * 40, step_batch=STEP, structure_batch=STRUCTURE)
            made[dtype] = SimpleNamespace(path=path, program=prog, gm=gm, gm2=gm2)
        return made[dtype]

    return get
