"""The suite's step: the lf model's tiny training step (`integrations/vllm/tests/program/train_step_lf.py`, loaded by its path)
with its log-free loss, derived by the training-step frontend at the step's batch shape and at the structure's, and handed
over as a lane hands it over (`grader.write_handover`); at a compute capability `cc`, the step's bf16 linears' GEMMs are lowered
to the routes `ROUTES` stands in for that GPU's cuBLAS, and the structure is derived without it, as its routes don't cover its
shapes."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import grader as G
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity_vllm.program.frontend.train_step import AdamW, export_step, primitives, step_report

LF_PATH = Path(__file__).resolve().parents[4] / "integrations" / "vllm" / "tests" / "program" / "train_step_lf.py"
#: the step's batch and the structure Program's, as rows of tokens
STEP, STRUCTURE = (2, 17), (3, 5)
#: the compute capability a step's bf16 products may be derived at, and the routes its cuBLAS stands in to take for the tiny
#: step's linears' GEMMs at M = 2 x 16 tokens (its capture covers the workloads' shapes, not these): unsplit, but for a split-K
#: over a bf16 workspace and a serial split-K, one route of each kind the grader reads
CC = CB.TRAIN_CC
ROUTES = {**{(32, K, N, role): {"kernel": "nosplit"} for K, N in ((64, 64), (64, 192), (192, 64)) for role in CB.TRAIN_LINEAR_ROLES},
          (32, 64, 64, "dW"): {"kernel": "splitk", "SPLIT_K": 2, "partial": "bf16"},
          (32, 64, 192, "dX"): {"kernel": "serialk", "SPLIT_K": 3}}


def _lf():
    spec = importlib.util.spec_from_file_location("grader_tests_lf", LF_PATH)
    mod = importlib.util.module_from_spec(spec)
    # registered before it runs: dataclasses resolve the module's string annotations through sys.modules
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


LF = _lf()


def _derive(dtype, rows: int, tokens: int, cc: tuple[int, int] | None = None):
    model, batch = LF.meta_step("tiny", rows, ctx=tokens - 1, vocab=64, matmul_dtype=dtype)
    opt = AdamW(**LF.ADAMW)
    ex = export_step(model, LF.square_loss, opt, batch)
    with pytest.MonkeyPatch.context() as mp:
        for key, route in ROUTES.items() if cc else ():
            mp.setitem(CB.TRAIN_ROUTES, key, route)
        rep = step_report(model, LF.square_loss, opt, batch, exported=ex, cc=cc)
    assert rep.program is not None, rep.unsupported
    return rep.program, ex[0]


@pytest.fixture(scope="session")
def steps(tmp_path_factory):
    """`steps(dtype, cc)`: the tiny step with its linears' operands in `dtype` (None: f32), its bf16 products at the compute
    capability `cc` (None: torch CPU's), derived once a session: its hand-over (`path`), its step Program, and the step's and
    the structure's ATen graphs (`gm`, `gm2`)."""
    made: dict = {}

    def get(dtype=None, cc: tuple[int, int] | None = None) -> SimpleNamespace:
        if (dtype, cc) not in made:
            (prog, gm), (prog2, gm2) = _derive(dtype, *STEP, cc), _derive(dtype, *STRUCTURE)
            path = G.write_handover(tmp_path_factory.mktemp("handover"), prog, prog2, run="tiny", checkpoints=[("start", 1)],
                                    frontend="0" * 40, step_batch=STEP, structure_batch=STRUCTURE)
            made[dtype, cc] = SimpleNamespace(path=path, program=prog, gm=gm, gm2=gm2)
        return made[dtype, cc]

    return get


@pytest.fixture
def unit_prices(monkeypatch):
    """|C| (`circuit_size`) with every primitive at one gate and one AND, so a class's gates are its word gates (each call of a
    primitive one gate): `test_circuit_size_on_the_tiny_step` prices the step at the Boolean prices themselves."""
    monkeypatch.setattr(G, "boolean_prices", lambda prims: {p: {"boolean": p, "and": 1, "xor": 0, "not": 0, "gates": 1}
                                                            for p in prims})


def dot_alone(monkeypatch, program) -> dict:
    """|C| (`circuit_size`) at the tensor-core dot alone: the dot `program`'s GPU routes call each k16 step at its Boolean
    price, every other primitive at none. Returns that price."""
    prims = primitives(program)
    (dot,) = [p for p in prims if p.startswith("HopperBF16WgmmaDot16")]
    price = G.boolean_prices({dot: prims[dot]})[dot]
    none = {"and": 0, "xor": 0, "not": 0, "gates": 0}
    monkeypatch.setattr(G, "boolean_prices", lambda ps: {p: price if p == dot else {"boolean": p, **none} for p in ps})
    return price
