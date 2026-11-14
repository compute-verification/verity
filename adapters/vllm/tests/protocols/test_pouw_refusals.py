"""The pouw option's refusals that no other CPU test of the adapter reaches, each held to its own words: `Settings.of`'s of
a transcript directory that is no string; `refusals`' of a tensor-parallel world, of a forward anchor under an executor
that counts its own, and of a schedule Pearl-C's path can't run; `_layers`' of a model with no linear layer and of two
layers that share a quant method; and H1's install's of a layout without its rows or its steps.  Under the option,
`circuit`'s of a scheme without a circuit and of an input group of another length than its Definition's, the
quantizer's of a value that is not finite, and the native prover's of a unit outside a call's population."""
from __future__ import annotations

import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.protocols.accounting.pouw.prover.circuit import SCHEMES
from verity.protocols.accounting.pouw.prover.circuit.ncp2 import Shape
from verity.protocols.accounting.pouw.verifier.coins import Challenger
from verity.protocols.accounting.pouw.verifier.serving import SCHEDULES, Schedule

from verity_vllm.protocols import pouw as P
from verity_vllm.protocols.interface import Context
from verity_vllm.protocols.pouw import circuit as PC
from verity_vllm.protocols.pouw import native as PN
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.operands import quantize
from verity_vllm.protocols.pouw.pearl_c.lanes import schedule_of

HOST = ("ncp-v1", "ncp-v1-shift24", "ncp-v2")
#: the least (n, k) of a linear the host schemes' domain takes
SHAPE = (1600, 1088)


def _serve(scheme: str, world: int = 1, **settings) -> Context:
    return Context(stage="serve", enabled=(f"{P.NAME}:{scheme}",), world=world,
                   settings={P.NAME: {"coins": Challenger(0, random.Random(0).randbytes), **settings}})


@pytest.mark.parametrize("given", [3, Path("/t"), b"/t", ("/t",)])
def test_settings_refuse_a_transcript_directory_that_is_no_string(given):
    said = "pouw setting 'transcript_dir' is a path string"
    with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
        P.Settings.of({"transcript_dir": given})
    assert P.refusals(HOST[0], _serve(HOST[0], transcript_dir=given)) == (said,)


def test_settings_take_a_path_string_and_none_for_no_transcript_directory():
    assert P.Settings.of({"transcript_dir": "/t"}).transcript_dir == "/t"
    assert P.Settings.of({"transcript_dir": None}).transcript_dir is None
    assert P.Settings.of({}).transcript_dir is None
    assert P.refusals(HOST[0], _serve(HOST[0], transcript_dir="/t")) == ()


@pytest.mark.parametrize("scheme", HOST)
def test_a_tensor_parallel_world_is_refused(scheme):
    assert P.refusals(scheme, _serve(scheme)) == ()
    assert P.refusals(scheme, _serve(scheme, world=2)) == (
        "TP 2: PoUW's units and transcript are per rank, which is not wired",)


def test_a_forward_anchor_is_refused_under_an_executor_that_counts_its_own_forwards():
    """``start_forward`` places the circuit-backed executor's first pass.  The host's other executors count their forwards
    from 0, so an anchor past 0 is refused there, and an anchor of 0 is none."""
    circuits = ", ".join(P.pouw_circuit.SCHEMES)
    for scheme in ("ncp-v1", "ncp-v1-shift24"):
        assert P.refusals(scheme, _serve(scheme, start_forward=0)) == ()
        said = (f"{scheme}: the pouw setting 'start_forward' anchors the circuit-backed executor's forward indices "
                f"({circuits}); this executor counts its own from 0")
        assert P.refusals(scheme, _serve(scheme, start_forward=3)) == (said,)
    assert "ncp-v2" in P.pouw_circuit.SCHEMES
    assert P.refusals("ncp-v2", _serve("ncp-v2", start_forward=3)) == ()


def test_a_schedule_pearl_cs_path_cannot_run_is_refused_under_the_schemes_name(monkeypatch):
    """Every registered schedule runs under every Pearl-C scheme, so `refusals` has `schedule_of`'s refusal to pass on only
    once a schedule is registered that the path can't run.  One that takes the GEMM to a side lane stands in for it."""
    for scheme in PPD.SCHEMES:
        for name, schedule in SCHEDULES.items():
            assert schedule_of(name, PPD.scheme_of(scheme)) is schedule
    monkeypatch.setitem(SCHEDULES, "side-gemm", Schedule("side-gemm", frozenset({"gemm"}), 2, "greatest", "least"))
    for scheme in PPD.SCHEMES:
        assert P.refusals(scheme, _serve(scheme, device="cuda", kernel="k")) == ()
        said = (f"{scheme}: side-gemm: gemm can't leave the main lane under h3: a later step of the call reads what it "
                "writes; side-gemm: this path runs only a_tree, tile_hash on a side lane, not gemm")
        assert P.refusals(scheme, _serve(scheme, device="cuda", kernel="k", schedule="side-gemm")) == (said,)
    assert P.refusals("ncp-v1", _serve("ncp-v1", schedule="side-gemm")) == (
        "ncp-v1: the pouw setting 'schedule' is Pearl-C's; this executor runs serially",)


class Linear:
    """A linear as far as `_layers` reads one: its weight's dtype and (n, k), and its quant method."""

    def __init__(self, method: object):
        self.weight = SimpleNamespace(dtype="float32", shape=SHAPE)
        self.quant_method = method


def _model(*layers: object) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(modules=lambda: iter(layers)))


@pytest.mark.parametrize("scheme", HOST)
def test_a_model_with_no_linear_layer_and_layers_that_share_a_quant_method_are_refused(monkeypatch, scheme):
    """An executor replaces each layer's own ``quant_method.apply``.  Two layers on one method would be one site, so every
    layer after the first on a method is refused by its ordinal; layers each on their own are taken with their (n, k)."""
    monkeypatch.setattr(P, "linear_types", lambda: (Linear,))
    with pytest.raises(ValueError, match=f"^pouw:{scheme}: the model has no linear layer to execute$"):
        P._layers(scheme, _model())
    with pytest.raises(ValueError, match=f"^pouw:{scheme}: the model has no linear layer to execute$"):
        P._layers(scheme, _model(SimpleNamespace(weight=None, quant_method=None)))
    one, other = object(), object()
    shared = "its quant method is shared with another layer"
    with pytest.raises(ValueError, match=f"^pouw:{scheme} refused at install: linear 1: {shared}; linear 3: {shared}$"):
        P._layers(scheme, _model(Linear(one), Linear(one), Linear(other), Linear(one)))
    layers = (Linear(one), Linear(other))
    _scheme, took, shapes = P._layers(scheme, _model(*layers))
    assert took == list(layers) and shapes == {0: SHAPE, 1: SHAPE}


@pytest.mark.parametrize(("rows", "steps"), [(None, 2), (4, None), (None, None)])
def test_h1s_install_refuses_a_layout_without_its_rows_or_its_steps(monkeypatch, rows, steps):
    """The refusal comes after the committed weights are read and before any other setting is: these settings hold no
    other."""
    monkeypatch.setattr(P.PHK, "layers", lambda model, types, ks: [])
    monkeypatch.setattr(P, "linear_types", lambda: ())
    target = SimpleNamespace(model=None, weight_commitments={"entry": {"weight_roots": {}, "shapes": {}}})
    with pytest.raises(ValueError, match=r"^entry: the static layout needs the pouw settings 'rows' \(T\) and 'steps' \(S\)$"):
        P._install_h1("entry", SimpleNamespace(ks=None, rows=rows, steps=steps), target)


@pytest.mark.parametrize("scheme", ["ncp-v1", "ncp-v1-shift24", "ncp-v3", ""])
def test_a_scheme_without_a_circuit_has_no_permutation_and_no_program(scheme):
    assert scheme not in SCHEMES and "ncp-v2" in SCHEMES
    said = f"{scheme}: no circuit (verity.protocols.accounting.pouw.prover.circuit.SCHEMES = {', '.join(SCHEMES)})"
    with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
        PC.kind_of(scheme)
    with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
        PC.program(scheme, 16, 64, 16)


def test_a_scheme_with_a_circuit_has_its_permutation_and_its_two_programs():
    for scheme, kind in SCHEMES.items():
        assert PC.kind_of(scheme) == kind
        c = PC.program(scheme, 16, 64, 16)
        assert (c.traced.scheme, c.traced.kind, c.shape) == (scheme, kind, Shape(16, 64, 16))
        assert (len(_groups(c.program)), len(_groups(c.weight_program))) == (11, 4)


def _groups(prog) -> list[int]:
    """How many values each of a Program's input groups takes, in the Definition's order."""
    return [len(group.refs) for group in prog.root.input_colls]


def test_a_program_is_not_evaluated_on_an_input_group_of_another_length(monkeypatch):
    """Each group of a call's Program and of its weight side's, one value short and one over, with every other group
    at its length: the refusal names the group's two lengths and comes before any gate is evaluated."""
    c = PC.program("ncp-v2", 16, 64, 16)
    monkeypatch.setattr(PC, "evaluate_region", lambda *_: pytest.fail("a gate was evaluated before the refusal"))
    for prog in (c.program, c.weight_program):
        sizes = _groups(prog)
        for at, size in enumerate(sizes):
            for given in (size - 1, size + 1):
                args = [[0] * (given if g == at else s) for g, s in enumerate(sizes)]
                with pytest.raises(ValueError, match=f"^an input group has {given} values, the Definition takes {size}$"):
                    PC._transcript(prog, args)


def test_a_program_is_evaluated_whole_on_input_groups_of_its_lengths():
    c = PC.program("ncp-v2", 16, 64, 16)
    for prog in (c.program, c.weight_program):
        trace = PC._transcript(prog, [[0] * size for size in _groups(prog)])
        assert len(trace) == prog.circuit.size
        assert all(trace[gate] == 0 for group in prog.root.input_colls for (_space, _zero, gate) in group.refs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_the_quantizer_refuses_rows_with_a_value_that_is_not_finite_and_takes_them_without_it(bad):
    rows = np.array([[63.0, -31.0], [0.0, 0.0], [0.5, bad]], dtype=np.float32)
    with pytest.raises(ValueError, match="^pouw: the quantizer refuses NaN and infinity$"):
        quantize(rows)
    rows[2, 1] = -0.125
    q, scale = quantize(rows)
    assert (q.dtype, scale.dtype) == (np.int8, np.float32)
    assert q.tolist() == [[63, -31], [0, 0], [63, -16]]
    assert scale.tolist() == [1.0, 1.0, float(np.float32(0.5) * (np.float32(1.0) / np.float32(63.0)))]


class Rows:
    """A weight's int7 rows and their FP32 scale encodings, as `circuit.Rows` hands them on."""

    def __init__(self, b: np.ndarray, t: np.ndarray):
        self.b, self.t, (self.n, self.k) = b, t, b.shape

    def rows(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        return self.b[lo:hi], self.t[lo:hi]


def _native(m: int, k: int, n: int) -> PN.Native:
    """The native prover of a call of zeros at m × k → n, every scale FP32's 1 (its encoding ``0x3F800000``)."""
    one = 0x3F800000
    weight = Rows(np.zeros((n, k), np.int8), np.full(n, one, np.uint32))
    call = PC.Call(np.zeros((m, k), np.int8), np.full(m, one, np.uint32), weight, np.zeros(n, np.uint16), bytes(32),
                   forward=0, layer=0, site=0, weight=0, rowids=np.arange(m, dtype=np.uint32),
                   colids=np.arange(n, dtype=np.uint32))
    return PN.Native(call, "ncp-v2")


def test_a_unit_outside_a_calls_population_or_its_weight_sides_is_refused():
    """A call of 32 × 64 → 48 has 1,548 units, and its weight side has 3, one a 16-column strip.  The unit before the
    first and the unit after the last have no place, no template and no outputs; the first and the last have theirs."""
    nat = _native(32, 64, 48)
    assert (nat.population, nat.side.population) == (1548, 3)
    for u in (-1, 1548):
        for read in (nat.locate, nat.template, nat.outputs):
            with pytest.raises(IndexError, match=f"^unit {u} outside 0..1547$"):
                read(u)
    assert (nat.locate(0), nat.locate(1547)) == (("digest", (0,)), ("dequant", (1, 2, 255)))
    for u in (-1, 3):
        with pytest.raises(IndexError, match=f"^Y unit {u} outside 0..2$"):
            nat.side.outputs(u)
    assert [nat.side.outputs(u).shape for u in (0, 2)] == [(3088,), (3088,)]
