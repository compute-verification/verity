"""The pouw option's refusals that no other CPU test of the adapter reaches, each held to its own words: `Settings.of`'s of
a transcript directory that is no string; `refusals`' of a tensor-parallel world, of a forward anchor under an executor
that counts its own, and of a schedule Pearl-C's path can't run; `_layers`' of a model with no linear layer and of two
layers that share a quant method; and H1's install's of a layout without its rows or its steps."""
from __future__ import annotations

import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity.protocols.accounting.pouw.verifier.coins import Challenger
from verity.protocols.accounting.pouw.verifier.serving import SCHEDULES, Schedule

from verity_vllm.protocols import pouw as P
from verity_vllm.protocols.interface import Context
from verity_vllm.protocols.pouw import pearl_c as PPD
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
