"""The epoch's Program digest computed before the serve from its class (`window_circuit.fixed_program`)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from verity.commitments.periods import interface as IF

from tests.program.predict.test_window_rows import CLASS, GRAIN, RUN, _registered
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_record as WREC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_class import GREEDY

ONE = WC.WindowClass(1, 1, 1)


def _served(*units):
    """A compile as the close's `takes` reads it: each request's prompt length LP and decode steps T."""
    return SimpleNamespace(requests=[SimpleNamespace(request=i, LP=lp, T=t, prompt=ids, carried_in=False, carried_out=False,
                                                     C=0, parse=None, policy=GREEDY) for i, (lp, t, ids) in enumerate(units)])


def _docket_program(compiled, cls, cap=64):
    low = WC.lower(_registered(), cls, None, GRAIN, requests=WC.takes(compiled, cls))
    return IF.Window(low.circuit, RUN, WREC.window_design("rate:1/4", low.circuit), cap).window.partition["program"]


def test_the_digest_before_the_serve_is_the_closed_epochs_dockets_program_whatever_its_prompt_id_and_cap():
    fixed = WC.fixed_program(_registered(), ONE, GRAIN)
    assert fixed == _docket_program(_served((1, 0, [128000])), ONE)
    assert fixed == _docket_program(_served((1, 0, [17])), ONE, cap=32)


def test_an_epoch_with_its_slot_empty_has_no_circuit_to_close():
    with pytest.raises(Unexpressible, match="a window of no request has no circuit"):
        _docket_program(_served(), ONE)


def test_a_class_whose_circuit_the_serve_chooses_is_refused_by_name():
    with pytest.raises(Unexpressible, match="moves with the requests the serve takes"):
        WC.fixed_program(_registered(), CLASS, GRAIN)
    assert _docket_program(_served((4, 1, [1] * 4), (3, 1, [2] * 3)), CLASS) != \
        _docket_program(_served((4, 1, [1] * 4), (4, 1, [2] * 4)), CLASS)
