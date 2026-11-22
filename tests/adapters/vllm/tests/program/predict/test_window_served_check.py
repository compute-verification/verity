"""A served window's check (`window_builder.check_served`): a template past what Flock takes is refused by name, and the
templates the door can't prove are named under each reason (no row past the row floor; a block of all their units under
the prover's floor), and stay in the circuit's population."""
from __future__ import annotations

import json
import re
from collections import Counter

import pytest
from verity.circuits import partition_object as PO
from verity.commitments.periods import interface as IF

from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, GRAIN, WHOLE
from verity_vllm.pipeline import advice as ADV
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
from verity_vllm.program.predict.phi import served_phi
from verity_vllm.program.predict.steps import Registered, compile_steps

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")

#: the H100 smoke's serve.json, as φ reads it
SERVE = {"model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "execution": "bi-eager",
         "target": {"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": True, "attention_backend": None,
                    "vllm_revision": "d9105ea80", "numerics_profile": "b1-eager-v3", "flash_attn_version": None}}
G2 = {**S.GREEDY, "max_tokens": 2}


def _window(reg, cls, sizes, sched, grain=WC._GRAIN):
    advice, ingress = S._session(sizes, sched)
    return WC.lower(reg, cls, None, grain, requests=WC.takes(compile_steps(reg, advice, ingress), cls)).circuit


def _in_population(circuit, named):
    tp = PO.TemplatePartition(*circuit.program())
    ranges = Counter()
    for _k, t, _a, n in tp.ranges:
        ranges[t] += n
    n = {IF.descriptor_id(fn): k for fn, k in circuit.templates}
    assert {t: ranges[t] for t in named} == {t: n[t] for t in named} and all(n[t] > 0 for t in named)
    assert tp.population == sum(n.values())


@pytest.mark.slow
def test_a_tiny_window_the_older_composer_refuses_passes_and_names_its_small_templates():
    """About a minute, the population's Program built: the one window small enough to pass here that the older composer
    refuses."""
    circuit = _window(S._registered(), CLASS, *WHOLE, GRAIN)
    with pytest.raises(Unexpressible, match="no row past 111 B"):
        WC.check_provable(circuit)
    named = WC.check_served(circuit)
    assert named["row_floor"] and all("no row has a compression before its last" in why for why in named["row_floor"].values())
    assert named["prover_floor"] and all("under 25" in why for why in named["prover_floor"].values())
    _in_population(circuit, {**named["row_floor"], **named["prover_floor"]})


@pytest.mark.slow
def test_run_inputs_of_the_tiny_window_are_read_across_its_ranges():
    """About a minute, the population's Program built: the one class here whose run input nodes are each read by several
    ranges of C's root, which `interface.run_input_reads` takes as one read a node, unit i its row i."""
    circuit = _window(S._registered(), CLASS, *WHOLE, GRAIN)
    built = circuit.program()
    tp = PO.TemplatePartition(*built)
    filled = IF.filled_ports(circuit.registered, circuit.wires, circuit.ingress, circuit.boundary_reads)
    runs = IF.run_input_reads(circuit.templates, *built, filled, circuit.cuts)
    assert runs and len({r.param for r in runs}) == len(runs)
    assert any(Counter(t for _k, t, _a, _n in tp.ranges)[r.reader] > 1 for r in runs)
    assert IF.members(circuit, 64, built, runs)


def test_a_template_past_the_rows_flock_takes_is_refused_by_name(monkeypatch):
    circuit = _window(S._registered(), CLASS, *WHOLE, GRAIN)
    widest = max(circuit.templates, key=lambda t: len(IF.layout(t[0]).cut(circuit.cuts.get(IF.descriptor_id(t[0])))))[0]
    rows = len(IF.layout(widest).cut(circuit.cuts.get(IF.descriptor_id(widest)))) + len(IF.layout(widest).outs)
    monkeypatch.setattr(IF, "MAX_ROWS", rows - 1)
    with pytest.raises(Unexpressible, match=f"Flock wouldn't prove: .*{re.escape(IF.descriptor_id(widest))}: {rows} rows, "
                                            f"past a proof block's {rows - 1}"):
        WB.check_served(circuit)


@pytest.mark.slow
def test_a_served_window_at_the_class_of_record_passes_and_names_what_the_door_cannot_prove():
    """The H100 smoke's window (4 requests of 27 prompt ids and 2 outputs, one window): passes, naming the small templates
    under the row floor and the prover's, each of whose units is in the population; prints each list's templates and units.
    Some four minutes: the class's own templates, which no smaller class has."""
    hf = json.loads(pinned_hf_config(SERVE["model"], SERVE["revision"]).read_text())
    reg = Registered.of(served_phi(SERVE, hf), 657832114885280620, ADV.CAP_BITS)
    circuit = _window(reg, WC.WindowClass(4, 27, 2), [(27, 2, G2)] * 4,
                      [[(i, 0, 27) for i in range(4)], [(i, 27, 1) for i in range(4)]])
    named = WC.check_served(circuit)
    n = {IF.descriptor_id(fn): k for fn, k in circuit.templates}
    for reason, got in named.items():
        _in_population(circuit, got)
        print(f"UNPROVABLE {reason}: {len(got)} of {len(n)} templates, {sum(n[t] for t in got)} of {sum(n.values())} units: "
              + " ".join(sorted(got)))
    assert named["row_floor"] == {}
