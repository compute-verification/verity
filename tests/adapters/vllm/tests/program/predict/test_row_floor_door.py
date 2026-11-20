"""The row floor (`window_builder.check_served`'s ``row_floor``, Lean's ``compose``): each template of a served window,
its first unit's block lowered by Lean (`one_circuit.lower`, ``flock-lower-check compose``), has a row past it, and the
door refuses a unit with none."""
from __future__ import annotations

import json
import shutil

import pytest
from verity.circuits import partition_object as PO
from verity.circuits.codec import descriptor_chunks
from verity.circuits.defs import bind
from verity.commitments.periods import interface as IF
from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF

from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, GRAIN, WHOLE
from tests.program.predict.test_window_served_check import _window
from verity_vllm.definitions.registry import io_shell as IO
from verity_vllm.program.predict import window_circuit as WC

pytestmark = [pytest.mark.usefixtures("code_versions_stood_in"),
              pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]

ROW_FLOOR = "no row has a compression before its last block"


def door_verdicts(circuit) -> dict[str, str | None]:
    """Each template's first unit, a block of its own, through the door's lowering: None where Lean composes it, else
    its refusal."""
    program, query = circuit.program()
    c = b"".join(descriptor_chunks(program))
    tp = PO.TemplatePartition(program, query)
    first: dict[str, int] = {}
    for _k, t, a, n in tp.ranges:
        if n and t not in first:
            first[t] = a
    cuts, blocks, at = sorted(first.values()), [], 0
    for u in cuts:
        blocks += [(at, u)] if at < u else []
        blocks.append((u, u + 1))
        at = u + 1
    blocks += [(at, tp.population)] if at < tp.population else []
    part = PO.build(json.loads(c), query, [blocks])
    cut, exe = OC.cuts(c, part), PF.lower_executable()
    got = {}
    for t, u in first.items():
        try:
            OC.lower(OC.restrict(c, part, [(u, u + 1)]), part, lambda: exe, cut=cut,
                     formed=OC.formed_in(c, part, [(u, u + 1)]))
            got[t] = None
        except OC.Refused as e:
            got[t] = str(e)
    return got


@pytest.mark.slow
def test_no_template_of_the_window_is_under_the_row_floor_by_the_plan_or_the_door():
    """Some minutes: a Lean build of ``flock-lower-check`` and one lowering a template of the tiny window."""
    circuit = _window(S._registered(), CLASS, *WHOLE, GRAIN)
    door = door_verdicts(circuit)
    assert set(door) == {IF.descriptor_id(fn) for fn, _ in circuit.templates}
    print("DOOR " + json.dumps({t: (why or "admitted")[-300:] for t, why in sorted(door.items())}))
    named = WC.check_served(circuit)
    print("NAMED " + json.dumps(named))
    assert named["row_floor"] == {}
    assert not [t for t, why in door.items() if why and ROW_FLOOR in why]


@pytest.mark.slow
def test_the_door_refuses_a_unit_whose_rows_are_all_under_the_floor():
    """`OutCollect_v1{KO=2}`, rows of 128 bits at most: the floor's own refusal, by Lean's ``compose``."""
    fn = bind(IO.OutCollect, KO=2, FIRST=False)
    why = door_verdicts(IF.Circuit("short", ((fn, 1),)))[IF.descriptor_id(fn)]
    assert why is not None and ROW_FLOOR in why, why
