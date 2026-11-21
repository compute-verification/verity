"""The plan's row floor (`window_builder.check_served`'s ``row_floor``) is the door's own refusal: for each template of a
served window, its first unit's block is lowered by Lean (`one_circuit.lower`, ``flock-lower-check compose``), and
``compose`` refuses it for no compression before its last block exactly where the plan names the template. A template
the door refuses before ``compose`` (not admitted, or refused by the lowering) is not compared, and none is named."""
from __future__ import annotations

import json
import shutil

import pytest
from verity.circuits import partition_object as PO
from verity.circuits.codec import descriptor_chunks
from verity.commitments.periods import interface as IF
from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
from verity.protocols.proof.spot.verifier.one_stage import proofs as PF

from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, GRAIN, WHOLE
from tests.program.predict.test_window_served_check import _window
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
def test_the_plans_row_floor_is_composes_refusal_on_every_template():
    """Some minutes: a Lean build of ``flock-lower-check`` and one lowering a template of the tiny window."""
    circuit = _window(S._registered(), CLASS, *WHOLE, GRAIN)
    named = WC.check_served(circuit)["row_floor"]
    door = door_verdicts(circuit)
    assert set(door) == {IF.descriptor_id(fn) for fn, _ in circuit.templates}
    compared = {t for t, why in door.items() if why is None or ROW_FLOOR in why}
    before = {t: door[t][-200:] for t in set(door) - compared}
    print(f"ROW-FLOOR compared {len(compared)} of {len(door)} templates; refused before compose: {before}")
    assert named and set(named) <= compared, f"named, but the door refuses them before compose: {set(named) - compared}"
    refused = {t for t in compared if door[t] is not None}
    assert set(named) == refused, (f"named by the plan, composed by Lean: {sorted(set(named) - refused)}; "
                                   f"refused by Lean, not named: {sorted(refused - set(named))}")
    padded = {t for t in door if t.startswith(("OutCollect_v2{", "SlotEntry_v2{"))}
    assert len(padded) == 3 and all(door[t] is None for t in padded), {t: door[t] for t in padded}
