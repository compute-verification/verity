"""A unit class of any partition as a ``verity/flock-circuit`` statement (`verity_flock.class_statement`): on circuit-check's tiny
Serve_v1 under `Q_word`, two table-free classes compose, their laid-out unit rows agree with the reference on synthetic lanes, and,
when ``FLOCK_CIRCUIT`` names a built ``flock-circuit`` binary, a loopback session accepts each."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytest.importorskip("verity.ir.units")
pytest.importorskip("verity_vllm.program.registry")
from circuit_check import targets as TG  # noqa: E402
from verity.ir.defs import bind  # noqa: E402
from verity.ir.program import Program  # noqa: E402

from verity_flock import circuit as FC  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity_flock import partition_units as PU  # noqa: E402


@pytest.fixture(scope="module")
def units():
    TG.load_registries()
    from verity_vllm.program.registry import b1

    CS.cache_slot_circuits()
    return PU.partition_units(Program(bind(b1.Serve, C=TG.TINY, LP=3, STEPS=2)), "q-word")


def test_table_free_classes_compose_and_prove(units, tmp_path: Path):
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": "test", "seed": 0})
    picked = []
    for s, cl in sorted(units.classes().items(), key=lambda kv: len(kv[1].representative.gates)):
        cc = PU.lower_class(units.activation(cl.definition), cl.representative)
        if cc.circuit is not None and "lookup" not in cc.circuit.kind:
            picked.append((s, cl))
        if len(picked) == 2:
            break
    binary = os.environ.get("FLOCK_CIRCUIT")
    for i, (s, cl) in enumerate(picked):
        lanes = PU.lanes(units.activation(cl.definition), [cl.representative], key, 8)[(cl.representative.definition, cl.representative.unit)]
        d = tmp_path / s[:16]
        st = CS.stage(units, "q-word", s, cl, lanes, d, hashlib.sha256(b"test").digest())
        text = (d / "circuit.txt").read_text()
        assert st["n"] == 8 and st["k_log"] <= FC.K_MAX and st["circuit_sha512"] == FC.digest(text)
        assert '"rule":"verity/partition/v1"' in text                      # the Q_word partition's own digest, not a provisional cover
        if binary and Path(binary).exists():
            live = CS.prove(binary, d, st["n"], 7690 + i)
            assert live["accepted"] is True and live["proof_bytes"]
