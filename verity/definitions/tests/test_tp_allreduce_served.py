"""tp_allreduce's sidecar binds GLM-5.3 TP8's all-reduces at served statics (`_glm53_served_roots`), and circuit-check's
catalog keeps a binding over `MAX_GATES` only while a pin is held for it. So each such binding holds a ``digests`` pin equal to
its circuit's digest, and no other is held: a binding without one would drop out of ``--all`` unseen, and a stale one fails
there."""
from __future__ import annotations

import importlib.util
import json
from functools import cache
from pathlib import Path

from verity.circuits.codec import program_digest
from verity.circuits.program import Program
from verity.definitions.registry import tp_allreduce as TA

SIDECAR = Path(TA.__file__).with_name("tp_allreduce.circuit-check.py")
PINS = Path(TA.__file__).with_name("tp_allreduce.circuit-check.json")
#: circuit-check's `targets.MAX_GATES`: a specialisation over it is checked only when its family has none smaller or it is pinned
MAX_GATES = 150_000


@cache
def _sidecar():
    spec = importlib.util.spec_from_file_location("tp_allreduce_circuit_check", SIDECAR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _served() -> list:
    mod = _sidecar()
    return mod._glm53_served_roots() + mod._glm53_multimem_alone_roots()


def test_the_served_bindings_are_every_mnnvl_call_and_the_multimem_rows():
    mod = _sidecar()
    ids = [fn.id for fn in _served()]
    assert len(set(ids)) == len(ids)
    mnnvl = [i for i in ids if i.startswith("AllReduceMnnvl_v1{")]
    assert len(mnnvl) == 10 + 42 and "AllReduceMnnvl_v1{WORLD=8,N=61440,SHOT=1}" in mnnvl
    assert "AllReduceMnnvl_v1{WORLD=8,N=67584,SHOT=1}" not in mnnvl and "AllReduceMnnvl_v1{WORLD=8,N=258048,SHOT=2}" in mnnvl
    assert len(ids) - len(mnnvl) == len(mod.GLM53_MULTIMEM_ROWS)
    alone = mod._glm53_multimem_alone_roots()
    assert sorted(fn.statics["SIZE"] for fn in alone) == sorted(rows * mod.GLM53_N for rows in mod.GLM53_MULTIMEM_ALONE)


def test_every_served_binding_over_max_gates_holds_its_digest_pin():
    pins = json.loads(PINS.read_text()).get("digests", {}) if PINS.exists() else {}
    want = {fn.id: {"sha512": program_digest(Program(fn))} for fn in _served() if fn.gates > MAX_GATES}
    wrong = {k: v for k, v in want.items() if pins.get(k) != v}
    assert not wrong, (f"set these in {PINS.name}'s \"digests\" (json indent 1, keys sorted), and review why any changed: "
                       + json.dumps(wrong, indent=1, sort_keys=True))
    assert sorted(set(pins) - set(want)) == [], f"drop these from {PINS.name}'s \"digests\": no served binding holds them"
