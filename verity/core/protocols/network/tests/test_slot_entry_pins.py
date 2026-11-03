"""`period_case.SLOT_ENTRY_PINS` against `io_shell.SlotEntry_v1` at its circuit-check rows (`ROWS`, bound in
`io_shell.circuit-check.py`): each row's closure digest (`period_case.slot_entry_digest`, over its population of one
member) is its pin, so a change to `SlotEntry_v1`'s body, or to any Definition it reaches, fails here until it re-pins.
Run as a script, it prints each row's descriptor id and digest:

    uv run python verity/core/protocols/network/tests/test_slot_entry_pins.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import period_case as PC
from verity.ml.circuits import codec, population
from verity.ml.definitions.registry import io_shell as X

#: the sidecar's rows the pins are of, in `SLOT_ENTRY_PINS`' order
ROWS = ("SlotEntry_v1:2", "SlotEntry_v1:64")
SCRIPT = "verity/core/protocols/network/tests/test_slot_entry_pins.py"


def digests() -> dict[str, tuple[str, str]]:
    """Each of `ROWS`: its descriptor id and closure digest, from the sidecar's binding of `io_shell.SlotEntry`."""
    path = Path(X.__file__).with_name("io_shell.circuit-check.py")
    spec = importlib.util.spec_from_file_location("io_shell_sidecar", path)
    sidecar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sidecar)
    rows = dict(sidecar._model_roots())
    out = {}
    for r in ROWS:
        d = rows[r]()
        key = codec.descriptor_id(d)
        out[r] = key, PC.slot_entry_digest(codec.encode_program(population.population_program(d, 1)), key)
    return out


def test_the_pins_are_io_shells_slot_entry_at_its_rows():
    got = digests()
    now = ", ".join(f"{r} {key} {d[:16]}…" for r, (key, d) in got.items())
    assert tuple(d for _, d in got.values()) == PC.SLOT_ENTRY_PINS, (
        f"io_shell.SlotEntry_v1's closure digests are not SLOT_ENTRY_PINS ({now}). If the change to SlotEntry_v1, or "
        f"to a Definition it reaches, is meant, re-pin: set SLOT_ENTRY_PINS in verity/core/protocols/network/tests/"
        f"period_case.py to the digests `uv run python {SCRIPT}` prints, in its order, and the descriptor ids in its "
        f"comment to the ids it prints")


if __name__ == "__main__":
    for r, (key, d) in digests().items():
        print(r, key, d)
