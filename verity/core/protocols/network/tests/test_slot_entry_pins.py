"""`period_case.SLOT_ENTRY_PINS` and `SELECT_ENTRY_PINS` against `io_shell.SlotEntry_v1` and `SelectEntry_v1` at their
circuit-check rows (`ROWS` and `SELECT_ROWS`, bound in `io_shell.circuit-check.py`): each row's closure digest
(`period_case.slot_entry_digest`, over its population of one member) is its pin, so a change to either body, or to any
Definition it reaches, fails here until it re-pins. Run as a script, it prints each row's descriptor id and digest:

    uv run python verity/core/protocols/network/tests/test_slot_entry_pins.py
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import period_case as PC
from verity.ml.circuits import codec, population
from verity.ml.definitions.registry import io_shell as X

#: the sidecar's `SlotEntry_v1` rows the pins are of, in `SLOT_ENTRY_PINS`' order; `SlotEntry_v1:4` is no pin, since at
#: KO 4 a slot's entry is a `SelectEntry_v1` unit's
ROWS = ("SlotEntry_v1:2", "SlotEntry_v1:64")
#: the sidecar's `SelectEntry_v1` rows the pins are of, in `SELECT_ENTRY_PINS`' order
SELECT_ROWS = ("SelectEntry_v1:2", "SelectEntry_v1:4", "SelectEntry_v1:64")
SCRIPT = "verity/core/protocols/network/tests/test_slot_entry_pins.py"
PERIOD = Path(__file__).resolve().parents[1] / "lean" / "NetworkCheck" / "Period.lean"


def digests(rows: tuple[str, ...] = ROWS) -> dict[str, tuple[str, str]]:
    """Each of ``rows``: its descriptor id and closure digest, from the sidecar's binding of its Definition."""
    path = Path(X.__file__).with_name("io_shell.circuit-check.py")
    spec = importlib.util.spec_from_file_location("io_shell_sidecar", path)
    sidecar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sidecar)
    bound = dict(sidecar._model_roots())
    out = {}
    for r in rows:
        d = bound[r]()
        key = codec.descriptor_id(d)
        out[r] = key, PC.slot_entry_digest(codec.encode_program(population.population_program(d, 1)), key)
    return out


def _pinned(rows: tuple[str, ...], pins: str, what: str) -> None:
    got = digests(rows)
    now = ", ".join(f"{r} {key} {d[:16]}…" for r, (key, d) in got.items())
    assert tuple(d for _, d in got.values()) == getattr(PC, pins), (
        f"io_shell.{what}'s closure digests are not {pins} ({now}). If the change to {what}, or to a Definition it "
        f"reaches, is meant, re-pin: set {pins} in verity/core/protocols/network/tests/period_case.py and its Lean copy "
        f"in NetworkCheck/Period.lean to the digests `uv run python {SCRIPT}` prints, in its order, and the descriptor "
        f"ids in their comments to the ids it prints")


def test_the_pins_are_io_shells_slot_entry_at_its_rows():
    _pinned(ROWS, "SLOT_ENTRY_PINS", "SlotEntry_v1")


def test_the_select_pins_are_io_shells_select_entry_at_its_rows():
    _pinned(SELECT_ROWS, "SELECT_ENTRY_PINS", "SelectEntry_v1")


def test_leans_pins_are_the_mirrors():
    """`slotEntryPins` and `selectEntryPins` in `NetworkCheck/Period.lean` are the mirror's lists, digest for digest."""
    text = re.sub(r"\\\n\s*", "", PERIOD.read_text())
    for lean, pins in (("slotEntryPins", "SLOT_ENTRY_PINS"), ("selectEntryPins", "SELECT_ENTRY_PINS")):
        m = re.search(rf"^def {lean} : List String :=\s*\[(.*?)\]", text, re.M | re.S)
        assert m, f"{PERIOD.name} has no `def {lean} : List String := [...]`"
        assert tuple(re.findall(r'"([0-9a-f]+)"', m.group(1))) == getattr(PC, pins), (
            f"{PERIOD.name}'s {lean} is not period_case.{pins}: set both to the digests `uv run python {SCRIPT}` prints")


if __name__ == "__main__":
    for r, (key, d) in {**digests(ROWS), **digests(SELECT_ROWS)}.items():
        print(r, key, d)
