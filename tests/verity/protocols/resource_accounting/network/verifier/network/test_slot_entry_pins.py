"""`period_case.SLOT_ENTRY_PINS`, `SELECT_ENTRY_PINS` and `SCAN_SLOTS_PINS` against `io_shell.SlotEntry_v1`,
`io_shell.SelectEntry_v1` and `boolean_io_shell.ScanSlots_v3` at their circuit-check rows (`ROWS`, `SELECT_ROWS` and
`SCAN_ROWS`, bound in `io_shell.circuit-check.py` and `boolean_io_shell.circuit-check.py`): each row's closure digest
(`period_case.slot_entry_digest`, over its population of one member) is its pin, so a change to a body, or to any
Definition it reaches, fails here until it re-pins; and the period cases' fixtures (`FIXTURES`: `fixtures/slot_entry.json`,
`fixtures/select_entry.json` and `fixtures/scan_slots.json`) are the canonical descriptors of that population at each
one's row. Run as a script, it prints each row's descriptor id and digest and writes the fixtures:

    uv run python tests/verity/protocols/resource_accounting/network/verifier/network/test_slot_entry_pins.py
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import period_case as PC
from verity.circuits import codec, population
from verity_vllm.definitions.registry import boolean_io_shell as BX, io_shell as X

#: the sidecar's `SlotEntry_v1` rows the pins are of, in `SLOT_ENTRY_PINS`' order; `SlotEntry_v1:4` is no pin, since at
#: KO 4 a slot's entry is a `SelectEntry_v1` unit's
ROWS = ("SlotEntry_v1:2", "SlotEntry_v1:64")
#: the sidecar's `SelectEntry_v1` rows the pins are of, in `SELECT_ENTRY_PINS`' order
SELECT_ROWS = ("SelectEntry_v1:2", "SelectEntry_v1:4", "SelectEntry_v1:64")
#: the sidecar's `ScanSlots_v3` rows the pins are of, in `SCAN_SLOTS_PINS`' order
SCAN_ROWS = ("ScanSlots_v3:2:2:4", "ScanSlots_v3:4:64:1024")
#: each fixture of the period cases (`period_case.slot_entry`, `select_entry` and `scan_slots`) and the row whose
#: population of one member's canonical descriptor it is
FIXTURES = {"slot_entry.json": "SlotEntry_v1:2", "select_entry.json": "SelectEntry_v1:2",
            "scan_slots.json": "ScanSlots_v3:2:2:4"}
SCRIPT = "tests/verity/protocols/resource_accounting/network/verifier/network/test_slot_entry_pins.py"
PERIOD = Path(__file__).resolve().parents[7] / "verity" / "protocols" / "resource_accounting" / "network" / "verifier" / "lean" / "NetworkVerifier" / "Period" / "Decode.lean"


def descriptors(rows: tuple[str, ...]) -> dict[str, tuple[str, dict]]:
    """Each of ``rows``: its descriptor id and the descriptor of its population of one member, from its sidecar's
    binding of its Definition."""
    bound = {}
    for m in (X, BX):
        name = Path(m.__file__).stem
        spec = importlib.util.spec_from_file_location(f"{name}_sidecar", Path(m.__file__).with_name(f"{name}.circuit-check.py"))
        sidecar = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sidecar)
        bound |= dict(sidecar._model_roots())
    return {r: (codec.descriptor_id(d := bound[r]()), codec.encode_program(population.population_program(d, 1)))
            for r in rows}


def digests(rows: tuple[str, ...] = ROWS) -> dict[str, tuple[str, str]]:
    """Each of ``rows``: its descriptor id and closure digest."""
    return {r: (key, PC.slot_entry_digest(desc, key)) for r, (key, desc) in descriptors(rows).items()}


def fixtures() -> dict[str, str]:
    """Each fixture's name and its text, the canonical descriptor of its row's population of one member."""
    got = descriptors(tuple(FIXTURES.values()))
    return {name: PC.D.canonical(got[r][1]) for name, r in FIXTURES.items()}


def _pinned(rows: tuple[str, ...], pins: str, what: str) -> None:
    got = digests(rows)
    now = ", ".join(f"{r} {key} {d[:16]}…" for r, (key, d) in got.items())
    assert tuple(d for _, d in got.values()) == getattr(PC, pins), (
        f"io_shell.{what}'s closure digests are not {pins} ({now}). If the change to {what}, or to a Definition it "
        f"reaches, is meant, re-pin: set {pins} in tests/verity/protocols/resource_accounting/network/verifier/network/period_case.py and its Lean copy "
        f"in NetworkVerifier/Period/Decode.lean to the digests `uv run python {SCRIPT}` prints, in its order, and the descriptor "
        f"ids in their comments to the ids it prints")


def test_the_pins_are_io_shells_slot_entry_at_its_rows():
    _pinned(ROWS, "SLOT_ENTRY_PINS", "SlotEntry_v1")


def test_the_select_pins_are_io_shells_select_entry_at_its_rows():
    _pinned(SELECT_ROWS, "SELECT_ENTRY_PINS", "SelectEntry_v1")


def test_the_scan_pins_are_boolean_io_shells_scan_slots_at_its_rows():
    _pinned(SCAN_ROWS, "SCAN_SLOTS_PINS", "ScanSlots_v3")


def test_the_fixtures_are_io_shells_entries_at_their_rows():
    for name, text in fixtures().items():
        assert (Path(PC.__file__).parent / "fixtures" / name).read_text() == text, (
            f"fixtures/{name} is not the descriptor of {FIXTURES[name]}'s population of one member: write it with "
            f"`uv run python {SCRIPT}`")


def test_leans_pins_are_the_mirrors():
    """`slotEntryPins`, `selectEntryPins` and `scanSlotsPins` in `NetworkCheck/Period/Decode.lean` are the mirror's lists,
    digest for digest."""
    text = re.sub(r"\\\n\s*", "", PERIOD.read_text())
    for lean, pins in (("slotEntryPins", "SLOT_ENTRY_PINS"), ("selectEntryPins", "SELECT_ENTRY_PINS"),
                       ("scanSlotsPins", "SCAN_SLOTS_PINS")):
        m = re.search(rf"^def {lean} : List String :=\s*\[(.*?)\]", text, re.M | re.S)
        assert m, f"{PERIOD.name} has no `def {lean} : List String := [...]`"
        assert tuple(re.findall(r'"([0-9a-f]+)"', m.group(1))) == getattr(PC, pins), (
            f"{PERIOD.name}'s {lean} is not period_case.{pins}: set both to the digests `uv run python {SCRIPT}` prints")


if __name__ == "__main__":
    for r, (key, d) in {**digests(ROWS), **digests(SELECT_ROWS), **digests(SCAN_ROWS)}.items():
        print(r, key, d)
    for name, text in fixtures().items():
        (Path(PC.__file__).parent / "fixtures" / name).write_text(text)
        print("wrote", f"fixtures/{name}")
