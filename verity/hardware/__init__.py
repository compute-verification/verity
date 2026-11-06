"""The hardware cards: one TOML file per SKU (``cards/<sku>.toml``), the one source of each card's numbers.

Every number in a card file is an entry ``{value, unit, mark, source}``. ``mark`` is ``ceiling`` (no program on that
SKU and firmware exceeds it: what the model's premises need), ``measurement`` (a measured value, so only a lower bound
on the ceiling) or ``placeholder`` (neither: open). ``source`` names a data sheet, a document, a run id or an ``art:`` id.
The Lean card tables (``Security/Definitions/Core/Accel``) are held to these files number by number
(``verity/hardware/tests/test_lean.py``). Stdlib only, and it imports no other unit.

~~~python
from verity.hardware import load
card = load("h200-sxm-141gb")
card["clock"]["sm"]["value"]       # 1980 (MHz)
card["memory"]["bandwidth"]        # {'value': 4916736000000.0, 'unit': 'B/s', 'mark': 'ceiling', 'source': ...}
~~~
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from verity.errors import InvalidArtifact

CARDS = Path(__file__).parent / "cards"
MARKS = ("ceiling", "measurement", "placeholder")
ENTRY = frozenset({"value", "unit", "mark", "source"})


def skus() -> list[str]:
    """The SKUs with a card file."""
    return sorted(p.stem for p in CARDS.glob("*.toml"))


def load(sku: str) -> dict:
    """The card file of ``sku`` as parsed TOML, refused unless every number in it carries its unit, mark and source."""
    path = CARDS / f"{sku}.toml"
    if not path.is_file():
        raise KeyError(f"no card {sku!r}: the cards are {skus()} (verity/hardware/cards/<sku>.toml)")
    card = tomllib.loads(path.read_text(encoding="utf-8"))
    if card.get("sku") != sku:
        raise InvalidArtifact(f"{path.name}: sku = {card.get('sku')!r}; set it to the file's name, {sku!r}")
    _check(card, path.name)
    return card


def _check(table: dict, where: str) -> None:
    for key, item in table.items():
        at = f"{where}: {key}"
        if isinstance(item, dict) and "value" in item:
            _check_entry(item, at)
        elif isinstance(item, dict):
            _check(item, at)
        elif isinstance(item, (int, float)):
            raise InvalidArtifact(f"{at} is a bare number; write it as {{ value, unit, mark, source }}")


def _check_entry(entry: dict, at: str) -> None:
    if set(entry) != ENTRY:
        raise InvalidArtifact(f"{at} has keys {sorted(entry)}; an entry has exactly {sorted(ENTRY)}")
    if isinstance(entry["value"], bool) or not isinstance(entry["value"], (int, float)):
        raise InvalidArtifact(f"{at}: value {entry['value']!r} is not a number")
    if entry["mark"] not in MARKS:
        raise InvalidArtifact(f"{at}: mark {entry['mark']!r}; mark it one of {MARKS}")
    for field in ("unit", "source"):
        if not isinstance(entry[field], str) or not entry[field].strip():
            raise InvalidArtifact(f"{at}: give it a {field}")
