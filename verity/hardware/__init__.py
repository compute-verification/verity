"""The hardware cards: one TOML file per SKU (``cards/<sku>.toml``), the one source of each card's numbers.

A card is a GPU's (``kind = "gpu"``) or a host's (``kind = "host"``: a server's CPUs and DRAM, apart from the GPUs it
carries). Every number in a card file is an entry ``{value, unit, mark, source}``. ``mark`` is ``ceiling`` (no program
on that SKU and firmware exceeds it: what the model's premises need), ``measurement`` (a measured value, so only a lower
bound on the ceiling) or ``placeholder`` (neither: open). ``source`` names a data sheet, a document or an ``art:`` id
(a run's record), and a node by its kind, never its host name. The Lean card tables (``Security/Definitions/Core/Accel``)
are held to the GPU cards that name one (``lean``), number by number (``verity/hardware/tests/test_lean.py``). Stdlib
only, and it imports no other unit.

``rate`` is the one calculator of a GPU's peak rates, Lean's ``card.ρ``: a reader that needs the tensor peak at some
clock calls it rather than multiplying the card's numbers itself.

~~~python
from verity.hardware import load, rate, skus
skus("host")                       # ['xeon-6776p-2s', 'xeon-platinum-8468-2s']
card = load("h200-sxm-141gb")
card["clock"]["sm"]["value"]       # 1980 (MHz)
card["memory"]["bandwidth"]        # {'value': 4916736000000.0, 'unit': 'B/s', 'mark': 'ceiling', 'source': ...}
rate(card)                         # Fraction(989429760000000): s8 MACs a second, 132 SMs x 4096 x clock.tensor 1830 MHz
rate(card, fmt="f16", mhz=1700)    # the f16 MACs a second at a 1700 MHz clock
~~~
"""

from __future__ import annotations

import tomllib
from fractions import Fraction
from pathlib import Path

from verity.errors import InvalidArtifact

CARDS = Path(__file__).parent / "cards"
KINDS = ("gpu", "host")
MARKS = ("ceiling", "measurement", "placeholder")
ENTRY = frozenset({"value", "unit", "mark", "source"})
WORD = 4  # bytes in a 32-bit word, the unit of every store and of device memory's rate


def skus(kind: str | None = None) -> list[str]:
    """The SKUs with a card file, or those of one ``kind``."""
    every = sorted(p.stem for p in CARDS.glob("*.toml"))
    return every if kind is None else [sku for sku in every if load(sku)["kind"] == kind]


def load(sku: str) -> dict:
    """The card file of ``sku`` as parsed TOML, refused unless every number in it carries its unit, mark and source."""
    path = CARDS / f"{sku}.toml"
    if not path.is_file():
        raise KeyError(f"no card {sku!r}: the cards are {skus()} (verity/hardware/cards/<sku>.toml)")
    card = tomllib.loads(path.read_text(encoding="utf-8"))
    if card.get("sku") != sku:
        raise InvalidArtifact(f"{path.name}: sku = {card.get('sku')!r}; set it to the file's name, {sku!r}")
    if card.get("kind") not in KINDS:
        raise InvalidArtifact(f"{path.name}: kind = {card.get('kind')!r}; set it to one of {KINDS}")
    _check(card, path.name)
    return card


def number(entry: dict) -> Fraction:
    """A card entry's value as an exact rational, read from its decimal text (``1.597568e12`` is that number exactly)."""
    return Fraction(str(entry["value"]))


def rate(card: dict, resource: str = "tensor", *, fmt: str | None = None, mhz: float | None = None) -> Fraction:
    """Lean's ``card.ρ resource``: the units a second of ``resource`` on the whole GPU, ``sm.count × rate.<resource> ×``
    its clock in Hz (``clock.tensor`` for the tensor and ``clock.sm`` for the rest, unless ``mhz`` names the clock), and
    for device memory (``dram``) ``memory.bandwidth`` in 32-bit words. With ``fmt``, the tensor rate in MACs of that
    operand format (tensor units over ``weight.<fmt>``). ``tests/test_lean.py`` holds it to every Lean table's arms."""
    if card["kind"] != "gpu":
        raise ValueError(f"{card['sku']} is a {card['kind']} card; rate is a GPU's")
    if resource == "dram":
        if fmt is not None or mhz is not None:
            raise ValueError("device memory's rate is memory.bandwidth: no SM clock or operand format scales it")
        return number(card["memory"]["bandwidth"]) / WORD
    if resource not in card["rate"]:
        raise KeyError(f"{card['sku']} has no rate.{resource}: its rates are {sorted(card['rate'])} and dram")
    if fmt is not None and (resource != "tensor" or fmt not in card["weight"]):
        raise KeyError(f"{fmt!r} weighs tensor MACs only, and {card['sku']}'s formats are {sorted(card['weight'])}")
    clock = number(card["clock"]["tensor" if resource == "tensor" else "sm"]) if mhz is None else Fraction(str(mhz))
    units = number(card["sm"]["count"]) * number(card["rate"][resource]) * clock * 10 ** 6
    return units if fmt is None else units / number(card["weight"][fmt])


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
