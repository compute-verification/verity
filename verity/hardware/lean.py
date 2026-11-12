"""The Lean card tables, generated from the card files.

A GPU card file that names a Lean table (``lean``, the module's path) gives that module: the namespace
``Verity.Accel.<the file's name>`` with its ``clock``, its ``tensorClock`` where the card's tensor has a clock of its own,
and its ``card : Card``. ``table`` is the table's numbers as exact rationals, and ``render`` is the module's text.

The text's shape is fixed, since the Lean check's record of a definition is a digest of the term Lean elaborates from
it, so a number written another way, or an arm moved, is a changed record though no number changed:

- a clock, and device memory's rate, is its millions times ``10 ^ 6``; every other rate is ``SMs * the entry *`` the name
  of its clock;
- a per-SM store is ``SMs * the entry``; a store in bytes is the bytes in binary units (``binary``) over the bytes of a
  word, and ``text`` the same over an instruction's bytes;
- a field's arms are in the card file's order, and constructors whose arms read the same share the first one's
  (``arms``); ``weight``'s arms are the formats whose weight is not 1, in ``FORMATS``' order, and the rest are its
  default.

Lean's ℕ division truncates, so ``render`` writes one only where it divides exactly and refuses the card file where it
does not; a fraction is written ``p / q`` in ℚ≥0, where the division is exact.

    uv run python -m verity.hardware.lean            name each generated module its card file no longer gives; exit 1 if any
    uv run python -m verity.hardware.lean --write    write them

A card file that also names a PoUW record (``lean_prices``) gives that module too: ``Prices.<the table's name>Loop``,
PoUW's coarse record at the card (``prices``, ``W1Prices.toPrices``' rule with the add at ``price.fp32_loop``), written
``⟨add, bf16, ⟨fs, qa⟩⟩`` (``render_prices``).

``modules`` is every generated module by its path, which ``--write`` writes (``write``; ``stale`` names the ones to
write) and ``tests/test_lean.py`` holds the committed files to, byte for byte. Another kind of record generated from
the card files is one more ``render`` function and its line in ``modules``; ``header``, ``numeral``, ``binary`` and
``arms`` serve both. Stdlib only, and it imports no other unit.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from math import ceil
from pathlib import Path

from verity.errors import InvalidArtifact
from verity.hardware.cards import CARDS, WORD, load, number, rate, skus

#: The Lean module that declares ``Verity.Accel.Card``, which a card table imports.
CARD = "Definitions.Core.Accel.Card"
#: The Lean module that declares PoUW's ``Prices``, which a card's PoUW record imports.
DEVICE = "Definitions.Pouw.PearlC.Device"
#: ``weight``'s arms, in order: the formats that have one where their weight is not 1. A format this doesn't name goes
#: after these, in the card file's order. An arm's place is part of the term (a match's alternatives are positional).
FORMATS = ("nvf4", "mxf4", "f16", "bf16", "tf32", "f64", "b1")
#: The one command that writes the generated modules: each module's header names it, and so does the test's failure.
COMMAND = "uv run python -m verity.hardware.lean --write"


def root() -> Path:
    """The repository's root, which a card file's ``lean`` is a path from: the nearest directory above this file that
    holds the tree the first table is in."""
    top = Path(load(tabled()[0])["lean"]).parts[0]
    for parent in Path(__file__).resolve().parents:
        if (parent / top).is_dir():
            return parent
    raise FileNotFoundError(f"no directory above {__file__} holds {top}/: the generated modules belong to a checkout")


def tabled() -> list[str]:
    """The GPU cards that name a Lean table."""
    return [sku for sku in skus("gpu") if "lean" in load(sku)]


def table(card: dict | str) -> tuple[dict[str, Fraction], dict[str, dict[str, Fraction] | Fraction]]:
    """The Lean table a card (or the card of a SKU) gives: its clocks in Hz (``tensorClock`` the card's ``clock.tensor``,
    its ``clock.sm`` where it has none), each store's words, each resource's units a second
    (``verity.hardware.cards.rate``, so the arms hold the one calculator), each format's weight, and the model's ranges."""
    card = load(card) if isinstance(card, str) else card
    sms, clock = number(card["sm"]["count"]), card["clock"]
    clocks = {"clock": number(clock["sm"]) * 10 ** 6, "tensorClock": number(clock.get("tensor", clock["sm"])) * 10 ** 6}
    store = card["store"]
    size = {"reg": sms * (number(store["reg"]) + number(store["reg_extra"])),
            "smem": sms * number(store["smem"]) / WORD, "l2": number(store["l2"]) / WORD,
            "dram": number(store["dram"]) / WORD}
    rho = {r: rate(card, r) for r in [*card["rate"], "dram"]}
    return clocks, {"size": size, "ρ": rho, "weight": {f: number(e) for f, e in card["weight"].items()},
                    "exact": number(card["model"]["exact"]),
                    "text": number(store["dram"]) / number(card["model"]["instruction_bytes"])}


def numeral(q: Fraction, *, nat: str | None = None) -> str:
    """``q`` as a Lean literal: a numeral, or ``p / q`` in lowest terms for ℚ≥0, whose division is exact. ``nat`` names
    a number that goes in a field of ℕ, where a fraction is refused: ℕ's division would truncate it."""
    if q < 0:
        raise InvalidArtifact(f"{nat or 'a number of the table'} is {q}: a card's numbers are not negative")
    if q.denominator == 1:
        return str(q.numerator)
    if nat:
        raise InvalidArtifact(f"{nat} is {q}, not a whole number, and Lean's ℕ would truncate it: fix the card file's entry")
    return f"{q.numerator} / {q.denominator}"


def binary(q: Fraction, *, nat: str) -> str:
    """A whole number of bytes ``q`` in binary units: ``m * 2 ^ k`` at the largest of 2^30, 2^20 and 2^10 that divides it,
    and one numeral where none does. ``nat`` names the number, as ``numeral``'s."""
    text = numeral(q, nat=nat)
    for k in (30, 20, 10):
        if q and q % 2 ** k == 0:
            return f"{q.numerator >> k} * 2 ^ {k}"
    return text


def arms(pairs: list[tuple[str, str]], default: str | None = None) -> list[str]:
    """A matched field's arms, from each constructor and the text of its arm, in order: constructors whose texts are the
    same share the first one's arm, and ``default`` is the arm of every constructor not among them."""
    shared: dict[str, list[str]] = {}
    for kind, text in pairs:
        shared.setdefault(text, []).append(kind)
    out = [f"    | {' | '.join(f'.{kind}' for kind in kinds)} => {text}" for text, kinds in shared.items()]
    return out + ([f"    | _ => {default}"] if default is not None else [])


def header(sku: str) -> str:
    """The comment a generated module opens with: what wrote it, from which card file, and that nobody edits it."""
    source = (CARDS / f"{sku}.toml").resolve().relative_to(root()).as_posix()
    return f"/-! Generated by `{COMMAND}` from\n`{source}`; do not edit. -/"


def render(card: dict | str) -> str:
    """The Lean module of a card (or of a SKU's card) that names a Lean table: ``clock``, ``tensorClock`` where the
    card's ``clock.tensor`` is not its ``clock.sm``, and ``card``, in the shape the module's docstring gives."""
    card = load(card) if isinstance(card, str) else card
    sku = card["sku"]
    if "lean" not in card:
        raise KeyError(f"{sku} names no Lean table: its card file has no `lean`")
    name = Path(card["lean"]).stem
    clocks, fields = table(card)
    sms, store = number(card["sm"]["count"]), card["store"]
    own = clocks["tensorClock"] != clocks["clock"]

    def whole(q: Fraction, key: str) -> str:
        return numeral(q, nat=f"{sku}: {key}")

    def factor(q: Fraction) -> str:
        text = numeral(q)
        return f"({text})" if "/" in text else text

    def millions(q: Fraction) -> str:
        return f"{factor(q / 10 ** 6)} * 10 ^ 6"

    def over(key: str, unit: Fraction, named: str) -> str:
        size = number(store[key])
        whole(size / unit, named)
        return f"{binary(size, nat=f'{sku}: store.{key}')} / {whole(unit, f'what {named} divides by')}"

    def resource(r: str) -> str:
        if r not in card["rate"]:
            return millions(fields["ρ"][r])
        tick = "tensorClock" if r == "tensor" and own else "clock"
        return f"{factor(sms)} * {factor(number(card['rate'][r]))} * {tick}"

    count = whole(sms, "sm.count")
    size = [("reg", f"{count} * ({whole(number(store['reg']), 'store.reg')} + "
                    f"{whole(number(store['reg_extra']), 'store.reg_extra')})"),
            ("smem", f"{count} * {whole(number(store['smem']) / WORD, 'store.smem in words')}"),
            *((s, over(s, Fraction(WORD), f"store.{s} in words")) for s in ("l2", "dram"))]
    weights = fields["weight"]
    listed = sorted((f for f in weights if weights[f] != 1),
                    key=lambda f: FORMATS.index(f) if f in FORMATS else len(FORMATS))
    others = "every resource's clock but the tensor's" if own else "every resource's clock"
    return "\n".join([
        "module", "", f"public import {CARD}", "", header(sku), "", "@[expose] public section", "",
        "open scoped NNRat", "", f"namespace Verity.Accel.{name}", "",
        f"/-- The card file's `clock.sm` in Hz: {others}. -/", f"def clock : ℚ≥0 := {millions(clocks['clock'])}", "",
        *(["/-- The card file's `clock.tensor` in Hz: the tensor's clock. -/",
           f"def tensorClock : ℚ≥0 := {millions(clocks['tensorClock'])}", ""] if own else []),
        f"/-- The table of the card file `{sku}.toml`. -/", "def card : Card where",
        "  size", *arms(size), "  size_pos := by intro s; cases s <;> decide",
        "  ρ", *arms([(r, resource(r)) for r in fields["ρ"]]),
        "  weight", *arms([(f, numeral(weights[f])) for f in listed], "1" if len(listed) < len(weights) else None),
        f"  exact := {whole(fields['exact'], 'model.exact')}",
        "  text := " + over("dram", number(card["model"]["instruction_bytes"]), "store.dram over model.instruction_bytes"),
        "",
        f"end Verity.Accel.{name}", ""])


def prices(card: dict | str) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """PoUW's coarse record at a card (or the card of a SKU), in E4M3 MACs: the FP32 add in the timed loop
    (``price.fp32_loop``), the BF16 MAC, the credited forming per element (the k32 noise atom's 32 MACs and half a
    two-code cast, a cast an ALU lane's share of the tensor rate, rounded up) and the forming that reads the activations
    alone (one FFMA on every 8th element at the loop's add, rounded up): ``W1Prices.toPrices``' rule. Rounding up raises
    ``W_ref`` and not the credit, so the gap is never below its value at the unrounded prices."""
    card = load(card) if isinstance(card, str) else card
    weight, r = card["weight"], card["rate"]
    unit = number(weight["e4m3"])
    add = number(card["price"]["fp32_loop"]) / unit
    cast = number(r["tensor"]) / number(r["alu"]) / unit
    return add, number(weight["bf16"]) / unit, Fraction(ceil(32 + cast / 2)), Fraction(ceil(add / 8))


def render_prices(card: dict | str) -> str:
    """The Lean module of a card's PoUW record (``lean_prices``): ``Prices.<the table's name>Loop``, ``prices``' four
    numbers as ``⟨add, bf16, ⟨fs, qa⟩⟩``, the term the hand-written record had."""
    card = load(card) if isinstance(card, str) else card
    sku = card["sku"]
    if "lean_prices" not in card:
        raise KeyError(f"{sku} names no PoUW record: its card file has no `lean_prices`")
    stem = Path(card["lean"]).stem
    name = stem[0].lower() + stem[1:]
    add, bf16, fs, qa = prices(card)
    term = f"⟨{numeral(add)}, {numeral(bf16)}, ⟨{numeral(fs, nat=f'{sku}: fs')}, {numeral(qa, nat=f'{sku}: qa')}⟩⟩"
    return "\n".join([
        "module", "", f"public import {DEVICE}", "", header(sku), "", "@[expose] public section", "",
        "namespace Pouw.PearlC", "",
        f"/-- PoUW's coarse record of the card file `{sku}.toml`, in `E4M3` MACs",
        f"(`verity.hardware.lean.prices`): an `FP32` add in the timed loop `{numeral(add)}`, a `BF16` MAC `{numeral(bf16)}`,",
        f"the credited forming `{numeral(fs)}` and the forming that reads the activations alone `{numeral(qa)}` per element.",
        "-/",
        f"def Prices.{name}Loop : Prices := {term}", "",
        "end Pouw.PearlC", ""])


def modules() -> dict[str, str]:
    """Every generated Lean module: its path from the repository's root, and its text."""
    return {load(sku)["lean"]: render(sku) for sku in tabled()} | {
        load(sku)["lean_prices"]: render_prices(sku) for sku in tabled() if "lean_prices" in load(sku)}


def stale(top: Path | None = None) -> list[str]:
    """The generated modules that are missing under ``top`` (the repository's root), or are not ``modules``' text there."""
    top = top or root()
    return [path for path, text in modules().items()
            if not (top / path).is_file() or (top / path).read_bytes() != text.encode("utf-8")]


def write(top: Path | None = None) -> list[str]:
    """Write each stale module under ``top`` (the repository's root), and return their paths."""
    top, texts = top or root(), modules()
    out = stale(top)
    for path in out:
        (top / path).parent.mkdir(parents=True, exist_ok=True)
        (top / path).write_bytes(texts[path].encode("utf-8"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The Lean card tables, generated from the card files.")
    ap.add_argument("--write", action="store_true", help="write each generated module; without it, name each one its "
                    "card file no longer gives and exit 1 if there is one")
    a = ap.parse_args(argv)
    if a.write:
        for path in write():
            print(f"wrote {path}")
        return 0
    found = stale()
    for path in found:
        print(f"{path} is not what its card file gives; write it: `{COMMAND}`")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
