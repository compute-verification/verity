"""The Lean card tables agree with the card files, number by number.

Each card file names its Lean table (``lean``). Its ``clock`` and ``card`` definitions are evaluated as exact rationals
(``^`` a power, ``/`` exact, and a natural-number field refused where Lean's division would truncate) and each arm is
compared with what the card file gives, for every store, resource and format the ACCEL ops are priced in
(``Security/Definitions/Core/Accel/Isa.lean``): a constructor added there fails here until the card files carry it.
"""

from __future__ import annotations

import ast
import operator
import re
from fractions import Fraction
from pathlib import Path

import pytest

from verity.hardware import load, skus

ROOT = Path(__file__).resolve().parents[3]
ISA = ROOT / "Security/Definitions/Core/Accel/Isa.lean"
WORD = 4  # bytes in a 32-bit word, the unit of every store and of device memory's rate
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def constructors(name: str) -> list[str]:
    """The constructors of the inductive ``name`` in ``Isa.lean``."""
    found = re.search(rf"^inductive {name}\n((?:  \|.*\n)+)", ISA.read_text(encoding="utf-8"), re.M)
    assert found, f"Isa.lean declares no inductive {name}"
    return re.findall(r"\|\s*(\w+)", found.group(1))


def evaluate(expr: str, clock: Fraction | None = None, nat: bool = False) -> Fraction:
    """A Lean arithmetic literal (numerals, ``+ - * / ^``, parentheses, ``clock``) as an exact rational. With ``nat``,
    a division that doesn't divide exactly is refused, since Lean's ℕ division truncates."""
    def go(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node, ast.Name) and node.id == "clock" and clock is not None:
            return clock
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exp = go(node.left), go(node.right)
            assert exp.denominator == 1, f"{expr}: a non-integer exponent"
            return base ** int(exp)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = go(node.left), go(node.right)
            out = OPS[type(node.op)](left, right)
            assert not (nat and isinstance(node.op, ast.Div) and out.denominator != 1), (
                f"{expr}: ℕ division truncates here; write the field so it divides exactly")
            return out
        raise AssertionError(f"{expr}: not a literal this test evaluates ({ast.dump(node)})")
    return go(ast.parse(expr.replace("^", "**"), mode="eval").body)


def card_table(text: str) -> tuple[Fraction, dict[str, dict[str, str] | str]]:
    """The Lean table's ``clock`` and its ``card``'s fields: a matched field's arms (pattern to expression, ``_`` the
    default) or a plain field's expression."""
    clock = re.search(r"^def clock : ℚ≥0 := (.+)$", text, re.M)
    body = re.search(r"^def card : Card where\n(.*?)\n\nend ", text, re.M | re.S)
    assert clock and body, "the Lean table has no `def clock : ℚ≥0 := …` and `def card : Card where …`"
    fields: dict[str, dict[str, str] | str] = {}
    current = None
    for line in body.group(1).splitlines():
        if m := re.fullmatch(r"  (\S+) := (.+)", line):
            fields[m.group(1)] = m.group(2)
        elif m := re.fullmatch(r"  (\S+)", line):
            current = fields.setdefault(m.group(1), {})
        elif m := re.fullmatch(r"    \|(.+?)=> (.+)", line):
            for pattern in m.group(1).split("|"):
                current[pattern.strip().lstrip(".")] = m.group(2)
        else:
            raise AssertionError(f"a line of `card` this test doesn't read: {line!r}")
    return evaluate(clock.group(1)), fields


def number(entry: dict) -> Fraction:
    return Fraction(str(entry["value"]))


def expected(card: dict) -> tuple[Fraction, dict[str, dict[str, Fraction] | Fraction]]:
    """The Lean table the card file gives: its clock in Hz, each store's words, each resource's units a second, each
    format's weight, and the model's ranges."""
    sms, clock = number(card["sm"]["count"]), number(card["clock"]["sm"]) * 10 ** 6
    store, rate = card["store"], card["rate"]
    size = {"reg": sms * (number(store["reg"]) + number(store["reg_extra"])),
            "smem": sms * number(store["smem"]) / WORD, "l2": number(store["l2"]) / WORD,
            "dram": number(store["dram"]) / WORD}
    rho = {r: sms * number(rate[r]) * clock for r in rate} | {"dram": number(card["memory"]["bandwidth"]) / WORD}
    return clock, {"size": size, "ρ": rho, "weight": {f: number(e) for f, e in card["weight"].items()},
                   "exact": number(card["model"]["exact"]),
                   "text": number(store["dram"]) / number(card["model"]["instruction_bytes"])}


def disagreements(card: dict, text: str) -> list[str]:
    """Every number on which the Lean table ``text`` and the card file differ, and every constructor either lacks."""
    clock, fields = card_table(text)
    want_clock, want = expected(card)
    out = [] if clock == want_clock else [f"clock: Lean {clock}, card file {want_clock}"]
    for field, kinds, nat in (("size", "Store", True), ("ρ", "Res", False), ("weight", "Fmt", False)):
        arms, names = fields.get(field, {}), constructors(kinds)
        unknown = sorted(set(arms) - set(names) - {"_"})
        out += [f"{field}: Lean's arms {unknown} are no {kinds} constructor"] if unknown else []
        missing, extra = sorted(set(names) - set(want[field])), sorted(set(want[field]) - set(names))
        out += [f"{field}: the card file lacks {kinds} {missing}"] if missing else []
        out += [f"{field}: the card file's {extra} are no {kinds} constructor"] if extra else []
        for name in names:
            arm = arms.get(name, arms.get("_"))
            if arm is None:
                out.append(f"{field} {name}: Lean has no arm")
            elif name in want[field] and (got := evaluate(arm, clock, nat)) != want[field][name]:
                out.append(f"{field} {name}: Lean {arm} = {got}, card file {want[field][name]}")
    for field in ("exact", "text"):
        got = evaluate(fields.get(field, "0"), clock, nat=True)
        out += [] if got == want[field] else [f"{field}: Lean {fields.get(field)} = {got}, card file {want[field]}"]
    return out


@pytest.mark.parametrize("sku", skus())
def test_lean_table_agrees_with_the_card_file(sku: str) -> None:
    card = load(sku)
    found = disagreements(card, (ROOT / card["lean"]).read_text(encoding="utf-8"))
    assert not found, (f"{card['lean']} and verity/hardware/cards/{sku}.toml disagree; change the card file first, "
                       "then its Lean table to match:\n" + "\n".join(found))


def test_a_changed_number_is_a_disagreement() -> None:
    """The check above isn't vacuous: one changed rate, a changed default weight and a dropped arm are each reported."""
    card = load("rtx-pro-6000-server")
    text = (ROOT / card["lean"]).read_text(encoding="utf-8")
    assert disagreements(card, text) == []
    rate = text.replace("188 * 1024 * clock", "188 * 1025 * clock")
    assert disagreements(card, rate) == [f"ρ tensor: Lean 188 * 1025 * clock = {188 * 1025 * 2430 * 10 ** 6}, "
                                         f"card file {188 * 1024 * 2430 * 10 ** 6}"]
    default = text.replace("    | _ => 1\n", "    | _ => 2\n")
    assert sorted(m.split(":")[0] for m in disagreements(card, default)) == [
        f"weight {f}" for f in ("b1", "e2m1", "e4m3", "e5m2", "s8", "u8")]
    dropped = text.replace("    | .l2 => 188 * 5 * clock\n", "")
    assert disagreements(card, dropped) == ["ρ l2: Lean has no arm"]


def test_a_truncating_nat_division_is_refused() -> None:
    assert evaluate("50 * 2 ^ 20 / 4", nat=True) == 50 * 2 ** 18
    with pytest.raises(AssertionError, match="truncates"):
        evaluate("141 * 2 ^ 30 / 7", nat=True)
    assert evaluate("1 / 8") == Fraction(1, 8)
