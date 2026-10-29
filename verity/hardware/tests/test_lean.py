"""The Lean card tables agree with the card files, number by number.

A GPU card file names its Lean table (``lean``). Its ``clock``, ``tensorClock`` (``clock`` where the table has none) and
``card`` definitions are evaluated as exact rationals (``^`` a power, ``/`` exact, and a natural-number field refused
where Lean's division would truncate) and each arm is
compared with what the card file gives, for every store, resource and format the ACCEL ops are priced in
(``Security/Definitions/Core/Accel/Isa.lean``): a constructor added there fails here until the card files carry it.
The B200 has no card table yet, and its data sheet (``Accounting.TS.Sheets.B200``) is held to its card file instead.
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
SHEET = ROOT / "Security/Definitions/Accounting/Sheets/B200.lean"
LEAN = [sku for sku in skus("gpu") if "lean" in load(sku)]
WORD = 4  # bytes in a 32-bit word, the unit of every store and of device memory's rate
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def constructors(name: str) -> list[str]:
    """The constructors of the inductive ``name`` in ``Isa.lean``."""
    found = re.search(rf"^inductive {name}\n((?:  \|.*\n)+)", ISA.read_text(encoding="utf-8"), re.M)
    assert found, f"Isa.lean declares no inductive {name}"
    return re.findall(r"\|\s*(\w+)", found.group(1))


def evaluate(expr: str, clocks: dict[str, Fraction] | None = None, nat: bool = False) -> Fraction:
    """A Lean arithmetic literal (numerals, ``+ - * / ^``, parentheses, the names in ``clocks``) as an exact rational.
    With ``nat``, a division that doesn't divide exactly is refused, since Lean's ℕ division truncates."""
    def go(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node, ast.Name) and node.id in (clocks or {}):
            return clocks[node.id]
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


def card_table(text: str) -> tuple[dict[str, Fraction], dict[str, dict[str, str] | str]]:
    """The Lean table's clocks (``clock``, and ``tensorClock``, ``clock`` where it has none) and its ``card``'s fields:
    a matched field's arms (pattern to expression, ``_`` the default) or a plain field's expression."""
    clock = re.search(r"^def clock : ℚ≥0 := (.+)$", text, re.M)
    tensor = re.search(r"^def tensorClock : ℚ≥0 := (.+)$", text, re.M)
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
    hz = evaluate(clock.group(1))
    return {"clock": hz, "tensorClock": evaluate(tensor.group(1)) if tensor else hz}, fields


def number(entry: dict) -> Fraction:
    return Fraction(str(entry["value"]))


def expected(card: dict) -> tuple[dict[str, Fraction], dict[str, dict[str, Fraction] | Fraction]]:
    """The Lean table the card file gives: its clocks in Hz, each store's words, each resource's units a second (the
    tensor's at the tensor's clock), each format's weight, and the model's ranges."""
    sms = number(card["sm"]["count"])
    clocks = {"clock": number(card["clock"]["sm"]) * 10 ** 6, "tensorClock": number(card["clock"]["tensor"]) * 10 ** 6}
    store, rate = card["store"], card["rate"]
    size = {"reg": sms * (number(store["reg"]) + number(store["reg_extra"])),
            "smem": sms * number(store["smem"]) / WORD, "l2": number(store["l2"]) / WORD,
            "dram": number(store["dram"]) / WORD}
    rho = ({r: sms * number(rate[r]) * clocks["tensorClock" if r == "tensor" else "clock"] for r in rate}
           | {"dram": number(card["memory"]["bandwidth"]) / WORD})
    return clocks, {"size": size, "ρ": rho, "weight": {f: number(e) for f, e in card["weight"].items()},
                    "exact": number(card["model"]["exact"]),
                    "text": number(store["dram"]) / number(card["model"]["instruction_bytes"])}


def disagreements(card: dict, text: str) -> list[str]:
    """Every number on which the Lean table ``text`` and the card file differ, and every constructor either lacks."""
    clock, fields = card_table(text)
    want_clock, want = expected(card)
    out = [f"{name}: Lean {clock[name]}, card file {want_clock[name]}" for name in want_clock
           if clock[name] != want_clock[name]]
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


@pytest.mark.parametrize("sku", LEAN)
def test_lean_table_agrees_with_the_card_file(sku: str) -> None:
    card = load(sku)
    found = disagreements(card, (ROOT / card["lean"]).read_text(encoding="utf-8"))
    assert not found, (f"{card['lean']} and verity/hardware/cards/{sku}.toml disagree; change the card file first, "
                       "then its Lean table to match:\n" + "\n".join(found))


def test_the_gpu_cards_without_a_lean_table() -> None:
    assert sorted(set(skus("gpu")) - set(LEAN)) == ["b200-sxm-180gb"]


def test_the_b200_sheet_agrees_with_its_card_file() -> None:
    """``Accounting.TS.Sheets.B200`` states the B200 card's clock, SMs and pipe widths, and each format's peak as
    ``rate.tensor`` over that format's weight. The sheet's implied and FP64 clocks, its FP6 peak and its uniform pipe
    have no key in the card file."""
    card = load("b200-sxm-180gb")
    sheet = dict(re.findall(r"^def (\w+) : ℝ := (.+)$", SHEET.read_text(encoding="utf-8"), re.M))
    keys = {"fmaxMHz": ("clock", "sm"), "smCount": ("sm", "count"), "widthAlu": ("rate", "alu"),
            "widthImad": ("rate", "imad"), "widthShfl": ("rate", "shfl")}
    peaks = {"peakS8": "s8", "peakE4m3": "e4m3", "peakE5m2": "e5m2", "peakF16": "f16", "peakBf16": "bf16",
             "peakTf32": "tf32", "peakFp4": "nvf4", "peakFp4_f8f6f4": "e2m1", "peakF64": "f64"}
    want = {name: number(card[section][key]) for name, (section, key) in keys.items()}
    want |= {name: number(card["rate"]["tensor"]) / number(card["weight"][fmt]) for name, fmt in peaks.items()}
    assert {name: evaluate(sheet[name]) for name in want} == want


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


def test_the_tensor_runs_at_its_own_clock() -> None:
    """The H200's table prices the tensor at its 1830 MHz tensor clock and every other resource at the 1980 MHz SM
    clock, and a tensor arm at the SM clock is a disagreement."""
    card = load("h200-sxm-141gb")
    text = (ROOT / card["lean"]).read_text(encoding="utf-8")
    clocks, _ = card_table(text)
    assert clocks == {"clock": 1980 * 10 ** 6, "tensorClock": 1830 * 10 ** 6}
    assert expected(card)[1]["ρ"]["tensor"] == 132 * 4096 * 1830 * 10 ** 6
    at_sm = text.replace("132 * 4096 * tensorClock", "132 * 4096 * clock")
    assert disagreements(card, at_sm) == [f"ρ tensor: Lean 132 * 4096 * clock = {132 * 4096 * 1980 * 10 ** 6}, "
                                          f"card file {132 * 4096 * 1830 * 10 ** 6}"]


def test_a_truncating_nat_division_is_refused() -> None:
    assert evaluate("50 * 2 ^ 20 / 4", nat=True) == 50 * 2 ** 18
    with pytest.raises(AssertionError, match="truncates"):
        evaluate("141 * 2 ^ 30 / 7", nat=True)
    assert evaluate("1 / 8") == Fraction(1, 8)
