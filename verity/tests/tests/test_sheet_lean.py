"""``Verity.Accel.H200.sheetCard`` (``security/Proofs/Core/Accel/Sheet.lean``), the H200's figures handed to the honest
lemma as data, held field by field to its card file, ``hardware/cards/h200-sxm-141gb.toml``."""

from __future__ import annotations

import ast
import operator
import re
import tomllib
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SHEET = ROOT / "security/Proofs/Core/Accel/Sheet.lean"
CARD = ROOT / "hardware/cards/h200-sxm-141gb.toml"
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
WORD = 4
CLOCKS = {"sheetClock": ("clock", "sm"), "sheetTensorClock": ("clock", "tensor")}
UNITS = {("sm", "count"): "SM", ("clock", "sm"): "MHz", ("clock", "tensor"): "MHz", ("store", "reg"): "word/SM",
         ("store", "reg_extra"): "word/SM", ("store", "smem"): "B/SM", ("store", "l2"): "B",
         ("store", "dram_installed"): "B", ("memory", "bandwidth"): "B/s", ("model", "exact"): "bit",
         ("model", "instruction_bytes"): "B/instruction"}
PER_SM = {("size", "reg"), ("size", "smem")} | {("ρ", r) for r in
                                                ("tensor", "issue", "alu", "imad", "f64", "shfl", "smem", "sfu", "l2")}


def load_card() -> dict:
    """The card file as parsed TOML. Read here rather than through ``verity.device``, a trusted place no new file
    loads (``tests/import_order.toml``)."""
    return tomllib.loads(CARD.read_text(encoding="utf-8"))


def number(entry: dict) -> Fraction:
    """A card entry's value as an exact rational, read from its decimal text, as ``verity.device.number`` reads it."""
    return Fraction(str(entry["value"]))


def parse(expr: str) -> ast.AST:
    """A Lean arithmetic literal as a Python expression's tree."""
    return ast.parse(expr.replace("^", "**"), mode="eval").body


def evaluate(expr: str, names: dict[str, Fraction], nat: bool = False) -> Fraction:
    """A Lean arithmetic literal (numerals, ``+ - * / ^``, parentheses, the names in ``names``) as an exact rational.
    With ``nat``, a division that doesn't divide exactly is refused, since Lean's ℕ division truncates."""
    def go(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exp = go(node.left), go(node.right)
            assert exp.denominator == 1, f"{expr}: a non-integer exponent"
            return base ** int(exp)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            out = OPS[type(node.op)](go(node.left), go(node.right))
            assert not (nat and isinstance(node.op, ast.Div) and out.denominator != 1), (
                f"{expr}: ℕ division truncates here; write the field so it divides exactly")
            return out
        raise AssertionError(f"{expr}: not a literal this test evaluates ({ast.dump(node)})")
    return go(parse(expr))


def first_factor(expr: str) -> Fraction | None:
    """The leftmost factor of a product, when it is a numeral."""
    node = parse(expr)
    while isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        node = node.left
    return Fraction(node.value) if isinstance(node, ast.Constant) and isinstance(node.value, int) else None


def sheet(text: str) -> tuple[dict[str, str], dict[str, dict[str, str] | str]]:
    """``Sheet.lean``'s two clocks and ``sheetCard``'s fields, comments dropped: a matched field's arms (constructor to
    expression, ``_`` the default) or a plain field's expression. Proof fields (``:= by``) are skipped."""
    clocks = {name: m.group(1) for name in CLOCKS
              if (m := re.search(rf"^def {name} : ℚ≥0 := (.+)$", text, re.M))}
    body = re.search(r"^def sheetCard : Card where\n(.*?)\n\n", text, re.M | re.S)
    assert body, "Sheet.lean has no `def sheetCard : Card where …`"
    fields: dict[str, dict[str, str] | str] = {}
    current: dict[str, str] = {}
    for raw in body.group(1).splitlines():
        line = raw.split(" --")[0].rstrip()
        if re.fullmatch(r"  \S+ := by .*", line):
            continue
        if m := re.fullmatch(r"  (\S+) := (.+)", line):
            fields[m.group(1)] = m.group(2)
        elif m := re.fullmatch(r"  (\S+)", line):
            current = fields.setdefault(m.group(1), {})
        elif m := re.fullmatch(r"    \|(.+?)=> (.+)", line):
            for pattern in m.group(1).split("|"):
                current[pattern.strip().lstrip(".")] = m.group(2)
        else:
            raise AssertionError(f"a line of `sheetCard` this test doesn't read: {raw!r}")
    return clocks, fields


def expected(card: dict) -> tuple[dict[str, Fraction], dict[str, dict[str, Fraction] | Fraction]]:
    """What each of ``sheetCard``'s fields is to be, from the card file's keys its comments name: sizes in 32-bit
    words, rates a second, device memory the memory on the board (``store.dram_installed``)."""
    def n(section: str, key: str) -> Fraction:
        return number(card[section][key])

    sms, sm_hz, tensor_hz = n("sm", "count"), n("clock", "sm") * 10 ** 6, n("clock", "tensor") * 10 ** 6
    clocks = {name: n(*key) * 10 ** 6 for name, key in CLOCKS.items()}
    size = {"reg": sms * (n("store", "reg") + n("store", "reg_extra")), "smem": sms * n("store", "smem") / WORD,
            "l2": n("store", "l2") / WORD, "dram": n("store", "dram_installed") / WORD}
    rho = {r: sms * n("rate", r) * (tensor_hz if r == "tensor" else sm_hz)
           for r in ("tensor", "issue", "alu", "imad", "f64", "shfl", "smem", "sfu", "l2")}
    rho["dram"] = n("memory", "bandwidth") / WORD
    weight = {fmt: number(entry) for fmt, entry in card["weight"].items()}
    return clocks, {"size": size, "ρ": rho, "weight": weight, "exact": n("model", "exact"),
                    "text": n("store", "dram_installed") / n("model", "instruction_bytes")}


def disagreements(card: dict, text: str) -> list[str]:
    """Every field on which ``sheetCard`` (in ``text``) and the card file differ, the SM count each per-SM arm opens
    with among them."""
    lean_clocks, fields = sheet(text)
    want_clocks, want = expected(card)
    out = [f"{name}: Lean has no `def {name} : ℚ≥0`" for name in want_clocks if name not in lean_clocks]
    names = {name: evaluate(e, {}) for name, e in lean_clocks.items()}
    out += [f"{name}: Lean {names[name]}, card file {want_clocks[name]}" for name in want_clocks
            if name in names and names[name] != want_clocks[name]]
    sms = number(card["sm"]["count"])
    for field, nat in (("size", True), ("ρ", False), ("weight", False)):
        arms = fields.get(field)
        if not isinstance(arms, dict):
            out.append(f"{field}: Lean has no matched field")
            continue
        extra = sorted(set(arms) - set(want[field]) - {"_"})
        out += [f"{field}: Lean's arms {extra} have no key in the card file"] if extra else []
        for name, value in want[field].items():
            arm = arms.get(name, arms.get("_"))
            if arm is None:
                out.append(f"{field} {name}: Lean has no arm")
                continue
            if (got := evaluate(arm, names, nat)) != value:
                out.append(f"{field} {name}: Lean {arm} = {got}, card file {value}")
            if (field, name) in PER_SM and first_factor(arm) != sms:
                out.append(f"{field} {name}: Lean {arm} doesn't open with the SM count, card file {sms}")
    for field in ("exact", "text"):
        expr = fields.get(field)
        got = None if not isinstance(expr, str) else evaluate(expr, names, nat=True)
        out += [] if got == want[field] else [f"{field}: Lean {expr} = {got}, card file {want[field]}"]
    return out


def test_sheet_card_agrees_with_the_card_file() -> None:
    """Both clocks, the four sizes, the ten rates, the twelve weights, ``exact`` and the code size, each the value the
    card file's keys give, and the SM count each per-SM size and rate opens with."""
    found = disagreements(load_card(), SHEET.read_text(encoding="utf-8"))
    assert not found, ("Sheet.lean's sheetCard and hardware/cards/h200-sxm-141gb.toml disagree; change the one that's "
                       "wrong\n" + "\n".join(found))


def test_the_card_file_units_the_sheet_reads() -> None:
    """``GemmFigures.weight_s8`` (``weight .s8 = 1``) rests on the card file's tensor unit being one ``s8`` MAC: its
    ``rate.tensor`` counts ``s8`` MACs and ``weight.s8`` is one tensor unit a MAC. The other keys' units are the ones
    the conversions above assume (bytes over 4 to words, MHz to Hz)."""
    card = load_card()
    assert card["rate"]["tensor"]["unit"] == "s8 MAC/SM-clock"
    assert card["weight"]["s8"]["unit"] == "tensor unit/MAC" and number(card["weight"]["s8"]) == 1
    assert {key: card[key[0]][key[1]]["unit"] for key in UNITS} == UNITS
    assert all(card["rate"][r]["unit"].endswith("/SM-clock") for r in card["rate"])
    _, fields = sheet(SHEET.read_text(encoding="utf-8"))
    arms = fields["weight"]
    assert evaluate(arms.get("s8", arms.get("_")), {}) == 1
