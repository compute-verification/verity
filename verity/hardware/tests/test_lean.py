"""The Lean card tables are the generator's output, and agree with the card files number by number.

A GPU card file names its Lean table (``lean``), which ``verity.hardware.lean`` writes: the committed module is held to
``render`` byte for byte. Independently of the generator, its ``clock``, ``tensorClock`` (``clock`` where the table has
none) and ``card`` definitions are evaluated as exact rationals (``^`` a power, ``/`` exact, and a natural-number field
refused where Lean's division would truncate) and each arm is compared with what the card file gives
(``verity.hardware.lean.table``), for every store, resource and format the ACCEL ops are priced in
(``security/Definitions/Core/Accel/Isa.lean``): a constructor added there fails here until the card files carry it. The
B200 has no card table yet, and its data sheet (``Accounting.TS.Sheets.B200``) is held to its card file instead.
"""

from __future__ import annotations

import ast
import copy
import operator
import re
from fractions import Fraction
from pathlib import Path

import pytest

from verity.errors import InvalidArtifact
from verity.hardware import lean
from verity.hardware.cards import load, number, skus
from verity.hardware.lean import table

ROOT = Path(__file__).resolve().parents[3]
ISA = ROOT / "security/Definitions/Core/Accel/Isa.lean"
SHEET = ROOT / "security/Definitions/Accounting/Sheets/B200.lean"
LEAN = lean.tabled()
PRICED = [sku for sku in LEAN if "lean_prices" in load(sku)]
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


def disagreements(card: dict, text: str) -> list[str]:
    """Every number on which the Lean table ``text`` and the card file differ, and every constructor either lacks."""
    clock, fields = card_table(text)
    want_clock, want = table(card)
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
    assert not found, (f"{card['lean']} and verity/hardware/cards/{sku}.toml disagree; change the card file or the "
                       f"generator, then write the table: `{lean.COMMAND}`\n" + "\n".join(found))


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
    faster = text.replace("188 * 1024 * clock", "188 * 1025 * clock")
    assert disagreements(card, faster) == [(f"ρ tensor: Lean 188 * 1025 * clock = {188 * 1025 * 2430 * 10 ** 6}, "
                                          f"card file {188 * 1024 * 2430 * 10 ** 6}")]
    default = text.replace("    | _ => 1\n", "    | _ => 2\n")
    assert sorted(m.split(":")[0] for m in disagreements(card, default)) == [
        f"weight {f}" for f in ("b1", "e2m1", "e4m3", "e5m2", "s8", "u8")]
    dropped = text.replace("    | .l2 => 188 * 5 * clock\n", "")
    assert disagreements(card, dropped) == ["ρ l2: Lean has no arm"]


def test_the_tensor_runs_at_its_own_clock_where_the_card_has_one() -> None:
    """The H200's card file and its Lean table price the tensor at the 1830 MHz ``clock.tensor`` and every other resource
    at the 1980 MHz SM clock, and a tensor arm at the SM clock disagrees. The same card without ``clock.tensor`` prices
    the tensor at the SM clock, and disagrees with that table on the clock and the tensor arm."""
    card = load("h200-sxm-141gb")
    text = (ROOT / card["lean"]).read_text(encoding="utf-8")
    assert card_table(text)[0] == table(card)[0] == {"clock": 1980 * 10 ** 6, "tensorClock": 1830 * 10 ** 6}
    assert table(card)[1]["ρ"]["tensor"] == 132 * 4096 * 1830 * 10 ** 6
    at_sm = text.replace("132 * 4096 * tensorClock", "132 * 4096 * clock")
    assert disagreements(card, at_sm) == [(f"ρ tensor: Lean 132 * 4096 * clock = {132 * 4096 * 1980 * 10 ** 6}, "
                                           f"card file {132 * 4096 * 1830 * 10 ** 6}")]
    untimed = copy.deepcopy(card)
    del untimed["clock"]["tensor"]
    assert table(untimed)[0] == {"clock": 1980 * 10 ** 6, "tensorClock": 1980 * 10 ** 6}
    assert disagreements(untimed, text) == [
        f"tensorClock: Lean {1830 * 10 ** 6}, card file {1980 * 10 ** 6}",
        f"ρ tensor: Lean 132 * 4096 * tensorClock = {132 * 4096 * 1830 * 10 ** 6}, card file {132 * 4096 * 1980 * 10 ** 6}"]


def test_a_truncating_nat_division_is_refused() -> None:
    assert evaluate("50 * 2 ^ 20 / 4", nat=True) == 50 * 2 ** 18
    with pytest.raises(AssertionError, match="truncates"):
        evaluate("141 * 2 ^ 30 / 7", nat=True)
    assert evaluate("1 / 8") == Fraction(1, 8)


GENERATED = lean.modules()
#: What each table declares, its namespace first: the names its importers read.
DECLARED = {"h200-sxm-141gb": ["Verity.Accel.H200", "clock", "tensorClock", "card"],
            "rtx-pro-6000-server": ["Verity.Accel.Sm120", "clock", "card"]}


@pytest.mark.parametrize("path", GENERATED)
def test_a_generated_table_is_the_generators_output(path: str) -> None:
    """Byte for byte: nobody edits a generated module, and a changed card file or generator is a rewritten one."""
    file = ROOT / path
    assert file.is_file(), f"{path} is missing; write it: `{lean.COMMAND}`"
    assert file.read_bytes() == GENERATED[path].encode("utf-8"), (
        f"{path} is not what `verity.hardware.lean.render` gives for its card file. It is generated: change the card "
        f"file or the generator, never this file, then write it: `{lean.COMMAND}`")


def test_a_generated_table_declares_the_names_its_importers_read() -> None:
    """Each card's table is the module its card file names, a module of the definitions tree, and declares its
    namespace, its clocks and ``card``: a renamed or dropped one fails here before a Lean build does."""
    assert sorted(GENERATED) == sorted([load(sku)["lean"] for sku in LEAN] + [load(sku)["lean_prices"] for sku in PRICED])
    assert all(path.startswith("security/Definitions/") for path in GENERATED)
    found = {sku: re.findall(r"^(?:namespace|def|abbrev|theorem|lemma|instance|structure) (\S+)", lean.render(sku), re.M)
             for sku in LEAN}
    assert found == DECLARED


def test_a_generated_table_states_its_source_and_no_number_in_prose() -> None:
    """A generated module opens by naming the command that wrote it and its card file, and says not to edit it. Its
    comments name the card file's keys and restate none of its numbers."""
    for sku, text in [(sku, lean.render(sku)) for sku in LEAN] + [(sku, lean.render_prices(sku)) for sku in PRICED]:
        opening = text.split("@[expose] public section")[0]
        assert lean.COMMAND in opening and f"verity/hardware/cards/{sku}.toml" in opening and "do not edit" in opening
        comments = re.findall(r"/-[-!].*?-/", text, re.S)
        assert len(comments) >= 2
        assert not re.search(r"\d", re.sub(r"`[^`]*`", "", "\n".join(comments))), f"{sku}: a comment states a number"


def test_a_generated_table_follows_its_card_file() -> None:
    """The generator isn't a copy of today's tables: a changed entry of the card file is a changed arm, which the old
    card file disagrees with; a fractional rate or weight is written ``p / q``; and a card without ``clock.tensor``
    gets no ``tensorClock`` and a tensor at the SM clock."""
    was = load("h200-sxm-141gb")
    card = copy.deepcopy(was)
    card["rate"]["issue"]["value"], card["rate"]["sfu"]["value"], card["weight"]["s8"]["value"] = 5, "0.5", "0.75"
    text = lean.render(card)
    for line in ("    | .issue => 132 * 5 * clock", "    | .sfu => 132 * (1 / 2) * clock", "    | .s8 => 3 / 4"):
        assert f"\n{line}\n" in text
    assert disagreements(card, text) == []
    assert [m.split(":")[0] for m in disagreements(was, text)] == ["ρ issue", "ρ sfu", "weight s8"]
    del card["clock"]["tensor"]
    untimed = lean.render(card)
    assert "tensorClock" not in untimed and "\n    | .tensor => 132 * 4096 * clock\n" in untimed
    assert disagreements(card, untimed) == []


def test_a_generated_table_has_one_shape() -> None:
    """The Lean check's record of ``card`` digests the term, so the text's shape is fixed: bytes in the largest binary
    unit that divides them, arms in the card file's order with equal ones sharing the first's, and ``weight``'s arms
    in ``FORMATS``' order (a format it doesn't name after them) before the default, which every weight of 1 takes."""
    assert [lean.binary(Fraction(n), nat="n") for n in (50 * 2 ** 20, 2 ** 27, 141 * 2 ** 30, 3 * 2 ** 10, 1000)] == [
        "50 * 2 ^ 20", "128 * 2 ^ 20", "141 * 2 ^ 30", "3 * 2 ^ 10", "1000"]
    assert lean.arms([("a", "1"), ("b", "2"), ("c", "1")], "0") == ["    | .a | .c => 1", "    | .b => 2", "    | _ => 0"]
    card = copy.deepcopy(load("h200-sxm-141gb"))
    card["weight"]["u8"]["value"], card["weight"]["nvf4"]["value"], card["weight"]["s8"]["value"] = 3, "0.5", 3
    text = lean.render(card)
    assert text.split("  weight\n")[1].split("  exact")[0].splitlines() == [
        "    | .nvf4 => 1 / 2", "    | .f16 | .bf16 => 2", "    | .tf32 => 4", "    | .f64 => 59 / 2", "    | .b1 => 1 / 8",
        "    | .s8 | .u8 => 3", "    | _ => 1"]
    assert text.split("  ρ\n")[1].split("  weight")[0].splitlines() == [
        "    | .tensor => 132 * 4096 * tensorClock", "    | .issue => 132 * 4 * clock",
        "    | .alu | .imad | .f64 => 132 * 64 * clock", "    | .shfl | .smem => 132 * 32 * clock",
        "    | .sfu => 132 * 16 * clock", "    | .l2 => 132 * 512 * clock", "    | .dram => 1229184 * 10 ^ 6"]
    assert disagreements(card, text) == []
    for weight in card["weight"].values():
        weight["value"] = 2
    assert "| _ =>" not in lean.render(card)


@pytest.mark.parametrize("section, key, value, named", [
    ("store", "l2", 50 * 2 ** 20 + 2, "store.l2 in words"), ("store", "smem", 262146, "store.smem in words"),
    ("store", "reg", "65536.5", "store.reg"), ("sm", "count", "131.5", "sm.count"),
    ("model", "instruction_bytes", 7, "store.dram over model.instruction_bytes"), ("model", "exact", "30.5", "model.exact")])
def test_a_generated_table_has_no_truncating_division(section: str, key: str, value: object, named: str) -> None:
    """A field of ℕ in a generated table divides only where the division is exact, as ``evaluate`` reads it, since
    Lean's ℕ division truncates; a card file whose entry would leave a fraction there is refused, naming the entry."""
    for sku in LEAN:
        _, fields = card_table(lean.render(sku))
        divided = [arm for arm in [*fields["size"].values(), fields["exact"], fields["text"]] if "/" in arm]
        assert divided and all(evaluate(arm, nat=True).denominator == 1 for arm in divided)
    card = copy.deepcopy(load("h200-sxm-141gb"))
    card[section][key]["value"] = value
    with pytest.raises(InvalidArtifact, match=f"{re.escape(named)} is .*not a whole number"):
        lean.render(card)


def test_the_command_writes_the_generated_tables(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The committed modules are current, so the command without ``--write`` exits 0 and names none. Under a root that
    has none, every module is stale; ``write`` writes each, byte for byte, and a second ``write`` has nothing to do. An
    edited module is stale again."""
    assert lean.stale() == [] and lean.main([]) == 0 and capsys.readouterr().out == ""
    assert lean.stale(tmp_path) == list(GENERATED)
    assert lean.write(tmp_path) == list(GENERATED) and lean.stale(tmp_path) == [] and lean.write(tmp_path) == []
    assert all((tmp_path / path).read_bytes() == text.encode("utf-8") for path, text in GENERATED.items())
    edited = next(iter(GENERATED))
    (tmp_path / edited).write_text(GENERATED[edited].replace("* clock", "* clock * 2", 1), encoding="utf-8")
    assert lean.stale(tmp_path) == [edited]


def test_a_pouw_record_is_the_coarse_rule_at_its_card_file() -> None:
    """The RTX PRO 6000's PoUW record is ``W1Prices.toPrices``' rule at its card file, written as the hand-written record
    was, and the rule follows the card: at the issue-bound add, 8, it gives the issue-bound record ``Prices.sm120``
    (``⟨8, 2, ⟨40, 1⟩⟩``), whose A-only forming rounds down to 1 where the loop's rounds up to 2."""
    assert PRICED == ["rtx-pro-6000-server"]
    assert lean.prices("rtx-pro-6000-server") == (Fraction(1047, 125), 2, 40, 2)
    assert "\ndef Prices.sm120Loop : Prices := ⟨1047 / 125, 2, ⟨40, 2⟩⟩\n" in lean.render_prices("rtx-pro-6000-server")
    card = copy.deepcopy(load("rtx-pro-6000-server"))
    card["price"]["fp32_loop"]["value"] = 8
    assert "\ndef Prices.sm120Loop : Prices := ⟨8, 2, ⟨40, 1⟩⟩\n" in lean.render_prices(card)
    del card["lean_prices"]
    with pytest.raises(KeyError, match="lean_prices"):
        lean.render_prices(card)
