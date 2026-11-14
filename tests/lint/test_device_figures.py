"""A registered device's figure is written in its card (``hardware/cards/*.toml``, ``hardware/nodes/*.toml``) and nowhere
else.  Its copies outside ``hardware/`` are held to a count per file in ``allowlists/device_figures.json``, so no file gains
one while each area makes its sites read the card or take the figure as a parameter.

What it reads: every tracked file outside ``hardware/``, outside every ``tests/`` directory and outside the generated Lean (a
Lean file whose module docstring says it is generated: its generator is read, and its generator's test holds it), and in each
only the tokens that can be a figure: Python's ``NUMBER`` tokens by ``tokenize`` (never a string, a docstring or a comment),
Lean's numerals outside comments, docstrings and strings, the numerals of ``.cu``, ``.cuh`` and ``.sh`` outside comments, and
the numbers of ``.toml`` and ``.json`` data files.  A constant product (``141 * 2 ** 30``, ``141 * 2 ^ 30 / 4``, ``50 << 20``)
is folded first and counts once.

What it matches is computed from the cards and node files when it runs, never written here: each entry's value in its unit's
forms (MHz and Hz, B and bit and 32-bit words, B/s and bit/s), and each GPU's tensor peak at each of its card's clock
readings (``READINGS``) in every format its card weighs, in MACs and in FLOPs a second.  An entry counts alone only if it is
distinctive (``distinctive``: four significant digits or more), and an entry, a form or a peak only if it is rare (``rare``:
not a power of two, nor a small odd number times one, which a model's or a kernel's shape usually is, but for an entry of
2**20 or more in its own unit); an SM count, which is neither, counts only as the value of a name in ``SMS`` (an assignment,
a keyword, a key).  A file's count is its matching tokens.

It reads the cards' TOML itself rather than through ``verity.device``, since the ``loaded_by`` of that trusted place only
shrinks, and it imports only the standard library, since the ratchet gate runs ``test_no_new_device_figures`` without pytest.

An entry ``{"file", "count", "lane"}`` of ``allowlists/device_figures.json`` allows ``count`` copies in its file, to be read
from the card: that list only shrinks, and the ratchet gate holds it.  An entry of ``tests/device_figures_kept.json`` allows
numbers that stay and says why (``"why"``: another number that happens to be equal, a record, a made-up device, a
citation); that list may grow.  Neither has line numbers, and a file's allowance is its two entries' sum.  A file with more
than its allowance (an unlisted file's is 0), or an allowance above its file's count, warns (``tests/conftest.py``'s
``BOOKKEEPING``): a copy that equals its card makes no claim false when it lands.  A move doesn't warn: a file no list
names that has an untracked listed file's base name and finds exactly that file's allowance takes it over (``paired``).
It cannot see a copy that already differs from its card (it matches exact values) or a figure the card has no key for.

    python3 tests/lint/test_device_figures.py --record          # the lists are what this tree finds, each lane the registry's
    python3 tests/lint/test_device_figures.py --prune           # lower and delete what this tree no longer finds
    python3 tests/lint/test_device_figures.py --move OLD NEW    # a file (or directory) moved: its entries follow it
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tokenize
import tomllib
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

NAME = "device_figures"
ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
#: the numbers that equal a figure and stay, each entry saying why; outside the lint directory, so the ratchet gate, which
#: runs the base's lint directory, reads the change's copy, and holds only ``LIST`` to shrink
KEPT = ROOT / "tests" / f"{NAME}_kept.json"
REGISTRY = ".agents/skills/using-slack/registry.json"
CARDS, NODES = "hardware/cards/", "hardware/nodes/"
#: the readings of a GPU's tensor clock its card states: the tensor's own, then the SM clock (``verity.device.READINGS``;
#: each peak below is ``verity.device.readings(card, fmt=...)``'s, a card without ``clock.tensor`` reading ``clock.sm``)
READINGS = ("tensor", "sm")
#: the names an SM count is bound to, in lower case
SMS = frozenset({"sms", "num_sms", "sm_count"})
#: each unit's forms besides itself: (factor, the form's unit, how the reader writes the factor)
FORMS = {
    "MHz": ((10 ** 6, "Hz", " * 10**6"),),
    "GHz": ((1000, "MHz", " * 1000"), (10 ** 9, "Hz", " * 10**9")),
    "B": ((8, "bit", " * 8"), (Fraction(1, 4), "32-bit words", " / 4")),
    "B/s": ((8, "bit/s", " * 8"), (Fraction(1, 4), "32-bit words a second", " / 4")),
    "bit": ((Fraction(1, 8), "B", " / 8"),),
    "bit/s": ((Fraction(1, 8), "B/s", " / 8"),),
    "word/SM": ((4, "B/SM", " * 4"),),
}
for _u in ("B/SM", "B/core", "B/socket"):
    FORMS[_u] = ((8, _u.replace("B", "bit", 1), " * 8"), (Fraction(1, 4), _u.replace("B", "word", 1), " / 4"))
PY, LEAN, SLASH, HASH, DATA = (".py",), (".lean",), (".cu", ".cuh"), (".sh",), (".toml", ".json")
#: the operators a constant product folds, by precedence (Lean's ``^`` is a power, C's an exclusive or)
OPS = {"<<": 1, "*": 2, "/": 2, "//": 2, "**": 3, "^": 3}
C_OPS = frozenset(OPS) - {"^", "**", "//"}
LEAN_OPS = frozenset(OPS) - {"**", "//", "<<"}
NUMERAL = re.compile(r"(?<![\w.])(\d[\d']*(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)([fFuUlL]*)(?![\w.])")
BETWEEN = re.compile(r"\s*(\*\*|\*|//|/|\^|<<)\s*")
#: the name a value is bound to, at the end of the text before it: ``sms = ``, ``sms := ``, ``sms : ℕ := ``, ``"sms": ``
BOUND = re.compile(r"(\w+)['\"]?\s*(?::[^:=\n]{1,30}?)?(?::=|(?<![=!<>:])=|:)\s*$")
LEAN_TOKEN = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])\'|--|/-|-/')
LEAN_BLOCK = re.compile(r"/-|-/")
LEAN_DOC = re.compile(r"/-!.*?-/", re.DOTALL)
GENERATED = re.compile(r"\bgenerated\b", re.IGNORECASE)
SLASH_COMMENT = re.compile(r"(?<![:\\])//[^\n]*|/\*.*?\*/", re.DOTALL)
HASH_COMMENT = re.compile(r"(?:^|(?<=\s))#[^\n]*", re.MULTILINE)


class Figure(NamedTuple):
    """What a matching value equals (``h200-sxm-141gb clock.tensor (MHz)``) and how a reader reads it from the card."""
    what: str
    read: str


class Figures(NamedTuple):
    """The values that count alone, and the SM counts, which count only bound to a name in ``SMS``."""
    alone: dict[Fraction, Figure]
    sms: dict[Fraction, Figure]


def digits(value: int | Decimal) -> int:
    """A card value's significant digits as its card writes it: an integer's every digit, a decimal's to its last nonzero."""
    if isinstance(value, int):
        return len(str(abs(value)))
    return len(value.normalize().as_tuple().digits)


def power_of_two(v: Fraction) -> bool:
    return v > 0 and all(n & (n - 1) == 0 for n in (v.numerator, v.denominator))


def distinctive(value: int | Decimal) -> bool:
    """Whether a card value counts alone: four significant digits or more as its card writes it, and ``rare`` in its unit."""
    return digits(value) >= 4 and rare(Fraction(value), written=True)


def rare(v: Fraction, written: bool = False) -> bool:
    """Whether a value is unlikely to be some other number: an integer of four digits or more, not a power of two, and not
    a small odd number times one (an odd part under 64), which the shapes and sizes of models and kernels are (768, 6144,
    49152), unless it is at least 2**20 as its card writes it (50 MiB, 96 GiB; their forms in bits or words are as often
    a buffer or a float's bits); a fraction is."""
    if v.denominator != 1:
        return True
    n = v.numerator
    return len(str(abs(n))) >= 4 and not power_of_two(v) and (written and n >= 2 ** 20 or n // (n & -n) >= 64)


def _entries(table: dict, at: tuple[str, ...] = ()):
    for key, item in table.items():
        if isinstance(item, dict) and "value" in item and "unit" in item:
            if isinstance(item["value"], (int, Decimal)) and not isinstance(item["value"], bool):
                yield (*at, key), item
        elif isinstance(item, dict):
            yield from _entries(item, (*at, key))


def figures(cards: dict[str, str], nodes: dict[str, str]) -> Figures:
    """The figures of the card files ``cards`` and the node files ``nodes``, each {path: text}."""
    alone: dict[Fraction, Figure] = {}
    sms: dict[Fraction, Figure] = {}
    for path, text in sorted(cards.items()) + sorted(nodes.items()):
        doc = tomllib.loads(text, parse_float=Decimal)
        node = "node" in doc
        name = doc.get("node") if node else doc.get("sku", Path(path).stem)
        loader = f'load_node("{name}")' if node else f'load_card("{name}")'
        for keys, entry in _entries(doc):
            v, unit = Fraction(entry["value"]), entry["unit"]
            read = f"number({loader}{''.join(f'[{k!r}]' for k in keys)})".replace("'", '"')
            what = f"{name} {'.'.join(keys)}"
            if keys == ("sm", "count"):
                sms.setdefault(v, Figure(f"{what} ({unit})", read))
            if not distinctive(entry["value"]):
                continue
            alone.setdefault(v, Figure(f"{what} ({unit})", read))
            for factor, form, written in FORMS.get(unit, ()):
                if rare(v * factor):
                    alone.setdefault(v * factor, Figure(f"{what} in {form} ({entry['value']} {unit})", read + written))
        if doc.get("kind") != "gpu":
            continue
        per_sm = Fraction(doc["sm"]["count"]["value"]) * Fraction(doc["rate"]["tensor"]["value"])
        for reading in READINGS:
            if reading not in doc.get("clock", {}):
                continue
            mhz = doc["clock"][reading]["value"]
            for fmt, weight in doc.get("weight", {}).items():
                peak = per_sm * Fraction(mhz) * 10 ** 6 / Fraction(weight["value"])
                if peak.denominator != 1 or not rare(peak):
                    continue
                read = f'rate(card, fmt="{fmt}", mhz=card["clock"]["{reading}"]["value"]) with card = {loader}'
                at = f"{name}'s {fmt} tensor peak at clock.{reading} ({mhz} MHz)"
                alone.setdefault(peak, Figure(f"{at} in MAC/s", read))
                alone.setdefault(2 * peak, Figure(f"{at} in FLOP/s", read + ", times 2"))
    return Figures(alone, sms)


def fold(values: list[Fraction], ops: list[str]) -> Fraction | None:
    """``values[0] ops[0] values[1] ...`` by precedence (``**`` and ``^`` from the right), or None where it can't be."""
    def apply(a: Fraction, op: str, b: Fraction) -> Fraction:
        if op in ("**", "^", "<<"):
            if b.denominator != 1 or not 0 <= b <= 256 or op == "<<" and a.denominator != 1:
                raise ValueError(op)
            return a * 2 ** int(b) if op == "<<" else a ** int(b)
        return a * b if op == "*" else a / b if op == "/" else Fraction(a // b)

    def parse(i: int, level: int) -> tuple[Fraction, int]:
        if level > 3:
            return values[i], i
        a, i = parse(i, level + 1)
        while i < len(ops) and OPS[ops[i]] == level:
            if level == 3:
                b, j = parse(i + 1, level)
                return apply(a, ops[i], b), j
            b, j = parse(i + 1, level + 1)
            a, i = apply(a, ops[i], b), j
        return a, i

    try:
        return parse(0, 1)[0]
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def number(text: str, c: bool = False) -> Fraction | None:
    """A numeral's value (Python's, Lean's, C's with its suffix and digit separators, a shell's), or None."""
    if len(text) > 40:
        return None
    if text.isdigit():
        return Fraction(int(text))
    s = text.replace("_", "").replace("'", "")
    if c:
        s = s.rstrip("fFuUlL")
    if not s or s[-1] in "jJ":
        return None
    try:
        return Fraction(int(s, 0)) if s[:2].lower() in ("0x", "0o", "0b") else Fraction(Decimal(s))
    except (ArithmeticError, ValueError):
        return None


class Run(NamedTuple):
    """A numeral, or a constant product of numerals, with its line, its text and the name it is bound to."""
    line: int
    text: str
    values: tuple[Fraction, ...]
    folded: Fraction | None
    name: str | None


def match(run: Run, figs: Figures) -> Figure | None:
    """The figure ``run`` states: its folded value's, or one of its numerals', where that counts alone; an SM count where it
    is a lone numeral bound to a name in ``SMS``."""
    for v in (run.folded, *run.values):
        if v is not None and v in figs.alone:
            return figs.alone[v]
    if len(run.values) == 1 and run.values[0] in figs.sms and (run.name or "").lower() in SMS:
        return figs.sms[run.values[0]]
    return None


def python_runs(text: str) -> list[Run]:
    toks: list[tokenize.TokenInfo] = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(text).readline):
            if t.type not in (tokenize.NL, tokenize.COMMENT):
                toks.append(t)
    except (tokenize.TokenError, SyntaxError):
        pass
    out, i = [], 0
    while i < len(toks):
        if toks[i].type != tokenize.NUMBER:
            i += 1
            continue
        j, ops = i, []
        while j + 2 < len(toks) and toks[j + 1].type == tokenize.OP and toks[j + 1].string in OPS and \
                toks[j + 1].string != "^" and toks[j + 2].type == tokenize.NUMBER:
            ops.append(toks[j + 1].string)
            j += 2
        nums = [number(t.string) for t in toks[i:j + 1:2]]
        if all(v is not None for v in nums):
            out.append(Run(toks[i].start[0], " ".join(t.string for t in toks[i:j + 1]), tuple(nums),
                           fold(nums, ops) if ops else nums[0], _python_name(toks, i)))
        i = j + 1
    return out


def _python_name(toks: list[tokenize.TokenInfo], i: int) -> str | None:
    """The name the numeral at ``i`` is bound to: ``sms = 132``, ``sms: int = 132``, ``num_sms=132``, ``{"sms": 132}``."""
    if i < 2 or toks[i - 1].string not in ("=", ":"):
        return None
    k = i - 2
    if toks[i - 1].string == "=" and k >= 2 and toks[k].type == tokenize.NAME and toks[k - 1].string == ":":
        k -= 2
    t = toks[k]
    if t.type == tokenize.NAME:
        return t.string
    if t.type == tokenize.STRING:
        return t.string.strip("'\"bBrRuU")
    return None


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    """``text`` with each span's characters but its newlines made spaces, so offsets and lines stay."""
    out, at = [], 0
    for a, b in spans:
        out += [text[at:a], re.sub(r"[^\n]", " ", text[a:b])]
        at = b
    return "".join([*out, text[at:]])


def lean_code(text: str) -> str:
    """Lean ``text`` with its comments (``--``, nested ``/- -/``, docstrings among them) and strings blanked."""
    spans, at = [], 0
    while m := LEAN_TOKEN.search(text, at):
        if m.group() == "--":
            end = text.find("\n", m.start())
            end = len(text) if end < 0 else end
        elif m.group() == "/-":
            depth, end = 1, m.end()
            while depth and (b := LEAN_BLOCK.search(text, end)):
                depth, end = depth + (1 if b.group() == "/-" else -1), b.end()
            end = len(text) if depth else end
        elif m.group() == "-/":
            at = m.end()
            continue
        else:
            end = m.end()
        spans.append((m.start(), end))
        at = end
    return _blank(text, spans)


def text_runs(code: str, ops: frozenset[str], sms: frozenset[Fraction], c: bool = False) -> list[Run]:
    """The runs of numerals in ``code`` (comments already blanked), joined by the operators ``ops``; a lone numeral among
    ``sms`` with the name it is bound to."""
    found = list(NUMERAL.finditer(code))
    out, i, line, counted = [], 0, 1, 0
    while i < len(found):
        j, opl = i, []
        while j + 1 < len(found) and (b := BETWEEN.fullmatch(code, found[j].end(), found[j + 1].start())) and b[1] in ops:
            opl.append(b[1])
            j += 1
        nums = [number(m.group(), c) for m in found[i:j + 1]]
        start = found[i].start()
        if all(v is not None for v in nums):
            bound = None
            if len(nums) == 1 and nums[0] in sms:
                line_start = code.rfind("\n", 0, start) + 1
                bound = BOUND.search(code[max(line_start, start - 80):start])
            line, counted = line + code.count("\n", counted, start), start
            out.append(Run(line, code[start:found[j].end()], tuple(nums),
                           fold(nums, opl) if opl else nums[0], bound and bound[1]))
        i = j + 1
    return out


def data_runs(text: str, path: str) -> list[Run]:
    """Each number of a ``.toml`` or ``.json`` data file, with its key path in place of a line (0)."""
    try:
        doc = tomllib.loads(text, parse_float=Decimal) if path.endswith(".toml") else \
            json.loads(text, parse_float=Decimal)
    except ValueError:
        return []
    out: list[Run] = []
    todo: list[tuple[str, str | None, object]] = [("", None, doc)]
    while todo:
        at, key, v = todo.pop()
        if isinstance(v, dict):
            todo += reversed([(f"{at}.{k}" if at else str(k), str(k), x) for k, x in v.items()])
        elif isinstance(v, list):
            todo += reversed([(f"{at}[{n}]", key, x) for n, x in enumerate(v)])
        elif isinstance(v, int) and not isinstance(v, bool) or isinstance(v, Decimal) and v.is_finite():
            out.append(Run(0, f"[{at}] {str(v).lower().replace('e+', 'e')}", (Fraction(v),), Fraction(v), key))
    return out


def generated(text: str) -> bool:
    """Whether a Lean file's module docstring says it is generated."""
    return any(GENERATED.search(d.group()) for d in LEAN_DOC.finditer(text, 0, 6000))


def holds(path: str) -> bool:
    """Whether this lint reads ``path`` (a generated Lean file's text decides the rest: ``generated``)."""
    parts = path.split("/")
    return not path.startswith("hardware/") and "tests" not in parts[:-1] and path.endswith(PY + LEAN + SLASH + HASH + DATA)


def runs(text: str, path: str, sms: frozenset[Fraction] = frozenset()) -> list[Run]:
    """The runs ``path``'s text can state a figure in; a lone numeral among ``sms`` with the name it is bound to."""
    if path.endswith(PY):
        return python_runs(text)
    if path.endswith(LEAN):
        return [] if generated(text) else text_runs(lean_code(text), LEAN_OPS, sms)
    if path.endswith(SLASH):
        return text_runs(_blank(text, [m.span() for m in SLASH_COMMENT.finditer(text)]), C_OPS, sms, c=True)
    if path.endswith(HASH):
        return text_runs(_blank(text, [m.span() for m in HASH_COMMENT.finditer(text)]), frozenset({"*"}), sms)
    if path.endswith(DATA):
        return data_runs(text, path)
    return []


def sites(text: str, path: str, figs: Figures) -> list[tuple[Run, Figure]]:
    """Each run of ``path`` that states one of ``figs``, with the figure."""
    return [(r, f) for r in runs(text, path, frozenset(figs.sms)) if (f := match(r, figs)) is not None]


@lru_cache(maxsize=None)
def files() -> tuple[str, ...]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return tuple(sorted(p for p in out.decode().split("\0") if p and holds(p)))


@lru_cache(maxsize=None)
def tree_figures() -> Figures:
    read = lambda d: {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8")  # noqa: E731
                      for p in sorted((ROOT / d).glob("*.toml"))}
    return figures(read(CARDS), read(NODES))


@lru_cache(maxsize=None)
def scan() -> tuple[tuple[str, Run, Figure], ...]:
    """Each matching run's file, run and figure."""
    figs = tree_figures()
    return tuple((f, r, fig) for f in files() if (ROOT / f).is_file()
                 for r, fig in sites((ROOT / f).read_text(encoding="utf-8", errors="replace"), f, figs))


def parse(doc: object, kept: bool = False) -> dict[str, dict]:
    """Each file's entry, from a list's document: an entry of ``KEPT`` says why its numbers stay, one of ``LIST`` doesn't."""
    name = (KEPT if kept else LIST).relative_to(ROOT).as_posix()
    keys = {"file", "count", "lane", "why"} if kept else {"file", "count", "lane"}
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{name}: {{'entries': [...]}} expected"
    out: dict[str, dict] = {}
    for e in doc["entries"]:
        assert isinstance(e, dict) and set(e) == keys and all(isinstance(e[k], str) and e[k] for k in keys - {"count"}), (
            f"{name}: an entry is {{{', '.join(sorted(keys, key=['file', 'count', 'lane', 'why'].index))}}}"
            + (f"; one without a why is a copy to read from its card, in {LIST.relative_to(ROOT).as_posix()}" if kept
               else f"; one that stays says why, in {KEPT.relative_to(ROOT).as_posix()}") + f": {e}")
        assert type(e["count"]) is int and e["count"] >= 1, f"{name}: count must be a positive int: {e}"
        assert e["file"] not in out, f"{name}: duplicate entry {e}"
        out[e["file"]] = e
    return out


def lists() -> tuple[dict[str, dict], dict[str, dict]]:
    """``LIST``'s entries and ``KEPT``'s."""
    return tuple(parse(json.loads(p.read_text(encoding="utf-8")), p == KEPT) if p.is_file() else {}
                 for p in (LIST, KEPT))  # type: ignore[return-value]


def allowed() -> Counter:
    """Each file's allowance: its entries' counts in both lists."""
    out: Counter = Counter()
    for entries in lists():
        for f, e in entries.items():
            out[f] += e["count"]
    return out


def found() -> Counter:
    return Counter(f for f, *_ in scan())


def paired(allow: Counter, now: Counter, tracked: set[str]) -> dict[str, str]:
    """{new: old}: a file no list names that has an untracked listed file's base name and finds exactly its allowance, which
    it takes over silently: a move.  Each listed file pairs once."""
    gone = sorted(f for f in allow if f not in tracked)
    out: dict[str, str] = {}
    for f in sorted(now):
        if f not in allow:
            old = next((g for g in gone if g not in out.values() and g.rsplit("/", 1)[-1] == f.rsplit("/", 1)[-1]
                        and allow[g] == now[f]), None)
            if old:
                out[f] = old
    return out


def carried(entries: dict[str, dict], pairs: dict[str, str]) -> dict[str, dict]:
    """``entries`` with each paired file's entry moved to its new path."""
    for new, old in pairs.items():
        entries = moved(entries, old, new)
    return entries


def owners() -> list[tuple[str, str]]:
    reg = json.loads((ROOT / REGISTRY).read_text(encoding="utf-8"))
    return [(p, h["handle"]) for h in reg["handles"] for p in h.get("owns", [])]


def say(f: str, run: Run, fig: Figure) -> str:
    key, token = run.text.rsplit("] ", 1) if not run.line else ("", run.text)
    where = f"{f}:{run.line}" if run.line else f"{f} {key}]"
    return f"    {where}: {token} equals {fig.what}. Read it, {fig.read}, or take it as a parameter."


def warn(f: str, n: int, allow: int, kept: int, at: list[str], owner: str, gone: list[str]) -> str:
    """A file's warning: its sites, each with the card's reader, and the entry to add for the ones that are other numbers."""
    entry = json.dumps({"file": f, "count": kept + n - allow, "lane": owner, "why": "what the number is"})
    return "\n".join([f"  {f}: {n} found, {allow} allowed", *at,
                      f"    If one is another number, {'raise its entry to' if kept else 'add'} {entry} "
                      f"{'in' if kept else 'to'} {KEPT.relative_to(ROOT).as_posix()}.",
                      *(f"    If it moved from {g}, its entry follows it: `python3 tests/lint/test_{NAME}.py --move {g} {f}`."
                        for g in gone)])


def test_no_new_device_figures() -> None:
    allow, now, (_, kept) = allowed(), found(), lists()
    tracked = set(files())
    pairs = paired(allow, now, tracked)
    gone = [g for g in allow if g not in tracked and g not in pairs.values()]
    at: dict[str, list[str]] = {}
    for f, run, fig in scan():
        at.setdefault(f, []).append(say(f, run, fig))
    owns = owners()
    new = [warn(f, n, allow[f], kept[f]["count"] if f in kept else 0, at[f], lane(f, owns),
                [g for g in gone if g.rsplit("/", 1)[-1] == f.rsplit("/", 1)[-1]])
           for f, n in sorted(now.items()) if f not in pairs and n > allow[f]]
    assert not new, (f"device figures: {len(new)} file(s) state more of a card's figures than {LIST.relative_to(ROOT)} "
                     "allows; a figure is written in its card (hardware/) and read from it.\n" + "\n".join(new))


def test_the_list_is_current() -> None:
    allow, now = allowed(), found()
    moved_from = set(paired(allow, now, set(files())).values())
    stale = [f"  {f}: {allow[f]} allowed, {now[f]} found" for f in sorted(allow) if allow[f] > now[f] and f not in moved_from]
    assert not stale, (f"device figures: {len(stale)} entries of {LIST.relative_to(ROOT)} or {KEPT.relative_to(ROOT)} allow "
                       f"more than their file holds; lower them (`python3 tests/lint/test_{NAME}.py --prune`), or move a "
                       f"moved file's (`--move OLD NEW`):\n" + "\n".join(stale))


CARD = """kind = "gpu"
sku = "g"
[sm]
count = { value = 132, unit = "SM", mark = "ceiling", source = "s" }
[clock]
sm = { value = 1980, unit = "MHz", mark = "ceiling", source = "s" }
tensor = { value = 1830, unit = "MHz", mark = "ceiling", source = "s" }
[store]
dram = { value = 151397597184, unit = "B", mark = "ceiling", source = "s" }
l2 = { value = 52428800, unit = "B", mark = "ceiling", source = "s" }
reg = { value = 65536, unit = "word/SM", mark = "ceiling", source = "s" }
[rate]
tensor = { value = 4096, unit = "s8 MAC/SM-clock", mark = "ceiling", source = "s" }
[memory]
bandwidth = { value = 4.916736e12, unit = "B/s", mark = "ceiling", source = "s" }
bus = { value = 6144, unit = "bit", mark = "ceiling", source = "s" }
[weight]
s8 = { value = 1, unit = "tensor unit/MAC", mark = "ceiling", source = "s" }
bf16 = { value = 2, unit = "tensor unit/MAC", mark = "ceiling", source = "s" }
"""
NODE = """node = "n"
[memory.vdc]
bytes = { value = 499289948160, unit = "B", mark = "measurement", source = "s" }
"""


def test_what_is_a_figure_and_what_is_not() -> None:
    figs = figures({"hardware/cards/g.toml": CARD}, {"hardware/nodes/n.toml": NODE})
    assert figs.alone[Fraction(1830)].read == 'number(load_card("g")["clock"]["tensor"])'
    assert figs.alone[Fraction(1830 * 10 ** 6)].read.endswith(" * 10**6")
    assert Fraction(132 * 4096 * 1830 * 10 ** 6) in figs.alone and Fraction(132 * 4096 * 1980 * 10 ** 6) in figs.alone
    assert Fraction(132 * 2048 * 1830 * 10 ** 6 * 2) in figs.alone and Fraction(65536) not in figs.alone
    assert Fraction(52428800) in figs.alone and Fraction(52428800 * 8) not in figs.alone and Fraction(4096) not in figs.alone
    assert Fraction(151397597184 * 8) in figs.alone
    assert Fraction(6144) not in figs.alone and Fraction(768) not in figs.alone and Fraction(52428800 // 4) not in figs.alone
    assert figs.alone[Fraction(499289948160)].read == 'number(load_node("n")["memory"]["vdc"]["bytes"])'
    assert distinctive(1830) and distinctive(Decimal("1.597568e12")) and not distinctive(Decimal("4.26e12"))
    assert not distinctive(132) and not distinctive(65536) and not distinctive(Decimal("29.5"))
    assert not distinctive(6144) and not distinctive(49152) and distinctive(3201) and distinctive(103079215104)
    assert not rare(Fraction(2 ** 32), written=True) and rare(Fraction(4800)) and rare(Fraction(1047, 125))
    assert rare(Fraction(52428800), written=True) and not rare(Fraction(52428800))
    py = ('"""1830 MHz in a docstring."""\n'
          "CLOCK = 1830  # 1980 in a comment\n"
          "HZ = 1830e6\n"
          "DRAM = 141 * 2 ** 30\n"
          "PEAK = 132 * 4096 * 1830e6\n"
          "sms: int = 132\n"
          "f(num_sms=132, other=132)\n"
          "d = {'sms': 132, 'n': 1830.5, 'm': '1830'}\n"
          "if s != 132: x = 4096 * 1024\n"
          "L2 = 50 << 20\n")
    assert [(r.line, r.text) for r, _ in sites(py, "a.py", figs)] == [
        (2, "1830"), (3, "1830e6"), (4, "141 * 2 ** 30"), (5, "132 * 4096 * 1830e6"), (6, "132"), (7, "132"), (8, "132"),
        (10, "50 << 20")]
    lean = ('/-- 1830 in a docstring -/\ndef c : ℕ := 1830 * 10 ^ 6  -- 1980\ndef s := "1980"\n'
            "def d : ℕ := 141 * 2 ^ 30 / 4\nstructure X where\n  sms : ℕ := 132\n  w := 132\n/- /- 1980 -/ -/ def e := 1980\n")
    assert [(r.line, r.text) for r, _ in sites(lean, "A.lean", figs)] == [
        (2, "1830 * 10 ^ 6"), (4, "141 * 2 ^ 30 / 4"), (6, "132"), (8, "1980")]
    assert sites("/-! # T (generated)\n-/\ndef c := 1830\n", "B.lean", figs) == []
    cu = 'const double B = 132.0 * 4096 * 1830e6;  // 1980\n/* 1830 */ int x = 1 ^ 1830; printf("%.1f", 1830.0f);\n'
    assert [r.text for r, _ in sites(cu, "k.cu", figs)] == ["132.0 * 4096 * 1830e6", "1830", "1830.0f"]
    assert [r.text for r, _ in sites("# 1830\nCLOCK=1830 ip=10.0.0.1 sm_1830\nSMS=132\n", "r.sh", figs)] == ["1830", "132"]
    assert [r.text for r, _ in sites('[a]\nsms = 132\nmhz = 1980\nname = "1830"\nn = 132\n', "d.toml", figs)] == [
        "[a.sms] 132", "[a.mhz] 1980"]
    assert [r.text for r, _ in sites('{"x": [1830, 1.597], "bw": 4.916736e12}', "d.json", figs)] == [
        "[x[0]] 1830", "[bw] 4.916736e12"]
    [(run, fig)] = sites('{"r 7168 64": [1830]}', "d.json", figs)
    assert say("d.json", run, fig).startswith("    d.json [r 7168 64[0]]: 1830 equals g clock.tensor (MHz).")
    assert holds("benchmarks/x.py") and not holds("hardware/cards/g.toml") and not holds("a/tests/b.py")
    assert not holds("README.md") and holds("security/A.lean")


def test_the_lists_are_well_formed_and_an_entry_moves_with_its_file() -> None:
    assert parse({"entries": [{"file": "a.py", "count": 2, "lane": "x"}]})["a.py"]["count"] == 2
    assert parse({"entries": [{"file": "a.py", "count": 2, "lane": "x", "why": "a size"}]}, kept=True)["a.py"]["why"]
    for bad, kept in (({"file": "a.py", "count": 0, "lane": "x"}, False), ({"file": "a.py", "count": True, "lane": "x"}, False),
                      ({"file": "a.py", "count": 1}, False), ({"file": "a.py", "count": 1, "lane": "x", "why": "w"}, False),
                      ({"file": "a.py", "count": 1, "lane": "x", "line": 3}, False), ({"file": "a.py", "count": 1, "lane": "x"}, True),
                      ({"file": "a.py", "count": 1, "lane": "x", "why": ""}, True)):
        try:
            parse({"entries": [bad]}, kept)
        except AssertionError as e:
            assert str(e).startswith(f"{(KEPT if kept else LIST).relative_to(ROOT).as_posix()}: "), e
        else:
            raise AssertionError(f"{bad} parsed")
    e = {"a/x.py": {"file": "a/x.py", "count": 2, "lane": "x"}, "b.py": {"file": "b.py", "count": 1, "lane": "y"}}
    assert moved(e, "a", "c") == {"c/x.py": {"file": "c/x.py", "count": 2, "lane": "x"}, "b.py": e["b.py"]}
    assert moved(e, "a/x.py", "b.py")["b.py"]["count"] == 3
    allow = Counter({"a/x.py": 2, "a/y.py": 3, "b.py": 1})
    assert paired(allow, Counter({"c/x.py": 2, "c/y.py": 4, "b.py": 1}), {"c/x.py", "c/y.py", "b.py"}) == {"c/x.py": "a/x.py"}
    assert paired(allow, Counter({"c/x.py": 2}), {"a/x.py", "c/x.py"}) == {}
    assert carried(e, {"c/x.py": "a/x.py"})["c/x.py"]["count"] == 2
    w = warn("c/y.py", 4, 1, 1, ["    c/y.py:3: ..."], "x", ["a/y.py"])
    assert f'{{"file": "c/y.py", "count": 4, "lane": "x", "why": "what the number is"}} in {KEPT.relative_to(ROOT)}' in w
    assert "--move a/y.py c/y.py`" in w and "raise" in w and "allowlists" not in w


def moved(entries: dict[str, dict], old: str, new: str) -> dict[str, dict]:
    """``entries`` with the file ``old``'s, or the directory ``old``'s files', under ``new``: a count already there adds."""
    old, new = old.rstrip("/"), new.rstrip("/")
    out: dict[str, dict] = {}
    for f, e in entries.items():
        to = new + f[len(old):] if f == old or f.startswith(f"{old}/") else f
        out[to] = {**out[to], "count": out[to]["count"] + e["count"]} if to in out else {**e, "file": to}
    return out


def lane(path: str, owns: list[tuple[str, str]]) -> str:
    """The handle whose registry ``owns`` holds ``path`` most narrowly (``research msg owners``' rule), or ``unowned``."""
    best = max(((p, h) for p, h in owns if path == p.rstrip("/") or p.endswith("/") and path.startswith(p)),
               key=lambda x: len(x[0]), default=("", "unowned"))
    return best[1]


def dump(entries: dict[str, dict], kept: bool = False) -> None:
    keys, path = ("file", "count", "lane", "why"), KEPT if kept else LIST
    rule = ("a number that equals a device's figure and stays, saying why" if kept
            else "a device figure outside its card, to be read from it")
    lines = [f"    {json.dumps({k: e[k] for k in keys if k in e}, ensure_ascii=False)}"
             for _, e in sorted(entries.items()) if e["count"] > 0]
    path.write_text(f'{{\n  "rule": "{rule} (tests/lint/test_{NAME}.py)",\n  "entries": [\n'
                    + ",\n".join(lines) + "\n  ]\n}\n", encoding="utf-8")


if __name__ == "__main__":
    first, kept = lists()
    if sys.argv[1:2] == ["--move"] and len(sys.argv) == 4:
        dump(moved(first, *sys.argv[2:]))
        dump(moved(kept, *sys.argv[2:]), kept=True)
    elif sys.argv[1:] in (["--prune"], ["--record"]):
        now = found()
        pairs = paired(allowed(), now, set(files()))
        first, kept = carried(first, pairs), carried(kept, pairs)
        owns = owners() if sys.argv[1] == "--record" else []
        out = {}
        for f in set(first) | set(kept) | (set(now) if owns else set()):
            k = min(kept[f]["count"], now[f]) if f in kept else 0
            n = now[f] - k if owns else min(first[f]["count"] if f in first else 0, now[f] - k)
            out[f] = {**first.get(f, {}), "file": f, "count": n, "lane": first.get(f, {}).get("lane") or lane(f, owns)}
            if f in kept:
                kept[f] = {**kept[f], "count": k}
        dump(out)
        dump(kept, kept=True)
    else:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
