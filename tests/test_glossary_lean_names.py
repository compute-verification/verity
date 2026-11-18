"""Every Glossary entry is in its one form, every Lean name the Notation gives exists (`GLOSSARY.md`), and so does every
one the security argument's status table gives (`security/README.md`, "Where each step stands").

An entry is one line, `- **Term** (area): definition. Code: `name`, `name2`.`, the area and the code names optional. A
Notation entry's code names are declarations of the tree's Lean (`security/` and the Lean packages under `verity/`),
full (`Verity.Circuit`) or as a suffix of the full name (`Circuit.size`); a structure's field counts
(`Verity.RegisteredCircuit.θ`), and so does a namespace (`ResourceAccounting.Compose`). A part of a status table's cell,
between semicolons, that says "not yet in Lean" is not checked until that lands.
"""
from __future__ import annotations

import re

from test_repository import ROOT, _tracked

GLOSSARY = "GLOSSARY.md"
SECURITY = "security/README.md"
LEAN_ROOTS = ("security/", "verity/")

_DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|partial|unsafe)\s+)*"
                   r"(?:def|abbrev|structure|class|inductive|theorem|lemma|instance|opaque)\s+([^\s(:{\[]+)")
_FIELD = re.compile(r"^\s+([^\s:(/-][^\s:(]*)\s*:")
_NAME = re.compile(r"`([^`/\s]+)`")
_PENDING = re.compile(r"not yet in Lean")


def lean_names() -> set[str]:
    """Every declaration's, field's and namespace's full name in the tree's Lean."""
    names: set[str] = set()
    for path in _tracked():
        if not path.endswith(".lean") or not path.startswith(LEAN_ROOTS) or "/.lake/" in path:
            continue
        stack: list[list[str]] = []
        structure: str | None = None
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
            prefix = [part for frame in stack for part in frame]
            if structure is not None:
                if line and not line[0].isspace():
                    structure = None
                elif m := _FIELD.match(line):
                    names.add(f"{structure}.{m.group(1)}")
                    continue
            words = line.split()
            if words[:1] == ["namespace"] and len(words) > 1:
                stack.append(words[1].split("."))
                names.add(".".join(prefix + stack[-1]))
            elif words[:1] == ["section"]:
                stack.append([])
            elif words[:1] == ["end"] and stack:
                stack.pop()
            elif m := _DECL.match(line):
                name = m.group(1).removeprefix("_root_.")
                full = name if m.group(1).startswith("_root_.") else ".".join(prefix + [name])
                names.add(full)
                if re.match(r"(?:.*\s)?(?:structure|class)\s", line) and line.rstrip().endswith("where"):
                    structure = full
    return names


_ENTRY = re.compile(r"- \*\*(?P<term>[^*]+)\*\*(?: \((?P<area>[^()]+)\))?: (?P<definition>.+?)"
                    r"(?: Code: (?P<code>`[^`]+`(?:, `[^`]+`)*)\.)?")


def entries(section: str) -> list[tuple[str, str | None, str, list[str]]]:
    """(term, area, definition, code names) for each entry of a section of the Glossary; a line not in the form fails."""
    text = (ROOT / GLOSSARY).read_text(encoding="utf-8")
    body = text[text.index(f"\n## {section}\n") + len(section) + 5:]
    body = body[:body.find("\n## ")] if "\n## " in body else body
    parsed = []
    for line in body.splitlines():
        if not line:
            continue
        m = _ENTRY.fullmatch(line)
        assert m, f"{GLOSSARY}, {section}: not an entry `- **Term** (area): definition. Code: `name`.`: {line[:120]}"
        parsed.append((m["term"], m["area"], m["definition"], _NAME.findall(m["code"] or "")))
    return parsed


def test_every_entry_parses():
    terms, notation = entries("Terms"), entries("Notation")
    assert len(terms) > 100 and len(notation) > 20, f"{GLOSSARY}'s Terms or Notation section was not found"
    keys = [term.lower() for term, *_ in terms]
    assert keys == sorted(keys), f"{GLOSSARY}'s Terms are out of alphabetical order"
    assert len(set(keys)) == len(keys), f"{GLOSSARY} has a term twice: give it one entry"


def notation_cells() -> list[str]:
    """The code names of every Notation entry, one cell each."""
    return ["; ".join(f"`{name}`" for name in code) for *_, code in entries("Notation")]


def security_status_cells() -> list[str]:
    """The Lean column of the security argument's status table."""
    text = (ROOT / SECURITY).read_text(encoding="utf-8")
    section = text[text.index("### Where each step stands"):]
    section = section[:section.index("\n#", 1)]
    rows = [line for line in section.splitlines() if line.startswith("| ") and not line.startswith(("| Step", "|---"))]
    return [re.split(r"(?<!\\)\|", row)[2].strip() for row in rows]


def missing_names(cells: list[str]) -> list[str]:
    """The names in `cells` that no Lean declares, skipping each part that says it is not yet in Lean."""
    names = lean_names()
    missing = []
    for cell in cells:
        for part in cell.split(";"):
            if _PENDING.search(part):
                continue
            for name in _NAME.findall(part):
                if name not in names and not any(full.endswith("." + name) for full in names):
                    missing.append(name)
    return missing


def test_every_lean_name_in_the_symbol_table_exists():
    missing = missing_names(notation_cells())
    assert not missing, (f"{GLOSSARY}'s Notation names Lean that doesn't exist: {missing}. Fix the name, or write the "
                         "part as 'not yet in Lean (ISSUE)' until the PR that adds it lands")


def test_every_lean_name_in_the_security_argument_exists():
    cells = security_status_cells()
    assert len(cells) > 10, f"{SECURITY}'s status table was not found"
    missing = missing_names(cells)
    assert not missing, (f"{SECURITY}'s status table names Lean that doesn't exist: {missing}. Fix the name, or write "
                         "the part as 'not yet in Lean' until the PR that adds it lands")
