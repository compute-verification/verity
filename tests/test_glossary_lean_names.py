"""Every Lean name the one symbol table gives exists (`GLOSSARY.md`, "Notation").

A row's Lean column names declarations of the tree's Lean (`Security/` and the Lean packages under `verity/`), written
in backticks, full (`Verity.Circuit`) or as a suffix of the full name (`Circuit.size`); a structure's field counts
(`Verity.RegisteredCircuit.θ`), and so does a namespace (`Accounting.Compose`). A part of the cell, between semicolons, that
says "not yet in Lean" is not checked until that lands.
"""
from __future__ import annotations

import re

from test_repository import ROOT, _tracked

GLOSSARY = "GLOSSARY.md"
LEAN_ROOTS = ("Security/", "verity/")

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


def notation_cells() -> list[str]:
    """The Lean column of every row of the Notation section's tables."""
    text = (ROOT / GLOSSARY).read_text(encoding="utf-8")
    section = text[text.index("## Notation"):]
    section = section[:section.index("\n## ", 1)]
    rows = [line for line in section.splitlines() if line.startswith("| ") and not line.startswith(("| Symbol", "|---"))]
    return [re.split(r"(?<!\\)\|", row)[3].strip() for row in rows]


def test_every_row_names_its_lean_or_says_it_has_none():
    cells = notation_cells()
    assert len(cells) > 20, "the Notation section's tables were not found"
    bare = [cell for cell in cells if cell != "—" and not _NAME.search(cell) and "not yet in Lean" not in cell]
    assert not bare, f"a row's Lean column names no declaration and says no 'not yet in Lean (ISSUE)': {bare}"


def test_every_lean_name_in_the_symbol_table_exists():
    names = lean_names()
    missing = []
    for cell in notation_cells():
        for part in cell.split(";"):
            if _PENDING.search(part):
                continue
            for name in _NAME.findall(part):
                if name not in names and not any(full.endswith("." + name) for full in names):
                    missing.append(name)
    assert not missing, (f"{GLOSSARY}'s Notation names Lean that doesn't exist: {missing}. Fix the name, or write the "
                         "part as 'not yet in Lean (ISSUE)' until the PR that adds it lands")
