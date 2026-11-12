"""Where a suite's lists under `[tool.verity.tests]` (`inputs`, `untraced`, a `reads` entry's `paths` and `tests`) say why
an item is there.  The pyprojects' merge driver (`pyproject-test-lists`) merges two branches that each add an item under a
comment line of its own inside the list, and leaves a comment above the list's opening to git, so a paragraph there
conflicts whenever two branches that each add an item both extend it.  A list therefore holds its reasons in one place:
`test_no_list_holds_reasons_in_two_places` warns (bookkeeping, `conftest.py`) on a list with comment lines both right above
its opening and inside it, which is one whose reasons are part moved, and one whose paragraph came back."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: a table's header line, and the opening line of one of its lists
TABLE = re.compile(r"\s*\[\[?\s*([\w.\-\"' ]+?)\s*\]\]?\s*(#.*)?$")
OPENING = re.compile(r"\s*(inputs|untraced|paths|tests)\s*=\s*\[(.*)$")
STRING = re.compile(r'"[^"]*"|\'[^\']*\'')


def _comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def two_places(text: str) -> list[tuple[int, str]]:
    """(the opening's line number, the key) of each list under `[tool.verity.tests]`, its `reads` entries included, in a
    pyproject's ``text`` that has comment lines both right above its opening and between its opening and its close."""
    lines, found, table, n = text.splitlines(), [], "", 0
    while n < len(lines):
        header, opening = TABLE.match(lines[n]), OPENING.match(lines[n])
        if header and not opening:
            table = header.group(1)
        elif opening and (table == "tool.verity.tests" or table.startswith("tool.verity.tests.")):
            above = n > 0 and _comment(lines[n - 1])
            end, rest, inside = n, opening.group(2), False
            while "]" not in STRING.sub("", rest).split("#")[0] and end + 1 < len(lines):
                end += 1
                rest = lines[end]
                inside = inside or _comment(rest)
            if above and inside:
                found.append((n + 1, opening.group(1)))
            n = end
        n += 1
    return found


def test_the_scan_reads_each_form() -> None:
    above = '[tool.verity.tests]\n# why\ninputs = ["a", "b",\n          "c"]\n'
    inside = '[tool.verity.tests]\ninputs = [\n    # why a\n    "a",\n    # why b: a ] in it\n    "b",\n]\n'
    both = '[tool.verity.tests]\n# why a\ninputs = ["a",\n          # why b\n          "b"]\n# why\nuntraced = ["t"]\n'
    reads = ('[tool.verity.tests]\ninputs = ["a"]  # why\n\n[[tool.verity.tests.reads]]\ntests = ["t"]\n# why\n'
             'paths = [\n    "p",\n    # why q\n    "q",\n]\n')
    other = '[tool.other]\n# why\ninputs = [\n    # why a\n    "a",\n]\n'
    assert two_places(above) == [] and two_places(inside) == [] and two_places(other) == []
    assert two_places(both) == [(3, "inputs")]
    assert two_places(reads) == [(7, "paths")]
    assert two_places(both + other + reads.replace("[tool.verity.tests]\n", "")) == [(3, "inputs"), (19, "paths")]


def test_no_list_holds_reasons_in_two_places() -> None:
    files = subprocess.run(["git", "ls-files", "-z", "--", "pyproject.toml", "*/pyproject.toml"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.split("\0")
    found = [f"{f}:{n} `{key}`" for f in sorted(filter(None, files)) if (ROOT / f).is_file()
             for n, key in two_places((ROOT / f).read_text(encoding="utf-8"))]
    assert not found, (
        f"{len(found)} lists under [tool.verity.tests] hold their reasons in two places, in comment lines above the list's "
        f"opening and inside it: {', '.join(found)}. Move each reason from above the opening to its own comment line above "
        "the item it explains, inside the list, and give a new item its reason there too: the pyprojects' merge driver "
        "merges two branches that each add an item with its reason, and leaves a paragraph both extended as a conflict")
