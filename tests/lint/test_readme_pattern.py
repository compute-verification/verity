"""Each protocol's and subsystem's README is its architecture on one screen, in `README.template.md`'s form; held to
``allowlists/readme_pattern.json``.

The data file's ``directories`` is the one list of the protocols' and subsystems' directories. Each holds a README whose
first heading is its title and whose level-2 headings are ``PARTS``, in order and no other, within ``MAX_LINES`` lines
and ``MAX_BYTES`` bytes. Every other README under ``SCOPE`` is a sub-directory's, within ``SUB_LINES`` lines. The file's
``allowed`` names each README that breaks its form today, with its reason, and only shrinks: a listed README that passes
fails until its entry goes.

    python3 tests/lint/test_readme_pattern.py --prune   # drop each listed README that now passes
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / "readme_pattern.json"
ME = "python3 tests/lint/test_readme_pattern.py"
#: the trees whose READMEs are read, as git pathspecs; `verity/README.md` outlines the distribution and is no subsystem's
SCOPE = ("verity/*/", "adapters/", "library/", "applications/")
PARTS = ("Pieces", "What varies", "Interfaces", "Try it")
MAX_LINES, MAX_BYTES, SUB_LINES = 70, 6000, 25
FIX = (f"write it from `README.template.md` (a sub-directory's: at most {SUB_LINES} lines, what it is and one thing to "
       "try, or delete it)")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def headings(text: str) -> list[tuple[int, str]]:
    """Each heading's level and text, outside fenced code."""
    out, fenced = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced and (m := HEADING.match(line)):
            out.append((len(m[1]), m[2]))
    return out


def problem(text: str | None, own: bool) -> str | None:
    """What breaks the form of a README (``None``: missing); ``own``: a protocol's or subsystem's own."""
    if text is None:
        return "no README"
    lines, size = len(text.splitlines()), len(text.encode())
    if not own:
        return f"{lines} lines > {SUB_LINES}" if lines > SUB_LINES else None
    hs = headings(text)
    if not hs or hs[0][0] != 1 or any(level == 1 for level, _ in hs[1:]):
        return "its first heading and only level-1 heading is not its title"
    if (twos := tuple(t for level, t in hs if level == 2)) != PARTS:
        return f"its level-2 headings are {list(twos)}, not {list(PARTS)}"
    if lines > MAX_LINES or size > MAX_BYTES:
        return f"{lines} lines and {size} bytes > {MAX_LINES} and {MAX_BYTES}"
    return None


def listed() -> dict:
    return json.loads(LIST.read_text(encoding="utf-8"))


def found(data: dict) -> dict[str, str]:
    """{README: what breaks its form} over the tracked tree, a listed directory's missing README included."""
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", *(f"{s}*README.md" for s in SCOPE)],
                             capture_output=True, text=True, check=True).stdout.split("\0")
    own = {f"{d}/README.md" for d in data["directories"]}
    readmes = {p for p in tracked if p.endswith("/README.md") and (ROOT / p).is_file()} | own
    out = {}
    for p in sorted(readmes):
        text = (ROOT / p).read_text(encoding="utf-8") if (ROOT / p).is_file() else None
        if why := problem(text, p in own):
            out[p] = why
    return out


def test_a_readme_is_read_by_its_form() -> None:
    title = "# run\n\nWhat it does.\n\n"
    good = title + "".join(f"## {p}\n\nText.\n\n" for p in PARTS)
    assert problem(good, True) is None
    assert problem(good + "~~~sh\n## not a heading\n~~~\n", True) is None
    assert "level-2" in problem(good + "## History\n", True)
    assert "level-2" in problem(title + "".join(f"## {p}\n" for p in reversed(PARTS)), True)
    assert "title" in problem("## Pieces\n" + good, True)
    assert "lines" in problem(good + "x\n" * MAX_LINES, True)
    assert "bytes" in problem(good + "x" * MAX_BYTES, True)
    assert problem("x\n" * SUB_LINES, False) is None and problem("x\n" * (SUB_LINES + 1), False) == "26 lines > 25"
    assert problem(None, True) == "no README"


def test_the_template_is_in_the_form() -> None:
    assert problem((ROOT / "README.template.md").read_text(encoding="utf-8"), True) is None


def test_every_listed_directory_exists() -> None:
    gone = [d for d in listed()["directories"] if not (ROOT / d).is_dir()]
    assert not gone, f"{LIST.name}'s directories names what isn't a directory: {', '.join(gone)}; drop or rename each"


def test_no_new_readme_off_the_form() -> None:
    data = listed()
    new = {p: why for p, why in found(data).items() if p not in data["allowed"]}
    assert not new, "".join(f"\n  {p}: {why}" for p, why in new.items()) + f"\n{FIX}"


def test_the_list_only_shrinks() -> None:
    data = listed()
    stale = sorted(set(data["allowed"]) - set(found(data)))
    assert not stale, f"these READMEs are in the form now: {', '.join(stale)}; `{ME} --prune` drops them"


def main(argv: list[str]) -> int:
    if argv != ["--prune"]:
        print(f"usage: {ME} --prune", file=sys.stderr)
        return 2
    data = listed()
    now = found(data)
    stale = sorted(set(data["allowed"]) - set(now))
    data["allowed"] = {p: why for p, why in data["allowed"].items() if p in now}
    LIST.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n".join(f"dropped {p}" for p in stale) or "nothing stale")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
