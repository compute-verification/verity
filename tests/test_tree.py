"""The tracked tree holds to its layout of record, `tests/tree.toml` (its header says what each part holds), at five
levels: the root, `verity/`, `verity/protocols/`, each group's directory there (a directory of `verity/protocols/` that
holds protocols: `proofs/`, `accounting/`, `policy/`) and each protocol's directory. Every tracked entry at those levels
is in the tree or on the not-yet list; every not-yet line names a path the tree still holds, with where it goes, its
owner and its day; and an entry the tree has that main doesn't hold yet names its owner and day until it lands. A stray entry
fails; a line or an entry whose move landed only warns (`BOOKKEEPING` in `tests/conftest.py`), so a move is never
reverted for a line of this file: top deletes landed lines, and the metric counts them until then. The not-yet list
only shrinks: it holds at most `NOT_YET_MAX` lines, at most `UNOWNED_MAX` of them unowned and `ARRIVING_MAX` arriving.
No check compares a day with today.

    python3 tests/test_tree.py REF   # the same checks over REF's tree (`git ls-tree`), without checking it out
"""
from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
import tomllib

from test_repository import ROOT, _tracked

LAYOUT = "tests/tree.toml"
HANDLES = ".agents/skills/using-slack/registry.json"
#: the not-yet list's length, its unowned lines and its arriving lines: each only shrinks, so lower it with the list
NOT_YET_MAX = 8
UNOWNED_MAX = 0
ARRIVING_MAX = 0
UNOWNED = "unowned"
#: the tree's entries under this prefix are what each protocol's directory may hold
NAME = "verity/protocols/NAME/"
#: the tree's entries under this prefix are what each group's directory may hold beside its protocols
GROUP = "verity/protocols/GROUP/"
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2} PST)?$")
_KEYS = {"tree": ({"path", "means"}, {"protocol", "owner", "day"}),
         "not_yet": ({"path", "to", "owner", "day"}, {"note", "arriving"}),
         "open": ({"what", "answers", "by"}, set())}
ASK = "or, if the tree should change, ask top (the shape is Daniel's)"


def layout() -> dict:
    return tomllib.loads((ROOT / LAYOUT).read_text(encoding="utf-8"))


def _p(path: str) -> str:
    return path.rstrip("/")


def _protocols(doc: dict) -> set[str]:
    return {_p(t["path"]) for t in doc["tree"] if t.get("protocol")}


def _groups(doc: dict) -> set[str]:
    return {p.rsplit("/", 1)[0] for p in _protocols(doc)} - {"verity/protocols"}


def tree(doc: dict) -> set[str]:
    """The tree's paths, each protocol's directory's entries written out for each protocol and each group's for each
    group."""
    out = set()
    for t in doc["tree"]:
        path = _p(t["path"])
        if path.startswith(NAME):
            out |= {f"{p}/{path[len(NAME):]}" for p in _protocols(doc)}
        elif path.startswith(GROUP):
            out |= {f"{g}/{path[len(GROUP):]}" for g in _groups(doc)}
        else:
            out.add(path)
    return out


def entries(doc: dict, tracked: list[str]) -> set[str]:
    """The tracked entries at the five levels: each file's first component, and the component below `verity/`,
    `verity/protocols/`, a group's directory and a protocol's directory."""
    levels, out = {"verity", "verity/protocols"} | _groups(doc) | _protocols(doc), set()
    for f in tracked:
        parts = f.split("/")
        out.add(parts[0])
        out |= {"/".join(parts[:k]) for k in range(2, len(parts) + 1) if "/".join(parts[:k - 1]) in levels}
    return out


def holds(tracked: list[str], path: str) -> bool:
    """Whether the tracked tree holds `path`: a file, a directory, or a pattern (`*.toml`) some file matches."""
    path = _p(path)
    return any(f == path or f.startswith(path + "/") or ("*" in path and fnmatch.fnmatchcase(f, path)) for f in tracked)


def strays(doc: dict, tracked: list[str]) -> list[str]:
    known, lines = tree(doc), [_p(n["path"]) for n in doc["not_yet"]]
    return [f"{e}: move it to its place in {LAYOUT}'s [[tree]], {ASK}" for e in sorted(entries(doc, tracked) - known)
            if not any(n == e or n.startswith(e + "/") for n in lines)]


def landed(doc: dict, tracked: list[str]) -> list[str]:
    gone = [f"{n['path']}: main no longer holds it, so its move landed: delete its [[not_yet]] line"
            for n in doc["not_yet"] if not n.get("arriving") and not holds(tracked, n["path"])]
    here = [f"{t['path']}: main holds it now: drop its owner and day from its [[tree]] entry"
            for t in doc["tree"] if "owner" in t and not t["path"].startswith((NAME, GROUP))
            and holds(tracked, t["path"])]
    return gone + here


def malformed(doc: dict, handles: set[str]) -> list[str]:
    out = []
    for part, (required, optional) in _KEYS.items():
        rows = doc.get(part, [])
        paths = [r.get("path") for r in rows]
        out += [f"[[{part}]] {p}: listed more than once; keep one" for p in sorted({p for p in paths if p and paths.count(p) > 1})]
        for r in rows:
            name = r.get("path", r.get("what", "?"))
            out += [f"[[{part}]] {name}: give it its {k}" for k in sorted(required - set(r))]
            out += [f"[[{part}]] {name}: {k} is no key of [[{part}]] (its keys: {', '.join(sorted(required | optional))})"
                    for k in sorted(set(r) - required - optional)]
            out += [f"[[{part}]] {name}: its {k} is empty: write it" for k in sorted(required & set(r))
                    if not str(r[k]).strip()]
    out += [f"[[tree]] {t.get('path')}: an entry main doesn't hold yet names both its owner and its day"
            for t in doc.get("tree", []) if ("owner" in t) != ("day" in t)]
    for r in doc.get("not_yet", []):
        owner = r.get("owner")
        if owner != UNOWNED:
            out += [f"[[not_yet]] {r.get('path')}: an owned line names its {k}: write it, or the owner is {UNOWNED}"
                    for k in ("to", "day") if r.get(k) == UNOWNED]
    dated = [(f"[[tree]] {t['path']}", t) for t in doc.get("tree", []) if "owner" in t] + \
            [(f"[[not_yet]] {r.get('path')}", r) for r in doc.get("not_yet", [])] + \
            [(f"[[open]] {o.get('what')}", {"owner": o.get("answers"), "day": o.get("by")}) for o in doc.get("open", [])]
    for name, r in dated:
        if r.get("owner") not in handles | {UNOWNED}:
            out.append(f"{name}: its owner {r.get('owner')!r} is no handle in {HANDLES}: name one, or {UNOWNED}")
        if r.get("day") is not None and r.get("day") != UNOWNED and not _DAY.match(str(r.get("day"))):
            out.append(f"{name}: its day {r.get('day')!r} is a date (2026-11-19), with a time where it isn't 6 PM PST "
                       f"(2026-11-13 16:00 PST), or {UNOWNED}")
    return out


def longer(doc: dict) -> list[str]:
    lines = doc.get("not_yet", [])
    counts = (("lines", len(lines), NOT_YET_MAX, "NOT_YET_MAX"),
              ("unowned lines", sum(r.get("owner") == UNOWNED for r in lines), UNOWNED_MAX, "UNOWNED_MAX"),
              ("arriving lines", sum(bool(r.get("arriving")) for r in lines), ARRIVING_MAX, "ARRIVING_MAX"))
    return [f"{LAYOUT}'s not-yet list has {n} {what}, above tests/test_tree.py's {name} = {most}: it only shrinks, so "
            f"move the entry to its place in the tree, {ASK}" for what, n, most, name in counts if n > most]


def still_arriving(doc: dict, tracked: list[str]) -> list[str]:
    return [f"{r['path']}: main holds it now: drop its line's `arriving` and lower ARRIVING_MAX"
            for r in doc["not_yet"] if r.get("arriving") and holds(tracked, r["path"])]


def _handles() -> set[str]:
    return {h["handle"] for h in json.loads((ROOT / HANDLES).read_text(encoding="utf-8"))["handles"]}


def test_every_entry_is_in_the_tree_or_on_the_not_yet_list() -> None:
    off = strays(layout(), _tracked())
    assert not off, "tracked entries the layout of record doesn't place:\n  " + "\n  ".join(off)


def test_every_not_yet_line_and_coming_entry_is_still_to_come() -> None:
    """A not-yet line whose path main no longer holds, or a coming entry main now holds: its move landed. Bookkeeping: it
    warns, and top deletes the line or drops the entry's owner and day."""
    off = landed(layout(), _tracked())
    assert not off, f"{LAYOUT} says these are still to come, and main holds the change:\n  " + "\n  ".join(off)


def test_every_line_says_where_it_goes_who_and_when() -> None:
    off = malformed(layout(), _handles())
    assert not off, f"{LAYOUT}'s lines lack what its header asks of them:\n  " + "\n  ".join(off)


def test_the_not_yet_list_only_shrinks() -> None:
    off = longer(layout())
    assert not off, "\n".join(off)


def test_no_line_is_still_arriving() -> None:
    off = still_arriving(layout(), _tracked())
    assert not off, f"{LAYOUT}'s arriving lines whose change landed:\n  " + "\n  ".join(off)


def test_each_check_names_its_fix() -> None:
    """Its paths are made up: no tree holds them and no layout map names them, so a move leaves this test as it is. Only
    the levels the checks read (`verity/`, `verity/protocols/`) are real."""
    doc = {"tree": [{"path": "verity/", "means": "m"}, {"path": "verity/protocols/", "means": "m"},
                    {"path": "verity/protocols/made-up-protocol/", "means": "m", "protocol": True},
                    {"path": "verity/protocols/NAME/made-up-part/", "means": "m"},
                    {"path": "verity/made-up-coming.py", "means": "m", "owner": "top", "day": "2026-11-19"}],
           "not_yet": [{"path": "verity/made-up-dir/", "to": "deleted", "owner": "top", "day": "2026-11-19"},
                       {"path": "verity/made-up-gone.py", "to": "x", "owner": "top", "day": "2026-11-19"},
                       {"path": "verity/made-up-arriving/", "to": UNOWNED, "owner": UNOWNED, "day": UNOWNED,
                        "arriving": True}]}
    tracked = ["verity/made-up-dir/a.py", "verity/made-up-coming.py",
               "verity/protocols/made-up-protocol/made-up-part/V.lean",
               "verity/protocols/made-up-protocol/made-up-stray/x.py",
               "verity/protocols/made-up-stray/made-up-part/R.lean", "made-up-stray.md"]
    assert strays(doc, tracked) == [f"{e}: move it to its place in {LAYOUT}'s [[tree]], {ASK}"
                                    for e in ("made-up-stray.md", "verity/protocols/made-up-protocol/made-up-stray",
                                              "verity/protocols/made-up-stray")]
    assert landed(doc, tracked) == [
        "verity/made-up-gone.py: main no longer holds it, so its move landed: delete its [[not_yet]] line",
        "verity/made-up-coming.py: main holds it now: drop its owner and day from its [[tree]] entry"]
    assert still_arriving(doc, tracked + ["verity/made-up-arriving/__init__.py"]) == [
        "verity/made-up-arriving/: main holds it now: drop its line's `arriving` and lower ARRIVING_MAX"]
    bad = {"tree": [{"path": "made-up-root/", "means": "m", "owner": "top"}],
           "not_yet": [{"path": "a/", "to": UNOWNED, "owner": "top", "day": "Thu 19 Nov"},
                       {"path": "b/", "to": "", "owner": "nobody", "day": UNOWNED, "colour": 1}],
           "open": [{"what": "w", "answers": "top"}]}
    assert malformed(bad, {"top"}) == [
        "[[not_yet]] b/: colour is no key of [[not_yet]] (its keys: arriving, day, note, owner, path, to)",
        "[[not_yet]] b/: its to is empty: write it", "[[open]] w: give it its by",
        "[[tree]] made-up-root/: an entry main doesn't hold yet names both its owner and its day",
        "[[not_yet]] a/: an owned line names its to: write it, or the owner is unowned",
        "[[not_yet]] b/: an owned line names its day: write it, or the owner is unowned",
        "[[not_yet]] a/: its day 'Thu 19 Nov' is a date (2026-11-19), with a time where it isn't 6 PM PST (2026-11-13 "
        "16:00 PST), or unowned",
        f"[[not_yet]] b/: its owner 'nobody' is no handle in {HANDLES}: name one, or unowned"]
    assert longer({"not_yet": [{"owner": UNOWNED, "arriving": True}] * (NOT_YET_MAX + 1)})[0].startswith(
        f"{LAYOUT}'s not-yet list has {NOT_YET_MAX + 1} lines, above tests/test_tree.py's NOT_YET_MAX = {NOT_YET_MAX}")


def main(ref: str) -> int:
    out = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", ref], cwd=ROOT, check=True, capture_output=True).stdout
    tracked, doc = [p.decode() for p in out.split(b"\0") if p], layout()
    found = strays(doc, tracked) + malformed(doc, _handles()) + longer(doc)
    warned = landed(doc, tracked) + still_arriving(doc, tracked)
    print("\n".join(found + [f"(bookkeeping) {s}" for s in warned]) or f"{ref}: holds to {LAYOUT}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "HEAD"))
