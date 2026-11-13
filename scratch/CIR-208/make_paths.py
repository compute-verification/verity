"""make_paths.py LIST MAPS_DIR REHEARSED_LIST > paths.md: every path string the superseded-versions check holds, with the row of
the definitions move's maps that moves it.

LIST is the list at the check's head, MAPS_DIR holds the four maps (`definitions_{kept,merge,served,lake}_map.toml`) and
REHEARSED_LIST is the list as `cross_move.sh`'s run left it after the maps (its `list.after-move.json`): the row computed here
for each file must give the path that run wrote, or this stops."""
import collections
import json
import sys
import tomllib
from pathlib import Path

MAPS = ("definitions_kept_map", "definitions_merge_map", "definitions_served_map", "definitions_lake_map")
doc = json.load(open(sys.argv[1]))
maps = {m: tomllib.load(open(Path(sys.argv[2]) / f"{m}.toml", "rb")) for m in MAPS}
after = json.load(open(sys.argv[3]))


def row(path: str):
    """(map, the row's old side, the path after it) for the [paths] row with the longest old side that holds ``path``."""
    best = None
    for name, m in maps.items():
        for old, new in m["paths"].items():
            if (path == old or (old.endswith("/") and path.startswith(old))) and (best is None or len(old) > len(best[1])):
                best = (name, old, new + path[len(old):] if old.endswith("/") else new)
    return best


count = collections.Counter(e["file"] for e in doc["entries"])
wrote = {}
assert [e["detail"] for e in doc["entries"]] == [e["detail"] for e in after["entries"]], "the rehearsal's list is another list"
for a, b in zip(doc["entries"], after["entries"]):
    assert wrote.setdefault(a["file"], b["file"]) == b["file"], (a, b)
for f in count:
    assert (row(f)[2] if row(f) else f) == wrote[f], (f, row(f), wrote[f])
groups = collections.defaultdict(list)
for f in sorted(count):
    r = row(f)
    groups[(r[0], r[1], maps[r[0]]["paths"][r[1]]) if r else None].append(f)

TEST = [
    ("15, 43", "`tests/lint/allowlists/superseded_versions.json`",
     "the list: named in the docstring, and `LIST = REPO / \"...\"`, one whole string joined to the root", "none"),
    ("22, 50", "`adapters/vllm/manifests/records.json`",
     "prose, in the docstring and in the failure's text (`FIX`); the test never opens it", "none"),
    ("28", "`adapters/vllm/tools/ratchet_gate.py`", "prose, in the docstring", "none"),
    ("31, 44", "`tools/verity/circuit_check/tests/test_superseded_versions.py`",
     "the docstring's command line, and `RULE`, the text of the list's `rule`", "none"),
    ("42", "`Path(__file__).resolve().parents[4]`",
     "`REPO`, the root: four directories above the test file, so it holds while the file keeps its depth",
     "none: no map moves `tools/verity/circuit_check/`"),
    ("52", "`Path(__file__).resolve().relative_to(REPO)`", "`PRUNE`, the command the failure prints: the file's own place",
     "follows the file"),
    ("58", "`circuit_check` (a module)", "the fresh interpreter imports `sidecars` and `targets` from it", "none"),
    ("59", "`verity.circuits.defs` (a module)", "the fresh interpreter's `REGISTRY`",
     "none: the kept and merge maps move modules into `verity.circuits` and rename none that is there"),
]
OTHER = [
    ("`tools/verity/circuit_check/pyproject.toml` 205", "`tests/lint/allowlists/superseded_versions.json`",
     "the suite's `inputs`: the one path this change adds", "none"),
    ("the same, 227 and 256", "`tests/test_superseded_versions.py`",
     "the test's name in two `[[tool.verity.tests.reads]]` groups, from the suite's directory", "none"),
    ("`tests/README.md` 35", "`lint/allowlists/superseded_versions.json`", "prose, from `tests/`", "none"),
    ("`tests/README.md` 36", "`tools/verity/circuit_check/tests/test_superseded_versions.py`", "prose", "none"),
    ("the list, line 2", "`tools/verity/circuit_check/tests/test_superseded_versions.py`", "inside its `rule` text", "none"),
]

p = print
p("# CIR-208: every path string the superseded-versions check holds, and the move's row for each\n")
p("Made at #8463's head `b1409cffb06f0e1b46ca8ee97fc85c28c412523b` (`main` `8c11a1d5ab` merged). The maps are read on")
p("`origin/cursor/top-moves-c-b396` at `680f1541f8c3`: `tools/move/definitions_kept_map.toml`, `definitions_merge_map.toml`,")
p("`definitions_served_map.toml` and `definitions_lake_map.toml`, which run in that order. \"Row\" is the `[paths]` row whose old")
p("side is the longest prefix of the string. \"None\" means no row of the four maps holds it, so the move leaves it as it is.\n")
p("## Run, checked, and only read\n")
p("- **Run.** `r20261113-025528-cc6a` took this head merged with `680f1541f8c3` (scratch commit `2ef2f56688`) and ran the kept,")
p("  merge and served maps with `tools/move/layout.py --map M --no-lock`, one commit each. On the moved tree every one of the")
p("  list's 239 `file` strings names a file; the test file passes (`3 passed`); `--prune` keeps the same 239 entries and")
p("  changes only the order of 25 lines; and the test file's graph task passes with `undeclared_reads: []`. The maps rewrote")
p("  207 of the list's lines and no line of the test file or of `tests/README.md`. The job is `scratch/CIR-208/cross_move.sh`.")
p("- **Checked against that run.** The row computed below for each of the list's 64 files gives the path that run wrote, for")
p("  all 64 (`make_paths.py` stops otherwise).")
p("- **Only read.** The Lake packages' map: none of its rows holds a string here. `cross_move.sh` runs it as a second phase")
p("  from its next run on; `inflight.md` names that run.\n")
p("## The test file, `tools/verity/circuit_check/tests/test_superseded_versions.py`\n")
p("No path in it is put together from parts: each is one whole string from the repository's root, or is taken from `__file__`.\n")
p("| line | string | what it is | row |\n|---|---|---|---|")
for line, s, what, r in TEST:
    p(f"| {line} | {s} | {what} | {r} |")
p("\nThe registries' loader, `circuit_check.targets.load_registries`, is not this change's, and it names two modules the served")
p("map renames (`[modules]`): `verity.definitions.registry` to `verity_vllm.definitions.registry` and")
p("`verity.definitions.numerics` to `verity_vllm.definitions.numerics`. The test calls the loader and names neither.\n")
p("## The suite's declaration, the README and the list's `rule`\n")
p("| where | string | what it is | row |\n|---|---|---|---|")
for where, s, what, r in OTHER:
    p(f"| {where} | {s} | {what} | {r} |")
p("\nThe two reads groups the test's name joins are the tree's own, shared with the other tests that load the registries;")
p("this change adds no path to either. The maps rewrite their paths for all of those tests at once.\n")
p(f"## The list's `file` strings: {len(doc['entries'])} entries in {len(count)} files\n")
p("| map | row | files | entries |\n|---|---|---|---|")
for key in sorted(groups, key=lambda k: (k is None, k)):
    files = groups[key]
    where = f"`{key[0]}.toml` | `\"{key[1]}\" = \"{key[2]}\"`" if key else "none | none: the file stays where it is"
    p(f"| {where} | {len(files)} | {sum(count[f] for f in files)} |")
p("\nAfter the move the entries of the moved files are out of the list's order (by file, then name, then version), which the")
p("test doesn't read. `--prune` writes the order again and changes nothing else.\n")
for key in sorted(groups, key=lambda k: (k is None, k)):
    files = groups[key]
    p(f"### {'`' + key[0] + '.toml`, row `' + key[1] + '`' if key else 'No row'}: {len(files)} files\n")
    p("| file in the list | entries | after the move |\n|---|---|---|")
    for f in files:
        p(f"| `{f}` | {count[f]} | {'`' + wrote[f] + '`' if wrote[f] != f else 'the same'} |")
    p()
