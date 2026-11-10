"""regen_hand_edits.py IN OUT [LAYOUT]: the hand edits `regen.py` leaves in the inference-only policy's and the slowdown
calculator's maps, for the layout's rows (each of `benchmarks/rho/`, `benchmarks/nci/` and the policy's check keeps its
directory's name under `verity/protocols/inference_only/`).

IN is `regen.py`'s output directory; OUT gets the maps as they run on the moved tree; LAYOUT is the layout's map
(`layout_one_map.toml` beside this script unless given), whose new package roots the two path maps here start from.
- The calculator's rename stands where the layout puts the calculator: its path map's rows are on the moved paths, and
  the names map keeps the rows for its outputs and for the references the path map's tools don't follow. The command
  `verity bench rho` runs `benchmarks/NAME`, so it stops existing with the layout, and its row is dropped with the row
  of `benchmarks/README.md`'s table.
- The check's directory, which the word map names `verifier/inference_only/`, is flattened into `verifier/` by a path
  map of one row, `inference_only_check_map.toml`, written here: it runs last of the certified moves, so the ordinary
  map after it names the check's lock where that row puts it.
- Two references the path maps' certificate doesn't credit get a row each: `rho/params.py` in a suite outside the
  calculator's directory, which the calculator's path map keeps and the names map renames; and the check's README's
  pattern below `../tests/`, which `inference_only_check_tests_map.toml`, written here, turns into the tests' directory
  before the moves.
Each edit must hit exactly the count given, so a map that changed under it fails here rather than silently.
"""
import shutil
import sys
import tomllib
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
layout = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parent / "layout_one_map.toml"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out)
(out / "regen-report.json").unlink(missing_ok=True)

POLICY = "verity/protocols/inference_only"
ROOTS = tomllib.loads(layout.read_text())["roots"]["new"]


def table(roots: dict) -> str:
    return "{ " + ", ".join(f'{k} = "{v}"' for k, v in roots.items()) + " }"


#: the calculator's package root before and after its move, and after it for the map that runs next
RHO, SLOWDOWN = {"rho": f"{POLICY}/rho/rho"}, {"slowdown": f"{POLICY}/slowdown/slowdown"}
#: a path map's [rewrite], as the layout's own maps have it on the moved tree
REWRITE = '''exclude = ["fixtures/**", "**/fixtures/**", "**/*vectors*.json", "**/*vectors*.txt.gz", "**/CHANGELOG.md", "*.patch",
           "**/TRUSTED.sha256", "**/lake-manifest.json.lock", "uv.lock", "tools/move/**", "archive/**",
           "tools/verity/check/tests/test_move_check.py", "tools/verity/check/move_check.py",
           "tools/research/src/research/store/moves.json"]
include = ["fixtures/artifacts.json", "**/fixtures/README.md", "**/fixtures/**/README.md", "**/lake-manifest.json"]
'''
#: the suite outside the calculator's directory whose comment names a file of it from that directory
PROPERTIES = "benchmarks/circuit_properties/pyproject.toml"
#: kept by the calculator's path map: `move_check.py` credits a path from a moved directory only in a file of it
KEEP_IN = f'''# Kept in one file: `rho/params.py`, a path from the calculator's directory in a comment of a suite outside it, which
# `move_check.py` credits only in a file of the moved directory. `slowdown_calculator_names_map.toml` renames it.
keep_in = {{ "{PROPERTIES}" = ["{POLICY}/rho/rho/params.py"] }}
'''

EDITS = {
    "inference_only_map.toml": [
        # the check's directory after the layout: verity/protocols/inference_only/verifier/nci/
        ('"network/nci" = "network/inference_only"', '"verifier/nci" = "verifier/inference_only"', 1),
        # the policy's data, kept under its name where the layout puts it; the old path's keep stays for what the
        # layout leaves spelling it (a file it archives)
        (r"'\bbenchmarks/nci\b'", r"'\bbenchmarks/nci\b', '\binference_only/nci\b'", 1),
        # the benchmarks' index has no row for the policy's data once the layout has moved it
        ('"benchmarks/README.md" = ["nci"]\n', "", 1),
        ("its own\n#   `nci`, its row `nci/` of `benchmarks/README.md`'s table ([keep_in]), and in",
         "its own\n#   `nci` ([keep_in]), and in", 1),
        ("# - `nci` becomes `inference_only`: `verity/core/protocols/network/nci/` and each module, file and identifier that\n"
         "#   spells it (",
         "# - `nci` becomes `inference_only`: the check's directory, `verity/core/protocols/network/nci/`, which\n"
         "#   `inference_only_check_map.toml` then flattens into `verifier/`, and each module, file and identifier that\n"
         "#   spells it (", 1),
    ],
    "inference_only_ids_map.toml": [
        # the check's lock, where the word map and then the check's path map put it
        ('files = ["verity/core/protocols/network/inference_only/lean/lean-audit.json"]',
         f'files = ["{POLICY}/verifier/lean/lean-audit.json"]', 1),
        ("renames, on a tree `inference_only_map.toml`'s move has run on (its files are\n"
         "# named where that move puts them).",
         "renames, on a tree the moves of `inference_only_map.toml` and\n"
         "# `inference_only_check_map.toml` have run on (its files are named where those moves put them).", 1),
    ],
    "slowdown_calculator_map.toml": [
        ("`benchmarks/rho/`, the calculator of the slowdown lower bound, is `benchmarks/slowdown/`, and its package\n"
         "# `rho` is `slowdown`, so `verity bench rho` is `verity bench slowdown` (the command runs the directory it names). The\n",
         f"`{POLICY}/rho/`, the calculator of the slowdown lower bound, is\n"
         f"# `{POLICY}/slowdown/`, and its package `rho` is `slowdown`. The\n", 1),
        ('"benchmarks/rho/" = "benchmarks/slowdown/"', f'"{POLICY}/rho/" = "{POLICY}/slowdown/"', 1),
        ('"benchmarks/rho/rho/" = "benchmarks/slowdown/slowdown/"', f'"{POLICY}/rho/rho/" = "{POLICY}/slowdown/slowdown/"', 1),
        ('paths = ["benchmarks/rho/"]', f'paths = ["{POLICY}/rho/"]', 1),
        ('"benchmarks.rho" = "benchmarks.slowdown"\n"rho" = "slowdown"\n',
         '"rho" = "slowdown"\n"verity.protocols.inference_only.rho" = "verity.protocols.inference_only.slowdown"\n', 1),
        ("# whose bytes a digest pins, and `move_check.py`, whose docstring names this directory as its example of a moved one.\n",
         "# whose bytes a digest pins, the archive, and `move_check.py`, whose docstring names this directory as its example of a\n"
         "# moved one.\n", 1),
    ],
    "slowdown_calculator_names_map.toml": [
        # the outputs follow the directory's rename; the command's row is dropped with the command
        ('[[words_in]]\nfiles = ["benchmarks/slowdown/**", "verity/composite/verifier/remote/README.md"]\n'
         'words = { "verity bench rho" = "verity bench slowdown", "rho.md" = "slowdown.md", "rho.json" = "slowdown.json" }\n',
         f'[[words_in]]\nfiles = ["{POLICY}/slowdown/**"]\n'
         'words = { "rho.md" = "slowdown.md", "rho.json" = "slowdown.json" }\n', 1),
        # the one join the path map's tools leave; the layout's own row rewrote the other file's
        (f'files = ["benchmarks/circuit_properties/run.py", "{POLICY}/nci/statement.py"]', f'files = ["{POLICY}/nci/statement.py"]', 1),
        # the trees with an import root of their own, which the root's tests name from `verity/`, where the index's row was
        ('[[words_in]]\nfiles = ["benchmarks/README.md"]\nwords = { "`rho/`" = "`slowdown/`" }\n',
         '[[words_in]]\nfiles = ["tests/test_boundaries.py"]\n'
         'words = { "protocols/inference_only/rho" = "protocols/inference_only/slowdown" }\n\n'
         f'[[words_in]]\nfiles = ["{PROPERTIES}"]\nwords = {{ "rho/params.py" = "slowdown/params.py" }}\n', 1),
        # the calculator's own files, where its path map has put them by then
        (f'"{POLICY}/nci/**", "{POLICY}/rho/**"]', f'"{POLICY}/nci/**", "{POLICY}/slowdown/**"]', 1),
        ("# - The command: `verity bench rho` is `verity bench slowdown`.\n", "", 1),
        ("# - The joins the move's tools don't follow: `HERE.parent / \"rho\"` in the two runners that put the calculator on\n"
         "#   `sys.path`, and the row `rho/` of `benchmarks/README.md`'s table.\n",
         "# - The references the move's tools don't follow: `HERE.parent / \"rho\"` in the runner that puts the calculator on\n"
         "#   `sys.path`, its directory among the trees with an import root of their own (`tests/test_boundaries.py`), and\n"
         "#   `rho/params.py` in a comment of the circuit properties' suite, a path from the calculator's directory that its\n"
         "#   move keeps there ([rewrite] keep_in).\n", 1),
        ("# What stays: the symbol ρ, every `rho` that is a number (`Overhead.rho`, `Estimate.rho`, the JSON's `rho`), and the\n"
         "# red team's codec, whose `rho` is a function of its own ([keep_in]).\n",
         "# What stays: the command `verity bench rho` where a document still gives it (the command runs a directory of\n"
         "# `benchmarks/`, so none runs the calculator: a hand edit), the symbol ρ, every `rho` that is a number\n"
         "# (`Overhead.rho`, `Estimate.rho`, the JSON's `rho`), and the red team's codec, whose `rho` is a function of its own\n"
         "# ([keep_in]).\n", 1),
    ],
}
#: a directory a comment names, where the layout puts it
COMMENT_DIRS = [
    ("verity/core/protocols/network/nci", f"{POLICY}/verifier/nci"),
    ("verity/core/protocols/network/tests", "verity/protocols/certifier/verifier/network/tests"),
    ("verity/core/protocols/remote", "verity/composite/verifier/remote"),
    ("verity/core/protocols/composition", "verity/composite/verifier/composition"),
    ("benchmarks/nci", f"{POLICY}/nci"),
    ("benchmarks/rho", f"{POLICY}/rho"),
    ("examples/lifecycle", "workloads/lifecycle"),
    ("Security/", "security/"),
]

for name, edits in EDITS.items():
    p = out / name
    text = p.read_text()
    for old, new, n in edits:
        assert text.count(old) == n, f"{name}: {old!r} occurs {text.count(old)} times, not {n}"
        text = text.replace(old, new)
    p.write_text(text)

# the calculator's path map: its roots and [rewrite] as a path map has them on the moved tree
p = out / "slowdown_calculator_map.toml"
data = tomllib.loads(p.read_text())
head, rest = p.read_text().split("[roots]\n", 1)
_, rest = rest.split("\n[paths]\n", 1)
body, tail = rest.split("exclude = [", 1)
_, tail = tail.split("\n[modules]\n", 1)
p.write_text(f"{head}[roots]\nold = {table(ROOTS | RHO)}\nnew = {table(ROOTS | SLOWDOWN)}\n\n[paths]\n{body}{REWRITE}{KEEP_IN}\n[modules]\n{tail}")
assert tomllib.loads(p.read_text()).keys() == data.keys(), "the calculator's path map lost a table"

(out / "inference_only_check_map.toml").write_text(f'''# The inference-only check's place: what `tools/move/layout.py --map tools/move/inference_only_check_map.toml` moves and
# rewrites. `{POLICY}/verifier/inference_only/`, the policy's check under the name
# `inference_only_map.toml` gives its directory, is `{POLICY}/verifier/`: the layout leaves a
# directory's name out where the new directory, less `verifier/`, already ends in it. It runs after that map and after
# `slowdown_calculator_map.toml`, whose package root it names.
# Nothing is split, edited or deleted, and no file is added. The format is `layout_map.toml`'s.

[roots]
old = {table(ROOTS | SLOWDOWN)}
new = {table(ROOTS | SLOWDOWN)}

[paths]
"{POLICY}/verifier/inference_only/" = "{POLICY}/verifier/"

[retired]
paths = ["{POLICY}/verifier/inference_only/"]

[drop]
paths = []

[rewrite]
# Never rewritten: records of what was (the earlier moves and their maps, changelogs, research's recorded moves), files
# whose bytes a digest pins, the archive, and `move_check.py` and its tests.
{REWRITE}
[modules]
# the renames [paths] and [roots] imply (`layout.py --modules`)
"verity.protocols.inference_only.verifier.inference_only" = "verity.protocols.inference_only.verifier"
''')

(out / "inference_only_check_tests_map.toml").write_text(f'''# The inference-only check's tests, named by their directory: what `tools/move/rename.py --map
# tools/move/inference_only_check_tests_map.toml` rewrites in the check's README, before the moves. The README names the
# tests by a pattern below `../tests/`, a directory the check no longer has beside it, and a relative path that names
# nothing can't follow `inference_only_check_map.toml`'s move. So the README names the tests' directory, which that
# move's tools follow. It renames no path, module or Lean declaration, so the change it makes is an ordinary one,
# reviewed as such.

[words]

[[words_in]]
files = ["{POLICY}/verifier/nci/README.md"]
words = {{ "`../tests/test_nci_*.py`" = "the `test_nci_*.py` of `../../../certifier/verifier/network/tests/`" }}

[rewrite]
# Never rewritten: the earlier moves and their maps, and lock files.
exclude = ["tools/move/**", "uv.lock", "**/*.lock"]

[keep]
patterns = []
strings = []
''')

changed = 0
for p in sorted(out.glob("*.toml")):
    lines = p.read_text().split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            new = line
            for old_dir, new_dir in COMMENT_DIRS:
                new = new.replace(old_dir, new_dir)
            if new != line:
                lines[i], changed = new, changed + 1
    p.write_text("\n".join(lines))
    tomllib.loads(p.read_text())
print(f"{len(list(out.glob('*.toml')))} maps written to {out}; {changed} comment lines' directories rewritten")
