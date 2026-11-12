"""`tools/ratchet_gate.py` on toy repositories: a branch's own commits judged together against its merge base (the merge of its
merge bases, when it has several), each queue merge on its own, no ratchet growing (a counted kind by its total, which a move
between its keys keeps, and one keyed anew at its total is compared in the base's form;
a placing table's edit judging the edges of the modules a change adds and never those between its base's; the other entries by number, per
ratchet, or by an entry's count, P10's sizes among them), a new,
moved or renamed ratchet, the changes it doesn't judge, the base's lint code run on the change's tree (a
loosened, deleted or bypassed lint fails, naming what it still finds; a tightened one passes once its findings are fixed), the
ruff lint rerun with the base's config and pin (a loosened config, lint or pin fails), the repository's ruff lint beside it
(`tests/lint`: its list, a count per (file, rule) pooled by rule, held like the integration's, and a change to the root config
both ruff lints read reruns both), the source lint (the top change's Python under the root config: a new finding fails it by
name, a fixed one passes, one moved to another function passes it and the repository's lint, a rename keeps its findings once
its entry moves with it and fails naming the entry when it doesn't, a file the change doesn't touch is never linted, nor
a member below the top), the environment-read check beside it (a new direct read failing by name with the settings-registry
fix, one in a registry passing, a read moved within its file passing, a rename moving its entry, a registry mark counted from
the change's base, the list's total held, a loosened lint failing, a base lint that doesn't load leaving the change unjudged),
the silent-except check beside that (a new swallowing handler, typed or broad, failing by name with the fix, one that
re-raises or records its outcome passing, a rename moving its entry, the list's total held, a loosened lint failing, both
counted lints counting one change), the retired-terms check (a new use refused with the term the checkout's glossary gives),
a site the base's tree holds compared by its file and what it is, with a count, and never by its line (one an edit above it
moves is the base's, one more of its kind is the change's, a file's count that a mended site lowers is no new violation),
a repository list (ruff's, a counted lint's) catching up to what its base holds and no
further, a counted lint's list rising with an edit to its lint at the files the change leaves untouched and no others, a
counted or ruff lint's violations in a lint module the gate runs as the base has it judged on the change's own copy
(the integration's lint under vLLM's rules as its ruff lint, each against its own list, ruff run where the lint runs it), a
public command P16's list gains (passing when the file it names runs it, failing when that file doesn't or isn't
there), the source checks' findings kept beside the verdict, where ruff comes from (this
Python, uv's cache, the index, else the gate can't judge and names the host's fix), the `noqa` count (no change adds more
than it removes; a moved one is kept; only comments count; a new copy of a line is named where it was added), the exemptions
(a revert, a pure move the base's move_check certifies), and its command line."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "ratchet_gate.py"
_spec = importlib.util.spec_from_file_location("ratchet_gate", SCRIPT)
assert _spec is not None and _spec.loader is not None, SCRIPT
G = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("ratchet_gate", G)
_spec.loader.exec_module(G)

P01 = "adapters/vllm/tests/lint/allowlists/p01_core.json"
P02 = "adapters/vllm/tests/lint/allowlists/p02_more.json"
LINT = "adapters/vllm/tests/lint/test_p01_core.py"
REPO_RUFF_LINT = "tests/lint/test_repo_ruff.py"
REPO_RUFF_JSON = "tests/lint/allowlists/repo_ruff.json"
#: a stand-in for move_check: certified when every change is a 100% rename. Like move_check, it finds the root by where its
#: own file is and loads the move script and the pins list from there
MOVE_CHECK = """\
import importlib.util, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
for path in ("tools/move/rename.py", "tools/verity/check/pins.py"):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
out = subprocess.run(["git", "diff", "--name-status", "-M100%", sys.argv[1], sys.argv[2]], capture_output=True, text=True).stdout
sys.exit(0 if out and all(line.startswith("R100") for line in out.splitlines()) else 1)
"""
#: what the stand-in loads, each at its path: the move script loads the one beside it, as `rename.py` imports `layout`
MOVE_SCRIPTS = {
    "tools/move/rename.py": """\
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("layout", Path(__file__).resolve().parent / "layout.py")
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)
""",
    "tools/move/layout.py": "ROWS = ()\n",
    "tools/verity/check/pins.py": "PINS = ()\n",
}
#: a toy lint in the real ones' shape: a library file matching WORD is a violation, keyed by its file
TOY_LINT = '''\
import json
import re
from pathlib import Path

import pytest

NAME = "p01_core"
WORD = re.compile(r"BAD")
HERE = Path(__file__).resolve().parent


def found():
    return sorted(f"verity_vllm/{p.name}" for p in (HERE.parents[1] / "verity_vllm").glob("*.py") if WORD.search(p.read_text()))


@pytest.mark.slow
def test_no_new_violations():
    allowed = {e["file"] for e in json.loads((HERE / "allowlists" / f"{NAME}.json").read_text())["entries"]}
    new = [f for f in found() if f not in allowed]
    assert not new, "P1: not in its allowlist:\\n" + "\\n".join(f"  {f} [k]" for f in new)


def test_allowlist_has_no_stale_entries():
    assert False, "the gate runs no stale-entry test"
'''
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _entry(file: str, symbol: str, count: int = 1) -> dict:
    return {"file": file, "kind": "k", "symbol": symbol, "detail": "d", **({"count": count} if count != 1 else {})}


def _allowlist(*entries: dict) -> str:
    """one entry a line, as the lints' allowlists are kept"""
    return '{"rule": "p01", "entries": [\n' + ",\n".join(json.dumps(e) for e in entries) + "\n]}\n"


def _noting(seen: list, note, call):
    """``call``, with ``note`` of each call's arguments appended to ``seen`` before it runs"""
    def noted(*args, **kwargs):
        seen.append(note(*args, **kwargs))
        return call(*args, **kwargs)
    return noted


class Repo:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir()
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.root, env=ENV, check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, message: str, files: dict[str, str | None]) -> str:
        for path, text in files.items():
            p = self.root / path
            if text is None:
                self.git("rm", "-q", path)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            self.git("add", path)
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")

    def queue(self, pr: int, title: str, head: str) -> str:
        """the queue's merge of ``head`` onto the current branch (`research.queue._message`)."""
        self.git("merge", "-q", "--no-ff", "-m", f"Queue: #{pr} {title}\n\nPR: #{pr}\nHead: {head}\nOwner: lane\n", head)
        return self.git("rev-parse", "HEAD")


@pytest.fixture(autouse=True)
def verdict_cache(tmp_path, monkeypatch) -> Path:
    """each test its own cache of verdicts, so none reads another's"""
    monkeypatch.setenv("RATCHET_GATE_CACHE", str(tmp_path / "gate-cache"))
    return tmp_path / "gate-cache"


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Repo:
    r = Repo(tmp_path / "repo")
    r.commit("base", {"adapters/vllm/verity_vllm/a.py": "A = 1\n", "adapters/vllm/verity_vllm/b.py": "B = 1\n",
                      P01: _allowlist(_entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B")),
                      "tools/verity/check/move_check.py": MOVE_CHECK, **MOVE_SCRIPTS})
    monkeypatch.chdir(r.root)
    return r


@pytest.fixture
def linted(repo) -> Repo:
    """``repo`` with the toy lint, whose two violations (a.py, b.py) P01 allows."""
    repo.commit("lint", {LINT: TOY_LINT, "adapters/vllm/verity_vllm/a.py": "A = 'BAD'\n",
                         "adapters/vllm/verity_vllm/b.py": "B = 'BAD'\n"})
    repo.git("checkout", "-q", "-b", "pr")
    return repo


def _verdicts(doc: dict) -> list[tuple[str, str]]:
    return [(c["name"], c["verdict"]) for c in doc["changes"]]


def test_a_branch_is_judged_against_its_merge_base_and_may_not_grow_a_ratchet(repo):
    base = repo.git("rev-parse", "main")
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("one", {"adapters/vllm/verity_vllm/a.py": "A = 2\n"})
    assert _verdicts(G.gate("main", "pr")) == [(f"{base[:12]}..{repo.git('rev-parse', 'HEAD')[:12]}", "untouched")]
    head = repo.commit("two", {"adapters/vllm/verity_vllm/c.py": "C = 1\n",
                               P01: _allowlist(_entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B"),
                                               _entry("verity_vllm/c.py", "C"))})
    doc = G.gate("main", "pr")
    [c] = doc["changes"]
    assert (c["base"], c["commit"], c["verdict"], c["before"], c["after"], c["touched"]) == (base, head, "failed", 2, 3, [P01])
    assert c["why"].startswith(f"grows {P01} 2 -> 3 entries: no change adds a ratchet exception")
    assert c["fix"] == [f'{P01}: new {G.key(_entry("verity_vllm/c.py", "C"))}']
    assert doc["failures"] == [{"commit": head, "pr": None, "why": c["why"], "fix": c["fix"]}]
    repo.commit("three: the entry moves with the code", {"adapters/vllm/verity_vllm/b.py": None,
                                                        P01: _allowlist(_entry("verity_vllm/a.py", "A"),
                                                                        _entry("verity_vllm/c.py", "C"))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("passed", 2, 2)


def test_no_counted_kind_grows_and_one_ratchet_does_not_pay_for_another(repo):
    repo.commit("p02, and a count", {P02: _allowlist(_entry("verity_vllm/a.py", "X")),
                                     P01: _allowlist(_entry("verity_vllm/a.py", "A", 2), _entry("verity_vllm/b.py", "B"))})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("raises a count", {P01: _allowlist(_entry("verity_vllm/a.py", "A", 3), _entry("verity_vllm/b.py", "B"))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("failed", 3, 3)
    assert c["why"].startswith(f"grows {P01} k 3 -> 4 in total: no change adds a ratchet exception")
    assert c["fix"] == [f'{P01}: {G.key(_entry("verity_vllm/a.py", "A"))} 2 -> 3']
    repo.commit("raises a count, retires another of its kind", {P01: _allowlist(_entry("verity_vllm/a.py", "A", 3))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["pooled"]) == ("passed", 3, 2, [f"{P01} k 3 -> 3 in total over 2 moved key(s)"])
    repo.git("checkout", "-q", "-b", "trade", "main")
    repo.commit("retires from p01, adds to p02", {P01: _allowlist(_entry("verity_vllm/b.py", "B")),
                                                 P02: _allowlist(_entry("verity_vllm/a.py", "X"), _entry("verity_vllm/b.py", "Y"))})
    [c] = G.gate("main", "trade")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("failed", 3, 3) and f"{P02} 1 -> 2 entries" in c["why"]
    repo.git("checkout", "-q", "-b", "lower", "main")
    repo.commit("lowers a count", {P01: _allowlist(_entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B"))})
    assert _verdicts(G.gate("main", "lower"))[0][1] == "passed"


TYPED = "adapters/vllm/tests/lint/allowlists/typed_records.json"
CHECK, COMMIT, RECORDS = "verity_vllm/check/run.py", "verity_vllm/commit/plan.py", "verity_vllm/records/workload.py"


def _or_empty(file: str, n: int) -> dict:
    """typed_records' count of `.get(k) or []` reads in one file"""
    return {"file": file, "kind": "or-empty", "symbol": "<module>", "detail": ".get(k) or [] / {}", "count": n}


def _schema(file: str) -> dict:
    """typed_records' per-site entry: a schema id literal outside records/"""
    return {"file": file, "kind": "schema-literal", "symbol": "<module>", "detail": "verity/x/v1"}


def test_a_move_between_packages_keeps_its_kinds_total_as_4545_moves_reads_from_check_to_commit(repo):
    """A move of `.get(k) or []` reads from check/ to commit/, to break their cycle, raises commit's count 53 -> 87 as
    check's falls 252 -> 218.  A new file is paid for the same way; a new per-site entry isn't, by any kind."""
    gates = _schema("verity_vllm/check/gates.py")
    repo.commit("typed_records", {TYPED: _allowlist(_or_empty(CHECK, 252), _or_empty(COMMIT, 53), gates)})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("moves check's reads to commit", {TYPED: _allowlist(_or_empty(CHECK, 218), _or_empty(COMMIT, 87), gates)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["fix"]) == ("passed", 5, 5, [])
    assert c["pooled"] == [f"{TYPED} or-empty 305 -> 305 in total over 2 moved key(s)"]
    assert c["why"] == f"5 -> 5 ratchet exceptions; pooled: {c['pooled'][0]}"
    repo.commit("a new file, which check pays for", {
        TYPED: _allowlist(_or_empty(CHECK, 210), _or_empty(COMMIT, 87), _or_empty(RECORDS, 8), gates)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("passed", 5, 6)
    new = _schema("verity_vllm/commit/x.py")
    repo.commit("a new schema literal", {
        TYPED: _allowlist(_or_empty(CHECK, 200), _or_empty(COMMIT, 87), _or_empty(RECORDS, 8), gates, new)})
    [c] = G.gate("main", "pr")["changes"]
    assert c["verdict"] == "failed" and c["why"].startswith(f"grows {TYPED} 1 -> 2 entries outside its counted kinds (or-empty)")
    assert c["fix"] == [f"{TYPED}: new {G.key(new)}"]


def test_a_kinds_total_may_not_grow_and_the_failure_names_the_totals_and_the_keys_that_moved(repo):
    repo.commit("typed_records", {TYPED: _allowlist(_or_empty(CHECK, 252), _or_empty(COMMIT, 53))})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("moves 34 reads and adds one", {TYPED: _allowlist(_or_empty(CHECK, 218), _or_empty(COMMIT, 88))})
    doc = G.gate("main", "pr")
    [c] = doc["changes"]
    assert (c["verdict"], c["pooled"]) == ("failed", [])
    assert c["why"].startswith(f"grows {TYPED} or-empty 305 -> 306 in total: no change adds a ratchet exception or raises a "
                               "recorded count, past what the same kind's other counts lose")
    assert c["fix"] == [f"{TYPED}: {G.key(_or_empty(COMMIT, 0))} 53 -> 88", f"{TYPED}: {G.key(_or_empty(CHECK, 0))} 252 -> 218"]
    assert doc["failures"][0]["fix"] == c["fix"]
    repo.git("checkout", "-q", "-b", "fresh", "main")
    one = {k: v for k, v in _or_empty(RECORDS, 1).items() if k != "count"}
    repo.commit("a new file of one read, nothing paid", {TYPED: _allowlist(_or_empty(CHECK, 252), _or_empty(COMMIT, 53), one)})
    [c] = G.gate("main", "fresh")["changes"]
    assert c["verdict"] == "failed" and c["why"].startswith(f"grows {TYPED} or-empty 305 -> 306 in total")
    assert c["fix"] == [f"{TYPED}: {G.key(one)} 0 -> 1"]


TYPED_LINT = "adapters/vllm/tests/lint/test_typed_records.py"
#: a toy typed_records: the `or []` reads under verity_vllm, counted per package, or, once PER is "file", per file
COUNTING_LINT = '''\
import json
from collections import Counter
from pathlib import Path

NAME = "typed_records"
PER = "package"
HERE = Path(__file__).resolve().parent


def test_no_new_violations():
    found = Counter()
    for p in sorted((HERE.parents[1] / "verity_vllm").rglob("*.py")):
        rel = p.relative_to(HERE.parents[1]).as_posix()
        found[rel if PER == "file" else rel.rpartition("/")[0]] += p.read_text().count("or []")
    entries = json.loads((HERE / "allowlists" / f"{NAME}.json").read_text())["entries"]
    allowed = Counter({e["file"]: e.get("count", 1) for e in entries})
    new = [f"  {k}: [or-empty] ({n} found, {allowed[k]} allowed)" for k, n in sorted(found.items()) if n > allowed[k]]
    assert not new, "typed records:\\n" + "\\n".join(new)
'''


def test_a_counted_kind_keyed_anew_at_its_total_is_compared_in_the_bases_form(repo):
    """A lint that comes to count or-empty reads per file where it counted them per package leaves each of the base's
    package keys uncovered by the change's list, with no read added: the base's lint is then held to the base's entries.
    A read the same change adds still fails, at the base's key, and so does a re-keying that lowers the total, which the
    base's lint can't tell from a lint loosened to drop its entries."""
    x, y, z = (f"adapters/vllm/{f}" for f in (CHECK, "verity_vllm/check/gates.py", COMMIT))
    per_file = {TYPED_LINT: COUNTING_LINT.replace('PER = "package"', 'PER = "file"')}
    repo.commit("typed_records, per package", {
        TYPED_LINT: COUNTING_LINT, x: "A = d.get('a') or []\nB = d.get('b') or []\n", y: "C = d.get('c') or []\n",
        z: "D = d.get('d') or []\n", TYPED: _allowlist(_or_empty("verity_vllm/check", 3), _or_empty("verity_vllm/commit", 1))})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("per file", {**per_file, TYPED: _allowlist(_or_empty(CHECK, 2), _or_empty("verity_vllm/check/gates.py", 1),
                                                           _or_empty(COMMIT, 1))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["typed_records"]), c["why"]
    assert c["pooled"] == [f"{TYPED} or-empty 4 -> 4 in total over 5 moved key(s)",
                           (f"{TYPED} or-empty keeps its total of 4 under other keys, so the base's lint is held to the "
                            f"base's entries for it")]
    repo.git("checkout", "-q", "-b", "adds", "main")
    repo.commit("per file, a read more in commit and gates.py's left out", {
        **per_file, z: "D = d.get('d') or []\nE = d.get('e') or []\n", TYPED: _allowlist(_or_empty(CHECK, 2), _or_empty(COMMIT, 2))})
    [c] = G.gate("main", "adds")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["typed_records: verity_vllm/commit: [or-empty] (2 found, 1 allowed)"]), c["why"]
    repo.git("checkout", "-q", "-b", "drops", "main")
    repo.commit("per file, with commit's entry dropped", {**per_file, TYPED: _allowlist(_or_empty(CHECK, 2),
                                                                                       _or_empty("verity_vllm/check/gates.py", 1))})
    [c] = G.gate("main", "drops")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["typed_records: verity_vllm/check: [or-empty] (3 found, 0 allowed)",
                                                   "typed_records: verity_vllm/commit: [or-empty] (1 found, 0 allowed)"]), c["why"]
    assert c["why"].endswith(f"; {TYPED} or-empty is keyed anew and its total goes 4 -> 3: a kind keyed anew is compared in the "
                             "base's form only at the base's total, so change its keys in one change and its counts in another")


IMPORTS = "adapters/vllm/tests/lint/_imports.py"
LAYERING = "adapters/vllm/tests/lint/test_p09_layering.py"
P09 = "adapters/vllm/tests/lint/allowlists/p09_layering.json"
LAYER_MAP = '''\
LAYER = {
    "verity_vllm": "pipeline",
    "verity_vllm.engine": "engine",
    "verity_vllm.program": "program",
}
'''
#: a toy P9: an `import MODULE` line of a module under verity_vllm that names one in a higher layer of `_imports.LAYER`
LAYERING_LINT = '''\
import json
import re
from pathlib import Path

from . import _imports as I

NAME = "p09_layering"
RANK = {"pipeline": 3, "engine": 2, "program": 1, "core": 0}
HERE = Path(__file__).resolve().parent


def rank(module):
    return RANK[I.LAYER[max((k for k in I.LAYER if module == k or module.startswith(k + ".")), key=len)]]


def test_no_new_violations():
    root = HERE.parents[1]
    known = {p.relative_to(root).with_suffix("").as_posix().replace("/", "."): p for p in sorted(root.glob("verity_vllm/**/*.py"))}
    found = [f"{src[12:]} -> {dst[12:]}" for src, p in known.items() for dst in re.findall(r"^import (\\S+)$", p.read_text(), re.M)
             if dst in known and rank(src) < rank(dst)]
    allowed = {e["detail"] for e in json.loads((HERE / "allowlists" / f"{NAME}.json").read_text())["entries"]}
    new = [f"  {v}" for v in found if v not in allowed]
    assert not new, "layering:\\n" + "\\n".join(new)
'''


def test_a_placing_tables_edit_judges_the_edges_of_the_modules_a_change_adds_and_never_those_between_its_bases(repo):
    """A change that places a module of its base in core and adds the module that first imports it passes: the base's
    table finds that import alone, at a new module's edge, which the change's table admits.  The same row with an import
    from a module the base holds fails, naming that import and what to do, beside a new module's or alone; and so does a
    new module's import that the change's own table refuses."""
    seed, old, rule = (f"adapters/vllm/verity_vllm/{f}" for f in ("engine/seed.py", "program/old.py", "program/rule.py"))
    core, uses = (lambda m: {IMPORTS: LAYER_MAP.replace("}", f'    "verity_vllm.{m}": "core",\n}}')}), "import verity_vllm.engine.seed\n"
    repo.commit("layers", {IMPORTS: LAYER_MAP, LAYERING: LAYERING_LINT, P09: _allowlist(), seed: "S = 1\n", old: "O = 1\n"})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("seed is core, and a new module imports it", {**core("engine.seed"), rule: uses})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["p09_layering"]), c["why"]
    assert c["why"].endswith(f"the base's lint code (p09_layering) finds 1 violation(s) only at the edges of the module(s) the "
                             f"change adds, which the change's tables admit ({IMPORTS}'s LAYER)")
    repo.commit("and a module of the base imports it too", {old: uses})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["p09_layering: program.old -> engine.seed"]), c["why"]
    repo.git("checkout", "-q", "-b", "direct", "main")
    repo.commit("seed is core, and a module of the base imports it", {**core("engine.seed"), old: uses})
    [c] = G.gate("main", "direct")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["p09_layering: program.old -> engine.seed"]), c["why"]
    assert c["why"].endswith(f"; the change edits {IMPORTS}'s LAYER, which judge(s) the edges of the modules it adds: between "
                             "modules its base holds the base's tables judge, so the row lands in one change and that import in "
                             "the next")
    repo.git("checkout", "-q", "-b", "refused", "main")
    repo.commit("another row, and a new module's import neither table admits", {**core("program.old"), rule: uses})
    [c] = G.gate("main", "refused")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["p09_layering: program.rule -> engine.seed"]), c["why"]


def _size(symbol: str, lines: int) -> dict:
    return {"file": "verity_vllm/a.py", "kind": "function", "symbol": symbol, "detail": "over the cap", "count": lines}


def test_a_size_is_not_a_count_so_p10_does_not_pool(repo):
    size = "adapters/vllm/tests/lint/allowlists/p10_size.json"
    repo.commit("sizes", {size: _allowlist(_size("f", 120), _size("g", 140))})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("g shrinks, f grows", {size: _allowlist(_size("f", 130), _size("g", 100))})
    [c] = G.gate("main", "pr")["changes"]
    assert c["verdict"] == "failed" and c["why"].startswith(f"grows {size} count 120 -> 130 for {G.key(_size('f', 0))}")


def test_main_moving_on_keeps_the_branch_judged_against_its_merge_base(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("pr", {"adapters/vllm/verity_vllm/a.py": "A = 2\n", P01: _allowlist(_entry("verity_vllm/a.py", "A"))})
    repo.git("checkout", "-q", "main")
    repo.commit("main moves", {"adapters/vllm/verity_vllm/b.py": "B = 2\n", "other.txt": "x\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["touched"]) == ("passed", 2, 1, [P01])


def test_a_branch_with_two_merge_bases_is_judged_against_their_merge(repo):
    """``pr`` merged ``main`` and a side branch that ``main`` landed since, so the two have two merge bases; against either,
    what ``main`` brought by the other route (a noqa each, and an allowlist entry) would count as ``pr``'s."""
    repo.git("checkout", "-q", "-b", "side")
    repo.commit("side: a noqa", {"adapters/vllm/verity_vllm/c.py": "import os  # noqa: F401\n"})
    repo.git("checkout", "-q", "main")
    repo.commit("main: a noqa and its entry", {"adapters/vllm/verity_vllm/d.py": "import re  # noqa: F401\n",
                                               P01: _allowlist(_entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B"),
                                                               _entry("verity_vllm/d.py", "D"))})
    repo.git("checkout", "-q", "-b", "pr", "main~1")
    repo.commit("pr", {A_PY: "A = 2\n"})
    repo.git("merge", "-q", "--no-ff", "-m", "merge main", "main")
    repo.git("merge", "-q", "--no-ff", "-m", "merge side", "side")
    repo.git("checkout", "-q", "main")
    repo.queue(7, "side", "side")
    assert len(repo.git("merge-base", "--all", "main", "pr").split()) == 2
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["noqa_added"], c["touched"]) == ("untouched", 0, [])
    repo.git("checkout", "-q", "pr")
    repo.commit("pr's own noqa", {"adapters/vllm/verity_vllm/e.py": "import sys  # noqa: F401\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["verity_vllm/e.py:1: import sys  # noqa: F401"])


def test_merge_bases_that_conflict_are_refused_with_how_to_leave_one(repo):
    repo.git("checkout", "-q", "-b", "side")
    repo.commit("side", {A_PY: "A = 'side'\n"})
    repo.git("checkout", "-q", "main")
    repo.commit("main", {A_PY: "A = 'main'\n"})
    repo.git("checkout", "-q", "-b", "pr", "main~1")
    repo.commit("pr", {"adapters/vllm/verity_vllm/c.py": "C = 1\n"})
    repo.git("merge", "-q", "--no-ff", "-m", "merge main", "main")
    repo.git("merge", "-q", "--no-ff", "-X", "ours", "-m", "merge side", "side")
    repo.git("checkout", "-q", "main")
    repo.git("merge", "-q", "--no-ff", "-X", "ours", "-m", "Queue: #7 side", "side")
    with pytest.raises(G.GitError, match="don't merge cleanly, so it has no one base to be judged against: merge .* into the"):
        G.gate("main", "pr")
    repo.git("checkout", "-q", "pr")
    repo.git("merge", "-q", "--no-ff", "-m", "merge main again", "main")
    assert _verdicts(G.gate("main", "pr"))[0][1] == "untouched"


def test_each_queue_merge_is_one_pr_against_the_one_below(repo):
    base = repo.commit("p02", {P02: _allowlist(_entry("verity_vllm/a.py", "X"))})
    repo.git("checkout", "-q", "-b", "one")
    one = repo.commit("retires A", {"adapters/vllm/verity_vllm/a.py": "A = 2\n", P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    repo.git("checkout", "-q", "-b", "two", base)
    two = repo.commit("adds code and its exception", {"adapters/vllm/verity_vllm/c.py": "C = 1\n",
                                                      P02: _allowlist(_entry("verity_vllm/a.py", "X"), _entry("verity_vllm/c.py", "C"))})
    repo.git("checkout", "-q", "-b", "three", base)
    three = repo.commit("code and tests only", {"adapters/vllm/verity_vllm/d.py": "D = 1\n",
                                                "adapters/vllm/tests/test_c.py": "def test(): pass\n"})
    repo.git("checkout", "-q", "-b", "next", base)
    m1 = repo.queue(1, "retire", one)
    m2 = repo.queue(2, "feature", two)
    m3 = repo.queue(3, "tests", three)
    doc = G.gate("main", "next")
    assert _verdicts(doc) == [("#1", "passed"), ("#2", "failed"), ("#3", "untouched")]
    assert [(c["base"], c["before"], c["after"]) for c in doc["changes"][:2]] == [(base, 3, 2), (m1, 2, 3)]
    assert [(f["commit"], f["pr"]) for f in doc["failures"]] == [(m2, 2)]
    assert doc["changes"][2]["commit"] == m3


def test_runs_over_a_growing_train_judge_each_change_once_and_a_cached_verdict_is_the_one_judged(repo, verdict_cache,
                                                                                                   monkeypatch):
    """A run judges every merge above its base, those an earlier run from that base judged included: with the cache, N runs
    over a train growing by a merge each cost N judgments, not N(N+1)/2."""
    monkeypatch.setenv("RATCHET_GATE_JOBS", "1")  # judgments are counted in this process
    base = repo.commit("p02", {P02: _allowlist(_entry("verity_vllm/a.py", "X"))})
    heads = []
    for i, files in enumerate([{P01: _allowlist(_entry("verity_vllm/b.py", "B"))},
                               {P02: _allowlist(_entry("verity_vllm/a.py", "X"), _entry("verity_vllm/c.py", "C"))},
                               {"adapters/vllm/verity_vllm/d.py": "D = 1\n"}]):
        repo.git("checkout", "-q", "-b", f"b{i}", base)
        heads.append(repo.commit(f"pr {i}", files))
    repo.git("checkout", "-q", "-b", "next", base)
    tops = [repo.queue(i + 1, f"pr {i}", h) for i, h in enumerate(heads)]
    monkeypatch.setenv("RATCHET_GATE_CACHE", "")
    fresh = [G.gate("main", t) for t in tops]
    monkeypatch.setenv("RATCHET_GATE_CACHE", str(verdict_cache))
    judged: list[str] = []
    monkeypatch.setattr(G, "judge", _noting(judged, lambda c, b: c.hi, G.judge))
    assert [G.gate("main", t) for t in tops] == fresh
    assert judged == tops
    assert [G.gate("main", t) for t in tops] == fresh and judged == tops
    next(p for p in sorted(verdict_cache.glob("*.json")) if "verdict" in json.loads(p.read_text())).write_text("{")
    assert G.gate("main", tops[-1]) == fresh[-1] and len(judged) == len(tops) + 1
    monkeypatch.setattr(G, "SELF", "another gate")
    assert G.gate("main", tops[-1]) == fresh[-1] and len(judged) == 2 * len(tops) + 1


def test_changes_judged_at_once_report_what_changes_judged_in_turn_report(linted, verdict_cache, monkeypatch, capsys):
    """A train's merges are judged by a pool of processes, each with its own worktree for the base's lints: the report, its
    order and the exit status are the ones judging them in turn gives, failures included, cached or not."""
    linted.git("checkout", "-q", "-b", "next", "main")
    for pr, files in enumerate([
            {"adapters/vllm/verity_vllm/a.py": "A = 'ok'\n", P01: _allowlist(_entry("verity_vllm/b.py", "B"))},
            {P01: _allowlist()},
            {"adapters/vllm/verity_vllm/d.py": "D = 1\n"},
            {"adapters/vllm/verity_vllm/c.py": "C = 'BAD'\n", P01: _allowlist(_entry("verity_vllm/c.py", "C"))},
            {"adapters/vllm/verity_vllm/e.py": "E = 1  # noqa: E1\n"},
            {"adapters/vllm/verity_vllm/f.py": "F = 1\n"}], 1):
        linted.git("checkout", "-q", "-b", f"b{pr}", "next")
        head = linted.commit(f"pr {pr}", files)
        linted.git("checkout", "-q", "next")
        linted.queue(pr, f"pr {pr}", head)
    pools: list[int] = []
    monkeypatch.setattr(G, "ProcessPoolExecutor", _noting(pools, lambda n, **k: n, G.ProcessPoolExecutor))

    def report(jobs: str, cache: str = "") -> list[tuple[int, str]]:
        monkeypatch.setenv("RATCHET_GATE_JOBS", jobs)
        monkeypatch.setenv("RATCHET_GATE_CACHE", cache)
        return [(G.main(["main", "next", *flag]), capsys.readouterr().out) for flag in ([], ["--json"])]

    serial = report("1")
    assert pools == [] and [v for _, v in _verdicts(json.loads(serial[1][1]))] == [
        "passed", "failed", "untouched", "failed", "failed", "untouched"]
    assert [code for code, _ in serial] == [1, 1] and "#2 (" in serial[0][1] and "p01_core: verity_vllm/b.py [k]" in serial[0][1]
    assert report("4") == serial and pools == [4, 4]
    assert report("4", str(verdict_cache)) == serial and report("1", str(verdict_cache)) == serial and pools == [4, 4, 4]
    assert len(linted.git("worktree", "list").splitlines()) == 1
    monkeypatch.setenv("RATCHET_GATE_JOBS", "none")
    assert G.main(["main", "next"]) == 2 and "RATCHET_GATE_JOBS is 'none'" in capsys.readouterr().err


def test_a_new_ratchet_is_its_own_baseline_and_a_moved_or_renamed_one_is_still_compared(repo):
    repo.git("checkout", "-q", "-b", "pr")
    moved = "adapters/vllm/tests/lint/moved/allowlists/p01_core.json"
    repo.commit("adds p02, moves p01 and retires from it", {
        "adapters/vllm/verity_vllm/a.py": "A = 2\n", P01: None, moved: _allowlist(_entry("verity_vllm/b.py", "B")),
        "adapters/vllm/tests/p02_allowlist.json": _allowlist(*(_entry("verity_vllm/a.py", str(i)) for i in range(5)))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("passed", 2, 1)
    repo.git("checkout", "-q", "-b", "slug", "main")
    by_slug = "adapters/vllm/tests/lint/allowlists/core.json"
    repo.commit("names p01 by its slug and grows it", {P01: None, by_slug: _allowlist(
        _entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B"), _entry("verity_vllm/c.py", "C"))})
    [c] = G.gate("main", "slug")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("failed", 2, 3) and c["why"].startswith(f"grows {by_slug} 2 -> 3")


def test_an_allowlist_left_unreadable_fails(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("breaks it", {"adapters/vllm/verity_vllm/a.py": "A = 2\n", P01: "{"})
    [c] = G.gate("main", "pr")["changes"]
    assert c["verdict"] == "failed" and f"leaves {P01} unreadable" in c["why"]


def test_what_is_a_ratchet_lint_code_and_a_slug():
    assert G.is_ratchet("adapters/vllm/tests/lint/allowlists/p07.json")
    assert G.is_ratchet("adapters/vllm/tests/by_name_allowlist.json")
    assert not G.is_ratchet("adapters/vllm/tests/fixtures/x.json")
    assert G.is_lint_code(LINT) and G.is_lint_code("adapters/vllm/tests/lint/_ratchet.py")
    assert not any(map(G.is_lint_code, [P01, "adapters/vllm/tests/lint/README.md", "adapters/vllm/tests/test_x.py"]))
    assert {G.slug(P01), G.slug(LINT), G.slug("x/allowlists/core.json"), G.slug("x/test_core.py")} == {"core"}
    assert G.lint_selection([LINT, P02, "adapters/vllm/tests/by_name_allowlist.json"]) == ["core", "more"]
    assert G.lint_selection([LINT, "adapters/vllm/tests/lint/_ratchet.py"]) is None


def test_the_repositorys_lints_are_a_second_lint_directory():
    assert G.is_ratchet(REPO_RUFF_JSON) and G.is_lint_code(REPO_RUFF_LINT)
    assert not any(map(G.is_ratchet, ["tests/fixtures/x.json", "tests/allowlists/x.json"]))
    assert not any(map(G.is_lint_code, ["tests/test_repository.py", "tests/lint/README.md"]))
    assert G.lint_selection([REPO_RUFF_JSON, P02], G.REPO_LINT) == ["repo_ruff"] and G.lint_selection([REPO_RUFF_JSON]) == []
    assert G.lint_selection(["tests/lint/_shared.py"], G.REPO_LINT) is None and G.lint_selection(["tests/lint/_shared.py"]) == []
    assert (G.lint_selection([G.REPO_RUFF_CONFIG]), G.lint_selection([G.REPO_RUFF_CONFIG], G.REPO_LINT)) == (["ruff"], ["repo_ruff"])
    assert (G.lint_selection([G.RUFF_CONFIG]), G.lint_selection([G.RUFF_CONFIG], G.REPO_LINT)) == (["ruff"], [])


def test_a_revert_and_a_pure_move_are_exempt(linted):
    linted.commit('Revert "Queue: #9 retire B"', {"adapters/vllm/verity_vllm/c.py": "C = 'BAD'\n",
                                                  P01: _allowlist(*(_entry(f"verity_vllm/{f}.py", f) for f in "abc"))})
    assert _verdicts(G.gate("main", "pr"))[0][1] == "exempt"
    linted.git("checkout", "-q", "-b", "move", "main")
    linted.git("mv", P01, "adapters/vllm/tests/lint/allowlists/p01_other.json")
    linted.commit("move p01's allowlist off its lint's name", {})
    [c] = G.gate("main", "move")["changes"]
    assert (c["verdict"], c["why"], c["lints"]) == ("exempt", f"a pure move ({G.MOVE_CHECK} certifies it)", ["p01_core"])
    linted.git("checkout", "-q", "-b", "next", "main")
    linted.queue(4, 'Revert "x"', "pr")
    assert _verdicts(G.gate("main", "next")) == [("#4", "exempt")]


def test_once_the_base_moves_a_passed_change_is_not_judged_again_and_a_failed_one_is(repo, monkeypatch):
    """A later run judges from a newer base: keyed by BASE, every verdict would miss and each run would judge every change above
    it cold. Only a failed change's exemptions read BASE, so only it is judged again."""
    base = repo.git("rev-parse", "main")
    repo.git("checkout", "-q", "-b", "ok", base)
    ok = repo.commit("retires A", {"adapters/vllm/verity_vllm/a.py": "A = 2\n", P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    repo.git("checkout", "-q", "-b", "bad", base)
    bad = repo.commit("grows P01", {"adapters/vllm/verity_vllm/c.py": "C = 1\n",
                                    P01: _allowlist(_entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B"),
                                                    _entry("verity_vllm/c.py", "C"))})
    judged: list[str] = []
    monkeypatch.setattr(G, "judge", _noting(judged, lambda c, b: c.hi, G.judge))
    first = [G.gate("main", t) for t in ("ok", "bad")]
    repo.git("checkout", "-q", "main")
    repo.commit("main moves on", {"adapters/vllm/verity_vllm/d.py": "D = 1\n"})
    again = [G.gate("main", t) for t in ("ok", "bad")]
    assert judged == [ok, bad, bad]
    assert [[(c["commit"], c["verdict"], c["why"]) for c in d["changes"]] for d in again] == \
           [[(c["commit"], c["verdict"], c["why"]) for c in d["changes"]] for d in first]


def test_a_worktree_git_wont_remove_decides_no_verdict_and_goes_by_hand(linted, monkeypatch):
    """When `git worktree remove --force` fails ("Directory not empty"), the gate still decides the verdict: it doesn't exit 2,
    which would fail the check's step without judging the change."""
    linted.commit("fixes a.py and retires its entry", {"adapters/vllm/verity_vllm/a.py": "A = 'ok'\n",
                                                       P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    run = subprocess.run
    monkeypatch.setattr(G.subprocess, "run", lambda args, *a, **k: subprocess.CompletedProcess(args, 1)
                        if list(args[:3]) == ["git", "worktree", "remove"] else run(args, *a, **k))
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["p01_core"])
    assert len(linted.git("worktree", "list").splitlines()) == 1


def test_a_worktree_that_leaves_files_behind_decides_no_verdict(linted, monkeypatch, tmp_path):
    """When git won't remove the worktree and the removal by hand leaves a file, the scratch directory's own cleanup raises
    on 3.12, past `main`'s `GitError`: the gate still reaches its verdict, rather than exiting 1 naming no failure."""
    linted.commit("fixes a.py and retires its entry", {"adapters/vllm/verity_vllm/a.py": "A = 'ok'\n",
                                                       P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    run = subprocess.run
    monkeypatch.setattr(G.subprocess, "run", lambda args, *a, **k: subprocess.CompletedProcess(args, 1)
                        if list(args[:3]) == ["git", "worktree", "remove"] else run(args, *a, **k))

    def stubborn(path, ignore_errors=False, onerror=None, *, onexc=None, dir_fd=None):
        """`shutil.rmtree` on a directory a file it can't delete keeps: its error goes to `onexc`, as shutil's does"""
        try:
            raise OSError(39, "Directory not empty", str(path))
        except OSError as e:
            if onexc is not None:
                onexc(os.rmdir, str(path), e)
            elif not ignore_errors:
                raise

    monkeypatch.setattr(G.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(G.shutil, "rmtree", stubborn)
    # 3.12's TemporaryDirectory cleans up through shutil.rmtree; a module-level tempfile._rmtree exists only on later Pythons
    monkeypatch.setattr(G.tempfile, "_rmtree", stubborn, raising=False)
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["p01_core"])


def _turn_is_free(turn: Path) -> bool:
    """whether whoever asks for the repository's worktree turn now, without waiting, gets it"""
    with open(turn, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def test_a_worktree_is_added_and_removed_only_in_the_repositorys_turn(linted, monkeypatch, tmp_path):
    """git keeps no lock on a repository's worktrees, so of two processes that add and remove one at once the add can fail
    ("could not create directory of '.git/worktrees/tree'") and the gate exit 2 with no verdict.  Each `git worktree` command
    the gate runs, the ``prune`` after a remove git refuses too, holds the lock on ``ratchet-gate-turn`` in the repository's
    common git directory, from a linked checkout as from the first: whoever else asks for it meanwhile is refused, and gets
    it once the gate is done."""
    linted.commit("fixes a.py and retires its entry", {"adapters/vllm/verity_vllm/a.py": "A = 'ok'\n",
                                                       P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    turn, linked = linted.root / ".git" / "ratchet-gate-turn", tmp_path / "linked"
    linted.git("worktree", "add", "-q", "--detach", str(linked), "pr")
    monkeypatch.setenv("RATCHET_GATE_CACHE", "")
    seen, run = [], subprocess.run

    def watched(args, *a, **k):
        if list(args[:2]) == ["git", "worktree"]:
            seen.append((args[2], _turn_is_free(turn)))
            if args[2] == "remove":
                return subprocess.CompletedProcess(args, 1)
        return run(args, *a, **k)

    monkeypatch.setattr(G.subprocess, "run", watched)
    for checkout in (linted.root, linked):
        monkeypatch.chdir(checkout)
        [c] = G.gate("main", "pr")["changes"]
        assert (c["verdict"], c["lints"]) == ("passed", ["p01_core"])
    assert seen == [("add", False), ("remove", False), ("prune", False)] * 2 and _turn_is_free(turn)
    assert len(linted.git("worktree", "list").splitlines()) == 2


def test_with_no_turn_the_gate_cannot_judge_and_names_the_file(linted, capsys):
    """A turn the gate can't take (here a directory sits where its file goes) is exit 2 naming the file, as a git error is:
    not a traceback, whose exit 1 would read as a failure that names no change."""
    linted.commit("fixes a.py and retires its entry", {"adapters/vllm/verity_vllm/a.py": "A = 'ok'\n",
                                                       P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    (linted.root / ".git" / "ratchet-gate-turn").mkdir()
    assert G.main(["main", "pr"]) == 2
    assert "no turn at the repository's worktrees (.git/ratchet-gate-turn: " in capsys.readouterr().err


def test_a_retired_entry_passes_once_the_base_lint_agrees_it_is_gone(linted):
    linted.commit("fixes a.py and retires its entry", {"adapters/vllm/verity_vllm/a.py": "A = 'ok'\n",
                                                       P01: _allowlist(_entry("verity_vllm/b.py", "B"))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["lints"]) == ("passed", 2, 1, ["p01_core"])
    linted.commit("drops b.py's entry with b.py unchanged", {P01: _allowlist()})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["p01_core: verity_vllm/b.py [k]"])
    assert "the base's lint code" in c["why"] and "finds 1 violation(s)" in c["why"] and G.MOVE_CHECK not in c["why"]


def test_a_move_the_base_lint_refuses_is_pointed_at_move_check(linted):
    linted.git("mv", "adapters/vllm/verity_vllm/b.py", "adapters/vllm/verity_vllm/c.py")
    linted.commit("moves b.py, and its entry with it, but not as a pure move", {P01: _allowlist(_entry("verity_vllm/a.py", "A"))})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["p01_core: verity_vllm/c.py [k]"])
    assert f"is exempt once `python3 {G.MOVE_CHECK} " in c["why"]


@pytest.mark.parametrize("loosen", [
    {LINT: TOY_LINT.replace('r"BAD"', 'r"WORSE"')},
    {LINT: None},
    {LINT: TOY_LINT.replace("if WORD.search(p.read_text())", "if WORD.search(p.read_text()) and False"),
     "adapters/vllm/tests/__init__.py": "raise SystemExit(0)\n"},
], ids=["threshold", "deleted", "bypassed"])
def test_loosening_a_lint_to_drop_its_entries_fails_naming_what_the_base_lint_finds(linted, loosen):
    head = linted.commit("loosens the lint and drops its entries", {**loosen, P01: _allowlist()})
    doc = G.gate("main", "pr")
    [c] = doc["changes"]
    assert (c["verdict"], c["before"], c["after"], c["lints"]) == ("failed", 2, 0, ["p01_core"])
    assert c["fix"] == ["p01_core: verity_vllm/a.py [k]", "p01_core: verity_vllm/b.py [k]"]
    assert doc["failures"] == [{"commit": head, "pr": None, "why": c["why"], "fix": c["fix"]}]


def test_deleting_a_lint_and_its_allowlist_passes_once_nothing_is_left(linted):
    linted.commit("fixes both", {"adapters/vllm/verity_vllm/a.py": "A = 1\n", "adapters/vllm/verity_vllm/b.py": "B = 1\n",
                                 LINT: None, P01: None})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["lints"]) == ("passed", 2, 0, ["p01_core"])


def test_tightening_a_lint_passes_once_its_findings_are_fixed_and_fails_if_they_grow_the_allowlist(linted):
    linted.git("checkout", "-q", "main")
    linted.commit("c.py", {"adapters/vllm/verity_vllm/c.py": "C = 'BAR'\n"})
    linted.git("checkout", "-q", "-b", "tighten")
    tighter = {LINT: TOY_LINT.replace('r"BAD"', 'r"BA[DR]"')}
    linted.commit("tightens the lint and fixes c.py", {**tighter, "adapters/vllm/verity_vllm/c.py": "C = 'ok'\n"})
    [c] = G.gate("main", "tighten")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["lints"]) == ("passed", 2, 2, ["p01_core"])
    linted.git("checkout", "-q", "-b", "tighten-and-allow", "main")
    linted.commit("tightens the lint and allows c.py", {**tighter, P01: _allowlist(
        *(_entry(f"verity_vllm/{f}.py", f) for f in "abc"))})
    [c] = G.gate("main", "tighten-and-allow")["changes"]
    assert (c["verdict"], c["before"], c["after"]) == ("failed", 2, 3) and c["why"].startswith(f"grows {P01} 2 -> 3")


def test_shared_lint_code_runs_every_lint_and_an_error_is_named(linted):
    linted.commit("a test that isn't a lint", {"adapters/vllm/tests/lint/test_other.py": "import no_such_package\n"})
    linted.git("checkout", "-q", "main")
    linted.git("merge", "-q", "pr")
    linted.git("checkout", "-q", "-b", "helper")
    linted.commit("a helper", {"adapters/vllm/tests/lint/_helper.py": "X = 1\n"})
    [c] = G.gate("main", "helper")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["p01_core"])
    assert G.violations([{"lint": "p10_size", "test": "test_no_new_x", "error": True,
                          "message": "Traceback:\n  File x\nFileNotFoundError: gone.py\n"}]) == [
        "p10_size: test_no_new_x raised FileNotFoundError: gone.py"]


REAL_LINT = SCRIPT.parents[1] / "tests" / "lint"
RUFF_LINT = "adapters/vllm/tests/lint/test_ruff.py"
RUFF_JSON = "adapters/vllm/tests/lint/allowlists/ruff.json"
RUFF_TOML = '[lint]\nselect = ["F"]\n'
F401 = {"file": "verity_vllm/a.py", "kind": "F401", "symbol": "<module>", "detail": "`os` imported but unused"}


def _pyproject(pin: str, *more: str) -> str:
    return "[dependency-groups]\ndev = [" + ", ".join(f'"{d}"' for d in (*more, pin)) + "]\n"


@pytest.fixture
def ruffed(repo) -> Repo:
    """``repo`` with the real ruff lint, its config selecting pyflakes, and this Python's ruff pinned: a.py's unused import
    is its one finding, which ruff.json allows."""
    version = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True, check=True).stdout
    repo.pin = f"ruff=={version.split()[1]}"
    repo.commit("ruff", {RUFF_LINT: (REAL_LINT / "test_ruff.py").read_text(),
                         "adapters/vllm/tests/lint/_ratchet.py": (REAL_LINT / "_ratchet.py").read_text(),
                         G.RUFF_CONFIG: RUFF_TOML, G.PYPROJECT: _pyproject(repo.pin), RUFF_JSON: _allowlist(F401),
                         "adapters/vllm/verity_vllm/a.py": "import os\nA = 1\n"})
    repo.git("checkout", "-q", "-b", "pr")
    return repo


@pytest.mark.parametrize("loosen", [
    {G.RUFF_CONFIG: RUFF_TOML + 'ignore = ["F401"]\n'},
    {G.RUFF_CONFIG: RUFF_TOML + '[lint.per-file-ignores]\n"verity_vllm/*" = ["F401"]\n'},
    {G.RUFF_CONFIG: 'extend-exclude = ["verity_vllm"]\n' + RUFF_TOML},
    {RUFF_LINT: (REAL_LINT / "test_ruff.py").read_text().replace(
        "for f in json.loads(r.stdout))", "for f in json.loads(r.stdout) if f['code'] != 'F401')")},
    {G.PYPROJECT: _pyproject("ruff==0.0.1")},
    {G.RUFF_CONFIG: None, RUFF_LINT: None, RUFF_JSON: None},
], ids=["ignore", "per-file-ignore", "exclude", "lint-code", "pin", "deleted"])
def test_loosening_ruff_to_drop_its_findings_fails_naming_what_the_base_ruff_finds(ruffed, loosen):
    ruffed.commit("loosens ruff and drops its entry", {RUFF_JSON: _allowlist(), **loosen})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["lints"]) == ("failed", 3, 2, ["ruff"]), c["why"]
    assert c["fix"] == ["ruff: verity_vllm/a.py:1 [F401] <module>: `os` imported but unused"]


RUFF_VLLM_LINT = "adapters/vllm/tests/lint/test_ruff_vllm.py"
RUFF_VLLM_JSON = "adapters/vllm/tests/lint/allowlists/ruff_vllm.json"
RUFF_VLLM_TOML = '[lint]\nselect = ["T201"]\n'
T201 = {"file": "verity_vllm/b.py", "kind": "T201", "symbol": "<module>", "detail": "`print` found"}


@pytest.mark.parametrize("loosen", [
    {G.RUFF_VLLM_CONFIG: RUFF_VLLM_TOML + 'ignore = ["T201"]\n'},
    {G.RUFF_VLLM_CONFIG: RUFF_VLLM_TOML + '[lint.per-file-ignores]\n"verity_vllm/*" = ["T201"]\n'},
], ids=["ignore", "per-file-ignore"])
def test_loosening_the_vllm_rules_config_alone_is_judged_like_ruffs(ruffed, loosen):
    """``ruff_vllm.toml`` sits in the lint directory but isn't Python: the gate takes it as that lint's config."""
    assert G.lint_selection([G.RUFF_VLLM_CONFIG]) == ["ruff_vllm"] and G.lint_selection([G.RUFF_CONFIG]) == ["ruff"]
    assert G.python_for(ruffed.pin, ["ruff_vllm"]) == [sys.executable, "-I"]
    ruffed.git("checkout", "-q", "main")
    ruffed.commit("ruff with vLLM's rules", {RUFF_VLLM_LINT: (REAL_LINT / "test_ruff_vllm.py").read_text(),
                                             G.RUFF_VLLM_CONFIG: RUFF_VLLM_TOML, RUFF_VLLM_JSON: _allowlist(T201),
                                             "adapters/vllm/verity_vllm/b.py": "print(1)\n"})
    ruffed.git("checkout", "-q", "-B", "pr")
    ruffed.commit("loosens it and drops its entry", {RUFF_VLLM_JSON: _allowlist(), **loosen})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("failed", ["ruff_vllm"]), c["why"]
    assert c["fix"] == ["ruff_vllm: verity_vllm/b.py:1 [T201] <module>: `print` found"]
    ruffed.git("checkout", "-q", "-B", "pr", "main")
    ruffed.commit("only loosens it", loosen)
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["touched"], c["lints"]) == ("passed", [G.RUFF_VLLM_CONFIG], ["ruff_vllm"]), c["why"]


def test_tightening_ruff_passes_once_its_findings_are_fixed_and_a_kept_pin_is_not_ruffs_change(ruffed):
    ruffed.git("checkout", "-q", "main")
    ruffed.commit("c.py", {"adapters/vllm/verity_vllm/c.py": "f = lambda: 0\n"})
    ruffed.git("checkout", "-q", "-B", "pr")
    ruffed.commit("selects E731, fixes c.py, and leaves the formatter off", {
        G.RUFF_CONFIG: RUFF_TOML.replace('["F"]', '["E731", "F"]') + '[format]\nexclude = ["*"]\n',
        "adapters/vllm/verity_vllm/c.py": "def f():\n    return 0\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["touched"], c["lints"]) == ("passed", [G.RUFF_CONFIG], ["ruff"]), c["why"]
    ruffed.git("checkout", "-q", "-b", "deps", "main")
    ruffed.commit("another dev dependency, the same ruff", {G.PYPROJECT: _pyproject(ruffed.pin, "pytest==9")})
    assert _verdicts(G.gate("main", "deps"))[0][1] == "untouched"
    ruffed.commit("a new pin", {G.PYPROJECT: _pyproject("ruff==0.0.1", "pytest==9")})
    [c] = G.gate("main", "deps")["changes"]
    assert (c["verdict"], c["touched"], c["lints"]) == ("passed", [G.PYPROJECT], ["ruff"]), c["why"]


@pytest.fixture(autouse=True)
def _ruff_looked_up_afresh():
    """``ruff_python`` remembers what it found for the gate's one run; each test is a run."""
    G.ruff_python.cache_clear()


def test_ruff_runs_on_this_python_with_the_pin_else_from_uvs_cache_else_the_index_else_the_gate_cannot_judge(ruffed,
                                                                                                           monkeypatch):
    here = [sys.executable, "-I"]
    assert G.ruff_pin("main") == ruffed.pin and G.ruff_pin(ruffed.git("rev-parse", "main~1")) is None
    assert G.python_for(None, None) == here and G.python_for("ruff==0.0.1", ["core"]) == here
    assert G.python_for(ruffed.pin, ["ruff"]) == here
    uv, pin = ["/opt/uv", "run", "--no-project", "--quiet", "--python", sys.executable], ["--with", "ruff==0.0.1", "--", "python", "-I"]
    cached, fetched = [*uv, "--offline", *pin], [*uv, *pin]
    monkeypatch.setattr(G.shutil, "which", lambda name: f"/opt/{name}")
    for has in (cached, fetched):
        G.ruff_python.cache_clear()
        tried: list[list[str]] = []
        monkeypatch.setattr(G, "has_ruff", _noting(tried, lambda python, pin, exact: list(python),
                                                   lambda python, pin, exact, has=has: list(python) == has))
        assert G.python_for("ruff==0.0.1", None) == has
        assert tried == [here, cached, fetched][:tried.index(has) + 1]
    G.ruff_python.cache_clear()
    monkeypatch.setattr(G, "has_ruff", lambda python, pin, exact: False)
    fix = f"`uv pip install --python {sys.executable} --break-system-packages ruff==0.0.1`"
    with pytest.raises(G.GitError, match=r"neither .* nor `uv run --with` \(from uv's cache, then the index\)") as e:
        G.python_for("ruff==0.0.1", ["repo_ruff"])
    assert fix in str(e.value)
    monkeypatch.setattr(G.shutil, "which", lambda name: None)
    with pytest.raises(G.GitError, match="no `uv` to provide it") as e:
        G.python_for("ruff==0.0.1", ["core", "ruff"])
    assert fix in str(e.value)


TOOLS_F401 = {"file": "tools/a.py", "rule": "F401", "count": 1}
BOTH_FOUND = ["ruff: verity_vllm/a.py:1 [F401] <module>: `os` imported but unused",
              "repo_ruff: tools/a.py [F401]: 1 found, 0 allowed", "repo_ruff: tools/a.py:1 `os` imported but unused"]


@pytest.fixture
def both_ruffed(ruffed) -> Repo:
    """``ruffed`` with the repository's real ruff lint too, the root config selecting pyflakes and the integration's extending
    it: tools/a.py's unused import is the root lint's one finding, which repo_ruff.json allows."""
    ruffed.git("checkout", "-q", "main")
    ruffed.commit("repo ruff", {REPO_RUFF_LINT: (SCRIPT.parents[3] / REPO_RUFF_LINT).read_text(),
                                G.REPO_RUFF_CONFIG: RUFF_TOML, G.RUFF_CONFIG: 'extend = "../../ruff.toml"\n',
                                REPO_RUFF_JSON: _allowlist(TOOLS_F401), "tools/a.py": "import os\nA = 1\n"})
    ruffed.git("checkout", "-q", "-B", "pr")
    return ruffed


def test_the_repositorys_ruff_list_may_not_grow(both_ruffed):
    """Its entries are counts per (file, rule), pooled by rule: a new pair's count grows its rule's total."""
    b401 = {**TOOLS_F401, "file": "tools/b.py"}
    both_ruffed.commit("a new finding, allowed", {"tools/b.py": "import sys\n", REPO_RUFF_JSON: _allowlist(TOOLS_F401, b401)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["before"], c["after"], c["touched"]) == ("failed", 4, 5, [REPO_RUFF_JSON]), c["why"]
    assert c["why"].startswith(f"grows {REPO_RUFF_JSON} F401 1 -> 2 in total"), c["why"]
    assert c["fix"] == [f"{REPO_RUFF_JSON}: {G.key(b401)} 0 -> 1"]


@pytest.mark.parametrize("loosen", [
    {G.REPO_RUFF_CONFIG: RUFF_TOML + 'ignore = ["F401"]\n'},
    {G.REPO_RUFF_CONFIG: 'extend-exclude = ["tools", "verity_vllm"]\n' + RUFF_TOML},
    {G.REPO_RUFF_CONFIG: None},
    {REPO_RUFF_LINT: (SCRIPT.parents[3] / REPO_RUFF_LINT).read_text().replace('"verity/definitions/")', '"verity/definitions/", "tools/")')},
], ids=["ignore", "exclude", "deleted", "lint-code"])
def test_loosening_the_root_config_or_lint_fails_naming_what_both_base_ruff_lints_find(both_ruffed, loosen):
    both_ruffed.commit("loosens ruff and drops both entries", {RUFF_JSON: _allowlist(), REPO_RUFF_JSON: _allowlist(), **loosen})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"], c["fix"]) == ("failed", ["ruff", "repo_ruff"], BOTH_FOUND), c["why"]


def test_a_root_config_change_reruns_both_ruff_lints_and_passes_when_they_find_nothing_new(both_ruffed):
    both_ruffed.commit("a comment", {G.REPO_RUFF_CONFIG: "# the rules\n" + RUFF_TOML})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["touched"], c["lints"]) == ("passed", [G.REPO_RUFF_CONFIG], ["ruff", "repo_ruff"]), c["why"]


def test_a_new_ruff_finding_in_a_changed_file_fails_naming_it(both_ruffed):
    head = both_ruffed.commit("a second unused import in a.py, and a new file with one", {
        "tools/a.py": "import os\nimport sys\nA = 1\n", "tools/c.py": "import re\n"})
    doc = G.gate("main", "pr")
    [c] = doc["changes"]
    assert (c["verdict"], c["touched"], c["linted"]) == ("failed", [], ["tools/a.py", "tools/c.py"])
    assert c["why"].startswith("adds 2 ruff finding(s) to the Python it changes, under the base's root ruff.toml "
                               "(tools/a.py F401 1 -> 2; tools/c.py F401 0 -> 1): fix the code"), c["why"]
    assert "--write" not in c["why"]
    assert c["fix"] == ["tools/a.py:2 F401 `sys` imported but unused", "tools/c.py:1 F401 `re` imported but unused"]
    assert doc["failures"] == [{"commit": head, "pr": None, "why": c["why"], "fix": c["fix"]}]


def test_a_fixed_ruff_finding_passes_and_does_not_pay_for_another_rule(both_ruffed):
    both_ruffed.commit("fixes a.py's unused import", {"tools/a.py": "A = 1\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["linted"], c["why"]) == ("passed", ["tools/a.py"],
                                                     "ruff finds nothing new in the 1 Python file(s) it changes")
    both_ruffed.commit("and reads an undefined name", {"tools/a.py": "A = B\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["tools/a.py:1 F821 Undefined name `B`"])


def test_a_ruff_finding_in_a_lint_module_a_change_edits_is_judged_on_its_own_copy(both_ruffed):
    """The gate runs the base's copy of a lint module the change edits, and both ruff lints read that copy too: a change
    that fixes each module's finding and drops its entry passes, judged on its own copy, and one that edits the module, keeps
    its finding and drops its entry fails, naming the finding where the change's copy holds it."""
    util, toy = "tests/lint/_util.py", LINT.removeprefix(f"{G.INTEGRATION}/")
    f541 = {"file": toy, "kind": "F541", "symbol": "<module>", "detail": "f-string without any placeholders"}
    both_ruffed.git("checkout", "-q", "main")
    both_ruffed.commit("a lint module of each directory with an f-string, listed", {
        LINT: TOY_LINT + 'X = f"x"\n', util: 'X = f"x"\n', RUFF_JSON: _allowlist(F401, f541),
        REPO_RUFF_JSON: _allowlist(TOOLS_F401, {"file": util, "rule": "F541", "count": 1})})
    both_ruffed.git("checkout", "-q", "-B", "pr")
    both_ruffed.commit("fixes both and drops their entries", {
        LINT: TOY_LINT + 'X = "x"\n', util: 'X = "x"\n', RUFF_JSON: _allowlist(F401), REPO_RUFF_JSON: _allowlist(TOOLS_F401)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["p01_core", "ruff", "repo_ruff"]), c["why"]
    assert (f"finds 3 violation(s) in the base's copy of {util}, {toy}, which it runs in place of the change's, and the "
            "change's own copy is within its list") in c["why"], c["why"]
    both_ruffed.git("checkout", "-q", "-B", "pr", "main")
    both_ruffed.commit("edits the module, keeps its finding, and drops its entry", {
        LINT: TOY_LINT + '# kept\nX = f"x"\n', RUFF_JSON: _allowlist(F401)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", [
        f"ruff: {toy}: 1 F541 found, 0 allowed",
        f"ruff: {toy}:{len(TOY_LINT.splitlines()) + 2} [F541] f-string without any placeholders"]), c["why"]


def test_a_vllm_rules_finding_in_a_lint_module_a_change_edits_is_judged_on_its_own_copy(ruffed):
    """As ruff's: the lint under vLLM's rules reads the base's copy of a lint module the change edits too.  A change that
    fixes one of the module's two findings and drops its entry passes, judged on its own copy against that lint's own list,
    which still holds the other; and one that edits the module, keeps both and drops an entry fails, naming the findings
    where the change's copy holds them.  Ruff runs on that copy from the integration, as the lint runs it, so a path the
    config names means there what it means to the lint: here `print` is flagged under ``tests/`` alone."""
    toy, n = LINT.removeprefix(f"{G.INTEGRATION}/"), len(TOY_LINT.splitlines())
    say, tell = ({"file": toy, "kind": "T201", "symbol": f, "detail": "`print` found"} for f in ("say", "tell"))
    says, tells = 'def say():\n    print("x")\n', 'def tell():\n    print("y")\n'
    ruffed.git("checkout", "-q", "main")
    ruffed.commit("ruff with vLLM's rules, and a lint module that prints twice, listed", {
        RUFF_VLLM_LINT: (REAL_LINT / "test_ruff_vllm.py").read_text(), RUFF_VLLM_JSON: _allowlist(say, tell),
        G.RUFF_VLLM_CONFIG: RUFF_VLLM_TOML + '[lint.per-file-ignores]\n"!tests/**" = ["T201"]\n',
        LINT: TOY_LINT + says + tells, "adapters/vllm/verity_vllm/b.py": "print(1)\n"})
    ruffed.git("checkout", "-q", "-B", "pr")
    ruffed.commit("fixes one and drops its entry", {LINT: TOY_LINT + tells, RUFF_VLLM_JSON: _allowlist(tell)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["p01_core", "ruff_vllm"]), c["why"]
    assert (f"finds 1 violation(s) in the base's copy of {toy}, which it runs in place of the change's, and the "
            "change's own copy is within its list") in c["why"], c["why"]
    ruffed.git("checkout", "-q", "-B", "pr", "main")
    ruffed.commit("edits the module, keeps both findings, and drops an entry", {
        LINT: TOY_LINT + "# kept\n" + says + tells, RUFF_VLLM_JSON: _allowlist(tell)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", [
        f"ruff_vllm: {toy}: 2 T201 found, 1 allowed", f"ruff_vllm: {toy}:{n + 3} [T201] `print` found",
        f"ruff_vllm: {toy}:{n + 5} [T201] `print` found"]), c["why"]


def _repo_lint(repo: Repo) -> str | None:
    """the repository's ruff lint, as the root suite runs it after landing, on ``repo``'s checkout: its failure, or None"""
    r = subprocess.run([sys.executable, "-c", ("import sys; sys.path.insert(0, 'tests/lint'); import test_repo_ruff as T; "
                         "T.test_no_new_findings()")], cwd=repo.root, capture_output=True, text=True)
    return None if r.returncode == 0 else r.stderr


def _move_entries(repo: Repo, old: str, new: str) -> str:
    """``--move``, the fix the gate names: the list with ``old``'s entries under ``new``"""
    subprocess.run([sys.executable, REPO_RUFF_LINT, "--move", old, new], cwd=repo.root, check=True, capture_output=True)
    return (repo.root / REPO_RUFF_JSON).read_text()


def test_a_finding_moved_to_another_function_in_its_file_passes_the_gate_and_the_repository_lint(both_ruffed):
    """Both count per (file, rule), so a refactor that moves an allowed finding into a new function changes neither."""
    both_ruffed.git("checkout", "-q", "main")
    d841 = {"file": "tools/d.py", "rule": "F841", "count": 1}
    both_ruffed.commit("d.py, whose unused local the list allows", {"tools/d.py": "def f():\n    x = 1\n",
                                                                    REPO_RUFF_JSON: _allowlist(TOOLS_F401, d841)})
    both_ruffed.git("checkout", "-q", "-B", "pr")
    both_ruffed.commit("the unused local moves to a new function", {
        "tools/d.py": "def f():\n    return g()\n\n\ndef g():\n    x = 1\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["linted"]) == ("passed", ["tools/d.py"]), c["why"]
    assert _repo_lint(both_ruffed) is None


def test_a_rename_that_moves_its_entry_passes_the_gate_and_the_repository_lint_and_a_copy_is_new(both_ruffed):
    """``--move`` carries the entry; the gate pools it within its rule, the base's lint agrees, and the source lint sees the
    renamed file's finding covered at its new path."""
    both_ruffed.commit("moves a.py, with a line more", {"tools/a.py": None, "tools/sub/a.py": "import os\nA = 1\nB = 2\n"})
    both_ruffed.commit("its entry", {REPO_RUFF_JSON: _move_entries(both_ruffed, "tools/a.py", "tools/sub/a.py")})
    assert '{"file": "tools/sub/a.py", "rule": "F401", "count": 1}' in (both_ruffed.root / REPO_RUFF_JSON).read_text()
    assert both_ruffed.git("diff", "--name-status", "-M", "main", "pr", "--", "tools").startswith("R")
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["touched"], c["lints"], c["linted"]) == ("passed", [REPO_RUFF_JSON], ["repo_ruff"],
                                                                     ["tools/sub/a.py"]), c["why"]
    assert _repo_lint(both_ruffed) is None
    both_ruffed.git("checkout", "-q", "-b", "copy", "main")
    both_ruffed.commit("copies a.py", {"tools/sub/a.py": "import os\nA = 1\nB = 2\n"})
    [c] = G.gate("main", "copy")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", ["tools/sub/a.py:1 F401 `os` imported but unused"])


def test_a_rename_that_leaves_its_entry_behind_fails_naming_the_entry_to_move_even_as_a_pure_move(both_ruffed):
    """The repository's lint reads no history, so it would fail the repository's suite; the gate fails the change that leaves
    the entry, naming it."""
    both_ruffed.commit("moves a.py as it is", {"tools/a.py": None, "tools/sub/a.py": "import os\nA = 1\n"})
    assert G.is_pure_move(G.changes("main", "pr")[0], "main")
    [c] = G.gate("main", "pr")["changes"]
    assert c["verdict"] == "failed", c["why"]
    assert c["why"].startswith(f"renames Python whose findings {REPO_RUFF_JSON} allows at the old path, and leaves 1 entry "
                               f"behind"), c["why"]
    assert c["fix"] == [(f'{REPO_RUFF_JSON}: move {{"file": "tools/a.py", "rule": "F401", "count": 1}} to "tools/sub/a.py", as '
                         f"the change moves the file (`python3 {REPO_RUFF_LINT} --move tools/a.py tools/sub/a.py`)")]
    assert "tools/sub/a.py [F401]: 1 found, 0 allowed" in (_repo_lint(both_ruffed) or "")
    both_ruffed.commit("its entry", {REPO_RUFF_JSON: _move_entries(both_ruffed, "tools/a.py", "tools/sub/a.py")})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("passed", []), c["why"]
    assert _repo_lint(both_ruffed) is None


def test_a_file_the_change_does_not_touch_is_never_linted(both_ruffed, monkeypatch):
    """a.py's finding is untouched; the integration's files and a deleted one are left to others."""
    seen: list[list[str]] = []
    monkeypatch.setattr(G, "ruff_findings", _noting(seen, lambda ruff, config, files: sorted(files), G.ruff_findings))
    both_ruffed.git("checkout", "-q", "main")
    both_ruffed.commit("definitions", {"verity/definitions/word.py": "X = 1\n"})
    both_ruffed.git("checkout", "-q", "-B", "pr")
    both_ruffed.commit("a clean file, and others the lint doesn't hold", {
        "tools/c.py": "C = 1\n", "adapters/vllm/verity_vllm/c.py": "import os\n",
        "verity/definitions/tiny.py": "import os\n", "verity/definitions/word.py": None, "tools/notes.md": "import os\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["linted"]) == ("passed", ["tools/c.py"]), c["why"]
    assert seen == [[], ["tools/c.py"]]
    both_ruffed.git("checkout", "-q", "-b", "docs", "main")
    seen.clear()
    both_ruffed.commit("docs only", {"tools/notes.md": "x\n"})
    assert [(c["verdict"], c["linted"]) for c in G.gate("main", "docs")["changes"]] == [("untouched", [])] and seen == []


def test_only_the_top_change_is_linted_as_each_member_below_was_at_its_own_merge(both_ruffed, verdict_cache):
    """What the lint finds is kept apart from the verdict: #1's verdict kept from below the top neither fails by it there nor
    spares it the lint at the top."""
    base = both_ruffed.git("rev-parse", "main")
    one = both_ruffed.commit("one: an unused import", {"tools/b.py": "import sys\n"})
    both_ruffed.git("checkout", "-q", "-b", "two", base)
    two = both_ruffed.commit("two: clean", {"tools/c.py": "C = 1\n"})
    both_ruffed.git("checkout", "-q", "-b", "next", base)
    m1 = both_ruffed.queue(1, "one", one)
    assert _verdicts(G.gate("main", "next")) == [("#1", "failed")]
    both_ruffed.queue(2, "two", two)
    doc = G.gate("main", "next")
    assert [(c["name"], c["verdict"], c["linted"]) for c in doc["changes"]] == [("#1", "untouched", None),
                                                                                ("#2", "passed", ["tools/c.py"])]
    assert list(verdict_cache.glob("*.json")) and _verdicts(G.gate("main", m1)) == [("#1", "failed")]


def test_with_no_ruff_to_run_the_gate_cannot_judge_and_names_the_hosts_fix(both_ruffed, monkeypatch, capsys):
    both_ruffed.commit("docs only", {"tools/notes.md": "x\n"})
    monkeypatch.setattr(G, "has_ruff", lambda python, pin, exact: False)
    monkeypatch.setattr(G.shutil, "which", lambda name: None)
    assert G.main(["main", "pr", "--json"]) == 0
    both_ruffed.commit("a clean file", {"tools/c.py": "C = 1\n"})
    assert G.main(["main", "pr", "--json"]) == 2
    err = capsys.readouterr().err
    assert "there is no `uv` to provide it" in err and f"--break-system-packages {both_ruffed.pin}`" in err, err


REPO_ENV_LINT = "tests/lint/test_repo_env_reads.py"
REPO_ENV_JSON = "tests/lint/allowlists/repo_env_reads.json"
TOOLS_A = {"file": "tools/a.py", "count": 1}
A_READS = "import os\nA = os.environ.get('A')\n"


@pytest.fixture
def env_linted(repo) -> Repo:
    """``repo`` with the repository's real environment-read lint: tools/a.py's read of A is its one, which the list allows."""
    repo.commit("env reads", {REPO_ENV_LINT: (SCRIPT.parents[3] / REPO_ENV_LINT).read_text(),
                              REPO_ENV_JSON: _allowlist(TOOLS_A), "tools/a.py": A_READS})
    repo.git("checkout", "-q", "-b", "pr")
    return repo


def _repo_env_fix() -> str:
    """the fix the repository's environment-read lint names, which the gate quotes from the base's copy"""
    spec = importlib.util.spec_from_file_location("repo_env_reads", SCRIPT.parents[3] / REPO_ENV_LINT)
    assert spec is not None and spec.loader is not None, REPO_ENV_LINT
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    return lint.FIX


def _env_lint(repo: Repo) -> str | None:
    """the repository's environment-read lint, as the root suite runs it after landing, on ``repo``'s checkout"""
    r = subprocess.run([sys.executable, "-c", ("import sys; sys.path.insert(0, 'tests/lint'); import test_repo_env_reads as T; "
                         "T.test_no_new_reads()")], cwd=repo.root, capture_output=True, text=True)
    return None if r.returncode == 0 else r.stderr


def test_a_new_direct_environment_read_fails_naming_it_and_the_settings_registry_and_one_there_passes(env_linted):
    """Reads outside the lint's scope (the integration's) are left to others."""
    env_linted.commit("a second read in a.py, a new file with one, and reads the lint doesn't hold", {
        "tools/a.py": A_READS + "B = os.environ['B']\n", "tools/c.py": "import os\nif 'C' in os.environ:\n    pass\n",
        "adapters/vllm/verity_vllm/c.py": "import os\nX = os.getenv('X')\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["touched"], c["counted"].get("repo_env_reads")) == ("failed", [], ["tools/a.py", "tools/c.py"]), c["why"]
    assert c["why"].startswith(f"adds 2 direct environment read(s) to the Python it changes, as the base's {REPO_ENV_LINT} "
                               f"counts them (tools/a.py 1 -> 2; tools/c.py 0 -> 1): {_repo_env_fix()}"), c["why"]
    assert "settings registry" in _repo_env_fix()
    assert c["fix"] == ["tools/a.py:3 B (os.environ[...])", "tools/c.py:2 C (in os.environ)"]
    assert "tools/c.py: 1 found, 0 allowed" in (_env_lint(env_linted) or "")
    env_linted.commit("reads them through the area's registry", {
        "tools/a.py": A_READS, "tools/c.py": "from tools import settings\nif settings.C:\n    pass\n",
        "tools/settings.py": "import os\nB = os.environ['B']\nC = 'C' in os.environ\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["counted"].get("repo_env_reads"), c["why"]) == ("passed", ["tools/c.py", "tools/settings.py"],
                                                       ("no direct environment read is new in the 2 Python file(s) it "
                                                        "changes")), c["why"]
    assert _env_lint(env_linted) is None


def test_a_read_moved_within_its_file_passes_the_gate_and_the_repository_lint(env_linted):
    env_linted.commit("A's read moves into a function", {"tools/a.py": "import os\n\n\ndef a():\n    return os.environ.get('A')\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["counted"].get("repo_env_reads")) == ("passed", ["tools/a.py"]), c["why"]
    assert _env_lint(env_linted) is None


def test_a_rename_must_move_its_entry_which_passes_both_and_fails_naming_the_entry_when_it_doesnt(env_linted):
    env_linted.commit("moves a.py as it is", {"tools/a.py": None, "tools/sub/a.py": A_READS})
    [c] = G.gate("main", "pr")["changes"]
    assert c["verdict"] == "failed" and c["why"].startswith(
        f"renames Python whose direct environment reads {REPO_ENV_JSON} allows at the old path, and leaves 1 entry behind: "
        f"{REPO_ENV_LINT} counts a file's direct environment reads under its path, so it would fail after landing"), c["why"]
    assert c["fix"] == [(f'{REPO_ENV_JSON}: move {{"file": "tools/a.py", "count": 1}} to "tools/sub/a.py", as the change '
                         f"moves the file (`python3 {REPO_ENV_LINT} --move tools/a.py tools/sub/a.py`)")]
    assert "tools/sub/a.py: 1 found, 0 allowed" in (_env_lint(env_linted) or "")
    subprocess.run([sys.executable, REPO_ENV_LINT, "--move", "tools/a.py", "tools/sub/a.py"], cwd=env_linted.root, check=True,
                   capture_output=True)
    env_linted.commit("its entry", {REPO_ENV_JSON: (env_linted.root / REPO_ENV_JSON).read_text()})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"], c["fix"]) == ("passed", ["repo_env_reads"], []), c["why"]
    assert _env_lint(env_linted) is None


def test_a_registry_mark_counts_from_the_changes_base_so_it_lands_before_the_reads_it_exempts(env_linted):
    mark = {"file": "tools/config.py", "registry": True}
    reads = {"tools/config.py": "import os\nB = os.environ.get('B')\n"}
    env_linted.commit("marks config.py a registry, and reads B there", {REPO_ENV_JSON: _allowlist(TOOLS_A, mark), **reads})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["touched"], c["fix"]) == ("failed", [REPO_ENV_JSON], ["tools/config.py:2 B (os.environ.get)"])
    assert "(a mark counts from the change's base, so a new one lands before what it exempts)" in c["why"], c["why"]
    assert _env_lint(env_linted) is None
    env_linted.git("checkout", "-q", "-b", "mark", "main")
    env_linted.commit("the mark alone", {REPO_ENV_JSON: _allowlist(TOOLS_A, mark)})
    [c] = G.gate("main", "mark")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["repo_env_reads"]), c["why"]
    env_linted.git("checkout", "-q", "main")
    env_linted.git("merge", "-q", "--ff-only", "mark")
    env_linted.git("checkout", "-q", "-b", "reads")
    env_linted.commit("reads B in the registry", reads)
    [c] = G.gate("main", "reads")["changes"]
    assert (c["verdict"], c["counted"].get("repo_env_reads")) == ("passed", ["tools/config.py"]), c["why"]


def test_the_environment_read_list_may_not_grow(env_linted):
    b = {"file": "tools/b.py", "count": 1}
    env_linted.commit("a new read, allowed", {"tools/b.py": "import os\nos.getenv('B')\n", REPO_ENV_JSON: _allowlist(TOOLS_A, b)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["touched"]) == ("failed", [REPO_ENV_JSON]), c["why"]
    assert c["why"].startswith(f"grows {REPO_ENV_JSON} 1 -> 2 in total"), c["why"]
    assert c["fix"] == [f"{REPO_ENV_JSON}: {G.key(b)} 0 -> 1"]


def test_loosening_the_environment_read_lint_fails_naming_what_the_base_lint_finds(env_linted):
    lint = (SCRIPT.parents[3] / REPO_ENV_LINT).read_text().replace('"adapters/vllm/",)', '"adapters/vllm/", "tools/")')
    env_linted.commit("leaves tools/ out and drops its entry", {REPO_ENV_LINT: lint, REPO_ENV_JSON: _allowlist()})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"], c["fix"]) == ("failed", ["repo_env_reads"], [
        "repo_env_reads: tools/a.py: 1 found, 0 allowed", "repo_env_reads: tools/a.py:2 A (os.environ.get)"]), c["why"]


def test_the_source_checks_are_kept_beside_the_verdict_and_never_fail_a_member_below_the_top(env_linted, verdict_cache,
                                                                                            monkeypatch):
    base = env_linted.git("rev-parse", "main")
    one = env_linted.commit("one: a new read", {"tools/b.py": "import os\nB = os.environ['B']\n"})
    env_linted.git("checkout", "-q", "-b", "two", base)
    two = env_linted.commit("two: clean", {"tools/c.py": "C = 1\n"})
    env_linted.git("checkout", "-q", "-b", "next", base)
    m1 = env_linted.queue(1, "one", one)
    assert _verdicts(G.gate("main", "next")) == [("#1", "failed")]
    env_linted.queue(2, "two", two)
    doc = G.gate("main", "next")
    assert [(c["name"], c["verdict"], c["counted"].get("repo_env_reads")) for c in doc["changes"]] == [("#1", "untouched", None),
                                                                                  ("#2", "passed", ["tools/c.py"])]
    monkeypatch.setattr(G, "find", lambda c, base: pytest.fail("judged again"))
    assert _verdicts(G.gate("main", m1)) == [("#1", "failed")] and len(list(verdict_cache.glob("*.json"))) == 4


def test_a_base_whose_environment_read_lint_does_not_load_cannot_judge(env_linted, capsys):
    env_linted.git("checkout", "-q", "main")
    env_linted.commit("breaks the lint", {REPO_ENV_LINT: "raise SystemError('no')\n"})
    env_linted.git("checkout", "-q", "-B", "pr")
    env_linted.commit("a clean file", {"tools/c.py": "C = 1\n"})
    assert G.main(["main", "pr", "--json"]) == 2
    assert f"{REPO_ENV_LINT} at " in capsys.readouterr().err


REPO_SILENT_LINT = "tests/lint/test_repo_silent_excepts.py"
REPO_SILENT_JSON = "tests/lint/allowlists/repo_silent_excepts.json"
TOOLS_S = {"file": "tools/s.py", "count": 1}
S_SWALLOWS = "def s(p):\n    try:\n        return open(p).read()\n    except OSError:\n        return None\n"


@pytest.fixture
def silent_linted(repo) -> Repo:
    """``repo`` with the repository's real silent-except lint: tools/s.py's ``except OSError: return None`` is its one, which
    the list allows."""
    repo.commit("silent excepts", {REPO_SILENT_LINT: (SCRIPT.parents[3] / REPO_SILENT_LINT).read_text(),
                                   REPO_SILENT_JSON: _allowlist(TOOLS_S), "tools/s.py": S_SWALLOWS})
    repo.git("checkout", "-q", "-b", "pr")
    return repo


def _silent_lint(repo: Repo) -> str | None:
    """the repository's silent-except lint, as the root suite runs it after landing, on ``repo``'s checkout"""
    r = subprocess.run([sys.executable, "-c", ("import sys; sys.path.insert(0, 'tests/lint'); "
                         "import test_repo_silent_excepts as T; T.test_no_new_silent_excepts()")],
                       cwd=repo.root, capture_output=True, text=True)
    return None if r.returncode == 0 else r.stderr


def test_a_new_swallowing_handler_fails_naming_it_and_the_fix_and_one_that_records_its_outcome_passes(silent_linted):
    """Typed handlers count like broad ones; handlers outside the lint's scope (the integration's) are left to others."""
    silent_linted.commit("a second handler in s.py, a new file with one, and handlers the lint doesn't hold", {
        "tools/s.py": S_SWALLOWS + "    except ValueError:\n        pass\n",
        "tools/t.py": "def t(xs, out):\n    for x in xs:\n        try:\n            out[x] = int(x)\n"
                      "        except ValueError:\n            continue\n",
        "adapters/vllm/verity_vllm/c.py": "try:\n    X = 1\nexcept Exception:\n    pass\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["touched"], c["counted"].get("repo_silent_excepts")) == ("failed", [], ["tools/s.py", "tools/t.py"]), \
        c["why"]
    assert c["why"] == (f"adds 2 silent except handler(s) to the Python it changes, as the base's {REPO_SILENT_LINT} counts them "
                        "(tools/s.py 1 -> 2; tools/t.py 0 -> 1): Re-raise; or record the outcome in your area's typed record or "
                        "log, saying what was caught and what the code did instead; or narrow it away: test for the case you "
                        "expect (`path.exists()`, `d.get(k)`) and let the rest raise, since a narrower handler that still "
                        "swallows still counts (why: adapters/vllm/tests/lint/README.md (Recorded outcomes)). If the file "
                        f"moved, move its entry in the same change: `python3 {REPO_SILENT_LINT} --move OLD NEW`."), c["why"]
    assert c["fix"] == ["tools/s.py:6 except ValueError: pass", "tools/t.py:5 except ValueError: continue"]
    assert "tools/t.py: 1 found, 0 allowed" in (_silent_lint(silent_linted) or "")
    silent_linted.commit("re-raises in s.py, and records what t.py skips", {
        "tools/s.py": S_SWALLOWS + "    except ValueError:\n        raise\n",
        "tools/t.py": "def t(xs, out, skipped):\n    for x in xs:\n        try:\n            out[x] = int(x)\n"
                      "        except ValueError as e:\n            skipped.append(f'{x}: {e}')\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["counted"].get("repo_silent_excepts"), c["why"]) == (
        "passed", ["tools/s.py", "tools/t.py"], "no silent except handler is new in the 2 Python file(s) it changes"), c["why"]
    assert _silent_lint(silent_linted) is None


def test_a_renamed_file_must_move_its_silent_except_entry_and_fails_naming_it_when_it_doesnt(silent_linted):
    silent_linted.commit("moves s.py as it is", {"tools/s.py": None, "tools/sub/s.py": S_SWALLOWS})
    [c] = G.gate("main", "pr")["changes"]
    assert c["verdict"] == "failed" and c["why"].startswith(
        f"renames Python whose silent except handlers {REPO_SILENT_JSON} allows at the old path, and leaves 1 entry behind: "
        f"{REPO_SILENT_LINT} counts a file's silent except handlers under its path, so it would fail after landing"), c["why"]
    assert c["fix"] == [(f'{REPO_SILENT_JSON}: move {{"file": "tools/s.py", "count": 1}} to "tools/sub/s.py", as the change '
                         f"moves the file (`python3 {REPO_SILENT_LINT} --move tools/s.py tools/sub/s.py`)")]
    assert "tools/sub/s.py: 1 found, 0 allowed" in (_silent_lint(silent_linted) or "")
    subprocess.run([sys.executable, REPO_SILENT_LINT, "--move", "tools/s.py", "tools/sub/s.py"], cwd=silent_linted.root,
                   check=True, capture_output=True)
    silent_linted.commit("its entry", {REPO_SILENT_JSON: (silent_linted.root / REPO_SILENT_JSON).read_text()})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"], c["fix"]) == ("passed", ["repo_silent_excepts"], []), c["why"]
    assert _silent_lint(silent_linted) is None


def test_the_silent_except_list_may_not_grow_and_loosening_its_lint_fails_naming_what_the_base_lint_finds(silent_linted):
    b = {"file": "tools/b.py", "count": 1}
    silent_linted.commit("a new handler, allowed", {"tools/b.py": "try:\n    B = 1\nexcept KeyError:\n    B = None\n",
                                                    REPO_SILENT_JSON: _allowlist(TOOLS_S, b)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["touched"]) == ("failed", [REPO_SILENT_JSON]), c["why"]
    assert c["why"].startswith(f"grows {REPO_SILENT_JSON} 1 -> 2 in total"), c["why"]
    assert c["fix"] == [f"{REPO_SILENT_JSON}: {G.key(b)} 0 -> 1"]
    silent_linted.git("checkout", "-q", "-B", "pr", "main")
    lint = (SCRIPT.parents[3] / REPO_SILENT_LINT).read_text().replace('"adapters/vllm/",)', '"adapters/vllm/", "tools/")')
    silent_linted.commit("leaves tools/ out and drops its entry", {REPO_SILENT_LINT: lint, REPO_SILENT_JSON: _allowlist()})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"], c["fix"]) == ("failed", ["repo_silent_excepts"], [
        "repo_silent_excepts: tools/s.py: 1 found, 0 allowed",
        "repo_silent_excepts: tools/s.py:4 except OSError: return None"]), c["why"]


def _caught_up(listed: str, total: str, n: int) -> str:
    return (f"{listed} {total} in total, {n} of it in {n} key(s) catching up to what its lint finds in the base's tree (a list "
            f"that records its base's findings adds none)")


def test_a_counted_lint_s_list_catches_up_to_what_its_base_holds_and_no_further(silent_linted):
    """A handler that landed with no entry (a PR on `next` before the lint did) may be listed in any file, the list's
    own or one the change edits; a count past what the base's lint finds there, or a site the lint doesn't count (an edit to
    the lint's module buys nothing), is growth."""
    u = {"file": "tools/u.py", "count": 1}
    silent_linted.git("checkout", "-q", "main")
    silent_linted.commit("a handler with no entry", {"tools/u.py": S_SWALLOWS})
    silent_linted.git("checkout", "-q", "-B", "pr")
    silent_linted.commit("the list takes it", {REPO_SILENT_JSON: _allowlist(TOOLS_S, u)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"], c["pooled"]) == (
        "passed", ["repo_silent_excepts"], [_caught_up(REPO_SILENT_JSON, "1 -> 2", 1)]), c["why"]
    assert _silent_lint(silent_linted) is None
    silent_linted.commit("past it", {REPO_SILENT_JSON: _allowlist(TOOLS_S, {**u, "count": 2})})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", [f"{REPO_SILENT_JSON}: {G.key(u)} 0 -> 2"]), c["why"]
    assert c["why"].startswith(f"grows {REPO_SILENT_JSON} 1 -> 3 in total"), c["why"]
    silent_linted.commit("a second handler in u.py, listed", {"tools/u.py": S_SWALLOWS + "    except ValueError:\n        pass\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", [f"{REPO_SILENT_JSON}: {G.key(u)} 0 -> 2"]), c["why"]
    v = {"file": "tools/v.py", "count": 1}
    silent_linted.git("checkout", "-q", "-B", "pr", "main")
    silent_linted.commit("a suppress(...) the base's lint doesn't count, listed beside an edit to the lint", {
        "tools/v.py": "import contextlib, os\n\n\ndef rm(p):\n    with contextlib.suppress(OSError):\n        os.remove(p)\n",
        REPO_SILENT_LINT: (SCRIPT.parents[3] / REPO_SILENT_LINT).read_text() + "\n# it counts suppress(...) too\n",
        REPO_SILENT_JSON: _allowlist(TOOLS_S, v)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", [f"{REPO_SILENT_JSON}: {G.key(v)} 0 -> 1"]), c["why"]


def test_the_repositorys_ruff_list_catches_up_to_what_its_base_holds_and_no_further(both_ruffed):
    b401 = {**TOOLS_F401, "file": "tools/b.py"}
    both_ruffed.git("checkout", "-q", "main")
    both_ruffed.commit("a finding with no entry", {"tools/b.py": "import sys\n"})
    both_ruffed.git("checkout", "-q", "-B", "pr")
    both_ruffed.commit("the list takes it", {REPO_RUFF_JSON: _allowlist(TOOLS_F401, b401)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["pooled"]) == ("passed", [_caught_up(REPO_RUFF_JSON + " F401", "1 -> 2", 1)]), c["why"]
    assert _repo_lint(both_ruffed) is None
    both_ruffed.commit("past it", {REPO_RUFF_JSON: _allowlist(TOOLS_F401, {**b401, "count": 2})})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", [f"{REPO_RUFF_JSON}: {G.key(b401)} 0 -> 2"]), c["why"]


def test_both_counted_lints_count_the_top_change_each_naming_its_own_fix(env_linted):
    env_linted.git("checkout", "-q", "main")
    env_linted.commit("silent excepts", {REPO_SILENT_LINT: (SCRIPT.parents[3] / REPO_SILENT_LINT).read_text(),
                                         REPO_SILENT_JSON: _allowlist()})
    env_linted.git("checkout", "-q", "-B", "pr")
    env_linted.commit("reads B where it swallows a missing one", {
        "tools/b.py": "import os\ntry:\n    B = int(os.environ['B'])\nexcept (KeyError, ValueError):\n    B = 0\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["counted"]) == ("failed", {"repo_env_reads": ["tools/b.py"],
                                                       "repo_silent_excepts": ["tools/b.py"]}), c["why"]
    assert c["fix"] == ["tools/b.py:3 B (os.environ[...])", "tools/b.py:4 except (KeyError, ValueError): B = 0"]
    first, second = c["why"].split("; and ")
    assert first.startswith("adds 1 direct environment read(s)") and "settings registry" in first, c["why"]
    assert second.startswith(f"adds 1 silent except handler(s) to the Python it changes, as the base's {REPO_SILENT_LINT} "
                             "counts them (tools/b.py 0 -> 1): Re-raise;"), c["why"]


CALLERS_LINT = "adapters/vllm/tests/lint/test_p16_cli_callers.py"
CLI = "adapters/vllm/verity_vllm/pipeline/cli.py"
ROW = {"row": "benchmarks/row.sh"}


def _cli(*names: str) -> str:
    """cli.py's COMMANDS table, each command running a module of its own"""
    return "COMMANDS: dict[str, tuple[str, str]] = {\n" + "".join(
        f'    "{n}": ("verity_vllm.{n}", "help"),\n' for n in names) + "}\n"


def _callers(public: dict[str, str]) -> str:
    return json.dumps({"rule": "P16", "public": public, "entries": []}, indent=1) + "\n"


@pytest.fixture
def callers_linted(repo) -> Repo:
    """``repo`` with the real P16 lint, whose one command, row, is public, run by benchmarks/row.sh."""
    repo.commit("p16", {CALLERS_LINT: (REAL_LINT / "test_p16_cli_callers.py").read_text(),
                        "adapters/vllm/tests/lint/_ratchet.py": (REAL_LINT / "_ratchet.py").read_text(),
                        CLI: _cli("row"), G.CALLERS_LIST: _callers(ROW), "benchmarks/row.sh": "verity-vllm row --x 1\n"})
    repo.git("checkout", "-q", "-b", "pr")
    return repo


def test_a_public_command_a_change_adds_passes_when_its_caller_runs_it_and_fails_when_it_does_not(callers_linted):
    """A command run only from benchmarks/, as weights-tie is from benchmarks/prove_path/run.sh: the base's P16 lint, run on
    the change's tree, reads the change's public table, and the gate checks each caller the table gains (a new command's, or
    one moved to another file) as that lint finds a call."""
    callers_linted.commit("adds tie, run from benchmarks", {
        CLI: _cli("row", "tie"), G.CALLERS_LIST: _callers({**ROW, "tie": "benchmarks/tie/run.sh"}),
        "benchmarks/tie/run.sh": 'WT="$PY -m verity_vllm.pipeline.cli tie"\n$WT --x 1\n'})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["touched"], c["lints"]) == ("passed", [G.CALLERS_LIST], ["p16_cli_callers"]), c["why"]
    assert "public, each run by its caller: tie (benchmarks/tie/run.sh)" in c["why"], c["why"]
    callers_linted.git("checkout", "-q", "-b", "uncalled", "main")
    callers_linted.commit("adds tie, which its caller only names, and gives row a caller that isn't there", {
        CLI: _cli("row", "tie"), G.CALLERS_LIST: _callers({"row": "benchmarks/gone.sh", "tie": "benchmarks/tie/run.sh"}),
        "benchmarks/tie/run.sh": "# run verity-vllm tie by hand\necho tie\n"})
    [c] = G.gate("main", "uncalled")["changes"]
    assert (c["verdict"], c["lints"]) == ("failed", []), c["why"]
    assert c["why"].startswith(f"declares 2 command(s) public in {G.CALLERS_LIST} that the file it names doesn't run"), c["why"]
    assert c["fix"] == [f'{G.CALLERS_LIST}: public "row": "benchmarks/gone.sh" is no file in the change\'s tree',
                        f'{G.CALLERS_LIST}: public "tie": benchmarks/tie/run.sh doesn\'t run `verity-vllm tie`']


A_PY = "adapters/vllm/verity_vllm/a.py"


def test_a_change_adds_no_more_noqa_lines_than_it_removes(repo):
    repo.commit("noqas", {A_PY: "import os  # noqa: F401\nA = 1\n", "adapters/vllm/verity_vllm/b.py": "B = 1  # noqa: E1\n"})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("a new noqa, and one that only names it", {"adapters/vllm/verity_vllm/c.py": (
        '"""Say why in a `# noqa: CODE`."""\nimport sys  # noqa: F401\nX = "# noqa"\n')})
    doc = G.gate("main", "pr")
    [c] = doc["changes"]
    assert (c["verdict"], c["noqa_added"], c["noqa_removed"], c["touched"]) == ("failed", 1, 0, [])
    assert c["why"].startswith("adds 1 `# noqa` line(s) under adapters/vllm/ and removes 0: a noqa is an exception, so "
                               "fix the finding each new one suppresses, or delete a stale noqa elsewhere in the same change")
    assert c["fix"] == ["verity_vllm/c.py:2: import sys  # noqa: F401"] and doc["failures"][0]["fix"] == c["fix"]
    repo.commit("deletes a stale one", {"adapters/vllm/verity_vllm/b.py": "B = 1\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["noqa_added"], c["noqa_removed"]) == ("passed", 1, 1)
    assert c["why"] == "1 `# noqa` line(s) added, 1 removed"


def test_a_new_copy_of_a_noqa_line_is_named_where_the_diff_adds_it(repo):
    bare = "    except Exception:  # noqa: BLE001"
    repo.commit("one", {A_PY: f"try:\n    A = 1\n{bare}\n    A = 0\n"})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("another", {A_PY: f"try:\n    A = 1\n{bare}\n    A = 0\ntry:\n    B = 1\n{bare}\n    B = 0\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["noqa_added"], c["fix"]) == ("failed", 1, [f"verity_vllm/a.py:7: {bare.strip()}"])


def test_a_noqa_that_moves_with_its_code_is_kept_and_a_file_wide_one_counts(repo):
    repo.commit("a noqa", {A_PY: "import os  # noqa: F401\nA = 1\n"})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("moves the import to c.py, edits a line beside it", {
        A_PY: "A = 2\n", "adapters/vllm/verity_vllm/c.py": "C = 0\n\nimport os  # noqa: F401\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["noqa_added"], c["noqa_removed"]) == ("untouched", 0, 0)
    repo.commit("edits the line it's on", {"adapters/vllm/verity_vllm/c.py": "C = 0\n\nimport re  # noqa: F401\n"})
    assert _verdicts(G.gate("main", "pr"))[0][1] == "passed"
    for name, text in [("ruff", "# ruff: noqa\nA = 1\n"), ("flake8", "# flake8: noqa: E1\nA = 1\n"),
                       ("unparsed", "A = (  # NOQA:E1\n")]:
        repo.git("checkout", "-q", "-B", name, "main")
        repo.commit(f"a {name} noqa", {"adapters/vllm/verity_vllm/d.py": text})
        [c] = G.gate("main", name)["changes"]
        assert (c["verdict"], c["fix"]) == ("failed", [f"verity_vllm/d.py:1: {text.splitlines()[0]}"]), name


def test_a_noqa_outside_the_integration_or_in_a_revert_is_not_held(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("elsewhere", {"tools/x.py": "import os  # noqa: F401\n", "adapters/vllm/notes.md": "# noqa\n"})
    assert _verdicts(G.gate("main", "pr"))[0][1] == "untouched"
    repo.git("checkout", "-q", "-b", "revert", "main")
    repo.commit('Revert "Queue: #9 drops a noqa"', {A_PY: "A = 1  # noqa: E1\n"})
    assert _verdicts(G.gate("main", "revert"))[0][1] == "exempt"


def test_the_command_line(repo):
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("feature", {"adapters/vllm/verity_vllm/c.py": "C = 1\n",
                            P01: _allowlist(_entry("verity_vllm/a.py", "A"), _entry("verity_vllm/b.py", "B"),
                                            _entry("verity_vllm/c.py", "C"))})
    def run(*a):
        return subprocess.run([sys.executable, str(SCRIPT), *a], cwd=repo.root, capture_output=True, text=True)

    r = run("main", "pr")
    assert r.returncode == 1 and f"failed: grows {P01} 2 -> 3 entries" in r.stdout and "verity_vllm/c.py" in r.stdout, r.stdout
    r = run("main", "pr", "--json")
    assert r.returncode == 1 and json.loads(r.stdout)["kind"] == G.KIND
    assert run("main", "main").returncode == 0
    r = run("main", "no-such-ref")
    assert r.returncode == 2 and "ratchet gate: git rev-parse" in r.stderr


REPO_LANE_LINT = "tests/lint/test_repo_lane_text.py"
REPO_LANE_JSON = "tests/lint/allowlists/repo_lane_text.json"
DOCS_A = {"file": "docs/a.md", "count": 1}


@pytest.fixture
def lane_linted(repo) -> Repo:
    """``repo`` with the repository's real lane-text lint: docs/a.md's PR number is its one site, which the list allows."""
    repo.commit("lane text", {REPO_LANE_LINT: (SCRIPT.parents[3] / REPO_LANE_LINT).read_text(),
                              REPO_LANE_JSON: _allowlist(DOCS_A), "docs/a.md": "# A\n\nSee #1234.\n"})
    repo.git("checkout", "-q", "-b", "pr")
    return repo


def _lane_lint(repo: Repo) -> str | None:
    """the repository's lane-text lint, as the root suite runs it after landing, on ``repo``'s checkout"""
    r = subprocess.run([sys.executable, "-c", ("import sys; sys.path.insert(0, 'tests/lint'); "
                         "import test_repo_lane_text as T; T.test_no_new_lane_text()")],
                       cwd=repo.root, capture_output=True, text=True)
    return None if r.returncode == 0 else r.stderr


def test_lane_text_is_counted_in_every_kind_of_file_its_lint_reads_naming_where_the_text_goes(lane_linted):
    """Markdown, a Lean comment and a Python docstring count, each by its path; a Python string literal is code, not prose."""
    lane_linted.commit("an attribution, a Linear id, a PR and a literal", {
        "docs/a.md": "# A\n\nSee #1234 (Daniel, 8 Oct).\n", "Lean/B.lean": "-- VLLM-136\ndef b := 1\n",
        "tools/c.py": '"""Built for #4321."""\nX = "#1234"\n', "tools/d.json": '{"pr": "#1234"}\n'})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["counted"].get("repo_lane_text")) == (
        "failed", ["Lean/B.lean", "docs/a.md", "tools/c.py"]), c["why"]
    assert c["why"].startswith(
        f"adds 3 lane-text site(s) to the files it changes, as the base's {REPO_LANE_LINT} counts them (Lean/B.lean 0 -> 1; "
        "docs/a.md 1 -> 2; tools/c.py 0 -> 1): Say what is true now, and put the rest where it lives: history in git "
        "(the commit message), work in Linear (the issue), rulings in Notion (the decisions log), results in the store"), c["why"]
    assert c["fix"] == ["Lean/B.lean:1 linear: VLLM-136", "docs/a.md:3 attribution: (Daniel, 8 Oct)",
                        "tools/c.py:1 pr: #4321"]
    assert "docs/a.md: 2 found, 1 allowed" in (_lane_lint(lane_linted) or "")


def test_a_lane_text_count_that_falls_fails_nobody_and_its_list_may_run_ahead_or_be_pruned(lane_linted):
    """Every area cleans its own while the list stands at its landing counts: a list ahead of its tree is fine, and only
    growth fails."""
    lane_linted.commit("says what is true now", {"docs/a.md": "# A\n\nSee the queue's README.\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["counted"].get("repo_lane_text"), c["why"]) == (
        "passed", ["docs/a.md"], "no lane-text site is new in the 1 file(s) it changes"), c["why"]
    assert _lane_lint(lane_linted) is None
    subprocess.run([sys.executable, REPO_LANE_LINT, "--prune"], cwd=lane_linted.root, check=True, capture_output=True)
    assert json.loads((lane_linted.root / REPO_LANE_JSON).read_text())["entries"] == []
    lane_linted.commit("prunes its entry", {REPO_LANE_JSON: (lane_linted.root / REPO_LANE_JSON).read_text()})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["repo_lane_text"]), c["why"]
    assert c["why"].startswith("3 -> 2 ratchet exceptions"), c["why"]


def test_a_lane_text_list_catches_up_to_what_its_base_holds_and_a_pinned_file_has_no_entry(lane_linted):
    e = {"file": "tools/e.py", "count": 1}
    lane_linted.git("checkout", "-q", "main")
    lane_linted.commit("lane text with no entry", {"tools/e.py": "X = 1  # VLLM-136\n"})
    lane_linted.git("checkout", "-q", "-B", "pr")
    lane_linted.commit("the list takes it", {REPO_LANE_JSON: _allowlist(DOCS_A, e)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"], c["pooled"]) == (
        "passed", ["repo_lane_text"], [_caught_up(REPO_LANE_JSON, "1 -> 2", 1)]), c["why"]
    assert _lane_lint(lane_linted) is None
    lane_linted.git("checkout", "-q", "-B", "pr", "main")
    lane_linted.commit("lane text in AGENTS.md, listed", {"AGENTS.md": "Owned by @vllm (Daniel, 8 Oct).\n",
                                                         REPO_LANE_JSON: _allowlist(DOCS_A, {"file": "AGENTS.md", "count": 1})})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", [f"{REPO_LANE_JSON}: {G.key({'file': 'AGENTS.md'})} 0 -> 1"]), c["why"]
    assert "repo_lane_text.json: AGENTS.md is pinned at zero (PINNED), so it has no entry" in (_lane_lint(lane_linted) or "")


def test_a_lint_that_comes_to_count_more_records_its_list_at_the_files_it_leaves_untouched_and_no_others(lane_linted):
    """The base's lint can't count a new kind, so the rise is taken from a change to the lint's module at each file of the
    base's tree it leaves untouched, where it adds no text; a file it edits, or a rise with no edit to the lint, is
    held as before, and main stays green after it lands."""
    b = {"file": "docs/b.md", "count": 2}
    lane_linted.git("checkout", "-q", "main")
    lane_linted.commit("text the lint will count", {"docs/b.md": "# B\n\nTOY1 and TOY2.\n"})
    lane_linted.git("checkout", "-q", "-B", "pr")
    real = (SCRIPT.parents[3] / REPO_LANE_LINT).read_text()
    lane = '    ("codename", r"\\[lane ", None),\n'
    wider = real.replace(lane, '    ("toy", r"TOY\\d\\b", None),\n' + lane, 1)
    assert wider != real
    lane_linted.commit("counts TOY ids, and records them", {REPO_LANE_LINT: wider, REPO_LANE_JSON: _allowlist(DOCS_A, b)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"], c["pooled"]) == ("passed", ["repo_lane_text"], [
        (f"{REPO_LANE_JSON} 1 -> 3 in total, 2 of it in 1 key(s) at files the change leaves untouched, recorded as the lint "
         "it edits counts them")]), c["why"]
    assert _lane_lint(lane_linted) is None
    lane_linted.commit("and one more in the file", {"docs/b.md": "# B\n\nTOY1, TOY2 and TOY3.\n",
                                                    REPO_LANE_JSON: _allowlist(DOCS_A, {**b, "count": 3})})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", [f"{REPO_LANE_JSON}: {G.key({'file': 'docs/b.md'})} 0 -> 3"]), c["why"]
    lane_linted.git("checkout", "-q", "-B", "pr", "main")
    lane_linted.commit("records them with the lint as it is", {REPO_LANE_JSON: _allowlist(DOCS_A, b)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", [f"{REPO_LANE_JSON}: {G.key({'file': 'docs/b.md'})} 0 -> 2"]), c["why"]


REPO_TERMS_LINT = "tests/lint/test_repo_retired_terms.py"
REPO_TERMS_JSON = "tests/lint/allowlists/repo_retired_terms.json"
TOY_GLOSSARY = '# Glossary\n\n## Older names\n\n| Old name, where it stays | Term |\n|---|---|\n| "teapot" | kettle |\n'


def test_a_retired_term_is_counted_as_the_glossary_of_the_tree_the_gate_runs_in_retires_it(repo):
    """The retired-terms lint, loaded alone from the base, reads the checkout's GLOSSARY.md: a new use in a file the change
    touches is refused with the term to say, and saying it passes."""
    repo.commit("retired terms", {REPO_TERMS_LINT: (SCRIPT.parents[3] / REPO_TERMS_LINT).read_text(), "GLOSSARY.md": TOY_GLOSSARY,
                                  REPO_TERMS_JSON: _allowlist(DOCS_A), "docs/a.md": "# A\n\nA teapot boils it.\n"})
    repo.git("checkout", "-q", "-b", "pr")
    repo.commit("another use", {"docs/a.md": "# A\n\nA teapot boils it, and a teapot pours it.\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["counted"].get("repo_retired_terms")) == ("failed", ["docs/a.md"]), c["why"]
    assert c["fix"] == ["docs/a.md:3 'teapot' is retired; say kettle (GLOSSARY.md)"], c["why"]
    repo.git("checkout", "-q", "-B", "pr", "main")
    repo.commit("says the term", {"docs/a.md": "# A\n\nA kettle boils it.\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["counted"].get("repo_retired_terms")) == ("passed", ["docs/a.md"]), c["why"]


@pytest.fixture
def terms_linted(repo) -> Repo:
    """``repo`` with the repository's real retired-terms lint and a list behind its tree: docs/a.md's use is listed, and
    docs/b.md's two, on lines 5 and 7, are not."""
    repo.commit("retired terms", {REPO_TERMS_LINT: (SCRIPT.parents[3] / REPO_TERMS_LINT).read_text(), "GLOSSARY.md": TOY_GLOSSARY,
                                  REPO_TERMS_JSON: _allowlist(DOCS_A), "docs/a.md": "# A\n\nA teapot boils it.\n",
                                  "docs/b.md": "# B\n\nOne.\nTwo.\nA teapot pours it.\n\nA teapot whistles.\n"})
    repo.git("checkout", "-q", "-b", "pr")
    return repo


#: what a change that touches the list writes beside docs/b.md: the term said in docs/a.md, whose entry goes
SAYS_THE_TERM = {"docs/a.md": "# A\n\nA kettle boils it.\n", REPO_TERMS_JSON: _allowlist()}
B_SITE = "repo_retired_terms: docs/b.md:{} 'teapot' is retired; say kettle (GLOSSARY.md)"


def test_a_site_that_an_edit_above_it_moves_is_the_bases_still_and_is_named_at_the_bases_line(terms_linted):
    """A change that touches the list, so the base's lint runs on its whole tree, and deletes a line above two uses the
    base holds past its list: each use moves up a line and is no new site, so the change passes, and each is named where
    the base holds it, with the change that wrote it."""
    wrote = terms_linted.git("rev-parse", "main").strip()[:12]
    terms_linted.commit("b.md is a line shorter", {
        **SAYS_THE_TERM, "docs/b.md": "# B\n\nOne.\nA teapot pours it.\n\nA teapot whistles.\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["repo_retired_terms"]), c["why"]
    assert ("the base's lint code (repo_retired_terms) finds 3 violation(s) its own tree already holds, which aren't this "
            f"change's: added by {wrote}: {B_SITE.format(5)}, {B_SITE.format(7)}, "
            "repo_retired_terms: docs/b.md: 2 found, 0 allowed") in c["why"], c["why"]


def test_one_more_site_of_a_term_in_a_file_is_the_changes_though_the_others_moved(terms_linted):
    """The same change with a third use of the term at the file's end: the file holds one more site of that term than its
    base does, so the change fails, naming the file's count and one site, the last."""
    terms_linted.commit("b.md is a line shorter and uses the term again", {
        **SAYS_THE_TERM, "docs/b.md": "# B\n\nOne.\nA teapot pours it.\n\nA teapot whistles.\n\nA teapot sings.\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert c["verdict"] == "failed" and "finds 2 violation(s) on its tree" in c["why"], c["why"]
    assert c["fix"] == ["repo_retired_terms: docs/b.md: 3 found, 0 allowed", B_SITE.format(8)], c["why"]


def test_a_file_whose_count_a_mended_site_lowers_is_no_new_violation(terms_linted):
    """The same change with one of the two uses mended: the file's count is lower than its base's and its one site left is
    the base's, so the change passes, and both are named as the base holds them."""
    wrote = terms_linted.git("rev-parse", "main").strip()[:12]
    terms_linted.commit("b.md is a line shorter and says the term once", {
        **SAYS_THE_TERM, "docs/b.md": "# B\n\nOne.\nA teapot pours it.\n\nA kettle whistles.\n"})
    [c] = G.gate("main", "pr")["changes"]
    assert c["verdict"] == "passed", c["why"]
    assert ("finds 2 violation(s) its own tree already holds, which aren't this change's: "
            f"added by {wrote}: {B_SITE.format(5)}, repo_retired_terms: docs/b.md: 2 found, 0 allowed") in c["why"], c["why"]


def test_a_violation_is_compared_by_its_file_and_what_it_is_with_a_count_and_never_by_its_line():
    """`beyond`: of a file's sites of one kind, as many as its base holds are the base's (those at a line the base names
    first, then the earliest), each named at the base's line; a count is the base's while it is no more than the base's
    against the same allowance; a key's violation that names each of its lines counts a site a line; and a violation that
    names neither a line nor a count is the base's when the base has the same one."""
    a, b, count = "l: f.py:3 x", "l: f.py:9 x", "l: f.py: 2 found, 0 allowed"
    assert G.beyond([a], ["l: f.py:4 x"]) == (["l: f.py:4 x"], [])
    assert G.beyond([a, b], ["l: f.py:9 x"]) == ([b], [a])
    assert G.beyond([a, b], ["l: f.py:4 x"]) == (["l: f.py:4 x"], [b])
    assert G.beyond([a, a], [a]) == ([a], [a])
    assert G.beyond([a, b], ["l: f.py:4 x", "l: f.py:5 x", "l: f.py:6 x"]) == (["l: f.py:4 x", "l: f.py:5 x"], [])
    assert G.beyond([a], ["l: g.py:3 x", "l: f.py:3 y", "m: f.py:3 x"]) == ([], [a])
    assert G.beyond([count, a], [count]) == ([count], [a])
    more, fewer = "l: f.py: 3 found, 0 allowed", "l: f.py: 1 found, 0 allowed"
    assert G.beyond([count], [more]) == ([more], []) and G.beyond([count], [fewer]) == ([], [count])
    assert G.beyond([count], ["l: f.py: 3 found, 1 allowed"]) == ([], [count])
    assert G.beyond(["l: f.py [k]: 2 found, 0 allowed"], ["l: f.py [j]: 2 found, 0 allowed"])[0] == []
    keyed, moved = "p: f.py:3,9 [k] s: d  (2 found, 1 allowed)", "p: f.py:4,10 [k] s: d  (2 found, 1 allowed)"
    one, two = "p: f.py:4 [k] s: d", "p: f.py:4,10 [k] s: d"
    assert G.beyond([keyed], [moved]) == ([moved], []) and G.beyond([keyed], [one]) == ([], [keyed])
    assert G.beyond([two], [one]) == ([], [two]) and G.beyond([one], [two]) == ([two], [])
    assert G.beyond(["l: test_x raised E", "l: nothing here"], ["l: test_x raised E"]) == (["l: test_x raised E"], ["l: nothing here"])


def test_a_site_a_lint_below_it_counts_fails_the_pr_that_adds_it_and_never_a_later_one(lane_linted):
    """A PR widens the lane-text lint on a train (`next` here), a second, from `main`, adds a site only that lint counts, which
    `main`'s lint passes, and a third touches the list, so the gate runs the lint on the whole tree, which finds the second's
    line.  A merge on the train is counted with the lints the tree below has, so the second is the one that fails; and a
    violation the tree below already holds is named with the PR that added it, and leaves the third in place."""
    b = {"file": "docs/b.md", "count": 2}
    lane_linted.git("checkout", "-q", "main")
    lane_linted.commit("text the lint will count", {"docs/b.md": "# B\n\nTOY1 and TOY2.\n"})
    main = lane_linted.git("rev-parse", "HEAD")
    lane = '    ("codename", r"\\[lane ", None),\n'
    wider = (SCRIPT.parents[3] / REPO_LANE_LINT).read_text().replace(lane, '    ("toy", r"TOY\\d\\b", None),\n' + lane, 1)
    lane_linted.git("checkout", "-q", "-B", "lint", main)
    lint = lane_linted.commit("counts TOY ids", {REPO_LANE_LINT: wider, REPO_LANE_JSON: _allowlist(DOCS_A, b)})
    lane_linted.git("checkout", "-q", "-b", "adds", main)
    adds = lane_linted.commit("a hook named for TOY9", {"tools/hook.py": '"""Runs as TOY9."""\nX = 1\n'})
    assert [c["verdict"] for c in G.gate("main", "adds")["changes"]] == ["passed"], "main's lint counts no TOY id"
    lane_linted.git("checkout", "-q", "-b", "next", main)
    lane_linted.queue(1, "counts TOY ids", lint)
    m2 = lane_linted.queue(2, "a hook named for TOY9", adds)
    doc = G.gate("main", m2)
    assert _verdicts(doc) == [("#1", "passed"), ("#2", "failed")], doc
    assert doc["changes"][1]["fix"] == ["tools/hook.py:1 toy: TOY9"] and doc["failures"][0]["pr"] == 2

    lane_linted.git("checkout", "-q", "-b", "prunes", lint)
    prunes = lane_linted.commit("says what is true now in a.md", {"docs/a.md": "# A\n\nSee the README.\n",
                                                                  REPO_LANE_JSON: _allowlist(b)})
    lane_linted.git("checkout", "-q", "next")
    lane_linted.queue(3, "says what is true now in a.md", prunes)
    doc = G.gate("main", "next")
    assert _verdicts(doc)[2] == ("#3", "passed") and not [f for f in doc["failures"] if f["pr"] == 3], doc
    why = doc["changes"][2]["why"]
    assert ("the base's lint code (repo_lane_text) finds 2 violation(s) its own tree already holds, which aren't this "
            "change's: added by #2: repo_lane_text: tools/hook.py:1 toy: TOY9, repo_lane_text: tools/hook.py: 1 found, "
            "0 allowed") in why, why
    assert "tools/hook.py: 1 found, 0 allowed" in (_lane_lint(lane_linted) or "")

    lane_linted.git("checkout", "-q", "-b", "drops", lint)
    drops = lane_linted.commit("drops b.md's entry", {REPO_LANE_JSON: _allowlist(DOCS_A)})
    lane_linted.git("checkout", "-q", "-B", "next", m2)
    lane_linted.queue(4, "drops b.md's entry", drops)
    [*_, c] = G.gate("main", "next")["changes"]
    assert (c["name"], c["verdict"]) == ("#4", "failed") and "finds 3 violation(s) on its tree" in c["why"], c["why"]
    assert c["fix"] == ["repo_lane_text: docs/b.md: 2 found, 0 allowed", "repo_lane_text: docs/b.md:3 toy: TOY1",
                        "repo_lane_text: docs/b.md:3 toy: TOY2"], "the violation the tree below holds is #2's, not #4's"


def test_a_lint_module_whose_own_prose_a_change_cleans_is_counted_on_its_own_copy(lane_linted):
    """The gate runs the base's copy of a lint module the change edits, and the lane-text lint reads that copy's prose too:
    a change that rewrites the module's docstring and drops its entry passes, counted on its own copy, and one that edits
    the module's code, keeps its site and drops its entry fails, naming the site where the change's copy holds it."""
    toy = {"file": LINT, "count": 1}
    lane_linted.git("checkout", "-q", "main")
    lane_linted.commit("a lint whose docstring names its issue", {LINT: '"""P1, after VLLM-136."""\n' + TOY_LINT,
                                                                  REPO_LANE_JSON: _allowlist(DOCS_A, toy)})
    lane_linted.git("checkout", "-q", "-B", "pr")
    lane_linted.commit("the docstring says what is true now", {LINT: '"""P1."""\n' + TOY_LINT, REPO_LANE_JSON: _allowlist(DOCS_A)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["lints"]) == ("passed", ["p01_core", "repo_lane_text"]), c["why"]
    assert (f"the base's lint code (p01_core, repo_lane_text) finds 2 violation(s) in the base's copy of {LINT}, which it "
            "runs in place of the change's, and the change's own copy is within its list") in c["why"], c["why"]
    assert _lane_lint(lane_linted) is None
    lane_linted.git("checkout", "-q", "-B", "pr", "main")
    wider = TOY_LINT.replace('WORD = re.compile(r"BAD")', 'WORD = re.compile(r"BAD|WORSE")')
    assert wider != TOY_LINT
    lane_linted.commit("counts WORSE, and drops the entry", {LINT: '"""P1.\n\nAfter VLLM-136."""\n' + wider,
                                                             REPO_LANE_JSON: _allowlist(DOCS_A)})
    [c] = G.gate("main", "pr")["changes"]
    assert (c["verdict"], c["fix"]) == ("failed", [f"repo_lane_text: {LINT}: 1 found, 0 allowed",
                                                   f"repo_lane_text: {LINT}:3 linear: VLLM-136"]), c["why"]
    assert f"{LINT}: 1 found, 0 allowed" in (_lane_lint(lane_linted) or "")
