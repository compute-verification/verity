"""The suite family: one task per suite (`test_file.discover`: as `suites.discover` finds them) but the build family's lean-suites
tasks' (`build.lean_suites`: in a tree with the Lean verifier, those check runs after its lean-build), each
`tools/verity/check/suites.py` on the suite as check's pytest step runs it, or one task per part of a suite that declares
`[tool.verity.tests] parts` (`suites.py NAME --files K/M`, by test id); its var `suite` is the suite's directory, which its
failed tests' ids are relative to.  A task's inputs are what
suites.py keys the suite on (`test_file.whole`, its `*.md` only when the suite reads Markdown: `suites.input_ids`), the files
every one of its keys includes (`suites.RUNNER`) and what a fetch of a fixture reads of the tree (`suites.STORE_READS`), each
read from suites.py's source.  A suite that declares `[tool.verity.tests] parallel` takes the family's slot and another one
process of its `worker_gb` (`suites.TEST_PROCESS_GB` undeclared); each runs at its `timeout_s`, else the family's."""
from __future__ import annotations

from pathlib import Path

from .build import lean_suites, suites_literals
from .test_file import Tree, _join, discover, whole

KIND = "suite"
REPORT = "suites.json"


def tasks(root: Path, files: list[str]) -> list[dict]:
    tree = Tree(root, files)
    suites = discover(tree)
    after = set(lean_suites(root, files))
    lit = suites_literals(root, files, "TEST_PROCESS_GB", "RUNNER", "STORE_READS")
    common = [p for p in (*lit.get("RUNNER", ()), *lit.get("STORE_READS", ())) if p in tree.files]
    out = []
    for s in suites:
        if not after.isdisjoint((s.name, s.path)):
            continue
        tests = tree.toml(_join(s.dir, "pyproject.toml")).get("tool", {}).get("verity", {}).get("tests", {})
        own = [f for f in whole(tree, s, suites) if s.markdown or not f.endswith(".md")]
        task = {"kind": KIND, "inputs": sorted({*own, *common}), "outputs": [{"name": "report", "path": REPORT}]}
        if tests.get("parallel") is not True:
            gb = s.mem_gb or lit.get("TEST_PROCESS_GB")
            task["resources"] = {"cores": 1, **({"mem_gb": gb} if gb else {})}
        if s.timeout_s:
            task["timeout_s"] = s.timeout_s
        m = tests.get("parts") if type(tests.get("parts")) is int and tests["parts"] > 1 else None
        if m:
            out += [{"id": f"{KIND}:{s.name}:{k}/{m}", **task, "vars": {"suites": s.name, "suite": s.path, "files": f"{k}/{m}"},
                     "about": f"suites.py on part {k} of {m} of {s.name}'s tests by id"} for k in range(1, m + 1)]
        else:
            out.append({"id": f"{KIND}:{s.name}", **task, "vars": {"suites": s.name, "suite": s.path},
                        "about": f"suites.py on {s.name}, as check's pytest step runs it"})
    return sorted(out, key=lambda t: t["id"])
