"""`lean_audit.NOT_READ`: the key holds the files an audit loads and no other. A package's key holds the auditor's files
(`lean_audit.tool_files`) and its audit runs on a scratch tree of them. A load of a `NOT_READ` file from one of them
either decides a verdict the key doesn't cover or fails the audit, and only on the path that reaches the load; a file of
the key that no audit loads audits every package again on every node whenever it changes."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("lean_audit_for_not_read", ROOT / "tools" / "verity" / "lean" / "lean_audit.py")
LA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LA)

#: the loads of a `NOT_READ` file by a file of the key, each {(the file, the name it loads): why no verdict under a key
#: reads it}; it only shrinks
BESIDE = {(f"{LA.LEAN_TOOL}lean_audit.py", "unread.py"):
          "the ratchet, which `run` runs after the audits from the checkout on every run: no pass is kept of it",
          (f"{LA.LEAN_TOOL}check.py", "reserve.py"):
          "what a run reserves and how many jobs it runs: it refuses a run before the run audits, a refusal is never kept, "
          "and no step's verdict reads the plan (`test_reserve.py` holds it to numbers and functions of pins, steps and a "
          "reservation)",
          (f"{LA.LEAN_TOOL}lean_audit.py", "reserve.py"):
          "`CARRIED`: copied into every scratch tree for check.py's load, and read by nothing of the audit's"}


def _loaders(sources: dict[str, str]) -> set[str]:
    """The functions of ``sources`` that load a file as a module: a call of one with a file's stem loads that file."""
    return {n.name for text in sources.values() for n in ast.walk(ast.parse(text))
            if isinstance(n, ast.FunctionDef) and any(isinstance(c, ast.Attribute) and c.attr == "spec_from_file_location"
                                                      for c in ast.walk(n))}


def _named(file: str, text: str, names: list[str], loaders: set[str]) -> set[str]:
    """The files of ``names`` that ``file``'s code names: a Python file by a path that ends in one, an import of its stem
    or a loader's call with it, outside docstrings and `NOT_READ` itself; a shell script anywhere outside its comments."""
    if not file.endswith(".py"):
        code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        return {n for n in names if n in code}
    tree, got = ast.parse(text), set()
    prose = {id(n.body[0].value) for n in ast.walk(tree)
             if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
             and ast.get_docstring(n, clean=False) is not None}
    prose |= {id(c) for n in tree.body if isinstance(n, ast.Assign) and "NOT_READ" in [getattr(t, "id", "") for t in n.targets]
              for c in ast.walk(n.value)}
    stems = {n[:-3]: n for n in names if n.endswith(".py")}
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in prose:
            got |= {name for name in names if n.value.rsplit("/", 1)[-1] == name}
        elif isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) in loaders:
            got |= {stems[a.value] for a in n.args if isinstance(a, ast.Constant) and a.value in stems}
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] + ([n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
            got |= {stems[m.rsplit(".", 1)[-1]] for m in mods if m.rsplit(".", 1)[-1] in stems}
    return got


def _key() -> tuple[dict[str, str], set[str]]:
    """The Python and shell files of the key, each with its text, and the loaders they define."""
    keyed = {f: (ROOT / f).read_text() for f in LA.tool_files(LA.COMMON.tracked_files()) if f.endswith((".py", ".sh"))}
    loaders = _loaders({f: text for f, text in keyed.items() if f.endswith(".py")})
    assert {f"{LA.LEAN_TOOL}check.py", f"{LA.LEAN_TOOL}setup.sh"} <= set(keyed) and {"sibling", "_module", "_load"} <= loaders
    return keyed, loaders


def test_no_file_of_the_key_loads_a_file_no_audit_reads():
    """Every Python file `NOT_READ` lists is loaded by no file of the key: not by `check.py`'s `sibling`, by a path, by an
    import, nor from the shell scripts an audit runs. The Lean programs load no Python. The one exception is `BESIDE`."""
    keyed, loaders = _key()
    names = sorted(n for n in LA.NOT_READ if n.endswith(".py"))
    got = {(f, n) for f, text in keyed.items() for n in _named(f, text, names, loaders)}
    assert got == set(BESIDE), (
        "a file of every package's key loads a file `lean_audit.NOT_READ` lists, so a verdict under the key would read what "
        f"the key leaves out: take the file out of `NOT_READ` or the load out of the keyed file: {sorted(got ^ set(BESIDE))}")


def test_every_file_of_the_key_is_one_an_audit_reaches():
    """The converse: each file of the key is named by the code of a file `lean_audit.py` reaches, itself first, so a tool
    no audit loads is listed in `NOT_READ` rather than keyed."""
    keyed, loaders = _key()
    tool = LA.tool_files(LA.COMMON.tracked_files())
    at = {f.rsplit("/", 1)[-1]: f for f in tool}
    assert len(at) == len(tool), "two files of the key share a name: tell them apart here"
    reached, todo = set(), ["lean_audit.py"]
    while todo:
        if (name := todo.pop()) not in reached:
            reached.add(name)
            todo += sorted(_named(at[name], keyed[at[name]], sorted(at), loaders)) if at[name] in keyed else []
    assert not (lost := sorted(set(at) - reached)), (
        f"no file an audit runs loads {', '.join(lost)}, so its change would audit every package again for nothing: "
        "list it in `lean_audit.NOT_READ` with what does read it")


def test_each_way_to_load_a_file_is_seen():
    """What the test above reads a load from, on one line each: a control that it still sees them."""
    names, loaders = ["modules.py", "siblings.py"], {"sibling", "_load"}
    for line, want in [('M = sibling("modules")', {"modules.py"}),
                       ('M = A.sibling("siblings")', {"siblings.py"}),
                       ('M = _load("modules", HERE / "modules.py")', {"modules.py"}),
                       ('P = ROOT / "tools" / "verity" / "lean" / "siblings.py"', {"siblings.py"}),
                       ('P = "tools/verity/lean/modules.py"', {"modules.py"}),
                       ("import modules", {"modules.py"}),
                       ("from . import siblings", {"siblings.py"}),
                       ("from tools.verity.lean.modules import header", {"modules.py"}),
                       ('NOT_READ = {"modules.py": "the codemod"}', set()),
                       ('def f():\n    """Loads `modules.py`."""\n    return doc.get("modules")', set()),
                       ('log("ported by modules.py")', set())]:
        assert _named("x.py", line, names, loaders) == want, line
    assert _named("setup.sh", '#!/bin/sh\n# modules.py ports\npython3 "$here/siblings.py"\n', names, loaders) == {"siblings.py"}
