"""The package stands apart from the job's machinery: but for the job itself (`run.py`) and the port check
(`of_record.py`), which counts a config's step under the job's own harness and loop, no module of it imports a `run`
module (the command, the job), the sandbox, the port check or a `research` module, so the machinery can move by path
and what the library imports brings none of it."""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "training_slowdown"
JOB = {PACKAGE / "run.py", PACKAGE / "sandbox.py", PACKAGE / "of_record.py"}
MACHINERY = {"run", "sandbox", "research", "of_record"}


def machinery(path: Path) -> set[str]:
    """The modules and names `path` imports from the machinery, wherever in the file the import is."""
    named = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            named += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            named += [".".join(filter(None, [node.module, a.name])) for a in node.names]
    return {n for n in named if n.removeprefix("training_slowdown.").split(".")[0] in MACHINERY}


def test_no_module_but_the_job_imports_the_jobs_machinery():
    library = sorted(set(PACKAGE.rglob("*.py")) - JOB)
    names = {p.relative_to(PACKAGE).as_posix() for p in library}
    assert names >= {"config.py", "screen.py", "grade.py", "calculator.py", "library.py", "traced.py",
                     "codecs/__init__.py", "routes/subset.py"}
    assert {(p.relative_to(PACKAGE).as_posix(), n) for p in library for n in machinery(p)} == set()
    assert machinery(PACKAGE / "run.py") == {"sandbox"} and machinery(PACKAGE / "of_record.py") == {"run"}
