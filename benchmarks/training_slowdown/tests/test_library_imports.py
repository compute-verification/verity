"""The library and the grade stand apart from the job's machinery: the routes, the config, the codecs, the grade and
the calculator import neither a `run` module (the command, the job), the sandbox nor a `research` module, so the
machinery can move by path."""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "training_slowdown"
LIBRARY = [*(PACKAGE / "routes").rglob("*.py"), *(PACKAGE / "codecs").rglob("*.py"),
           *(PACKAGE / name for name in ("config.py", "grade.py", "calculator.py"))]
MACHINERY = {"run", "sandbox", "research"}
#: the imports of the machinery that stand, which only shrinks: `config.route` screens a route's source with the
#: job's `screen` before it imports it
KNOWN = {("config.py", "run.screen")}


def machinery(path: Path) -> set[str]:
    """The modules and names `path` imports from the machinery, wherever in the file the import is."""
    named = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            named += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            named += [".".join(filter(None, [node.module, a.name])) for a in node.names]
    return {n for n in named if n.removeprefix("training_slowdown.").split(".")[0] in MACHINERY}


def test_the_library_and_the_grade_import_none_of_the_jobs_machinery():
    assert len(LIBRARY) >= 7 and all(p.is_file() for p in LIBRARY)
    assert {(p.relative_to(PACKAGE).as_posix(), n) for p in LIBRARY for n in machinery(p)} == KNOWN
