"""Every file a serve-tax writer keeps on the node (each name given to its `_kept`) is one `research_outputs.PRIVATE`
names, plain and gzipped, so no stage's outputs carry it."""
import ast
from pathlib import Path

import pytest
from verity_vllm.pipeline import research_outputs

ROOT = Path(__file__).resolve().parents[3]
WRITERS = ("benchmarks/serve_tax/servetax_plugin.py", "benchmarks/serve_tax/replay.py",
           "benchmarks/replay_ladder/ladder.py")


def kept_names(path: Path) -> list[str]:
    """The last argument of each call of a function named ``_kept`` in ``path``, an f-string's fields rendered as 1."""
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "_kept":
            arg = node.args[-1]
            if isinstance(arg, ast.Constant):
                names.append(arg.value)
            elif isinstance(arg, ast.JoinedStr):
                names.append("".join(v.value if isinstance(v, ast.Constant) else "1" for v in arg.values))
            else:
                raise AssertionError(f"{path}:{node.lineno}: _kept's name is not a literal, so this test can't check it")
    return names


@pytest.mark.parametrize("writer", WRITERS)
def test_every_kept_name_is_private(writer: str) -> None:
    names = kept_names(ROOT / writer)
    assert names, f"{writer} keeps nothing through _kept"
    missed = [n for name in names for n in (name, name + ".gz") if not research_outputs.private(n)]
    assert not missed, f"{writer} keeps {missed} on the node, which research_outputs.PRIVATE names no pattern for: add one"


def test_plugin_refuses_without_private(tmp_path: Path) -> None:
    import servetax_plugin as P

    with pytest.raises(RuntimeError, match="private"):
        P._kept({"out": str(tmp_path)}, "step_trace.1.jsonl")
    assert P._kept({"out": str(tmp_path / "out"), "private": str(tmp_path / "p")}, "ids.1.jsonl") \
        == tmp_path / "p" / "server" / "ids.1.jsonl"
