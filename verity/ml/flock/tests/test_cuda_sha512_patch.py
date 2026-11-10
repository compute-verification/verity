"""``cuda_sha512_patch.py``'s first step, the one that needs nothing of upstream's: it copies the GPU's one SHA-512
(``verity.ml.kernels.SHA512``) into the Flock checkout as ``kernels_sha512.cuh`` and installs ``cuda/sha512.cuh`` beside
it, its include of that header pointed at the copy. Only a pod runs the script, with a bare ``python3``, so a move of
either header that the script doesn't follow fails here first."""
from __future__ import annotations

import ast
import re
import runpy
import sys
from pathlib import Path

import pytest

from verity.ml.kernels import SHA512

SCRIPT = Path(__file__).resolve().parents[1] / "cuda_sha512_patch.py"
HEADER = Path(__file__).resolve().parents[1] / "cuda" / "sha512.cuh"


def _includes(text: str) -> list[str]:
    return re.findall(r'^#include "([^"\n]+)"$', text, re.MULTILINE)


def test_it_installs_the_gpus_sha512_and_the_header_including_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ghash = tmp_path / "cuda-ghash"
    ghash.mkdir()
    (ghash / "merkle_open.hpp").write_text("")              # upstream's, not yet patched
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as stopped:       # at the first of upstream's files this stand-in lacks
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert Path(stopped.value.filename).parent == Path("cuda-ghash"), stopped.value
    assert (ghash / "kernels_sha512.cuh").read_bytes() == SHA512.read_bytes()
    ours, installed = HEADER.read_text(), (ghash / "sha512.cuh").read_text()
    (theirs,) = [name for name in _includes(ours) if name not in _includes(installed)]
    assert (HEADER.parent / theirs).resolve() == SHA512.resolve()
    assert installed == ours.replace(f'#include "{theirs}"', '#include "kernels_sha512.cuh"')


def test_it_imports_only_the_standard_library() -> None:
    """A pod runs it with ``python3`` from the Flock checkout, where no package of this tree is installed."""
    imports = [node for node in ast.walk(ast.parse(SCRIPT.read_text())) if isinstance(node, ast.Import | ast.ImportFrom)]
    names = {alias.name.split(".")[0] for node in imports if isinstance(node, ast.Import) for alias in node.names}
    names |= {"." if node.level else (node.module or ".").split(".")[0] for node in imports if isinstance(node, ast.ImportFrom)}
    assert names <= sys.stdlib_module_names, sorted(names - sys.stdlib_module_names)
