"""A suite that declares custody's reader of the private names declares the list it reads.

`research.store.private` reads the tree's one list of private names when it is loaded, so every suite whose declared inputs
hold that module reads the list too, and the read guard fails a suite that reads a file outside its inputs."""
from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READER = "tools/research/src/research/store/private.py"
LIST = "adapters/vllm/verity_vllm/pipeline/private_names.json"


def _holds(declared: list[str], path: str) -> bool:
    return any(path == d.rstrip("/") or path.startswith(d.rstrip("/") + "/") for d in declared)


def test_a_suite_that_declares_custodys_reader_declares_the_private_names():
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*pyproject.toml"], capture_output=True, text=True, check=True)
    missing = []
    for rel in listed.stdout.split():
        tests = tomllib.loads((ROOT / rel).read_text()).get("tool", {}).get("verity", {}).get("tests") or {}
        declared = [*(tests.get("inputs") or []),
                    *(p for r in tests.get("reads") or [] if isinstance(r, dict) for p in r.get("paths") or [])]
        if _holds(declared, READER) and not _holds(declared, LIST):
            missing.append(rel)
    assert not missing, (f"add \"{LIST}\" to [tool.verity.tests] inputs of: {', '.join(missing)}. Each declares {READER}, which "
                         "reads that list when it is loaded")
