"""The private names, `private_names.json` beside this file, and the one matcher over them: which files of a run no keeper
keeps. `research_outputs` imports it; a keeper outside the adapter's environment (`benchmarks/poub_session/keep.py`) loads
this file by path, so it needs only the standard library and imports nothing of `verity_vllm`."""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path, PurePosixPath

LIST = Path(__file__).with_name("private_names.json")

Names = tuple[tuple[str, ...], tuple[str, ...]]


def load(path: Path = LIST) -> Names:
    """(kept_apart, private): the directory prefixes and the file name patterns no keeper keeps."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    return tuple(doc["kept_apart"]), tuple(doc["private"])


def private(rel: str, names: Names | None = None) -> bool:
    """Whether `rel` is private by the list: under a kept-apart directory, or a file a private pattern names."""
    kept_apart, patterns = names if names is not None else load()
    parts = PurePosixPath(rel).parts
    return any(p.startswith(kept_apart) for p in parts) or any(fnmatch.fnmatchcase(parts[-1], q) for q in patterns)
