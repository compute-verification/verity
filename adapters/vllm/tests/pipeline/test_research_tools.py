"""`research_tools.CLOSURE` is the integration's import closure: `source_identity.code_identity` over it keys the hot Commit worker and
names the TP Commit's tree, so a Verity core file outside it would be code those keys never see."""
from __future__ import annotations

from pathlib import Path

import verity
from research.store.tool import glob_re, list_source_files

from verity_vllm.pipeline import research_tools
from verity_vllm.pipeline.source_identity import TREE


def test_closure_covers_every_core_file():
    """Every directory the import system finds `verity` in (one per member that ships part of it), not a hand-written one."""
    pats = [glob_re(p) for p in research_tools.CLOSURE]
    files = list_source_files(TREE)
    portions = [Path(p).resolve().relative_to(TREE.resolve()).as_posix() for p in verity.__path__]
    assert portions
    for portion in portions:
        core = [p for p in files if p.startswith(portion + "/")]
        outside = [p for p in core if not any(rx.match(p) for rx in pats)]
        assert core and not outside, f"core files under {portion} outside research_tools.CLOSURE: {outside}"
