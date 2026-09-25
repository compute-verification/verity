"""`research_tools.CLOSURE` is the integration's import closure: `source_identity.code_identity` over it keys the hot Commit worker and
names the TP Commit's tree, so a Verity core file outside it would be code those keys never see."""
from __future__ import annotations

from research.store.tool import glob_re, list_source_files

from verity_vllm.pipeline import research_tools
from verity_vllm.pipeline.source_identity import TREE


def test_closure_covers_every_core_file():
    pats = [glob_re(p) for p in research_tools.CLOSURE]
    core = [p for p in list_source_files(TREE) if p.startswith("packages/verity/src/verity/")]
    outside = [p for p in core if not any(rx.match(p) for rx in pats)]
    assert core and not outside, f"core files outside research_tools.CLOSURE: {outside}"
