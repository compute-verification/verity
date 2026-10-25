"""Every merge driver `.gitattributes` names is set where drive and the gates merge (`research.clone.RECORD_DRIVER`): a clone's
own config holds none of them, and without its driver git merges the file as text."""
from __future__ import annotations

import re
from pathlib import Path

from research.clone import RECORD_DRIVER

ROOT = Path(__file__).resolve().parents[1]


def test_every_merge_driver_the_attributes_name_is_set_in_drives_clone_and_its_script_exists():
    named = {m for line in (ROOT / ".gitattributes").read_text().splitlines() if not line.lstrip().startswith("#")
             for m in re.findall(r"\bmerge=([\w-]+)", line)}
    assert {"lean-audit", "fixture-map", "vllm-allowlist"} <= named
    missing = sorted(n for n in named if f"merge.{n}.driver" not in RECORD_DRIVER)
    assert not missing, ("`.gitattributes` names merge drivers that drive's clone doesn't set, so it merges their files as "
                         "text: add merge.<name>.name and merge.<name>.driver to RECORD_DRIVER in "
                         f"tools/research/src/research/clone.py: {missing}")
    absent = sorted(n for n in named if not any((ROOT / p).is_file()
                                                for p in re.findall(r"[\w./-]+\.py", RECORD_DRIVER[f"merge.{n}.driver"])))
    assert not absent, f"RECORD_DRIVER runs a script that isn't in the tree for: {absent}"
