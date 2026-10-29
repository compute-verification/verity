"""The frozen table of v0's tags before MEM-152's move (`retired`) covers every tag today's PoUB code names, each the
same tag under v0's name then."""

from __future__ import annotations

import re
from pathlib import Path

from verity.core.protocols.poub import retired

ROOT = Path(retired.__file__).parent


def test_the_table_holds_every_tag_the_code_names() -> None:
    named = {m[0] for f in ROOT.rglob("*") if f.suffix in (".py", ".lean") and f != Path(retired.__file__)
             and not {"tests", ".lake"} & set(f.relative_to(ROOT).parts)
             for m in re.finditer(r"verity/poub[a-z0-9/_-]*", f.read_text(encoding="utf-8"))}
    assert named == set(retired.V0), f"tags the table lacks: {sorted(named - set(retired.V0))}; tags no code names: " \
                                     f"{sorted(set(retired.V0) - named)}"
    assert all(old == retired.RETIRED + new.removeprefix("verity/poub") for new, old in retired.V0.items())
