"""Every environment variable exact_kit reads, declared once: the shape of ``benchmarks/pouw/settings.py`` for this
distribution, which has no workspace dependency. All four are `research run`'s and say only where a run happens, so none is
in a verdict. Read one as an attribute (``settings.OUT``): the variable's string, or None when it is unset or empty."""
from __future__ import annotations

import os

#: name -> doc
SETTINGS = {
    "OUT": "the run's out/ directory, where finish writes <script>.verdict.json without --verdict",
    "RESEARCH_RUN_DIR": "the run's directory, where finish writes result.json",
    "RESEARCH_RUN_ID": "the research run this process is part of",
    "RESEARCH_SOURCE_SHA": "the commit of the tree the run ships, in result.json's fingerprint",
}


def __getattr__(name: str) -> str | None:
    if name not in SETTINGS:
        raise AttributeError(f"{name} is not a declared setting (exact_kit/settings.py)")
    return os.environ.get(name) or None
