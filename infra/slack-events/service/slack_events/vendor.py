"""The one routing implementation and its registry, as the service runs them: `_vendor/` (what `build.py` copied from the
repository at build time), else the repository's own files where the service runs from a checkout (its tests, `vercel dev`
after no build)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parents[1]
VENDOR = HERE / "_vendor"
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE      # a deployed bundle (/var/task) has no repository around it
NAME = "research_slack"


def _pick(vendored: str, rel: str) -> Path:
    for p in (VENDOR / vendored, REPO / rel):
        if p.is_file():
            return p
    raise RuntimeError(f"neither {VENDOR / vendored} nor {REPO / rel} exists: run `python3 build.py` (the Vercel build does)")


def slack() -> ModuleType:
    """`tools/research/src/research/slack.py`, run as the single file it is built to be."""
    if NAME in sys.modules:
        return sys.modules[NAME]
    spec = importlib.util.spec_from_file_location(NAME, _pick("research_slack.py", "tools/research/src/research/slack.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def registry() -> dict:
    """The registry bundled at build, checked as `research msg` checks it."""
    path = _pick("registry.json", ".agents/skills/using-slack/registry.json")
    return slack().check_registry(json.loads(path.read_text()), str(path))
