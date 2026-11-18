"""The environment variables `served_session.py` reads, declared once (`tests/lint/test_repo_env_reads.py`: reads belong
here)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

#: ``1`` restores the refusals that rest only on where a party runs; unset, the run goes on and its record says where
HOLD_PLACEMENT = "VERITY_HOLD_PLACEMENT"


def hold_placement() -> bool:
    """`verity_cli.settings.hold_placement`, the one reader, loaded by its path: `served_session.py` runs under any
    python3."""
    path = Path(__file__).resolve().parents[2] / "tools" / "verity" / "cli" / "verity_cli" / "settings.py"
    spec = importlib.util.spec_from_file_location("verity_cli_settings", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli.hold_placement()


def host(name: str) -> str | None:
    """The address the variable `name` holds, or None while it is unset or empty. ``site.json``'s gate names one,
    ``VY_CHALLENGER_1_HOST`` (the challenger host's), which no file here holds."""
    return os.environ.get(name) or None
