"""The fleet loop's settings (`fleet.py`): the environment variables its usage panels and its archive read, declared once."""
from __future__ import annotations

import os
from pathlib import Path

#: the site the usage panels go to (`fleet.put_panel`); unset, no panel is sent
PANELS_URL = os.environ.get("FLEET_PANELS_URL") or None
#: the site's panels key
PANELS_KEY = Path(os.environ.get("FLEET_PANELS_KEY", str(Path.home() / ".config" / "verity" / "panels.key")))


def research_bin() -> Path:
    """The `research` command the loop runs (`data put`, `msg send`)."""
    return Path(os.environ.get("FLEET_RESEARCH", "/workspace/.venv/bin/research"))


def store_root() -> str:
    """The evidence store the samples go into: $RESEARCH_STORE, else ~/fleet-store."""
    return os.environ.get("RESEARCH_STORE") or str(Path.home() / "fleet-store")


def store_env() -> Path:
    """The KEY=VALUE file holding the store remote's credentials (the pod's r2.env), which the loop's environment doesn't hold."""
    return Path(os.environ.get("FLEET_STORE_ENV", str(Path.home() / ".config" / "verity" / "r2.env")))


def tenant() -> str:
    """The id of the tenant whose projects and quotas the loop lists: $FLEET_TENANT, else $NEBIUS_TENANT_ID (common.sh's
    name for it). No file holds it, and a read that needs it fails naming the variable when neither is set."""
    if not (got := os.environ.get("FLEET_TENANT") or os.environ.get("NEBIUS_TENANT_ID")):
        raise RuntimeError("NEBIUS_TENANT_ID is unset: the id of the tenant whose projects and quotas fleet.py lists "
                           "(FLEET_TENANT, when set, is the loop's own)")
    return got


def jump() -> str | None:
    """USER@HOST a machine registered at a private address (10.*) is reached through, $FLEET_JUMP; no file holds it."""
    return os.environ.get("FLEET_JUMP") or None
