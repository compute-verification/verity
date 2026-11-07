"""The Lean tools' settings: the environment variables they read, each declared once here."""
from __future__ import annotations

import os

#: the GB the kept warm builds may take on a node before their least recently used go (`lean_audit.BUILDS_MAX_GB`)
BUILDS_MAX_GB = float(os.environ.get("VERITY_LEAN_BUILDS_MAX_GB", "120"))
