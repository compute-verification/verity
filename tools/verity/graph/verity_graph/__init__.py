"""`verity graph`: the check as a task graph that depends only on the commit (README.md).

`contract` holds the schema names, canonical JSON, digests and the key rule; `graph` lists a checkout's graph/v2 from its
families (`families/`, declared in `tasks.toml`); `runners` runs one task and makes its result/v1; `local` is the serial
reference executor; `cli` is the command."""
from __future__ import annotations

from pathlib import Path

#: the checkout this package is in, where `verity graph run` runs tasks
ROOT = Path(__file__).resolve().parents[4]
