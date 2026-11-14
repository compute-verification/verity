"""`verity-vllm verdict` -- a row's Commit verdict record (`check.verdict`), rebuilt from the row's evidence.

    verity-vllm verdict from-record <row_dir> [--out PATH]      rebuild the record from a row's evidence
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from verity_vllm.check.verdict import from_record
from verity_vllm.config import option

__all__ = ["COMMANDS", "main"]


@dataclass(frozen=True, kw_only=True)
class FromRecordOptions:
    """rebuild <row_dir>/verdict.json from the row's Commit evidence"""

    row_dir: str = option("row_dir")
    out: str | None = option("--out", default=None, help="where to write (default <row_dir>/verdict.json)")
    rc: int | None = option("--rc", type=int, default=None,
                            help="commit_delta's exit code (row_pod.sh passes it; a pre-P5 row's commit_summary.json supplies it otherwise)")
    cmd: str = field(default="from-record", init=False)


COMMANDS = {"from-record": FromRecordOptions}


def main(a: FromRecordOptions) -> int:
    v = from_record(a.row_dir, rc=a.rc)
    out = Path(a.out) if a.out else Path(a.row_dir) / "verdict.json"
    v.write(out)
    gs = " ".join(f"{k}={v_}" for k, v_ in v.groups().items())
    print(f"{v.outcome} {out} [{gs}]")
    return 0
