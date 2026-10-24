"""`verity-vllm verdict` -- a row's Commit verdict record (`check.verdict`), rebuilt from the row's evidence.

    verity-vllm verdict from-record <row_dir> [--out PATH]      rebuild the record from a row's evidence
    verity-vllm verdict mapping                                  the old -> new check-name table (markdown)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from verity_vllm.check.verdict import from_record
from verity_vllm.check.verdict_names import mapping_markdown
from verity_vllm.config import option


@dataclass(frozen=True, kw_only=True)
class FromRecordOptions:
    """rebuild <row_dir>/verdict.json from the row's Commit evidence"""

    row_dir: str = option("row_dir")
    out: str | None = option("--out", default=None, help="where to write (default <row_dir>/verdict.json)")
    rc: int | None = option("--rc", type=int, default=None,
                            help="commit_delta's exit code (row_pod.sh passes it; a pre-P5 row's commit_summary.json supplies it otherwise)")
    stage_ok: str | None = option("--stage-ok", default=None,
                                  help="a stage decision made outside this verdict, if any (recorded as compat.stage_ok; a disagreement is a note)")
    quiet: bool = option("--quiet", default=False, action="store_true")
    cmd: str = field(default="from-record", init=False)


@dataclass(frozen=True, kw_only=True)
class MappingOptions:
    """print the old -> new check-name table (markdown)"""

    cmd: str = field(default="mapping", init=False)


COMMANDS = {"from-record": FromRecordOptions, "mapping": MappingOptions}


def main(a: FromRecordOptions | MappingOptions) -> int:
    if a.cmd == "mapping":
        print(mapping_markdown(), end="")
        return 0
    v = from_record(a.row_dir, rc=a.rc, stage_ok=a.stage_ok)
    out = Path(a.out) if a.out else Path(a.row_dir) / "verdict.json"
    v.write(out)
    if not a.quiet:
        gs = " ".join(f"{k}={v_}" for k, v_ in v.groups().items())
        print(f"{v.outcome} {out} [{gs}]")
    return 0
