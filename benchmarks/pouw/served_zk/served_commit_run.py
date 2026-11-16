"""`served_commit.py`'s commands (PoUW's prover, `kernels/served_zk/`) with what verity does not import: the challenger's
checks of record with Lean's side (`of_record_lean.Record`) and the commit's batch tile checks
(`benchmarks/pouw/pearl_c_vllm/fast_tile.py`'s `install`).

  python benchmarks/pouw/served_zk/served_commit_run.py COMMAND ...    as `served_commit.py COMMAND ...`
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "pearl_c_vllm")]

import fast_tile  # noqa: E402
from of_record_lean import Record  # noqa: E402

from verity.protocols.resource_accounting.compute.pouw.prover.kernels.served_zk import served_commit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return served_commit.main(argv, record=Record, tiles=fast_tile.install)


if __name__ == "__main__":
    sys.exit(main())
