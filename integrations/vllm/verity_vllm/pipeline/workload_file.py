"""`verity-vllm workload-file ROW_OR_PATH...`: each workload made present and checked (`config.workload_file`: fetched by its art id
when absent, refused unless its SHA-256 is the one its directory's ``SHA256SUMS`` lists), its path printed, one a line.  A row id is
found as a row finds it (`config.row_file`); a ``.json`` path is taken as it is.  The shell scripts that read a workload themselves
run it first, and code outside the integration reads one through this module's ``workload_file``."""
from __future__ import annotations

from dataclasses import dataclass

from verity_vllm.config import option, row_file, workload_file

__all__ = ["workload_file"]


@dataclass(frozen=True)
class Options:
    workloads: list[str] = option("workloads", nargs="+", metavar="ROW_OR_PATH", help="row ids or workload files")


def main(o: Options) -> int:
    for w in o.workloads:
        print(workload_file(w) if w.endswith(".json") else row_file(w))
    return 0
