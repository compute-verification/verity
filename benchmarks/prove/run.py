"""`verity bench prove`: what a proof system's prover costs on a workload, `verity-bench prove` (its README beside this
file).

    verity bench prove --system flock --workload glm53               every block proved: a row a block type, the totals
    verity bench prove --system spot --workload glm53 [--sampling ART]
                                                                     Flock at the workload's sampling plan of record
    verity bench prove --system flock --workload glm53 --provers 2   each type priced at two provers (default: one)
    verity bench prove --system flock --workload glm53 --rows 12     the 12 costliest block types, the rest in one row
    verity bench prove --system flock --workload glm53 --json        the table as JSON
"""

from __future__ import annotations

import sys

from verity_bench.cli import main

if __name__ == "__main__":
    sys.exit(main(["prove", *sys.argv[1:]]))
