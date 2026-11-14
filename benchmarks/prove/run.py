"""`verity bench prove`: what a proof system's prover costs on a workload, `verity-bench prove` (its README beside this
file).

    verity bench prove --system flock --workload glm53               every block proved: a row a block type, the totals
    verity bench prove --system spot --workload glm53 [--sampling ART]
                                                                     Flock at the workload's sampling plan of record
    verity bench prove --system flock --workload glm53 --provers 2   each type priced at two provers (default: one)
    verity bench prove --system flock --workload glm53 --cores 24    each type priced on a host of 24 CPU cores a card
                                                                     (default: 10)
    verity bench prove --system flock --workload glm53 --rows 12     the 12 costliest block types, the rest in one row
    verity bench prove --system flock --workload glm53 --json        the table as JSON
    verity bench prove lean-verify --lean VERIFIER --proof TREE      the Lean verifier timed on one kept proof, as the
                                                                     record the table's Lean verifier figures read
"""

from __future__ import annotations

import sys

from verity_bench.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if sys.argv[1:2] == ["lean-verify"] else ["prove", *sys.argv[1:]]))
