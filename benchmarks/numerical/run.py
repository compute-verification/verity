"""`verity bench numerical JOB [ARGS]`: one of ``verity_numerical``'s tools, started with ARGS as ``python -m`` starts its
module.  With no JOB it prints the jobs, one line each (``verity_numerical/__main__.py`` holds the list; README.md says what
each reads, writes and needs).

    run.py                                       the jobs
    run.py instances verify                      the committed input set against its manifest
    run.py unit_costs --out DIR                  the per-unit-type cost inventory, written under DIR

None of them needs a GPU.  ``cell`` and ``placement`` reach the pods a cell names, through ``research``.
"""
from __future__ import annotations

import sys

from verity_numerical.__main__ import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
