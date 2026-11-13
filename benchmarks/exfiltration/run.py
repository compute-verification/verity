"""`verity bench exfiltration [ARGS]`: GLM-5.3's exfiltration capacity in GB over a monitoring period at 10% of total
compute, the compute audit's part beside the network certifier's, every input labelled measured, modeled or assumed
(the README beside this file). Its code is `verity_bench.exfiltration`.

    verity bench exfiltration                 the figure, the inputs, each part, and the two side by side a period
    verity bench exfiltration --json          the same as one document
    verity bench exfiltration --help          every parameter and its default
"""

from __future__ import annotations

import sys

from verity_bench.exfiltration import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
