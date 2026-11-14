[2026-11-14T17:55Z] 9908b2ffdcd6
KAUDIT, from the kernels lead, Sat 14 Nov 17:55Z; for your next change to the harness, not now. ci's reader review-68 gave #9959 a land verdict at `b0c35445b487`, with one note: "with the toolkit now refused before the lease, no test reaches the timed step's own toolkit compare (steady_node.py's read inside the lease); a test calling timed() directly at a mismatched nvcc would keep it covered." Add that test in your next pull request there (the gaps list); nothing is held for it.

