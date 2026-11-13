[2026-11-13T07:07Z] 8788cf2f1d30
From benchmarks (KITMARK). Top, Thu 11:05 PM PST: tonight every path-only move toward the page tree lands as two chains on main by Fri 7 AM PST. Your rename PR (stacked on #9590) is NOT marked: do not mark it until the chain is on main; then carry it across with `tools/move/restack.py` (not by hand), rerun its suites and mark it. #9590 itself has a land verdict at d4837e4c and is marked; leave it.

