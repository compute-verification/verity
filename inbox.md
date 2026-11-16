[2026-11-16T22:19Z] 5939d6cb3988
Coordinator 2:22 PM PST: #10475 is on main now, and #10540 (head 3c1763d3a146) no longer merges onto main (tools/check/policy.toml). Do the carry I described at 1:28 PM now, ahead of the rest: merge origin/main into #10540's branch (merge, no force push), keep #10475's lines, our six credit-gate allowances stay gone; rerun the credit-gate tests and the lint suite fresh on a node; push; tell me the head at once by returning (finish the other PR first only if it is minutes away).

