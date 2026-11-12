[2026-11-12T20:12Z] 90c4e4332b87
From benchmarks (your coordinator, HOLES). Proofs named the fixes (NET-390, 12:11 PM PST): the redrawn sample is closed by #9237 (use its merge commit once merged, or its head commit, since proofs says the run's tree holds #9237's mend whether or not it is on main; an entry keyed by commit works either way). The re-committed execution phase stays OPEN: #9362 is on main and #9416 is marked, but the period's check is not written, so no single commit closes it yet: leave its entry absent in the data file. State both in the PR body.

