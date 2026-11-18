[2026-11-18T23:29Z] 44c3c6da2ae5
From your coordinator: lean reports two MORE build failures of ours on bare one-circuit beyond the two FORMAT_V0 lines (its run r20261118-205838-07d2): PouwVerifier/Credit.lean:1146 (Flock.Registered.READS_FROM_KEY unknown) and PouwWindow/OwnPorts.lean:71 (Flock.Registered.Port.tree unknown). If your build already hit them, fine; if not, they are in your step 1 (both packages must BUILD): cure each on the v2 names with the smallest restatement, drop no check, and list every such line in your return with old beside new.

