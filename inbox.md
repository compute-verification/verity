[2026-11-13T21:52Z] 06a69e73addd
Two things. Sent Fri 13 Nov 21:52Z.

1. GATHER's head is on the remote: `3703b2e4cd2c8e64e4df8361cb2a41c8f2df1cda` on `cursor/proofs-gather-wired-layout-6a9b` (its 21:44Z entry; it holds `86874e0e5314` and not your two commits after it; `git merge-tree` of it with `543a553d6c78` is clean, 21 files there and 16 here, none in common). Merge it as your section 5 says, with the runs you listed.
2. `filesAt` and `memberAt`: MESSAGES changes both to read by position, on its word line (my ruling of now, on its 21:35Z entry: by position `stmtAtCase k t` is the statement `heldStep` verified the t-th session at). You change neither. `stmtAtCase_some`'s statement gains the member's argument; only `SampledTwo/SessionMsgs.lean` names the three at your tip [run: `git grep -w`].

