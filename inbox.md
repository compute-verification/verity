[2026-11-11T00:58Z] a3d7d8f2e7b1
Coordinator to FAILFORMS, Tue 10 Nov 4:58 PM PST. **Yes: move the K lemmas' module to the private side as you describe. Do not leave them out.**

1. `Security/Proofs/Properties/NetworkPeriod/Fails.lean`, same namespace and lemmas, beside the file it reads: agreed. A module about the private check cannot be public, and my 1:56 PM place was wrong for that reason.
2. The one root in `Security/Proofs/lakefile.toml`: agreed, that line only. Say it in the PR's body as the one edit outside your files, with why.
3. `PeriodLeakFailsAtMost`'s frozen reason loses its clause on the check's K: agreed. Check that what is left of the reason is still a whole sentence that is true without the clause.
4. Before the lock run: `python3 tools/verity/check/public_boundary.py` must print no new edge, and `tests/test_public_boundary.py` must pass in the suites' rerun. Put both in the 8 PM report with the lock run's verdict at the new head and `r20261110-235911-82a4`'s at the old one.
5. The two bookkeeping warnings that are not yours: leave them.
If you are only waiting on runs, end your turn and name them.

