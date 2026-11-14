[2026-11-14T08:30Z] fbf80ae9e5aa
READCALC, Sat 14 Nov, 12:29 AM PST. PR #9999 (lock), read at its new head under the new ruling.

**ea8f747441fecd5d61b8b2a5a2a2807e04eb9c0e READY, 0 must-fixes: the 26 differ in `issue` alone, exactly as ruled.**

- RAN, parsed JSON against base 512644d14f (main's lock at 9a520e76c0 is the same):
  - The 26 open entries `@training` owns differ only in `issue`. 16 carry TRN-167: the eleven named, plus CheatingAtMost, IdentitySchedule, PouwPerWindow, SampledPerWindow and UnmonitoredAtMost. 10 carry TRN-168: CoversEfficientMethods, DescriptionHard, MinEntropyHard, Recovers, SourceHard, OverlapWeights, RequiresState, Consumers.Upstream, HeldPriceAtMost and Selects.
  - Every owner stays `@training`, and `owner`, `digest` and `due` are unchanged.
  - No other open entry differs. Key order is equal, and the rest of the lock is equal.
  - The textual diff is 26 `issue` lines out and 26 in.
- RAN: `merge.py` main→head exits 0 with the head's lock; `tests/test_lean_packages.py` 35 passed; `git merge-tree` with main is clean.
- RAN, the body: its 26 table rows match the lock, and it says nothing older (no owner move, no 15).
- READ, the citations it names: `classes.py:280` cites CoverTraining, `envelopes.py:486` cites DampenerRemote, and `theorem.py:268` names DampenerRemotePeriod in a docstring. The lock's `DampenerRemotePeriod` takes the five.

NOTE:
1. INFERRED, not checked: "the server-block sum has that bound's form" (nothing cites ServingBlockBound), and "network accounting read all ten".
2. READ: the first commit's subject ("the policy's 15 to network accounting") is false at the head, and commits land on main as they are.
3. I found no run at this head. The landing needs `suites.py --changed` and `security`'s Lean records on a node.

