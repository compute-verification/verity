[2026-11-10T13:12Z] aac4b2da5887
PATHJOINS (NET-306), #8648 at 2da7a5b6be: not ready to mark yet. What each run has reported, read from its record at 5:12 AM PST, as I said I would send.

- **49 of the 55 suites have passed at 2da7a5b6be in one run or the other; none has failed in either.** [read]
- `r20261110-112220-045a` (vy-nebius-cpu-3): running, 47 of 55 reported, all passed, none reused.
- `r20261110-115706-b6a1` (vy-nebius-cpu-2): running, 25 of 55 reported, all passed, 9 of them reused (8 from the run at 7e8fffb3ba, 1 from aa97376a5f), each because no file it read has changed.
- Of the 49, 48 ran at this head; `verity/ml/commitments` has so far only the reused pass from 7e8fffb3ba, and cpu-3 has not reached it. I will name any suite that is still only reused when I say "ready to mark".
- **In neither yet, six:** `verity-numerical`, `verity-core-evaluate`, `verity/core/protocols/pouw`, `verity/ml/circuits/ops`, `verity/ml/circuits`, `verity/ml/windows`. They are the suites the runner leaves to the end; in the 7e8fffb3ba run the six took 28 minutes together on 7 workers, and cpu-3 runs them on 2.
- **My estimate for "ready to mark" is still about 6:00 AM PST**, past the 5:30 AM PST you set. [argued] I send it with the list of suites the moment the sixth passes, and tell you at once if one fails.

Nothing is pushed to #8636. Both runs stay to their end.

