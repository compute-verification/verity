[2026-11-13T04:13Z] 55a40d3d4ad9
network-accounting (lead) to TERMGB, Thu 12 Nov 8:13 PM PST. **#9454 is on `main` (`d1a5cb986a2c`), so #9533 and #9501 each take `main` now. And soundness ruled both things I had left for Monday: controls and the rate. Both go into #9501's reader before its mark.**

**1. #9533, first**
- Merge `origin/main` (tip `e2665bcddcd` at 8:11 PM PST) into it, run its suites in the pool, and send me one line: head, run, verdict. I mark it when the run is exit 0. From my mark, push nothing to it.

**2. Controls** (soundness on NET-403, Linear's stamp 7:41 PM PST; read whole by me)
- **The rule:** an execution phase is held only where each verifier that accepted has, at the same commit, a control of record that it refused. One control a verifier a commit serves every phase checked under that commit. A phase whose verifier lacks its control at its commit is charged its whole schedule, with the line naming which verifier.
- **Which control each owes:**
  - stop record: a stop word changed (the count off by one);
  - response tie: a released id changed;
  - request tie: a request word changed (`max_tokens`);
  - compute audit's second stage: a committed output changed.
- Row 12 stands as I ruled it.

**3. The rate** (soundness on NET-390, Linear's stamp 8:01 PM PST)
- **The rate is the registration's, not the verdict's own word.** The stop record's and the tie's populations and rates are registered before any coin. Each of those verdicts carries the rate it ran at.
- The reader holds a class only where each verdict's rate equals the registered one and the registered one is `1/1` for both. A verdict that states `1/1` under a registration that says otherwise is refused by name.
- So the reader is handed the registration itself, checks its digest against the `registration` the four verdicts name, and reads both rates from it. Where the rate sits in proofs' registration I have asked on NET-390. Until proofs names it, keep the path in one named constant, with a test that a registration without it is refused by name.

**4. Where it goes, and when**
- **Both in #9501.** A reader that can hold a class soundness's rule would not hold does not land, and proofs builds its join on one reader with one head.
- **The control record's form is yours**, as the reader's. Post it on NET-403 as soon as it is fixed, and no later than #9501's head: proofs writes three of the four control writers and needs it first.
- **Friday noon PST for #9501's head stands if it fits.** If it does not, tell me by Fri 13 Nov 10 AM PST which hour it is. Later than Sat 14 Nov noon PST needs my word first, since proofs decides then whether its Monday holds.
- #9501 also takes `main`. Not marked until its nine harness tests are accounted for in the body.

