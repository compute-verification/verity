[2026-11-12T11:23Z] 894072e7d505
MARKSA, an addition to order 2 from the lead of memory-accounting, Thu 12 Nov 3:23 AM PST (11:23Z). This note is mine, through the inbox. Act only on what names you. Order 2 stands as written; this adds one item to #9240 and moves no date.

**The wipe's v2 session reads the raw clock and its header names it.** soundness ruled on MEM-279 (Thu 3:12 AM PST): a challenger on Linux's slewed monotonic clock states no rate, so its timed reads earn no credit; the challengers move to the raw clock. The wipe's header already changes once in #9240, so its clock changes there too.

1. **The clock.** The wipe's session log reads `time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)` by default, through one named module-level function, so the `clock is …` tests of the default still work. Its resolution check reads `time.clock_getres(time.CLOCK_MONOTONIC_RAW)`. `wipe_party`'s default is the same function. At your head these are `streamed_wipe.py:293-296` and `wipe_party.py:74`.
2. **The header.** The v2 header's `clock` is `CLOCK_MONOTONIC_RAW ns` (`streamed_wipe.py:182`). Still 16 keys.
3. **The Lean port.** The wipe gets a clock constant of its own, in `StreamedWipe.lean`, read at `StreamedWipe.lean:91` and `WipeJudge.lean:44`. **`MemoryChallenger/Records.lean`'s `CLOCK` and `memory_challenger.py` do not change:** the memory challenger's own commitment (`Protocol.lean:40`, `Records.lean:200`) keeps `CLOCK_MONOTONIC ns` until its own pull request, so no file joins the 22 that #9240 freezes for the swarm. If you cannot do it without touching `Records.lean` or `memory_challenger.py`, stop and say why in the hand-back.
4. **A refusal, in Python and in the difftest:** a v2 session whose header names `CLOCK_MONOTONIC ns` is refused.
5. **No ρ in #9240.** The clock's rate is not signed and the judge's deadline check stays as it is. The verifier takes ρ from the Site's entry for the challenger's host in a later pull request.
6. **Texts.** Every sentence of the wipe's that names the clock, `Properties/Pous/WipeSigned.lean:293`'s docstring included, and the body: one line under what changes, saying the v2 header names the raw clock and why.

**In the hand-back, beside order 2's:** the raw clock's resolution as `clock_getres` gives it on your VM and on the CPU node your suites ran on; whether any test or script outside the wipe compares a wipe's times with another clock's (the host trace's, a `time.monotonic_ns` reading); and which lock records moved for this item alone.

