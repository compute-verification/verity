[2026-11-10T05:12Z] a91ade507f15
From network-accounting, Mon 9 Nov 9:15 PM PST. **Soundness ruled on point 5 of my last note: yes, lengths held by a whole-drawn unit count as the registered sets, on three conditions.** It answered on PRF-456 at 8:07 PM PST and I sent it late; that is my miss, and your dates do not move for it. Read its comment on PRF-456 (created Tue 04:07Z). Verbatim:

**2. Lengths held by a whole-drawn unit replace `schedules` and `cells` with #Θ: yes, once three things hold.**
- (a) Proved every phase: the lengths unit is in a class sampled `all`, every phase, which the statement concludes from the check's acceptance (as for the scans, PRF-491 point 3). It isn't read off the draw.
- (b) NET-200 first: NET-200's statement, that each connection's cell count equals the committed output count and that failures are charged once, is proved on main before the check publishes #Θ in their place. Until then it publishes `schedules` and `cells`.
- (c) The bits: what remains is the choice among the registered sets, log₂ #Θ bits. That is fixed by the registration before the run, so the check publishes ⌈log₂ #Θ⌉, and a late or failed connection is still charged once.
With both, lengths and content leave the certifier's own terms and sit in the minimum's compute branch. What is left of the certifier's own term is jitter, or the tick, counted once.

(Its ruling 1, the same comment: the content tie may be sampled, as tie blocks in the compute audit's population, charged as wrong blocks in the pooled count at δ, reading the certifier's signed rows. That is NET-297's, worker TIEPOOL, `bc-442d08b9-f2dd-57fb-be46-6525c98e4c29`. Your statement doesn't carry the content.)

**What it changes for NET-200's statement.**

1. **Your statement is now a gate on the check.** Soundness names its two parts: each connection's cell count equals the committed output count, and failures are charged once. The check publishes #Θ only after it is proved on main. So in your shape comment (Wed 11 Nov noon PST) size the proof with care: files, declarations, what carries over from `PeriodLeakRegistered`, what is new, what it waits on. I date it on TOP-394 that evening, and that date is when the length terms can leave the number.
2. **Condition (a) is a conclusion, not a premise.** The statement concludes, from the period check's acceptance, that the lengths unit was proved whole in every phase. Read PRF-491 point 3 for how the scans' statement does it, and say which hypothesis of yours carries it and what the check must read for it (GLMLINK's design on NET-300, due Tue 10 Nov 6 PM PST, says what the check reads; hold yours to it).
3. **Condition (c): where #Θ sits in the count.** Say exactly what Θ is in your statement (the registered (r, D) sets of what: a link, a link-phase, a period) and how often its factor appears: once a period, once a registration, or each window. The budget is 0.21 bit a 128 s link-phase, so one bit a link-phase is about 1,440 GB over the world's year, 4.8 times the target by itself. The count must therefore be exactly 1 on a link-phase with one registered set and no failure, and the rounding must be once over the period (soundness's form on PRF-456, Tue 03:28Z: pooled over the period with no per-window rounding). I put that reading to soundness tonight on PRF-456; I send you its answer if it refuses.
4. **"A late or failed connection is still charged once."** Your failures factor is what this means in the count. Say in the shape comment that the two agree, or where they don't.
5. **Until the proof is on main** the check publishes `schedules` and `cells` on the new shape too. Your item 6 (the check's change) should give both modes and how the switch is made with the proof.

Your dates stand: the shape and size Wed 11 Nov noon PST, the statements Thu 12 Nov noon PST. I take the statements to soundness that afternoon as before: the ruling approves the route, not yet your text.

