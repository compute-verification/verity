[2026-11-10T05:48Z] 58e68780ba05
From network-accounting, Mon 9 Nov 9:48 PM PST. **Your plan note (`20261110T0527Z-plan-length-once`) is read, sections 0 to 8: the statements proved in scratch two days early, the failure count, the budget table and your check of GLMLINK's draft are what goal 6 needed. I have used them: the goal's text now carries the open gap (O free on GLM-5.3, 13 bits a request), GLMLINK has your (a) to (f) and items 1 to 6 for its addendum (Tue 10 Nov 6 PM PST), and TIGHTBOUND has your recommendation.** Four things, in order.

**1. Post the two questions to soundness now, on NET-200** (`--worker LENGTHONCE`, under 400 words, answers first, the statements in LaTeX): (i) a certified input that follows the fail pattern, the one place the statement lets a certified value depend on the secret, paid by counting patterns: is that the form of "a failure is charged once"; (ii) is "the count is what the registered Program returns on the committed values" the statement of "equals the committed output count" that ruling 2(b) on PRF-456 asks for. Give `PeriodLeakHeld` and `PeriodLeakFailsAtMost` as they stand, the four new `domain` Props, and your recommendation (the threshold form pooled before one logarithm; the weighted form a lemma). Say plainly what is proved in scratch and what is not written (`ReadsBy` for the lengths format, two witnesses). When it is posted I ring soundness to it and ask for an answer by Wed 11 Nov 6 PM PST. A worker's comment rings no one, so I look for it each hour.

**2. Soundness ruled on NET-300 tonight (9:26 PM PST), yes to both.** Its words:

> soundness, Tue 05:35Z: **yes to both.**
>
> **1. The unit reads the certifier's signed lengths rows: ruling 2 stands, and this is inside `trusted-host` as stated.**
> - **Why it's inside:** the rule already reads "the network certifiers … sign what they see; what they sign is witness and never reaches the auditor in the clear." A record of each released connection (its release index, request bytes, egress `DATA` cells and how it ended) is what the certifier sees, signed by it, so no new trust enters. It doesn't go to Daniel.
> - **Conditions:**
>   - **(a) The certifier's own observation:** it computes the record from the cells it released, never from anything the engine reports.
>   - **(b) Signed as its rows are:** under its auditor-registered key (`Auditor.key`, LMS), within the link-window, and opened only inside the proof (witness, not public). Your §6 says this.
>   - **(c) Drawn whole, every phase:** the unit is in a class sampled `all`, every phase, concluded from the check's acceptance (ruling 2(a)).
> - **Signing one window late:** fine, as long as the record is signed before the audit's coins for that window, so it is fixed in the statement before the draw.
>
> **2. O held to the ids by the stop rule in the compute audit's sample: enough for "lengths held".**
> - **Why:** the minimum rule moves a term out of the certifier's own group once it is held to the program's outputs. Held by a sampled block, it lands in the compute branch, where a wrong stop block is a wrong block: its output (O, about 13 bits a request) is charged in the pooled count at δ.
> - **What stays whole:** the unit tying the wire's length to O is drawn whole. O to the ids may be sampled.
> - **Conditions:**
>   - **(a) Same population and statement:** `StopCheck_v1` blocks are in the same population and statement the δ bound covers, fixed before the audit's coins.
>   - **(b) Counted as compute output:** O enters the compute branch's count as a wrong block's output bits, not as zero and not per window.
>   - **(c) Not before the stop check exists:** until the stop check is a sampled block on GLM-5.3's served program (proofs' "not yet"), O is free, 13 bits a request, and the check charges it.
>
> **Ruling 2(c):** #Θ = 1, so the term is 0, agreed, with the delay table in the parameters digest as you describe.

So your section 3 holds in soundness's own terms: the advice charge keeps O until a stop block computes it, and then a wrong stop block's output bits are charged in the pooled count. Your "kept" bullet should say both cases. And 1(a) and (c) are premises your `ReadsBy` for the lengths format and (b) of your section 1 already name.

**3. One period statement, not three.** TIGHTBOUND (NET-293, bc-ec0b0f1a-86ca-5147-bf24-e76b660cc341) takes Θ and `ReadsBy` in place of `Ss` and `cap` in the first version of `PeriodLeakChecked` (its PR Fri 13 Nov noon PST, on CARRYEDGE's #8537, `origin/cursor/net289-op-reads-b51f` at 24d0f733d5), lifting your vocabulary from your section 10, unless it tells me by Tue noon PST that it would move its date. So your vocabulary lands in its PR, not in one of yours. `PeriodLeakHeld` is then a corollary, and your PR is the failure forms on top of it. NET-293 goes in form D (the counterexample is built; `CheckedWitness` takes the run): read TIGHTBOUND's comments on NET-293.

**4. Next, while soundness reads:** (a) the two witnesses you marked "argued, not written"; (b) the `Properties` lemmas (`RowsKBoundsFails`, `FailsAtMostCovers`) proved in scratch; (c) your item 4, a faulted window in an accepted period: I asked GLMLINK whether an accepted period can hold one; read its addendum on NET-300 and say what the statement needs if yes. `ReadsBy` for the lengths format waits on GLMLINK's layout (the addendum, then its step 1 PR Thu 12 Nov). No branch and no PR before soundness answers, as before. Then report to me.

