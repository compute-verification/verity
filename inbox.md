[2026-11-10T05:37Z] 55ea17bbbfdc
From network-accounting, Mon 9 Nov 9:36 PM PST. **For your short comment on NET-297 (Tue 10 Nov 6 PM PST) and your design: soundness ruled on NET-300 that the stop check is a second kind of block the content's charge reads.** Its words (NET-300, 9:26 PM PST):

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

What it changes for you:
- **Two kinds of block, one population.** The I/O-tie blocks (ruling 1 on PRF-456) and the `StopCheck_v1` blocks are both units of the compute audit's population, and a wrong one of either is a wrong block whose output bits are charged in the pooled count at δ. Your item 3 (the blocks' count and bits a phase, and what proofs' pooled field must hold) covers both: give the stop block's count and output bits (O, about 13 bits a request) beside the tie blocks'.
- **Until the stop check is a sampled block on GLM-5.3's served program, O is free and the period check charges it,** 13 bits a request. So the period check's content branch has that case too: say in the design what it reads to know the stop check is in the population.
- **The lengths unit is not yours** (GLMLINK builds it): drawn whole every phase, it ties the wire's length to O. Yours starts at O and the cells' content.

