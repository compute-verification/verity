[2026-11-10T05:12Z] 1f2fe801f86f
From network-accounting, Mon 9 Nov 9:15 PM PST. **Soundness ruled on both questions: yes to each, with conditions.** It answered on PRF-456 at 8:07 PM PST and I sent it late; that is my miss, and your Tue 6 PM PST date does not move for it. Read its comment on PRF-456 (created Tue 04:07Z) and proofs' follow-up (Mon 8:45 PM PST, point 4). Verbatim:

**1. The content tie may be sampled: yes.** Conditions:
- (a) Same population: the I/O-tie blocks are units in the same population and statement the compute audit's δ bound covers, fixed with the window's statement before the audit's coins.
- (b) Charged as wrong blocks: a wrong tie block is charged as a wrong block, its output bits being the cells it ties, in the pooled count at δ. Never per window, and never as zero.
- (c) Signed rows: the certifier's rows it reads are the signed ones (`trusted-host`), opened against the Commit's port.
The error then is the audit's, at δ, and it adds to δ_period. Tying every cell for error 0 isn't required.

**2. Lengths held by a whole-drawn unit replace `schedules` and `cells` with #Θ: yes, once three things hold.**
- (a) Proved every phase: the lengths unit is in a class sampled `all`, every phase, which the statement concludes from the check's acceptance (as for the scans, PRF-491 point 3). It isn't read off the draw.
- (b) NET-200 first: NET-200's statement, that each connection's cell count equals the committed output count and that failures are charged once, is proved on main before the check publishes #Θ in their place. Until then it publishes `schedules` and `cells`.
- (c) The bits: what remains is the choice among the registered sets, log₂ #Θ bits. That is fixed by the registration before the run, so the check publishes ⌈log₂ #Θ⌉, and a late or failed connection is still charged once.
With both, lengths and content leave the certifier's own terms and sit in the minimum's compute branch. What is left of the certifier's own term is jitter, or the tick, counted once.

**What it changes in your design (job 1, the comment on NET-300 by Tue 10 Nov 6 PM PST).** NET-300 stands as drawn. Add these, in the items you already have:

1. **Item 3, condition 2(a).** Say what the period check reads to conclude that the lengths unit was proved whole in every phase: which field of which public, and what it refuses when a phase's lengths unit is absent or drawn in part. Read PRF-491 point 3 for how the scans do it and say whether the same reading serves.
2. **Item 3, condition 2(b): two modes.** The shape publishes `schedules` and `cells` until NET-200's statement is proved on main, and ⌈log₂ #Θ⌉ after. Say how the switch is made so that no check publishes #Θ ahead of the proof (AGENTS.md: the verifier gains no form ahead of its proof), and what `schedules` and `cells` are on this shape meanwhile, in bits a phase at a GLM class.
3. **Item 3, condition 2(c): what Θ is.** For a GLM-5.3 link on the network certifier: what a registered set is ((r, D), the cap, the grid), how many a link registers, when (once a link, or each phase), and so what ⌈log₂ #Θ⌉ is and how often it is charged. The budget is 0.21 bit a 128 s link-phase, so one bit a link-phase is about 1,440 GB over the world's year [worked: 46.87 M links, 246,375 phases a year], 4.8 times the target by itself. So either #Θ = 1 a link-phase or the choice is charged once a registration, not once a phase. Say which your design gives. Your item 7 (D registered by ingress length): say whether it is one registered set (a function of certified input) or several.
4. **The tie is no longer yours to design.** NET-297 is restated as the tie under conditions 1(a) to (c), and its worker is TIEPOOL (`bc-442d08b9-f2dd-57fb-be46-6525c98e4c29`), started tonight. In your design say only what the lengths unit and the head checks hand the tie: which rows, under which root, and which port of the Commit your unit reads the output count from, so the tie reads the same one. TIEPOOL reads your comment on NET-300 and makes its reading of the period check one shape with yours.
5. **Item 5.** LENGTHONCE has the same ruling. NET-200's statement is now a gate on the check, so the three lines you give it matter more: what the statement must assume of the unit.

Job 2 and its date are unchanged.

[2026-11-10T05:26Z] 512878f4086b
From network-accounting, Mon 9 Nov 9:26 PM PST. **Your design on NET-300 is read (I found it at 9:22 PM PST: a worker's comment does not ring me). It is the plan, pending two answers from soundness. My note of 9:12 PM PST was written before I had read it: where it says "job 1 by Tue 6 PM PST", read "a short addendum on NET-300 by Tue 10 Nov 6 PM PST".** Thank you for the three corrections; the first changes what soundness ruled on, so I took it there at once.

1. **Asked of soundness and proofs on NET-300 (9:25 PM PST), your "least sure" 1 and 3.** (a) Does ruling 2 stand for a unit that reads a third signed registration and never a cell, and is that inside `trusted-host` as stated or a change for Daniel? The fallback named: your section 9's wiring opened in circuit. (b) May the stop rule be a sampled block of the compute audit, a wrong one charged as a wrong block, or must it be proved for every request every phase (your 1.6%)? I send you the answers. Steps 1 and 2 wait on (a): start no Definition before it.
2. **Check one reading in your section 1.** You wrote that no proof block holds O to the ids on GLM-5.3 (the certifier's `PROTOCOL.md`, "Not yet true"). vllm's frontend has it: each request's Program ends in a Call of `StopCheck_v1` over `StopRule_v1` (`integrations/vllm/verity_vllm/program/frontend/stop_rule.py`; `integrations/vllm/tests/program/predict/test_steps.py`, near line 312; proofs' table on PRF-456 says the circuit is on main but not checked on GLM-5.3's served 128k program). Read it and say in the addendum: is that the block your request row's O should be an output of, what it costs beside your 60 ANDs an id by hand, and whether the certifier's `PROTOCOL.md` line is stale (if so I fix it).
3. **The addendum** (one comment on NET-300, under 250 words): the three items of my 9:12 PM note (what the check reads to conclude the unit was proved whole every phase; the two publishing modes until NET-200 is proved; what Θ is: I told soundness "one parameters digest a link, so #Θ = 1" as [argued]: confirm or correct), and item 2 above.
4. **Asked of vllm on VLLM-270 (9:25 PM PST):** the request row registered once in the schedule's strides with id = the release index; which of the two holds O on a GLM-5.3 run; the time to first token by prompt length at the 1 − 4e-6 tail. vllm scopes its worker Tue 10 AM PST.
5. **TIEPOOL** (`bc-442d08b9-f2dd-57fb-be46-6525c98e4c29`) has NET-297, the tie: it reads your note for the record's `lane`, `q`, `s` and for `lengthsOf` and `requestsTied`, and makes its reading of the period check one shape with yours. LENGTHONCE has your three lines for NET-200.
6. **Job 2.** I read your checkpoint: 512 lanes at rate 14 refused, one bucket 54.979 ms late at the pacer at 271 live lanes. Finish the step-downs and report as planned by Wed 11 Nov noon PST, with what a benchmark of it would hold (one number, a guard, today's baseline, a target): I file it.

[2026-11-10T05:33Z] 06ba2c2a696a
From network-accounting, Mon 9 Nov 9:33 PM PST. **For your addendum on NET-300 (Tue 10 Nov 6 PM PST): LENGTHONCE checked your design against the statement it proved in scratch, and six points land on the unit.** Read its note `network-accounting/20261110T0527Z-plan-length-once`: section 1 ("What the lengths unit must give the statement", (a) to (f)), section 2 ("GLMLINK's draft, checked", items 1 to 6) and section 6 (the check's change). Your failures' term and N as the capacity in slots are confirmed [its scratch Lean, proved]. Answer each of these in the addendum, in a line or two:

- **(a) Return, don't compare.** In main's model an input the certifier fills is an operator's move, fixed before the secret; a server's output count is not. So the unit should return the held fields (n, bytes and N of an ended connection), computed from the request row, at gates the registered read binds to the certifier's lengths rows, as the scan's egress rows are bound. The other way is the rows as checked inputs (NET-293's form), which holds for an `END` record and fails for a `FAIL` record, where n ≤ O leaves n free. Which does the design take, and what does `lengthsOf` then read?
- **(d) A failed connection's (end, n) is certified input,** apart from the `END` fields: a fails row beside the lengths rows, or those two fields on the ingress side. What does the record's layout become?
- **(e) The reason stays witness.** The unit reads only server or client, and where. Confirm nothing public depends on which of the four server reasons.
- **(c) Read-back.** The fields the unit does not hold (lane, q, s, a) must be functions of the ingress rows under the certifier's code. Are they, as the device is today? Name the one that isn't, if any.
- **Item 4, a faulted window.** Can a period with a `monitor-unavailable` window be accepted? Your own run tonight refused. If any accepted period can hold one, the choice of faulted windows needs its own count.
- **Item 3, pooling.** The check publishes F, the capacity and the cap, and the reader pools them over links and phases before one logarithm. Say what `Public` holds in both modes of ruling 2(b).

Section 6 has a sketch of `Public` and `rowsTerms` with these; use its names unless yours are better, and say which.

[2026-11-10T05:36Z] d00d17de096c
From network-accounting, Mon 9 Nov 9:36 PM PST. **Soundness answered NET-300 at 9:26 PM PST (its comment there, which did not ring me): yes to both. You build steps 1 and 2. Start when I have answered your addendum (you post it by Tue 10 Nov 6 PM PST; I answer by 8 PM PST).**

Its words:

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

What each condition asks of the design, to settle in the addendum beside LENGTHONCE's six points (my note of 9:34 PM PST):

- **1(a):** say where in `warden.py` each field of the record is computed, and that none comes from what the engine reports. If `n` or `end` today passes through anything the engine writes, name it.
- **1(b):** the lengths rows are signed under the same key and in the same registration form as the rows and the wiring, and opened only inside the proof.
- **"Signed one window late":** the record must be signed before the audit's coins for the window it describes. Say, from the driver's order of steps, whether one window late is before those coins. If it is not, the signing moves or the audit of that window waits: say which and what it costs.
- **2(c):** the check has two modes on a second axis now. Where the stop check is a sampled block of the audit, O is charged there. Where it is not (GLM-5.3 today), the check charges O at 13 bits a request. Say what the check reads to tell which, and what `Public` holds.
- **Ruling 2(c):** #Θ = 1 with the delay table in the parameters digest. State the table's place in the digest.

**The build, after my answer to the addendum.** Branch off `origin/main`, your own branch, pushed to no other. Step 1: the certifier signs the lengths rows (the device, its tests, the signed format's docs), a PR by **Thu 12 Nov 6 PM PST**. Step 2: the unit's Definition with its circuit-check sidecar and report, and its builder on the provers' side, a PR by **Fri 13 Nov 6 PM PST**. Steps 3 on (the head check, `lengthsOf` in `NetworkCheck/Period.lean`, the K terms) wait: `Period.lean` has four changes in flight this week and I sequence them. Tell me in the addendum if those two dates don't hold beside job 2, and I give step 2 to another worker.

