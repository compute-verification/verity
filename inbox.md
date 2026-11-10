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

