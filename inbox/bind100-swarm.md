# The 100-bit swarm's mailbox (read before each comment)

The shared context is PRF-523 (a copy: `inbox/attachments/bind100-context.md`). The draft: `inbox/attachments/spot-security-definitions.md`.

## Tue 10 Nov 23:36Z, coordinator

- Nothing new since your brief. Daniel's rulings and other workers' headline findings appear here, newest last.

## Wed 11 Nov 00:00Z, coordinator: Daniel wants a yes or no straightaway. Lead your next comment with one line. One candidate to attack first.

Daniel, Tue 3:51 PM PST: "What is the fastest way you think we can get clarity on whether we have something working? More bits of security? Like I want to know straightaway because I'll be sending this to people."

**Every option worker (PRF-524 to PRF-533):** post your first comment as soon as you have a headline, inside the 90 minutes, and lead it with one line:

`REACHES 100: yes | no | only at <points> — <the bound, one formula> — floor / B / A in bits — rests on <premise, on main or new> — confidence: derived and checked | derived | guess`

**Candidate R (PRF-542; `inbox/attachments/bind100-candidate-r.md`):** replace the two-run finder by one that repeats the prover until two runs open a node differently, so it succeeds with constant probability and pays in time. The coordinator's sketch gives about 2^-96 to 2^-117 at one session an audit, SHA-512, no random oracle, `ExpectedCR` as on `main`. It is unchecked.
- **KILIAN (PRF-524), NFACTOR (PRF-527), TAIL (PRF-526): read it before anything else and say in your first comment whether it holds, where it fails, and what the literature's version of it gives.** It is the "finder of cost T w / eps with constant success" form of your briefs.
- SCALE (PRF-533): step 5's cost carries n_sess; can the finder extract one session a run?
- ATTACK (PRF-532): is there an opening distribution that defeats step 3?
- TARGET (PRF-531): the premise is asked of a finder with about 2^254 expected calls; is that how the systems we compare with state collision resistance?

**Two facts the coordinator read or computed since your brief** (check, do not trust):
- The Lean's premises take any `H : Msg → Msg`; the digest's width enters only as the constants `2^256` and `2^513` in `ExpectedCR` and `StrictBirthday` [read, `Security/Definitions/Core/Interaction.lean:276-287` at `2b869ed35db5`]. WIDEHASH: a 768-bit digest does not reach 100 under today's theorem (2^-97.6 at best, 2^-58.5 at worst); 1,024 bits gives 2^-161.6 to 2^-122.5 [the context's script with the exponent changed].
- ROM: if the base argument's extractor still rewinds, a collision among all calls of the run and one session's extraction is bounded by 1.9 (E^2/2^513)^(1/3) with only its expected cost E known: 2^-104.6 at one session and the smallest k, 2^-95.9 at the largest. A tail bound on the extractor, or a straight-line one, is what makes this route large.

## Wed 11 Nov 00:15Z, coordinator: two pointers for the auditors (AUDIT-FLOCK, AUDIT-CRYPTO, AUDIT-SPOT, AUDIT-HEADLINE, AUDIT-TABLE). Check them; they are not findings yet.

- **A deployed guarantee whose number is "below 1".** `Flock.Guarantees.DeltaPubDeployed` states only that `Bridge.deltaPub … < 1` at the deployed parameters (`k = kDep = ⌈2^24/e⌉`, `ρ = 1/2`, `M = R_w = 2^192`, `t' ≤ 2^80`, `v ≤ 2^80`, `W ≤ 2^40`, `K₀ ≤ n ≤ K₀·k_d`) [read by the coordinator: `Security/Proofs/Flock/Bridge/PublicDeployed.lean:248-257` at `2b869ed35db5`; it is in the lock's `deployed` list]. Its proof bounds the sampling term by `e^-1 < 1/2`. **AUDIT-SPOT and AUDIT-TABLE: evaluate each term of `Bridge.deltaPub` at those parameters, and say what the terms other than the sample's miss come to in bits.** AUDIT-HEADLINE: where is this guarantee quoted, and as what?
- **A soundness guarantee that reads a hiding premise.** `Flock.SecurityProofs.RecursiveKSDeployed`'s record has `Claims.Hm96Hiding` among its open premises, with `Claims.Flock.ListWithin` and `RecDeploy.DenseAt` [read by the coordinator: `Security/lean-audit.json` at `2b869ed35db5`]. `Hm96Hiding`'s docstring gives `δ₁ ≤ 2^-193` a leaf "at all but `2^-64` of keys" (`Security/Definitions/Core/Hm96.lean:31-37`); it is read in `Security/Proofs/Flock/Recursive/KSCoinTreeStatement.lean:48`. **AUDIT-FLOCK: how does `δ₁` enter the recursion's knowledge-soundness bound (times how many leaves and levels), and what is the number? AUDIT-CRYPTO: this is one more place the `2^-64` split reaches, and it is a soundness bound, not a zero-knowledge one.**
- The open premise under SPOT's deployed guarantee `SampledKSLiveV5` is `RecordCustodyZKJPLaw` (custody of the record between the challenger and the verifier), with `SHA512CRExpected`, `SHA512CRStrict` and `UniformRandomBytes` assumed [the same lock]. It is not a number, so it is outside the audit's table, but DRAFT-CORE and DRAFT-REST should check that the draft says it.

## Wed 11 Nov 00:25Z, coordinator: Candidate R's step 3 fails; what stands, and the three questions that now decide it

Read PRF-542's comments of 00:13Z (TAIL), 00:19Z and 00:22Z (TAIL's final) before anything else on Candidate R.

- **Step 3 fails as I wrote it** [the 00:19Z family checked by me by hand: bad runs of mass δ each open one uniform node to false; a run opens every node to true with probability δ/N′; the tie goes to true; two bad runs never disagree, so M runs find a disagreement with probability at most Mδ/N′]. The expected count of disagreeing pairs is constant and sits on a rare event.
- **What stands** [two workers' derivations, the core inequality rederived by me, simulated at small sizes, not in Lean]: `N′` in place of `√N′` (or `√(N′·n_open)` where one run opens at most `n_open` nodes), and one factor of `L+1`. With the 00:19Z constants, `14.64·128·(N+1)·c_run/2^256`: floor / B / A at one session, k = 1,507, T = 2^80: 2^-103.1 / 2^-99.0 / 2^-88.1 [arithmetic rechecked by me]; at `n_sess = n_PU`, which is what `main`'s instances have, 2^-87.3 / 2^-74.8 / 2^-53.0.
- **The three questions, in the order that decides most:**
  1. **ATTACK, SCALE:** in SPOT's own two-stage game, can a prover's rare runs open far more nodes than a bad run must (`n_open ≫ n_min`)? TAIL's conjecture is `M = O(max(√N′, n_open/n_min)/δ)`, which gives back 2^-123.9 / 2^-121.8 / 2^-116.3 where every block is sampled independently. Build the prover or say what in the game stops it. TWOPROVE reads a refused second stage as opening everything (`Reg.lean:94`): check it.
  2. **REPPROVE, REPBREAK (finals by 02:30Z):** the corrected statement in the two-stage game, where the second stage's commitments follow the first sample. Neither comment checked it. And the 00:19Z lever: the least-bad assignment in place of the plurality.
  3. **NFACTOR, KILIAN:** is `N′` (linear in the proof length) the known barrier? TAIL cites Chiesa, Dall'Agnol, Guan, Spooner and Yogev (ePrint 2024/1434): Theorem 1 linear in ℓ, improving it open; Theorem 2 carries the opened count. Say whether anything in the literature gets below linear for a tree commitment under a linear collision premise.
- **One session an audit** is now worth 16 to 35 bits (the columns above), so TWOPROVE's route 3b and SCALE's answer on `n_sess` matter more than they did.

## Wed 11 Nov 00:32Z, coordinator: where the options stand after six finals (TAIL, COMMIT, WIDEHASH, AUDIT-CRYPTO, AUDIT-RESOURCE, DRAFT-REST)

Nothing below is checked in Lean. "Reaches 100" is at T = 2^80 unless said.

- **Two routes reach 100 at every point, both by changing the protocol:**
  - **PAD** (COMMIT, PRF-530 final 00:21Z): a leaf that binds statistically (the row XOR a keyed pad, opened in the clear), so the binding finder never extracts: 2^-130.8 / 2^-126.6 / 2^-115.7 at floor / B / A, at every session count and k. Hiding becomes computational. Being checked now: soundness by TAIL, hiding by AUDIT-CRYPTO, cost by WIDEHASH, finals 03:15Z on PRF-530.
  - **A 1,024-bit digest** (WIDEHASH, PRF-529 final 00:23Z): today's theorem with the exponent moved, worst cell 2^-122.5; 2.2 to 2.4 times the serve-time hashing. 768 bits does not reach it.
- **Proof-only routes:** Candidate R as corrected reaches 100 only at the floor at one session (2^-103.1 / 2^-99.0 / 2^-88.1). With a 640-bit digest under it, every cell at 30 sessions or fewer. Capping the extractor is dead: while the term is `2N·sqrt(β')` no rewinding extractor passes 79.2 bits (TAIL, PRF-526 final 00:25Z).
- **Two facts every remaining card must use:**
  1. **Count T in compressions.** The premises count whole SHA-512 evaluations (`Core/Interaction.lean:262-288`, `Assumptions/Flock.lean:38-78`, read by me), and for an iterated hash the generic bound holds per compression: L-block messages with a common suffix collide L times as often (WIDEHASH's smell, simulated on a toy hash; the argument checked by me). Say what your bound loses with T in compressions at 4 to 64 KiB rows.
  2. **`main`'s instances have `n_sess = n_PU`** (`SamplingGeneric.lean:250`, `Deployed.lean:124`; TAIL's read). Give that column, and say whether your route needs one session an audit.
- **A lever two workers found from different sides** (DRAFT-REST's first smell, COMMIT's UNITS): the rewinding lemma proves `sqrt(N·Q·β')` and the proof passes `Q = N` (`Core/Game/Rewinding.lean:175`, `SamplingGeneric/Reg.lean:411`). **NFACTOR: this is yours: the bound with Q the positions one run opens and N the boundary positions, at the three points.**
- **Still owed from you** (first comment with the lead line if you have not posted one): KILIAN, NFACTOR, ROM, EXTFORM, SCALE, ATTACK, TARGET; AUDIT-SPOT, AUDIT-FLOCK, AUDIT-HEADLINE, AUDIT-TABLE; DRAFT-CORE. ATTACK and SCALE: the question in the entry above (can rare runs open far more than a bad run must) now decides whether Candidate R alone is enough.

## Wed 11 Nov 00:43Z, coordinator: for AUDIT-FLOCK (PRF-535), one number another lane needs

compute-research posted its own table on PRF-537 (Tue 10 Nov 4:27 PM PST; `note:compute-research/20261111T0023Z-finding-security-numbers`). Its Dither guarantees carry a term `κ_S`, C-Flock's error for **one outer proof**, against a design budget of 2^-57.1 a proof (2^-43.4 a replica-week). With the 2^-33.6 we quote for one audit's binding terms it would be 2^-19.8 a replica-week.

- **AUDIT-FLOCK: in your final, give C-Flock's error for one outer proof as Dither reads it** (which statement of ours `κ_S` is, at which sizes, each term, in bits at T = 2^80 and 2^64), and say whether 2^-57.1 is met. If `κ_S` is not one of our statements as written, say which one it should be.
- AUDIT-TABLE, AUDIT-HEADLINE: its table also has Pearl-C at 2^-64 at the cap and 2^-48 at 2^80 calls (agrees with AUDIT-CRYPTO's and AUDIT-RESOURCE's), and a paused logging-chip specification that took "κ <= 2^-80 a proof" from an older note. Cite its table for its rows; do not rederive them.

## Wed 11 Nov 00:48Z, coordinator: a proof-only route found four times, and a checked number already on `main`. Targets change for REPPROVE, REPBREAK, SCALE, ATTACK, NFACTOR, TAIL and the three SPOT auditors

**1. `main` proves the binding to a committed trace twice, and the 2^-33.6 is the weaker proof** (DRAFT-CORE, PRF-540 final row 16 and addendum; ROM, PRF-525's first smell). Read by me at `2b869ed35db5`:
- `Flock.Guarantees.SampledKSLiveV3` (`Security/Proofs/Flock/Bridge/ScopeGate.lean:116-183`), in the lock's `deployed` list: for the live audit with one session and a uniform sample of fixed size, `Pr[accept and not RelS at the plurality trace] <= miss + ksAvgStrictZ + Q_s * (2(t'+v)(1+k)/2^256 + 1/(eM) + k/(e R_w)) / (1 - rho)`. `Q_s` is the expected number of commit strings a sample reads. Linear, no factor N, no square root.
- `Flock.Bridge.deltaPub_le_deployed` (`Security/Proofs/Flock/Bridge/PublicDeployed.lean:150-153`): that error with `Q_s <= W = sum over units of |io u|` is at most `miss + 2^-100` for `W <= 2^40`, budgets up to `2^82` and a link cost `t' <= 2^81` a run.
- The 2^-33.6 is `compilerKSReg`'s (the draft's Theorem B.2: any number of sessions, the five designs, what #8512 and #8527 build on). It is not in `deployed`.

**2. The same finder carried to the generic theorem, found independently by four of you:** KILIAN (K-EXP and K-EXP-RS, PRF-524 final), EXTFORM (WAIT, PRF-528 final), DRAFT-CORE (row 16, option B), REPBREAK (PRF-542, 00:30Z). Keep the plurality. The finder runs once at a tape, then reruns at that tape until each node the first run opened is opened again, and outputs the two openings when they differ. Success at least half the off-plurality mass; expected cost `(1 + w) * c_run`; so under `ExpectedCR` the binding term is `2(1 + w) * c_run / 2^256`, with no levels and no `sqrt`.
- `w = N`: 113.0 / 108.8 / 97.9 bits at one session, k = 1507, T = 2^80 (against the 103.1 / 99.0 / 88.1 of my 00:2xZ entry, which this replaces).
- The three of you count `w` differently: blocks `n_PU` (EXTFORM, through `WitOpens`), the values one session's blocks read `R` or the expected pairs `W'` (KILIAN), `sum_g pi_g` (DRAFT-CORE), `Q_s <= W` (`main`). DRAFT-CORE's addendum, item 3: the generic contract's `opens` clause does not confine what a session reveals, so anything below `N` needs an eleventh clause.
- Simulated by KILIAN and EXTFORM on toy provers. Not in Lean. Restating `KSReg`'s error changes a locked record: Daniel's.

**3. What each of you does with it. Finals as before unless a time is given.**
- **REPBREAK:** your target is this finder. Break (a) "the first later run that opens u opens it to another value with probability at least 1/2", (b) the cost `(1 + w) * c_run`, in particular a prover whose reruns cost more than the first run, or for which `T` does not cap a rerun's calls of both parties, (c) the two-stage game, where the second stage's commitments follow the first sample. On PRF-542, `HOLDS | FAILS AT`, by 02:30Z.
- **REPPROVE:** read `Security/Proofs/Flock/Soundness/LinkFinder.lean` (`succ_ge`, `expect_finderCost_le`, `link_mass_le`) and `Discharge/ZkLink/CompiledLink.lean:627-640` before you prove anything again. Say whether `pr_offReg_le` (`SamplingGeneric/Reg.lean:387-441`) can be proved from them by a bridge, and what the bridge is. On PRF-542 by 02:30Z.
- **SCALE:** the one count that decides which cells reach 100. At the floor, B and A, and at the one-stage run of record: `W = sum_u |io u|` (`Bridge.ioSum`, `Security/Proofs/Flock/Bridge/Public.lean:73-77`; say whether a wire is a bit or a row), the most values one session's blocks read (`R`), the expected (session, value read) pairs a run's sample has (`W'`), and the sessions a run has at the design's sample, not at the partition's size. A Python count over the partitions and the planner's rows; no GPU. On PRF-533 by 02:30Z, with what each number was read from.
- **ATTACK:** the finder's weak point is the contract, not the lemma: a session that reveals openings outside its blocks' boundary, and a refused second stage that reads every block (`Reg.lean:89-94`). Build the prover that uses them, or say why none exists.
- **NFACTOR:** KILIAN's K-CS card is your lever with no Q at all (`#bad^2 <= N * #conflict`): 56.5 / 54.4 / 49.0. Say whether you agree, then finish.
- **TAIL** (after your first comment on PAD): KILIAN disputes your "Theorem 2 with a linear premise keeps a square root": the additive term enters only as `log(q/eps)`. One line on PRF-542 saying who is right.
- **AUDIT-SPOT, AUDIT-HEADLINE, AUDIT-TABLE:** for every binding number you quote, say which theorem it is: `compilerKSReg`'s (generic, 2^-33.6) or `SampledKSLiveV3`'s (deployed, one session). AUDIT-HEADLINE: which of the two does the headline run rest on?
- **Everyone: `T` in compressions** stands as in the entry above. KILIAN, EXTFORM, ROM, TARGET and AUDIT-FLOCK have answered it: nothing to pay if the premise and `T` both count compressions, 5 to 9 bits on a linear term if `T` stays in calls of 4 to 64 KiB rows.

## Wed 11 Nov 01:21Z, coordinator: eight finals in (REPBREAK, REPPROVE, ATTACK, NFACTOR, TAIL, AUDIT-SPOT, AUDIT-HEADLINE, AUDIT-TABLE). The finder stands; the sizes it is quoted at do not. New asks for KILIAN, EXTFORM, DRAFT-CORE, SCALE, AUDIT-CRYPTO, WIDEHASH

Daniel gets the checked answer at 04:00Z (8 PM PST). **Each of you still running: post your final by the time below, and end it with one line headed "For Daniel" that I can quote as it stands.**

**1. The units of the three sizes are wrong for the theorem, and I checked it myself** (AUDIT-SPOT S2/S3/S6, NFACTOR, AUDIT-HEADLINE S18). From `origin/main:benchmarks/spot_figures/cards/workload.glm-5.3.json`, 154 strata, 287 tokens, summed `per_token * 287 * field`:

| field | sum | what the swarm has been calling it |
|---|---|---|
| `macs` | 2^43.72 | "N at the floor" (2^43.73) |
| `ands` | 2^54.15 | (not used) |
| blocks (`per_token` alone) | 2^34.32 | (not used) |
| `in_bits + out_bits` | 2^53.63 | (not used; `W` of `deltaPub_le_deployed`) |
| largest `in_bits + out_bits` of one block | 2^20.05 | NFACTOR's rho |

- `n_PU = 57,344` is the `macs` field of one block type, `DsaMqaLogits_v3{N=14,H=32,D=128}` (`per_token` 0.585). It is not a count of blocks.
- `KSReg`'s `nG` is `s.prog.gates.length` (`SamplingGeneric.lean:553-560`). A multiply-accumulate is not a gate. So B.2's `nG * sqrt(Pr[Bconf])` at the floor's circuit is not 2^-33.6: with `nG = 2^54.15` it is 2^-23.2 (AUDIT-SPOT), and `gates.length` may count more than the ANDs (I did not read which gates the list holds; SCALE, say).
- `W <= 2^40` (`deltaPub_le_deployed`) holds for at most 1,965 tokens at the fused cut (AUDIT-HEADLINE): under seven requests. So the Lean-checked `miss + 2^-100` covers no served size, and my 00:48Z item 1 overstated what it gives.
- Every number in this file at "floor / B / A" used `N = 2^43.73 / 2^47.9 / 2^58.8` and `n_PU = 57,344 / 2^24.2 / 2^35.1`. A term linear in `N` loses 10.4 bits at the floor if `N` is gates; a term in `n_PU` loses 18.5 if `n_PU` is the census's blocks.

**2. The finder itself: seven derivations agree, and none of the attacks broke it.** REPBREAK (PRF-542 00:30Z and final), REPPROVE (final 00:57Z), ATTACK and NFACTOR (finals) add to KILIAN, EXTFORM and DRAFT-CORE. What differs is the count:
- **Node form** (REPPROVE): `N' * (2 c_run / 2^256 + 1/(eM))`, `c_run = T (1 + n_sess * e_s)`. Degrades with sessions: 113.0 / 108.8 / 97.9 bits at one session, 97.2 / 84.6 / 62.8 at `n_sess = n_PU`.
- **Targeted second run** (REPBREAK `B_tgt`, ATTACK per position): rerun only the sessions that open the gate. `4 N T (1 + d e_s) / 2^256`: 112.0 / 107.8 / 96.9 at any session count, 99.0 / 94.8 / 83.9 at the largest k. Needs one clause in `Sessions`: the positions a session reveals are fixed by (s, u). The deployed reveal meets it (`SpotFlock/Reveals.lean:38-42, 64-65`, `mem_reveal` 121).
- **By block** (NFACTOR WAIT-PU, through `WitOpens`, `Opens.lean:290-305`): count `n_PU`, no new clause. **Skip by gate** (WAIT-S): `2 T (1 + e_s)(1 + m (1 + rho)) / 2^256`.
- **NFACTOR's warning, for EXTFORM and KILIAN: skip by gate, never by block.** A finder that stops at the first rerun opening the same block, not the same gate, is refused by NFACTOR's counterexample (its final, section WAIT-S).
- ATTACK: with `X = sum_p Pr[sample reads p]`, the term is `X * (2 E[trial] / 2^256 + 1/(eM))`; the generic contract lets a session reveal outside its block, and then `X = N`.

**3. KILIAN (03:15Z): do not prove the first-hit lemma again.** `Security/Proofs/Flock/Soundness/LinkFinder.lean` on `main` is it (`finderG` 42, `plurality_le` 208, `succ_ge` 229, `expect_finderCost_le` 284, `link_mass_le` 315-336, `outOf_spec` 358). REPPROVE's final on PRF-542 lays out the bridge: eight steps that replace `pr_offReg_le` (`SamplingGeneric/Reg.lean:387-441`) where `ksReg_of_sessionKS` (`Reg.lean:512`) uses it, with the premise `LinkCRReg` per (tape, gate) shaped like `LinkCRZC` (`CompiledLink.lean:618-632`), about 250 to 500 lines, and it needs C5's `runCostAt_exp` (branch `cursor/proofs-bconf-cost-5987`, not on `main`). **Your deliverable is that bridge, or as many of its eight steps as compile, on your branch, with the statement you would put in place of `KSReg`'s `epsilon_reg` written out.** State the count it uses in the theorem's own unit (item 1).

**4. EXTFORM (02:45Z):** which `w`, in the theorem's unit. For each of node, targeted, by block and by gate: the count, whether the generic contract as written gives it or which clause it needs, and whether the deployed protocol meets that clause. Use item 2's forms; say where you disagree with REPBREAK or NFACTOR.

**5. SCALE (02:30Z) and DRAFT-CORE (02:45Z): the sizes, in the theorem's units.** One table, each number with the file and field it was read from:
- For the floor (one request, 287 tokens), and for B and A if they are anything more than scalings: `gates.length` of the program `KSReg` is instantiated at (ANDs and the rest), the committed positions a session can open (rows, and bits), the proof blocks (`n_PU`), `W = ioSum`, the largest `|io u|`, the sessions a run has at the design's sample, the blocks the sample reads, and the expected positions a sample reads (`Q_s`, ATTACK's `X`).
- SCALE: where did `n_PU = 57,344` and B's and A's `N` and `n_PU` come from (the draft, the planner, a card)? If nothing sources them, say so.
- DRAFT-CORE: at those sizes, what does `SampledKSLiveV3` give with `Q_s` as it is defined (not bounded by `W`)? AUDIT-SPOT computes 2^-115.6 to 2^-130.6 for a 27,713-block sample and says no lemma bounds `Q_s` that way. Is that right, and what is the smallest lemma that would?
- Both: restate 112.0 / 107.8 / 96.9 (targeted) and 113.0 / 108.8 / 97.9 (node, one session) with the counts in the theorem's unit.

**6. AUDIT-CRYPTO and WIDEHASH (03:15Z): PAD holds only with four changes** (TAIL, PRF-530 final 00:52Z: 2^-130.8 / 2^-126.6 / 2^-115.7 with positions in bits; 2^-121.8 / 2^-117.6 / 2^-106.7 with T in calls on 64 KiB rows). (1) The order is M, then c_k, then the roots, then the sample. (2) **Every commitment a session opens must be opened in the clear and bind statistically**: the trace, the interiors, the registered values, the advice, the input anchors and the network certificate. Any one left computational drops it to 2^-76 to 2^-82. (3) T in compressions. (4) New Lean: a two-run finder without `runExt`, `regVal` over published strings, Naor's count, the false-statement lemma, the two-stage committed form.
- **AUDIT-CRYPTO:** your hiding verdict with change 2 applied to all six kinds of commitment, not the trace alone: what is the hiding assumption then, what leaks if one pad key covers them all, and what the design rule "everything zero-knowledge" needs said.
- **WIDEHASH:** PAD's cost with change 2: pads on all six, per row and per session, against the 1,024-bit digest's 2.2 to 2.4 times.
- PAD's positions are bits. Restate its three numbers with item 1's counts.

**7. TAIL conceded to KILIAN** (PRF-542): Theorem 2 of ePrint 2024/1434 is linear under a linear premise. No square root comes from the literature's side.

Nothing here is in Lean beyond `LinkFinder` and the two theorems of the 00:48Z entry. Restating `KSReg` changes a locked record and is Daniel's; say what you would restate, do not land it.

## Wed 11 Nov 01:29Z, coordinator: one more for SCALE, on `W`

Two counts of `W = ioSum` disagree by 2^16 a token. The workload card gives 2^53.63 bits of block inputs and outputs for 287 tokens, 2^45.5 a token (AUDIT-SPOT S6; my sum). AUDIT-HEADLINE's "`W <= 2^40` holds for at most 1,965 tokens at the fused cut" is 2^29.1 a token. **SCALE: say which cut and which field each count uses, whether a block's weight inputs are in `io u` (and counted once a block or once a circuit), and which of the two is `ioSum` for the run of record.** In your 02:30Z final.

The draft's differences are on one page for Daniel: https://app.notion.com/p/415399515d9e816d85bef67acb677bcb (DRAFT-CORE's and DRAFT-REST's tables, verbatim, with a note that B.4's sizes are being recounted). DRAFT-CORE: if your 02:45Z final changes a row, say which row and give the new cell.
