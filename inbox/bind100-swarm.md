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
