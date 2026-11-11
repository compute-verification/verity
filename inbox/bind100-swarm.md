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
