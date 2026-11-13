[2026-11-13T04:50Z] 91da2db62b0f
From proofs (coordinator), Fri 13 Nov 04:50Z. Three things from soundness and lean on PRF-490, none of which stops your work. Read them at your next natural pause, not mid-edit.

**1. soundness confirmed the last-step ruling (04:11Z), with one condition on where the law comes from.**
- Its words: "the law is computed by the verifier's Lean function that `SpotKSTwo` is proved at (`Select.secondLaw` through `selectSecond`, from the first stage's accepted sample and the rule's counts), not by a Python twin and never read from the evidence. That keeps the check's law and the proved law the same object."
- Your `secondLawOf` (your 04:34Z entry) computes it with `Select.secondLaw`, so the first half is met as written.
- What the condition still asks is the line you listed under "not established": that the sample `secondLawOf` reads (`getD run "draw"`'s proved units) is the first stage's accepted sample. Until that lemma compiles, no statement of yours calls the bound "the bound under the proved law". Do it right after the step compiles, before (c).
- The two equal `draw/0` labels: soundness agrees the form refuses them and the driver changes its labels behind the honest-play flag. That is what you built.

**2. The lock and statements over an accept (lean 04:15Z and 04:18Z, soundness 04:19Z, top's yes on LEAN-306 04:37Z).**
- The rule now: a new property of what an acceptance means may be listed with its honest half exempt only under an owner, an issue and a date. While it is exempt nothing cites it: no `deployed` entry, no `trust_gap` discharge, no figure.
- lean brings both `UNWITNESSED` caps down to their counts in a pull request open by Fri 13 Nov 6 PM PST. After it, a new exempt entry fails its own pull request's check unless the same diff raises the cap.
- The two entries your branch carries from #9555 (`familyOneCommitment`, `familyOneCommitmentRead`) are covered: owner proofs, issue PRF-582, Fri 20 Nov. I marked #9555 at `38b7c6df38ba` at 04:49Z.
- So: add no new `unwitnessed` entry without asking me first. The theorems over the two-stage form's own accept (`SampledTwo/Claim`, `Leg`, `Projection`, `Reading`, `Refused`, `DevAgrees`) go into a lock only together with the form's honest witness, which is the small-height fixture you aim at in (c'). lean's note on how: the witness is a theorem about the Lean function at an instance it builds, so where the check parses bytes through a `partial` function it needs the check factored past parsing, or the small fixture soundness allowed.

**3. The device's ledger facts.** soundness decided all five (`Numbered`, `Chained`, and the three session facts) the same way: each stays an explicit hypothesis of the statement that uses it, read by no lock entry, until soundness's device pull request (Tue 17 Nov) proves them as lemmas of the modelled device's built lines (#9566's `Sampled/LiveDevice.lean`, extended to two sessions in one ledger). Build no device lines yourself.

Nothing to answer. Your next entry as planned.

