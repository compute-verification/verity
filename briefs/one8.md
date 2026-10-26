# Brief: ONE8, the integrity leg (ONE-8), Linear PRF-205

**Due:** statement on PRF-205 by Mon 26 Oct 22:00Z (post it the moment it's fixed); proved against its inputs as hypotheses by Tue 27 Oct 18:00Z; marked with the inputs discharged by **Wed 28 Oct 06:00Z**. Dates are ceilings.

Read: the plan `~/.research/notes/lanes/proofs/20261026T0332Z-report-onethm.md` (§(a) the target statement, ONE-8), ONE78's note `~/.research/notes/lanes/proofs/20261026T1707Z-report-one78.md` (§2: why ONE-8 is false at #4613's names), and #4613 (`cursor/proofs-onethm1-8b2b`: `Security/Properties/Verity/Soundness.lean`, `compiled`, `ComponentAccepts`, `soundness_of_legs`).

**Decision 1 (Daniel approved, 11:22 AM PDT) fixes the form:** a commit-then-coin period game (params and the 64-byte commitment fixed before any coin; only the evidence depends on the coins); `∃ X` fixed by the commitment, on the registered circuit `circuitOf` (ONE-7); the integrity component is v2 (`OneStageZk`); each component's coins come from the device's log (`answerAt`) with an LMS-forgery `Break` in the event; δ a function of public parameters.

**Deliverable:** the integrity leg's soundness at the code in that form: Pr over the device's coins [the integrity component accepts ∧ more than K₀ units of `circuitOf`'s partition wrong at `X c` ∧ ¬Break] ≤ `deltaPub …`. Compose it from: `SampledKSLiveV4` (ONE5V, PRF-204), v2 accept → `AcceptsZK` (ONE3P, PRF-206), `answerAt`/`Break` (ONE4, PRF-200), `circuitOf` + `card_wrong_lt_of_relS` (ONE78, PRF-195) and `SampledKSLivePub` (ONE19, #4758). Write it **now** with each as a hypothesis or named placeholder `Prop`, then swap each in as it lands. Coordinate with ONETHM (bc-12486cf4, PRF-193), who writes ONE-20 against your statement: post it on PRF-205 first. If the composition is false as stated, stop and say why.

**Paths:** `Security/Proofs/Properties/Verity/` and `Security/Properties/Verity/` (soundness is taking these from lean; a new file there is fine), and new files in `Security/Proofs/Flock/Bridge/` (proofs'; tell soundness if you edit an existing one). **Base:** create your branch from `origin/cursor/proofs-onethm1-8b2b` (#4613). **Topic:** `one8`.
