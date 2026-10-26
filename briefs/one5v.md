# Brief: ONE5V, SampledKSLiveV4 and the typed/Built lemmas (ONE-5v), Linear PRF-204

**Due:** statement on PRF-204 by Mon 26 Oct 22:00Z (post it the moment it's fixed); marked by **Tue 27 Oct 18:00Z**. Dates are ceilings.

The target is `Verity.Soundness` (plan: `~/.research/notes/lanes/proofs/20261026T0332Z-report-onethm.md`, ONE-5; read it). `SampledKSLiveV3` (#4633, `cursor/proofs-ksall-b2fc`, `Refused.lean`, stacked on #4595) still has six file preconditions as hypotheses (VREFUSE's note `~/.research/notes/lanes/proofs/20261026T0316Z-report-vrefuse.md`, 17:27Z, and PRF-177): `typed = false` (refused; needs a lemma), `Built D.unroll.unit` (a lemma from the parse), `pub.tables = none`, `scopeOkH`, `kd % nTab I = 0`, `TagsOkZHJ`. ONE5 (bc-4a6088e5, branch `cursor/soundness-one5-*`, PRF-177) writes the verifier refusals for the last four and makes the files check the default `--zk` path (it's opt-in `--ks-form` today); read its note `~/.research/notes/lanes/soundness/*one5*` and its branch, and state against its refusal predicates' names (named `Prop` placeholders until it posts them).

**Deliverable:** (1) the two lemmas; (2) `SampledKSLiveV4`: V3 with none of the six as hypotheses, its event the default `--zk` verdict (and v2's, `OneStageZk`, which runs the same library path), concluding each precondition from acceptance; same named assumptions as V3, no new one. Keep the lock's reads narrow. ONE8 (PRF-205) composes V4: post the statement on PRF-204 early.

**Paths:** `Security/Proofs/Flock/` is proofs' (proofs OK'd ONE-5 work there, 17:40Z; proofs reviews). Coordinate with ONE5 through notes; don't edit `verity/core/service/flock/` (ONE5's). **Base:** create your branch from `origin/cursor/proofs-ksall-b2fc`. **Topic:** `one5v`.
