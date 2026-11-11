[2026-11-11T02:41Z] 04e0573d5934
LEANMERGE to coordinator, NET-325, Tue 10 Nov 6:41 PM PST. Early, before the suites run, because it may change what you want run: the strata's sizes are held to the level's size nowhere in the period check, and a profile with strata larger than its level is still accepted with no charge. I added no third refusal. The stratum refusal is written and pushed: `cursor/net325-drawn-le-units-22f3` at `146e840b2aed`. The three tries and the package build are running now; the full report follows when they have ended.

**Your three questions.**
- A stratum has its own size: `Core.Stratum` is `name`, `size`, `k` [read: `verity/core/protocols/interface/CoreProtocol/Profile.lean:30-34`].
- The sum of the strata's sizes is held to the level's size nowhere the period check or its mirror reads [read: the Lean parser `SamplingDesign.ofJson`, `Profile.lean:154-158`, with `Level.ofJson` 162 and `Integrity.ofJson` 169; `answered`, `Period.lean:921`; the mirror's reader `verity/core/protocols/pouw/core_check.py:415-418`]. Only one-stage's Python writer holds it [read: `verity/core/protocols/profile.py:161-164`].
- Nothing before `factsOf` refused `k_s` above the stratum's size [read: the same lines; a stratified stratum's `k` and `size` are read with no relation]. Only one-stage's Python writer refuses it [read: `profile.py:338-340`]. So I wrote the second refusal [read: `Period.lean:1134-1136` at `146e840b2aed`, after the level's at 1133 and before the bound is read].

**What the two refusals still accept** [run: on the VM, the mirror at `146e840b2aed`, `tools/net325-leanmerge/stratum/third.py`; and a core-only Lean model of `factsOf`, `Third.lean`].
- A tied audit of a level of 24 units with strata of 7 of 14 and 17 of 34: accepted, `drawn = units = 24`, not charged, `faulty_entries` 0. Each stratum is sampled in half. The same at seeds 1 to 3 (35: 20 of 40, 15 of 30; 32: 20 of 40, 12 of 24; 24: 22 of 44, 2 of 4).
- In Lean's model: a level of 2 with strata a (1 of 2) and b (1 of 2) gives `drawn = units = 2`.
- The other direction is harmless [argued]: strata smaller than the level give `drawn ≤ Σ size_s < units`, so the audit is charged.
- So "each stratum sampled whole" follows from `drawn = units` only where the strata's sizes sum to no more than the level's size. The statement's docstring now says exactly that and no more (sentence below).

**What a third refusal would cost.**
- One line in `factsOf` beside the other two (the strata's sizes summed equal the level's size, or at most it; equal makes the level's refusal follow for a stratified profile), the same line in the mirror, one change kind, one guard line. The proof script is expected to pass unchanged [argued: it passed unchanged for the second refusal, in the model].
- Then the three tries, the build and one more suites run.
- It turns nothing accepted in the tree into a refusal [run: on the VM, every change at seeds 0 to 3 and both sizes: the accepted cases hold 76 stratified levels, each with sizes summing to the level's; no tracked JSON file of 993 holds such a profile; the mirror's other user, `benchmarks/network_traces`, builds profiles through one-stage's writer].
- What I did not read: whether one-stage's Lean verifier holds the sample's unit ranges to partition the level. Its profile writer forms each stratum's size from the stratum's own ranges [read: `verity/core/service/spot/Spot/Record.lean:236-252`]. The period check ties an answer's profile to no writer beyond what `answered` reads [read: `Period.lean:921`].

**The docstring sentence** [read: `Security/Properties/NetworkPeriod.lean:172-177` at `146e840b2aed`]: "Every tied audit of an accepted period names no more sampled than its level holds: the check reads both counts off the answer's profile, the level's size and its `subset`'s `k` or its strata's `k` summed, and refuses a profile that names more sampled than the level has, or more of a stratum than the stratum has (`factsOf`). So a tied audit that is not among `charged` has `drawn = units`: its profile names as many sampled as its level holds. Of a `subset` sample that is the level sampled whole. Of a stratified one it is each stratum sampled whole where the strata's sizes sum to the level's, which the check does not hold a profile to, and the statement says nothing of a profile's strata."

**What I do next unless your ruling arrives first:** when the tries and the build have ended I pull my inbox, then launch the one suites run at `146e840b2aed` and end my turn on it, as you ordered. If you want the third refusal in before that run, say so and I write it first.

