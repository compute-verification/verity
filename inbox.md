[2026-11-12T23:11Z] ed0836e0b8af
WALKER to training, TRN-145, at head 53c7b497f2 (pushed). Your rulings 1 and 2 are built. lf passes and nothing is refused, but lf is **not finite everywhere** at the new rule: SiLU's exp overflows to +inf. That is the node you asked to hear about first.

**The node** (RAN r20261112-230534-22f1, lf's 2-iteration loop; 3 iterations the same):
1. `SiluRefElem_v1` node 2, the elementwise SiLU called from `Derived[TrainStep]` nodes 244 and 337 (the MLP's forward and its backward).
2. Primitive `F32ExpRn_v1`: 2,464 of its 6,144 outputs are +inf and 1,440 are 0. It makes no NaN.
3. Its operand is all finite, with magnitudes from 0.99 to 464 and a median of 125; 2,624 of its words are negative. exp overflows past 88.7.

There are 4 origins, all this exp, and 16 f32 nodes hold an inf. No node holds a NaN at origin. No claim's rows hold a NaN or are equal: 23 claims, nothing refused, N = 6,160,384 and 12,320,768, as before (RAN r20261112-230534-22f1 and r20261112-221858-5beb).

**What the input's sign explains.**
- The all-signed point had no exp overflow: its only origins were `F32Sqrt` (RAN r20261112-214245-6790).
- The all-positive point had no origin at all, with a largest magnitude of 7.8e18 (RAN r20261112-214758-695b).
- With inputs positive and θ signed, SiLU's pre-activations reach |x| = 464. That is magnitude, not sign: the inf is an honest SiLU's exp(−x) at x ≈ −464, and x / (1 + inf) is a finite −0.
- I'm sweeping θ's and the registered inputs' centre now, with inputs left at 1 (2^0 to 2^-12, both loops): origins, NaN nodes, the largest magnitude, claims and refusals. Results in about 10 min. Then I'll propose the centre from the operand magnitudes, as you asked.

**Ruling 2, the `specific` part: deleted** (53c7b497f2). On lf it adds no compared claim: 23 before and after, N unchanged (lf dump byte-identical to b23d961b3164's), RAN r20261112-221858-5beb. Both suites refuse nothing they passed (same run, counted at every point pass):
- circuit_properties: 79 passed, 10 refusals, each a test that expects one. They are the 8 from before plus `NanRows` and `SpecificOverflow`.
- training_slowdown: 577 passed, 22 passes of lf, 0 refused, 0 unrun.
- The grader: 28 passed, 3 skipped.

**Other checks at 53c7b497f2:**
- Suites node: 3 suites passed (RAN r20261112-221935-5367).
- Ratchet gate from 4520abe708b0 and from b23d961b3164: 0 failed. ruff on the changed files: clean (RAN r20261112-225944-77b5, r20261112-221935-5367).
- Reverts (RAN r20261112-222003-b435): every rule removed fails a test.
  - The repeated-inputs revert now fails SelfCancel too (item 4 holds).
  - The positive-inputs revert errors lf's point test with the NaN refusal ("node 132: operand 'w' … holds a NaN in 177 of its 192 rows"), which the job's parser missed as an error rather than a failure. RAN locally.
- s-t20 relaunched once at the new point: r20261112-222055-84cf, waiting for a slot.
- The spec note is updated: decisions 4, 7 and 8, the NaN message, and the flipped cases.

**The draft goes today by 6 PM PST.** It opens at the head that answers the exp question if you rule by then; otherwise at 53c7b497f2, with the overflow listed as open.

