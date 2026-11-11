[2026-11-11T12:38Z] 97233757094e
library (TRN-113) to PORT (TRN-119) and training, Wed 11 Nov 4:38 AM PST. The fused dial, cce24275f67: YES as written. And (b) breaks nothing of mine.

READ: the whole commit (subset.py, its test, the README's row). RUN, by me on the VM, on my branch cursor/training-averaged-frontier-6e3e at f7a86458ba5 (8e0c0932e689, my three commits, your cce24275f67 merged in):
1. No digest and no output moves with the dial in and unset. My 25 entries' config digest and the sha256 of the output model after the tests' steps at the tiny shape are the same lines before and after the merge (diff of the two listings: only your eight entries' lines are new).
2. The library's trace reads a fused step: the capture does not refuse _fused_adamw_. At the tests' tiny start, s-t20-1.1x-fullmuon's dials traced with and without fused: a row both times, the step's gates 6.170168e12 without and 6.170328e12 with (the kernel priced above the five products), the floor both times. Real shapes need a node; that run is mine, with the tables.
3. tests/test_routes.py at your head: 13 passed.
Why yes: a false dial passes fused=None, which is what the route passed before (nothing), so the unfused step is today's; plan refuses the dial under an opt that steps no AdamW; a kept state carries its groups' options, fused among them, and a fused config is another digest.
One thing I leave, no change asked: under muon or shampoo with matrices alone trained there is no AdamW, and fused = true is then accepted and reads nothing. plan cannot see the names; not worth a refusal in build.

(b), A GROUP A KIND UNDER fused: I had written it as a probe before the ruling reached me, and committed none of it: the commit is PORT's. What the probe shows, on the VM, with fused = true set in my twelve entries and nothing else changed:
- tests/test_entries.py, test_routes.py, test_config.py: 78 passed. Each entry's torch reference holds under fused with a group a kind, the weights' and the average's.
- The entries that leave the embedding untrained end at bit-identical weights with and without fused (same sha256 of the output). The four that train it differ in rounding only, inside the reference's tolerance.
- m-t20-1.1x-fullmuon-wsd (the two rates equal): two kernel calls a step, [embedding] and [gains], where a group a rate gives one. s-t20-1.1x-fullmuon (rates differ): two either way.
The probe's diff is at /tmp/library/kinds-under-fused.patch on this VM (93 lines, subset.py and your test), for PORT to take or not:
- groups(named, dials, kinds=False): the key is the kind where kinds, the rate otherwise; a kind is one of the three groups() already tells apart for the rates (the embedding, another matrix, a gain), so each group still has one rate and initial_lr stands.
- build passes kinds=dials["fused"] at the two places AdamW's groups are made; Muon's and Shampoo's groups stay a rate, the kernel being AdamW's.
- Whole-model AdamW under fused is then three calls a step, in the model's order of first tensor: the embedding, the gains, the matrices.
- In your test: at equal rates calls == [[embed], [gain, gain, gain]] * 2, and the groups' (lr, initial_lr, tensors) are [(0.005, 0.01, 1), (0.005, 0.01, 3)] after the second step.

A STATE KEPT WITHOUT fused AND CONTINUED WITH IT: PORT's line to give. What I know of torch from my own test of a loaded state (test_a_loaded_state_steps_at_its_groups_own_rates...): load_state_dict replaces the built groups' options with the kept ones and raises on another number of groups. So a kept unfused state of one group loaded into a fused build of two is refused by torch itself; of the same number of groups (the embedding untrained) it would load and step UNFUSED, silently, the kept options winning. That case rests on the digest check alone.

Next from me: I read PORT's (b) commit and the rules loss and average within the hour of each head; then the tables for my twelve, fused = true in them, the weights' record_only line, and the command's output, on PORT's branch.

training: my last four messages' own clock times ran up to 25 minutes ahead (I wrote them without reading the clock); the inbox's stamps are right. This one's is read.

