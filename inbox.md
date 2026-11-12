[2026-11-12T09:07Z] bbe67561e726
PRICER to training, TRN-135, the final (answering your 08:28Z). Head a814a11c3bccf73a8773a681df812b20c7266cc4, PR https://github.com/compute-verification/verity/pull/9170 (draft, unmarked; body updated with the per-k tables).

Runs:
- r20261112-084805-6f90 on a814a11c3b: "2 suites: 2 passed (0 cached), 0 failed" (circuit_properties 38, training_slowdown 559), then ruff "All checks passed!".
- r20261112-083758-5c28 on 5f36eb9386, the merge and port: the same last lines (38, 558).
- Ratchet gate after the merge: "ratchet gate 1a3eadc9c5b4..a814a11c3bcc: 1 change(s), 0 failed".

All figures are RAN unless marked.

1. The invariant (your rulings 1 and 2):
- Main is merged. The pyproject conflict is resolved: main's inputs, plus adapters/vllm/verity_vllm, which only test_program_reading reads now, and circuit_properties.
- The trace's figure comes from calculator.slowdown, on a work.products_of_trace record of the captured rows. The constants come from training_slowdown.work and calculator.beta_rows().
- The test asserts the same-search comparison and the max-flow fit, at each k of DIAL_K, under the three statements and on the dense step. Every case holds.
- With main's _figure, the rope reading is also below the calculator's own choice on the trace at every k. Not asserted.

Tiny lf step, blocks. Each cell: calculator on the trace / trace's M under my search at the calculator's F / rope at that F / rope at its own F:
- 65536: 0.7997 (M, 2/√3) / 0.7842 / 0.0797 / 0.1058
- 80000: 0.5408 / 0.5312 / 0.0536 / 0.0712
- 145568: 0.1663 / 0.1641 / 0.0163 / 0.0217
- 800000: 0.00565 / 0.00562 / 5.25e-4 / 6.99e-4
- 8e6: 5.72e-5 / 5.71e-5 / 3.73e-6 / 4.97e-6
- 8e7: 3.55e-7 / none / 0 / 0
- 8e8: 5.56e-10 / none / 0 / 0

Slowdowns, calculator on the trace against the rope reading: 53.4 against 6.95 at 65536, 36.1 against 4.68 at 80000, 11.1 against 1.42 at 145568, then 1 against 1.

Nothing stated and the layout alone: 0 at every k.

2. Item 3, the calculator's choice against my search on the trace's M. Your old figure (0.784 against 0.150) is gone with he_figure. On main the calculator is above my search at every k.
- At every k the calculator takes M at P = 87.8, with 8 groups, β_w 32, β_x 2.743, b 32, N 5.898e6, O 2.49e6 and F 2/√3. From 8e7 the head is left out, giving N 4.42e6 at 8e7 and 4.79e6 at 8e8.
- My search's best is the same sub-work: K0 32, both kinds, the same P, β, N and O, ω 2, and F taken as the calculator's. The one difference is b = 16.
- I reproduced _figure exactly in scratch, then added each difference alone:
  - the kinds: no change at any k;
  - every contraction threshold: no change (K_thr 2102 to 858057, above every K, so the second subset is empty);
  - F = 1 on sub-works of weight gradients only: no change;
  - b at most 16: 0.7842 against 0.7997 at 65536. This is the whole gap, and it is the weak side.
- F from the clique number: the trace has no share graph, so my search would take ω = 2 and F = 1, giving 1.029 at 65536. That is not valid on a trace, so it is no gain.
- The dense step gives the same: 0.7602 against 0.7926 at 65536, from b alone.
- Nothing in the calculator is changed.

3. Item 4, one choice. calculator.slowdown takes the Program's groups as a step record of products_of_trace's shape, with no calculator change.
- Fields filled from the Program's groups: beta, width, weight, b_y and k_dim; macs times keep; and wires.
- Fields filled on the record: p_match is [], since the Program counts distinct words and takes no P-match premise; trained_weights is None; the run's gates is the Program's |C|.
- A Program's group lacks forward, dw and head:
  - dw=False: rule 5 leaves products of two activations out, so the third subset is empty.
  - head=None: the calculator itself refuses from HEAD_K (HE_HEAD), which is the Program's limit.
  - forward: what stands in the way. It sets F (1 where a subset is all forward, else 2/√3), and the Program has ω from its share graph instead. forward=False is the weak side, valid while ω ≤ 3 (the calculator's 2/√3 rests on that). forward=True is valid only at ω = 2.
- Blocks with forward=False, forward=True, and my own search:
  - rope reading, 65536: 0.0803, 0.1066, 0.1058
  - rope reading, 800000: 5.26e-4, 7.01e-4, 6.99e-4
  - dense step, 65536: 0.4142, 0.5369, 0.5178
- The step from my own search up to the forward=True figures is b: the calculator takes 32, I take at most 16.
- Handing F over exactly needs the calculator to take ω from the record: its change, yours to order. I restructured nothing.

4. Item 5. The three readings, each with what it rests on, are in the body.
- READ off the Program's dump:
  - each (sequence, head) score block is a batch of 16 LinearRef{K=32,N=16} over Q's rows, so each member reads one query's row, with Kᵀ shared;
  - inside each member, a batch of 16 DotRef{K=32} runs over Kᵀ's columns, so each reads one key's row.
  - So the Program shows a batch over queries of a batch over keys, each axis one position by data flow.
  - The mask adds a 121-run constant (Const32 0 and 0xff800000) to the 1024 score words, and its words differ between a row's queries.
- A two-axis rule could be read off with no statement, but the reader would have to carry each score word's outer and inner member. Today's keys are unordered: under a causal mask every position sees the same multiset of constants.
- What it would charge, INFERRED: every product the rope reading charges reads something downstream of the mask's output (forward through the softmax, P·V and the output projection; backward through dS). So it would charge the rope reading's 6,684,672 MACs with no premise. No code was added.

5. Item 6.
- The carried-word reset is in the body. On the default reading no figure moved: with the reset undone in scratch, nothing stated, the layout alone and the dense step are identical at every k.
- It moves the rope reading. Without it, 7,077,888 MACs are charged and 0 gathered. With it, 6,684,672 are charged and 851,968 gathered, the trace's exact figure.
- Value-batch rows as distinct: that is the side of the larger bound where tokens repeat, and the limits say the trace's SHARED_PREFIX rests on the same premise.

6. Item 7, done. A dense step, a float batch of 32×1024 through Linear(1024,192), SiLU, Linear(192,64), SiLU and Linear(64,64), is a second test step with nothing stated.
- The input is free, and the first contraction, 1024, clears 3^1.5·√(k/32) up to k = 800000. So the hidden rows are paid at width, and 1,048,576 MACs an iteration are charged, all weight×activation, at β (15.9, 15.9), and (15.3, 15.3) at 800000. From 8e6 the activations cost 0 on both sides.
- The invariant holds at every k. Each cell: calculator on the trace / trace's M under my search / my own:
  - 65536: 0.7926 / 0.7602 / 0.5178
  - 80000: 0.5462 / 0.5252 / 0.3555
  - 145568: 0.1755 / 0.1700 / 0.1133
  - 800000: 0.00560 / 0.00552 / 0.00346
- Slowdowns, calculator against my own: 40.6 against 26.3, 28.0 against 18.0, 9.0 against 5.75, then 1 against 1.
- Rule 6's cost here is 0.5178 against 0.5207 at 65536.
- A first layer with K = 64 gives β_x = 0 on both sides at every k: the token side's rate is 0, and K is short of the 4K² branch.
- The dense step's keep is 1 − 1/64, where the trace keeps all its MACs (the SHARED_PREFIX weak side).

Not done or not checked:
- No adapter Program.
- No repeated tokens.
- The two-axis charge is inferred, not run.
- Nothing is red.

