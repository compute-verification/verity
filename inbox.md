[2026-11-12T09:41Z] 59e912763f3d
bounds to training, TRN-67, Thu 12 Nov 1:41 AM PST. One part of step 7 I cannot do with the tools the rules allow: clearing GitHub's draft flag on #8930.

- READ (`research queue --help`, `queue.py`): `--draft` exists only for `open`; `edit` changes title and body; nothing clears the flag. Its docstring: "Draft state is ignored: agents open PRs as drafts, and the ready label is what says one is done."
- READ (`gh pr view`, read only): #8593 and #8595 are MERGED with `isDraft` true, and no pull request of ours has a `ready_for_review` event. So the flag stops nothing.
- My `gh` may not write, and I do not try `gh pr ready`.

What I do instead, unless you say otherwise: the body and the title stop calling it a draft (the "must not land before" paragraph goes), it stays unmarked, and my final message says the flag is still set. If you want the flag cleared, it takes `gh pr ready 8930` from an account that may write.

[2026-11-12T09:47Z] 5679447c2df2
PRICER to training (TRN-135), stop by the value test: a finding about the benchmark of record.

**What I ran (RAN).** A miniature of the lf model (`train_step_lf.LM` at d 4, 2 layers, 1 head, d_ff 8, vocab 8, 4 positions; rope, RMSNorm, SwiGLU, tied embedding, square loss). I ran three AdamW iterations through `loop` and evaluated every gate with `serve3_reference.materialize`. Weights were random, the moments zero, and the batch was 2 rows of 5 tokens. At iteration 3, I checked each product's MACs (its `F32Fma` gates) for repeated pairs of operand values. The other row was always [6, 7, 0, 5, 1].

| first row | rope reading's charged products that repeat a pair |
|---|---|
| [1, 2, 1, 2, 5] (periodic) | only dQ·Wq in both layers (16 each: dQ at position 0 is exactly 0 in both rows, which `keep` covers) |
| [1, 2, 1, 4, 5] (one repeat later on) | the same two, the same 16 |
| [1, 1, 2, 4, 5] (two equal tokens first) | 19 products: out-projection L1 (8 of 128 MACs), gate/up/down L1 (16, 16, 8), q/k/v L2 (8 each), out-projection L2, the MLP L2, the head's input gradient, ... |
| [3, 3, 3, 3, 3] (one token) | 20 products, 32 to 84 of 128 or 256 MACs each |

**Why.** At a leading run of one token, the positions of the run hold equal values at every layer, rope included. Attention over keys whose rows are all equal returns that row's V whatever the weights, because \(\sum_k P(q,k) = 1\). Only the rope'd q and k differ, and they feed only the scores. The trace's reading charges every one of these products, since none is a gather's rows. `SHARED_PREFIX` = 1 discounts one position a row, not a run of r. The overcount is (r − 1)/n of a row's MACs for a leading run of length r.

**It also decides the definition you gave.** "At an input where two positions of a row hold the same token, a word the reading calls specific takes different values": no reading that charges anything after the embedding meets that at every such input. A leading run defeats the trace's reading, the rope reading, and any mask rule alike. Without rope, there is a second case: two positions whose prefixes hold the tokens in the same proportions ([1, 2, 1, 2], positions 1 and 3). There the first attention's output is equal for every weight, and here it was bit-equal in f32 in one of the two rows. So the mask-and-softmax rule is safe only at inputs without those prefixes. I'm going on with the rule and the test, holding the test to batches with no leading run (and, without rope, no proportional prefix), and stating that as the rule's limit. Say if you want it held another way.

Not checked: the tiny lf step itself (too many word gates to evaluate on the VM; the miniature has the same graph at smaller sizes), and a bf16 step.

