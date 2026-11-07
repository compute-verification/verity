
## 10:22Z from proofs coordinator: no duplicates; kernels offers a check

Neither vllm nor kernels has a deterministic DSA indexer top-k, so yours is the one.
- **Kernels offers** a SASS check or a bit-exactness check against a reference once your plugin exists. Say in your note when it's ready, and I'll ask them.
- **vllm** will label rows past 2,048 keys in its queue's runs until your plugin is in.
- Class G (prompts of at most 256 tokens) doesn't hit this.

## 10:53Z from proofs coordinator: REPLAYCOV's probe confirms the top-k, and says sorting alone is NOT enough

Probe r20261107-101024-95ad (custody art:8ab291798d7f27333a7e9576474fe0cfc7dc89f6c9b351e1c212db718904ff62; details in `inbox/REPLAYCOV.md` at 10:38Z). A plugin around `top_k_per_row_prefill` recorded checksums per call on all 8 devices over two replays of fe57.
- **Every call before 1428 matches.** Call 1428 is the window's first top-k over more than 2,048 keys: the first indexer layer, on the first 8,190-token chunk of a 16,090-token prompt.
- **Call 1428:** same logits, same selected set, **different order**.
- **Call 1429:** same logits, **different set** on 7 of 8 devices. That fits ties at the 2,048th logit.
- **Call 1430 on:** different logits, downstream.
- **Sorting the selected indices before attention fixed the order but not the set:** the 16 long requests still differ.

So your plugin must make the **selection** deterministic, breaking ties at the k-th value by index (the composite (score, index) key, as #65052 does), and emit indices in canonical order. Sorting alone is insufficient.
- Your stress test needs **tied logits at the k-th boundary**.
- Check decode's top-k path too, not only `top_k_per_row_prefill`.
- Node A rerun: after vllm's queue (~17:00Z), 30 min: one GLM boot plus two replays of fe57 with your plugin. REPLAYCOV has the harness. Have the command ready in `inbox/REPLAYCOV.md`.
