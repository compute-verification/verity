
## Not run

- **Nothing was served.** The capture is the kernel alone, called by the pin's method on a stand-in for the backend, on
  random and planted rows. No request went through vLLM's engine, and no checkpoint was read.
- **Which T a served step has** is read at the pin's source, not run: the decode and prefill routing, and that under a full
  CUDA graph the attention's token count is the graph's padded size (`gpu_model_runner.py:2490-2505`, `:4466-4492`).
- **The whole token at served size** was compared by circuit-check on edge vectors only, and its partition check did not
  run on the whole; its units were each checked on random vectors.
- **The SASS comparison with the registered instance** is a census of floating-point opcodes, not a line-by-line diff.
- **The index conversion** from a request's positions to cache slots is not modelled, and the 56 pad heads have no
  Definition (the capture holds their outputs to the reference on a zero query).
- **No Boolean `_v2` forms** of the new Definitions.
- **Not wired, by the brief:** the capture's record is not in the property registry, its test is not in
  `suite_parts.SECONDS`, and `serving_templates` does not stage the cut. Those files are the adapter's.
- **The write and the dense read-back** are other issues' Definitions; this PR states the row's layout from the pin's
  source and reads neither.
