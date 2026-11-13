[2026-11-13T11:46Z] 10f3eb2685d7
DGRAD, Fri 13 Nov 3:46 AM PST (TRN-156, your 3: E's size)

**Of the seven entries, only q4muon-full's E size depends on the data, through `scaledquant`'s xz. The four LoRA entries, grow and ia3 have an E size fixed by the shapes and the dials, and for grow by the step too.** All of this is READ from the code (`codecs/__init__.py`, `codecs/wire.py`, `routes/dgrad.py`, `routes/ia3.py`, `run.py` `_encode`); I ran nothing for it.

- **E's container:** `Plan.encode_parts` and `wire.pack` write a JSON header (the keys, the rule indexes, each array's dtype and shape) followed by the raw arrays. Nothing at that level is compressed.
- **s-t20-1.1x-dgrad-q4r-mlp, -q4r, -q2r, -vote: fixed.** The codec is `lowrank` at `e_bits` 4 (dgrad's default). Each adapter's factors are 4-bit codes packed raw, plus an f16 scale a rank component. The header holds `bits` and the shapes. Every adapter is handed over as its parts whatever its values, so which tensors are in E doesn't depend on the data either.
- **m-t20-1.1x-dgrad-grow: fixed by the shapes, the dials and the step.** It uses the same `lowrank` encoding. The factors' rank at a checkpoint is the `grow` phase that step is in, so E's size changes through the run, but only with the step.
- **s-t20-1.1x-dgrad-q4muon-full: depends on the data.**
  - The codec is `scaledquant`, which is xz (preset 1) of the codebook, the scales and the codes, so its size is the codes' compressibility.
  - Beside it, the gains (`gains = true`) are `identity`, raw at a fixed size each. A gain is in E only where it differs from θ, so which gains E holds is the data's in principle.
- **m-t20-1.1x-ia3: fixed.** The codec is `rowscale`: one fp32 scale a row, packed raw. Every scaled matrix is handed over as its parts whatever its values.

ia3: suites at its head passed (RAN r20261113-111857-ba3c: repository 355, grader 25 with 3 skipped, training_slowdown 654, verity-graph 220). ruff and the ratchet gate find nothing new there (RAN locally). Its body comes when r20261113-111731-3f3e is read.

