# Direction `quant`: how far an update compresses without losing the loss it bought

A search direction of the training-efficiency search. Harness v3 has no classes: `--max-e-bytes` caps a sparse code (on a
v2.x pin `--klass` names the class, whose cap applies). The theorem's β is read from
the witnessed encoding E, so for a fixed bpb gain the direction pushes |E| down: train an update, then encode it in low
bits (k-bit codes with per-group scales, sign and ternary codes, codebooks), or train it in that parameterization, and
entropy code E.

- `qformat.py`: CWQUAN1, the low-bit encoding (`delta`, `lora` and `full` entries, f16 codebooks and scales, xz), with its
  decoder and quantizers. te_core 3.2's `prepare.encode_quant` and `prepare.decode` write and read the same bytes; with
  `--emit quan` a candidate's E is CWQUAN1 (train.py refuses it on a pin whose prepare lacks them, or writes or reads
  different bytes). With `--emit full` (the default) E is the model through `encode_full`, and the notes record its
  CWQUAN1 size, checked to rebuild that model bit for bit. `qformat.recompress` re-encodes a CWQUAN1 at xz preset 9e,
  or at the smallest of `qformat.XZ_TRIES` (LZMA2 match-finder settings, 2-6% under 9e on sparse codes): the decoders
  read any one xz stream, and sparse codes take ~1.1-1.2x their order-0 entropy against preset 1's ~1.5x.
- `train.py`: the candidate (`--mode ptq`: the reference recipe, then each `--variants` quantizer on Δ; `--mode qat`: the
  update trained as a sign, ternary, sparse ternary (`--qat sparse --qat-keep SHARE`) or k-bit code, `--xz 9e|min` for
  the submitted E; `--mode qlora`: LoRA with k-bit factors; its docstring has the rest). The notes' `delta_grad` has
  islands/lowrank's keys: `bits` is the fp32 gradient plus optimizer state (Muon's buffer, AdamW's m and v) and
  `latent_bits`, apart from it, the fp32 latents a straight-through code trains (the update application's Δ_t).
- `test_qformat.py`: CPU round trips (`python islands/quant/test_qformat.py` from `benchmarks/covert_witness/`).

Run: `python benchmarks/covert_witness/launch.py s100 --island quant --note "MECHANISM ..." -- --mode qat --qat sparse
--qat-keep 0.08 --emit quan --xz min --budget-fraction 0.1`.
