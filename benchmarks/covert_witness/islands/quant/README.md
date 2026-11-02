# Direction `quant`: how far an update compresses without losing the loss it bought

A search direction of the training-efficiency search, not a class: it runs in `reference`, `e1mb` and `e100kb` (on this
branch `islands/<class>/train.py` links to `islands/quant/train.py`, which takes `--klass`). The theorem's β is read from
the witnessed encoding E, so for a fixed bpb gain the direction pushes |E| down: train an update, then encode it in low
bits (k-bit codes with per-group scales, sign and ternary codes, codebooks), or train it in that parameterization, and
entropy code E.

- `qformat.py`: CWQUAN1, the low-bit encoding (`delta`, `lora` and `full` entries, f16 codebooks and scales, xz), with its
  decoder and quantizers. No fixed decoder reads it yet: until the toolsmith adds it to `prepare.decode`, a candidate
  submits each variant's model through `encode_full` and records its CWQUAN1 size in the notes, checked to rebuild that
  model bit for bit.
- `train.py`: the candidate (`--mode ptq`: the reference recipe, then each `--variants` quantizer on Δ; the first variant
  is the submission, the rest snapshots at the full step log, so the evaluator scores every one).
- `test_qformat.py`: CPU round trips (`python islands/quant/test_qformat.py` from `benchmarks/covert_witness/`).

Run: `python benchmarks/covert_witness/launch.py reference s100 --note "MECHANISM" -- --klass reference --budget-fraction 0.3
--muon-lr 0.01 --variants fp,rtn4g128,rtn2g128,sign`.
