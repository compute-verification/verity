# hardness_ml

The effective-compute penalty of training a language model whose large linears are hardened
(`inbox/HARDNESS.md` Q3 on the compute-research inbox branch): a small decoder-only transformer on TinyStories V2 with a
4096-entry BPE, trained on CPU.

- `env.sh`: the CPU torch venv and the token cache under `$HML_BASE`, built once.
- `prep.py`: downloads TinyStories V2 (GPT-4 split), trains the tokenizer, writes `train.bin` and `val.bin`.
- `train.py`: the model and the designs (`dense`, `A`, `B`, `lowrank`, `frozen`, `lsb`, with int8 operands and live
  noise); its docstring states each. `ablate` adds eval-time ablations (permute or drop the public random part, codes the
  noise leaves unchanged), and `init` / `save` start from and write a saved model.
- `sweep.py`: a preset of configs in parallel processes, results under `$OUT/runs/`. The `redml*` presets are Q4's
  (A-LSB, A at the 1% rank, ablations, and `redml3a` then `redml3b`: a float model converted by PTQ and QAT).
- `analyze.py`: loss gaps and the effective-compute penalty against the baseline runs.
- `redml_scale.py`, `redml_numerics.py`: numpy only, seconds. Unhardenable, KV, coin and rank-cap shares at real model
  shapes; B's ball, adapter cancellation and committed-precision routes.

~~~bash
research run --on vy-nebius-cpu-6 --project verity --source . --cwd source -- bash -c \
  '. benchmarks/hardness_ml/env.sh && python benchmarks/hardness_ml/prep.py --data $HML_DATA && \
   python benchmarks/hardness_ml/sweep.py --preset bench --data $HML_DATA --out $OUT'
~~~
