# hardness_ml

The effective-compute penalty of training a language model whose large linears are hardened
(`inbox/HARDNESS.md` Q3 on the compute-research inbox branch): a small decoder-only transformer on TinyStories V2 with a
4096-entry BPE, trained on CPU.

- `env.sh`: the CPU torch venv and the token cache under `$HML_BASE`, built once.
- `prep.py`: downloads TinyStories V2 (GPT-4 split), trains the tokenizer, writes `train.bin` and `val.bin`.
- `train.py`: the model and the designs (`dense`, `A`, `B`, `lowrank`, `frozen`, with int8 operands and live noise);
  its docstring states each.
- `sweep.py`: a preset of configs in parallel processes, results under `$OUT/runs/`.
- `analyze.py`: loss gaps and the effective-compute penalty against the baseline runs.

~~~bash
research run --on vy-nebius-cpu-6 --project verity --campaign hardness --source . --cwd source -- bash -c \
  '. benchmarks/hardness_ml/env.sh && python benchmarks/hardness_ml/prep.py --data $HML_DATA && \
   python benchmarks/hardness_ml/sweep.py --preset bench --data $HML_DATA --out $OUT'
~~~
