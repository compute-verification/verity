# hardness_ml

The effective-compute penalty of training a language model whose large linears are hardened
(`inbox/HARDNESS.md` Q3 on the compute-research inbox branch): a small decoder-only transformer with a 4096-entry BPE,
trained on CPU, on TinyStories V2 and, to tell that corpus's low loss floor from the constraint, on FineWeb-Edu.

- `env.sh`: the CPU torch venv and the token caches under `$HML_BASE` (`$HML_DATA` TinyStories, `$HML_FW` FineWeb-Edu),
  built once.
- `prep.py`: downloads the corpus (`--source tinystories`, TinyStories V2's GPT-4 split; `--source fineweb-edu`, one
  shard of FineWeb-Edu's sample-10BT), trains the tokenizer, writes `train.bin` and `val.bin`.
- `train.py`: the model and the designs (`dense`, `A`, `B`, `lowrank`, `frozen`, with int8 operands and live noise);
  its docstring states each. Each result records its corpus.
- `sweep.py`: a preset of configs in parallel processes, results under `$OUT/runs/`. `fw1` and `fw2` are the FineWeb-Edu
  presets; the rest ran on TinyStories.
- `analyze.py`: loss gaps and the effective-compute penalty against the baseline runs, one corpus at a time.

~~~bash
research run --on vy-nebius-cpu-6 --project verity --campaign hardness --source . --cwd source -- bash -c \
  '. benchmarks/hardness_ml/env.sh && python benchmarks/hardness_ml/prep.py --data $HML_DATA && \
   python benchmarks/hardness_ml/sweep.py --preset bench --data $HML_DATA --out $OUT'
research run --on vy-nebius-cpu-6 --project verity --campaign hardness --source . --cwd source -- bash -c \
  '. benchmarks/hardness_ml/env.sh && python benchmarks/hardness_ml/prep.py --source fineweb-edu --data $HML_FW \
   --max-train-bytes 2000000000 && python benchmarks/hardness_ml/sweep.py --preset fw1 --data $HML_FW --out $OUT --workers 16'
~~~
