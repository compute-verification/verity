# randomized_arith

Does randomizing the arithmetic with verifier coins stop the honest int8 budget from being a free exact substrate for a
stacking cheater (workstream S3 of approach:stacking-properties/program)? This directory measures both sides.

- `honest.py`, run by `job.sh` on one GPU: the honest accuracy cost of each randomization (stochastic rounding, a
  dither offset, ±1 LSB noise, noise on every int32 accumulator, a fresh Hadamard rotation, a signed permutation) on a
  W8A8 model with exact int8 dots, against round-to-nearest-even. Perplexity on the ppl bench's pinned windows, MMLU
  5-shot on a fixed subset and GSM8K 8-shot.
- `cheater.py`: the cheater's cheapest exact squaring mod p1024 under each randomization, built in the property red
  team's port of Model.lean (`benchmarks/property_redteam/squaring.py`) with randomized gates, run on random inputs and
  fresh coins, its units and MACs counted.
- `verdict.py`: f at the census's operating points (Llama-3.1-8B and 70B, 10 min and 1 h) under each mechanism, from
  `cheater.py`'s JSON, and the honest side's tables from `honest.py`'s results.

~~~text
research run --on vy-dc-h200 --project verity --campaign stacking-properties --source . --cwd source \
  --env GPU_LEASE_WHO=memory-accounting-s3 --send /path/wikitext-2-raw-v1-test.json --send /path/c4-validation.00000-of-00008.json.gz \
  -- bash benchmarks/randomized_arith/job.sh

research run --on vy-dc-h200 --project verity --campaign stacking-properties --source . --cwd source -- bash -c \
  'mkdir -p $RESEARCH_RUN_DIR/out && PYTHONPATH=. python benchmarks/randomized_arith/cheater.py --out $RESEARCH_RUN_DIR/out/cheater.json &&
   PYTHONPATH=. python benchmarks/randomized_arith/verdict.py --cheater $RESEARCH_RUN_DIR/out/cheater.json > $RESEARCH_RUN_DIR/out/verdict.md'

python verdict.py --honest RESULTS.jsonl ...   # the honest tables, from a run's out/<model>/results.jsonl
~~~

The two sent files are the ppl bench's pinned sets (`benchmarks/ppl/datasets.json`, by `art:` id). `cheater.py` runs on
a CPU in about a minute.
