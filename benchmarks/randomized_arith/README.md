# randomized_arith

Does randomizing the arithmetic with verifier coins stop the honest int8 budget from being a free exact substrate for a
stacking cheater (workstream S3 of approach:stacking-properties/program)? This directory measures both sides.

- `honest.py`, run by `job.sh` on one GPU: the honest accuracy cost of each randomization (stochastic rounding, a
  dither offset, ±1 LSB noise, noise on every int32 accumulator, a fresh Hadamard rotation, a signed permutation) on a
  W8A8 model with exact int8 dots, against round-to-nearest-even. Perplexity on the ppl bench's pinned windows, MMLU
  5-shot on a fixed subset and GSM8K 8-shot.

~~~text
research run --on vy-dc-h200 --gpus 1 --max-min 300 --project verity --campaign stacking-properties --source . \
  --cwd source --send /path/wikitext-2-raw-v1-test.json --send /path/c4-validation.00000-of-00008.json.gz \
  -- bash benchmarks/randomized_arith/job.sh
~~~

The two sent files are the ppl bench's pinned sets (`benchmarks/ppl/datasets.json`, by `art:` id).
