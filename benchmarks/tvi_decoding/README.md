# Decoding under self-generated data (training-vs-inference program, crux 4)

How much of a fine-tuned `theta'_s` (and of its linear-map outputs) a block can predict cheaply from the registered
`theta`, and what fresh bits per value the residual still carries at bf16, FP8 (per-tensor E4M3) and NVFP4.

- `train.py`: full fine-tuning with the beta sweep's recipe (AdamW 5e-5, batch 8 x 256, stdlib corpus), fp32 snapshots of
  the main run at steps 0-1000, and the cheap proxies on the same batch stream: batch 1 (`b1`, `b1lr`) and every 10th
  batch (`sub10*`, 10x fewer steps at 1x, 3x and 10x the learning rate).
- `predict.py`: for targets 1, 10, 100 and 1000, each predictor (theta, the proxies, extrapolation along early
  snapshots, per-tensor least squares, an elementwise linear or MLP map trained on half the layers and tested on the
  other half) and two conditionals (layer l-1's fresh update; each map's fresh inputs). Bits are the sweep's
  estimators (`quant.py`, unchanged): the best of context entropy, lzma (middle layer) and low-rank MDL, with every
  coefficient fit on the target charged at 32 bits.
- `table.py`, `summary.py` and `condensed.py` render a run's `b2_<model>.json` as Markdown.

Launch from a checkout, on a GPU node of the program, with every file sent as an input:

~~~text
research run --on vy-nebius-3 --mem-gb 32 --timeout 9000 --project verity --source . --cwd source \
  --send run.sh --send train.py --send predict.py --send common.py --send quant.py \
  --env "MODELS=EleutherAI/pythia-160m Qwen/Qwen2.5-0.5B" \
  -- env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py --by training \
  -- gpu-lease 1 --wait -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run.sh"'
~~~

`SMOKE=1 B2_DEV=cpu` caps training at 10 steps on the CPU; `TARGETS`, `B2_LAYERS` and `B2_MLP_STEPS` narrow `predict.py`.
