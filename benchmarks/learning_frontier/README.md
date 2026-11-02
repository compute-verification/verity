# learning_frontier: loss against cost under restrictions, at small scale

Small decoder-only language models trained on one fixed corpus, to measure what a restriction on training costs at a fixed
loss: the compute penalty κ of a restricted family against dense AdamW, and how much of the restricted quantity (the fresh
input each period of training creates, bits communicated) the family saves at that loss, the implied α. Each is a ratio at a
held-out loss target every compared configuration reaches:

- κ(ℓ) = C_family(ℓ) / C_dense(ℓ), the cost to first reach loss ℓ, in analytic training FLOPs, in tokens and in measured
  training seconds;
- α(ℓ) = Ω_dense(ℓ) / Ω_family(ℓ), Ω the restricted quantity spent up to ℓ:
  - per-window low rank, per period: the fresh input of every window up to ℓ, a window's being every value it trains (what the
    next window takes in when the base is registered at every boundary);
  - per-window low rank, on the stock: the values in the departure from the initialization at ℓ (what a server block must
    hold when the base is registered once, at the start), every parameter for dense;
  - local SGD: the bits communicated up to ℓ.

These are small-scale ratios on one corpus, one token budget and one window length or sync interval each, not frontier-scale
numbers; a size trend across three sizes is not an extrapolation past them.

**The corpus** (`prep.py`): fixed shards of `HuggingFaceFW/fineweb-edu` (`sample-10BT`) at a pinned revision, GPT-2 tokens
(tiktoken `gpt2`, each document prefixed by the end-of-text token), and a fixed held-out split from a shard no training run
reads. Held-out loss is in nats per byte: the summed negative log-likelihood of the held-out targets over their UTF-8 bytes.
`manifest.json` records the dataset, revision, shards and their hashes, the tokenizer and the split.

**The model** (`model.py`): Llama-style (RMSNorm, SwiGLU, RoPE, no biases), context 1024, the 50,257-token vocabulary
padded to 50,304, input and output embeddings tied.

| size | d_model | layers | heads | d_ff | non-embedding parameters |
|---|---|---|---|---|---|
| `s` | 512 | 6 | 8 | 1408 | 19.3M |
| `m` | 768 | 8 | 12 | 2048 | 56.6M |
| `l` | 1024 | 12 | 16 | 2816 | 154.2M |

**Training** (`train.py`): 64 sequences of 1024 tokens per step, 8 tokens per non-embedding parameter by default; AdamW
(β 0.9, 0.95; weight decay 0.1 on matrices), gradient clipping at 1.0, linear warmup over 5% of the steps and cosine decay to
10%, bf16 autocast. Every family at a size starts from the same initialization and reads the same sequences in the same order.
The held-out set is evaluated 20 times, at equal step intervals. The families:

- `dense`: AdamW on every parameter, the reference.
- `lowrank`, per-window low rank: each window of 200 steps trains a rank-r factor pair on every matrix (attention, MLP and
  the tied embedding) and the norms in full; at the window's end the pairs merge into the frozen matrices and fresh ones are
  drawn (`model.py`). The factors' optimizer state is reset at every merge, the norms' persists, and the learning rate
  re-warms over 20 steps after each merge. There is no full-rank warmup: every step is low rank. A window's fresh input:
  every value trained in it, factors and norms, at fp32; dense's is every parameter, at fp32, per window of the same length.
  The embedding gets a factor pair because counting it in full each window would cap α near the ratio of all parameters to
  the embedding's, whatever the rank.
- `localsgd`, local SGD with an outer Nesterov step (DiLoCo-style): 4 replicas, each with its own AdamW kept across syncs,
  take a quarter of every global batch; every H steps their mean delta from the global model is the outer gradient of SGD
  with Nesterov momentum (learning rate 0.7, momentum 0.9, the same at every H), and every replica restarts from the new
  global model. Bits communicated per sync: each replica's delta, every parameter at fp32; dense's are data parallel over 4
  workers, each sending its full fp32 gradient every step. The global model is evaluated at syncs. When the step count is
  not a multiple of H, the last outer step follows a partial period and still applies the full outer momentum, so the last
  evaluation can sit above the one before it; `analyze.py` reads each curve's running minimum.

**Cost.** Analytic FLOPs per token: dense, 6 per matrix entry (the tied head counted once as a matrix, the lookup a gather)
plus causal attention, 6 · layers · context · d_model; per-window low rank, 4 per frozen matrix entry (no weight gradient)
plus 6 per factor entry, plus attention. The Kaplan count, 6 per non-embedding parameter, is recorded beside it. Measured:
tokens per second, achieved FLOP/s and the fraction of the census's bf16 peak (`benchmarks/census/hardware.json`), over
training steps only (evaluation and the first 10 steps after each start excluded). Local SGD's replicas run one after
another on one GPU, so its measured time is the global batch's compute with no communication.

**Tuning parity.** Every family at the two smaller sizes sweeps the same three peak learning rates, and the largest size
takes each configuration's best at `m`; where that rate is unstable at `l`, the configuration takes the next value down the
grid's factor-2 ladder. The outer optimizer of local SGD and the low-rank re-warm are not tuned.

**Envelopes.** A checkpoint inside a cosine schedule has not annealed, so a configuration can also have shorter-horizon runs
(`--tokens-per-param` below the main budget) at its main run's learning rate. `analyze.py` then reports κ and α between
envelopes as well: each side's least cost to reach a target over its main run and its shorter ones.

**Run:** `prep.py --dest DATA` builds the corpus (CPU, network). `train.py --data DATA --out OUT --ckpt-root CKPT --size s
--family lowrank --rank 32 --lr 3e-3` trains one configuration; it stops after `--chunk-minutes`, saves and exits 3, and
resumes when started again, so `leased.sh PYTHON ARGS...` runs it as a chain of preemptible 30-minute `gpu-lease` leases.
`analyze.py --out DIR RUN_DIR...` reads fetched runs (`research fetch --all RUN`) and writes the analysis.

**Reads and writes:** `prep.py` writes `train.bin`, `val.bin` and `manifest.json` under `--dest`; `train.py` writes
`config.json` (arguments, counts, FLOPs per token, the corpus manifest, the data order's hash), `curve.json` (one point per
evaluation: step, tokens, FLOPs, fresh-input bits as `novel_bits`, bits communicated, held-out loss, throughput) and
`result.json` into `--out`;
`analyze.py` writes `analysis.json`, `tables.md` and one plot per size and family into `--out`. **Backs:** no Lean guarantee
or claim; its numbers are measurements of training under restrictions.

## The covert-training speedrun

From a registered θ (a dense run's weights at 25%, 50% or 100% of its steps), how much compute does each structural class of
training method need to lower held-out loss by Δ, against the best unrestricted method: κ_class(Δ) = C_class(ℓ₀ − Δ) /
C_ref(ℓ₀ − Δ), in analytic FLOPs, tokens and GPU-seconds, and each class's ceiling, the lowest loss it reaches in the budget.

- `train.py --save-theta 0.25,0.5,1` keeps the starts: `theta-pNNN.pt` (fp32 weights and their digest, `model.theta_digest`)
  and `theta.json` (each start's step, schedule, data order, commit and held-out loss) in `--out`.
- `speedrun.py --data DATA --start THETA.pt --method adamw|muon|lora|mezo|icl --lr LR --out OUT --ckpt-root CKPT` continues
  one start under one class at a fixed FLOP budget (default 2 tokens per non-embedding parameter at dense AdamW's FLOPs per
  token), reading the training stream from past every window a base run read, and writes the run's witness record
  (`witness.json`: commit, configuration, seed, θ's digest, ℓ₀, analytic FLOPs, tokens, GPU-seconds, the held-out trajectory
  at every 5%, update bits, fresh input per window, the result's digest). The classes and their structural knobs are its
  docstring. It resumes like `train.py`, so `LEASED_SCRIPT=speedrun.py leased.sh PYTHON ARGS...` runs it under leases.
- `plan.py` lists the runs (per size: 3 starts × S0 AdamW and Muon, S1 LoRA at 100 kB, 1 MB, 10 MB and 100 MB, each at 3
  learning rates, and S3) with each one's steps, tokens, FLOPs and estimated GPU time; `plan.py --chain SIZE FRACTION METHOD
  UPDATE_BYTES` prints the shell loop one `research run` takes for a configuration at its 3 rates.
- `kappa.py --out DIR RUN_DIR...` reads fetched runs: per start, each configuration at its best rate, the reference (the best
  S0 configuration), κ at Δ up to the reference's gain at half the budget (so the budget is twice the reference's cost to the
  deepest target), the ceilings, S3's gain against its inference cost, and the plots (`analysis.json`, `tables.md`,
  `loss-budget-SIZE.png`, `kappa-SIZE.png`, `ceilings-SIZE.png`).
- `smoke.py` runs all of it on a tiny model and a synthetic corpus on one CPU thread (about a minute): digests, evaluation
  parity with `train.py`, FLOP counts, every class, LoRA's merge, resuming, and `kappa.py` on the outputs.
