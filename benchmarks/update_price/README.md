# Update price

Upper bounds on the price in update hardness (`Nci.Assumptions.DescriptionHard` over updated parameter slots): for a
selection U of a fine-tune's updated slots, one version each, LDL_{t,R}(x_U) ≥ Σ_{i∈U} p_i − λ, where the decoder holds
the registration R (θ, the registered model) and runs within t steps. The premise is computational, so an attack gives
upper bounds only: each decoder found here rebuilds x_U bit for bit from a description of L bits within its budget t,
which refutes every price with Σ p_i − λ > L. Failing to find a shorter one is evidence, not proof.

## The fine-tune

`train.py`: SmolLM2-135M (`HuggingFaceTB/SmolLM2-135M` at a pinned revision, published in bf16, loaded exactly into
float32) fine-tuned on the first shard of `wikitext-103-raw-v1` (pinned revision), with text packed into fixed-length
chunks in file order and drawn in a seeded permutation. AdamW runs with linear warmup then cosine decay, weight decay on
the matrices, and gradient clipping. Everything runs in float32 on CPU: the weights, activations, gradients and both Adam
moments. One thread count, `torch.use_deterministic_algorithms` and no dropout make the run a deterministic function of
R, the data, the seed and the hyperparameters on one machine. `--replay N` reruns the first N steps in the same process
and compares the parameters' SHA-256 at each saved step. The updated slots are every parameter (the embedding is tied to
the head). Each is priced at the representation it is trained in (fp32) and at its bf16 rounding, which is what a bf16
forward code holds.

## The decoders

`codec.py` (numpy and the standard library). Every description is decoded and compared with its target before its length
counts.

- **ulp** (a): the target's distance from θ in units in the last place (the float's order as an integer). Each distance is
  a symbol (sign, bit length, the two bits below the leading one) plus its remaining bits raw. The symbols go through an
  interleaved rANS coder with one frequency table per context, the context being θ's exponent and sign within a tensor.
  Contexts with fewer than 4096 slots merge, and the decoder derives the same merge from θ. The tables are in the
  stream.
- **lowrank** (b): per matrix, rank-r integer factors of the update (from its SVD, scaled by powers of two, rounded). The
  decoder adds their exact integer product to θ in float64 and rounds it to the format, and the residual is coded by
  `ulp` against that prediction. Rank and factor precision are the cheapest found.
- **pred0 and predM** (c): θ plus a float32-coefficient combination of float64 bases the decoder computes, with the
  residual coded by `ulp`. pred0 uses θ alone (weight decay's shrinkage). predM uses the first M steps replayed: θ,
  x_M − θ and the Adam direction m_M/(√v_M + ε) at step M. The coefficients are least squares per tensor, or x_M itself
  where that is cheaper.
- **replay** (d): the run record (model and data revisions, seed, hyperparameters, N) and `train.py`, both xz-compressed,
  replayed for N steps. When R doesn't hold the data, the description also carries the tokens. They are coded either
  prequentially, by the model being trained at each step (the sum of the training losses in bits, the first token of
  each chunk at log2 of the vocabulary), or by xz.

The selections (`price.py`) are:

- every slot, per class (q, k, v, o, gate, up, down, norms, embedding) and per layer (0, 15, 29);
- the largest and smallest changes (0.1% and 1%, and 1% and 10%), indexed by log2 C(n, k) bits;
- the largest and smallest |θ| (10% each, which R determines);
- seeded random subsets of 10^3, 10^5 and 10^7 slots;
- the embedding rows of tokens absent from the first N steps' data, named by R when R holds the data and indexed
  otherwise;
- θ's binades, priced from the ulp coder's per-slot code lengths.

The decoder budget t is counted in training steps of the run's machine, with seconds beside it. The codecs decode in a
fraction of a step; predM needs M steps and replay needs N.

`tables.md` reports:

- bits per updated weight for every slot, by method and t;
- per class and per layer;
- per selection;
- the λ a uniform price at the all-slots rate would need to survive every selection tried, and the largest uniform price
  that survives them at λ = 0;
- the replay descriptions;
- xz and zstd on one layer's XOR and delta byte planes, as baselines.

## Run it

~~~sh
research run --on auto --cpus 12 --mem-gb 32 --max-min 300 --cwd source -- bash benchmarks/update_price/run.sh
~~~

`run.sh` passes its arguments to `train.py`, whose defaults are the run of record: 1000 steps of 16×512 tokens, saves at
steps 1, 10, 100, 300 and 1000, Adam moments at 1, 10 and 100, and `--replay 100`. For a trial on the machine, pass
`--steps 3 --save 1,3 --state 1 --replay 1`. `train.json`, `price.json` and `tables.md` go to `$OUT`. Checkpoints (about
6.5 GB) stay in the run's scratch directory and are removed at exit. Reprice a kept checkpoint directory with
`python price.py --ckpt DIR --train train.json --out OUT [--versions 10,1000] [--only REGEX]`.

The tests (`tests/`) check that every codec round-trips exactly (with pinned encodings) and run the pricing end to end
on a toy checkpoint.
