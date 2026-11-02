# covert_witness: the best compute efficiency at lowering loss, per class of a training computation's structure

The theorem prices a training computation's slowdown from its structural parameters: the update's size (β, K′), its depth
D, its deep share φ and its block size m. This speedrun measures the other factor: for each class of those parameters, the
lowest held-out loss a candidate reaches from a registered start at a fixed training-FLOP budget, and its FLOPs-to-target
curve. The attacker's effective cost in a class is the theorem's slowdown for the class times that rate. Every candidate
carries witnesses to its parameters, which the evaluator checks; nothing here meters a chain of blocks.

It is autoresearch-shaped: a fixed side (`prepare.py`, `evaluate.py`, `guard.py`, `candidate.sh`, `launch.py`, with
`data.py`, `starts.py` and `model.py` behind them) and one island per class, each with one editable `train.py` and its
program, `README.md`, for a researcher agent.

## Classes

Caps on the witnessed parameters (`prepare.CLASSES`); every class also has the budget below.

| class | cap | seed (`islands/<class>/train.py`) |
|---|---|---|
| `reference` | none | Muon on the blocks' matrices, AdamW on the embedding and norms; E every tensor |
| `e1mb` | E ≤ 1,000,000 bytes | LoRA on every block matrix and the norm gains, the largest rank that fits (bf16) |
| `e100kb` | E ≤ 100,000 bytes | LoRA on each block's o and down projections and the norm gains |
| `shallow` | only the top 2 blocks and the final norm differ | AdamW on those; the tied head stays frozen (it is layer 0 too) |
| `seed` | E a seed log, ≤ 100,000 bytes | MeZO: two forwards a step, the step's coefficient stored, z replayed from seeds |

**Budget**: analytic training FLOPs ≤ 9e16 (`TRAIN_FLOP_BUDGET`, about 10 GPU-minutes of m's dense AdamW at its measured
1.49e14 FLOP/s) and train.py's GPU-seconds ≤ 1200 (`GPU_SECONDS_CAP`). A class's score at a start is its lowest valid
search bpb.

## The fixed side

- **Held-out sets** (`data.py`): `search`, fineweb-edu shard 013's first whole documents up to 2^21 GPT-2 tokens (2,092
  documents, 9,797,536 bytes), the only set candidates are scored on; `final`, shard 012's by the same rule, never shown to
  the search. Each is tokenized by every start's tokenizer; bits per byte divides by the documents' UTF-8 bytes, so scores
  compare across tokenizers. Training streams: learning_frontier's GPT-2 shards 000-002, and shard 000 in SmolLM2's tokens.
- **Starts** (`starts.py`, `prepare.register_smollm2`): `s50`, `s100`, `m50`, `m100`, learning_frontier's dense s and m
  (seed 0, data order 0, its best rates) at 50% and 100% of their 8-tokens-per-parameter runs, retrained here since that
  branch kept no final checkpoint; and `smollm2`, SmolLM2-135M at its hub revision. Each is registered as
  `starts/<name>/theta.safetensors` with `start.json`: the digest θ (`prepare.digest`), config, tokenizer and provenance.
- **Evaluator** (`evaluate.py`, `prepare.evaluate_bpb`): bits per byte at context 1024 of decode(θ, E). The model must
  have θ's tensors, shapes and dtypes, so its parameter count, evaluation FLOPs and weight bytes read per byte (every
  parameter's bytes once per window, a lookup table in full) are the start's: no lookup table rides free.
- **Witness checker** (`evaluate.py`): θ's digest; decode(θ, E) bit-equal to the submitted model; 8·|E| bits; the
  declared changed set covers what differs, and gives φ and the lowest updated layer (`prepare.deep_share`); training
  FLOPs from the step log and the declared trainable set (`prepare.train_flops`), which must cover the changed set;
  GPU-seconds; the class's caps. Decoders are fixed (`prepare.decode`: `full`, `lora`, `seedlog`, `sparse`) and read
  only θ and E; the witness records each E's rebuild cost in MACs (`prepare.rebuild_macs`).
- **Guard** (`guard.py`): train.py runs with an audit hook refusing any open under the held-out directory.
- **Launcher** (`launch.py` on an agent VM, `candidate.sh` on the node): one candidate a run, `research run --on vy-nebius-2
  --pool gpu` with `gpu-lease 1` inside (the node takes runs only through its slot pools, and `--gpus` is refused there);
  the record goes to the store as `research data put --kind evidence/v1 --tree OUT --preserve`.

**Run**: `python benchmarks/covert_witness/launch.py CLASS START --note TEXT -- TRAIN_ARGS` from a committed checkout.
`data.py` and `starts.py --size s|m` build the fixed side once (CPU with network; one GPU); `evaluate.py --start NAME --out F`
scores a start. `CW_DEVICE=cpu` runs the whole path on a CPU for a smoke test.

**Reads and writes**: everything lives under `/workspace/research/scratch/covert-witness` (`CW_ROOT`, kept under @training):
`eval/` (held out), `starts/`, `train/`, `ckpt/`, and `candidates/` (each candidate's submission, removed when it ends).
A candidate's run `out/` holds `witness.json` (schema `covert-witness/witness/v1`), `timing.json`, `meta.json`,
`steps.jsonl`, `E.bin` when at most 1 MB, `train.py` and its logs. **Backs**: no Lean guarantee or claim; its numbers are
measurements of training under structural caps.
