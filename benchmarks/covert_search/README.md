# covert_search

A continuing search over training methods per structural class, scored by a fixed evaluator: an entry continues a registered
pretrained start θ as a chain of metered blocks and outputs a model; a fixed bits-per-byte evaluator scores it on held-out
fineweb-edu text at context 1024, with evaluation compute per byte capped at the start's forward cost. In the shape of
karpathy/autoresearch: a fixed `prepare.py`, a fixed meter, and one editable `train.py` per island with its `program.md`.

| file | role |
|---|---|
| `prepare.py` | fixed: the data (training stream; SEARCH = shard 013's first 2,097,153 GPT-2 tokens, the first curves' held-out text; FINAL = shard 012's, which no candidate is scored on), the registered starts, and `evaluate_bpb` |
| `meter.py` | fixed: the block executor; refuses a violation and writes the witness |
| `islands/<class>/train.py`, `program.md` | one island per class: the editable seed and the rules for the agent that searches it |
| `launch.py`, `job.sh` | one candidate: `research run --on vy-nebius-2 --pool research --gpus 1`, then the evidence store and the island's `results.tsv` |
| `prep_job.sh` | the integrator's one-time node jobs: retrain the dense starts, build the data, register the starts |
| `smoke.py` | CPU, seconds: every seed completes in its class; the meter refuses an over-k carry and state outside the carry |

## Classes and c

| island | class | k (fresh input per block beyond θ) | seed |
|---|---|---|---|
| `reference` | ref | unlimited | full-parameter AdamW; Muon with `--hp opt=muon` |
| `mb10` | 10MB | 8·10⁷ bits | LoRA on the hidden matrices, rank sized to fit, AdamW, merged at export |
| `mb1` | 1MB | 8·10⁶ bits | the same, scope narrowed until rank 1 fits |
| `kb100` | 100kB | 8·10⁵ bits | MeZO on the norm gains, the update kept as its seed log |

c is the FLOPs of one forward and backward over 64 sequences of 1024 tokens on the start, every parameter trained, as torch's
`FlopCounterMode` counts them (matrix products and attention; `start.json` `c_flops`). A block may do at most c, and
charged compute is blocks × c. The budget per candidate is 1000 blocks plus the export block. Everything crossing into a
block counts against k: the carry, at its canonical serialization; the data, at 16 bits per token; and the block's 64-bit
seed. `meter.py`'s docstring has the rest.

## Running

Preparation, once, by the integrator, on vy-nebius-2 (`research run --on vy-nebius-2 --pool research --source . --cwd source
-- bash benchmarks/covert_search/prep_job.sh ...`):

1. `starts s 3e-3` and `starts m 1.5e-3`: the first curves' dense s and m, retrained with the same configuration (their
   checkpoints weren't kept), with θ at 25%, 50% and 100% of each run.
2. `data`: the SEARCH, FINAL and training streams for both tokenizers, then SmolLM2-135M's registration.
3. `register s RUNDIR`, `register m RUNDIR`: s50, s100, m50 and m100 from step 1's run directories.

A candidate, from an island agent's checkout: `python benchmarks/covert_search/launch.py ISLAND --start m50 --note "..."`.
