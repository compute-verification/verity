# NCI literature index and slowdown board

Data and a calculator for the question negotiators will ask about the NCI policy ("bounding accumulation"): if both
sides adopt it, how fast can a determined actor still advance frontier capability? Four parts:

- every source we rely on, each claim quoted verbatim (`sources/`);
- the parameters the bounds depend on, each estimate saying which findings it merges and how (`parameters.json`);
- the policies they are evaluated under (`calibrations.json`);
- the board: one training method in one configuration per row, with a slowdown bracket (`rows.json`, `slowdown.py`).

Nothing in `verity` reads this directory. The theorems it cites are in `Security/` (`Nci.SecurityProofs.ThreadedTraining`,
`Security/Models/Nci/Guarantees/Threads.lean`).

## Files

| Path | Holds |
| --- | --- |
| `sources/<topic>.json` | sources filed under their first topic, each with `findings`: `claim` (ours), `quote` (verbatim), `location`, `quantities`, `grade`, `bears_on` |
| `parameters.json` | parameters, each with estimates: `low`, `central`, `high`, `confidence`, `evidence` (finding ids), `merge` |
| `calibrations.json` | a policy (`K` bits a block may read besides committed parameters, `Gamma` gates per block, `b_tok`) and the model it is calibrated to |
| `rows.json` | the board's rows: method, configuration, state representation, calculator inputs (a number or `estimate-id:low|central|high`), notes, sources, demonstrated capability |
| `catalog.py` | loading, and `problems()`: every rule below that a file breaks |
| `slowdown.py` | the bracket per row; `python benchmarks/nci/slowdown.py [--calibration ID] [--row ID]` |
| `render.py` | the whole index as Markdown on stdout |

Topics: `weights`, `updates`, `seed-based`, `gradients`, `activations`, `tokens`, `capability-per-bit`, `theory`,
`governance`, `integration-measures`.

## Rules for adding to it

`tests/` enforce these through `catalog.problems()`:

- A source has a canonical `https` URL (`arxiv.org/abs/…` where there is one), no URL twice, and `verified: fetched` (we
  read the page and saw the quote) or `search-snippet` (a search result showed it).
- A finding's `quote` is text seen in the source, never a paraphrase. A derived number says so in its `condition`
  ("derived: 16 bits / 10x").
- Finding ids are `<source id>/<n>`, in order. Append findings; never renumber, since estimates and rows cite them.
- Every estimate cites at least one finding and says in `merge` how it combines them. Every row input resolves.
- A source goes in the file of its first topic.

Agents writing from memory produce plausible but wrong citations. Every source here was found and checked by a
web-search agent, and an auditor re-fetched a seeded random sample of 40 findings (2026-10-17). All 40 quotes were
verbatim; three findings were corrected, and those carry an `audit` field saying what changed.

## How estimates merge findings

An estimate answers one question in one context, such as "bits per activation crossing a training pipeline boundary at
retained quality". Its three values follow one convention:

- `low` is what an attacker could reach on the best published evidence, however thin.
- `central` is what the strongest measurement at the largest scale supports.
- `high` is what standard practice uses.

Where sources conflict, `merge` names the conflict and the reading taken: matched-loss results over lossy ones, and
large models on uncontaminated data over small ones. Rates that need training to decompress are excluded from weight
rates, because decompressing is itself training. "Compression" by soft prompts counts positions, not bits, so it is
not a bit ratio.

## The bracket

For each row under each calibration of its model, `slowdown.py` computes a bracket:

- `lower` is the proven bound, `Nci.SecurityProofs.ThreadedTraining`:
  \(\rho \ge \kappa\, b_a / (2\alpha\Lambda)\) with \(\alpha = K/\Gamma\) and \(\Lambda = \max(e,\ cK/(4 d\, b_w))\). It
  holds under the row's rates and the theorem's assumptions. `rates` caps the inputs where a row's own numbers make an
  assumption false:
  - the weight rate is at most the whole update spread over its weights;
  - the activation rate is at most `b_tok`, since a token id is a thread value, and at most `w_cut / d`, since a block
    may take the values it needs from a compressed boundary crossing;
  - when one block can hold the state a token touches, the entry fee vanishes (`d = 1`, `e` at least the counted
    multiplications per token).
- `upper` is the cost of an explicit compliant construction, the cheapest that applies:
  - `fit`: the state fits in a block;
  - `decode-shard`: the description fits, but decoding it does not;
  - `shard`: layer sharding with activations crossing the boundaries.

  The constructions are analytic; none has been run.
- `chain` is the earlier bound \(\kappa F\), for comparison only. The lock marks its assumption as falsified, and on the
  small-adapter rows it exceeds a compliant construction's cost.

The invariant `lower <= upper` is tested on every row. The two caps on threading and token rates exist because rows
violated it without them, so a violation means a row's inputs break the theorem. A tight bracket settles a row; a wide
one is a research target.

## What the literature says now

All values below are from `parameters.json`; render the index for the findings behind each.

- **Weights barely matter.** Trained weights need about 2.5 bits per weight at retained quality, and about 0.8 at the
  best published MoE result (`b_w.pretrained/frontier-lossy`). Their information capacity is about 2–3.6 bits per
  parameter (`b_w.capacity/measured`). The thread bound is flat in `b_w` above about 0.1–0.5 bits, so compressing a
  model's weights doesn't weaken it.
- **Updates are the hole.**
  - Fine-tuning deltas code at about 0.1 bits per base weight (`b_w.delta/sft`).
  - RL needs far less: rank-1 LoRA matches full RL at 8B, and RL conveys about 1 bit per episode (`b_w.delta/rl`).
  - Whole seed-based runs are 3.2e5 bits (`update.bits_total/seed-run`).
  - So `U`, the description of the update, sits at or below `K` for every demonstrated post-training method except
    full-weight training. At a 1T MoE, though, decoding seeds costs steps × parameters, and that is what binds.
- **Activations set the floor.**
  - Production systems send 8-bit activations across parallel links. The one boundary test in training is stable only
    from about 4.4 bits (`b_a.activation/boundary-training`).
  - Retrained bottlenecks cut the bits a token carries across a shard boundary by 2–4x, and one unreplicated preprint
    claims 100x (`activation.boundary_bits/d7168`).
  - The sharding floor is about `w_cut / b_tok`: centrally 7e2, and 60 at the most aggressive claim.
- **Tokens are the auditor's lever.**
  - Raw token ids carry about 17.3 bits. Arithmetic-coded against a committed base model they carry about 2.2
    (`b_tok/entropy-coded`).
  - Requiring coded prompts shrinks `K` about 8x for the same inference. That multiplies every full-weight bracket by
    about 5–10, but it also shrinks the free headroom for non-text state.
- **Gradient compression does not evade a per-block cap.** DiLoCo-family methods cut traffic 100–3000x between workers
  that each hold a full model (`grad.compression_ratio/frontier`). Their achieved per-step updates, 0.04–0.6 bits per
  parameter, are thousands of times `K` at frontier scale (`update.bits_per_param_step/pretraining`).
- **Shared headroom is elicitation.**
  - Every demonstrated gain from a few MB or less elicits skill the base model has: TinyLoRA, one-shot RL, LIMA and s1,
    prompt and playbook optimization (`capability.bits/*`).
  - No generational jump has been shown within about 10 MB.

## The board today

Under `kimi-k2-1m-raw` (K = 2e7 bits, a 1M-token session at 32B active parameters):

| Row | Lower (proven) | Upper (construction) |
| --- | --- | --- |
| `pretrain-full` | 1.1e5 | 1.5e7 (shard) |
| `rl-full-weights` | 1.4e5 | 9.2e6 (shard) |
| `rl-rank1-lora-all-experts` | 4 | 6.3e4 (shard) |
| `mezo-seed-finetune` | 1 | 4.7e3 (decode-shard) |
| `rl-tinylora`, `context-playbook` | 1 | about 1 (fit) |

Each row's numbers move with the calibration. The 11 MB calibration lets a fixed seed pool fit (`zo-seed-pool`, about
1.2) but not a 20k-step MeZO run. Coded prompts raise full-weight training's lower bound about 5x.

The widest brackets are the research targets:

- **`rl-rank1-lora-all-experts`** (4 to 6.3e4) decides whether frontier RL escapes.
- **The seed-based rows.** Their lower bound is 1, so their cost is whatever the best construction pays.
- **`pretrain-full`**. The proven bound (1e5) and the construction (1.5e7) are 140x apart.

## What changed from the prior draft's appendix

The prior review's citations all resolve, but these uses don't hold up. Each item cites its finding:

- **TT-LoRA MoE.** Its "rank 5" is a tensor-train rank (33,920 parameters), and it trails rank-16 LoRA rather than
  matching it (`kunwar-2025-ttloramoe/2`, `/3`).
- **Wang et al. (OREO)** covers math and ALFWorld, not coding (`wang-2024-oreo/1`).
- **Song et al.'s coverage result** is for offline contrastive methods only (`song-2024-onlinedata/1`).
- **Rinberg et al.**
  - The 33x is a lossy rewrite; the lossless result is about 11x (`rinberg-2026-haikutoopus/2`, `/3`).
  - The 250–1667x buys part of a capability gap, not compressed responses (`rinberg-2026-haikutoopus/4`).
- **Zhang et al.** is an empirical study whose gains are at about 2x (`zhang-2025-promptcompression/1`).
- **Brown et al.'s 16–100x** is measured only at 1.5B. At 7B, about 10x costs 22 MMLU points
  (`brown-2026-weight-theft/7`, `/8`).
- **Unsourced claims.** "Updates compressible 50–100x" and the DeepSeek-V4-Pro LoRA sizes have no source here.

## Not yet here

- Scenario columns (today, ordinary progress to 2028, a crash program) and capability ratings on agreed scales.
  Rows record only the capability the literature has demonstrated.
- Constructions run at small scale.
- Rows for distillation, model merging, test-time training and offline RL.

## Rendering

~~~sh
python benchmarks/nci/render.py > /tmp/nci-index.md     # never commit the output
python benchmarks/nci/slowdown.py --calibration kimi-k2-1m-raw
~~~
