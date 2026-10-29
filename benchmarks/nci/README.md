# The inference-only policy's literature index and overhead board

Data and a calculator for the question negotiators will ask about the
[inference-only policy](../../verity/core/protocols/network/nci/README.md) ("bounding accumulation"): if both sides
adopt it, how fast can a determined actor still advance frontier capability? Four parts:

- every source we rely on, each claim quoted verbatim (`sources/`);
- the parameters the bounds depend on, each estimate saying which findings it merges and how (`parameters.json`);
- the policies they are evaluated under (`calibrations.json`);
- the board: one training method in one configuration per row, with an overhead bracket (`rows.json`, `slowdown.py`).

Nothing in `verity` reads this directory. The theorems it cites are in `Security/`: `Nci.SecurityProofs.ThreadedTraining`
(`Security/Definitions/Nci/Guarantees/Threads.lean`) and `Nci.SecurityProofs.ContainedTraining`
(`Security/Definitions/Nci/Guarantees/Contained.lean`).

The slowdown lower bound's numbers come from [`benchmarks/rho`](../rho/README.md), the one slowdown calculator.
`tiers.py` gives the remote composition's terms per tier (`remote_terms`), and the slowdown on them is rho's
`rho.theorem.multiple_bound` in Lean's form (`Nci.Dampener.remoteDampener` at the reference's MFU), which `statement.py`,
`fit.py` and `figures.py` read. Hosts carry no compute and no run beats the reference's efficiency (`GLOSSARY.md`), so
`host_share` is 0 and the residual trains at MFU₀ (`tiers.evaluate`'s `residual_mfu`), while the counted
multiplications (NCI's term, χ, the memory cap) stay at peak: at `tiers.py`'s defaults the slowdown is 3.02, and rho's
remote scenario at the H200 gives 3.04. Without the reference-efficiency assumption they are 1.21 and 1.22, without the
hosts assumption 2.19 and 2.80, and without both 1 and 1.12. The code and its outputs keep the older names (`slowdown.py`, and `tiers.py`'s `efficiency` and `residual`;
`GLOSSARY.md`, Older names).

## Files

| Path | Holds |
| --- | --- |
| `sources/<topic>.json` | sources filed under their first topic, each with `findings`: `claim` (ours), `quote` (verbatim), `location`, `quantities`, `grade`, `bears_on`. A fetched fixture (`fixtures/artifacts.json`), since it would put the tree over its data budget: `research data fetch-fixtures benchmarks/nci/sources` brings it; after an edit, `research data put --kind fixture/v1 --tree benchmarks/nci/sources --preserve` and the new id in the registry (the control pod exports it once the PR is admitted) |
| `parameters.json` | parameters, each with estimates: `low`, `central`, `high`, `confidence`, `evidence` (finding ids), `merge` |
| `calibrations.json` | a policy (`K`, a block's fresh-input budget in bits; `Gamma`, its gate budget; `b_tok`) and the model it is calibrated to |
| `rows.json` | the board's rows: method, configuration, state representation, calculator inputs (a number or `estimate-id:low|central|high`), notes, sources, demonstrated capability |
| `algorithms.json` | the algorithm library: training algorithms by architecture-agnostic traits (`library.TRAITS`), measured multipliers and their evidence |
| `models.json` | model shapes the library queries against |
| `arguments.json` | the argument map: claims on a ladder, the premises each rests on, the attacks against it, and the proposed evidence with its review |
| `arguments.py` | the map's checks (every reference resolves, including Lean names against the lock) and its rendering |
| `catalog.py` | loading, and `problems()`: every rule below that a file breaks |
| `library.py` | queries over the library: a bracket and the per-block read breakdown for any algorithm, model, policy and data mode |
| `slowdown.py` | the bracket per row; `python benchmarks/nci/slowdown.py [--calibration ID] [--row ID]` |
| `kappa.py` | kappa per token and over each frontier model's pretraining schedule, by part; `python benchmarks/nci/kappa.py [--model ID]` |
| `covering_priced.py` | Kimi K2's overhead under the covering bound's forms (one cost, two costs, a price per weight, the priced additive form) at bf16, FP8 and FP4, from per-class weight prices; `python benchmarks/nci/covering_priced.py [--prices FILE.json] [--dgrad]`; `--baselines [--prices FILE.json ...]` gives the priced form's gain over the uniform form under each `c_w` a registration (the weights alone, or with the optimizer state) makes selectable, per price source |
| `prices_olmoe.json` | a price file for `covering_priced.py --prices`: per-role prices of an updated weight measured on OLMoE-1B-7B-0924's checkpoints (AdamW, 64 experts, top-8), the routed experts (`mlp_in`, `mlp_out`) apart from a dense MLP (`dense_mlp_in`, `dense_mlp_out`); its `about`, `runs` and `method` say how |
| `prices_olmo2.json` | a price file for `covering_priced.py --prices`: per-class prices of an updated weight measured on OLMo-2 7B and 13B at early, mid and late stage 1, theta's forward codes (`points`) and with its fp32 master (`points_optimizer`), and the theta-selectable tiers over the block matrices; its `about` and `runs` say how |
| `tiers.py` | the remote composition's terms under the verification stack, per tier (its `efficiency`, and `remote_terms`, the `s` and `x` rho's `multiple_bound` takes); `python benchmarks/nci/tiers.py [--json]` |
| `exposure.py` | what `tiers.py`'s `efficiency` is most exposed to: ranges, tornados, elasticities, thresholds on its inverse, and the closed forms of mu, nu and chi |
| `fit.py` | the fit bound: the memory cap when no server window holds a model that reaches the target loss; `python benchmarks/nci/fit.py [--json]` |
| `figures.py` | `tiers.py`'s talk figures, with the slowdown from rho's `multiple_bound`, to the evidence store |
| `statement.py` | every number that the Notion page on `tiers.py`'s claims, model and tiers quotes, at the commit it runs on; `python benchmarks/nci/statement.py [--json]` |
| `render.py` | the whole index as Markdown on stdout |

Topics: `weights`, `updates`, `seed-based`, `gradients`, `activations`, `tokens`, `capability-per-bit`, `theory`,
`governance`, `integration-measures`, `efficiency-metrics`, `strategy-local`, `strategy-lowrank-sparse`,
`strategy-gradient-free`, `strategy-reuse`, `capability-measures`, `rl-scaling`.

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
web-search agent.

- An auditor re-fetched a seeded random sample of the findings. A finding the audit corrected carries an `audit` field
  saying what changed.
- The scaling corpus (findings marked `from_review: scaling-corpus`) was checked by its agent's script against the
  fetched pages but not yet audited.

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

## How we measure overhead

**The yardstick.** Suppose an actor's last pretrain reached held-out loss \(L^*\) with compute \(C^*\). After the deal,
how much compute must a compliant circuit spend to reach \(L^*\) again? The overhead, net of better recipes and of
progress, is

\[
R \;=\; \frac{1}{g}\,\min_A \frac{\text{overhead}_A}{m_A}.
\]

- \(\text{overhead}_A\) is the policy's overhead for training strategy \(A\): the board and the library compute it from
  \(A\)'s traits.
- \(m_A(C) = C_{\text{ref}}(L_A(C))/C\) is \(A\)'s compute multiplier: the compute the reference recipe needs to reach the
  held-out loss \(A\) reaches with compute \(C\), divided by \(C\). It is the compute-equivalent gain (`davidson-2023-ceg/1`),
  with the reference and the test set made explicit. The reference is a compute-optimal dense AdamW Transformer (`hoffmann-2022-chinchilla/1`), with
  \(C_{\text{ref}}(L)\) from a Chinchilla-form law (`besiroglu-2024-chinchilla-replication/2`).
- \(g\) is algorithmic progress in the base recipe (architecture, data, optimizer, allocation) since the last pretrain. It
  applies with or without the policy, so the talk shows it and then brackets it. Loss-based estimates put it near 2.5x
  per year, range 1.1-5x (`ho-2024-algorithmic-progress/3`, `gundlach-2025-origin/14`, `whitfill-2025-selection-bias/1`).
  Capability-based measures, which include post-training and data, run faster: 3.5-10x per year
  (`whitfill-2025-time-horizon/1`, `ho-2025-rosetta-stone/5`, `ho-2026-least-understood/1`). Most of it is
  scale-dependent (`gundlach-2025-origin/14`).

So the strategies that matter are those that trade compute for less fresh input per block: low-rank and
sparse state, local or decoupled training, seed-described updates, reuse of registered models. Their \(m_A\) is measured
against the same base recipe (DiLoCo against data-parallel training of the same model, a student against training from
scratch). Base-recipe improvements enter only through \(g\).

**The loss.** Bits per UTF-8 byte, which does not depend on the tokenizer (`gao-2020-pile/1`,
`huang-2024-compression-represents-intelligence/5`), on a broad multi-domain corpus (`magnusson-2023-paloma/1`) written
after every compared model's training cutoff (`li-2024-compression-time-split/1`), fixed and published in advance. The
corpus matters: a different validation set can reverse which recipe wins (`lourie-2025-unreliable/2`), and
quality-filtered data can lose on broad web text while winning benchmarks (`penedo-2024-fineweb/3`). Loss is preferred
to benchmarks because it scales smoothly and extrapolates within a recipe (`gadre-2024-overtraining/2`,
`brandfonbrener-2024-loss-to-loss/1`), and equal loss means equal downstream scores when corpus and architecture are
fixed (`du-2024-emergent-loss/1`). Benchmark-based multipliers appear in the library only flagged.

**Scale.** A multiplier is a function of compute. At 1e24-1e26 FLOP every value is extrapolated, and the library marks
it so: a 1% loss error is a factor of 2-7 in compute there (`choshen-2024-hitchhiker/1`), and the loss floor itself is
fitted anywhere from 1.68 to 2.01 (`gundlach-2025-origin/18`). Teacher, generator and rephraser compute would count
toward \(m_A\) (`ho-2026-least-understood/5`), except that a registered model runs free under the policy, so the library
counts the student's compute only.

**RL and inference.** Held-out pretraining loss is the right yardstick for "can they train a new base model", but it is
blind to RL, which is reported in capability, and to inference-time compute. The measure for those is a
compute-equivalent gain on a stitched capability index (`ho-2025-rosetta-stone/1`), reported as a pair:

- weights-only: the RL model against its base at the same test-time tokens, the part the update carries and the only
  part the inference-only policy meters;
- deployed: at the model's actual thinking budget, the difference being inference, which the policy leaves free
  (`ord-2025-mostly-inference-scaling/1`, `ho-2025-reasoning-ceg/3`).

Reasoning RL is worth about 10x deployed (range 1-100x; `ho-2025-reasoning-ceg/1`, `ho-2025-reasoning-ceg/2`), roughly
3x of it in the weights. Today's RL fits in a block: DeepSeek-R1-Zero's whole RL run carried at most about 5.3 Mbit
(`schulman-2025-lora-without-regret/3`), well under \(K\). The policy therefore leaves elicitation of a registered model
unthrottled, about an order of magnitude of effective compute, and bounds it by what a registered model plus \(K\) bits
of new state per block can reach. What it throttles is moving the base model's frontier.

## The bracket

For each row under each calibration of its model, `slowdown.py` computes a bracket:

- `lower` is the proven bound, `Nci.SecurityProofs.ThreadedTraining`:
  \(\text{overhead} \ge \kappa\, b_a / (2\alpha\Lambda_T)\) with \(\alpha = K/\Gamma\) and the thread term
  \(\Lambda_T = \max(e,\ cK/(4 d\, b_w))\) (not the Notation's Λ). It holds under the row's rates and the theorem's
  assumptions. `rates` caps the inputs where a row's own numbers make an
  assumption false:
  - the weight rate is at most the whole update spread over its weights;
  - the activation rate is at most `b_tok`, since a token id is a thread value, and at most `w_cut / d`, since a block
    may take the values it needs from a compressed boundary crossing;
  - when one block can hold the state a token touches, the entry fee vanishes (`d = 1`, `e` at least the counted
    multiplications per token).
- `upper` is the cost of the cheapest construction `search` finds. In it, a block covers a slice of every token's
  computation. It reads its slice's state, or the whole description when that doesn't split (seeds), plus the
  activations crossing into the slice and the tokens' data. It decodes its own slice and is padded to `Gamma`. The
  result is labeled `fit` (one block holds the whole computation), `shard` (slices split the state) or `decode-shard`
  (each slice reads a whole description and decodes its part). One sequential chain of blocks is assumed. The
  constructions are analytic; none has been run.
- `chain` is the partial-sum law, `Nci.SecurityProofs.ContainedTraining`:
  \(\text{overhead} \ge \kappa_{\text{chain}} f(K)/(2\alpha)\),
  with \(f\) the read factor of \(b_w\), \(b_a\) and the chain's smallest inner dimension. Its only assumptions are that
  one chain of the reference training algorithm's circuit is incompressible at \(b_w\) and \(b_a\), in support form. A
  chain charges each weight once per layer, so it is one forward path, and a row's `chain` gives that path's share
  \(\kappa_{\text{chain}}\) and inner dimension. For Kimi K2 the path is the dense one (MLA projections and the shared
  expert), 7% of training's work. The activation rate is capped by a compressed boundary's bits per residual entry.
  Rows without a chain get none.

\(\kappa\) is the thread theorem's hypothesis \(\kappa\,|\text{computed}| \le 2|M|\): twice the multiplications of an
updated weight by an activation or an activation gradient, over every gate of the reference training algorithm's
circuit, one gate per scalar operation (a multiply-add is two). `kappa.py` counts both from each model's config and
its report's pretraining
schedule. The weight gradients, the LM head's forward, attention's score and value products, softmax, norms, routing
and the optimizer step are in the denominator only, so \(\kappa < 2/3\). Kimi K2's schedule gives 0.60 (0.603 at 4k,
0.40 at 32k), DeepSeek-V3's 0.56, Llama 3 405B's 0.63 and Qwen3-235B's 0.57. The adversarial 0.5 is an attention-heavy
shape with a fifth of its tokens at 32k; reported runs put at most about 5% of their tokens above their base context
(Llama 3's 800B of 15.6T). The other rows are counted the same way (`kappa.py`'s `ROWS`). The dense-100b example on
Llama 3's schedule gives 0.61. RL's update needs the backward through every sampled token, so its circuit is
pretraining's at the rollout's length: 0.39 on-policy at R1-Zero's 32k cap (0.56 at 8k, 0.28 at 64k; R1-Zero's
reference-model and off-policy forwards give 0.47). Expert-specialized fine-tuning's frozen weights are committed in
θ, so their products are in the denominator only and they take no weight gradients: 0.11. Evolution strategies run
forwards alone: 0.58 at 32k. A lower \(\kappa\) is
the defender's conservative side: it lowers the thread bound and raises \(\nu\) and \(\mu\) (`tiers.py`). A cheater's
own lower-\(\kappa\) recipe (longer contexts, more heads) buys gates that aren't the reference recipe's work: what they
are worth enters through `m`. A faulty proof block is worth at most the counted multiplications a block can hold, and a
cheater can declare blocks of weight matmuls, so `tiers.py` takes such a block's own \(\kappa_{\text{block}} = 1\)
(each multiply-add's multiplication counted) rather than the recipe's average. The cheating share \(\chi\) grows as
\(\sqrt{\kappa_{\text{block}}}\): 1.27x what 0.60 gives, and 1.77x at 2, a block of multiplications alone whose
additions are left to another block.

`--sweep` prints the bracket over a grid of `b_w` and `b_a`. `--efficiency` prices training recipes (BF16, FP8,
4-bit, ternary, and extrapolated 2- and 1-bit) at the rates their matmuls read, each with its compute multiplier
`train.compute_multiplier`, and gives each recipe's multiplier over its overhead, \(m/\text{overhead}\): a ceiling from
the partial-sum law and the attack's from the construction. The attacker picks the best recipe.

The invariant `lower <= upper` is tested on every row. The two caps on threading and token rates exist because rows
violated it without them, so a violation means a row's inputs break the theorem. A tight bracket settles a row; a wide
one is a research target.

## The argument map

`arguments.json` organizes everything here around six claims, from most to least trustworthy:

| Rung | Claim | Status |
| --- | --- | --- |
| 1 | inference complies as declared | established |
| 2 | within the circuit on registered hardware, contained dense pretraining pays an overhead of about 1e6 to 1e7 | conditional |
| 3 | the policy throttles pretraining however long the run | conditional |
| 4 | every known method is throttled or weak | open |
| 5 | translated into capability and time | contested |
| 6 | robust to unknown algorithms | open |

Each claim lists:

- **what it rests on:** the proved theorems, assumptions, constructions, literature estimates and policy rules, each
  with its status;
- **what attacks it:** the red teams' findings, by severity, each with its repair;
- **what comes next.**

What registered hardware computes is one circuit, built by appending as it streams, over one registration up front. The
red teams' top attack, recommitting θ partway through, is closed by that: nothing computed under monitoring ever enters
θ, so a run of any length is charged inside the one circuit. Any later re-registration channel would reopen the attack,
so it would have to be slow and priced. A proposed rule puts every multiplication in a public matmul template, which
would let the partial-sum law apply to the declared circuit directly rather than to one contained recipe. The proposed evidence carries the round-2 red team's verdicts. Render the index
for the full map.

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
  - Retrained bottlenecks cut the bits a token carries across a shard boundary by 2–4x. One preprint claims 100x, and
    its matched-token replication loses 14–15% in loss at only 8x (`activation.boundary_bits/d7168`).
  - The sharding floor is about `w_cut / b_tok`: centrally 7e2, and 60 at the most aggressive claim.
- **Tokens are the auditor's lever.**
  - Raw token ids carry about 17.3 bits. Arithmetic-coded against a committed base model they carry about 2.2
    (`b_tok/entropy-coded`).
  - Requiring coded prompts shrinks `K` about 8x for the same fresh prompt tokens. That multiplies every full-weight
    bracket by about 5–10, but it also shrinks the free headroom for non-text state.
  - `K`, a block's fresh-input budget, caps what it takes in, not what it serves: its outputs, the tokens it generates,
    and prompts committed in `θ` use none of it. Threads crowd out only fresh inputs read outside `θ`.
- **Gradient compression does not evade a block's fresh-input budget.** DiLoCo-family methods cut traffic 100–3000x
  between workers that each hold a full model (`grad.compression_ratio/frontier`). Their achieved per-step updates,
  0.04–0.6 bits per parameter, are thousands of times `K` at frontier scale (`update.bits_per_param_step/pretraining`).
- **Shared headroom is elicitation.**
  - Every demonstrated gain from a few MB or less elicits skill the base model has: TinyLoRA, one-shot RL, LIMA and s1,
    prompt and playbook optimization (`capability.bits/*`).
  - No generational jump has been shown within about 10 MB.

## The algorithm library

`library.py` answers questions across training algorithms.

- **The library holds 83 algorithms in 17 families.** 59 come from a scaling corpus of 91 sources.
- **60 algorithms carry a measured `multiplier`:** a compute-equivalent gain over a dense AdamW Chinchilla-optimal
  reference, citing its findings, with the full `scaling` evidence beside it.
- **Each algorithm states its `convention`.** `as-published` keeps the method's state at its stored precision
  (weights and optimizer moments). `attacker` assumes the cheapest rates at about the same quality. Queries default
  to the attacker convention, which caps every algorithm's state and boundary bits at the literature's central rates,
  so all compare alike.

An algorithm is a set of traits that scale with the model's shape (`algorithms.json`; `library.TRAITS` lists them):

- the fraction of parameters it updates;
- the bits of state per updated parameter, and whether that state splits across slices;
- the width and bits of what crosses a layer boundary, and how many times per token;
- the work per token and the fresh input per token;
- any decoding work;
- the theorem's own parameters.

A query takes an algorithm, a model (`models.json`), a policy and whether the training data is registered information
(committed in θ) or fresh input. It returns:

- the bracket;
- where a block's reads go: state, activations, data;
- the multiplier over the overhead under the policy, both achievable and at the ceiling.

~~~sh
python benchmarks/nci/library.py table --model kimi-k2 --policy kimi-k2-1m-raw    # brackets and read shares
python benchmarks/nci/library.py data                                             # what training data is worth, per model
python benchmarks/nci/library.py query lora-16 --model dense-7b --data committed  # one query, as JSON
~~~

**What the multipliers say.** Most algorithmic changes are worth 1–2×, constant with scale, and they don't stack.
Optimizer gains shrink to about 1.1× near 1B. Fine-grained MoE is the one large multiplier that grows with scale, about
20× at 10²⁰ FLOP. Against overheads of 10⁵–10⁷, a better recipe matters only if it also lowers the overhead. The
algorithms that lower it (compressed boundaries, low-rank updates, local learning, zeroth-order) all have multipliers
of 1 or less.

**How big a deal is training data?** Not much, once a model doesn't fit in a block.

- At Kimi K2 scale, the cheapest construction for every algorithm spends about half its reads re-reading weights and
  half on activations crossing in. Data is under 1%, and committing the data changes the cost by under 1%.
- Data is the whole story only when one block can hold a token's entire state. That covers models of a few million
  parameters (a factor of about 5,000 at 4M), fine-tuning deltas and seed runs on models up to about 1–7B, and
  rank-1 RL on small models.
- On the proof side it matters a great deal: with the data committed, the theorem's containment hypothesis fails, and
  the proven bound drops to 1.

Three caps keep the library's lower bounds valid. Each was added because a query broke lower ≤ upper without it:

- a token's charge is at most `b_tok`;
- an activation's charge is at most the bits a compressed boundary carries per value;
- a block that holds the whole state is charged no more per token than the fresh input it reads.

## What the committed parameters may hold

Today the policy puts no limit on θ, the registered information. That matters most for seed-based training.
A developer may commit a noise table, one entry per parameter per direction, and then decode an update with one
multiply-add per parameter instead of a PRNG draw.

- **Without a cap** (`kimi-k2-1m-raw-uncapped`), `zo-seed-pool`, `es-population-30` and `mezo-seed-finetune` cost
  about 1.2, 2 and 3.
- **With a cap**, which every other calibration assumes (`theta_cap`), they cost about 3, 2,200 and 2,400. A fixed seed
  pool fits in one block even then: its decoding, 4.3×10¹⁶ flops, is under `Gamma`.

**Design constraint.** The policy serves an agreement between rivals, so a rule on θ must neither reveal nor narrow the
architecture. Checks run inside the zero-knowledge proof, and the verifier learns only that they pass.

**Recommended rule: two public budgets.**

- At most `B_matrix` bits of θ may be read only as a multiplication's operand against a value not in θ (matrix weights,
  in any architecture).
- At most `B_other` bits may be anything else.

Both budgets are set alike for all parties and generously, a few times the largest permitted model, so they reveal
nothing about any party's model. A noise table for a 1T-parameter model costs about 125 GB per direction, so a budget
of a few TB leaves room for too few directions to train with. Frontier corpora don't fit either.

**Rejected** because they would reveal or narrow the architecture: typed roles, declared shapes or layers, and a
per-model registry. A rule that committed text enter only through the embedding lookup is optional. It is generic, but
it needs the proof to identify that operation.

**Not closed.** Structure can't tell weights from data: a one-hot activation copies any θ entry. A corpus can also hide
in a checkpoint's slack bits, about 2×10¹¹ tokens for a 1T-parameter model requantized within 1% loss. So small-model
training on smuggled data stays bounded, not closed.

**The proof gap.** The thread guarantees say nothing about a developer whose data is in θ, since their premise
`ThetaRegistered` fails. The fix to pursue states the guarantee with a term for the committed data, degrading by
at most the bits of data the budgets allow.

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
