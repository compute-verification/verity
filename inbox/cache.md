# CACHE: what each departure from stock vLLM costs (Daniel's question of Wed 11 Nov, 5:44 PM PST)

Branch `cursor/proofs-cache-7c3e`, cut from `origin/main` at `be9a3d5669`. Append-only; newest entry last.

Tags on every number and claim: **[code]** read in the tree at that commit (the whole file unless "in part");
**[relayed]** from a note I read and did not recheck; **[arithmetic]** mine, from the numbers beside it;
**[modeled]** the cost model's output; **[vLLM source]** vLLM at our pin `d9105ea80`, read on GitHub; **[guess]**.

## Entry 1, Thu 12 Nov 02:20Z (due 03:00Z): what `main` charges, the five lines, the ways I will price

### Where this is heading (not final)

- **Prefix caching is possible, and the cost model's figure of record already assumes it is on.** It serves with
  the cache on and has every sampled replay prove again the prefix its request hit, so no replay block reads another's
  values (fan-out 1). What turns caching off is the serve we built and measure (the `two-stage` arm), not the design
  the figure is computed for. The two disagree, and nobody has said so.
- **The price of that design is not a larger charge per missed block. It is audit compute:** a sampled request's
  replay proves its whole prompt, cached part included. On our plan that is 1.14 times the proving. On agent traffic
  where 86% to 98% of prompt tokens are cached it is 7 to 46-59 times, and the figure at a fixed audit budget
  grows by about that much. Runs below give the exact factors.
- **Fan-out does not multiply the charge for a missed block in any form of the bound on `main`.** It raises a
  ceiling in one form, is absent from a second, and is refused by the third, which is the one the figure of record uses.
- **Speculative decoding needs no new proof system but does need three things built** (the draft head in the
  registered circuit, the accept rule, the scheduler's step relation), and vLLM's accept rule changes only how many
  tokens come out a step for greedy requests, but which tokens come out for seeded sampling requests.

### 1. What `main` charges when one request reads K/V rows another computed

**Today nothing lets it happen on the two-stage path** [code]:

- The serve's boundary commitment refuses to start while the scheduler caches prefixes:
  `adapters/vllm/verity_vllm/pipeline/boundary_rows.py:600-603` [in part].
- The replay engine turns caching off: `adapters/vllm/verity_vllm/engine/engine_profile.py:146` and `:151-156`.
- The `two-stage` arm passes `--no-enable-prefix-caching`: `benchmarks/serve_tax/serve_tax.py:237-239` [in part].
- The verifier refuses every two-stage run at its last check: `FORM = False`,
  `verity/protocols/spot/verifier/second_stage.py:86` [in part]. So no two-stage run is audited yet, cached or not.

**What the bound says** [code: `security/Properties/Spot/TwoStageCheck.lean` whole; `verity/protocols/spot/optimize.py:1163-1265`
whole; `verity/protocols/spot/plancheck.py` whole]. The two-stage bound has three forms, and fan-out F enters them
differently:

| Form | What an incorrect block is charged | Where F enters |
|---|---|---|
| Sets' form, uncapped | its own committed output bits, b(T), whoever reads them | nowhere |
| Sets' form, capped (`TwoStageCheckServedSound`) | its committed bits, but all incorrect blocks of one replay block together at most the bits that replay block serves, S | S is read as F·S (`optimize.py:1203`, `served = served * fanout`; the Lean's `ServedCover` with `N` = F) |
| Patterns' form (`TwoStageCheckPatternsSound`) | counts served-output patterns; each replay block at most 2^S of them | refused when F > 1 (`optimize.py:1200-1202`, `plancheck.two`) |

So the answer to "is that the fan-out F": yes, F is the Glossary's term for it ("the most replay blocks one group
joined by such reads holds"), but it is **not a multiplier on the charge for a missed block**. In the capped form it
lifts the ceiling on one replay block's incorrect blocks from S to F·S, and that ceiling can never exceed the uncapped
form. The figure "grows in proportion to F" in no form; how much it grows in the capped form is what run B below
measures.

**The figure of record is in the patterns' form** [code]: `headline.py`'s rows are on the lines `tokens` and
`per_expert`, both `form="patterns"` (`benchmarks/spot_figures/total_cost.py:165-178`), at S = 28,000 bits (875 output
tokens at 32 bits, `headline.py:42`). The cost model hard-codes fan-out 1 (`benchmarks/spot_figures/figure1.py:123`) and
`total_cost.py` passes none. So for a serve where cached rows are shared, `main` has no figure of record at all, rather
than a larger one.

**Nothing on the two-stage path computes F from a served run** [code]. `partition.fanout`
(`verity/protocols/spot/verifier/one_stage/partition.py:291-325`) computes it for the one-stage partition, by joining a
template query's root nodes that read each other's results; `plancheck.two` takes F as an argument. Separately, rule R8
(`shared_inputs`, same file) refuses a run Input that two proof blocks read unless it is registered or tied to the
network period check, and that tie's composition is marked unproved (`security/Properties/Spot.lean:124-136`).

**What the figure of record assumes about the cache** [code: `benchmarks/spot_figures/breakeven/headline.py:319-366`
whole, `sweep.py` whole]:

- Serving has the cache on: a request of the plan is 3,320.5 computed tokens (`reprice.py:510`), which is prompt tokens
  less the hit tokens plus output tokens, (2,235,146 − 259,712 + 186,242) / 651 [arithmetic on
  `benchmarks/serve_tax/prefix_plan.jsonl`].
- The K/V pool is carried across the 600 s edges and its pages are committed again at each edge (a copy block a page
  and a hashing charge).
- Each sampled replay proves again the prefix its request hit: proving is scaled by 1 + 0.000441 + 0.1386
  (`headline.py:334`; `CHARGES = "certified hits, replays reprove hits"`, `:45`).
- So every replay block is closed and F = 1. A probe run of `headline.py rows` on `main` gives row (i) at 14,228.6 GB a
  world year at 10% of compute times 1.05 [modeled; run `r20261112-020641-ccde`, two budgets only], the same figure
  the lane already reports for the tree's own card [relayed].

**F in the three cases you asked for:**

- (i) *One session's chain of turns.* Each turn reads the turn before, so the whole session is one joined group: F is
  the session's request count. On our plan it is 2 (two of 651 requests hit, each reading one earlier request)
  [arithmetic on the plan]. On the 766-request trace the vllm lane served at 128k the lane measured a median of 4 and at
  most 6 under a rule that lets a turn hit only its own conversation [relayed, 9 Nov note]. On AgentX it is the
  session: median 86 requests, mean 251.5, with sub-agents nested inside [relayed].
- (ii) *Stock vLLM's cache, keyed by content and shared by every request.* Any request whose leading blocks match
  any earlier request's joins that request's group, so one shared system prompt joins everything that carries it. No
  bound but the requests served while the block stays in the pool. The lane measured 54 on that 766-request trace, through
  a 23-block preamble [relayed]. On AgentX every request carries Claude Code's system prompt and tool list, so I
  expect one group per machine [guess; not read from the corpus].
- (iii) *The cost model's pool per window.* As the figure of record prices it, F = 1, because replays prove hits again.
  If instead hits were read as committed rows and the pool were emptied at each window edge, a group is at most one
  machine's requests in a window: about 263 at 600 s and 1,577 at 3,600 s at our plan's rate of 651 requests in 1,486 s
  [arithmetic]. That is a cap, not a small one.

### 2. The five lines

1. **Right, with four things to add.** (a) The lines are `engine_profile.py:146` and `:151-156`, and
   `adapters/vllm/README.md:77-78` and `:109-110`. (b) The refusal is on the serving side too
   (`boundary_rows.py:600-603`). (c) The arm is the measured serve, not an audited path: its replay block's interior
   holds only the token selection until a request Definition lands, and the verifier refuses every two-stage run
   (`README.md`, the Replay section; `second_stage.py:86`). The arm also leaves out admission on the certified tick and
   the edge snapshot (`benchmarks/serve_tax/README.md`). (d) The one-stage check "reads a hit" in this sense only: rule
   KvPrefix lets a warm request's Program take rows from a `prefix` input, and `check_hits` holds the engine's hit count
   and each cached block's producer to what the Program declares. It checks that the circuit matches the engine. It
   carries no charge.
2. **The constants are right; the reading is off in three places.** 11.62% is right (259,712 of 2,235,146)
   [arithmetic], but only 2 of 651 requests hit, so this plan says little about caching. 13.86% is the hits' FLOPs over
   the computed FLOPs (12.0% in tokens [arithmetic]), and it is the price of the caching-off case, which the figure of
   record does **not** use: it keeps the cache on and adds the 13.86% to proving. The 0.7% to 4.4% is the hits lost when
   the pool is emptied each window; the figure of record carries the pool instead and pays to recommit it. And the
   circuit that "proves the lookups" proves which blocks hit (the schedule), not the cached rows' values; its 1.10e11
   to 3.05e11 AND gates a window are on top [code, `sweep.py`].
3. **86% and 98% are right [relayed]; "7 to 46" needs a caveat.** 1 / (1 − 0.8567) = 6.98. For AgentX prompt
   tokens alone give 1 / (1 − 0.9831) = 59.2; 46 follows only if output tokens are about half a percent of all tokens,
   and the traffic note gives no means, so I can't check it. I will carry 46 to 59. These are token counts in both
   directions: vLLM's own AgentX run serves 8,340 tokens a GPU-second counting cached ones, which is about 140 to 180
   computed tokens a GPU-second [arithmetic: 8,340 / 59.2 and / 46], below the 4,994 the cost model assumes at 0.4
   utilization. So on that traffic the computed tokens are far more expensive each than the model's average token.
4. **Wrong in its main clause.** The charge for each missed incorrect block does not grow with the number of requests
   a cached prefix feeds (section 1). What grows, in the capped sets' form only, is the ceiling on one replay block's
   total charge. The F values themselves (tens to hundreds on AgentX within a session, no limit under stock vLLM) are
   right.
5. **Mostly right; "needs no change to the proof system" is true and misleading.** The capture map line is right.
   `replay_step.py:78-82` lists speculative decoding as not modeled rather than refusing it; the refusal I found is
   `adapters/vllm/verity_vllm/pipeline/boundary_tables.py:230-231`, which rejects a step with draft tokens. 329 / 145 =
   2.27 [relayed]. Stock vLLM does not turn speculative decoding on by default; it is what vLLM's published GLM runs
   turn on. The proof system needs no change, but the audited path needs three things built (section 3, B). And vLLM's
   AgentX figure uses an *assumed* acceptance length; the sampler at our pin has a `synthetic` mode that accepts drafts
   by a fixed rate whatever the model says (`rejection_sampler_utils.py:583-585` [vLLM source]), so that figure is not
   evidence of what a real draft head accepts on agent traffic.

### 3. The ways I will price

**A. Prefix caching.** Four traffic shapes (our plan; Moonshot conversation; GitHub Copilot; AgentX), hits over
computed r = 0.14, 0.57, 5.9, 45 to 58 [arithmetic, in tokens, from the traffic note's shares; Moonshot and Copilot
use medians or prompt tokens alone where the note has no means]. In FLOPs the lane's own replay of the Moonshot
conversation trace gave 0.626 [relayed, 8 Nov memo], and our plan is 0.1386 [code].

| Way | Serve | Replay | Fan-out | Priced by |
|---|---|---|---|---|
| A1. Caching off (the arm today) | computes every prompt token: 1 + r times the compute | as today | 1 | arithmetic on r; figure from run A at r = 0 |
| A2. Caching on, each sampled replay proves the prefix again (the figure of record's assumption) | stock | proving × (1 + r) | 1 | run A: `headline.py rows` at r = 0, 0.1386, 0.57, 5.9, 45, 58.2 |
| A3. Caching on, cached rows read as committed inputs, charged at F·S | stock, plus committing the rows | no extra proving | F > 1 | run B: capped sets' form at S × F, F = 2 … 252, 10,000 |
| A4. A public cap on how many requests may read one cached prefix | loses the hits past the cap | none | the cap | run B at F = cap, plus the hits lost [relayed for the Moonshot trace; not measurable for AgentX without the corpus] |
| A5. A widely shared prefix proved in full once (or registered) | stock | one full proof a period | 1 for that prefix | arithmetic: one prefix's proving against a period's budget |
| A6. A session (or joined group) as the replay block | stock | a sampled session replays all its turns | 1 | run B: G requests as one block, G = 4, 16, 86 (a rough stand-in: tokens, replay and S each × G) |
| A7. The pool emptied each window of 600 to 3,600 s | loses 0.7% to 4.4% of hits on our plan | lookups proved by the built slice | ≤ a window's requests | `sweep.py` constants; no trace of the other shapes to price the loss |

**B. Speculative decoding.** By reading, no run: what the registered circuit needs (the draft head; the accept
rule; the step relation), what the accept rule does to tokens [vLLM source, read], what the accepted count shows and to
whom, and the figure's move with serving rate (it is proportional to tokens a GPU-second [relayed]).

**C. The other departures of the arm** (the step record, the boundary commitment, the history tables, the canonical
top-k; and what the arm leaves out: the certified tick, the edge snapshot, paced egress, the engine pins): a list
with file and line, and the measured serving cost of each where the lane's serving notes have one [relayed].

### 4. Runs

- `r20261112-020641-ccde`: probe, done, exit 0. Row (i), main's settings, two budgets, all six scopes: 13,961.6
  (world week), 14,059.9 (world month), 14,228.6 (world year), 13,429.7 / 13,528.4 / 13,697.9 (datacenter week / month /
  year) GB at 10% × 1.05 [modeled].
- `r20261112-020726-bb59` (run A) and `r20261112-020801-9d1e` (run B): running on the CPU pools, each a few minutes
  once it holds a slot.
- The driver is `~/.research/notes/lanes/proofs/tools/cache_price.py`. It sets three of the cost model's values from
  outside (`sweep.PREFIX_OFF`, `headline.SERVED_BITS`, the line's form) and edits nothing in the tree. **No code
  changed on this branch.**

### 5. What I would ask other lanes (not sent)

- **vllm:** does the certified serve with prefix caching on that you are rehearsing commit the cached rows, or does
  its replay recompute them? Which hit rule: vLLM's content-keyed one, or per conversation? What F did you see?
- **vllm:** GLM-5.3's draft head: is `num_nextn_predict_layers` 1 a single extra decoder layer over the last hidden
  state, and does the MTP method hand draft logits to the sampler or only draft tokens?
- **network-accounting:** under paced egress on the certified tick, does the number of tokens released at a tick
  depend on how many the engine produced that step?
