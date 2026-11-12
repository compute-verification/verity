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

## Entry 2, Thu 12 Nov 02:35Z: your 02:00Z and 02:22Z sections read; two corrections to what Daniel was told; first prices

**First line, as you asked.** What you told Daniel at 6:00 PM PST is right in two parts: it is not the proof system
(Flock and the sampling take no notice of the cache), and the demo's choice, caching off, does not extrapolate to agent
traffic. Two things to correct to him:

1. **Item 1 is not open. The Lean decides it, and the answer is no:** a pool committed once an execution phase does
   not leave every replay block closed. A request reads a proof block when one of its served outputs reads a value that
   block commits (`blocksTo`, `security/Definitions/Claims/SamplingServed.lean:33-36` [code, in part]). A cached page
   is the committed output of the block that computed it, in another request. Committing the pool fixes the page before
   the coins; it does not change who computed it. The patterns' form, which the figure of record uses, requires that no
   two requests read one proof block and that each request reads in one replay block (`RequestsCover`,
   `security/Properties/Spot/TwoStageCheck.lean:53-62` [code, whole file]); its theorem is proved
   (`twoStageCheckPatternsSound`, `security/Proofs/Properties/Spot/TwoStageCheck.lean:308` [code, in part]). The cost
   model's pool unit (`CacheStep_v3` on `MemCheck_v1` and `BlockKey_v1`) proves which blocks hit, not what the pages
   hold, so it does not close a reader either.
2. **The figure of record does not extrapolate either,** and that is neither the demo nor the proof system. It keeps
   every replay block closed by having each sampled replay prove the cached prefix again (entry 1). That costs proving
   in proportion to 1 plus hits over computed. At AgentX's reuse the same budget buys a figure 44 to 57 times larger
   (table below). So the honest sentence for Daniel is: **not the proof system, not the demo's size in machines, but
   the traffic's shape acting on how the sampling design accounts for shared values; and one design already stated and
   proved on `main` avoids it.**

**The design that avoids it: make the cache's sharing group the replay block.** If the cache is keyed by session
(vLLM's own `cache_salt` request field [memory of vLLM's API; I will check it at the pin]) and a session is the replay
block, every page a request reads was computed inside its own replay block. The block is closed, nothing is proved
twice, and the patterns' form on `main` covers it with the served cap read as the session's (the merged request serves
G·S bits, `N` = 1). The model puts it at about 1.2 to 1.4 times the figure for 4 to 86 requests a session.

**First prices** [modeled: `headline.py rows` on `main` at `be9a3d5669`, row (i), world year, the GB at which the
total is 10% of compute, times 1.05; runs `r20261112-020726-bb59` and `r20261112-020801-9d1e`; only the three values
named in entry 1 were changed from outside; every other input is the plan's]:

| Case | GB a world year | Against no cached reads |
|---|---:|---:|
| No cached reads proved again (r = 0), S = 28,000 bits | 12,552 | 1 |
| **Replays prove hits again, the figure of record: our plan (r = 0.1386)** | 14,229 | 1.13 |
| … at the Moonshot conversation trace's reuse (r = 0.57) | 19,446 | 1.55 |
| … at GitHub Copilot's (r = 5.9) | 83,834 | 6.7 |
| … at AgentX's (r = 45 and 58.2) | about 552,000 and 710,000 [arithmetic from the rows at 9.95% total; the reader failed on a datacenter's week, where the bound is everything served] | 44 and 57 |
| **Session as the replay block**, G = 4 / 16 / 86 requests (tokens, replay and S each × G; a rough stand-in) | 14,767 / 15,595 / 17,787 | 1.18 / 1.24 / 1.42 |
| Patterns' form with S read as F·S, F = 2 / 4 / 8 / 16 / 86 / 252 / 10,000 (**no statement on `main` covers this for F > 1**; I have a proof sketch only) | 14,043 / 14,801 / 15,289 / 15,857 / 19,039 / 24,882 / 99,039 | 1.12 / 1.18 / 1.22 / 1.26 / 1.52 / 1.98 / 7.9 |
| Capped sets' form, which `main` does state for F > 1 (`N` = F), F = 1 to 10,000 | 24.4 million, flat in F | 1,945 |

The datacenter's year is within 4% of the world's in every row, and a week or a month within 2% of the year: at a
fixed share of compute the bound is nearly the same total over any period [modeled, run A, all six scopes for the
first six rows]. The bound falls as 1 over the audit budget in every row (fitted slope −1.0000).

What the table says:

- **Line 4 of your five is half right, and I was too quick in entry 1.** A single incorrect block is charged its own
  committed bits whoever reads it. But a replay block that is incorrect wholesale was charged what it serves, S, and
  with F readers it is charged what they all serve, F·S. That is the growth with F, and in the patterns' form it is mild
  (1.5 times at F = 86), because a wholesale-incorrect block is also the easiest to catch.
- **The capped sets' form is no use here:** it is about 1,900 times the patterns' form before any sharing.
- **Caching off** stays at the first row's figure per request, and pays 1 + r in serving compute: 1.14, 1.57, 6.9, 46
  to 59 times [arithmetic].

**What changes in the final** (your 02:22Z order: 1, 3, 2, 5, 4): caching off priced once; the routes with the
cache on (session as the replay block; pages read as committed inputs under a new statement; prove again) each with
what serving commits against 1% of serving time and what the replay reads; speculative decoding on; AgentX first;
one recommended route with its changes by lane. Still to read: vLLM's `cache_salt` at the pin, the serving notes'
measured rows for the commitment, AgentX's session lengths against a cap.

## Entry 3, 02:50Z Thu 12 Nov (my VM's clock; entry 2's header said 02:35Z and was written at 02:22Z)

Read your 02:20Z correction and 02:28Z section at 02:48Z; following both. The final gains advice by class, the
circuit compiled ahead of time, and a context column at 1,048,576 beside 131,072.

**Your point to Daniel, "a long context and a cache hit are one question": right for the replay, not for the
guarantee.** File and line:

- `RequestsCover`'s first clause (`security/Properties/Spot/TwoStageCheck.lean:58`) needs every proof block a request
  reads to be in one replay block. A request cut into spans that are replay blocks breaks it, as a hit across
  requests does. So the patterns' form on `main`, the form the figure of record is in, covers neither.
- `ServedCover` (the same file, lines 89-95), the capped form's hypothesis, has no such clause. It needs only that
  no two requests read one proof block, and bounds the bits served by the requests reading in each replay block by
  `N` times `S`. A span reading its own request's earlier spans satisfies it with `N` = 1: inside one request the
  capped form charges no fan-out. A hit across requests breaks its first clause unless the readers are one "request".
- `req` is any map from served outputs to a type (`∀ R ... req`, lines 80-82 and 102-103). So naming the session
  as the request needs no new theorem, in either form.
- What I did not check: which form the 128k row's figure is in when its replay blocks are spans. The capped form at
  the plan prices 1,945 times the patterns' form in the cost model [modeled, entry 2].

**#8403** [PR body read whole]: agreed, and it is the charge the figure of record makes (`PREFIX_OFF`). My entries
price it as "prove again": 14,229 GB at the plan, 552,000 GB on AgentX [modeled]. Its own "not covered" matters more
on agent traffic: a hit whose producers ran in an earlier execution phase, "most of a steady-state window's hit
FLOPs". The pool read does not give F = 1 (entry 2).

**New since entry 2:**

- AgentX's dataset card [read whole]: 21,635,381,376 prompt tokens and 106,474,498 output tokens over 98,827
  requests in 393 sessions, so r = 45 cached tokens a computed token [arithmetic]. It is a sample of the largest
  sessions (`--sampling top`, at least 20 requests). Its 98.31% counts blocks "the same session sent earlier", so a
  cache keyed by session (vLLM's `cache_salt`, in the pin) keeps all of it. Block ids are local to a session: the
  corpus cannot say what sessions share.
- Session as the replay block, more sizes [modeled, run `r20261112-022258-7eb7`, world, one year, GB at 10% of
  compute]: 252 requests 22,922; 1,000 requests 33,304; 252 requests with each request's cap at 4,298 tokens 34,427.
  Against 552,000 for proving the prefix again and 14,229 for the figure of record.
- The cost model holds 13.45 GPU-seconds of egress at each edge (`headline.HOLD`): 2.2% of serving time at 600 s
  phases, whatever the cache does [arithmetic from the constant]. The 1% goal needs phases of 1,345 s or longer
  under that rule.

**Running:** `r20261112-024202-2e5d` (AgentX's mean session in the model) and `r20261112-024625-487a` (the AgentX
corpus on a CPU pool: requests and served tokens a session, times between reuses, what splitting a session costs).
Both about 03:15Z. The draft of the final is `lanes/proofs/20261112T0240Z-report-stock-vllm-departures.md`, synced
now; it does not yet hold your 02:20Z and 02:28Z asks.

## Entry 4, 03:18Z: the note holds your 02:20Z and 02:28Z asks; one correction to my entries 2 and 3

The note is `lanes/proofs/20261112T0240Z-report-stock-vllm-departures.md`, synced at 03:16Z. It is complete as a
first draft of `proofs/cached-serving-audit-design`; I will keep checking it until 05:15Z.

**Correction to entries 2 and 3: what is missing is a counting lemma of the plan check, not a theorem of the
guarantee.** [statements read whole; proofs not read]

- The two-stage scope has no clause on what a replay block reads (`InScopeTwo`,
  `security/Definitions/Claims/SamplingTwo.lean:183-185`). A value one replay block reads from another is on the
  writer's boundary, which serving commits (`Prog.Boundary`, `security/Definitions/Claims/Sampling.lean:155-160`).
- The guarantee holds "at every `δ`, input values `a` and split `req`" (`InfluenceTwoPatterns`,
  `security/Definitions/Claims/Guarantees/SamplingTwo.lean:29-35`, proved at
  `security/Proofs/Flock/SamplingGeneric/TwoServed.lean:628`). A request costs at most its served bits, however
  many of the blocks it reads are wrong. `CapTwo` (lines 41-45) has no such clause either.
- Only the plan check's lemma, which turns that sum into a number, assumes each request reads one replay block
  (`RequestsCover`, `security/Properties/Spot/TwoStageCheck.lean:58`, in `TwoStageCheckPatternsSound`, lines 78-85).
- So a hit read from committed rows, and a long request cut into spans, are covered by the guarantee as proved. The
  checked number is not: it needs the count for a request that reads several replay blocks. My argument for it,
  unproved: the worst case is each wrong replay block in a different session, each charged one session's served
  cap, which is the Glossary's F times S.
- I assumed a table read is a gate's read of its writer in the circuit the statement is instantiated at. Not
  verified.

**The read mechanism is on `main`:** history tables, "rows other proof blocks output, committed as one table that
proof blocks read at hidden positions" (`verity/commitments/history.py:1-12`). The verifier computes the root from
the writers' public commit strings. A read's binding is proved (`Hash.Guarantees.HistoryReadBinding`,
`security/Definitions/Hash/Guarantees.lean:156-166`). A table spans phases: a position is written by this phase's
proof blocks or "carried as rows ... of a value the verifier holds (a window edge's rows of an earlier window)"
(`verity/protocols/spot/verifier/lean/Spot/HistoryTable.lean:18-23`). Its capacity is a constant, `2 ^ 17` (line
56), and its property is stated for every capacity (`security/Properties/Spot/HistoryTable.lean:35-42`).

**AgentX, counted over the corpus** [computed, run `r20261112-025248-fa51`; totals equal the dataset card's]:

- 393 sessions, 98,827 requests; 98.31% of prompt tokens are blocks the same session sent earlier, by vLLM's
  leading-run rule; 45 cached tokens a computed token.
- A session: 86 requests at the median, 251 on average; 96,027 served tokens at the median, 270,927 on average, 5.06
  million at most; 2.8 hours long at the median; a request in flight in 7 phases of 600 s at the median.
- 53% of requests exceed 131,072 prompt tokens and 18% exceed 402,240, the pool a rank.
- A pool emptied at each edge loses 439% of compute at 600 s and 123% at 3,600 s, against the model's 4.37% at
  the plan.
- Cutting sessions so that each piece recomputes its context: 83% more compute at 75,250 served tokens a piece,
  0.35% at 2,000,000.

**The routes on agent traffic** [modeled, world, one year, GB at 10% of compute; run `r20261112-030533-082f`]:

| Route | GB | Times the figure of record (14,229) |
|---|---|---|
| Cache off | 12,552, and serving compute 46 times stock's | |
| Recompute the hit in the replay (#8403, the figure of record's charge) | 552,000 | 39 |
| Read, AgentX's mean session one replay block, cap its own served tokens | 24,753 | 1.7 |
| The same under a cap of 2,000,000 served tokens a session | 41,043 | 2.9 |
| Read, a replay block the plan's request, each wrong one charged a session's cap of 220,500 tokens (unproved count) | 24,882 | 1.7 |
| The same at a cap of 2,000,000 | 44,327 | 3.1 |
| Any request reads one pool (capped form) | 24.4 million | 1,700 |

**Your other asks, in the note:** advice by class (section 7: neither route adds to today's 5.6 bits if the pool's
model reproduces vLLM's evictions; else 12 bits a request at AgentX's mean prompt); the circuit compiled ahead of
time (section 8: recomputing a hit needs a circuit that depends on the served record in class G; reading does not,
except through the hit's length; speculative decoding needs GLM-5.3's draft layers, and positions as values so
that accepted lengths stop being a fixed list); context length (section 9: with reads nothing in the replay grows
with context; what contradicts "not a hard limit" is the merging unit that is full at 131,072 keys, the compile
at 788 GiB, and the pool of 402,240 tokens a rank, which a 1,048,576-token context does not fit with or without
the audit).

**Runs cancelled as duplicates, never started:** `r20261112-024202-2e5d`, `r20261112-030122-ede5`,
`r20261112-024625-487a`. Nothing of mine is running.

## Entry 5, 03:49Z: your 03:17Z section read; the line for 07:00Z; the published serve; REPLAY's two figures

The note is synced (notes commit `437b610c1d`); it opens with the line below and has a new section, "Under vLLM's
published serve for eight H200s". No tree code changed.

**The line for 07:00Z.** To prove: `AdmissibleTwoRankinPatterns` without its hypothesis that each request reads
one replay block (`security/Definitions/Claims/SamplingTwoRankin.lean:79-88`); `RequestsCover` then loses its first
clause (`security/Properties/Spot/TwoStageCheck.lean:58`), and a wrong replay block is charged the served cap of the
session that reads it. Its price against today's bound of 14,229 GB (world, one year, audit at 10% of compute)
[modeled]: AgentX 1.7 to 3.1 times, at session caps of 220,500 to 2,000,000 served tokens; GitHub Copilot about 1.2
times [between two modeled rows]; the Moonshot conversation trace not computable, it names no sessions; our plan
0.88 times, 0.99 if its sessions hold two requests. Recomputing the hit, as `main` does: 39, 5.9, 1.4 and 1 times.

**The date: I give none.** My instructions bar me from estimating calendar time, so I size it by what changes.
- One definition loses one hypothesis; the right side of its inequality does not change.
- One proof of 106 lines is redone (`admissibleTwo_rankin_patterns`,
  `security/Proofs/Flock/SamplingGeneric/TwoRankin.lean:267-372`). Two facts replace its use of the hypothesis. A
  request's cost is at most the sum of its costs in each replay block, which `min_sum_le_sum_min` (line 374) and
  `sum_min_inter_le` (line 386) already give for the capped form. And a request's pattern is fixed by its patterns
  in each replay block: if any is heavy the whole is, and light ones record their sets. [my argument, unproved]
- The clause is used once, as `hreq.1` in `twoStageCheckPatternsSound_form`
  (`security/Proofs/Properties/Spot/TwoStageCheck.lean:243-260`). The per-replay-block factor and the check's bound
  on it do not use it. Then the lock's records.
- No ruling: the guarantee's statement, `InfluenceTwoPatterns`, does not change.
- Whether it fits Fri 13 Nov 9 AM PST is settled by one attempt at the restated lemma, which the Lean lane can
  start now; nothing else gates it.

**What that lemma does not give: the count across an execution phase's edge.** `InfluenceTwoPatterns` is one
statement on fixed input values, and phases are joined only in that their misses multiply
(`security/Definitions/Claims/SamplingTwoReg.lean:8-11`). A wrong block in one phase that shapes tokens served in
a later one is charged by neither. Stating that changes what the guarantee says, so it is an ask to Daniel. It is
avoided, not closed, by carrying a session to the phase in which it ends, as a running request is carried today
(`adapters/vllm/verity_vllm/pipeline/boundary_rows.py:5-7`); the coin then waits for the session, 2.8 hours at
AgentX's median [computed].

**Under the published serve** (FP8 KV cache, MTP with five draft tokens, Model Runner V2, prefix caching on):
- **Steps or positions?** The statement reads neither: gates, proof blocks, replay blocks, requests and served
  bits. Positions enter in the verifier's schedule (`Spot/Schedule.lean:1-6`), steps only in the adapter. They meet
  where a run of positions ends inside a step that emitted up to six tokens: the replay re-runs the whole step and
  keeps its own positions. vLLM pads each request's query to include rejected positions (`speculator.py:240-245`
  at the pin), so a step's shape does not depend on what was accepted.
- **The page's element type?** One term reads it, and it can only fall: the bits a lightly wrong block commits
  (`bitsOf`). A session's cap does not. Outside the bound, it moves the circuits: every attention reads FP8 pages,
  cache on or off, and the tree's Definitions take bf16 ("quantised KV not modelled",
  `verity/definitions/numerics/fa2_model.py:286-287`). Not priced.
- **Prices.** Every figure scales by tokens served a GPU-second, not known at five draft tokens (2.27 at three
  [relayed], at most 6). The ratios between routes do not move. Proving a kept token costs one draft-layer row
  more, about 1.3% if that layer is one of 78 [arithmetic; its size not read]. Bytes committed a row halve to
  46,314 B a token [arithmetic].
- **Found in vLLM's source:** the draft's later steps reuse its first step's indexer top-k
  (`mtp_speculator.py:28-34`), and the draft samples with each request's seed and temperature and hands its logits
  to the accept rule (`speculator.py:204-262`). Both must be in the circuit.

**REPLAY's two figures, as I would say them now.** "About 1.5 times today's bound for 86 requests sharing a
prefix" is against the no-repeat figure (12,552 GB) and for a session cap of 86 times ours, 75,250 served tokens;
against today's bound it is 1.34. AgentX's sessions serve more: 1.7 times today's bound at 220,500 served tokens,
3.1 at 2,000,000. "About 1,900 times" for the sets' form is 1,945 times the no-repeat figure, 1,715 times today's
bound.

**Since entry 4, in the note:** the cost model's figure is its column "GB@10% x1.05" (the bound at 10% of compute
times `LAMBDA`), and it charges serving 4.34% of compute at the plan, of which 2.29% is the egress wait; a serve at
1% would leave 8.89% for proving and multiply every figure by 0.62 [arithmetic].

**Running:** `r20261112-032526-949b` and `r20261112-033048-7551`, the same job twice (the routes at week, month and
year, world and datacenter), both still queued. They fill cells the note marks "not run" and change no answer. I
cancel both before my final if neither has started.

## Entry 6 and final, 04:08Z: the line for 07:00Z made exact; the routes at six scopes; nothing of mine is running

Your inbox is unchanged since `1cdc71f2b7`. The note is synced at notes commit `e850c70417`. No tree code changed.
Entry 5's header said 03:58Z; it was pushed at 03:49Z.

**The line for 07:00Z, with what entry 5 left out.** The statement is `AdmissibleTwoRankinPatterns` without its
one-replay-block hypothesis, read with a session as its request. `RequestsCover` loses only its first clause
(`security/Properties/Spot/TwoStageCheck.lean:58`). Its other two stay, and they bind the serve:
- **No two sessions read one proof block** (lines 59-60). So the cache is keyed by session, and a prefix that
  sessions share is recomputed in each or registered; it is not a block that all of them read.
- **Every session that reads a replay block serves the point's one cap** (lines 61-64). One cap for every session,
  or a cap a class, is undecided.

**The proof, step by step** [my argument, checked against the definitions and each step of the present proof; not
compiled]. Write T_r for the proof blocks of a set T that session r reads. The present proof
(`security/Proofs/Flock/SamplingGeneric/TwoRankin.lean:267-372`) uses the hypothesis three times; each has a
replacement in the same file.
1. Take the tuple of per-replay-block patterns from the set, ρ ↦ the pattern of T ∩ ρ, and not from the pattern.
2. The tuple fixes the pattern. If T_r is heavy in some replay block it is heavy, since `bitsOf` is a sum. If in
   none, the tuple records every part and T_r is their union (`biUnion_inter_blocksIn`, line 83).
3. The pattern's bits are at most the tuple's: `lightBits |r| (lightOf |r| U)` is `min(b(U), |r|)`; `b(T_r)` is
   the sum over replay blocks (`bitsOf_eq_sum`, line 96); and `min(Σ a, c) ≤ Σ min(a, c)` (`min_sum_le_sum_min`,
   line 374).
4. Rankin's step, the supremum over a pattern's sets and the sum over tuples stand as written. The right side is
   the same product, which the check already bounds without the hypothesis.

**The routes at week, month and year, world and datacenter** [modeled, run `r20261112-033048-7551`]. GB, the
headline's column "GB@10% x1.05":

| | World: week | month | year | Datacenter: week | month | year |
|---|---|---|---|---|---|---|
| Today's bound (the plan, recompute) | 13,962 | 14,060 | 14,229 | 13,430 | 13,528 | 13,698 |
| The plan with no repeats | 12,318 | 12,404 | 12,552 | 11,851 | 11,938 | 12,086 |
| Read, a replay block the plan's request, a session's cap of 220,500 served tokens | 24,573 | 24,687 | 24,882 | 23,932 | 24,045 | 24,240 |
| The same at a cap of 2,000,000 served tokens | 43,952 | 44,088 | 44,327 | 43,188 | 43,316 | 43,551 |
| Read, AgentX's mean session one replay block | 24,490 | 24,586 | 24,753 | 23,940 | 24,037 | 24,205 |
| The same, AgentX's median session | 18,277 | 18,375 | 18,543 | 17,723 | 17,821 | 17,989 |
| Read, Copilot's session of 31 calls one replay block | 15,813 | 15,911 | 16,080 | 15,257 | 15,356 | 15,524 |
| Recompute at AgentX's ratio, 45 | 541,073 | 545,096 | 552,003 | 318,360 at every budget up to 6% | 524,460 | 529,244 |

A week's figure is 1 to 2% under a year's, and a datacenter's 2 to 4% under the world's. The datacenter's size is
the census's and the execution phase 600 s, as the headline has them. All of it is the model's scaling from one
plan; none is measured.

**Read in the tree since entry 5:** GLM-5.3's config has one draft layer (`num_nextn_predict_layers` 1) and asks
the draft to reuse its first step's indexer top-k (`index_share_for_mtp_iteration` true):
`adapters/vllm/verity_vllm/engine/profiles/hf_configs/GLM53.config.json:20, 202`.

**What I would ask, and of whom** (I messaged no one):
- **lean:** one attempt at the restated lemma, to say whether it fits Fri 13 Nov 9 AM PST.
- **vllm:** which accept method and temperature production uses; what an FP8 page is for GLM-5.3 at the pin
  (which elements, at what scale); tokens served a GPU-second at five draft tokens on AgentX.
- **Daniel, as an ask:** the count across an execution phase's edge, since it changes what the guarantee says.

**Runs.** `r20261112-033048-7551` is done and preserved. `r20261112-032526-949b`, its duplicate, I cancelled
before it started. Nothing of mine is running or queued.

## Entry 7 and final, 04:20Z: your 04:06Z section; what differs at `v0.34.0rc1`; vllm's (a) and (b); two corrections to my entry 5

Entry 6 is no longer my last. The note holds all of this in "At the build `v0.34.0rc1`" (notes commit named at
the end). No tree code changed. Nothing of mine is running.

**Two corrections to entry 5, found by reading the new build.**
- **The draft is an argmax by default, at both builds.** I wrote that the draft samples with the request's seed
  and hands its logits to the accept rule. That is `draft_sample_method` "probabilistic". The default is "greedy":
  the draft token is the draft's argmax and counts as one-hot in the test (`vllm/config/speculative.py:682-688`
  at `v0.34.0rc1`, `:588` at the pin) [read].
- **An FP8 row is not half.** Under `fp8_ds_mla` it is 656 bytes a layer against 1,152 [relayed from vllm's
  reader], so 53,940 B a token in place of 92,628, not 46,314 [arithmetic].

**What I rest on in vLLM, read again at `v0.34.0rc1` (`52aeb5806585`)** [read at the lines named, none run]:
- **Same:** `cache_salt` enters the first block's hash (`vllm/v1/core/kv_cache_utils.py:655-665`). The standard
  accept test: `u = tl_rand32(seed, pos)`, accept when `log p(x) > log u + log q(x)`
  (`rejection_sampler_utils.py:616, 714-716`). The MTP speculator's reuse of the indexer's top-k, moved to
  `spec_decode/mtp/speculator.py:27-34`. A step padded to its rejected positions
  (`spec_decode/target_dependent_ar/speculator.py:144-145`).
- **Differs:** a block's key now covers the tokens after it that the drafter reads
  (`vllm/v1/core/block_pool.py:124, 335-343`), your fact 4. My AgentX ratio of 45 is by the pin's rule; one block
  in a mean prompt of 219,000 tokens moves it under 1% [arithmetic].
- **Differs:** a fourth accept method, "fly", lossy (`speculative.py:98, 610-617`). Refuse it and "synthetic".
- **Differs, and not in your seven:** the pin refuses adaptive verification unless the method is "dspark"
  (`speculative.py:1519` there). At the new build I did not find that refusal, and the class MTP extends takes the
  flag (`target_dependent_ar/speculator.py:51`). It is off by default. Under it the count verified depends on
  confidences and measured step times, so the admitted profile must set it off.
- **Not read by me at the new build:** the scheduler, the FP8 write, the sampler's noise, `trace_replay.py`
  beyond its kernel's signature. Your facts 1, 2, 3 and 5 I rest on as relayed.

**(a) Does the request's block compute the accept test, so the kept count is derived?**
- [design; rule read at both builds; not built for sampling] Yes. The count is then a value and no advice.
- [design] What the block is given: the request's seed, temperature and top-p, as ports of round 1's record
  (case 1 of your ruling). The draft tokens are wires (case 2), because the draft layer's rows are in the block.
  Under the default no draft probabilities are needed at all. Under "probabilistic" the draft's logits row is a
  wire too; it is never a port.
- [arithmetic] A draft token given and not computed would be the prover's choice: 17.2 bits each.
- [design] Not at all K slots. A slot's test reads the target's row of the token before it, which is kept. The
  block needs the tests up to and including the first rejection and the target's rows of kept tokens only. The
  tree's greedy `VerifyBranch_v1` computes all K + 1, "rejected slots included"
  (`verity/definitions/registry/spec.py`).
- [my arithmetic from GLM-5.3's config, products of the linear layers only; AND gates not computed] Cost a
  decode row: the test is one log-sum-exp over the processed logits, one log and one compare, under one token
  select in the census's work model, and an accepted token needs no Gumbel select. The draft is the cost: about
  two draft-layer rows a kept token (2.6%) and one more unembedding and argmax (9.5e8 products, about 3%).
  Together 5 to 6% a decode row; 1.3% a computed prompt token.
- [unread] The token select's and the draft layer's AND counts, so I give no gate figure.

**(b) The step record's counts: registered, or charged?**
- [fact 5 relayed; the rest design] Two counts. The kept count is derived, (a). The scheduled count, how many
  draft tokens the scheduler gave a row in a step, is the server's choice and reaches the tokens.
- [design] A one-request replay needs its own scheduled count, not its batch-mates'. It is a shape: it chooses
  the step's Program. Round 1's record fixes it before the first coin; nothing about it is given at replay.
- [design, from the output-accounting rule] Registered and charged are not alternatives. Round 1's record makes
  the input sound, which is what your ruling decides. A choice of the server that reaches the tokens is still
  charged unless a block or the registered compiler computes it. So: in round 1's step record, and advice at its
  coded length.
- [arithmetic] Raw, log2(6) = 2.58 bits a row a step at five draft tokens. Coded against vLLM's rule, 0 where the
  rule predicts it: the count is K unless the token budget, the context limit or `max_tokens` cuts it.
- [design] The route: the advice code computes the scheduled count from the registered scheduler rule and codes
  only departures, as it derives hits today. That needs each step's kept count in the step record as a shape,
  checked equal to the block's computed count, which is the form the tree has for greedy (`ServeSpec_v1`'s
  schedule `A`, "an equality obligation of the record"). A false count is then a wrong block.
- [design] Why this one: it is 0 bits where vLLM follows its rule, it reuses a form on `main`, and it keeps the
  target's rows to kept tokens. Its price: a step's kept count becomes public structure, as step shapes are
  today (`adapters/vllm/verity_vllm/pipeline/boundary_rows.py:18`). The private way, K + 1 slots every step with
  positions as values, costs up to (K + 1) / (kept + 1) times the target's rows [arithmetic].
- [unread] The scheduler's rule at the new build, and `advice_code/replay.py`.

**Unchanged by the new build:** the line for 07:00Z in entries 5 and 6, the two counts, and every figure in GB.

**Added 04:21Z, read at `v0.34.0rc1` after the lines above** (so two of entry 7's "not read" are now read):
- [read] **The scheduled count.** A decode row is padded to 1 + K tokens or left out of the step, "Prefer to not
  schedule than schedule un-padded", unless the context limit is near
  (`vllm/v1/core/sched/scheduler.py:1362-1380`); the drafts are cut to what was scheduled (`:1026-1037`). So under
  (b)'s route the count costs 0 bits except at the context limit: a row left out is a choice of batch, which the
  advice code already carries.
- [read] **One more switch to pin off:** K can follow the batch's size from a configured table
  (`num_speculative_tokens_per_batch_size`, `scheduler.py:318-332, 1765-1769`). Off unless set. Set, it is a
  registered table and a function of the batch, so still no advice, but a Program a value of K.
- [read] **The noise:** the Gumbel noise has MurmurHash3 helpers and the accept test's `u` keeps Philox, "retained
  Philox helper for rejection sampling" (`vllm/v1/worker/gpu/sample/gumbel.py:9, 67-70, 81-105`). So the accept
  test's draw is the same at both builds, your fact 6, and the token select's Definition is per build, your fact 1.
- [read] Our advice code "refuses what it doesn't model (priority policy, spec decode, …)"
  (`adapters/vllm/verity_vllm/pipeline/advice_code/replay.py`, docstring), your fact 7.
- The note is at notes commit `7977a30cab`. Entry 7's header said 04:20Z; it was pushed at 04:17Z.

**Closing line, 05:00Z.** Your inbox is unchanged since `c5863189b5` (read again at 04:59Z). The note's last
commit is `bdae9ff510`: since the addendum it gained a short paragraph under RESULT on the published serve, the
`cache_salt` citation at `v0.34.0rc1`, and, under "A join I assumed", that its prices need the second of the two
ways your 03:55Z section names for the ruling's fourth case. Nothing of mine is running or queued.
