# REPLAY's entries (to the proofs coordinator)

Branch `cursor/proofs-replay-136d`, cut from `origin/main` `be9a3d5669`. Note:
`proofs/20261112T0217Z-report-cheapest-replay` (first checkpoint synced 02:17Z). Marks: [read] a file at `main`
`be9a3d5669` with its line, or a note with its id; [relayed] another lane's statement I did not check at its source;
[mine] my arithmetic, with its inputs; [guess]. I ran nothing on a node and changed no code.

## Entry 1, Thu 12 Nov 02:32Z (due 03:30Z): every route found, where it is, and which columns the tree fills

**In three lines.** I found 17 routes in five families. Three things your list of five does not have, each of which
moves the answer: (a) the registered compiler on `main` already compiles a class G hit the cheap way, as the request's
Program reading its hit's K/V rows as boundary inputs, the producers' rows (`steps.py:45-46`), so the read
route is not only a cost-model constant; (b) the reason a recomputed hit is dear in class G is on record: a position's
K/V bits depend on the launch it was computed in, so a recompute must rerun each producer's launch [read `proofs/20261107T2045Z-report-lifetime`, section 1;
`steps.py:63-65`], and at 98% repeated tokens the producers are the session's whole history; (c) tonight's serving-cost run shows the two-stage serve's
boundary commitment does not keep up (about 460 s of host thread for a 30 s window of 107 requests, run
`r20261112-013012-03bb` [relayed `vllm/20261111T0131Z-checkpoint-servecost-full`, 02:14Z]), which is today's cost at
serving and is not the K/V's.

### The routes

Columns I can fill from the tree or the notes: 1 what serving records, 2 advice, 3 replay cost, 4 bit-exactness,
5 circuit, 6 guarantee. "M" a measured number exists (run id in the table of entry 2), "m" modeled, "r" read in code
only, "–" nothing found.

| # | Route | Where [read] | Built? | Fills |
|---|---|---|---|---|
| **A. Replay a whole stretch of serving** | | | | |
| 1 | The whole execution phase from an empty engine, every request real | `benchmarks/serve_tax/replay.py:1-19`; `adapters/vllm/README.md:84-89` | yes; M at TP8, 131,072 limit, 11,350 steps | 1M 2r 3M 4M 5r 6r |
| 2 | From an edge snapshot (KV pool, scheduler, runner, generators) to the next edge | `benchmarks/serve_tax/restore.py:1-18`, `boundary.py:1-27`, `tp_restore.sh:1-22`, `tp_restore.py` | yes; M at TP8, 40,960 limit | 1M 2r 3M 4M 5r 6r |
| 3 | From an edge by re-adding each request with its K/V pages from host copies (no runner state) | `rebuild.py:4`, `:10-33` | yes; M | 1M 3M 4M |
| 4 | From any step inside a phase, from a per-step mirror of the pages ("replay over the request's lifetime") | `rebuild.py:5-6`, `:34-43`; pricer switches `sweep.py:99`, `:361`, `:406` | replay side yes on one GPU; the serving side (pages copied and hashed as written) is not built | 1m 3M(model on a measured trace) 4M(one GPU) |
| 5 | One replay engine kept up and fed replay block after replay block | no script [relayed `vllm/20261111T0837Z-report-replay-how-and-cost`] | no | 3: engine start M (131.5 to 140.3 s) |
| **B. Replay one request, the others as stand-ins of their shapes** | | | | |
| 6 | Shape-only: batch-mates are zero dummies of their recorded shapes; the target's hit pages loaded from serving's bytes | `rebuild.py:7-8`, `:44-53`, `:62-70` (`--shape-only RID --dummy zero`, `--served` at TP) | yes; M bit-exact at TP8, 40k limit, four targets | 1r 2r 3M 4M 5r 6r |
| 7 | Shape-only with the hit recomputed (#8403) | `rebuild.py:54-61`; `steps.py:60-67`, `:227-228`; `predict/hits.py`; `check/replay/recomputed_prefix.py:1-14`; `request_replay.py:7-10` | yes; M at TP8 for hits of 128 to 384 tokens | 1r 2r 3M 4M 5r 6r |
| 8 | The request on the CPU from its Program (only the rows a read reaches, never its batch-mates) | `check/replay/request_replay.py:1-10`, `:30-41`; `pipeline/interior_rows.py:9-11`; CLI `commit-replay` | yes; M core-seconds a position | 3M 4r 5r |
| 9 | The request alone on a GPU under batch-invariant kernels | `benchmarks/replay_ladder/README.md:34`; `benchmarks/replay_determinism/README.md`; capture map `pins.inventory_runs` | not for GLM-5.3: "no sparse MLA backend supports batch invariance" | 1M(other models: 14 to 31% of serving) 4r |
| 10 | A span of positions inside a request, its earlier positions read from per-layer history tables | `verity/commitments/history.py:1-22`, `:41`; `pipeline/boundary_tables.py:1-30`; `adapters/vllm/README.md:109-110` ("not on this path") | tables and row check on `main`; serve-time tables and the span replay not built | 1M(idle card) 3– 4– 5r 6r |
| 11 | Stand-in interior: one token-select check per served id on the replay's logits | `pipeline/interior_rows.py:13-17`; `serve_tax.py:236-239`; `adapters/vllm/README.md:97-100` | yes (what the `two-stage` arm audits today) | 1M 5r 6r |
| **C. What a hit reads** | | | | |
| 12 | Recomputed inside the hit request's replay block (the same as 7, as a statement shape) | `steps.py:60-67` | yes, with four refusals by name (`steps.py:66-67`) | 3m 5r 6r |
| 13 | Read as boundary inputs, the K/V rows their producers committed | compiler: `steps.py:23-24`, `:45-46` (class G), GP-01's producers per block; one-stage readers `check/replay/cached_prefix.py:1-8`, `check/match/prefix_reads.py:1-14`; two-stage tables `boundary_tables.py:1-12` (a request's own rows only) | one-stage yes; two-stage no: `boundary_rows.py:600-603`, `request_replay.py:39-41` refuse | 1M(tables, idle card)/m(pages hashed as written) 2r 3M(route 6) 5r 6r |
| 14 | Read from a committed pool whose calls the certifier's cache step checks | `verity/protocols/certifier/verifier/cache_step.py:1-40`; `benchmarks/mempool/`; `sweep.py:54-63` | circuit yes (`CacheStep_v1` to `_v3`); it proves which pages hit, not what they hold | 1m 2r 6r |
| 15 | Hits certified by a rule on the certified record (same conversation, registered prefixes, retention by certified events) | notes only [read `proofs/20261108T2305Z-report-cacheopt-memo`] | no | 2r 6m |
| 16 | Hits as advice | `advice_code/codec.py` (v1 "sends none") ; `sweep.py:51` (5,181 bits a window if sent) | superseded: derived at 0 bits by the scheduler replay | 2M |
| **D. Change what the replay block is** | | | | |
| 17 | The cache's sharing group (a session) as one replay block or one request of the split | Lean: `security/Properties/Spot/TwoStageCheck.lean:53-64` (`RequestsCover`: a class may hold `N` requests); CACHE's entry 2 | no code | 3mine 6m(CACHE) |
| **E. Today** | | | | |
| 18 | Prefix caching off | `serve_tax.py:232-239`; `engine_profile.py:146`, `:151-156`; `sweep.py:48` | yes; ruled out as a route (Daniel, 6:02 PM PST) | all, as the baseline |

### What I can and cannot fill, column by column

1. **Recorded at serving.** Measured on GLM-5.3 at TP8: the step record, token commits and edge snapshots
   (`r20261107-155724-3f53`), the two-stage arm tonight (`r20261112-013012-03bb`, in flight), one layer's history
   table on an idle card (`r20261110-185544-dba6`). Modeled only: pages hashed as written. Not measured anywhere:
   history tables or page copies **while serving**, and anything at an agentic load.
2. **Advice.** By class from the codec's own docstring and the certified run (5.6 bits a request, all length terms,
   `r20261111-160319-6d24` [relayed `vllm/20261111T1704Z-certserve-results`]). No route adds a class except
   speculative decoding (accept lengths) and unseeded sampling (seeds). Fillable for every route.
3. **Replay cost.** Measured for routes 1, 2, 6, 7, 8 at our loads (12% repeated or none; 40,960 or 131,072 limit).
   Nothing is measured at 86% to 98% repeated or past 131,072; those cells will be my arithmetic from the measured
   step times, marked so.
4. **Bit for bit.** Measured sizes: 11,350 steps at the 131,072 limit (route 1), 226 steps (route 2), four targets up
   to a 19,170-token prompt (route 6), hits up to 384 tokens (route 7). Never shown: a hit past 384 tokens, a span
   replay, anything with draft tokens, a step above 2,048 tokens against the circuit.
5. **Circuit.** Fillable by reading: the compiler's refusals (`steps.py:57-58`, `:66-67`, `:81-103`), the Definitions'
   pinned ranges, the capture map. Compile cost: only the numbers you gave (28,360 s, 788 GiB) plus the notes' derive
   memory; I will not add one.
6. **Guarantee.** From the Lean I read tonight (`TwoStageCheck.lean`, `SamplingServed.lean:21-36`,
   `SamplingTwo.lean:183-186`) and CACHE's prices, which I build on and do not redo.

### Your list, checked

- **#8403, its limits.** Confirmed [read `gh pr view 8403`, `steps.py:60-67`]. Two more refusals than you list: a run
  whose row started before the hit or read another history (two requests computed one block in the same step), and a
  resume on the KV.
- **`boundary_rows.py:602`.** Confirmed at `:600-603`. #8897 (open, not marked) deletes that refusal and the flag in
  the two arms; it says itself that nothing of it has been served and that a sampled hit is not replayable on the
  two-stage path until the replay takes #8403's form [read its body].
- **Advice.** "About 5.6 bits a request" confirmed (5.8 on the first try of 240 requests, 5.6 on 2,000).
  "Advice v1 sends no hits" is `main`'s codec docstring. One thing to add: the step compiler says each request Program
  already returns its length rule as its last check (`steps.py:72-73`), so the lengths are advice because the
  certifier's compile slice holds no stop rule, not because no circuit has one.
- **Speculative decoding.** Confirmed (capture map line 74; `boundary_tables.py:230-231`; `sampler_capture.py:51-53`).
  Two sources disagree on the draft length and I read neither: VLLM-377 says vLLM's post of 23 Oct uses three draft
  tokens; vllm-benchsweep on VLLM-339 (21:43Z) says the recipe page's guide text for eight H200 uses "MTP, 5 tokens"
  and its command builder leaves it off. Both [relayed].
- **"Stock settings".** Class G serves at `CLASS_G_SERVED`: steps of 2,048 tokens and prefill chunks of 256 rows
  (`engine_classes.py:37`), where stock vLLM steps 8,192 tokens and has no chunk threshold. The compiler refuses a
  step above 2,048 tokens by name (`steps.py:57-58`). So the circuit known ahead of time covers a serve that is not
  stock in two engine limits. This is a departure nobody has priced as serving time; I will put what the notes
  measured (mean lifetimes 40.0 s at the default against 79.1 s at the registered limits, two different runs [relayed `vllm/20261111T0837Z-report-replay-how-and-cost`]) in the table.
- **131,072.** Found so far: `CLASS_G_LIMITS` and `CLASS_G_SERVED` (`engine_classes.py:34`, `:37`); a history table's
  tree, `CAPACITY = 1 << 17` (`verity/commitments/history.py:41`), and a Lean witness "at the pinned capacity, `2^17`"
  (`security/Proofs/Hash/HistoryWitness.lean:48`, `:70`); the pinned ranges of the long-row Definitions
  (`boolean_dsa.circuit-check.py:13` `N=1..131072`; `boolean_dsa_topk_mask.circuit-check.py:14-37`
  `N=4096..131072 step 2048`; `dsa_topk_reads.circuit-check.py:41`; `stop_check.circuit-check.py:24`). Not a limit of
  the model: `max_position_embeddings` is 1,048,576 (`hf_configs/GLM53.config.json`), and the recipe page serves
  262,144 [relayed, VLLM-339 21:43Z]. The one physical limit I found: the KV pool held 401,408 tokens in the stock run,
  three prompts of 131,072 [relayed, same comment, `server.log:249`]. The section in the final has each place and its
  cost.

### Next (06:00Z entry)

The table, six columns for each route, AgentX's shape first. Reading left: the lifetime and 128k coverage notes for
the span and the pages' sizes, the carry spec, CACHE's final (05:15Z), `origin/cursor/proofs-inbox:inbox/replay.md`
at each hour.

## Entry 2 and final, Thu 12 Nov 03:06Z (entry 2 was due 06:00Z, the final 16:00Z): the table, the recommended route, what is undecided, the context limit, the message for Daniel

**Why the final is early.** I read vllm's 02:50Z on VLLM-377: Daniel wants the goal done by Friday morning and vllm
asks for the route to be chosen there at 07:00Z, with your word by 04:00Z on whether that can hold. This entry is
complete for that choice. Three inputs arrive after it; a one-off timer wakes me at about 05:45Z to read them, and
what they change goes in an addendum entry below this one:

- `r20261112-025327-f570`, the measured cost of the step record and the canonical top-k on this model at
  saturation (ends about 05:20Z to 05:35Z by its launcher's guess, no later than 06:15Z);
- CACHE's final (05:15Z), for column 6's prices on the four traffic shapes;
- vllm-prodserve's settings table and accept rule (05:30Z on VLLM-377).

**Read since entry 1:** your 02:28Z section (the context column is 1,048,576 beside 131,072; your point checked below
with file and line), CACHE's entry 2 (`cursor/proofs-cache-7c3e` at `8dfb1e1dbe`), vllm's
`20261112T0229Z-report-cachedserve-load`, the goals' texts, vllm-prodreplay's answer on VLLM-377 (02:48Z), the end of
tonight's serving-cost run `03bb`, and vLLM's rejection sampler, draft layer, request and step defaults at the pin
`d9105ea80`. The note `20261112T0217Z-report-cheapest-replay` holds the same text. Nothing was run and no code
changed.

**Revised since `c348d79df3` (02:54Z):**

1. The Lean on `main` does cover a replay block that reads another's committed values, in the sets' form, at about
   1,900 times the figure's bound; it is the patterns' form that does not. I had written "refuses".
2. A recomputed hit re-runs the conversation so far, the earlier requests' decode steps included: vLLM caches the
   pages a request fills while decoding.
3. The ratios of replay to own compute are a request's (18 and 291 times); I had set a compute-weighted figure
   against an unweighted one.
4. From vllm-prodreplay's reading of the records: the 131,072 run's replay ratio is 1.35 by its record and 0.87
   less the time it spent writing values; a whole-run replay reproduced a 128,960-token hit by re-running its
   producer; the CPU replay has never run at full depth; the table of the indexer's keys is not built.
5. Tonight's two-stage arm ended: 194 of 3,272 requests committed, 7% of output rate over three repeats.
6. The longest hit read back from serving's pages is 384 tokens (6 pages), not 256.
7. Added: row 16 against CACHE's route; the grain of the page commitment; who is charged across execution phases;
   the draft layer; whose each 131,072 is; the design sentence on `main` that this route changes
   (`commit/history_tables.py:22-26`); the message for Daniel, 298 words.

### The answer


1. **The cheapest replay the tree's parts make.** The replay block is the sampled request's own computed positions
   (for a long request, a span of them). The prover re-runs only the engine steps those positions ran in, with every
   other request a zero stand-in of its recorded shape. Everything the positions did not compute, the cached prefix
   and the request's own earlier positions, it reads from KV pages that serving committed as it wrote them. Row 16.
2. **Why it is cheapest.** It is the only route whose replay, interior commitment and proving do not grow with the
   share of repeated prompt tokens (12% to 98%), and grow with the context (131,072 to the model's 1,048,576) no
   faster than serving's own cost a token does: what it recomputes is what the request itself computed, and a
   position's index scores every earlier key in the engine and in its proof alike. Recomputing the hit, which is what `main` does, multiplies the
   tokens replayed, committed and proved by 1.13 at our plan's 11.62% repeated and by 7 to 59 at 85.67% to 98.31%
   [mine: 1/(1 − share) of prompt tokens; CACHE's 6.9 and 46 to 59 count FLOPs and outputs]. In this engine class
   the replay must also re-run every step that first computed a page of the prefix, the earlier requests' decode
   steps included: for an agentic conversation, the conversation so far.
3. **What it costs at serving** over stock vLLM: a hash of each KV page as it fills and a host copy kept until the
   coins, modeled at 0.036% to 0.045% of one card's time, **never run beside serving**; the step record and the
   token ids, measured inside noise. It does not grow with the repeated share: only computed tokens write pages. The
   table builder the tree has today is too slow for this (2% to 4% of serving GPU time by my arithmetic). Two things
   outside the replay decide whether the whole serve is under 1%: the certified tick, which removes the scheduling
   advice and cost 1.0% of output rate in the one pair of runs there is (1.6% modeled), and the two-stage boundary
   commitment as built, which in tonight's serving-cost run committed 194 of 3,272 requests and cost 7% of output
   rate over three repeats (row 10).
4. **Advice.** 0 bits for cache hits, admission and release buckets under the certified tick; 5.6 bits a request
   measured for the terminator and the output length, 0 once the verifier holds the stop rule each request's circuit
   already ends with. Speculative decoding adds the accept lengths, at most 2 bits a step at three draft tokens, until
   its accept rule is in the circuit.
5. **What is missing.** (a) The guarantee: the form of the bound our figure uses needs each request to read in one
   replay block, so it does not cover a replay block that reads another's committed values, a cache hit and a long
   request's earlier span alike; the form on `main` that does cover it prices about 1,900 times higher [m, CACHE].
   (b) The request's circuit on the two-stage path: today's replay block is a stand-in of token selects. (c) The
   draft layer and its accept rule: not built. (d) Measurements at size: no hit past 384 tokens has been replayed
   this way (a whole-run replay reproduced a 128,960-token hit by re-running the request that wrote it), nothing has
   run at an agentic load, and no page commitment has run beside serving.
6. **Against CACHE's route** (a conversation as one replay block, row 14): that route is on `main` and cheaper in the
   bound (1.4 times at 86 requests against 1.5), but its replay block for AX's median conversation holds 142,016
   positions or more, the size the 128k goal already found too large for one replay block. At agentic lengths it
   needs spans too, and a span reads committed values, which is the statement row 16 needs. So the long context
   forces that statement, and with it the cache hit costs nothing more.

### How to read the table

**States.** [M run] measured, with the run; I read it in the note named, not in the run's own record. [m] modeled,
with its inputs. [mine] my arithmetic, with its inputs. [read file:line] code or Lean. [relayed] told to me, not
read. [argued]. [guess]. "Not measured" means I found no number.

**Configurations.** Every number is GLM-5.3 with FP8 weights, tensor parallel over eight H200 (node A), vLLM
`d9105ea80`, compiled (torch.compile and CUDA graphs), chunked prefill, async scheduling, unless the cell says
otherwise. No run has draft tokens.

- **P40**: context limit 40,960; the production plan (ShareGPT, multi-turn, 3% long prompts of 8k to 32k tokens);
  saturated; prefix cache on; canonical DSA top-k.
- **P128**: context limit 131,072; the 651-request plan (`benchmarks/serve_tax/prefix_plan.jsonl`: 11.62% of prompt
  tokens repeated, largest prompt 130,808 tokens) at 0.4 requests a second; prefix cache on. Served twice: at vLLM's
  defaults, steps of 8,192 tokens and no cap on a prefill chunk (`r20261107-213820-c343`), and at the registered
  engine limits, steps of 2,048 tokens and prefill chunks of 256 rows (`r20261108-195819-b00b`).
- **Chat**: 1,024 short prompts a pass, 384 in flight (`r20261112-013012-03bb`, tonight, ended 02:35Z).
- **AX**: AgentX's shape: median prompt 142,016 tokens, 98.31% of prompt tokens repeated within the conversation,
  median output 444 tokens [read `proofs/20261111T2108Z-report-traffic-published-traces`]; median 86 requests a
  conversation [relayed, CACHE]. **No run of ours is at this shape**: every AX cell is my arithmetic from the
  other configurations.

**Two readings of "replay cost".** Over the steps a replay re-runs, replay GPU-seconds are about serving's: 0.83
times in busy GPU-seconds (317 against 381, ±12%) and 0.85 in wall seconds over 226 steps at P40 [M
`r20261107-125416-f460`, read in `vllm/20261111T0837Z-report-replay-how-and-cost`]; over 11,350 steps at P128 the
record's own ratio is 1.35 in wall seconds (2,022.5 against 1,495.4), or 0.87 less the 716.0 s the replay spent
writing three steps' values [M `r20261108-203339-cf6c` against `b00b`; relayed, vllm-prodreplay on VLLM-377, 02:48Z,
who read the records]; 0.68 to 0.73 for one request with stand-ins (96 to 113 s against 141 to 158 s served for the
same 150 to 249 steps) [M `r20261110-063224-38ef`; relayed, the same comment]. That ratio does not separate the
routes: every replay on the engine does the whole batch's work for the steps it covers. What separates them is
**which steps a sampled request makes the prover re-run**, against the request's own share of serving. On P128 a
request's own compute is 17.6 GPU-seconds on average, its lifetime's steps 320 (18 times), and every 600 s execution
phase its lifetime touches 5,120 (291 times) [m, same note and `proofs/20261107T2045Z-report-lifetime`: lifetimes
of `c343`, eight cards times the time, at a replay factor of 1.0, a request; weighted by compute the two are 458 and
5,258, and the cells below quote those]. Column 3 gives that second reading.

**What a request's bits depend on.** GLM-5.3 at TP8 is not batch invariant. A position's values depend on its step's
shapes: the step's token count padded to its capture size, its count of sequences, every row's key count, and which
rows are prefills [read `vllm/20261110T1946Z-report-vllm270-answers`, section 13]. They do not depend on the other
requests' values: four targets replayed bit for bit with every other request zero [M `r20261108-080145-ef0f`]. So a
replay on the engine must reproduce the shapes of the steps it runs, and nothing else of the other requests. This is
also why a recomputed prefix is dear here: each stretch of it is recomputed in the shapes of the step that first
computed it (`steps.py:63-66`).

### The table

One row a route. The first table holds columns 1 to 3, the second columns 4 to 6, same rows. Rows 1 to 15 are in the
tree or the notes; 16 is built from their parts; 17 is the floor and is not built. `steps.py` is
`adapters/vllm/verity_vllm/program/predict/steps.py`; `rebuild.py` is `benchmarks/serve_tax/rebuild.py`;
`TwoStageCheck.lean` is `security/Properties/Spot/TwoStageCheck.lean`.

#### Columns 1 to 3: at serving, advice, replay cost

**Advice common to every row** (not repeated in the cells), a request, by class, with the prefix cache on [read
`adapters/vllm/verity_vllm/pipeline/advice_code/codec.py:1-55`]:

| Class | Bits a request | What removes it |
|---|---|---|
| Cache hits | 0: the scheduler replay derives every hit, the page pool's eviction included, from the certified record and the registered pool size | done; sent whole they were 1.54 to 1.57 bits [M `r20261108-050523-0cb3`] |
| Admission | 0 under the certified tick. Without it: the `arrived` decisions and the order's deviations, 5,563 bits in the largest 600 s phase of the no-tick corpus trace [M `r20261111-083805-5b6d`, read in `vllm/20261111T0814Z-checkpoint-advicezero`]; a request, on GLM-5.3 with the cache on: not measured | the certified tick. The 0 is the codec's: no compile slice holds the tick's rule yet |
| Release buckets | 0: certified | done |
| Terminator | 1 (how the request ended) [M `r20261111-222159-f3f1`: 2,000 of 11,186 bits, 2,000 requests] | the verifier holding the stop rule, which each request's Program returns as its last check (`steps.py:72-73`) |
| Lengths | 4.6 (the output length; 9,186 bits over the same 2,000 requests, outputs up to 256 tokens, cache off) [M same run]; 8.5 with the terminator at outputs up to 1,024 [M `r20261108-214658-2b1f`] | the same |
| Aborts | none: a client's abort cannot reach the engine behind the network certifier [read `vllm/20261112T0229Z-report-cachedserve-load`, item 5] | n/a |
| Seeds | 0 for a request that sends its seed; one without a seed is refused under the tick [read `vllm270-answers`, section 13] | a registered rule deriving the seed from the certified request |
| Routing | none on one node; nothing in the tree for several | not designed |
| Speculative accept lengths | no field: no draft token is served | the accept rule as a check in the request's Program (undecided, item 7) |

| # | Route | 1. What serving records or commits beyond stock, and its cost against 1% | 2. Advice beyond the common list | 3. Replay cost for one sampled request; how it moves with the repeated share and the context |
|---|---|---|---|---|
| 1 | The whole execution phase from an empty engine, every request real (`benchmarks/serve_tax/replay.py`) | The step record: −0.3% [−3.3, +2.6] of output rate at saturation, 3 repeats, P40 [M `r20261107-155724-3f53`, arm `record`]. Token ids, 4 bytes a token: 528,448 B for a 131,072-token request; its host thread 81.3 to 89.8 s a phase of one such request, no GPU time seen [M `r20261109-220303-3352`, `r20261109-223637-4078`]. | none | Every step of the phase: 5,258 GPU-s on P128 at 600 s phases [m]. Every hit on a page older than the phase is recomputed: 4.37% of computed FLOPs at 600 s on P128 [m `sweep.py:63`]; at AX each conversation's whole prefix once a phase, 586 GPU-s for a 142,016-token one [mine: 4.2 GPU-ms a token, from 546 GPU-s for one request of 129,024 prompt and 1,024 output tokens alone, `r20261111-141333-8d9d`]. Grows with both. |
| 2 | The phase from a snapshot at its edge (`restore.py`, `tp_restore.sh`) | Row 1's, plus at each edge a pause to hash every rank's page pool and copy one: 0.984 s median, 1.06 s at most over 21 edges, 298.1 GB hashed and 37.26 GB copied, P40 [M `3f53`, arm `boundary`]; 1.5 s at P128 [read `proofs/20261107T2045Z-report-lifetime`]. That is 0.16% to 0.25% of a 600 s phase [mine]. One edge a 30 s: −0.1% [−2.1, +2.0] [M `3f53`]. The drain before the pause, 2.7 s at the median on P128, is not priced. | none | Every step of the phase, at 0.83 times serving's busy GPU-s [M `f460`]; engine start 131.5 to 140.3 s and snapshot load 52.9 s a replay [M, `replay-how-and-cost`]. 5,258 GPU-s a sampled request on P128 [m]. Flat in the repeated share: the snapshot holds the cached pages. The snapshot is the pool, 37 GB a rank, at any context. |
| 3 | From an edge, each request added again with its pages from host copies (`rebuild.py:10-31`) | Row 2's, less the runner's state. | none | Row 2's. Its time I did not read. |
| 4 | The request's lifetime only, started inside the phase from pages copied as they were written, every request real (`rebuild.py:33-42`; pricer `sweep.py:361`) | No edge. Each KV page copied to host memory when it fills and hashed: 0.14 to 0.18 GB/s a rank written; the hash 0.036% to 0.045% of one card; host memory 22.5 GB on average and 61 GB at most with the coin 1 s after the request, 60 to 68 GB and 127 to 147 GB with coins at 600 s phase ends [m, `report-lifetime` on `c343` and the P40 trace]. **The serving side is not built; its effect on output rate is not measured.** | none | The lifetime's steps: 458 GPU-s on P128 at vLLM's defaults, 1,160 at the registered limits [m]. The other requests' pages must be loaded: 8.5 GB a rank on average, 24 GB at most [m]. Flat in the repeated share. Grows with context through the load only. |
| 5 | One request, the others zero stand-ins of their shapes, its hit pages read from serving's bytes (`rebuild.py --served --shape-only RID --dummy zero`, `:44-52`) | Row 1's, plus the target's own pages kept and committed (row 4's copy and hash; the measured runs took them from an edge's dump). | none: a stand-in keeps its lengths, arrival, hits and finish (`rebuild.py:45-46`), which are the schedule, and the registered compiler computes the schedule from the certified record | The lifetime's steps: 19 to 32 s on eight cards from the edge to the target's finish, targets of 18 to 65 output tokens, P40 [M `r20261108-080145-ef0f`, `-084116-2a14`, `-085223-c677`, `-090249-b2cd`, read in `proofs/20261108T0245Z-report-shape-replay`]. Nothing of the others is loaded. Flat in the repeated share: a 256-token hit was 4 pages loaded. With context the target's pages loaded grow: 92.6 KB a token a rank [mine: the pool's 37.18 GB over 401,408 tokens], 12.1 GB at 131,072. |
| 6 | The same with the hit recomputed (#8403: `rebuild.py --recompute-hits`, `compile_steps(recompute_hits=True)`). **What `main` does.** | Row 1's only. Serving is unchanged. | none: the compiler takes each recomputed stretch's step from the schedule | Row 5's plus the hit. Measured only where the hit is small: 96, 109, 113 s against 107 s with serving's pages, hits of 128 to 384 tokens [M `r20261110-063224-38ef`]. Tokens recomputed: 13.88% more FLOPs on one P128 record (#8403's body); 7.0 times the request's own tokens at 85.67% repeated, 59 times at 98.31% [mine]. Bit for bit, the hit is recomputed in every step that first computed a page of it (`rebuild.py:58-60`), and vLLM caches the pages a request fills while decoding too [read vLLM `vllm/v1/request.py:265-281`]. So at AX the replay re-runs the conversation so far: 85 earlier requests of 444 outputs at the medians are 37,740 decode steps, 44,000 to 47,000 GPU-s at Chat's step time [mine; AX's step time is not measured], about 85 times row 16. Prefilled once alone, which is not bit for bit, the prefix is 586 GPU-s [mine]. Grows with both. |
| 7 | The request on the CPU from its Program (`check/replay/request_replay.py`) | Row 1's. | none | 290 core-seconds a prefill position a rank and 770 a decode step [M, a fit of two requests, read in `vllm/20261109T2018Z-report-128k-path`]; 84,500 core-hours for a 131,072-token request over 8 ranks [scaled by that note]. Only the request's own rows. |
| 8 | The request alone under batch-invariant kernels | 14.1% to 31.1% of output rate on three other models on one H200 [m, card `serve_tax.glm-5.3.json`, `tokens_bi`]. None exists for GLM-5.3's sparse attention. Over 1%. | none, and the schedule stops mattering | The request's own tokens. |
| 9 | A span of positions inside a request, earlier positions read from per-layer history tables (`verity/commitments/history.py`, `pipeline/boundary_tables.py`) | One table a request a layer: 176 MB a layer and 13.7 GB over 78 layers for a 131,072-token request; 0.28 to 0.30 s a layer on one idle H200 [M `r20261110-185544-dba6`]. That is 0.17 to 0.18 GPU-ms a token over 78 layers: **2.3% to 4.3% of serving's GPU time** (against 7.5 GPU-ms a token on P40's mix and 4.2 for a long prefill), and 32% to 34% of a long prefill's wall time if the step waits on it [mine]. The builder runs when the phase closes, for finished requests only (`boundary_tables.py:1-2`, `:14-17`). While serving: not measured; tonight's arm asks for one layer and builds none [read `vllm/20261111T0131Z-checkpoint-servecost-full`]. | none | Not measured: no span replay exists. The span's steps only [argued]. |
| 10 | Stand-in interior: one token-select check for each served id, on logits a replay produced (`pipeline/interior_rows.py:14-16`) | Today's `two-stage` arm: the boundary commitment by size class, prefix cache off. Its host thread took 458 s for the first 30 s phase (107 requests in 40 classes) and 759 s for the second, and committed 194 of 3,272 finished requests, none with tables: it does not keep up. Serving cost +7.16% of output rate over 3 paired repeats (−2.07%, +7.16%, +12.53%, rising rep by rep; no interval), Chat [M `r20261112-013012-03bb`, ended 02:35Z, read in `servecost-full`, 02:31Z and 02:41Z]. | none | Whatever replay produces the logits (rows 1, 2, 5). Interior: 309,760 bytes an output on rank 0 [M, `128k-path`]. |
| 11 | The hit held to a committed pool by the certifier's cache step (`verity/protocols/certifier/verifier/cache_step.py`) | A pool's commitment a phase; the pricer's pool per phase recomputes 4.37% of P128's FLOPs at 600 s [m `sweep.py:63`]. | none: it is how the 0 bits for hits are held in a circuit | Not a replay. It proves which pages a request hit, not what they hold [read `cache_step.py:1-40`; CACHE's entry 2]. Pairs with row 5 or 16. |
| 12 | Hits allowed only by a rule on the certified record (same conversation; registered prefixes) | Nothing at serving; hits the rule forbids are recomputed. Notes only [read `proofs/20261108T2305Z-report-cacheopt-memo`]. | none | As the route it pairs with. |
| 13 | Hits sent as advice | nothing | 1.54 to 1.57 bits a request; 5,181 bits a 600 s phase (`sweep.py:51`) | None of its own. Superseded: the scheduler replay derives hits. |
| 14 | A conversation as one replay block | Row 1's, and the cache keyed by conversation: vLLM's `cache_salt` enters the first page's key [read vLLM `d9105ea80`, `vllm/v1/core/kv_cache_utils.py:605-613`]. | none | The conversation's computed tokens once, no hit recomputed. But its interior is every request's: every Definition's ports are 58.3 MiB a position a rank [M, `128k-path`], 7,462 GiB a rank for 131,072 positions [scaled by that note], against 721 GB of disk on the largest CPU node. Not feasible as one interior. |
| 15 | Prefix caching off (today's two-stage arm) | Serving recomputes every repeated token: 13.86% more FLOPs on P128 [m `sweep.py:48`], 7.0 and 59 times at 85.67% and 98.31% [mine]. Ruled out by Daniel. | none | The request's lifetime, which is longer. |
| **16** | **Recommended: row 5's replay, cut to a span as in row 9, started as in row 4. The sampled request's own computed positions, in their served step shapes with zero stand-ins, started at its first step on an engine kept up, reading the cached prefix and its own earlier positions from KV pages committed as written** | Row 1's, plus a hash and a host copy of each page as it fills. The hash: 0.036% to 0.045% of one card [m, row 4]; by the 117 GB/s a card the edge hash ran at [M `3f53`: 37.26 GB in 318 ms], 0.019% of serving's GPU time, or 0.15% of a long prefill's wall time if the step waits on it [mine: 92.6 KB a token]. At AX 263 MB a request a rank: 2,844 computed tokens [mine]. The latent rows are the same on every rank (`boundary_tables.py:8`), so one rank commits them. Does not grow with the repeated share. The host copy holds a page until the coins of the last phase a request read it in: at most the pool, 37 GB a rank, plus a phase's writes, 84 to 108 GB at 600 s [mine, from row 4's rates; argued]. With today's table builder instead: row 9's 2.3% to 4.3%. **Not built; not measured beside serving.** | none | The steps of the span: for a whole request, row 5's 19 to 32 s at P40. That is its own compute times about the rows in its steps: 18 times on P128 [m: 320 against 17.6 GPU-s a request]. At AX at least 444 decode steps: 65 to 69 s, 519 to 550 GPU-s, at Chat's 146 to 155 ms a step [mine; AX's step time is not measured], plus loading 13.2 GB a rank of prefix, 19 s at the snapshot load's rate [mine]. Flat in the repeated share. With context: tokens recomputed flat, pages loaded grow. |
| 17 | The floor: the request's rows alone on a GPU, each kernel forced to the settings its served step chose | Row 16's. | none | The request's own compute, about 1 time. Not built: it needs a GPU form of the row kernels the CPU replay evaluates (`check/replay/evaluate.py`), each taking the step's shape as a setting. [argued] |

#### Columns 4 to 6: bit for bit, the circuit, the guarantee

**Common to every row that replays on the engine**, what keeps a replay bit for bit [read `adapters/vllm/README.md:
90-94`, `shape-replay`]: the recorded schedule, every row of every step; the canonical DSA top-k
(`VERITY_DSA_TOPK=canonical`); the engine's environment (`engine/env.py`: Inductor's reduction tiling fixed,
`PYTHONHASHSEED=0`, request ids not randomized); the served seeds. **What breaks it:** one extra 64-token row at the
head of the queue changed the target's first token [M `2a14`]; a request without a seed, whose sampler advances the
generator 32 draws for every row of the batch, so even a finished batch-mate's extra row under async scheduling must
be re-run; a stand-in value that is not finite. **Never run:** a step with draft tokens (`boundary_tables.py:230-231`
refuses one); a replay at an agentic load; a start inside the phase at TP8.

**Common to every two-stage row:** the verifier refuses every two-stage run at its last check, `FORM = False`
(`verity/protocols/spot/verifier/second_stage.py:86`), until the form's proof lands.

| # | 4. Why bit for bit; what the tree has shown, at what size | 5. What it needs from the circuit ahead of time | 6. What the guarantee charges |
|---|---|---|---|
| 1 | It is the serve again. Shown: all 11,350 steps and 184,922 tokens, 8 of 8 ranks, P128 at the registered limits [M `cf6c`], among them a 128,960-token hit replayed as a hit, its producer re-run too [relayed, vllm-prodreplay, from `b00b`'s step trace]. It starts only where the engine was empty (`replay.py:62`, `:65` [relayed, the same]): under traffic that never drains, its block is the server's whole life. | The whole phase's Program if the replay block is the phase. One 131,072-token request alone: 28,360 s and 788 GiB on one rank [relayed, the brief; the 788.1 GiB read in `128k-path`]. | A phase as the replay block is closed only if nothing is carried in. Its interior is every request's: not feasible. |
| 2 | The restored state is checked leaf by leaf. Shown: 226 steps, 329 requests, 50,517 tokens equal, P40 [M `f460`]. | As row 1. | As row 1; the snapshot's pool is a committed input every request of the phase reads. |
| 3 | Shown on one process (`rebuild.py:10-31`); sizes not read. | As row 1. | As row 1. |
| 4 | Start state from a per-step copy: shown on one GPU only; the tensor-parallel path has none (`rebuild.py:68`), and it refuses a start step that runs a request the step before finished (`:40-41`). | The request's Program (row 5). | As row 5. |
| 5 | The target's values depend on the others' shapes, not their values. Shown: 4 of 4 targets, among them a 256-token hit loaded as 4 pages, a 19,170-token prompt decoding at 19,360 keys, and a request preempted 3 times; an all-real control of 307 steps; P40 [M `ef0f`, `c677`, `b2cd`]; and a 384-token hit loaded as 6 pages [M `38ef`, read in `vllm/20261110T0704Z-report-replayhit-gpu-verdict`]. Not shown: a hit past 384 tokens in a one-request replay; a 131,072-token target; more than one KV cache group, which `rebuild.py:870` refuses [relayed, vllm-prodreplay]. | The request's Program after its history: the step compiler writes it in this engine class with the hit's rows as boundary inputs, the producers' (`steps.py:23-24`, `:45-46`), and a request carried over a phase's edge reads its own earlier rows the same way (`:46-51`). Refused today: a step above 2,048 tokens (`:57-59`), positions past 131,072 (`:95-96`), a recompute after preemption (`:56-57`), sampling other than greedy or seeded top-p (`:98`). Stock steps are 8,192 tokens on an H200 [read vLLM `vllm/engine/arg_utils.py:2663-2666`]. On the two-stage path the replay block is still the stand-in (`adapters/vllm/README.md:98-100`), and the CPU replay refuses a Program that reads a cached prefix (`request_replay.py:39-41`). Compile: 135 s a Program a rank over a 256-step span, 3.1 to 3.8 core-hours a sampled request [m `sweep.py:52-53`]. | Not closed: it reads values another replay block computed. On `main` the patterns' form, which the figure of record uses, refuses it: each request must read in one replay block (`TwoStageCheck.lean:58`), and the pricer refuses a fan-out above 1 (`optimize.py:1200-1202`). The capped form (`ServedCover`, `:89-94`) covers it when no two requests read one proof block, so with the conversation taken as the request and the cache keyed by conversation; it prices at about 1,945 times the patterns' form [m, CACHE's entry 2]. The patterns' form with S read as F·S: 1.52 times at F = 86, and no statement on `main` [m, CACHE; a proof sketch]. |
| 6 | Each stretch of the hit is recomputed in the step that first computed it, so every kernel choice is the served one (`steps.py:63-66`). Shown: hits of 128, 256 and 384 tokens, prompts up to 910 tokens [M `38ef`]. Refused by name: a producer an earlier execution phase carries in, a stretch whose row started before it, two requests computing one page in one step, a resume (`steps.py:66-67`). At AX most producers are from earlier phases. | The Program with the hit recomputed: built (#8403). For a producer in an earlier phase it would read that phase's schedule: not built. | Closed, F = 1: the form on `main` and the figure of record's. The interior and the proving grow with 1 + hits over computed: 14,229 GB a world year on our plan (1.13 times no reuse), 44 to 57 times at AX's reuse [m, CACHE's entry 2, `r20261112-020726-bb59` and `-020801-9d1e`]. |
| 7 | It evaluates the registered circuit's own rows, so it is exact by definition; it is the reference the others are held to. Never run on a GLM-5.3 request at full depth: 4 of 78 layers, synthetic weights, one rank [relayed, vllm-prodreplay, `r20261108-091453-bf6d`]. | The request's Program, as row 5 or 6. | As the Program it evaluates. |
| 8 | A position's row would not read its batch. Not available for this model. | The uncached request's Program, one a (prompt, output, hit) shape, shared by requests (`steps.py:27`). | As row 5 or 6, by what the hit reads. |
| 9 | Not shown. The table's rows are the engine's cache rows read at the step's slots (`boundary_tables.py:8-12`); the span's replay would be row 5 cut shorter. | The long row's Definitions exist to 131,072 keys (`verity/definitions/registry/boolean_dsa.circuit-check.py:13`, `boolean_dsa_topk_mask.circuit-check.py:14-37`); the table's tree is pinned at 2^17 rows in the Lean verifier (`Spot/HistoryTable.lean:56`) and in Python (`history.py:41`). Missing: the second stage reading a table (`two_stage.py second` passes none [relayed, vllm-prodreplay]); a table of the indexer's keys, 132 B a position a layer, which a long row's scores read at every earlier position [relayed, the same, from #8434's body]; the span's replay Definition; the committed schedule a span's range comes from (`boundary_tables.py:28-29`). A cached request's reads are refused (`pipeline/history_rows.py:94-99`): the design on `main` has them open a table whose rows below the hit the replay recomputes (`commit/history_tables.py:22-26`). | As row 5: the span reads a table another replay block wrote. Its readers are its own request only, so the capped form on `main` holds with `N` = 1; the patterns' form refuses it by the same clause. |
| 10 | The ids are held to logits; nothing ties the logits to the model's layers [read `128k-path`]. | `TokenSelect_v4{V=154880}` and its chunk and combine Definitions: built. | A stand-in: it audits the token select alone. |
| 11 | n/a | `CacheStep_v1` to `_v3`: built; no compile slice runs on GLM-5.3 [read `vllm270-answers`, section 9]. | The pricer's pool unit: 1.10e11 to 3.05e11 ANDs a phase [m `sweep.py:58-59`]. It does not close a reader [CACHE]. |
| 12 | n/a | A rule Definition over the certified record: not built. | Bounds the group a page's readers form, so it caps F. |
| 13 | n/a | none | 5,181 bits a phase charged; ruled "never" [read `cacheopt-memo`]. |
| 14 | The conversation replays as it was served. Not shown. | Every request of the conversation in one replay block. | Closed if the cache is keyed by conversation. The patterns' form on `main` allows `N` requests a replay block (`TwoStageCheck.lean:60-64`): 1.18, 1.24, 1.42 times at 4, 16, 86 requests [m, CACHE]. |
| 15 | Nothing is shared. | Today's. | Closed, F = 1. |
| **16** | Rows 5 and 9 together. Each part is shown at small size on this model at TP8: zero stand-ins, a hit read from serving's pages, a long prompt, a preempted target. Not shown: a start inside the phase at TP8, a span, a hit past 384 tokens, an agentic load. | Row 5's Program, cut into spans that read their history (row 9). Missing: the size class's replay Definition on the two-stage path; a Definition that opens a committed page inside the replay block; captures for steps up to 8,192 tokens and prefill chunks past 256 rows, or the serve stays at the registered limits; the draft layer and accept rule; ranges past 131,072. | Row 5's. The one change to the guarantee's statement: the patterns' form for a replay block read by `N` requests, and for a request that reads in several replay blocks, with `N` public and capped by the conversation. |
| 17 | Would need each forced kernel shown equal to the served one, step shape by step shape. | Row 16's. | Row 16's. |

#### What grows with context, from 131,072 to the model's 1,048,576

Nothing has been served past 131,072; every cell is [mine] or [argued] from the sizes above.

| Routes | At serving | In the replay of one sampled span | In advice | In what the guarantee charges |
|---|---|---|---|---|
| Recompute what came before (1, 6, 15) | Token ids only, 4 B a token: 0.53 MB to 4.2 MB a request. | Everything before the span: 8 times the tokens, and each position's index reads every earlier key, so more than 8 times the work (not measured). | The length terms, 3 bits wider at most, until the stop rule is held. | Closed, F = 1; the interior and the proving grow with the tokens recomputed. |
| A snapshot at the edge (2, 3) | The pool hashed at each edge: set by the cards' memory, not by the context. | The phase's steps; lifetimes lengthen. | the same | As the first row. |
| Read what came before from committed pages (5, 9, **16**) | 92.6 KB a computed token a rank at any context. A table padded to its size class (`boundary_tables.py:20-23`): the tree of a 2^20-row table is 9.6 GiB against 1.2 GiB at 2^17. The engine's own decode row scores keys up to `max_model_len` (capture map `:1533`): stock's cost, not ours. | The span's own tokens: no growth. Pages loaded: 12.1 GB to 97.1 GB a rank, which is past this node's 37 GB pool, so such a request cannot be served here at all. If the replay builds the history's row tables: 24 to 189 GPU-s at today's builder's rate. A sampled row's index proof reads every earlier key: 837,038 ANDs a key [relayed]. | the same | At most the served bits of the requests that read the block: for a span, its own request's, which does not grow with the prompt. No statement on `main` gives that in the patterns' form. |

### The recommended route

**Row 16.** It is the only route whose replay, interior and proving stay flat as traffic goes from 12% to 98%
repeated and grow no faster than serving's own cost a token as context goes from 131,072 to the model's limit, and
it leaves serving stock but for a page hash and a host copy. Each part of its replay is already bit for bit on GLM-5.3 at TP8 at small size, so what it lacks is one
statement in the guarantee and measurements at size.

The changes, by lane, each with what I read that makes it possible:

- **vllm's serve.**
  - Serve the registered class with the prefix cache on: `verity_vllm.LLM(engine_class="G")` turns it off
    (`engine_classes.py:12-13`) though the registration may state it on (`:5-6`), and the boundary commitment refuses
    a caching scheduler (`pipeline/boundary_rows.py:600-603`); #8897 (open, not marked) deletes that refusal.
  - Commit each KV page when it fills, one hiding leaf a page, and keep a host copy until the phase's coins and its
    replays: a plugin on vLLM's KV connector interface, sized at 300 to 500 lines for the copy and 150 to 250 for
    the commitment in `report-lifetime` (section 1), which reads that vLLM's own offloading connector stores each
    page as it is computed. The copy is keyed by the write, not by content: two writes of the same tokens can hold
    different bits. One rank's copy is enough: the latent rows and the indexer's keys are the same on every rank.
  - The grain is the open choice [argued]. A leaf a page a layer is what serving can afford: it is the leaf the
    edge snapshot's kernels hash the pool by, at the rate above (`benchmarks/serve_tax/boundary.py:9`, `:698`). A
    leaf a position, the history table, is what a proof block opens cheaply and what today's builder makes at 2.3%
    to 4.3%.
    The two fit together if serving commits pages and the replay builds the row tables of the sampled request's
    history, with proof blocks that hold a table's rows to their page's leaf: at today's builder's rate that is
    26 GPU-s for a 142,016-token history [mine: 0.18 GPU-ms a token], paid only for sampled requests. Not designed.
  - Make the boundary commitment keep up: its thread builds a layout for every phase and class
    (`boundary_rows.py:574-579`; PRF-456 [relayed]).
  - Give a request without a seed one by rule from its certified bytes.
- **vllm's replay.** `rebuild.py --served --shape-only` is the replay. It needs a start inside the phase at TP8 (the
  per-step copy is one-GPU only, `:33-42`, `:68`), a cut to a span, and one engine kept up and fed replay block
  after replay block (engine start is 131.5 to 140.3 s; no script does this).
- **circuits' Definitions.** The size class's replay Definition for the two-stage path, the model's computation in
  place of the stand-in [read `vllm270-answers`, section 9, step B]; a Definition that opens a committed page or
  table row. For a hit the design on `main` is one sentence away: a cached request's reads open a table "its replay
  builds, its own recomputed rows below C and then a held run of this table's strings"
  (`commit/history_tables.py:22-26`), and a held run is already how a table takes its request's rows from earlier
  execution phases (`pipeline/history_rows.py:18-20`); this route makes the rows below C held runs of the
  producers' tables, not recomputed rows. Also missing: the table of the indexer's keys; the draft layer (the model's config has one, `num_nextn_predict_layers` 1; the capture map does not
  build it, `:74`) and the accept rule (vLLM `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:548-667`);
  captures for the step shapes stock vLLM runs.
- **The certifier's cache step.** Nothing new for this route's replay. For the 0 bits to be held in a circuit: a
  compile slice with the hits (`CacheStep`) and the tick's admission (`ReplayStepTick_v1`, designed, not built
  [read `vllm/20261111T1243Z-draft-tick-replay-step`]) on GLM-5.3, and the stop rule held by the verifier.
- **The guarantee's statement.** The patterns' form with `RequestsCover`'s first clause (`TwoStageCheck.lean:58`)
  relaxed: a request may read in several replay blocks, several requests may read one, and a replay block is charged
  at most what the requests reading it serve. One statement covers a cache hit and a long request's spans. CACHE has
  a proof sketch and the price.
- **The cost model.** The replay card's factor is Moonlight's and unmeasured for this model at 131,072. A sampled
  request's replay is its steps, not its tokens. The hit's reproving (`PREFIX_OFF`, `sweep.py:48`,
  `headline.py:334`) goes; the page commitment and the readers' term come in.

### What I could not decide

Each with the run or statement that decides it.

1. **What the page commitment costs beside serving.** 0.036% to 0.045% of a card modeled as a page hash; 2.3% to
   4.3% of serving's GPU time as today's history tables on an idle card. Decided by the serving-cost benchmark with
   the tables at all 78 layers (`TABLES`, `serve_tax.py:201`, holds 1) and with a page-leaf plugin, at one load.
2. **Whether a long hit read from committed pages replays bit for bit.** Shown at 256 and 384 tokens. Decided by
   `rebuild.py --served --shape-only RID` on one of P128's two follow-ups that hit 131k and 83k tokens.
3. **Whether the patterns' form holds with `N` readers,** and its price at AX. Decided by the Lean statement;
   CACHE's final has the model. The statement must also say who is charged across execution phases [argued]: at AX
   most hits are on pages an earlier phase wrote, which this phase's Program reads as committed inputs (the compiler
   has that form for a request carried over an edge, `steps.py:46-51`), and the readers of an incorrect page are then
   requests of later phases, which the earlier phase's statement does not count: an audit statement holds one
   Program (`security/Definitions/Claims/Sampling.lean:338-343`).
4. **How large `N` is under stock keys.** vLLM keys the cache by content, so any two requests with one first page
   share it. Decided by Daniel or vllm: `cache_salt` a conversation (one request field; hits across conversations
   are lost; AX's 98.31% is inside the conversation) or a public cap in the statement.
5. **The tick or the advice.** 0 bits of admission needs the certified tick: output rate 442.2 against 446.7 tokens
   a second at 1.67 requests a second, first token at 20.4 s against 3.4 s at the median, cache off [M
   `r20261108-200706-1758`'s rows T100 and B100, one run each, read in `vllm/20261108T2205Z-result-certtick-rerun`];
   1.6% in the card [m]. Decided by the same pair with repeats at an agentic load, and by what admission costs
   without the tick on a cache-on trace (vllm's VLLM-368).
6. **Stock step sizes.** The compiler covers steps up to 2,048 tokens and prefill chunks of 256 rows; stock runs
   8,192 with no chunk cap. Mean lifetime was 40.0 s at the defaults and 79.1 s at the registered limits, one run
   each [M `c343`, `b00b`]. Decided by circuits' captures at 8,192, or one serving-cost pair at the two limits.
7. **Speculative decoding.** Nothing is built. Read at the pin: under greedy a draft token is accepted only when it
   equals the target's argmax, so the tokens are the target's; under sampling the test and the resample change which
   tokens come out, from a draw keyed by seed and position. The accept lengths set every row's positions, so the
   schedule needs them: at most log₂(k + 1) bits a step (2 at k = 3) as advice, until the accept rule is a check
   whose outputs a lemma shows unique. Three more things follow. A stand-in's accept length depends on values it
   does not have, so a replay with stand-ins must force each one's from the step record: no code does. The draft
   layer is a 79th layer with its own attention and its own index, run up to k times a step
   (`vllm/v1/worker/gpu/spec_decode/mtp/speculator.py:12-58`; where its cache rows live I did not read), so the
   request's circuit and the committed pages grow by it. And vLLM can vary k with the batch size
   (`vllm/config/speculative.py:479-482`). Decided by vllm's production settings (k is 3 in one source and 5 in
   another, both relayed) and one served run with drafts.
8. **A recompute after preemption** has no Program in this engine class (`steps.py:56-57`), and the engine can
   preempt under the tick [read `cachedserve-load`, item 2]: 65 preemptions in the no-tick row B100, 0 in T100.
   Decided by circuits, or by a pool sized so nothing is preempted.
9. **The canonical top-k's and the step record's cost on this model at saturation.** 0.25% modeled and inside
   noise at P40. Decided tonight by `r20261112-025327-f570` (arms `record`, `canonical-topk`, `no-prefix-cache`,
   Chat, six repeats), which ends about 05:20Z to 05:35Z by its launcher's guess and no later than 06:15Z [read
   `vllm/20261111T2113Z-checkpoint-vllm-servecost-night`, 02:56Z].

### The context limit

**Not hard in the model, and in vLLM only through memory.** GLM-5.3's config has 1,048,576 positions
(`engine/profiles/hf_configs/GLM53.config.json`, `max_position_embeddings`, no RoPE scaling); the published recipe
serves 262,144 [relayed, vllm-benchsweep on VLLM-339]. Where 131,072 is held, and what raising it costs:

| Where | What holds it | To raise it | Whose |
|---|---|---|---|
| The engine class | `CLASS_G_LIMITS` and `CLASS_G_SERVED`, `max_model_len` 131,072 (`engine_classes.py:33`, `:36`); the compiler refuses a position past it (`steps.py:95-96`) | Two constants, after the rows below. | vllm |
| The long row's Definitions | pinned ranges end at 131,072 keys (`boolean_dsa.circuit-check.py:13`; `boolean_dsa_topk_mask.circuit-check.py:14-37`; `dsa_topk_reads.circuit-check.py:41`; `stop_check.circuit-check.py:24`); the block that merges the selected positions fills the largest proof block, 2^27 bits, at 131,072 keys [relayed, the coordinator's 02:28Z] | New pins and circuit-check runs; the merge needs a cut. | circuits (CIR-244) |
| The history table | `CAPACITY = 2^17` rows in the Lean verifier (`Spot/HistoryTable.lean:56`; `check` is `checkAt` at it, `:189-191`) and its Python (`history.py:41`); a witness at that capacity (`security/Proofs/Hash/HistoryWitness.lean:70`) | One constant in each, the vectors and the witness; `checkAt` already takes the capacity. Trees grow with it: 1.2 GiB a request at 2^17 [M, `128k-path`], 9.6 GiB at 2^20 [mine]. | proofs |
| A decode row's index | the engine scores a decode row's keys up to `max_model_len` (capture map `:136`, `:1533`), and the circuit's index logits are 837,038 ANDs a key a row [relayed, CIR-186] | The one cost that grows with a request's length on every decode row, in serving and in proving. | the model's; circuits for the proof's side |
| The compile | one 131,072-token request on one rank: 28,360 s and 788 GiB [relayed] | Does not scale as one Program; spans do (row 9). | circuits and vllm |
| The page pool | 401,408 tokens on this node with the cache in bf16: 3.06 prompts of 131,072, 1.53 of 262,144, 0.38 of 1,048,576 [mine, from the pool's size; the count relayed from the stock arm's `server.log:249`] | **The one physical limit.** A 1,048,576-token request does not fit this node at any setting I read. An FP8 cache, which the recipe uses, is not captured: the capture map records `kv_cache_dtype` auto (`:35`). | vllm (the FP8 cache's capture) |
| `serve_tax.py` | nothing: its base is 4,096 and a run passes its own (`:315`, `:423-428`) | none | n/a |

**What I would set it to: 262,144.** It is the published recipe's limit, it holds AgentX's 256,000-capped copy whole
(largest prompt 255,808), and it is one doubling of each constant. "No restriction" is true of the design in row 16:
nothing in its serving cost or its replay grows with context but the pages loaded and the decode row's index, which
stock vLLM pays too. It is not true of this node, whose pool holds 401,408 tokens.

**The coordinator's point, checked.** "A replay that recomputes everything before its span grows with context, and
one that reads the earlier attention values from a committed table does not; so a long context and a cache hit are
one question": confirmed, with two additions.

- Both read a value another replay block committed, and the same clause refuses both: `RequestsCover`'s "each request
  reads in one replay block" (`TwoStageCheck.lean:58`), over reads that follow gates back to the proof block that
  computed the value (`security/Definitions/Claims/SamplingServed.lean:21-36`).
- They differ in who reads. A span's table is read by its own request, so the block is charged at most that
  request's served bits; a cached page is read by every later request that shares it. In the Glossary's terms
  (`GLOSSARY.md:362-364`) the first is a group of one request's blocks and the second a group of the conversation's.
- "That is what a 128k request's replay blocks already do" is the design's statement, not the tree's: on the
  two-stage path the replay block is a stand-in, the second stage does not read a table, and the table's range has
  no committed schedule to come from (`boundary_tables.py:28-29`). The step compiler does have the form, for a
  request carried over a phase's edge (`steps.py:46-51`).

Of the coordinator's list I checked these: the limits' lines are `engine_classes.py:33` and `:36` (not `:33-36`); the
mask's pinned range is 4,096 to 131,072 in steps of 2,048, which is 63 values
(`boolean_dsa_topk_mask.circuit-check.py:14-37`). The AND counts, the 2^27 merge and the compile figures I did not
find in the tree and carry as relayed.

### Wrong or stale statements in maintained docs (listed, not fixed)

- `adapters/vllm/README.md:109-110`: "a served prefix-cache hit [is] not on this path". True of the two-stage serve
  only; #8403's replay with the hit recomputed is on `main` (`rebuild.py --recompute-hits`, `steps.py:60-67`).
- `benchmarks/spot_figures/cards/replay.glm-5.3.json`: `status` "measured" with a factor whose `source` says it is
  Moonlight's on one H200 and that GLM-5.3's busy GPU-seconds are not measured. The newer note
  (`replay-how-and-cost`, Wed 11 Nov) reads 317 against 381 busy GPU-seconds for `f460` from its GPU samples.
- `benchmarks/spot_figures/cards/serve_tax.glm-5.3.json`: the certified tick's 1.626% is simulated on Llama-3.2-1B
  traces; the one GLM-5.3 pair gives 1.0% (442.2 against 446.7). The card is the older.

### Message for Daniel

**Cheapest replay.** Re-run only the sampled request's own computed tokens, in the steps it was served in, with every
other request replaced by zeros of the same shape. The cached prefix, and a long request's own earlier positions, are
read from KV pages that serving committed as it wrote them. So the cost does not grow with repeated tokens, and
with context grows no faster than serving.

**At serving.** A hash of each KV page as written and a host copy until the coins: modeled under 0.2% of
serving time, not yet measured. Zero scheduling advice also needs the certified tick, 1.0% of output rate in our one
pair of runs, so the total is near 1%.

**At audit.** A sampled request's replay is its own steps: 19 to 32 seconds, measured on short requests. Recomputing
the cached prefix, as `main` does, multiplies what is committed and proved by 7 to 59 at Copilot's and AgentX's
reuse, and replays the conversation so far.

**Advice.** 0 bits for cache hits and scheduling; 5.6 bits a request for the output's length and ending, 0 once the
verifier checks that itself. Speculative decoding adds at most 2 bits a step until its accept rule is in the
circuit.

**Missing.**

1. The guarantee: the bound we quote needs each request to read in one replay block. One new statement covers cache
   hits and long contexts, at about 1.5 times today's bound for 86 requests sharing a prefix (modeled); without it,
   about 1,900 times.
2. The request's circuit on this path, the draft layer and its accept rule.
3. Measurements: the longest cache hit read back this way is 384 tokens; nothing at an agentic load or past
   131,072 context. The design does not limit context; this node's memory stops near 400,000 tokens.

## Addendum, Thu 12 Nov 03:49Z, revised 05:21Z for the two builds, the pin first (due 06:15Z): row 16 for vLLM's published line: an FP8 KV cache, five draft tokens a step, Model Runner V2

**Row 16 stays the route. For this serve none of it is shown bit for bit: every showing was at a BF16 cache, with no
draft token, at the pin, and the FP8 cache puts another attention kernel under every layer. What serving commits
changes in two places. The page is the FP8 row as stored, 656 bytes a token a layer (0.57 of BF16's 1,152, not
half), taken once the page is in the prefix cache and the step that wrote its last kept row has ended, and it
holds the draft layer's rows. And each
decoding request's kept count in each step is a value the logits decide and the certified record does not fix:
advice for every request, at most 2.585 bits a decoding request a step, until each request's Program checks its own. The FP8 write reads
nothing outside the token's own row, so zero stand-ins stay enough for the write. For the FP8 read, and for
anything with draft tokens, that is not established.**

**Revised 05:21Z, the pin first** (the lead's 04:06Z, 04:30Z and 04:38Z sections). **My 03:49Z lines were all read
at vLLM `main`, `20ad1db3fe90`, which is neither the build Friday serves, the pin `d9105ea8001e` (P), nor the tag
`v0.34.0rc1` (T).** Section 0 puts each of the 69 cited places at both: at P 34 have the same text, 26 differ and 9
are not there; at T 66 have the same text and 3 differ. **The conflict, at each build apart: at P a hit under `mtp`
drops its trailing block; at T it drops none, and a block's key covers the token after it. vllm's reader is right at
T, and my line was wrong where I read it.** Three more lines of mine are corrected: when a page enters the prefix
cache; a row's draft count in a step, which is the scheduler's and not only short near a request's end; and vLLM's
own token replay, which refuses a draft at P and takes one at T. Between the builds the route sees six differences:
the FP8 write's scale rule, the drawn token's noise, the hit's rule, when a page is published, vLLM's token replay,
and T's skipping of draft passes nothing would verify. The accept rule and the scheduler's arithmetic for a row's
draft count are the same at both. **vllm's two questions: (a) yes at both builds: the request's block computes the
accept test and the kept count is its output, with no draft probabilities under the default; its gate count is not
read. (b) Registered, in round 1's step record, and charged until checked: at most 2.585 bits a decode row a step
for the kept count, and 0 for the scheduled count once the advice code models the scheduler under drafts. At P as
entry 2's showings were served, a BF16 cache and no draft token, entry 2's row 16 stands as written.**
`r20261112-025327-f570` ended 05:10:05Z, exit 0: the step record's and the canonical top-k's cost at saturation is
not told from zero, and a cost under about 1% is not resolved by it (section 5).

**Sources.** P: the pin `d9105ea8001e`, the build Friday's goal serves. T: the tag `v0.34.0rc1` at `52aeb5806585`.
M: vLLM `main` at `20ad1db3fe90` (12 Nov 03:11Z), where sections 1 to 6 were first read: **a line cited "M" below
is at the place section 0's table gives for P and for T, with the same text unless a note there says otherwise.**
N: `v0.32.0`. V: our tree at `be9a3d5669`. vLLM paths are under `vllm/` unless they start with
`recipes/`; V paths under `adapters/vllm/verity_vllm/` unless longer. "The capture map" is
`adapters/vllm/manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json`. Marks as in entry 2, except that entry 2's [M run] is written [measured] here, M being taken. [relayed,
prodserve] is vllm-prodserve's two comments on VLLM-377, 03:20Z and 03:23Z, which I read whole. Nothing was run on a
GPU; every number marked [mine] is arithmetic on numbers read, with its inputs.

**Two of the brief's "what changed" lines, checked.** Model Runner V2 changes nothing: it is the runner we serve on
at the pin (V `engine/env.py:20`) [read; the lead corrected this on VLLM-377 at 03:20Z]. DeepGEMM is in our serve
already: the FP8 linears from 32 rows in a step and the index's scores (the capture map `:65-66`,
`DeepGemmFp8Linear_v1`, `DsaMqaLogits_v3`, `_v4`) [read]. What is new to the circuit is the FP8 cache's write and
read, the draft layer, the accept rule and, at T, the sampler's noise (section 0, note n).

### 0. The two builds, the pin first: which commit is which, every cited line at each, the conflict, and vllm's two questions

**Which commit is which** [ran: `git log -1` on each; GitHub's compare between them]. **P is the build Friday's goal
serves** [relayed, the lead, 04:38Z: Daniel, 8:00 PM PST]: `d9105ea8001e`, a commit of `vllm-project/vllm` of 7 Sep
04:57Z, "[Qwen3.8-Flash-Next] Remove torch.compile for NVIDIA implementation (#55272)". It is the pin entry 2 names,
the same tree, and an ancestor of the tag. **T is the second stream's build:** the tag `v0.34.0rc1` at
`52aeb5806585`, 3,969 commits after P and none behind. N, `v0.32.0`, is 2,002 commits after P. **My 03:49Z lines are
all at M, vLLM `main` at `20ad1db3fe90`, which is neither build:** 287 commits after the tag and 6 behind it. I took
M as the newest I could fetch and did not say it was not the pin's tree; the table below puts every one of those
lines at P and at T.

**What Friday serves at P.** Each of entry 2's five showings was at P with a BF16 cache and no draft token. If
Friday's serve keeps those, entry 2's row 16 stands as written and none of this addendum applies to Friday. If it
turns the published line's flags on at P, the P column below applies. Which of the two it is, is not in what I was
relayed. The recipe that states the line is not in P's tree [ran: `git ls-tree`, no `recipes/` at P]; its parts are
code at P: `mtp` is a method (P `config/speculative.py:1859`), and an FP8 cache picks the same backends (P
`platforms/cuda.py:136-152`) [read].

**The conflict on a hit under `mtp`, settled at each build apart** [read; the lead's 04:30Z and 04:38Z]:
- **At P a hit drops its trailing block.** `use_eagle_block_drop()` is true for `mtp` unless turned off (P
  `config/speculative.py:1859-1863`); the scheduler hands it to the cache manager and to each lookup (P
  `v1/core/sched/scheduler.py:288`, `:305`, `:435-442`); the coordinator says why: the chunk's last position is
  written with the next token, "which may or may not be present after the matching prefix" (P
  `v1/core/kv_cache_coordinator.py:108-125`). P has no hash over the next token [ran: `git grep
  num_hash_lookahead_tokens` at P and at N, no match].
- **At T a hit drops no block.** With a draft that reads the target's hidden states, the prefix cache on and no KV
  connector, each block's hash covers the one token after the block (T `config/vllm.py:632-653`, `:1455`; T
  `v1/worker/gpu/spec_decode/speculator_config.py:121-125`; T `v1/core/kv_cache_utils.py:962-974`, `:999-1009`; T
  `v1/core/block_pool.py:331-344`), and the scheduler then turns the drop off: "Lookahead hashes prove the boundary
  token, so EAGLE needs no drop" (T `scheduler.py:342-345`). `disable_lookahead_block_hashes` brings the drop back
  (T `config/speculative.py:532-537`).
- **So vllm's reader is right at T, and my 03:49Z line was wrong where I read it:** M has T's mechanism (M
  `scheduler.py:341-344`). I cited the config's `use_eagle_block_drop()` and did not follow it into the scheduler.
  What I wrote is what P does, which is not where I had read it.
- **For the registered record** [argued]: the rule for a hit's length and a block's key is one a build. At P: the
  longest run of matching blocks less the last one. At T: a block matches only if the token after it matches too,
  and a block is hashed only once that token exists (T `kv_cache_utils.py:971-973`), so a prompt's last full block
  is not hit by a request that continues with another token.

**Every vLLM place this addendum cites, at P and at T** [ran: for each of the 69 places in 27 files, M's lines
looked for line by line in P's file and in T's; where they are not there verbatim, the nearest lines diffed and
read; each one-line cite read by eye]. No letter: the same text. A letter: the note under the table.

| File (under `vllm/` but for the recipe) | Cited at M | At P, the pin | At T, the tag |
|---|---|---|---|
| `recipes/models/zai-org/GLM-5.3.yaml` | `:107-109`, `:180-187`, `:238-249`, `:279-282` | not in P's tree: it has no `recipes/` (a) | same lines |
| `platforms/cuda.py` | `:136-152` | `:136-152` | same lines |
| `v1/attention/backends/mla/flashattn_mla_sparse.py` | `:43-46`, `:98` | `:36-39`, `:87` | same lines |
| `model_executor/layers/attention/mla_attention.py` | `:362-379`, `:538-560` | `:353-370` (b), `:495-517` | `:361-378`, `:537-559` |
| `v1/attention/backends/mla/flashmla_sparse.py` | `:63-74`, `:77-126`, `:149-153`, `:409-426`, `:440-444`, `:723-726`, `:1228-1232`, `:1289-1294`, `:1322-1332` | `:52-63`, `:66-103` (c), `:112-116`, `:322-339`, `:350` (d), `:587-590`, `:833-837`, `:875-880` (b), `:908-918` (e) | same lines |
| `v1/attention/ops/flashmla.py` | `:14-45` | `:14-43` (b) | same lines |
| `v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py` | `:66`, `:148-152` | `:63`, `:69-73` | same lines |
| `models/deepseek_v32/common/kernels.py` | `:23-27`, `:63-119`, `:204`, `:355-384`, `:368-374`, `:386-408`, `:409-515` | a literal 448 at `:280` (f), `:50-62` (g), `:163`, `:263-291` (f), `:276-280` (f), `:293-314` (h), `:315-415` (g) | same lines |
| `model_executor/layers/attention/attention.py` | `:148-168` | `:127-147` | same lines |
| `models/deepseek_v32/attention.py` | `:185`, `:214-216`, `:390-424` | `:185`, not there (i), `:336-366` (g, h) | `:184`, `:213-215`, `:366-400` |
| `v1/attention/backend.py` | `:820` | `:830` | same line |
| `models/deepseek_v32/nvidia/mtp.py` | `:61-165`, `:167-268` | `:58-158` (j), `:160-261` (b) | `:61-164` (j), `:166-267` (b) |
| `v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py` | `:136-140`, `:142-147`, `:233-246`, `:391-439` | the file is `spec_decode/autoregressive/speculator.py`: `:234-238` (b), `:240-245`, `:308-320` (b), `:582-670` (k) | same lines |
| `v1/worker/gpu/spec_decode/mtp/speculator.py` | whole | whole, `:1-58` (k) | same file |
| `v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | `:519-784`, `:582`, `:615-616`, `:634`, `:716`, `:746-752`, `:990-1000` | `:484-692` (l), `:541`, `:569` (l), `:587`, `:662`, `:588-594` (l), `:812-822` | same lines |
| `config/speculative.py` | `:613`, `:676`, `:734`, `:1888-1899` | `:479`, `:539`, `:588`, `:1859-1863` (m) | `:570`, `:633`, `:682`, `:1758-1769` |
| `v1/worker/gpu/sample/gumbel.py` | `:8-13`, `:67-71`, `:80` on | `:7-14` (b), `:83-87`, not there: the noise is Philox, `:120-125` (n) | same lines |
| `v1/worker/gpu/sample/states.py` | `:54-58` | `:50-54` | same lines |
| `v1/worker/gpu/states.py` | `:64-65` | `:61-62` | same lines |
| `v1/core/sched/scheduler.py` | `:982-997`, `:1326-1342`, `:1607-1610`, `:1728-1733`, `:2336-2354`, `:2394-2415` | `:766-781`, `:994-1010`, `:1220-1223`, `:1354-1359`, `:1934-1952`, not there (o) | `:1026-1041`, `:1357-1373`, `:1643-1646`, `:1764-1769`, `:2392-2410`, `:2450-2471` |
| `v1/worker/gpu/model_runner.py` | `:2598-2610` | `:2209-2221` | `:2563-2575` |
| `v1/worker/gpu/input_batch.py` | `:545-562` | `:519-536` | same lines |
| `v1/worker/gpu/spec_decode/rejection_sampler.py` | `:314-427` | `:183-275` (b) | same lines |
| `v1/worker/gpu/sample/trace_replay.py` | `:137-175` | not there (p) | same lines |
| `config/model.py` | `:321-326` | `:273-278` | `:320-325` |
| `v1/core/kv_cache_coordinator.py` | `:66-87` | not there (m) | `:66-87` (b) |
| `v1/kv_cache_interface.py` | `:1553-1554` | `:1275-1276` (b) | `:1549-1550` |

**At P: 34 places have the same text, 26 differ, 9 are not there. At T: 66 have the same text, 18 of them at other
line numbers, and 3 differ.** The notes [read, each]:
- **(a)** The published line's recipe exists at T and M only, the same file at both.
- **(b)** Wording, typing or argument names only: a type annotation; a docstring's quoting; a renamed argument
  (`kv_c_and_k_pe_cache` at P for `kv_pages`); an error message's layout; at P `num_speculative_steps` for
  `num_speculative_tokens`, and one argument the first draft pass does not pass (`skip_sample`); at P the draft
  group's comment reads "contains EAGLE/MTP draft attention layers". The verify chunks at P carry no sampling mask
  (P `rejection_sampler.py:183-275`).
- **(c)** P's layout text has the 656-byte row and not the later 528-byte row of another model; the 656-byte row's
  lines are the same.
- **(d)** At P the mixed-batch rule is the head count alone, `num_heads < 32` (P `:350`); T and M add two
  conditions, `use_hisparse` and `pcp_dcp_kv_gather`, whose values on node A I did not read. With 8 heads a rank the
  answer is the same at all three unless one of those is on.
- **(e)** At P the padded head count is a stored field (P `:908`), at M and T a call on that line; the padding to it
  is the same text (P `:910-918`).
- **(f)** **The FP8 write's scale differs.** P: a tile's largest magnitude over 448, floored at FLT_MIN, not rounded
  (P `:276-280`: `tl.maximum(tile_amax * (1.0 / 448.0), 1.1754944e-38)`). T and M: floored at 1e-4 and rounded up to
  a power of two (T `:368-374`). I had this for P as relayed from prodserve; it is now read.
- **(g)** **The index keys: the same quantizer and the same row at both builds on an H200.** One scale a token, its
  largest magnitude floored at 1e-4 over 448, rounded up to a power of two (P `:50-60`; T `:64-75`); the write puts
  a token's key bytes and its 4-byte scale at the same offsets (P `:77-86`; T `:98-119` with its options off). T's
  options are off on an H200: the shuffled layout is ROCm's (T `model_executor/models/deepseek_v2.py:670-673`; T
  `models/deepseek_v32/amd/rocm.py:29-32`), the packed scale is SM 10.x's (T `utils/deep_gemm.py:656-666`), and 448
  becomes 224 only for the FNUZ type (T `kernels.py:23-27`). T also gives the index cache a slot mapping of its own,
  which I did not follow.
- **(h)** Plain `fp8` at P always loads the layer's scale and divides by it (P `kernels.py:300-301`); T and M skip
  the division when the scale is 1.0 (T `attention.py:366-400`, `skip_scale`). The scale is 1.0 at both, so the
  stored bytes agree unless a division by 1.0 is not the identity, which it is in IEEE arithmetic [argued].
- **(i)** A comment on a ShardQ layout that P does not have. The reuse of the first draft pass's top-k is at P (P
  `spec_decode/mtp/speculator.py:28-34`).
- **(j)** `eh_proj`'s low-latency GEMM plan: P and T test the draft layer's own model type, M the target's. The plan
  exists only on SM 10.0, 10.3 and 10.7 (T `models/deepseek_v32/nvidia/glm52_low_latency_gemm.py:332-340`), so on an
  H200 `eh_proj` is a plain linear at all three. P also builds no index group there (P `:58-158`).
- **(k)** At P the draft passes are `AutoRegressiveSpeculator`'s: the first over the target step's whole batch with
  the rejected positions padded in (P `autoregressive/speculator.py:240-245`, the same text), then K - 1 passes of
  one row a request (P `:582-670`, a loop inside `_decode` there, a function of its own at T). The reuse of the
  first pass's top-k is at P (P `spec_decode/mtp/speculator.py:28-34`); T asks in addition that prefill context
  parallelism be 1.
- **(l)** **The accept rule is the same at P, T and M:** `u = tl_rand32(seed, pos, includes_zero=False)` (P `:569`),
  greedy accepts where the draft is the target's argmax (P `:587`), sampled accepts where `target_logprob > log(u) +
  draft_logprob` (P `:662`). T's kernel adds the "fly" window (T `:605`, `:717-745`) and, under block verification,
  a seed of its own a draft (T `:615`), both off by default; P's kernel has no "fly" and one seed.
- **(m)** The conflict above. `replay_boundaries` is T's and M's; P states the drop in the coordinator (P
  `kv_cache_coordinator.py:108-125`).
- **(n)** **The drawn token's noise differs.** P: Philox for the Gumbel noise as for the accept test's uniform (P
  `gumbel.py:83-87`, `:120-125`). T and M: MurmurHash3 for the Gumbel noise, Philox kept for the accept test (T
  `gumbel.py:8-13`, `:67-71`, `:80` on).
- **(o)** **When a page enters the prefix cache differs, and my "published after acceptance" was too short for
  either build.** At P and at T a block is cached when the scheduler allocates its step, for the tokens already
  final: "we cap the number at `request.num_tokens`, ensuring only 'finalized' tokens are cached" (P
  `v1/core/kv_cache_manager.py:553-562`; T `:637-646`), which is before the GPU has written that step's rows. T adds
  a publish at the update for the tokens the step made final, "Publish finalized states before retirement or
  completion" (T `scheduler.py:2467-2471`); P has no such line [ran: `rg` at P]. So at P a page that accepted drafts
  complete is cached at its request's next allocation, or never if the request ends first [argued]. Section 3, f is
  corrected with this.
- **(p)** **vLLM's token replay refuses a draft at P.** P's file has no kernel for verified drafts, and a request
  that carries forced tokens is refused on an engine with speculative decoding (P `sampling_params.py:941-944`; N
  `:1118-1121`). T takes one (T `trace_replay.py:137-213`; no such refusal at T, [ran: `rg`, six refusals at T
  `sampling_params.py:1122-1141`, none of speculative decoding]). Section 3, d is corrected with this.

**The lead's seven facts of 04:06Z, each checked** [read, at the places in the notes]: 1, the noise: agreed, note n.
2, the FP8 write's scale: agreed, note f. 3, vLLM's token replay: agreed at P, N and T, note p; `d14f72c7` itself I
did not read. 4, the block's key under `mtp`: agreed, the conflict above. 5, a row's draft count is the scheduler's:
agreed, and the same arithmetic is at P, with one rule that is T's alone (section 3, b, corrected). 6, the accept
test's draw carries no state, and "fly" is new, lossy and off by default: agreed, note l; `rejection_sample_method`
defaults to "standard" (T `config/speculative.py:610-617`); the route refuses "fly", "synthetic", "block" and a
probabilistic draft, each of which changes the test [mine]. 7, our advice code refuses speculative decoding whole:
read (V `adapters/vllm/verity_vllm/pipeline/advice_code/replay.py:23-25`); the charge is (b) below.

**vllm's question (a): does the request's block compute the accept test at all K positions, so the kept count is
derived and not advice? Yes at both builds, as a design; no code does it** [argued].
- What the block computes [read; the rule is note l's, the same at P and T]: the target's logits at the row's 1 + K
  positions, which the target pass runs whatever is kept; then for each draft in order, while none was rejected and
  the draft is not a -1 placeholder (T `rejection_sampler_utils.py:598-610`; P `:554-567`): one uniform from the
  seed and the position, the target's log probability of the draft token, which is its logit over the temperature
  less a log-sum-exp over the vocabulary taken in blocks (T `:680-706`), and one comparison. The kept count is the
  number accepted before the first rejection (T `:747-748`, `:759`). A circuit has fixed shape, so it evaluates the
  test at all K positions and counts a position only if every earlier one passed, the form `Accept_v1{K}` has for
  greedy: "Liveness of slot `j` is *derived* from the verdicts" (V `verity/definitions/registry/spec.py:6-9`) [read;
  argued].
- **What the block must be given, and from which record** [argued]: **the draft tokens from no record:** they are
  outputs of the draft layer inside the same block (case 2 of the 03:22Z ruling), with one exception I had not
  written. A step verifies the drafts proposed after the request's step before it ("Drafts proposed after a step are
  verified in the request's next decode step", T `scheduler.py:525-526`), so **a span's first step verifies drafts
  proposed before the span:** either the span's boundary record holds those K token ids as a port (case 1), or the
  span begins one step earlier. **The draft's probabilities: none.** The draft is an argmax by default and "the
  draft probabilities are treated as one-hot during rejection sampling" (T `config/speculative.py:682-688`; P
  `:588`), so `draft_logprob` is 0 and the test is `target_logprob > log(u)`. **From records:** the seed by its rule
  (case 3); temperature and top-p from the certified request; the positions, the scheduled count and the kept count
  to compare with, from the step record (case 1).
- **What it costs in gates a decode row: not read.** No Definition of the sampled accept test exists (section 4), so
  no gate count does, and I will not give one. What settles it is not a run on a GPU: the accept Definition under
  sampling at this vocabulary, counted by `circuit-check` [circuits]. In work, read off the kernel [mine: the
  vocabulary is 154,880, V `adapters/vllm/verity_vllm/engine/profiles/hf_configs/GLM53.config.json:223`]: at K = 5 a
  decode row gains five log-sum-exps over 154,880 logits (774,400 exponentials), five logs of a uniform and five
  comparisons, and still draws one token, the resample or the bonus token, where a row without drafts draws one. The
  census's work model for one drawn token is 5 V operations and 2 V logs (`benchmarks/census/subcircuits.json`,
  `gumbel-top-p-token-select`, `work_model`), which is 774,400 and 309,760 here [mine]. The test is small beside the
  five more rows through the target's 78 layers and the draft layer's passes, which row 16 runs in any case, as the
  served step's shape (section 1, column 3) [argued].
- **What stays outside the block:** the scheduled count, how many drafts the scheduler gave the row in the step. It
  is not a function of the request alone (section 3, b), so the block takes it from the step record.

**vllm's question (b): are the batch-mates' counts registered, or charged as advice? The route takes registered, and
a registered count is still charged until something checks it** [argued; the charge's form relayed, the lead's fact
7].
- **Registered:** the step record is a port of round 1's boundary record (case 1 of the lead's ruling), committed
  before the coins, and it holds for each row of each step the scheduled count and the kept count the GPU saw. Why:
  a one-request replay builds every step's shapes from it, its batch-mates' rows, positions and key counts; a
  stand-in is zeros and computes no count of its own (section 3, d); and a shape the server could fix after the
  coins is the hole FRAME found (the ruling). Counts supplied at the replay as advice are refused for that reason.
- **Charged, until checked:** a registered count nothing checks is still the server's choice at serving. **The kept
  count:** at most log2 6 = 2.585 bits a decode row a step, so at most 82.7 bits a step at 32 decode rows and 330.9
  at 128, vLLM's default `max_num_seqs` (T `config/scheduler.py:42`, `:61`) [mine: 32 and 128 times 2.585]. It is 0
  for a row whose request's Program computes the accept test and compares (a), which the route asks of every
  request; so the charge ends when the accept Definition lands and the lemma that its count is unique is stated.
  **The scheduled count:** 0 bits where the registered scheduler rule gives it from the step's rows, as the advice
  code gives a row's admission today; it is the scheduler's arithmetic on the token budget, the input budget, the
  context limit and `max_tokens` (T `scheduler.py:855-873`, `:1026-1037`, `:1358-1380`; P `:619-634`, `:766-781`,
  `:994-1016`). The advice code refuses speculative decoding whole (V `advice_code/replay.py:23-25`), so until it
  models those lines the scheduled count is charged too, at most 2.585 bits more a decode row a step.
- **Why a batch-mate's count reaches the sampled request at all** (section 3, b): through the batch-mate's length,
  which sets its row's key count and the step it finishes in, and through the token budget its rows take from a
  later row of the same step. A wrong registered count for a batch-mate is caught where that batch-mate is sampled,
  by its own block's comparison [argued; no Lean; I did not check it against `InfluenceTwoPatterns`].
- **CACHE answers both the same way** (its entry 7, section 5).

### 1. Row 16 restated, column by column

| Column | Does not change | With an FP8 KV cache | With five draft tokens a step |
|---|---|---|---|
| 1. What serving commits beyond stock | The step record. A hash and a host copy of each page. One rank's copy: the write is a function of the token's latent row alone (section 2), and that row is the same on every rank (V `pipeline/boundary_tables.py:8`) [argued]. | The page is the stored FP8 rows: 53,940 B a token a rank against 92,628 [mine: 78 × 656 + 21 × 132; 78 × 1,152 + 21 × 132, which is entry 2's 92.6 KB; 21 of the 78 layers hold index keys, V `engine/profiles/hf_configs/GLM53.config.json`, `indexer_types`]. The hash follows the bytes: 0.021% to 0.026% of one card [mine: 0.58 of the modeled 0.036% to 0.045%]. Not built, not measured. | The draft layer's rows join the page: 54,728 B a token a rank [mine: 79 × 656 + 22 × 132; the draft layer always holds an index, M `models/deepseek_v32/attention.py:185`], 0.59 of BF16's. The page's hash is taken once the page is in the prefix cache and the step that wrote its last kept row has ended, not at the GPU's write: a row at a rejected position is written and then overwritten (M `v1/core/sched/scheduler.py:2336-2354`), and when a page enters the prefix cache differs by build (section 0, note o) [read]. The step record holds, for each decoding request in each step, the draft count the scheduler gave it and the kept count the GPU saw (section 3, b). |
| 2. Advice beyond the common list | None for hits, admission, release buckets and seeds: every request is given its seed before the engine sees it (V `engine/served_seed.py:1-14`) [read]. | None [argued: the write has no choice in it]. | The kept counts: at most log₂ 6 = 2.585 bits a decoding request a step, charged for every request, sampled or not. That is 207 to 239 bits for a 256-token output and 360 to 414 for AgentX's median output of 444 tokens, against 5.6 bits for the length and the ending [mine: 80 to 92 and 139 to 160 steps at 3.19 and 2.77 tokens a step, the recipe's two figures, which are greedy runs on other hardware, M `recipes/models/zai-org/GLM-5.3.yaml:279-282`]. 0 once every request's Program checks its own kept counts and a lemma shows the count unique (section 3, e). The scheduled count is charged as well until the advice code models the scheduler under drafts (section 0, b). |
| 3. Replay cost of one sampled request | The span's steps on an engine kept up. Flat in the repeated share. [argued] | Pages loaded shrink: 7.2 GB a rank at 131,072 tokens and 7.8 GB at AgentX's 142,016, in place of 12.1 and 13.2 [mine: 54,728 B a token, the draft layer's rows included]. | Fewer and larger steps: 139 to 160 for 444 outputs in place of 444 [mine, same figures]. Each runs six rows a decoding request through the 78 layers, then the draft layer over the same rows, then four passes of one row a request (section 3, a): 1.9 to 2.2 target rows run for each token emitted [mine: 6 over 3.19 and 2.77]. A step's time with drafts on this node: not measured. A hit's length follows the build's rule (section 0). |
| 4. Why bit for bit; what is shown | The premise only: a row's bits depend on its step's shapes and not on the other rows' values. | **Nothing is shown.** Every layer's attention is another kernel: FlashMLA's FP8 sparse kernel, where we serve `FLASH_ATTN_MLA_SPARSE` and FlashAttention 3 over BF16 rows (the capture map `:63-64`) [read]. It is launched once over all the step's rows (section 2). | **Nothing is shown.** No step with draft tokens was ever run through the adapter, which refuses one (V `pipeline/boundary_tables.py:230-231`, `engine/sampler_capture.py:51-53`) [read]. A stand-in's kept count has to be forced from the record, and no code does (section 3, d). vLLM's own token replay refuses a draft at P and takes one at T (section 0, note p). |
| 5. What it needs from the circuit ahead of time | Row 5's Program cut into spans; the size class's replay Definition; a Definition that opens a committed page; the step sizes; ranges past 131,072. DeepGEMM's Definitions exist. | The quantizing write and the 656-byte row; FlashMLA's FP8 sparse kernel at this model's geometry. The write's rule differs between P and T, so it is one Definition a build (section 0, note f). | The draft layer, layer 78; launches at six rows a decoding request; the accept test under sampling and the resample; the compile slice with a decode row of K + 1 tokens, the roll-back and the build's rule for a block's key (section 4). |
| 6. What the guarantee charges | The statement: it is over the circuit [relayed, the lead; I agree]. Entry 2's cell is corrected by CACHE (section 5): what is missing for a shared page and for a span is a counting lemma of the plan check, not a theorem. | Nothing [argued]. | Kept counts are charged as advice until the accept check and its uniqueness lemma (section 3, e). The interior grows by the draft layer's rows: a page a reader opens holds them, so the producer's Program computes them at every position, prefill included [argued]. |

**What is no longer shown bit for bit, plainly.** All of it. Entry 2's five showings were: four targets replayed
with every other request zero; a 256-token and a 384-token hit read from serving's pages; a 19,170-token prompt; a
request preempted three times; an all-real control of 307 steps. Each was at vLLM `d9105ea80`, a BF16 cache,
`FLASH_ATTN_MLA_SPARSE`, no draft token [measured: the runs named in entry 2; the kernels read in the capture map]. Under the
published line the cached bytes are other bytes, attention is another kernel, every decode step has six rows a
request and a 79th layer, and at T a sampled token is drawn from another noise generator than P's (section 0, note n). The first
shape-only replay at FP8 without drafts, then one with drafts, are the two runs that put it back (section 6).

**Row 16 under the lead's ruling of 03:22Z** (every input of a replay block is a port of the boundary record, another
template's output in the block, a value the verifier computes from what it holds before the sample, or the opened
row of a read against a committed root) [relayed; the sorting is argued, mine]:

- **Case 1, committed at serving:** the request's prompt and output ids, which the certified record also holds;
  the step record, with each step's rows and each decoding request's scheduled and kept count; for a span, the
  draft tokens its first step verifies (section 0, a). Today's served driver registers a replay block's parameters after the sample
  [relayed, FRAME's finding in the ruling], so the step record's commitment before the coins is part of what row 16
  needs, with or without drafts.
- **Case 2, computed inside the block:** the logits, the draft tokens, the accept test's verdicts, the token at a
  rejection, the kept count the block compares with the record's.
- **Case 3, the verifier's own:** the zero stand-ins and their shapes, which come from the step record; the seed,
  by its rule from the window's seed coin and the certified id (V `engine/served_seed.py:1-14`).
- **Case 4, opened against a committed root:** every cache row the block reads and did not write: the cached
  prefix's, a span's earlier positions', and under drafts the draft layer's rows at those positions. Which page is
  read is fixed by the step record; under drafts that needs the kept counts, since they set the slots a step writes.
- **Nothing in row 16 is outside the four,** if the page's root is taken when its bytes are final (section 3, f) and the step record is committed
  before the coins. The one input I cannot place is whatever FlashMLA's kernel reads beyond a row's own keys
  (section 2, item 8).

### 2. What serving commits for a hit when the cache is FP8

**The page is the FP8 bytes as stored, and the scale is computed from the token's own row alone.** By line:

1. **The published line asks for plain `fp8` on H200s** (M `recipes/models/zai-org/GLM-5.3.yaml:180-187`:
   "H100/H200 use plain fp8 KV cache"; the command at `:238-249`) [read].
2. **The engine then picks the backend, and the backend picks the format.** On Hopper the sparse backends are tried
   in the order `FLASH_ATTN_MLA_SPARSE`, `FLASHMLA_SPARSE`, then FlashInfer's, since this model's cached row is 576
   wide, not 512 (M `platforms/cuda.py:136-152`; P `:136-152` the same; 512 + 64 in V `GLM53.config.json`) [read].
   The first refuses any cache type but `auto`, `float16`, `bfloat16` (M
   `v1/attention/backends/mla/flashattn_mla_sparse.py:43-46`, `:98`; P `:87`, `:176`) [read]. `FLASHMLA_SPARSE`
   serves every quantized type as `fp8_ds_mla` (M `model_executor/layers/attention/mla_attention.py:362-379`; P
   `:358-363`; M `flashmla_sparse.py:149-153`: `"fp8",  # alias for fp8_ds_mla`) [read]. If FlashMLA is not in the
   build (M `v1/attention/ops/flashmla.py:14-45`), FlashInfer's serves plain `fp8` (M
   `flashinfer_mla_sparse_sm90.py:66`, `:148-152`) [read]. **Which one node A's engine picks is not established by
   reading**: the start prints "Using %s KV cache format for %s backend" (M `mla_attention.py:538-560`).
3. **`fp8_ds_mla`, the format I expect** [read M `models/deepseek_v32/common/kernels.py:355-384`; the layout in M
   `flashmla_sparse.py:77-126`]: a token's row in a layer is 656 bytes: its 512 latent values as E4M3 in four tiles
   of 128, each tile's float32 scale, and its 64 rotated position values in BF16, not quantized. A tile's scale is
   its own largest magnitude over 448, floored at 1e-4, rounded up to a power of two (`:368-374`; 448 at `:23-27`;
   the comment at `:371` says FLT_MIN, the code at `:372` has 1e-4).
4. **Plain `fp8`** [read M `kernels.py:386-408`]: all 576 values over one scalar the layer holds, which is 1.0 (M
   `model_executor/layers/attention/attention.py:148-168`) and which nothing under `vllm/` at M recomputes [ran: `rg
   calculate_kv_scales`, no match]; at 1.0 the division is skipped (M `models/deepseek_v32/attention.py:390-424`).
5. **The index's keys** are FP8 with one scale a token at the pin already, 132 bytes (M `kernels.py:63-119`,
   `:409-515`) [read]; nothing changes for them, at P or at T on an H200 (section 0, note g).
6. **Nothing outside the token's own row is read.** The write is one Triton program a token (M `kernels.py:204`),
   which loads that token's row, the norm's weight, the rotation at its position and its slot [read]. No batch-mate,
   no other token of the request, no running statistic, in either format. **So zero stand-ins stay enough as far as
   the write goes** [read; prodserve's 03:23Z comment reads the same at P and N].
7. **The rule differs between the two builds.** At P a tile's scale is its largest magnitude over 448 floored at
   FLT_MIN, with no rounding to a power of two (P `kernels.py:276-280`) [read]; at T and M it is floored at 1e-4 and
   rounded up (T `:368-374`) [read]; N [relayed, prodserve]. The same values give other stored bytes, so the write's
   Definition is one a build.
8. **The read side is where a batch-mate could enter, and I could not settle it.** With 8 heads a rank (64 over 8 ranks, V `GLM53.config.json`) the FP8
   kernel takes every row of the step, prefill and decode, in one launch (M `flashmla_sparse.py:63-74`, `:440-444`,
   `:1228-1232`, `:1289-1294`), with the heads padded from 8 to 64 (`:723-726`, `:1322-1332`) and a tile scheduler
   sized from the card's SM count and the number of queries (`:409-426`) [read]. So a row's bits can depend on the
   step's row count, which the step record holds [argued]. Whether they depend on another row's values is inside
   FlashMLA's kernel, which is not in vLLM's tree and which I did not read: **not established**. The tree's twin of
   FlashMLA's sparse decode for DeepSeek V4 on the H200 ports its scheduler from the batch's shape (V
   `verity/definitions/numerics/flashmla_twin.py:7-9`) [read]; that is another model's row and entry point. vLLM's own code
   does not quantize the query on this path (M `v1/attention/backend.py:820`; `flashmla_sparse.py` never sets it)
   [read].
9. **So serving commits, for a page:** the bytes above for all 79 layers' rows and the 22 index-key rows, as they
   stand when the page's bytes are final (section 3, f). The reader's replay block opens the rows it reads
   against that root, which is case 4 of the lead's 03:22Z ruling [relayed]. **The Definitions that read the page
   take FP8 bytes and scales as inputs, never the BF16 values, which serving does not keep** [argued].

### 3. What serving commits for a speculative step: the lead's five sentences against the code

**(a) "The draft tokens and the accept test are values the replay recomputes." Right, with what recomputing
needs.**
- The draft layer is the model's layer 78: `enorm`, `hnorm`, `eh_proj` from 2H to H, one decoder block, the shared
  head (M `models/deepseek_v32/nvidia/mtp.py:61-165`, `:167-268`) [read]. Its block has its own sparse attention,
  its own index (M `attention.py:185`) and so its own cache rows at every position [read].
- Its first pass a step runs over the target pass's whole batch, prefill rows and rejected positions included: "we
  pad each request's query length to include any rejected positions" (M
  `v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py:142-147`, `:233-246`) [read]. Then four passes of one
  row a request (M `:391-439`), which reuse the first pass's top-k (M `spec_decode/mtp/speculator.py`, whole file;
  M `attention.py:214-216`; `index_share_for_mtp_iteration` true in V `GLM53.config.json`) [read].
- So the draft tokens need no commitment of their own, as the lead's ruling says, but the draft layer reads its own
  cache rows at every earlier position: for a span or a hit those are committed pages (section 2, item 9) [argued].
- The accept test (M `spec_decode/rejection_sampler_utils.py`, `_rejection_kernel`, `:519-784`) [read]: one
  uniform a position, `u = tl_rand32(seed, pos)` (`:615-616`; the seed is mixed with the draft's index only under
  block verification, which is off); greedy accepts when the draft equals the target's argmax (`:634`); sampled
  accepts when `target_logprob > log(u) + draft_logprob` (`:716`), and a draft is greedy by default so its log
  probability is 0 (M `config/speculative.py:734`). The same three lines at P are `:569`, `:587`, `:662`; at N `:572`,
  `:590`, `:665` [read].
- **Under sampling the tokens depend on the draft layer's bits, not only the counts.** At the first rejection the
  token is drawn from the target's distribution with the rejected draft token removed (M `:990-1000`: "the target
  distribution with the rejected draft token probability zeroed out") [read]. Production samples at temperature 1.0
  and top-p 0.95 [relayed, the lead, 03:30Z], so the draft layer is in the circuit; it cannot be skipped by taking
  the kept counts as given, which would do under greedy [argued].
- **The drawn token's noise is another generator at M and N than at P.** The accept test's uniform stays Philox (M
  `v1/worker/gpu/sample/gumbel.py:8-13`, `:67-71`), the Gumbel noise of a sampled, resampled or bonus token is a
  MurmurHash3 (M `gumbel.py:80` on; for N [relayed, prodserve: N `gumbel.py:97-125`]) [read].
- A request with no seed gets one from `np.random` (M `v1/worker/gpu/sample/states.py:54-58`; P `:53`) [read]. Ours
  never reaches that line (V `engine/served_seed.py:1-14`) [read].

**(b) "The step record must carry, for each row of each step, how many draft tokens it ran and how many it kept,
since those set the next step's shapes for every batch-mate." The fields are right; the reason needs correcting.**
- **A kept count does not change the next step's row count or row order.** Every decoding request is scheduled 1 + K
  rows whatever it kept, padded with −1 placeholders (M `scheduler.py:1326-1342`, `:1607-1610`; P `:1005`), and the
  runner orders rows by whether a request has drafts and by its row count (M `v1/worker/gpu/model_runner.py:
  2598-2610`) [read]. K is fixed: the table that varies it with the batch size is unset by default and adaptive
  verification is off (M `speculative.py:613`, `:676`; M `scheduler.py:1728-1733`) [read]. **Corrected (the lead's fact
  5):** I wrote that a request runs fewer than K drafts only near its end. The count is the scheduler's, at both
  builds (next bullet).
- **The drafts a row verifies in a step are what the scheduler gave it** [read at T, and at P where a P place is
  given]: a running row is cut to the step's token budget, to the input budget less the draft slots and to the
  context limit (T `scheduler.py:855-873`; P `:619-634`), and its drafts are cut to what was scheduled (T
  `:1026-1037`; P `:766-781`); a newly scheduled decode row is padded to 1 + K or left out of the step, "Prefer to
  not schedule than schedule un-padded" (T `:1358-1380`; P `:994-1016`); a request sure to have reached `max_tokens`
  gets no step (T `:813-830`; P `:598-605`). **One rule is T's alone:** T proposes no drafts in a step where no
  scheduled request could use them (T `:525-561`); P has none [ran: `git grep _drafts_unused` at P, no match], so
  whether a step ran the draft passes is one more thing the step record fixes at T. The token budget is what the
  rows ahead of it in the step left, 2,048 a step by default (T `config/scheduler.py:40`, `:47`; what node A's
  engine resolves it to, not read), so the count depends on batch-mates. Under sampling the ids depend on it too: a
  position with a draft emits the draft where the test passes, and a position without one is drawn from the target
  [argued from T `rejection_sampler_utils.py:598-610`, `:716`]. So the step record holds the scheduled count beside
  the kept count.
- **What a kept count changes** [read M `scheduler.py:2336-2354`: `request.num_computed_tokens -= num_rejected`]:
  its own request's positions in every later step, so that row's key count and the slots it writes; the step its
  request finishes in, so every later step's row count and admissions; and the pages allocated. A row's key count is
  among the shapes a batch-mate's bits depend on (entry 2, "What a request's bits depend on"). **So it does change
  the sampled request's kernel shapes, through each batch-mate's length, and not through the row count** [argued
  from the two readings].
- **The trace has the two numbers already, with one doubt.** A step's row is `[request, first position, length]`
  and `emitted` counts the tokens each request emitted (V `engine/step_trace.py:229-250`) [read]: the length is 1 +
  the drafts run, and `emitted` is the kept count + 1 [argued]. The doubt: the first position is the scheduler's,
  and under async scheduling the CPU's count is "Optimistic CPU mirror of num_computed_tokens (upper bound on GPU
  value)" (M `v1/worker/gpu/states.py:64-65`; P `:61`), the GPU correcting it from the rejections (M
  `v1/worker/gpu/input_batch.py:545-562`) [read]. The draft passes also read the batch's largest CPU-side length (M
  `speculator.py:136-140`) [read]. **Whether the recorded positions are the ones the GPU ran is not established.**
- The trace's header carries no speculative setting and no cache format, and `benchmarks/serve_tax/replay.py:79-90`
  builds the replay's engine from the header alone [relayed, prodserve].

**(c) "The sampled request's replay recomputes its own and is refused if they differ." Right as a design; no code
does it, and comparing tokens is not enough.** Under greedy the tokens are the target's argmaxes whatever was kept,
so a replay whose kept counts differ can still emit the served tokens in other shapes [argued from M `:634`,
`:746-752`]. The comparison has to be on the kept count of each step.

**(d) "A batch-mate's enter as shapes only." Right for the speculation's own code; not established for the kernels
it calls.**
- **The draft pass for one request reads nothing of another's in the speculator and the sampler:** every kernel
  there is one Triton program a request: the draft inputs (M `speculator.py`, `_prepare_prefill_inputs_kernel`,
  `_prepare_decode_inputs_kernel`), the accept test (M `rejection_sampler_utils.py:582`), the counts (M
  `input_batch.py:545-562`) [read]. What crosses requests is shape: the batch's largest row count and length (M
  `speculator.py:136-140`) and how many requests share a verify chunk (M `spec_decode/rejection_sampler.py:
  314-427`) [read].
- The draft layer's forward is the model's own kernels over the step's rows, so "shapes only" there is entry 2's
  premise, shown for 78 layers at BF16 and not for this layer or the FP8 kernel: **not established** [argued].
- **A stand-in's kept count must be forced.** Its values are zeros, so its own accept test gives another count than
  the served one. No code of ours or vLLM's forces the count itself [ran: `rg -i 'draft|spec_decode|speculative'
  benchmarks/serve_tax/rebuild.py`, no match].
- **Corrected (the lead's fact 3): vLLM's own token replay, at each build** [read]. It forces a request's tokens
  (off by default: P `config/model.py:273-278`; T `:320-325`). **At P it refuses a draft:** a request that carries
  forced tokens is refused on an engine with speculative decoding (P `sampling_params.py:941-944`; N `:1118-1121`).
  Our replay's engine profile turns that replay on (V `engine/engine_profile.py:144`, `:149`), so a replay with
  drafts at P needs another way to force tokens [argued]. **At T it takes one,** keeping a draft only where it
  equals the forced token (T `v1/worker/gpu/sample/trace_replay.py:137-184`, called at T
  `spec_decode/rejection_sampler.py:309`). For a request whose drafts are the served ones that repeats serving's
  kept count: serving emits the draft exactly where it accepted it, and at a rejection the token it draws has the
  draft removed (T `rejection_sampler_utils.py:990-1000`) or, under greedy, is an argmax other than the draft (T
  `:634`, `:751`) [argued from the readings]. On a stand-in's drafts, which come from zeros, it does not give the
  served count. What I wrote of this hook at 03:49Z holds at T and M and not at P.

**(e) "Until the accept rule is a Definition the kept counts are advice: at most log₂(K + 1) bits a step, 2 at
three, 2.6 at five." Right, read as a decoding request a step and for every request.**
- 2.585 bits a decoding request a step at K = 5, 2 at K = 3 [mine]. A request's total is column 2's: 207 to 414 bits,
  37 to 74 times the length and ending [mine].
- The certified record does not fix them, unlike a length, which the certified output fixes. It holds a release
  bucket for each output (V `pipeline/advice_code/codec.py:15`), and a step's outputs leave together, so the
  buckets bound the grouping and fix it only where no two steps of a request share a bucket [read; argued].
- After the Definition: each request's Program computes its own kept counts and checks them against the record's,
  and the record's are what the compiler builds every request's shapes from. That is the form the tree has for a
  greedy two-model serve: "The dynamic accepted length is a value ... the shapes it would have to drive ... are
  fixed by `A`, and the program's `accepted` outputs are checked against `A`" (V
  `verity/definitions/registry/spec.py:16-21`) [read]. The count carries no bits once a lemma shows the check's
  output unique, which it is: the test is a function of the seed, the position, the logits and the drafts [argued;
  no Lean].
- **The scheduled count is a second number a decode row a step** (b). It is the scheduler's arithmetic, so the
  advice code can give it from the registered rule as it gives a row's admission; it refuses speculative decoding
  whole today (V `pipeline/advice_code/replay.py:23-25`) [read], and until it models those lines the scheduled count
  is charged too, at most 2.585 bits more a decode row a step [argued; mine].

**(f) Not in the five sentences: the page and the hit under `mtp`.**
- **Corrected: when a page's bytes are final.** I wrote that a page is published after acceptance. At both builds a
  block enters the prefix cache when the scheduler allocates its step, for the tokens already final, which is before
  the GPU writes them (P `v1/core/kv_cache_manager.py:553-562`; T `:637-646`); T also publishes at the update what
  the step made final (T `scheduler.py:2467-2471`), and P does not [read; section 0, note o]. A row at a rejected
  position is written and later overwritten (P `scheduler.py:1934-1952`; T `:2392-2410`). So "committed as written"
  in row 16 must read: **the hash is taken once the page is in the prefix cache and the step that wrote its last
  kept row has ended.** A hash at the GPU's write can hold a rejected row, and a hash at the publish can come before
  the write [argued].
- **Corrected 04:39Z: the hit under `mtp`.** My line said vLLM drops the trailing block of every hit under `mtp`.
  That is P. At T and at M no block is dropped and a block's key covers the token after it. Section 0 has each
  build's lines and what the rule changes for the registered record.
- Rows per step: six a decoding request. A step's row count then crosses our two kernel thresholds at fewer
  requests: DeepGEMM's branch (32 rows or more) from 6 decoding requests, and the fused all-reduce norm (42 rows
  or fewer) stops at 8 [mine, from the capture map `:65-71`].

### 4. What is missing in the tree for this serve, beyond entry 2's list

One line each; the owner in brackets. "None" is what `rg` over `verity/` and `adapters/` found at V [ran:
`fp8_ds_mla`, `eh_proj`, `_rejection_kernel`, `murmur`] or what prodserve's table found [relayed]; it is not a claim
that nobody is building it.

**Definitions** [circuits, CIR-246, unless named]
- The FP8 cache's write: the tile scale and the 656-byte row, one a build, since P's rule and T's differ: none.
- The FP8 cache's read: FlashMLA's FP8 sparse kernel at 8 heads padded to 64, all the step's rows in one launch:
  none. `FlashMlaKvRow_v1`, `FlashMlaSplit_v1`, `FlashMlaCombine_v1`, `FlashMlaDecode_v1` exist for DeepSeek V4's
  row only (V `verity/definitions/numerics/flashmla_twin.py:1-13`).
- Plain `fp8`'s row and FlashInfer's sparse kernel: needed only if the engine start picks them; none.
- The draft layer: `eh_proj`'s launch, the shared head and top token, the reuse of top-k between draft passes; its
  block's parts exist for layers 0 to 77 [relayed, prodserve].
- The accept rule under sampling: the ratio test, the residual with one token removed, the bonus token. A greedy
  `Accept_v1{K}` exists for another workload (V `verity/definitions/registry/spec.py:1-21`).
- The sampler's noise at T: the MurmurHash3 Gumbel draw: none [relayed, prodserve: `rg -i murmur`]. P's is Philox,
  which the tree's sampler Definitions are written for (V `verity/definitions/registry/sampling.py:1-17`) [read].
- DeepGEMM: exists, pinned at P; to be pinned again for T; reached from 6 decoding requests under drafts.
- The compile slice: a decode row of K + 1 tokens, the roll-back, the draft count a step gives a request, the
  build's rule for a block's key (section 0): none in `verity/protocols/certifier/prover/compile_slice_v6.py` [relayed, prodserve].
- A kept-count class in the advice code: none, and its charge counts one output a row a step (V
  `pipeline/advice_code/codec.py:434-447`) [read]. Its scheduler model refuses speculative decoding whole (V
  `pipeline/advice_code/replay.py:23-25`) [read].

**Captures** [vllm: VLLM-381 for the build, VLLM-379 for drafts]
- A capture map for this serve: its pins, `FLASHMLA_SPARSE` with `fp8_ds_mla`, the draft layer built (today's map
  `:74` says it is not), CUDA graphs as the line leaves them (the map is eager).
- Launches at six rows a decoding request, in the target pass and the draft layer's first pass.
- The 656-byte row's bytes against the Definition, and one page's bytes at publish against its first hit.

**The step record** [vllm]
- The header: the speculative setting, the cache format, the vLLM build.
- A step's row: the draft count the scheduler gave it; the kept count the GPU saw, or the proof that `emitted` − 1
  is it; the first position the GPU ran.
- For a span: the draft tokens its first step verifies, or the span cut one step earlier (section 0, a).
- The refusals to lift: a step with draft tokens (V `pipeline/boundary_tables.py:230-231`,
  `engine/sampler_capture.py:51-53`, `observe/fold/patterns/sampling.py:81`, `:428`) and a cache that is not BF16
  (V `pipeline/boundary_tables.py:57-58`, `:142-145`: "serve with --kv-cache-dtype auto") [read].
- The table builder's hook takes positions from the CPU's count (V `pipeline/boundary_tables.py:240-241`), which
  under drafts is the upper bound of section 3, b [read].

**The replay** [vllm]
- A stand-in's kept count forced from the record; kept counts compared, not only tokens.
- The replay's engine built with the served speculative setting and cache format.
- A way to force tokens on an engine with drafts at P, where vLLM's token replay refuses one (section 3, d).
- `benchmarks/serve_tax/rebuild.py:869-870` copies one KV cache group; whether the draft layer makes a second is
  not established (M `v1/kv_cache_interface.py:1553-1554`).

**The page commitment** [vllm; not built at any cache type]
- Taken when the page's bytes are final (section 3, f); covering the draft layer's rows and the index keys; the
  reader opening FP8 bytes.

**The guarantee** [proofs]
- The plan check's counting lemma for a shared page and a span (CACHE, section 5).
- The uniqueness lemma for the kept count.

**Outside the circuit, which the line also changes** [vllm]
- The Rust front end and `--tokenizer-mode glm5_next`: what turns the certified request's bytes into token ids. I
  did not read what the tree registers for it. The line runs word for word on no release [relayed, vllm-newpin].
- The context: the recipe's own H200 setting is `--max-model-len 262144` (M `GLM-5.3.yaml:107-109`: "vLLM estimates
  ~450K max tokens, so the 1M default fails at startup") [read]. That is entry 2's recommended limit.

### 5. `f570` and CACHE

- **`r20261112-025327-f570` ended 05:10:05Z, exit 0. Its verdict: the step record's and the canonical top-k's cost
  on this model at saturation is not told from zero, and a cost under about 1% cannot be told by this run; the modeled
  0.25% is neither shown nor refuted** [relayed: read in its launcher's note
  `vllm/20261111T2113Z-checkpoint-vllm-servecost-night`, the 05:15Z and 05:19Z checkpoints; the ratios recomputed
  from its figures, mine].
  - Five arms of six passes each, none cut, every pass 1,024 requests sent and completed, 0 failed, about 1,425
    steps of 143 rows, 204,022 output tokens [relayed]. Output tokens a second: `compiled` 951.8 to 988.0, `record`
    945.3 to 1,028.3, `canonical-topk` 965.0 to 1,005.2, `no-prefix-cache` 974.7 to 999.4, `compiled-again` 975.4 to
    1,020.7 [relayed].
  - **Cost, as stock's output tokens a second over the arm's, less one, by pass; a negative cost is an arm that ran
    faster than stock** [relayed, its reader's lines; recomputed from the note's 30 figures, mine, to within 0.01
    points]: `record` median −0.36%, lowest −7.44%, highest +4.52%; `canonical-topk` +0.20%, −5.32% to +1.88%;
    `no-prefix-cache` +0.35%, −4.76% to +1.05%. **Stock against itself, `compiled-again` over `compiled`: −1.05%, −6.75% to +0.98%.**
    Every range holds zero, and stock differs from stock by more than any arm differs from stock. Without each arm's
    first pass, stock against itself is −0.66%, −1.51% to +0.98%. The job's own line says "not comparable"
    [relayed].
  - **So entry 2's item 9 stays open at the size it was modeled:** "six paired passes at this load do not tell a
    cost under 1% from zero" [relayed, the launcher's words]. What the run does show: each arm's median is within
    0.36% of stock's, inside what stock's second run differs from its first by [mine].
  - **What would settle it** [relayed; guess, the launcher's]: one of node A's eight cards is thermally throttled in
    nearly every sample under load, in this run and in two earlier ones, and that may be the passes' noise; "the
    same passes after infra mends the card's cooling", or more paired passes.
  - **What it does not price** [relayed]: it has no committing arm ("Not run ... `long`, `full`, the two committing
    arms; so no boundary reading"), so the page hash's cost at serving is not in it; and its load has almost no
    hits, 1,664 cached prompt tokens of 222,614 (0.75%), so `no-prefix-cache` says little about the hit. It is at
    the pin with a BF16 cache and no drafts, so it prices entry 2's item 9 and nothing of this addendum's FP8 rows
    or draft tokens.
- **CACHE's final is in** (entries 5 to 7, 03:49Z to 04:21Z, `origin/cursor/proofs-cache-7c3e` at `b2e01c27ba`; its
  closing line of 05:00Z and 05:02Z, at `f0f2bba078`, changes none of them and says that nothing of its is running
  or queued) [read; relayed in the lines under this one].
  - **The line for 07:00Z:** prove `AdmissibleTwoRankinPatterns` without its hypothesis that each request reads one
    replay block (`security/Definitions/Claims/SamplingTwoRankin.lean:79-88`), read with a session as its request.
    `RequestsCover` loses only its first clause (`security/Properties/Spot/TwoStageCheck.lean:58`); its other two
    stay. One proof of 106 lines is redone (`security/Proofs/Flock/SamplingGeneric/TwoRankin.lean:267-372`); CACHE
    gives the argument and has not compiled it. `InfluenceTwoPatterns` does not change, so no ruling is needed. Not
    given: the count across an execution phase's edge, which is an ask to Daniel.
  - **Prices, world, one year, against today's bound of 14,229 GB** [modeled, CACHE's run `r20261112-033048-7551`]:
    the cache off 12,552 GB; the hit read under a session cap of 220,500 served tokens 24,882 GB, 1.7 times, and of
    2,000,000 44,327 GB, 3.1 times; Copilot's session of 31 calls as one replay block 16,080 GB; the hit recomputed
    at AgentX's ratio 552,003 GB, 39 times.
  - **My entry 2's two figures, corrected by CACHE, which I take:** "about 1.5 times" was against the no-repeat
    figure (12,552 GB) and for a session cap of 75,250 served tokens; against today's bound it is 1.34. "About 1,900
    times" is 1,945 times the no-repeat figure and 1,715 times today's bound.
  - **On vllm's two questions CACHE and I answer alike** (its entry 7): (a) yes, with the seed, temperature and
    top-p as ports and the draft tokens as wires, no draft probabilities under the default, no gate figure; (b) the
    kept count derived, the scheduled count registered in round 1's step record and charged at its coded length, 0
    where vLLM's rule gives it. CACHE's form of the check is `ServeSpec_v1`'s schedule `A`: the count in the record
    as a shape, checked equal to the block's own. What I add: a span's first step needs the drafts proposed before
    it; the scheduled count is charged until the advice code models the scheduler under drafts; and CACHE read the
    tag and the pin for this, as section 0 does.
  - Neither CACHE's model nor the figure of record holds a draft token or an FP8 row [read: CACHE's entries;
    argued].

### 6. Not established, and the one run that settles each

All on node A, so the lead's to place; none launched by me. Each is a run at one build, and where section 0 has a
note a result at P says nothing of T. One question is not a run's: whether Friday's serve at P keeps a BF16 cache
and no draft token (section 0); it is vllm's to say, and I was not relayed it.

1. **Which attention backend and cache format the engine picks under `--kv-cache-dtype fp8`, and whether the draft
   layer is a KV cache group of its own:** one engine start with the line's flags; the log.
2. **Zero stand-ins at an FP8 cache, and a hit read from FP8 pages:** `rebuild.py --served --shape-only RID --dummy
   zero` on a target with a hit, the FP8 flag carried into the replay's engine, no drafts.
3. **A step with draft tokens replayed bit for bit, first all real, then with stand-ins forced to the recorded kept
   counts:** one served pass with `mtp` at K = 5 and its replay. It needs the forcing code first.
4. **Whether the trace's first position and `emitted` − 1 are what the GPU ran under async scheduling:** the same
   served pass, the trace read against the runner's `num_sampled` and positions.
5. **Whether a page's bytes, hashed when the step that wrote its last kept row ends, are its bytes at its first
   hit, the draft layer's rows included:** the same pass, hashing a page at both.
6. **The acceptance length and the step time on this node at production sampling,** which set column 3 and what
   the kept counts cost as coded advice: the same pass.
7. **How much of the kept counts the certified release buckets already fix,** which is what they cost as advice
   after coding: the same pass served through the network certifier, its record's buckets against the trace.
8. **What node A's engine resolves the step's token budget to, and how often it cuts a row's drafts,** which is what
   the scheduled count costs while the advice code does not model it: the same pass, the trace's row lengths.
9. **The accept test's gate count a decode row** (section 0, a): no run on a GPU; the accept Definition under
   sampling at a vocabulary of 154,880, counted by `circuit-check`.
10. **The step record's and the canonical top-k's cost at saturation, below about 1%** (section 5; entry 2's item
   9): `f570`'s arms again once node A's throttled card is mended, or with more paired passes [relayed; guess, its
   launcher's].

## Entry 3 and final, Thu 12 Nov 13:28Z (due 16:00Z): the comparison as final with the kept drafts' row, the route and what it needs by lane, the message for Daniel

**Tags.** [run]: a command I ran on this VM, named (git, `rg`, arithmetic on a file); nothing ran on a GPU. [read]: code
or a doc at the revision named, with file and line. [relayed]: a run's result or a fact I took from a Linear comment, a
note or the coordinator's message, with who and where; I did not read it in the run's own record. [derived]: my
arithmetic or argument, with its inputs. [guess]. Entry 2's [M run] and [m] are [relayed] in these terms (I read them
in notes), its [mine] and [argued] are [derived].

**Revisions** [run: `git rev-parse`, `git diff --stat` over the paths cited]. Our tree: `main` at `71a2afe671aa`
(fetched 13:02Z), equal on every path cited here to the checkout I read, `5456b937e7e5`. vLLM: the pin `d9105ea8001e`
(P), the build Friday serves. Entry 2 and the addendum read our tree at `be9a3d5669`. Paths under
`adapters/vllm/verity_vllm/` are written from `pipeline/`, `engine/` or `program/`; `draft_table`'s file is
`benchmarks/network_e2e/encodings/replay.py`.

### 0. The published line's flags first

1. **What Friday serves** [relayed: vllm lead on VLLM-377, 06:01Z; the coordinator's 06:16Z]: vLLM's published line for
   these cards, at the pin: the FP8 KV cache, five draft tokens a decode row (`mtp`, K = 5), the prefix cache on,
   `--max-model-len 262144`. Wednesday's settings (bf16 cache, no draft) are the last rung, printed only if the
   published line's replay differs.
2. **Shown at those flags** [relayed: vllm lead on VLLM-377, 07:22Z, 09:20Z, 12:26Z]: a replay of the whole pass from
   an empty engine (row 1) gave back every token: 3,406 of 3,406 in 12 requests (`r20261112-065751-cb56`; one step
   differs and no token), then 23,363 of 23,363 in 32 requests over 2,040 steps, 25 of them starting from a
   prefix-cache hit, the longest 102,336 tokens (`r20261112-074230-88d0`, the first end-to-end try). Every request was
   greedy. The limit was 262,144 and the load cut requests at 131,072.
3. **Not shown at those flags** [relayed, the same comments, unless marked]: a sampled request on GLM-5.3 (section 5);
   a replay of one request among stand-ins (rows 5, 6, 16), which has not run with a draft and has no code for a
   stand-in's kept count (section 4) [run]; a start inside a pass; a request past 131,072 tokens; any commitment
   beside serving: with the FP8 cache the serve refuses our history tables (`pipeline/boundary_tables.py:141-145`)
   [read] and Friday's run is served without them; the cost:
   +0.09% by the users' token rates, +3.3% by the seconds a pass took, +0.25% by the users' seconds, one server
   lifetime an arm, "under 1%: not shown".
4. **Certified, and at which flags** [relayed: the coordinator's 12:48Z, from vllm on VLLM-242; the rule read by me,
   `pipeline/advice_code/lengths.py:1-15`]: the certified run of record serves with the **draft off**. In its
   rehearsal (one card, Llama 3.2 1B, 2,000 requests, the prefix cache on) hits, admission and resumes cost 0 bits
   with each event occurring, and each request's output length and finish cost 11,186 bits in all.
5. **At Wednesday's settings, since entry 2** [relayed: vllm-advicezero on VLLM-319, 06:01Z and 06:40Z]: a
   prefix-cache hit of 9,088 tokens, twice, at a context of 40,960 (`r20261112-053510-0a4c`) and one of 120,704
   tokens at 131,072 (`r20261112-061536-f368`) were replayed bit for bit both ways: with every page of the hit
   recomputed inside the replay (row 6) and with the hit's pages overwritten from serving's bytes (row 5). Compared:
   the target's tokens, its logprob bits and every step's schedule up to its last. One serve each.

### 1. The answer

1. **The route is unchanged: row 16** [derived]. Replay only the sampled request's own computed positions (a span for
   a long request), in their served step shapes with zero stand-ins, on an engine kept up, reading the cached prefix
   and its own earlier positions from KV pages serving committed: under the published line, the FP8 row as stored.
2. **What moved for it since entry 2** [relayed, items 2 and 5 above]: its largest open showing closed at Wednesday's
   settings (a hit of 120,704 tokens read from serving's pages); at the published line's flags only row 1 has run.
3. **The kept drafts are the one class of advice that grows with output tokens and not with requests** [derived].
   They are in no route's certified form today: advice v1 has no field for them and its replay refuses a draft serve
   without them (`advice_code/codec.py:358-365`, `advice_code/replay.py:99-105`) [read]. So **no route can be
   certified with a draft today**, and Friday's certified zero is at draft off.
4. **Your figure is confirmed as the fixed-width charge and is not the least** [derived, section 4]: 3 bits a decode
   row that carried five drafts; 2.585 packed; never more than 1 bit an output token; 1.84 bits a row at the measured
   entropy and 1.36 by the code vllm built, both on one greedy pass.
5. **It does not separate the routes** [derived]: every route's circuit takes its step shapes from the registered
   compiler, which reads the certified record and the advice, and the count is in neither. The routes differ in
   whether the replay can run with a draft at all, and in who must force a stand-in's count.
6. **The replay did not differ in vllm's first end-to-end try** [relayed, item 2], so no FP8 cause was needed.
   Section 5 has the reading you asked for all the same, at the pin [read]: under the FP8 cache the attention backend
   is another than Wednesday's, every row but a full decode row of 1 + K tokens is a prefill row to it, and such a
   row on the MHA path reads its own step's keys unquantized where every later step reads them as stored. The
   14:50Z sampled row had not started when I wrote this.

### 2. The table, final

One row a route, entry 2's numbering. Entry 2's cells stand in full; this table restates each row's
deciding figures and adds what was shown since. **W** is Wednesday's settings (bf16 cache, no draft), **F**
the published line's flags. Every figure not marked otherwise is [relayed] as entry 2 cites it; the F and "since"
cells are [relayed] from section 0's comments.

| # | Route | At serving beyond stock, against 1% | Replay for one sampled request; with the repeated share and the context | Shown bit for bit: W, then F | What the guarantee charges | Verdict |
|---|---|---|---|---|---|---|
| 1 | The whole execution phase from an empty engine, every request real (`benchmarks/serve_tax/replay.py`) | The step record and the token ids: −0.3% [−3.3, +2.6] at saturation (`3f53`); −0.36% and not told from zero (`f570`); at F the three readings of item 3 | Every step of the phase: 5,258 GPU-s on P128 at 600 s phases. Grows with both | W: 11,350 steps, 184,922 tokens (`cf6c`). **F: 23,363 of 23,363 tokens, 2,040 steps, hits to 102,336 tokens, greedy (`88d0`)** | Closed only if nothing is carried in; its interior is every request's | The only route run at F. It starts only from an empty engine, so under traffic that never drains its block is the server's life |
| 2 | The phase from a snapshot at its edge (`restore.py`, `tp_restore.sh`) | Row 1's, plus a pause at each edge: 0.16% to 0.25% of a 600 s phase | Every step of the phase at 0.83 times serving. Flat in the share | W: 226 steps (`f460`). F: not run | As row 1 | A sampled request still costs its whole phase |
| 3 | From an edge, pages from host copies (`rebuild.py:10-31`) | Row 2's, less the runner's state | Row 2's | W: one process. F: not run | As row 1 | As row 2 |
| 4 | The request's lifetime from pages copied as written, every request real (`rebuild.py:33-42`) | A copy and a hash of each page as it fills: 0.036% to 0.045% of a card, modeled; not built | 458 GPU-s on P128 (1,160 at the registered limits). Flat in the share | W: one GPU only. F: not run | As row 5 | Row 16 without the stand-ins: loads the others' pages |
| 5 | One request, the others zero stand-ins, its hit pages read from serving's bytes (`rebuild.py --served --shape-only`) | Row 1's, plus the target's pages kept and committed | 19 to 32 s on eight cards, P40. Flat in the share; pages loaded grow with context | W: 4 of 4 targets; hits of 256 and 384 tokens; **since: 9,088 tokens (142 pages) and 120,704 tokens (1,886 pages), no step and no position differs (`0a4c`, `f368`)**. F: not run; no code forces a stand-in's kept count [run] | Not closed in the form the figure of record uses: it reads values another replay block computed | Row 16's replay, shown at W to a 120,704-token hit |
| 6 | The same with the hit recomputed (#8403); what `main` does | Row 1's only | Row 5's plus the hit: 1.13 times the tokens at 11.62% repeated, 7 to 59 times at 85.67% to 98.31% [derived, entry 2]; at AX the conversation so far | W: hits of 128 to 384 tokens; **since: 9,088 twice and 120,704 with every page recomputed in the replay (`0a4c`, `f368`)**. F: not run | Closed, one replay block a request: the form on `main` | Sound today, and dear exactly where traffic is agentic |
| 7 | The request on the CPU from its Program (`check/replay/request_replay.py`) | Row 1's | 84,500 core-hours for a 131,072-token request | Exact by definition; never run at full depth. F: no Program of a step with drafts exists [run] | As the Program it evaluates | The reference, not a route at size |
| 8 | The request alone under batch-invariant kernels | 14.1% to 31.1% on three other models: over 1% | The request's own tokens | None exists for this model's sparse attention | As row 5 or 6 | Not available |
| 9 | A span reading per-layer history tables (`pipeline/boundary_tables.py`) | 2.3% to 4.3% of serving's GPU time with today's builder [derived, entry 2] | Not measured: no span replay exists | Not shown. **F: refused: the tables hold bf16 words and refuse an FP8 cache (`:141-145`) and a step with draft tokens (`:231`)** [read] | As row 5, its readers its own request | The span's idea stays in row 16; the tables as built are not at F |
| 10 | Stand-in interior: one token-select check a served id (`pipeline/interior_rows.py`) | The two-stage arm: +7.16% over three paired repeats, 194 of 3,272 requests committed (`03bb`); since: it serves with the prefix cache on where it builds no tables (#8897, on `main`) [run: `git log`] | Whatever replay gives the logits | Holds ids to logits, not logits to the model. **F: refused: the sampler capture refuses a step with draft tokens (`engine/sampler_capture.py:51-53`)** [read] | A stand-in: it audits the token select alone | Not an audit of the model's computation |
| 11 | The hit held to a committed pool by the certifier's cache step | A pool commitment a phase | Not a replay | n/a. F: the certifier's step does not model a draft (`verity/protocols/certifier/verifier/replay_step.py:81`) [read] | It proves which pages a request hit, not what they hold | Pairs with row 5 or 16 |
| 12 | Hits allowed only by a rule on the certified record | Nothing | As its route | n/a. Since: the load's sessions share no cache block, by the corpus's construction (block ids are per session); a cache shared across sessions could add at most 3.82% (256k copy) or 1.69% (full corpus) of prompt tokens (VLLM-377, 09:20Z) | Caps the readers of a page | The readers of a page are one conversation in our load |
| 13 | Hits sent as advice | Nothing | None | n/a | 5,181 bits a phase | Superseded: the scheduler replay derives hits |
| 14 | A conversation as one replay block | Row 1's, the cache keyed by conversation | The conversation's computed tokens once | Not shown | Closed; 1.18 to 1.42 times at 4 to 86 requests | Its interior at 131,072 positions is 7,462 GiB a rank: not feasible as one |
| 15 | Prefix caching off | Serving recomputes every repeated token: 13.86% more FLOPs on P128, 7 to 59 times at agentic reuse | The request's lifetime | Nothing is shared | Closed | Ruled out by Daniel |
| **16** | **Recommended: row 5's replay, cut to a span, started at the request's first step on an engine kept up, reading committed KV pages** | Row 1's, plus a hash and a host copy of each page as it fills: 0.036% to 0.045% of a card, modeled; **not built, not measured beside serving**. At F the page is the FP8 row as stored, 656 bytes a token a layer, hashed once the page is in the prefix cache and the step that wrote its last kept row has ended [derived, addendum 2 and 3f] | The span's steps: 19 to 32 s at P40 for a whole request; at AX at least 444 decode steps without a draft, about 99 with five at the try's acceptance [derived: 1 + 443 / 4.5365]. Flat in the share; tokens recomputed flat in context | W: its replay, to a 120,704-token hit (row 5). Not shown at W: a start inside a phase at TP8, a span, an agentic load. F: nothing | Row 5's; one statement to add (section 6) | The only route flat in the repeated share |
| 17 | The floor: the request's rows alone on a GPU, kernels forced to the served step's settings | Row 16's | The request's own compute, about 1 time | Not built | Row 16's | Not built |

### 3. Advice in bits a request, by class, final

| Class | Bits a request | What holds it at 0 |
|---|---|---|
| Cache hits | 0, with hits occurring, in the certified rehearsal (`r20261112-112802-8e98`) [relayed: the coordinator's 12:48Z] | Done: the scheduler replay derives every hit (`advice_code/codec.py:26-29`, `:39-45`) [read] |
| Admission | 0 under the certified tick, with admissions occurring [relayed, the same] | The certified tick (`codec.py:50-55`) [read]. No compile slice holds the tick's rule on GLM-5.3 yet [relayed, entry 2] |
| Resumes | 0, with resumes occurring [relayed, the same] | The scheduler replay |
| Release buckets | 0: certified, set by the network certifier's fixed schedule (`codec.py:15-16`) [read] | Done |
| Finish | 1: 2,000 bits over 2,000 requests [relayed, the same] | A proof of the length rule (`advice_code/lengths.py:4-6`) [read] |
| Output length | ⌈log2(max_tokens + 1)⌉ (`lengths.py:7-8`) [read]: 9,186 bits over 2,000 requests, 4.59 a request [relayed, the same] | The same proof |
| Aborts | No field (VLLM-391) [relayed, the same] | Not designed |
| Seeds | 0 for a request that sends its seed [relayed, entry 2] | A registered rule deriving the seed from the certified request |
| Scheduled drafts (how many a row carried) | 0: the scheduler replay derives it, `Step.drafts` (`advice_code/replay.py:35-43`) [read]; held to vLLM's scheduler at K = 1, 3 and 5 (#9143, on `main`) [run: `gh pr view 9143`] | Done in the adapter's replay. Not in the certifier's step circuit |
| **Kept drafts, K = 5 (how many of them a row kept)** | **Not in advice v1: a draft serve is refused, so no certified figure exists.** Charged at fixed width: 3 bits a decode row that carried drafts; 293 bits a request on AX's median, 188.5 on our 128k plan, 0.66 bits an output token on both, at the try's acceptance [derived, section 4]. What `draft_table` prints today, uncertified: 295.7 bits a request, 0.405 an output token, on the try's greedy pass [relayed: VLLM-384] | The accept rule computed in each request's replay block with the count its output, and a lemma that the count is unique (section 4) |

### 4. The kept drafts at K = 5

**What the count is** [read: `advice_code/replay.py:35-43`, `:327-395`]. A decode row is its token and then the
drafts its request holds, K unless a budget or the context limit cuts the row. Its output is its token and the drafts
it kept; the rest roll back. How many it carried is the scheduler's arithmetic and is derived. "How many it kept is
the model's choice, so it is an input, `RequestIO.kept`, as where the request stopped is." It is no free choice of
the prover: it is a function of the row's logits, its drafts, the seed and the position (P
`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:569`, `:587`, `:662`) [read], so a block can compute it.

**Why the certified record does not fix it** [read: `advice_code/codec.py:15-16`]. The record holds each output's
release bucket, and the network certifier's fixed schedule sets that bucket from the request's own ingress bucket and
the output's index. So which outputs left one step together is not in what leaves. The addendum's line that the
buckets bound the grouping (3e) is withdrawn. network-accounting's worker reads the same [relayed: WATERLINE,
`network-accounting/20261112T0652Z-plan-advice-draft-tokens`, section A].

**Why every request's count is needed, not only the sampled one's** [derived, addendum 3b]. A kept count moves its
own request's positions in every later step, so that row's key count and the slots it writes, and the step its
request finishes in. A row's key count is among the shapes a batch-mate's bits depend on. So the sampled request's
kernel shapes read every batch-mate's counts.

**The charge while no proof holds the accept rule.**

- **Your 3 bits a decode row: confirmed as the fixed-width charge** [derived]. It is the tree's convention for a
  length (`lengths.py:7-8`: `width(max_tokens + 1)` bits) [read]. Exactly: `width(d + 1)` bits for a row that carried
  d drafts, d derived. That is 3 bits at d = 4 or 5 and 0 at d = 0, so a request's first decode row is free where it
  carries none.
- **It grows with output tokens, not with requests: confirmed** [derived]. A request of n output tokens has
  (n − 1) / (1 + its mean kept count) rows that carried drafts.
- **It is not the least valid charge** [derived]. Packed, a row is log2 6 = 2.585 bits. And a request's counts are
  fixed by marking which output tokens start a row: n − 1 bits, so never more than 1 bit an output token (0.988 by
  counting the ways to cut n tokens into rows of 1 to 6). Fixed width overstates wherever a row keeps fewer than two
  drafts on average.
- **Measured, greedy only** [relayed: vllm-advicedraft on VLLM-384; vllm lead on VLLM-377, 09:41Z]. The try's
  replayed pass: 5,143 rows that carried drafts, their counts 665, 553, 442, 257, 150, 3,076 for 0 to 5 kept, in 32
  requests of 23,363 output tokens. Entropy 1.840 bits a row. A code vllm built and decoded back sends 6,997 bits,
  1.360 a row, because counts come in runs (#9279, open, not on `main`) [run: `gh pr view 9279`].
- **From those counts** [derived: 23,331 tokens over 5,143 rows]: 4.54 output tokens a row that carried drafts, 70.7%
  of five drafts a row kept. The other pass at F, `cb56`, gives 4.44 (3,394 tokens over 765 rows).

**On the two shapes** [derived: rows = (n − 1) / 4.5365, the try's greedy acceptance; AX's median output of 444
tokens and median prompt of 142,016 from entry 2; the plan's 651 requests, 186,242 output and 2,235,146 prompt tokens
from `benchmarks/serve_tax/prefix_plan.jsonl`, run: a sum over the file]. No run is at either shape with a draft, and
acceptance under production sampling is not measured.

| | AX's median request (444 outputs) | The 128k plan (651 requests, mean 286.1 outputs) | An output token, both |
|---|---|---|---|
| Rows that carried drafts | 97.7 | 62.8 a request, 40,911 in all | 0.220 |
| **3 bits a row (fixed width)** | **293 bits a request** | **188.5 bits a request**, 122,733 in all | **0.66 bits** |
| 2.585 bits a row (packed) | 252 | 162.4 | 0.57 |
| 1.840 bits a row (the try's entropy) | 180 | 115.6 | 0.405 |
| 1.360 bits a row (vllm's code on the try) | 133 | 85.5 | 0.30 |
| Every draft kept, 3 bits a row | 222 | 142.5 | 0.50 |
| No draft kept, 3 bits a row; capped by marking row starts | 1,329; 443 | 855; 285 | 3; 1 |
| A served token, prompt and output, at 3 bits a row | 0.0021 bits | 0.051 bits | |
| Against the length and finish, 5.593 bits a request in the rehearsal | 52 times | 34 times | |

**Over a monitoring period** [derived; the rate is scaled, nothing here is proved]. The charge is a rate an output
token, so a period's total is the rate times the period's output tokens, for one datacenter as for all compute. At 3
bits a row and the try's acceptance it is 0.66 bits an output token, 3.8% of the 17.24 bits an output id carries (a
vocabulary of 154,880). For one node at 446.7 output tokens a second, the only rate of node A I have read (row B100 of
`r20261108-200706-1758`: no draft, cache off, 1.67 requests a second, not the published line's rate): 1.78e8 bits a
week, 7.76e8 a month of 30.44 days, 9.31e9 a year. A datacenter's and the world's totals multiply by their output
tokens, which I did not read from the census.

**Route by route.** The second and third columns are common to every route but where a row says otherwise, so they
are stated once:

- **Charged** [derived]: 3 bits a decode row that carried drafts, for every decoding request of the execution phase,
  as tabled above. It is charged whole until a proof holds the accept rule.
- **What makes a draft serve certifiable at all, charged** [relayed: VLLM-392, filed 12:59Z; WATERLINE's
  note, table B]. vllm's code: advice v1's record carries each row's kept count, `codec._ios` hands them to the
  replay, and the count is a class of the certified table (VLLM-392, owner vllm-advicedraft). network-accounting's
  code: the advice's form in `verity/protocols/certifier/verifier/advice.py` with a new version, the step's reference,
  a step circuit `ReplayStep_v2{…, K}` whose inputs gain `kept`, the state chain and the compile slices, and the
  charge beside the length terms (NET-358, NET-361; #9198, open, makes today's binding refuse a draft by name).
- **What holds it at 0**, by whose code [derived, from AGENTS.md's rule that a relation check stands in for
  computing only where a lemma shows its outputs unique]:
  - **circuits' Definitions**: the draft layer (the model's layer 78, in no capture map) and the accept test. The
    tree has the greedy two-model form: `Accept_v1{K}` and the static schedule its `accepted` output is checked
    against (`verity/definitions/registry/spec.py:6-9`, `:16-21`, `:64-73`) [read]. The sampled test is not built:
    the ratio test on the position's uniform and the draw from the residual (P `rejection_sampler_utils.py:662`,
    `:812-823`) [read].
  - **vllm's step compiler**: a request's Program for a step with drafts, whose count output is held equal to the
    record's. It has none [run: `rg -i 'draft|specul' adapters/vllm/verity_vllm/program/predict/steps.py`, no match].
    A sampled request also needs a certified seed [relayed: VLLM-392].
  - **proofs' Lean**: the lemma that any two count sequences the checks accept are equal, by induction over steps;
    and that a count which fails its check in an unsampled replay block is a wrong block.
  - **One thing is new to the accounting** [derived]. The compiler still needs the counts to build the shapes, and
    it cannot derive them as it derives hits, since they read the model's values. So they stay sent, and AGENTS.md
    charges advice at its length. The 0 needs a class the rules do not have: advice that a block's output is held
    equal to is charged 0, and a mismatch is a wrong block. A registered cap of 0, which is how the hits' class is
    off today and how WATERLINE's note turns this one off [relayed: its section C], is not that: under a cap of 0
    nothing is sent.
- **A cheaper check for the same count** [derived; read at P]. A row's count is the length of the longest prefix on
  which its drafts equal the tokens its request emitted from that row on, capped by the drafts it carried and by the
  request's end. Under greedy a draft is kept only when it equals the target's argmax and the token at a rejection is
  that argmax (P `rejection_sampler_utils.py:587-594`). Under the sampled rule the token at a rejection is drawn with
  the rejected draft removed (P `:812-823`). So the check needs the row's draft tokens and the certified output ids, K
  comparisons of token ids a row, and no logits. It still needs the draft layer in the circuit: the draft's tokens
  are in no record.

| Routes | Where the count comes from in the replay | What differs in the charge or in what holds it at 0 | With a draft today |
|---|---|---|---|
| 1, 2, 3, 14 (the engine again, every request real) | Recomputed, for every request, by the replay's own accept test on its own logits and drafts; nothing carried. The comparison holds it through each row's first position and each request's token count, not through the step's emitted counts (`benchmarks/serve_tax/replay.py:231-243`) [read] | Nothing. The replay needs no count, but the circuit's shapes do [derived] | Row 1 runs and is shown at F, greedy, uncertified. Rows 2, 3, 14: not run with a draft in anything I read. Not certifiable: advice v1 has no field |
| 4 (the lifetime, every request real) | Recomputed for the rows it re-runs; a row before its start is in the positions and pages it starts from [derived] | Nothing | Not built |
| 5, 6, **16**, 17 (one request among zero stand-ins) | The sampled request's own: recomputed. Each stand-in's: read from the committed step record (a step's emitted count less one, `engine/step_trace.py:268-282`) [read] and forced, since a stand-in's values are zeros and its own test gives another count [derived] | vllm's `rebuild.py` must force each stand-in row's count (roll back d − kept, emit 1 + kept, as `advice_code/replay.py:340-395` does on paper). At P vLLM's own token replay refuses an engine with drafts (P `vllm/sampling_params.py:941-944`) [read, addendum 3d] | Not run. No code forces a count [run: `rg -i 'draft|specul' benchmarks/serve_tax/rebuild.py`, no match]. Not certifiable |
| 7 (the CPU, from the Program) | Carried: the Program's shapes are the compiler's, so the count is advice, or the record's count held by the block [derived] | Nothing | Cannot run: no Program of a step with drafts; the CPU's sampler recompute skips a drafting row (`check/replay/stoch_recompute.py:553`) [read] |
| 8 (batch-invariant kernels) | Would recompute its own and need no other request's. Its own counts still set its circuit's shapes (`spec.py:19-21`) [read] | The same charge, by request | No such kernels |
| 9 (a span from history tables) | As row 16 for the span's rows [derived] | Nothing | **Cannot be certified, or served, with a draft: the tables refuse a step with draft tokens (`boundary_tables.py:231`) and an FP8 cache (`:141-145`)** [read] |
| 10 (stand-in interior) | Takes the count of the engine replay under it. Its own check, one token select a served id, holds no count; under sampling a kept draft is not a draw from its row [derived] | It cannot hold the count at 0 at all | **Cannot be certified with a draft: the capture refuses a step with draft tokens (`sampler_capture.py:51-53`)** [read] |
| 11 (the certifier's cache step) | Not a replay. In the step circuit the count would be an input, bitlen(K) bits a running slot [relayed: WATERLINE's note, table B row 4] | With a draft a hit gives up its last block (`advice_code/pool.py:87`, `:154`) [read]; `cache_step.py` is among the files that note lists to change [relayed: the same note, row 5] | **Cannot: the step does not model a draft (`replay_step.py:81`)** [read] |
| 12, 13, 15 | None of their own | Nothing | As the route each pairs with |

**What `draft_table` charges today, on the production row** [read: `benchmarks/network_e2e/encodings/replay.py`].
`kept_bits` (`:807-830`) replays the trace, refuses one the replay does not reproduce (`:819-821`), counts every row
that carried drafts and emitted (`:822-826`: a row's count is the step's emitted count less one), and returns
`charged = rows × entropy`, the entropy being that of those same counts (`:828-829`), beside `max = log2(K + 1)` and
`derived = 0` (`:829-830`). `draft_table` (`:833-848`) puts it beside the arrivals' bits, and the report prints the
row (`:908-918`).

- On the try's greedy pass that is 5,143 × 1.840 = 9,463.5 bits, 295.7 bits a request and 0.405 bits an output token;
  on `cb56`, 107.5 and 0.379 [relayed: VLLM-384; the products are mine].
- It is uncertified and from another run than the certified one [relayed: the coordinator's 12:48Z].
- It is an entropy, not a code's length [derived]: the distribution is fitted to the trace it charges and no codebook
  is registered before the run. #9279 replaces it with a code that is built and decoded back.
- It reads 0 as `derived` although no Program derives the count [read: `:830`; relayed: vllm lead, 09:41Z, "no
  Definition holds the accept test"].

### 5. The read of VLLM-377, and where the FP8 cache's read first enters at the pin

**The read** [relayed: vllm lead on VLLM-377, read through the Linear tool at 13:05Z]. The
first end-to-end try, `r20261112-074230-88d0`, ended 08:53Z with exit 0 and the replay did not differ: 23,363 of
23,363 tokens, all greedy. Of the four causes in the 06:11Z comment, the fourth was the FP8 cache: "a position's key
is read as two different values, by the step that writes it and by every later step". The first sampled serve and replay
under drafts on GLM-5.3 is today at 14:50Z (12:26Z); it had not started when I wrote this. I read VLLM-377 again after
it and append what it says.

**The reading, at P** [read, each line at `d9105ea8001e`], given so that it is at hand if a sampled replay differs. In
one line: vllm's fourth cause is in the code. The two values exist for a row the attention treats as a prefill row,
in a step that takes the MHA path; which rows and which steps those are is fixed by the step's row lengths and order.
Paths are under `vllm/`; `attention.py` is `models/deepseek_v32/attention.py`, `mla_attention.py` and
`sparse_mla_attention.py` are under `model_executor/layers/attention/`, `utils.py` is
`v1/attention/backends/utils.py`.

1. **The write comes first in each layer's call.** `attention.py:340-359`: `fused_norm_rope` writes the step's latent
   rows into the layer's cache in the cache's format (`mla_kv_cache_dtype`) and returns the step's own rows before
   any quantization.
2. **The backend is another than Wednesday's** [derived: I read no server log]. On an H200 the platform's list is
   FlashAttention's sparse backend, then FlashMLA's (`platforms/cuda.py:136-152`). The first takes only a bf16 cache
   (`v1/attention/backends/mla/flashattn_mla_sparse.py:36-40`); the second takes the FP8 one
   (`v1/attention/backends/mla/flashmla_sparse.py:110-117`). So W ran the first, as our capture map records, and F
   runs FlashMLA's.
3. **Which rows the attention treats as decode rows changes with it.** FlashMLA's sparse backend requires uniform
   decode rows (`flashmla_sparse.py:265`; the others do not, `sparse_mla_attention.py:117`), and its threshold at 8
   heads a rank is 256 tokens (`flashmla_sparse.py:279-292`, `config/model.py:1549-1561`). The split
   (`utils.py:803-839`, called at `sparse_mla_attention.py:273-278`): a step whose rows all carry the same number of
   tokens, at most 256, is all decode rows. Otherwise the decode rows are the leading rows that carry as many tokens
   as the first row, and from the first row that carries another number on, every row is a prefill row, whatever its
   length. The runner puts first the rows that hold drafts and carry the decode length, then the other rows that hold
   drafts, then the rest by length (`v1/worker/gpu/model_runner.py:2209-2221`). So with drafts the decode rows are
   the rows of 1 + K = 6 tokens. A decode row that a budget or the context limit cut short, a decode row that holds
   no draft, and every prefill chunk, of any length, are prefill rows for the attention.
4. **A decode row reads every key from the cache, its own included**: `attention.py:530-533`
   (`forward_mqa(mqa_q_arg, kv_cache, …)`), and in a step that also has prefill rows `mla_attention.py:1037`.
5. **A prefill row on the MHA path reads its own step's keys as the step computed them, not from the cache.**
   `attention.py:496-508` hands `forward_impl` the step's `kv_c` and `k_pe`; `mla_attention.py:923-932` passes the
   prefill rows' part to `forward_mha`; `mla_attention.py:3081-3085` builds the keys from them. On an H200 they stay
   unquantized: converting a prefill's keys to FP8 needs another device class (`mla_attention.py:1806-1811`,
   `:3087-3089`). **This is where the two values part: the same position is read unquantized by its own row in the
   step that wrote it, and as stored FP8 bytes by every later step.**
6. **The same row reads its earlier positions from the cache.** With a context, an earlier chunk of its prompt or a
   prefix-cache hit, `forward_mha` calls `_compute_prefill_context` (`mla_attention.py:3104`, `:3118-3120`), which
   for this cache format gathers the FP8 bytes and converts them up (`:2833-2834`,
   `cp_gather_and_upconvert_fp8_kv_cache`). **So the FP8 read first enters a request at the first of: its first row
   with a context (a second chunk, a row after a hit, any decode row), or a row in a step that does not take the MHA
   path.** A prompt prefilled in one step on the MHA path with no hit emits its first token without one FP8 read;
   its next row reads every earlier key as FP8.
7. **Whether a step takes the MHA path is set by the step, not by the row.** Dense MHA when the longest sequence
   among the step's prefill rows is at most `index_topk` (`sparse_mla_attention.py:290-292`, `:307-310`; 2,048 for
   GLM-5.3, our copy of its config, `GLM53.config.json:22`). Masked MHA by a table over that longest sequence and
   the longest prefill row, where the masked kernel is available and its workspace fits (`mla_attention.py:1113-1136`,
   `:1769-1793`). For GLM-5.3's head sizes (256, 256; `GLM53.config.json:207-208`, `:222`) on FlashMLA's backend at
   eight ranks (`mla_attention.py:1732-1740`): a step with no context up to 114,688 tokens, any chunk; a step with a
   context only where its longest prefill row is at least 4,096 tokens (to a sequence of 8,192), 8,192 (to 16,384),
   16,384 (to 32,768) or 32,768 (to 49,152). Otherwise every row of the step reads the cache
   (`mla_attention.py:894-897`).
8. **At W the two values do not exist, and our registered class's reason does not carry to F** [derived]. At W the
   cache holds the bf16 words the step computed. And the registered class caps a prefill chunk at 256 rows, "so each
   prefill chunk stays on DSA's decode route at 8 heads a rank" (our tree, `engine/profiles/engine_classes.py:5-6`)
   [read]: true of the bf16 backend, whose split is by length alone. Under FlashMLA's split a chunk of 256 rows or
   fewer is a prefill row unless it carries exactly the decode rows' length, and the published line caps no chunk
   [relayed, entry 2]. So a registration at F needs its own statement of which row takes which read.
9. **What follows for a replay** [derived]. Which value a row reads depends on its step's row lengths and their
   order, on how its prompt was cut into steps, and on what it hit. All are the schedule, so a replay that repeats the
   schedule reads the same values. That holds for row 1, as the try showed. It holds for row 16 under three
   conditions. A span starts at a step's edge, never inside a prefill step. The page's leaf is the stored FP8 row,
   not the bf16 row the step computed. And each stand-in row carries its served length in its served place, which
   with drafts needs its kept counts forced (section 4): a kept count moves where a budget cuts a later row, and so
   which rows of that step are decode rows. A prefix prefilled again in other chunks is not bit for bit. For the
   circuit it means two reads of one position: the Program of a prefill row on the MHA path reads its own step's
   rows as computed, and every other read is of the stored row; the compiler has the lengths and order that decide
   which.
10. **Not read**: node A's server log for the backend it picked; the draft layer's own attention in its later passes;
    whether the indexer's key cache behaves the same way; the order of rows that tie in the runner's sort beyond
    Python's stable order. The runner's trimming of drafts by measured timings is off by default and offered to
    another draft method only (`config/speculative.py:539`, `:1519`), so it does not enter here.

### 6. The recommended route, and what it needs by lane

**Row 16** [derived]. It is the only route whose replay, interior and proving stay flat as the repeated share goes
from 12% to 98%, and it leaves serving stock but for a page hash and a host copy. Friday's showing is row 1's: its
pass starts from an empty engine, which a benchmark's does and a server under traffic does not.

- **vllm's serve** [derived, on the lines read above].
  - Commit each KV page when it fills and keep a host copy until the coins: at F the stored FP8 row, taken once the
    page is in the prefix cache and the step that wrote its last kept row has ended. Not built. With the FP8 cache
    the history tables are refused, so this page leaf is the only commitment the published line has a design for.
  - The advice's form carries each row's kept count (VLLM-392). Until then no draft serve is certified.
  - Hold the count directly in the replay's comparison: it compares rows and tokens, not the emitted counts
    (`benchmarks/serve_tax/replay.py:231-243`).
- **vllm's replay** [derived]. `rebuild.py --served --shape-only` needs a start inside the phase at TP8, a cut to a
  span at a step's edge, one engine kept up, and, with a draft, a forced kept count for every stand-in row. None of
  the four is built.
- **circuits' Definitions** [derived]. The draft layer and the accept test, greedy and sampled; the FP8 cache's row,
  written and read, with both reads of section 5 (a prefill row's own step unquantized on the MHA path, everything
  else as stored); a Definition that opens a committed page; the size class's replay Definition on the two-stage
  path; ranges to 262,144. Entry 2's list stands otherwise.
- **network-accounting's certifier** [relayed: WATERLINE's note, table B]. The advice's form with the kept count, the
  step circuit with a draft, the state chain and the compile slices, the charge: rows 1 and 3 to 6 of that table,
  sized there at 250 to 350 lines for the step circuit and 200 to 400 over ten files for the slices.
- **proofs** [derived]. (a) The statement for a replay block that reads committed values: a request may read in
  several replay blocks and several requests may read one, each block charged at most what its readers serve. One
  statement covers a cache hit and a long request's spans; CACHE has the sketch and the price, 1.5 times at 86
  readers. Our load's readers of a page are one conversation [relayed, row 12]. (b) The accept check's uniqueness
  lemma, and that a failed count in an unsampled replay block is a wrong block. (c) `proofs/spot-advice-certified`
  stays titled at draft off until (b) and VLLM-392 land.
- **The cost model** [derived]. A kept drafts' term in bits an output token while charged; the hit's reproving goes
  and the page commitment and the readers' term come in, as entry 2 has it.

### 7. Corrections to my earlier entries, and stale statements in maintained docs (listed, not fixed)

- **Addendum 3e, withdrawn** [read: `advice_code/replay.py:35-43`; run: `gh pr view 9143`, merged 11:47Z]: "the advice
  code refuses speculative decoding whole" and "until it models those lines the scheduled count is charged too". On
  `main` the scheduler replay models one MTP module's draft, so the scheduled count is derived at 0 bits and only the
  kept count is an input.
- **Addendum 3e, withdrawn** [read: `codec.py:15-16`]: "the buckets bound the grouping" (section 4).
- **Entry 2, superseded** [relayed, section 0]: "at most 2 bits a step at three draft tokens" (K is 5; section 4 has
  the figures); "no hit past 384 tokens has been replayed this way" (now 120,704 at W); "#8897 (open, not marked)"
  (on `main`).
- **Entry 2, a path moved** [run: `git ls-files`]: `engine/engine_classes.py` is `engine/profiles/engine_classes.py`.
- **`benchmarks/prod_repro/README.md:60-62`, stale on `main`** [read]. "the tree has no split of the bits by class":
  `advice_code/by_class.py` is on `main` (#9076) [run: `git log`]. "Speculative decoding is off in every arm the
  benchmark has, and the replay builds its engine without it": the `published` arm serves five draft tokens
  (`benchmarks/serve_tax/README.md:48-49`) and the replay builds its engine with the header's speculative decoding
  (`benchmarks/serve_tax/replay.py:9-10`, `:81-83`).
- **`advice_code/by_class.py` has no class for the kept drafts** [run: `rg -i 'draft|specul'`, no match]: not wrong
  while a draft serve is refused; VLLM-392's part 2 adds it.

### 8. Message for Daniel

**What Friday serves.** vLLM's published line for our cards: FP8 KV cache, five draft tokens, prefix cache on, context
262,144. At those flags a whole pass replayed token for token, 23,363 of 23,363: greedy requests, cache hits up to
102,336 tokens. Sampled requests run today.

**Cheapest replay.** Re-run only the sampled request's own computed tokens, in its served steps, every other request
replaced by zeros of its shape; read the cached prefix and its earlier positions from KV pages serving committed.
Its cost does not grow with repeated tokens. At Wednesday's settings a 120,704-token cache hit, read from
serving's pages, replayed bit for bit; at the published line's flags this replay has not run.

**At serving.** A hash of each KV page and a host copy until the coins: modeled under 0.2% of serving time, not
measured.

**Advice.** Under the certified tick, rehearsed on one card, cache hits, admission and resumes cost 0 bits; length
and ending cost 5.6 bits a request. Friday's certified zero is at draft off, because the advice has no field for the
drafts each decode row kept; what closes it is that field, then the accept rule computed in each request's replay
block, so the count is derived, not charged. Until then it costs 3 bits a decode row: about 0.66 bits an output
token, 290 for a 444-token answer, at greedy requests' measured acceptance. It grows with output tokens, not
requests.

**Missing.**

1. The guarantee for a replay block that reads committed values: one statement covers cache hits and long contexts,
   at about 1.5 times today's bound.
2. Circuit: the draft layer, its accept rule, the FP8 cache's row.
3. Replay: nothing forces a stand-in's kept count, so the cheapest replay cannot yet run with drafts.

**The message's figures, by tag** (292 words). [relayed]: the flags, 23,363 of 23,363, 102,336, 120,704, the 0 bits
and 5.6 bits a request (11,186 over 2,000), draft off. [derived]: under 0.2% (entry 2: the page hash modeled at
0.036% to 0.045% of a card), 3 bits a row, 0.66 bits an output token and 290 bits (section 4), 1.5 times (CACHE's
model, entry 2).

### 9. Not established

Each with what would establish it.

1. **A sampled request under drafts on GLM-5.3** [relayed: VLLM-377, 12:26Z]: neither its replay nor its acceptance
   is shown. The first serve and replay is today's 14:50Z row; it had not started when I wrote this.
2. **Row 16's replay at the published line's flags** [run: `rg`, no code]: one request among stand-ins with an FP8
   cache and drafts has not run, and nothing forces a stand-in's kept count. Established by `rebuild.py` with that
   forcing, on one request of the try's pass.
3. **Row 16 at Wednesday's settings, two parts** [relayed, entry 2]: a start inside a pass at TP8, and a span cut
   inside a request. Neither has code.
4. **The page commitment beside serving** [derived, entry 2]: not built; its cost is modeled, 0.036% to 0.045% of a
   card. At the published line's flags its leaf is the stored FP8 row, which no code hashes.
5. **The cost of the record at the published line's flags** [relayed: VLLM-377, 09:20Z]: three readings from one
   server lifetime an arm, +0.09%, +0.25% and +3.3%. Established by the rehearsal's pairs.
6. **The kept drafts' charge under certification** [derived]: no certified figure exists, since a draft serve is
   refused. Every figure in section 4 is from one greedy pass's counts, scaled to two shapes no run has served with
   a draft. Acceptance under production sampling is not measured, so the rows a request has are not either.
7. **A registered code's length for the kept count** [derived]: `draft_table` prints an entropy fitted to the trace
   it charges. Whether a code registered before the run stays near 1.36 to 1.84 bits a row on another trace is not
   shown.
8. **That the count is unique given its checks** [derived]: argued from reading the pin's accept test; no Lean and
   no run. The longest-common-prefix form is read at P for the greedy and the sampled rule and not for the synthetic
   rule; block verification's test I did not read.
9. **That advice a block's output is held equal to may be charged 0** [derived]: the rules have no such class. It is
   a ruling and a statement, proofs' and Daniel's.
10. **Whether the step record's positions are the ones the GPU ran, under async scheduling with drafts** [derived,
    addendum 3b]: the scheduler's count is an upper bound the GPU corrects, and the replay's comparison does not read
    the emitted counts. Established by comparing them, one line in `replay.py`.
11. **The guarantee for a replay block that reads committed values** [relayed: CACHE's final]: a sketch and a
    modeled price; no statement on `main`.
12. **The FP8 reading's edges** [read, section 5 item 10]: the backend node A picked, which I derive from the pin's
    list and did not read in a server log; the draft layer's later passes; the indexer's key cache. And which rows
    take which read under a registration at F: our registered class's sentence is about the bf16 backend.
13. **A request past 131,072 tokens** [relayed: VLLM-377, 09:20Z and 12:26Z]: the try's load cut requests there. One
    request ending at 262,144 is in today's rehearsal.
14. **The kept drafts' charge pooled over a week, a month and a year for a datacenter and for all compute**
    [derived]: I give a rate an output token and one node's totals at a rate that is not the published line's. The
    census's output tokens are not read.

## Revision to entry 3, Thu 12 Nov 13:31Z, on the coordinator's 13:18Z: VLLM-392 is named in the kept drafts' row and in the message; what it will charge is the count's own code, 0.30 bits an output token on the try, under the fixed width's 0.66

**Why a revision** [read: my mailbox's 13:18Z section, read at 13:29Z]. My final was pushed at 13:28Z without it. Two
things change, the kept drafts' row (section 3) and the message for Daniel (section 8); both are restated whole here
and replace the ones above. Sections 4 to 7 and 9 stand, with the three notes below. Tags as in the final.

**1. The kept drafts' row, final.**

| Class | Bits a request | What holds it at 0 |
|---|---|---|
| **Kept drafts, K = 5 (how many of its drafts a decode row kept)** | **Not in advice v1 today: a draft serve is refused, so no certified figure exists and Friday's certified zero covers a serve with the draft off** [read: `advice_code/codec.py:358-365`, `advice_code/replay.py:99-105`]. **VLLM-392 (vllm, owner vllm-advicedraft, filed 12:59Z, due Mon 16 Nov) puts each row's kept count in the record and charges it, a class of its own in the certified table, at the length of the count's own code (`kept_encode`, #9279)** [relayed: your 13:18Z; VLLM-392]. That code on the try's greedy pass: 6,997 bits, 1.36 bits a decode row that carried drafts, 218.7 bits a request, 0.30 bits an output token [relayed: #9279's text; the quotients are mine]. At that rate: 133 bits a request on AX's median, 85.5 on our 128k plan [derived]. At fixed width, your figure: 3 bits a row, 293 bits a request on AX's median, 188.5 on the 128k plan, 0.66 bits an output token [derived]. What `draft_table` prints on `main` today, uncertified: 295.7 bits a request and 0.405 bits an output token on the same pass, an entropy and no string's length [relayed: VLLM-384] | The accept rule computed in each request's replay block with the count its output, and a lemma that the count is unique: circuits' Definitions, vllm's step compiler, proofs' Lean, and a ruling that advice a block's output is held equal to is charged 0 (section 4) [derived]. It comes after VLLM-392 and is not part of it: "What holds the charge at 0 afterwards is proofs' to say" [relayed: VLLM-392] |

Route by route nothing changes [derived]: section 4's table stands. No route can be certified with a draft until
VLLM-392 lands, and rows 5, 6, 16 and 17 need a stand-in's kept count forced besides.

**2. Three notes on sections 4 and 9.**

- **The code's length is a charge the rules can take; the entropy is not** [relayed: #9279's text; I did not read its
  diff]. `kept_encode` is an adaptive binary coder, and `kept_decode` gives the counts back "from the bits, K and the
  output lengths alone". So its decoder holds no table fitted to the trace it charges, and the string's length is an
  advice length. That answers the first half of "Not established" item 7. The second half stands: its length on a
  trace that is not greedy is not shown, and the other greedy pass gives another rate, 0.87 bits a row and 0.20 bits
  an output token on `cb56` (666 bits over 765 rows and 3,406 tokens) [derived from #9279's text].
- **The fixed width is the cap, the code is the charge** [derived]. Advice is charged at its length and its compiler
  has a public length cap. 3 bits a row is a cap known before a run; the code's worst case I did not read. Cut by
  certifier window the code "costs up to 2 bits more a window" [relayed: #9279's text].
- **#9279 is not on `main`** [run: `gh pr view 9279` at 13:30Z: open, head `e29b73f1ba37`; `main` at
  `3498d4ad6546`]. Your 13:18Z says it is marked today [relayed].

**3. Message for Daniel, final** (299 words; replaces section 8's).

**What Friday serves.** vLLM's published line for our cards: FP8 KV cache, five draft tokens, prefix cache on, context
262,144. At those flags a whole pass replayed token for token, 23,363 of 23,363: greedy requests, cache hits up to
102,336 tokens. Sampled requests run today.

**Cheapest replay.** Re-run only the sampled request's own computed tokens, in its served steps, every other request
replaced by zeros of its shape; read the cached prefix and its earlier positions from KV pages serving committed.
Its cost does not grow with repeated tokens. At Wednesday's settings a 120,704-token cache hit, read from
serving's pages, replayed bit for bit; at the published line's flags this replay has not run.

**At serving.** A hash of each KV page and a host copy until the coins: modeled under 0.2% of serving, not measured.

**Advice.** Under the certified tick, rehearsed on one card, cache hits, admission and resumes cost 0 bits; length
and ending cost 5.6 bits a request. Friday's certified zero is at draft off, because the advice has no field for the
drafts each decode row kept; VLLM-392 (vllm, due Monday) adds the field and charges it, and the accept rule
computed in each request's replay block makes the count derived, not charged. Charged, it is at most
3 bits a decode row, 0.66 bits an output token; vllm's code measured 0.30 on one greedy pass: 130 to 290 bits for a
444-token answer. It grows with output tokens, not requests.

**Missing.**

1. The guarantee for a replay block that reads committed values: one statement covers cache hits and long contexts,
   at about 1.5 times today's bound.
2. Circuit: the draft layer, its accept rule, the FP8 cache's row.
3. Replay: nothing forces a stand-in's kept count, so this replay cannot yet run with drafts.

**The message's figures, by tag.** [relayed]: the flags, 23,363 of 23,363, 102,336, 120,704, the 0 bits and 5.6 bits a
request (11,186 over 2,000), draft off, VLLM-392 and its date, 0.30 (6,997 bits over 23,363 output tokens). [derived]:
under 0.2% (entry 2: the page hash modeled at 0.036% to 0.045% of a card), 3 bits a row and 0.66 bits an output token,
130 to 290 bits (133 and 293, section 4), 1.5 times (CACHE's model, entry 2).

## Second revision to entry 3, Thu 12 Nov 13:48Z: the FP8 reading corrected on one point and counted on the try's own record: on these cards only a whole prompt of at most 2,048 tokens reads its own keys unquantized; 4 rows in 8,937 steps did, and none of the 4 is in the pass the try replayed

**Why** [read; run]. I held section 5 against the try's record in the store: its server log and its step trace, 370 KB
fetched with `research data fetch --path`; nothing ran on a GPU. One statement of section 5 does not apply to an H200
and is withdrawn, the backend is now read and not derived, and the reading is now a count. The run is
`r20261112-074230-88d0`; its files are `out/prod-record/server/server.log` and
`out/prod-record/server/step_trace.369984.jsonl.gz`, the recorded arm's whole server lifetime.

1. **Withdrawn: section 5 item 7's table for masked MHA** [read, at P]. Masked MHA is for another device class and
   never with a quantized cache (`sparse_mla_attention.py:89-90`, `:112`; set at `:553`). On an H200 the MHA path is
   dense MHA alone: a step whose prefill rows' sequences are all at most `index_topk`, 2,048 tokens
   (`sparse_mla_attention.py:290-292`, `:307-310`).
2. **So, at the published line's flags on these cards** [derived]. The two values exist only for a prefill row that
   ends within its sequence's first 2,048 positions, in a step with no longer prefill row. In every other step every
   row reads every key from the cache as stored. The FP8 read enters a request at its first row, unless that row is
   such a prefill row with no hit; then at its next row.
3. **The backend, read** [read: the server log, lines 62 to 64]. "Using FLASHMLA_SPARSE attention backend out of
   potential backends: ['FLASHMLA_SPARSE', 'FLASHINFER_MLA_SPARSE_SM90']"; "Using fp8_ds_mla KV cache format for
   FLASHMLA_SPARSE backend"; "Using FLASH_ATTN MLA prefill backend". Section 5 item 2 is confirmed, and the first
   part of "Not established" item 12 is closed.
4. **Counted on the step trace** [run: a count over the trace's 8,937 steps and 136 requests; its rows are in the
   runner's order].
   - Every decode row carried 6 tokens: 21,356 rows, none cut short. So section 5 item 3's cut decode row, which the
     attention would treat as a prefill row, did not occur in this serve; that part of the reading is from the code
     alone.
   - 302 prefill chunks in 280 steps: 226 longer than 2,048 tokens, 4 of at most 256, 267 with a context.
   - **4 of the 302 took dense MHA**: whole prompts of 448 or 640 tokens with no hit, in steps 0, 4,501, 4,503 and
     6,801. 2,176 positions were read unquantized, once each, by their own row. Every other row of the lifetime read
     stored bytes.
   - **None of the 4 is in the replayed pass.** The trace's first pass after the warm-up is steps 205 to 2,244: 2,040
     steps, 32 requests, 23,363 tokens, the figures vllm gave for the replay. The four rows are in the warm-up and in
     the third and fourth passes. **So the try's replay did not exercise the two values.** By the reading it repeats
     them, since it repeats the schedule; no run shows it. A replay of the third or fourth pass, or one that covers
     the warm-up's first request, would.
   - The header: `kv_cache_dtype` fp8, `long_prefill_token_threshold` 0, steps of 8,192 tokens,
     `rejection_sample_method` standard, `draft_sample_method` greedy, adaptive verification off. The standard rule is
     the one I read; the block rule and the synthetic rule are not what this serve ran, so the last clause of "Not
     established" item 8 does not bear on it.
5. **What it changes** [derived]. Nothing in the route or the kept drafts' row. For vllm: the fourth cause is real,
   rare on this load (a short prompt that arrives alone and hits nothing) and untested by the try's replay. For
   circuits: one more read to model, a whole prompt of at most 2,048 tokens that reads its own rows as computed. vLLM
   has a setting that sends every row through the cache's read (`sparse_mla_force_mqa`, read at
   `mla_attention.py:1134-1136` and `sparse_mla_attention.py:307-310`); it is not among the published line's flags,
   and I do not propose it. For row 16: its three conditions in section 5 item 9 stand.
6. **For you to post, if you see fit** [derived]: "vllm: at the pin, on these cards with the FP8 cache, a row reads
   its own step's keys unquantized only when it is a prefill row within its sequence's first 2,048 positions and its
   step has no longer prefill row (dense MHA; masked MHA is off on an H200). In the try's recorded lifetime that is 4
   rows, steps 0, 4,501, 4,503 and 6,801, none in the replayed pass (steps 205 to 2,244). A replay over one of them
   would test your fourth cause."

The kept drafts' row and the message for Daniel of my 13:31Z revision stand unchanged.

## Last update to entry 3, Thu 12 Nov 15:38Z: the 14:50Z sampled row is not reported; the kept counts read from the try's own trace (your 3 bits a row is 0.66 to 0.68 bits an output token over its four passes, 0.87 at a chat load's acceptance); the message for Daniel, final

**1. The sampled row** [relayed: VLLM-377, read through the Linear tool at 14:57Z, 15:26Z and 15:38Z; VLLM-392 has no
comment]. Nothing is posted on the 14:50Z sampled serve and replay under drafts. So I cannot say whether a sampled
replay differs, and I point at no cause. If it does differ, the candidates I would read first, in order [guess]: the
accept test's uniform, drawn from the request's seed and the row's position (P
`rejection_sampler_utils.py:569`), so a request with no seed or a position the replay gives differently; then the
four rows of my 13:48Z revision, which the try's replay never exercised.

The one comment since is vllm-newpin's of 15:00Z on moving the pin [relayed]. Three of its facts touch this entry.
Its recommendation is that Friday stays at the pin, the build I read. At vLLM's release the FP8 cache's write follows
another rule (its scale rounded up to a power of two) and the sampled select draws by another generator, so section 5's
reading and the accept test are the pin's and want a second read there. And at the pin a hit with drafts on finds 64
tokens of a 130-token prompt where the release finds 128, which agrees with row 11's cell that a hit gives up its last
block.

**2. The kept counts, read from the try's own step trace** [run: a count over
`out/prod-record/server/step_trace.369984.jsonl.gz` of `r20261112-074230-88d0`; a row's count is its emitted count
less one].

| Pass (the same plan of 32 requests, 23,363 output tokens each) | Decode rows that carried drafts and emitted | Kept 0, 1, 2, 3, 4, 5 | Entropy, bits a row | 3 bits a row: bits a pass; a request; an output token |
|---|---|---|---|---|
| 1, the replayed one | 5,143 | 665, 553, 442, 257, 150, 3,076 | 1.840 | 15,429; 482.2; 0.660 |
| 2 | 5,286 | 751, 652, 450, 259, 154, 3,020 | 1.898 | 15,858; 495.6; 0.679 |
| 3 | 5,280 | 777, 606, 466, 245, 152, 3,034 | 1.887 | 15,840; 495.0; 0.678 |
| 4 | 5,170 | 704, 558, 434, 236, 163, 3,075 | 1.845 | 15,510; 484.7; 0.664 |

- **Pass 1 is as vllm gave it** [run], so section 4's measured figures are now read by me and not only relayed.
- **The same plan kept other counts in other passes** [run]: the four passes have the same prompt lengths, output
  lengths and hits (25 a pass, the longest 102,336 tokens), and their rows differ by up to 2.8%. Whether the passes'
  prompts and token ids are the same text I did not read; a pass's first requests miss the cache, so no pass reuses
  the one before. If they are the same text, a request's kept counts are not fixed by its prompt and its length.
- **Every decode row carried five drafts** [run: 21,356 rows of 6 tokens in the lifetime], so 3 bits is the width of
  every row here and no row is free.
- **On the two shapes at pass 2's rate** [derived]: 301 bits a request on AX's median and 193.8 on the 128k plan, where
  pass 1's rate gave 293 and 188.5.
- **At a chat load's acceptance** [derived from a relayed figure: vllm-newpin, 15:00Z, stock vLLM at the pin, 1,024
  chat prompts, "draft tokens accepted 49.3%"; I take it as kept over carried and did not read how it was sampled]:
  3.47 output tokens a row, so 0.87 bits an output token at 3 bits a row, 384 bits a request on AX's median and 246.8
  on the 128k plan. The rate an output token moves with the load's acceptance: 0.5 if every draft is kept, 3 if none.

**3. Message for Daniel, final** (298 words; replaces the earlier ones).

**What Friday serves.** vLLM's published line: FP8 KV cache, five draft tokens, prefix cache on, context
262,144. At those flags a whole pass replayed token for token, 23,363 of 23,363: greedy requests, cache hits up to
102,336 tokens. Sampled requests ran today; no result yet.

**Cheapest replay.** Re-run only the sampled request's own computed tokens, in its served steps, every other request
replaced by zeros of its shape; read the cached prefix and its earlier positions from KV pages serving committed.
Its cost does not grow with repeated tokens. At Wednesday's settings a 120,704-token cache hit, read from
serving's pages, replayed bit for bit; at the published line's flags it has not run.

**At serving.** A hash of each KV page and a host copy until the coins: modeled under 0.2% of serving, not measured.

**Advice.** Under the certified tick, rehearsed on one card, cache hits, admission and resumes cost 0 bits; length
and ending cost 5.6 bits a request. Friday's certified zero is at draft off, because the advice has no field for the
drafts each decode row kept; VLLM-392 (vllm, due Monday) adds the field and charges it, and the accept rule
computed in each request's replay block makes the count derived, not charged. Charged, it is at most 3 bits a decode
row: 0.66 bits an output token at the try's acceptance, 0.87 at a chat load's; vllm's code measured 0.30 on one
greedy pass. It grows with output tokens, not requests.

**Missing.**

1. The guarantee for a replay block that reads committed values: one statement covers cache hits and long contexts,
   at about 1.5 times today's bound.
2. Circuit: the draft layer, its accept rule, the FP8 cache's row.
3. Replay: nothing forces a stand-in's kept count, so it cannot yet run with drafts.

**The message's figures, by tag.** [relayed]: the flags, 23,363 of 23,363, 102,336, 120,704, the 0 bits and 5.6 bits a
request (11,186 over 2,000), draft off, VLLM-392 and its date, 0.30 (6,997 bits over 23,363 output tokens), 49.3%.
[run]: the try's rows a pass. [derived]: under 0.2% (entry 2: the page hash modeled at 0.036% to 0.045% of a card),
3 bits a row, 0.66 and 0.87 bits an output token, 1.5 times (CACHE's model, entry 2).

**4. Not established, final** (section 9 stands but for these).

- **Item 1, a sampled request under drafts on GLM-5.3**: still not reported at 15:38Z [relayed].
- **Item 7, a registered code's length**: a code exists in #9279, open; its length is shown on two greedy passes only
  [relayed].
- **Item 8, the count's uniqueness**: the serve ran the standard rule with greedy drafts [run: the trace's header], so
  the rule I read is the one served; still no Lean and no run.
- **Item 12, the FP8 reading**: the backend is read from the try's log; the two values arose in 4 rows of the
  lifetime and in none of the replayed pass, so **no run shows that they replay** [run].
- **New: the reading at vLLM's release** [relayed: vllm-newpin]: the cache's write and the sampled select differ
  there; I read the pin only.
- **New: acceptance on the load Friday serves** [derived]: 70.7% on the try's greedy agentic pass and 49.3% on stock
  chat prompts give 0.66 and 0.87 bits an output token at fixed width. Friday's own figure is neither.

## Entry 4, Thu 12 Nov 19:29Z: the cap and the charge circuit for the kept field (VLLM-392): neither cap is reached in any size we serve, and reached each only refuses; the addend is about 4,700 gates for each "c > j" decision at K = 5, in a block proved on every window; no guarantee's statement and no named assumption changes

**My design note of today answers neither of vllm's two choices differently from your 18:58Z** [read:
`lanes/proofs/20261112T0217Z-report-cheapest-replay.md`]. It names no window for the kept counts, and since its 13:31Z
revision its kept row charges "the length of the count's own code"; the 3 bits a row in its body is your fixed-width
figure beside it. One thing in this read, which the note does not have, bears on choice 1: section 2.5.

Read at `origin/main` `0582899c4392` (fetched 19:01Z) [run]. `main` was `4a90174415b7` at 19:20Z; of the files cited
below only the certifier's `PROTOCOL.md` changed, one line added at 296, so its line numbers here are one lower than
on that commit [run: `git diff --stat`]. AGENTS.md reread whole on `main`, the same at both commits [run]. VLLM-392's
three comments read whole; none after your 18:58Z [read, 19:17Z]. Nothing built; nothing run but `rg`, `git` and
arithmetic. Tags: [read], [run], [derived], [relayed], [guess]. "Gates" are ANDs.

**The answers**

1. **The cap.** ℓ bounds one execution phase's advice string, and CW one certifier window's charge, 60 s by default.
   At the highest rate a card on `main` gives, a window holds 134,146 output tokens and a 600 s phase 1.34 million,
   against vllm's 3.5 million and 28 million: 26 and 21 times under [derived]. Reached, each is a refusal of the run,
   with no truncation and no uncharged tail, and neither is a soundness matter. ℓ is reached only by one string that
   covers more than a phase: 3.5 to 8 hours of serving in one record [derived].
2. **The charge circuit.** The addend is a second counted segment: for each "c > j" decision one
   `ArithCountStep_v1{18, 1, 20}` and one `BookKT_v1{K(K + 2)}`, 4,736 gates at K = 5, times the class's cap on a
   window's decisions. That is 2.8e8 gates for a 60 s window at the chat rate, in a block proved on every window
   [derived]; the one compile-slice block sized on `main` is 1.0e6 gates, at a class far under a chat window's
   [read]. `ChargeCount_v2` becomes a `_v3` with two more statics, and no new primitive Definition is needed. No
   guarantee's statement and no named assumption changes [run, read], so I did not stop there. About 11 files and
   600 to 800 lines for the charge [guess], not counting the replay step and the slice, which are
   network-accounting's.

### 1. The cap

**1.1 What each bounds, and where it is fixed.**

| | ℓ (`cap_bits`) | CW |
|---|---|---|
| Bounds | The length in bits of one advice string: one execution phase's advice a_j [read: ARCHITECTURE.md:21, :24, :127; `security/Definitions/Core/Compiler.lean:213-217`; `RegisteredCircuit.lean:41`, `a_le : ∀ x ∈ a, x.length ≤ b.B.ℓa`]. In the adapter it is the whole `v1.encode` of one certified record, every window's charge together and the padding [read: `codec.py:8`, `:553-554`; `steps.py:343`, "the session's certified record"] | One certifier window's charge: the counted bits of its segment, the finish's 2 and its length terms' tail, below 2^20 [read: `arith_encode.py:156-157`, `:307`, `:316`; `window_charge.py:27`, `:60`, `:277-283`, `:511-515`] |
| Kind | A registration parameter: the capped compiler's digest names it, "so the registration's `compiler` fixes ℓ before sampling" [read: `pipeline/advice_windows.py:268-284`]; set by `--cap` [read: `pipeline/advice.py:56`, `:627`] | A constant of the code: the width of the count and of the declared charge c_w. It is a static of `ArithCountStep_v1{K, BIN, CW}` and `ArithCountFinish_v1{CW}`, and every caller binds 20 [read: `arith_encode.py:747`, `:917-923`] |
| Fixed at | The default `1 << 23`: `adapters/vllm/verity_vllm/pipeline/advice_code/codec.py:299` (`Registered.cap_bits`), `pipeline/advice_windows.py:265`, `program/predict/steps.py:146` [read] | `verity/protocols/certifier/verifier/arith_encode.py:157`, `CW = 20`; in Lean `security/Definitions/NetworkCertifier/AdviceV1.lean:68` and `verity/protocols/certifier/device/program/lean/NetworkCertifier/AdviceV1.lean:35` [read] |
| Enforced at | `codec.py:553-554` (`_parse`), `:701-702` (`decode_bits`) and `steps.py:320-322` (`_within_cap`, "before any of it is read", `:337-341`) [read]. No file under `verity/` or `security/` reads `cap_bits` [run: `rg`] | In the circuit: each counting step, the finish and the tail's sum set ok to 0 on a carry out of bit 20 [read: `arith_encode.py:680-681`, `:699-701`; `window_charge.py:513-514`] |

- **A window** is T buckets of τ_b: `Registered.T = 600` (`codec.py:297`), `WardenParams.window = 600` and
  `bucket_ms = 100` (`warden.py:206`, `:202`) [read], so 60 s [derived]. **A phase** is 600 s by the serving path's
  default (`boundary_windows.py:129`) and the reference deployment's (`block_record.py:297`, 12,500 machines) [read],
  so a phase's string is W = 10 windows [derived].
- **ℓ is not a period's total or a registration's.** A period of m phases may carry m·ℓ [derived from the Lean
  above]. And nothing in the adapter cuts a record at a phase: W is the record's last bucket // T + 1
  (`codec.py:275-277`) [read], so one string covers whatever one record covers.
- **The two do not imply each other**: ten windows at 2^20 − 1 bits are 10,485,750 bits, over ℓ = 8,388,608
  [derived]. Where the string is never formed (the composition charges Σ c_w), nothing on `main` compares that sum
  with ℓ [run: the `rg` above].

**1.2 The sizes.** The workload card (`benchmarks/spot_figures/cards/workload.glm-5.3.json`) holds no rate: it holds
the request, 256 prompt tokens and 32 generated [read]. The rates on `main` are the serving-rate card's and the
serve-tax card's step, so the table has three rows, each for one engine on eight H200s:

| Rate | Output tokens a second | A 60 s window | A 600 s phase | A week | A month (30.44 d) | A year |
|---|---|---|---|---|---|---|
| A. The serving-rate card's 279.47 computed tokens a GPU-second, as if every computed token were an output: a ceiling for that card [read: `serving_rate.glm-5.3.json`; the product derived] | 2,235.76 | 134,146 | 1,341,456 | 1.35e9 | 5.88e9 | 7.05e10 |
| B. The serve-tax card's chat step, 143 rows a step at 147 ms a step, no draft [read: `serve_tax.glm-5.3.json`; the quotient derived] | 972.79 | 58,367 | 583,673 | 5.88e8 | 2.56e9 | 3.07e10 |
| C. Rate A at the workload card's request, 32 outputs in 288 computed tokens [derived] | 248.42 | 14,905 | 149,051 | 1.50e8 | 6.53e8 | 7.83e9 |

- **vllm's two figures are right** [derived]: 2^23 / 0.30 = 27.96 million tokens and 2^20 / 0.30 = 3.495 million.
- **A window against CW**: 3.495 million is 26.1 times rate A's window and 59.9 times rate B's [derived].
- **A phase's string against ℓ**: 27.96 million is 20.8 times rate A's phase and 47.9 times rate B's [derived].
- **At ten times the greedy rate**, 3 bits a token, which is the plain form's worst: still 2.6 times under CW and 2.1
  times under ℓ at rate A [derived].
- **With a draft, at most six tokens a row**: rate B's rows at six tokens each would be 5,837 tokens a second, which
  needs every draft kept and a step no slower. Even there the greedy rate is 10 times under CW and 8 times under ℓ
  [derived].
- **One string for longer than a phase is where ℓ is reached**: at 0.30 bits a token after 12,507 s (3.47 hours) at
  rate A and 28,744 s (7.98 hours) at rate B. A week in one string is 21 times ℓ at rate B and 48 times at rate A
  [derived].
- **The pooled charge, which no cap bounds**: at rate B and 0.30 bits a token, 1.77e8 bits a week, 7.68e8 a month and
  9.20e9 a year a machine, and 2.21e12, 9.59e12 and 1.15e14 for the reference deployment's 12,500 machines. Scaled
  from two uncertified greedy serves of one machine; the sampled passes' 0.03 to 0.10 would give a tenth to a third
  of it [relayed rates, derived products]. The world's is not computed.

**1.3 What happens when one is reached, and the smallest change.**

- **ℓ**: decode raises `AdviceCodeError` ("… bits, over the cap ℓ = …") and the compiler `Unexpressible` ("the advice
  is … bits, over the registered cap ℓ = … bits"), before any bit is read [read]. `encode` under `check` decodes its
  own string (`codec.py:665-680`), so the server learns it at encode [read]. The phase's advice does not compile and
  its audit fails: "the audit refuses advice over the cap" (PROTOCOL.md:1216-1217) [read]. Nothing is truncated and
  nothing is left uncharged [derived].
- **CW**: the window's proof block has ok = 0, "a charge, the segment's and the tail's, of 2^20 bits or more"
  (`protocol/charge-chain.md:177`), and the Python reference and the Lean refuse the same [read]. The count never
  wraps: the carry out of bit 20 is the refusal [read].
- **The smallest change: none is needed.** If one were, for ℓ it is a larger registered `cap_bits` and no code
  [read]; for CW a shorter registered window T, or else the constant in three places, new pins for every Definition
  that binds it, and the Lean difftests' vectors, whose edge case is at 2^CW [read: the certifier program's
  `lean-audit.json`, `runs`].
- **What vllm has to hold to**: one string a phase. A serve of record that runs hours as one record needs either
  that cut or a larger registered ℓ [derived].

**1.4 Soundness, or only a refusal: only a refusal of an honest run** [derived].

- What the bound counts is the realized charge, each window's c_w, a public output the block holds its count to
  (PROTOCOL.md:1478-1482, :1503) [read]. A prover gains nothing at a cap: past it he is refused.
- ℓ's size is a term of one form of the bound and of nothing else. With a registered cap c the advice strings number
  at most 2^(c+1) − 1, so the capped form charges c + 1 bits whatever was sent (PROTOCOL.md:1117, :1150-1152);
  without one "the realized charge is sound only in the indexed form" (A7, :1218-1219) [read]. So raising ℓ costs the
  capped form bit for bit and the realized form nothing. At 2^23 the capped form is 13,981 bits a second a machine,
  against 292 realized at rate B [derived].
- Two conditions on the new code are soundness, and both are met by building it as the tail is built [derived from
  `charge-chain.md:179-205` and `window_code.py:27-47`]. (a) The second segment's count is added with the same
  refusal on the carry out: a sum that wrapped past 2^20 would undercharge. (b) What its decoder reads beside the
  string is public and fixed before it: K registered, the output lengths certified. Then the window's code stays
  prefix-free and the count of 2^c values (`hAs`) holds as it does with the tail.

### 2. The charge circuit

**2.1 What `ChargeCount_v2` counts** [read: `verity/protocols/certifier/prover/window_charge.py`, whole]. It runs the
counting encoder over one window in `window_code.code_window`'s order: n_arrived and each decision at
`BookKT_v1{3}`'s estimate for min(run, 2); n_hits and the hits; n_resumes, 0; n_order and the entries; n_lag, 0; one
`ArithCountFinish_v1{20}`; then it adds the length terms' tail, bitlen(max_tokens) + 1 bits a term (`:1-27`,
`:233-283`, `:463-519`). Its carry is 149 bits in a 160-bit row (`:97`, `:288`). The caller holds its `bits` to c_w:
`CompileSlice_v7`, with `ChargeLink_v2{D, 0, 0, K}`, where K is the arrival slots and not the draft's K
(`compile_slice_v7.py:1-33`). **Their worker's reading is right: nothing in it counts a field after the tail.**

**2.2 What a circuit must count for the coded second segment** [derived from VLLM-392's 17:10Z text and from
`kept_encode` on `main`, `benchmarks/network_e2e/encodings/replay.py:811-831`, which I read]:

1. A second count from a fresh tally, after the tail.
2. For each request whose body lands in the window, in release order, and for each of its emitting rows in order:
   the bits "c > j" for j = 0, 1, … up to the first 0 or to m = min(K, output_len − done − 1). Each bit goes through
   `BookKT_v1{C}` at the context (prev, j), then through `ArithCountStep_v1{18, 1, 20}` on its (f0, f1). There are
   C = K(K + 2) contexts: prev is "the request's first row" or 0 to K, and j is 0 to K − 1. Then done grows by 1 + c
   and prev becomes c; a request's counts end where done is its output length.
3. A second `ArithCountFinish_v1{20}`, 2 bits when the segment coded anything, and the sum of the three parts with
   the refusal at 2^20.
4. The carry. The kept contexts' counts go from window to window as the decisions' do, 30 bits a context: 1,050 bits
   at K = 5 beside today's 149 [derived]. If vllm's code restarted them at the book's priors in each window the carry
   would not change; what that costs in bits is not measured [guess].
5. Computed in the circuit and never witness: the context, the bound m, where a request's counts end, and which slots
   are enabled. "A context or a frequency the prover supplied would pick the code per value" (`arith_encode.py:39-43`);
   a slot arrangement or a disabled value of the prover's passes bits uncharged (`charge-chain.md:225-231`) [read].
   The output lengths are `cert`'s, as the length slots' are. It refuses c > m, counts that do not end at the
   request's output length, and a context past C.
6. Decisions number at most the output tokens: a row that kept c takes at most c + 1 decisions and emits c + 1
   tokens [derived]. On the try's pass 1 my own row counts give 20,255 decisions over 5,143 rows, 3.94 a row and 0.87
   a token, which agrees with their "about 3.9 a row and 20,000 a greedy pass" [derived from the counts I read at
   15:38Z; theirs relayed].

**2.3 Its size in gates.**

- `ArithCountStep_v1{18, 1, 20}` is 2,243 and `ArithCountFinish_v1{20}` is 41 [read: `arith_encode.py:123-125`].
- `BookKT_v1{C}` is 148 + (61 + bitlen(C − 1))·C for C ≥ 2 [derived from `_select` and `_build_book_kt`,
  `arith_encode.py:704-735`: a one-hot over the contexts, then two 15-bit counts selected and written back for each].
  The same count gives the two pinned sizes, 210 at C = 1 and 337 at C = 3 [read: `:125`]. At C = 35 it is 2,493.
- **A window: D·(2,243 + 148 + (61 + bitlen(C − 1))·C + ρ) + 41 + σ, with C = K(K + 2)** [derived]. D is the class's
  kept-decision slots a window. ρ is a slot's bookkeeping (done, m, prev, j, the request's end: a few 32-bit adds and
  compares), not built: 100 to 300 [guess]. σ is the three-part sum, about 22, and the link's equality over the
  wider carry, 1,050 more at K = 5.
- **At K = 5: 4,736 + ρ a decision.** At K = 3: 3,366. At K = 1: 2,580, which is today's decision [derived].
- **D is the largest window's, and every window pays it.** Slots are fixed per class and a window with more than its
  slots fails (`charge-chain.md:234-235`), and "every advised window is proved, never sampled", at its engine's
  class (`protocol/compile-slice.md:69-74`) [read]. So D is at most the output tokens of the requests whose bodies land in one window: at rate
  B's 58,367 that is 2.76e8 gates a window at one decision a token, and 2.40e8 at the try's 0.87 [derived].
- **Against what is there**: the one compile-slice block sized on `main` is 1,021,302 gates, at the class
  "(R, W, A) = (8, 1, 9), Q 4, D 32, P 54, K 4" (`compile-slice.md:74-75`, `:85`) [read]. The addend at a chat
  window's size is 235 to 270 times that block, but no class for a chat window is on `main`, so I cannot give the
  addend as a share of its own slice [derived]. Against a token's circuit it is nothing, 4,736 gates in 3.2e13 (the
  workload card's strata, the estimator's upper bound) [derived]; but a token's blocks are sampled and this block
  is not.
- **A slot a row instead of a slot a decision**, with j static, uses `BookKT_v1{K + 2}`, 596: 5 × 2,839 = 14,195 a
  row at K = 5. That is fewer gates on greedy (3,125 a token against 4,106) and three times more on sampled
  [derived].
- **Plain, 3 bits a row, is a count of rows**: under the length slot's 109 gates a row [read: `window_charge.py:68-77`;
  the comparison derived]. That is the circuit's side of your choice 2: about 4,700 gates a decision is what the 2.2
  to 93 times in bits [relayed] costs.

**2.4 Which Definition, and whether a statement changes.**

- **No new primitive Definition.** `BookKT_v1{C}` takes any C ≥ 1 with a context of bitlen(C − 1) bits
  (`arith_encode.py:490-491`, `:888-891`), and the counting step and the finish are used as they are [read].
  `BookKT_v1{35}` is a new specialization, with its pin.
- **`ChargeCount_v2{N, H, O, L}` and `ChargeLink_v2` become `_v3`**, with two more statics, the kept slots and the
  draft's K: the book gains the kept priors, the inputs the kept slots, and the carry the kept counts. With no kept
  slot it has to be v2 bit for bit, as v2 at H = O = L = 0 is v1's charge (`window_charge.py:70-71`) [read; the shape
  derived]. It needs its sidecar binding and pins (`window_charge.circuit-check.py`, `.json`), by AGENTS.md's
  Circuits rule.
- **Then the slice that calls it.** `CompileSlice_v7` is the one slice that charges the length terms
  (`charge-chain.md:215-217`) [read]; its `adv` gains the kept values. That is network-accounting's.
- **No guarantee's statement changes, and no named assumption** [run, read]:
  - The leak guarantees take the charge as a premise of the run. `EncardAccSeqsSampledLe` holds for any `As` and `c`
    with `hAs : As.encard ≤ 2^c` (`security/Definitions/NetworkCertifier/Guarantees/SampledCharge.lean:86-88`), and
    `SampledLeakAccepted` (`:107-120`) and the deployed `PeriodLeakAccepted` (`Guarantees/Period.lean:127-128`) count
    the free inputs outside the registered ones [read]. That advice v1's code is prefix-free is prose there
    (`:63-66`) and in the PROTOCOL; no Lean statement says it [run: `rg -i 'prefix-free|kraft'` over the area's
    definitions and proofs finds that docstring alone].
  - `security/lean-audit.json` names neither `WindowCharge` nor `AdviceV1` [run: 0 matches], and only
    `Proofs/NetworkCertifier/Code.lean` and the root `Definitions.lean` import the two modules [run], so no listed
    guarantee reads the charge's definitions.
  - In the two guarantees that count free inputs, the kept values count at their committed width, as every advice
    value does; the code's length enters only through `EncardAccSeqsSampledLe`'s `c` [derived]. That is so for the
    decisions and the lengths today, and the kept field does not change it.
  - What changes is definitions and one code-agreement theorem: `NetTiming.WindowCharge`
    (`security/Definitions/NetworkCertifier/WindowCharge.lean`: `Window`, `Book`, `Carry`, `window`), its mirror in
    the certifier program, `CodeChargesWhole` (`security/Proofs/NetworkCertifier/Code.lean:528-531`, of which the
    proofs' README says "They aren't guarantees", `:414`), and the window difftest's vectors with their digests in
    the program's lock [read].
  - One sentence is false for a draft serve until the circuit lands: `SampledCharge.lean:65`, "Its charge, the coded
    length counted by the in-circuit encoder". It is a docstring, and it is your point 3 of 18:58Z.

**2.5 What `window_charge.py` does not show, and it bears on choice 1** [derived; not sized].

- The count has to run in the string's order: the estimate's counts move with every coded bit, and the block's
  charge is the string's own length (`charge-chain.md:179-181`) [read]. The body's-window form orders the counts by
  request.
- The replay reads a row's kept count at the row's step, with the requests interleaved. Every v1 field today is "in
  the order the replay reads them, … with no routing" (PROTOCOL.md:1437-1438), and the scan reads each "at a private
  head" (`:1466`) [read]. The kept field by request would be the first that is not.
- So in the block the charge's read, by request, and the replay's, by step, have to be of the same committed values,
  or the charge is on free witness. That tie, a routing of a window's rows, is not in `window_charge.py`, and I
  expect it to be of the count's own size or more [guess]. Today's slice is spared half of it: it "loads each window
  from the empty state, so a window that opens with a request running is no slice" (`compile_slice_v7.py:25-26`)
  [read]. A slice for requests that outlive their window also has to carry the counts across windows.
- The step-order form with the request's context, vllm's other way and the same bits [relayed], counts where the
  replay reads. It needs `prev` in each running request's state and the coder's counts carried between step blocks,
  which `charge-chain.md:48-50` took out of the step blocks for the decisions [read]. I cannot say from this read
  which of the two is smaller.
- **So your "the step-order form buys nothing" holds for the bits and for the Python decoder; for the circuit it is
  open.** Who can close it is network-accounting, with the replay step that takes `kept` (NET-358 and NET-361
  [relayed: WATERLINE's note, as in my final]). What a later switch costs vllm is its `_parse` and `_decode` becoming
  one loop [relayed]; the charged bits stay, the string changes.

**2.6 Size in files and lines** [guess for every change; the present sizes are read].

| File | Lines now | Change |
|---|---|---|
| `verity/protocols/certifier/prover/window_charge.py` | 678 | +200 to 260: the reference's kept loop, `Window`, `WindowBook` and `WindowCarry`, the carry row, the second count in `_build_window`, the signature, `size` |
| `window_charge.circuit-check.py` and `.json` | 17 | the `_v3` binding and its pins |
| `verifier/tests/test_network_certifier_window_charge.py` and `…_slots.py` | 360, 48 | +120 to 150 |
| `security/Definitions/NetworkCertifier/WindowCharge.lean` | 218 | +60 to 80 |
| the program's `NetworkCertifier/WindowCharge.lean` and `WindowDifftest.lean` | 159, 171 | +80 to 90 |
| the program's `scripts/window_vectors.py` and `lean-audit.json` | 242 | +50, and new digests |
| `security/Proofs/NetworkCertifier/Code.lean` (`CodeChargesWhole`) | 556 | +60 to 100 |
| `protocol/charge-chain.md`, `PROTOCOL.md` ("Sent", "The encoder in the circuit"), two READMEs | 252, 2,371 | +60 |

In all 11 or 12 files and 600 to 800 lines. Not in it: the slice (`compile_slice_v7.py`, 199 lines, and its tests),
the replay step, and section 2.5's tie.

### 3. Not established

1. The addend's proving time, and so its share of serving. The proof-overhead card's blocks give no one price a gate
   that I could carry over [read: `proof_overhead.glm-5.3.json`].
2. Section 2.5: which order is smaller in the circuit, and the size of the tie.
3. An output-token rate with the draft on. No card has one; rate A is a ceiling by its construction and not a
   measurement of a draft serve.
4. vllm's 0.29 to 0.30 and 0.03 to 0.10 bits a token, on two uncertified serves: not rerun by me. I checked the two
   quotients and the decision count only.
5. ρ and σ of section 2.3 and every line count of section 2.6: nothing was built.
6. That one certifier window and one string are one engine's: I took it so [guess].

### 4. Statements in maintained docs, listed and not fixed

- `security/Definitions/NetworkCertifier/AdviceV1.lean:13`, `RecordRows.lean:11-12` and `Tap.lean:17` cite
  `verity/core/protocols/certifier/…`, a path `main` does not have; the files are under
  `verity/protocols/certifier/verifier/` [run: `ls`].
- `pipeline/advice_windows.py:263` and `program/predict/steps.py:145` say ℓ is "far above a session's few hundred
  requests (I1 spends tens of bits a row)" and "far above a session's few thousand rows". True today; they are the
  sentences to change when a class grows with tokens [read].
- PROTOCOL.md:1437-1438, "each field's values … are in the order the replay reads them, … with no routing", goes
  false when the kept field lands by request (section 2.5) [derived].

### 5. The text for VLLM-392

~~~
@vllm proofs (lead), our two answers.

**The cap holds; build as agreed.** ℓ = 2^23 bits bounds one execution phase's string, CW = 20 one 60 s window's charge. At the serving-rate card's ceiling (2,236 tokens a second a machine) a window holds 134,146 output tokens and a 600 s phase 1.34 million: 26 and 21 times under your two figures, which check. Past a cap the run is refused, never truncated: not a soundness matter. Keep one string to one phase: at 0.30 bits a token ℓ is reached in 3.5 to 8 hours.

**The charge circuit is not built**: about 4,700 gates for each "c > j" decision at K = 5, proved on every window. Until it lands the field is charged by code and not checked.

**Open, ours:** whether it counts by request or by step; we say before it touches your string's order.
~~~

Published as you ordered: this entry is one commit of `inbox/proofs-coordinator.md` alone on
`cursor/verity-proofs-84b5`, appended at the end, no force. It is also the last section of `inbox/replay.md` on
`cursor/proofs-replay-136d` and in my note. Nothing is posted to Linear or Slack.
