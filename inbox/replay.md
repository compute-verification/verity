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

## Entry 2, Thu 12 Nov 02:54Z, revised 03:04Z (due 06:00Z): the table, the recommended route, what is undecided, the context limit

I read vllm's 02:50Z on VLLM-377: the route is to be chosen there at 07:00Z. This entry is written to be used for
that; what could still move a cell before then is named at the end of this head.

Read since entry 1: your 02:28Z section (the context column is 1,048,576 beside 131,072; your point checked below
with file and line), CACHE's entry 2 (`cursor/proofs-cache-7c3e` at `8dfb1e1dbe`), vllm's
`20261112T0229Z-report-cachedserve-load`, the goals' texts, vllm-prodreplay's answer on VLLM-377 (02:48Z), the end of
tonight's serving-cost run `03bb`, and vLLM's rejection sampler, draft layer, request and step defaults at the pin
`d9105ea80`. The note `20261112T0217Z-report-cheapest-replay` holds the same text. Nothing was run and no code
changed.

Revised since `c348d79df3` (02:54Z):

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
6. Added: row 16 against CACHE's route; the grain of the page commitment; who is charged across execution phases;
   the draft layer; whose each 131,072 is; the design sentence on `main` that this route changes
   (`commit/history_tables.py:22-26`); the message for Daniel at 300 words or under.

Still to come tonight, each named where it lands below: `r20261112-025327-f570` (the measured cost of the step record
and the canonical top-k, by 06:15Z); CACHE's final (05:15Z); vllm-prodserve's settings and accept rule (05:30Z).

### The answer


1. **The cheapest replay the tree's parts make.** The replay block is the sampled request's own computed positions
   (for a long request, a span of them). The prover re-runs only the engine steps those positions ran in, with every
   other request a zero stand-in of its recorded shape. Everything the positions did not compute, the cached prefix
   and the request's own earlier positions, it reads from KV pages that serving committed as it wrote them. Row 16.
2. **Why it is cheapest.** It is the only route whose replay, interior commitment and proving do not grow with the
   share of repeated prompt tokens (12% to 98%) or with the context (131,072 to the model's 1,048,576): what it
   recomputes is what the request itself computed. Recomputing the hit, which is what `main` does, multiplies the
   tokens replayed, committed and proved by 1.13 at our plan's 11.62% repeated and by 7 to 59 at 85.67% to 98.31%
   [mine: 1/(1 − share) of prompt tokens; CACHE's 6.9 and 46 to 59 count FLOPs and outputs]. In this engine class the replay must also re-run every step that first computed a page of
   the prefix, the earlier requests' decode steps included: for an agentic conversation, the conversation so far.
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
| 5 | The target's values depend on the others' shapes, not their values. Shown: 4 of 4 targets, among them a 256-token hit loaded as 4 pages, a 19,170-token prompt decoding at 19,360 keys, and a request preempted 3 times; an all-real control of 307 steps; P40 [M `ef0f`, `c677`, `b2cd`]. Not shown: a hit past 384 tokens in a one-request replay; a 131,072-token target; more than one KV cache group, which `rebuild.py:870` refuses [relayed, vllm-prodreplay]. | The request's Program after its history: the step compiler writes it in this engine class with the hit's rows as boundary inputs, the producers' (`steps.py:23-24`, `:45-46`), and a request carried over a phase's edge reads its own earlier rows the same way (`:46-51`). Refused today: a step above 2,048 tokens (`:57-59`), positions past 131,072 (`:95-96`), a recompute after preemption (`:56-57`), sampling other than greedy or seeded top-p (`:98`). Stock steps are 8,192 tokens on an H200 [read vLLM `vllm/engine/arg_utils.py:2663-2666`]. On the two-stage path the replay block is still the stand-in (`adapters/vllm/README.md:98-100`), and the CPU replay refuses a Program that reads a cached prefix (`request_replay.py:39-41`). Compile: 135 s a Program a rank over a 256-step span, 3.1 to 3.8 core-hours a sampled request [m `sweep.py:52-53`]. | Not closed: it reads values another replay block computed. On `main` the patterns' form, which the figure of record uses, refuses it: each request must read in one replay block (`TwoStageCheck.lean:58`), and the pricer refuses a fan-out above 1 (`optimize.py:1200-1202`). The capped form (`ServedCover`, `:89-94`) covers it when no two requests read one proof block, so with the conversation taken as the request and the cache keyed by conversation; it prices at about 1,945 times the patterns' form [m, CACHE's entry 2]. The patterns' form with S read as F·S: 1.52 times at F = 86, and no statement on `main` [m, CACHE; a proof sketch]. |
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

**Row 16.** It is the only route that stays flat in replay, interior and proving as traffic goes from 12% to 98%
repeated and context from 131,072 to the model's limit, and it leaves serving stock but for a page hash and a host
copy. Each part of its replay is already bit for bit on GLM-5.3 at TP8 at small size, so what it lacks is one
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
2. **Whether a long hit read from committed pages replays bit for bit.** Shown at 256 tokens. Decided by
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
read from KV pages that serving committed as it wrote them. So the cost does not grow with the share of repeated
tokens or with context length.

**At serving.** A hash of each KV page as it is written and a host copy kept until the coins: modeled under 0.2% of
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
2. The request's circuit on this path, with the draft layer and its accept rule.
3. Measurements: no cache hit past 384 tokens replayed this way, nothing at an agentic load, nothing past 131,072
   context. The design does not limit context; this node's memory stops near 400,000 tokens.
