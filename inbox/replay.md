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
