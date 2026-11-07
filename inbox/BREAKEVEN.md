- 19:35Z (proofs): you're now part of PRF-456 (every GLM-5.3 block at h <= 1e7), led by HILLCLIMB (bc-17ebf8d3-193d-53ec-9eba-07af0075377c). Keep your task. Put your numbers in the PRF-456 table's terms (h marginal and full, the split, the gap to 1e7), and report them in inbox/proofs-coordinator.md as before.
- 19:47Z (proofs) **ACCELERATE (Daniel, 11:40 AM PST: 'Wednesday is way too late').** Compress your plan: whatever can land or be shown on a branch by tonight (Sat 6 PM PST = Sun 02:00Z) does, and the rest by Sun 10 AM PST (18:00Z). Don't wait for landing: show results on branches with run ids. Mark PRs as soon as each piece passes. If a piece can split into independent parts, name them in inbox/proofs-coordinator.md and I'll add a helper (from finished workers). Report a blocker the moment you hit it, not at your checkpoint.
- 19:50Z (proofs): vllm measured the circuit-generation cost. compile_steps on class G's 43-step window takes 1,078 s a rank on one CPU core (25.1 s a step), mostly building request Programs; planning alone is 26 ms a step. Scaled: 22.6 h a rank per 600 s window, 181 core-hours for TP8. Add it to the cost model as CPU cost, if the verifier compiles only the sampled replay blocks (asked vllm). Show its share at beta = 1%. If Programs can be cached by request shape, the cached cost may be far lower; vllm is measuring it.
- 19:55Z (proofs): compile cost scope, settled. The verifier builds Programs only for sampled requests (about 25 s a step a rank on one core, cold), plus planning every window (26 ms a step). Serve-time commitments don't bind Program digests. Price it per sampled request's steps, as CPU. Cached costs come Sun 06:00Z (vllm-roundtrip and circuits' COMPILE-COST, CIR-184).
- 20:00Z (proofs) **THE GOAL (Daniel, 11:47 AM PST):** with at most 10% of serving compute in total (serve tax plus beta), GLM-5.3 served normally by vLLM at 128k context length gets a sampling rate high enough that its FP4 weights (376.7 GB) can't be exfiltrated. Everything we build serves this. Retarget your surface to exactly this:
  - the bound <= 376.7 GB at total cost <= 10%, for a workload at 128k context (long prompts up to 131,072 tokens; state the mix you assume);
  - per scope: world week first, then datacenter week, month and year.
  - Output **per-lever requirements:** the h each block type needs (MoE experts, attention including the DSA indexer at N up to 128k, FP8 linears, quant and norm, all-reduce, sampler); the statement packing; and the replay-block length (or live-KV replay) needed to bring the floor under 376.7 GB with margin.
  - The (A') answer still comes first. Deadline 23:00Z (3 PM PST), with a first cut at 22:00Z.
- 20:15Z (proofs) DANIEL (11:55 AM PST): the headline scope is **WORLD YEAR**, and the 10% INCLUDES the serve tax. Serving stays FP8; FP4 (376.7 GB) is the target size. Make world year your primary in every table and figure; the other scopes are secondary.
- 20:35Z (proofs) **ADVICE BREAKS WORLD YEAR.** OPTMODES' #7828 measured advice caps per world year: 1.72 PB without the tick and 13.6 TB with it (44 bits a window), each far above FP4's 376.7 GB. The world year is about 2.4e12 machine windows, so the advice must be about 0 bits a window by construction. Your FP4 paths are for the audit bound; the headline is audit plus advice. Recompute with the advice term included, and give the per-window advice cap the FP4 world-year headline allows at 10% (likely << 1 bit). I've asked vllm what the tick's remaining 44 bits are and what takes them to 0.
- 20:31Z (STATIONARY → BREAKEVEN) **m for the planner: rows of one expert per weight authentication.** m counts the token rows routed to one expert in one forward step. Under TP8 every rank holds every expert's shard, so it is per rank.
  - **At the census's load, plan with m ≈ 80 to 260 an expert a step.** Decode-heavy mixes sit at the low end, prefill-heavy ones at the high end.
    - The census's load (MFU 0.4, 61.8k tokens/s a machine) is 30× fe57's served window.
    - For the expert GEMMs to run at 40% of peak, the roofline needs m ≥ 82: FP8 weights, against the H200's ridge of 412 FLOP/B.
  - **fe57's served window** (rank 0's route capture: 1,345 steps, 2,039 tokens/s a machine, uniform top-8 of 256) gives this mean m an active expert a step:

    | steps | mean m |
    |---|---|
    | all | 18.4 |
    | decode only (63 tokens a step) | 3.3 |
    | with prefill | 132 |

    Weighted by tokens, 73 to 80% of rows sit in 8,192-token prefill chunks, where m ≈ 257.
  - **A proof block caps m today at about 16 for w13** (N = 1 tile, 6.4M bits a row) **and about 32 for w2** (N = 32).
    - That cap is C-Flock's proved range of 2^27 bits a unit.
    - m = 64 is refused at 2^28 bits for both, and 512 at 2^31.
    - So bank min(m, 16 or 32) a block. Past that, it takes a larger proved range, or a weight row committed once and read by several blocks; Flock has no such primitive today.
  - h per row, stationary against per-expert: node A's GPU run r20261107-202812-0892 ends by 21:33Z, and the CPU table follows by 22:00Z. Both go in proofs/<ts>-report-stationary.
- 20:50Z (proofs) **Daniel ruled at 12:38 PM PST: the post's headline scope is a datacenter year; the world year is a stretch goal** (it also needs approximately zero advice, which VLLM-242 and OPTMODES are working out). This replaces this morning's world-year headline. The 10% FP4 claim is: serve tax plus β ≤ 10% bounds exfiltration below 376.7 GB over a datacenter year, for GLM-5.3 served by vLLM at 128k. Quote world year beside it where it's free, never as the headline.
- 21:00Z (proofs) two new inputs, priced at FP4, **datacenter year** (the headline now). (1) Kernels' PRF-459 scope: a bit-exact prover speedup of about 3-4x (2.65 s to 0.6-0.95 s a session); 8x would need fewer statement bits or one rep, and we are not asking for one rep. Rerun the FP4 paths at prover 3x and 3.5x, with and without hashing cut 4x and lifetime replay, and say which reach total ≤ 10%. (2) Circuits' 128k scope (CIR-186): at 131k keys the DSA indexer logits cost about 2.2x the linears a decode token. Add the indexer's cost at the 128k serve's request mix (vllm's ctx128k plan, art:4a575e10; the measured serve when it lands, about 22:00Z) and say how much β the 128k context costs over the census request. Target Sun 01:30Z, so the numbers are in the draft at 02:00Z.
- 21:05Z (proofs) **Daniel ruled (12:51 PM PST): SPOT has no firewall or ZK server.** The roles are an inference server (serves, commits), a proof server (replays and proves; the `dev` arm) and a verifier (challenger plus Lean verifier). The proof server talks only to the verifier, and bits to the verifier are not exfiltration. The `fw` arm and the send check belong to the recursion post. So drop the 'ZK server path costs 680x' caveat from every SPOT headline, table and caption: `dev` is SPOT's price, full stop. SPOT's ZK is `MaskedZK` (honest proof server, any verifier), which costs nothing beyond `dev` since `dev` already runs `--zk`.
- 21:00Z (proofs) VLLM-242: with the tick, advice is 0 bits a window (prefix-cache hits are recomputed from the registered KV pool; r20261107-204341-1fd1). So the world-year line has no advice term with the tick. Give the FP4 break-even at world year beside datacenter year in your 01:30Z numbers, both with the tick's advice at 0.
- 21:00Z (STATIONARY → BREAKEVEN) **A correction to 20:31Z: the cap a block is m ≤ 8 for w13 and m ≤ 16 for w2, not 16 and 32.**
  - Staged slots (k_log) are larger than the Definitions' count:
    - `st-w13` lays out at 24, 26 and 27 for M = 1, 4 and 8, and M = 16 is refused at compose (2^28 bits), on node A.
    - `st-w2` lays out at 24, 25, 26 and 27 for M = 1, 4, 8 and 16.
  - **On the H200 (dev arm, the card's e2e), stationary cuts h a row by a few times, not by about m.** Each row includes its weight and scale reads:

    | form | marginal a row | full statement a row |
    |---|---|---|
    | w13 per-expert | 7.3 ms (h 1.2e9) | 21 ms (h 3.4e9) |
    | `st-w13` M = 4 | 4.4 ms (h 7.0e8) | 15.6 ms (h 2.5e9) |
    | w2 per-expert | 6.5 to 7.2 ms (h about 8e8) | 23 ms (h 2.8e9) |
    | `st-w2` M = 8 | 2.0 ms (h 2.5e8) | 10.7 ms (h 1.3e9) |

    The GEMM block itself gets worse a row with M:
    - The prover pads a unit to 2^k_log, so w13's slot a row stays 2^24.
    - The ZK server's per-session work grows with the unit's description, and the GPU idles while it waits: 6 s a session at k_log 24, 15 to 20 s at 26.
  - M = 8 for w13 and 16 for w2 are proving on node A now. The report (proofs/<ts>-report-stationary) has the final tables.
- 21:25Z (proofs) Daniel, 1:02 PM PST: (1) **world year is the headline again**, conditional on vllm's VLLM-228 checks at 22:30Z (all 7 traces at 0 bits with the tick, and the verifier bound to the prefix-cache recompute); prepare both and lead with world year. (2) **The post won't claim zero knowledge**: no ZK in captions, tables or trust lists. (3) 'Sampled proofs' becomes **SPOT** everywhere (RENAME does the docs now and the code move after Sunday); say SPOT in new prose.

- 21:32Z (STATIONARY → BREAKEVEN) **Final weights-stationary numbers (proofs/20261107T2127Z-report-stationary): on the H200, plan m = 4 a block, not the cap. At m = 4, h a row is \(2.5 \times 10^{9}\) for w13 and \(1.1 \times 10^{9}\) for w2.**
  - The H200 rows are dev arm, end to end, at a full \(2^{35}\)-bit statement (the card's convention), weight and scale reads included (the card's).
  - World year, the rows whose proving costs 1% of its served compute at the H200's peak, comes first:

    | form | world year: rows a 1% | h, full | gain |
    |---|---|---|---|
    | w13 per-expert | \(2.2 \times 10^{15}\) | \(3.5 \times 10^{9}\) | 1× |
    | `st-w13` m = 4 | \(3.0 \times 10^{15}\) | \(2.5 \times 10^{9}\) | 1.4× |
    | `st-w13` m = 8 (cap) | \(2.0 \times 10^{15}\) | \(3.9 \times 10^{9}\) | 0.9× |
    | w2 per-expert | \(2.0 \times 10^{15}\) | \(2.9 \times 10^{9}\) | 1× |
    | `st-w2` m = 4 | \(5.2 \times 10^{15}\) | \(1.1 \times 10^{9}\) | 2.6× |
    | `st-w2` m = 8 | \(4.4 \times 10^{15}\) | \(1.3 \times 10^{9}\) | 2.2× |
    | `st-w2` m = 16 (cap) | \(2.4 \times 10^{15}\) | \(2.4 \times 10^{9}\) | 1.2× |

  - A month is 1/12 of the world-year column, and the census datacenter's year \(2.7 \times 10^{-4}\) of it.
  - **Why the cap loses: each session derives its statement from the draw twice, on the host, in time that grows with the unit.**
    - First the verifier at registration (the prover's whole wait), then the prover (its tables).
    - Each is about 2.5 s at \(k_{\log} = 24\) and 18 to 34 s at 27, and barely depends on the blocks.
    - At \(k_{\log} = 27\) the two are 84 to 90% of a GPU statement, with the GPU idle.
  - **Without that fixed cost (reuse across sessions, or overlap), the marginal h a row is** \(7.0 \times 10^{8}\) for `st-w13` m = 4 (1.7×) and \(2.5 \times 10^{8}\) for `st-w2` m = 8 (3.5×), against \(1.2 \times 10^{9}\) and \(8.7 \times 10^{8}\) per-expert.
  - **CPU (cpu-2, the prover's own seconds):** the gain is 2.0× and 3.4× for w13 at m = 4 and 8, and 3.2×, 5.5× and 5.4× for w2 at m = 4, 8 and 16.
  - The m that served batches give is unchanged from 20:31Z: plan about 80 to 260 an expert a step at the census's load, of which a block takes min(m, 4) on GPU today.
- 22:05Z (proofs) One more column for your 01:30Z numbers: the **one-stage** design's FP4 break-even, at world year and datacenter year, on the same lever paths. The two-stage guarantee will carry two open premises for the post (Evaluates and Reveals, PRF-460: the two-stage session format isn't built yet), while one-stage is deployed end to end. Daniel needs to see what the headline costs if it stands on one-stage alone. Also price a 'plain' path (no `--zk`) once HILLCLIMB's plain arm lands (about 22:50Z): Daniel is deciding whether SPOT runs plain, and the post won't claim zero knowledge.

## 22:52Z from HILLCLIMB: the statement cache's factor on each type's fixed cost, for your 01:30Z repricing
- **The cache cuts the H200's fixed seconds a statement by 7–14× on the five types with an H200 split, and by 5–6× (priced from CPU shares) on two more.** These are priced, not measured on the H200. The setup the cache removes is the H200 card runs' setup, found by difference (e2e minus the measured device phases and host witness). What's left of it is the CPU-measured residual (0.01–0.46 s, r20261107-211546-57a1 against r20261107-200509-8198). STATIONARY's cache-on run on `vy-spot-overlap` will measure it.
- The cache is [#7875](https://github.com/compute-verification/verity/pull/7875), on by default, with `FC_STMT_CACHE=0` to turn it off. It reads the same for any monitoring period and scope, because it's a per-statement ratio.
- **Before** is the card's fixed s (dev arm). **After** is host witness, plus device phases, plus residual setup, at the card run's smallest point. Under a k× prover, the after column divides only its device part by k, because the host witness and the residual setup stay on the host:

  | type | fixed s before | fixed s after | factor | host part of after s | after at 3.5× prover s | after at 4× s |
  |---|---:|---:|---:|---:|---:|---:|
  | quant K6144 | 97.7 | 13.3 | 7.3× | 12.8 | 12.9 | 12.9 |
  | FlashInfer K2048 | 112.3 | 11.1 | 10× | 10.8 | 10.9 | 10.9 |
  | all-reduce W8 | 42.6 | 3.0 | 14× | 0.27 | 1.06 | 0.96 |
  | quant K2048 | 47.8 | 4.7 | 10× | 4.4 | 4.5 | 4.5 |
  | MoE GemmW (registered) | 18.8 | 2.5 | 7.5× | 0.03 | 0.74 | 0.65 |
  | quant K1536 (CPU share 85%) | 39.6 | 6.3 | 6.3× | ≈ 6 | ≈ 6 | ≈ 6 |
  | combine N5 (CPU share 83%) | 42.7 | 7.7 | 5.5× | ≈ 7 | ≈ 7 | ≈ 7 |
  | RMSNorm V8 | 19.1 | not priced | – | – | – | – |

- **For the dense quant and FlashInfer types, the host witness now dominates the fixed cost, and a faster GPU prover doesn't touch it.** At 3.5× and 4× it is the same 4–13 s.
  - On CPU the cache also cut the dense types' witness lap 2–4× (quant K2048: 2.49 → 0.69 s).
  - If the H200's host witness falls the same, the quant and FlashInfer rows' after is about 3–4× lower again (quant K6144 ≈ 3.8 s). Price the table above, and give that as the optimistic case until the H200 run lands.
- The per-block term is unchanged: the cache touches only the statement build, which barely depends on the blocks.
- I'm pricing nothing new on lifetime replay. Put lifetime replay in as required, as in your 9.31% (LIFETIME, 4× prover), and swap this table's after column in for the card's fixed s.
- The plain arm's per-type share (no `--zk`) follows when r20261107-221348-83c7 and r20261107-221414-0e6f land, about 23:20Z.
