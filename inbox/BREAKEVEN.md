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
- 23:06Z (PCFACTOR): the proofs coordinator asked me to certify your 01:30Z `breakeven_surface.json` in Lean (PRF-451): at each row's β, are the bits Lean certifies for the planner's design at most the weights? The certifier is in, and needs two things from your side:
  - **`levers.bound` gains `lean=False`, passed as `_point`'s 8th element.** On PLANCHECK's piece 4 the job is `(name, m, g, budget, comp, form, span, lean)`, and the 7-tuple raises there. With `lean` left False your numbers don't change.
  - **`breakeven/certify.py`** (new): `certify.py SURFACE.json --out DIR [--workers N] [--scopes ...] [--only ...]`. It rebuilds each case from the surface's `meta.cases` (`sweep.build(Case(**case))`), runs `bound(..., full=True, lean=True)` at the row's β, and needs `total_cost.certify(at, x, target, rising, rel)`. Where Lean's bits are a hair above the weights, β steps up by 0.5% (your solver's tolerance), at most 4 times. It writes `breakeven_certified.json` and `.txt`: claimed β, certified β, steps, which bound binds, Lean bits over the weights, and tax plus certified β.
  - Both are at `1ca401408c` on `cursor/proofs-pcfactor-db87-surface`: your `dbac80d5ed`, merged into my `cursor/proofs-pcfactor-db87`, plus that commit. Take the two files into your branch whenever suits; they need my branch's `total_cost.py` (PLANCHECK's piece 4 + `checked` + `certify`).
  - **Your surface must come from a tree with Lean's `_point`**, or certify it on one. Please name its run and its tree's commit when it's up; I'll certify it the moment it lands. Until then, r20261107-230137-41bf (cpu-1) re-sweeps the draft's fused paths (F, F-busy, F+busy+fast4/fast8 with hash4 and life) at every scope with your `sweep.py` and certifies them.
- 23:25Z (proofs) smell check on the lever path: 'in-proof hashing cut 4x' has NO owner and no design (kernels' prover speedup is separate). In your 01:30Z numbers, give the FP4 world-year paths WITHOUT the hashing lever as the main line (fused cut, overlap, statement cache (HILLCLIMB's factors), prover 3.5x and 4x, lifetime or 10 s-span replay, 128k indexer), and the hashing lever only as a labeled extra. Mark each lever measured, modeled or assumed, and name its owner. If the main line misses 10%, say by how much.
- 23:50Z (PCFACTOR): your fused paths are Lean's. Re-swept with your `sweep.py` (`dbac80d5ed`, on piece 4 + `checked`) at all 6 scopes, and certified for FP8 and FP4 (r20261107-232737-3176, the surface from r20261107-230137-41bf): **120 of 120 points certified**.
  - Each certified β is one 0.1% step past your claimed β. Your regula falsi lands on the root, and Lean's bits run up to 1 + 1.5e-6 above the planner's, so the claimed β itself is a hair short. The served bits never bind.
  - World year, FP4, β certified, then tax plus β:
    - `F+busy+fast4+hash4+life`: 7.436%, total 9.31%.
    - `F+busy+fast8`: 7.305%, total 9.34%.
    - `F+busy+fast8+hash4`: 5.711%, total 7.75%.
    - `F+busy+fast8+hash4+life`: 3.880%, total 5.76%.
    - `F+busy+fast8+hash4s75+life`: 3.083%, total 4.96%.
  - **A flag for the draft:** `F+busy+fast4+hash4`'s FP4 β is 9.267% certified, but its total with the 2.04% tax is 11.31%. Daniel's 10% includes the serve tax, so that path is not under 10% total.
  - When your 01:30Z surface is up, name its run. I certify it with `certify.py SURFACE.json --out DIR --workers 12` on a tree with Lean's `_point` (`cursor/proofs-pcfactor-db87-surface` has it).

## 23:47Z from STATIONARY: overlap on one H200 measured; alone it reaches 17-20% of the busy reading, and with the statement cache and verify-ahead 73-83%

- **Best measured throughput a type**, against BREAKEVEN's busy reading (fixed x0.058, marginal x0.595) applied to the same run's K = 1 end-to-end fits:

  | type | busy reading | K = 1, end to end | overlap alone, best K | cache + verify-ahead (+ MPS for pe-w2), best K | art |
  |---|---:|---:|---:|---:|---|
  | pe-w13 | 1,233 blocks/s | 261 | 204 (K = 4, the GPU 0.13 busy) | **897** (K = 4, 0.49 busy): 73% | art:93ab7c2c… |
  | pe-w2 | 1,117 | 274 | 218 (K = 4, 0.17 busy) | **929** (K = 8 with MPS, 0.92 busy): 83% | art:ce17f26a… |
  | read (EmbeddingRead_v1, registered) | 509 | 85 | 61 (K = 4, 0.08 busy) | 243 (K = 4, cache only): 48% | art:886d9c2a… |

  - In h (H200 FP8 peak-seconds over a block's MACs): pe-w13 goes from 6.2e8 per session at K = 1 to 1.8e8 effective, and pe-w2 from 4.4e8 to 1.3e8.
  - h is a per-statement ratio, the same for any monitoring period and scope.
  - The per-K table (e2e and h per session, blocks/s, h effective and busy at K = 1, 2, 4, 8, for every configuration) is in the note `20261107T2347Z-report-overlap`.
- **Device memory caps K.** Each prover holds a device arena of about 16.5 x 2^(m-3) bytes for its whole process, 66 GiB at a full statement.
  - K = 2 on full statements ran out of memory, and the sessions were rejected (R7; art:6be3b090…).
  - So each of K provers proves at most full/K blocks.
  - That is why overlap alone stays at x1.1-1.3: the per-session fixed cost (about 6 s at k24, 12 s at k25) is paid per statement, and statements shrink by K.
  - The host is not the limit: 3.7 of 20.4 CPUs, never throttled.
- **The cache measured on the H200** (K = 1, end-to-end basis): fixed seconds pe-w13 5.73 to 0.56 (10x) and pe-w2 5.54 to 0.67 (8.3x). The marginal per block is unchanged.
- **The two levers together, as fits at K = 4:**
  - fixed x0.045 (pe-w13) and x0.043 (pe-w2 with MPS) a statement;
  - marginal x0.58 and x0.54 a block.
  - So the marginal reaches the busy reading. The fixed per block is x0.17-0.18, because a statement holds only full/K blocks.
- **The remaining lever:** free each prover's device arena between sessions, with the GPU phases serialized, so that K provers prove full statements. It is a CUDA change to the prover, and it isn't measured.
- The pod's lease ends 00:08Z, and it stops itself.
- All four runs are preserved (`research data put --preserve`, evidence/v1), and every session in runs 2-4 was accepted.
- Branch `cursor/proofs-stationary-9ca7` @ 6f4377828f: `zk_cost.py prove --provers K --stmt-cache` and `stationary_gpu.sh` with `ST_OVERLAP`, `ST_CACHE`, `ST_MPS` and `FC_VERIFY_AHEAD`.
- 23:52Z (proofs) STATIONARY measured overlap on an H200 (note 20261107T2347Z-report-overlap; your inbox has the table): overlap ALONE reaches only 17-20% of the busy reading you assumed; with the statement cache and verify-ahead (and MPS for pe-w2) it reaches 73% (pe-w13) and 83% (pe-w2). Marginal reaches the busy reading; fixed stays at about x0.17-0.18 per block (not x0.058) because K provers can each hold only full/K blocks (66 GiB a prover). Reprice the main line with these MEASURED factors instead of the assumed busy reading, and show the extra lever 'prover frees GPU memory between sessions' (unmeasured, kernels) as a labeled extra like the hashing cut. HILLCLIMB and STATIONARY send the fused block's measured price.

## 00:00Z from HILLCLIMB: the plain arm's per-type share, and one scope on the cache's factor
- **On CPU, `--zk` costs 30% or more of a session only for all-reduce (35%) and the registered MoE GemmW (30%).** For the fused cut's per-expert GEMMs it's 18% (w2) and 0% (w13), and for the dense quant, FlashInfer and combine types it's −4% to 11%.
  - Runs r20261107-221348-83c7 and r20261107-221414-0e6f, on cpu-2 at 16 cores. Each type's design statement is 2 blocks, padded to 8. The statement cache was on.
  - `nofw` (`--zk`, the prover alone) is compared with `plain` (no `--zk`). The numbers are medians of 2 timed sessions after a warm one, and every session was accepted in both arms.
  - The share is 1 − plain / zk:

  | type | e2e s, zk → plain | `--zk` share of e2e | witness + PIOP s, zk → plain | `--zk` share of witness + PIOP |
  |---|---|---:|---|---:|
  | all-reduce W8 | 21.08 → 13.72 | 35% | 19.91 → 13.06 | 34% |
  | MoE GemmW K256 (registered) | 12.36 → 8.69 | 30% | 11.77 → 8.47 | 28% |
  | expert w2 K256 N32 (per-expert) | 1.71 → 1.41 | 18% | 1.43 → 1.25 | 13% |
  | EmbeddingRead H3072 | 3.70 → 3.30 | 11% | 3.33 → 3.08 | 8% |
  | quant K1536 | 1.87 → 1.67 | 11% | 1.42 → 1.31 | 8% |
  | quant K2048 | 2.59 → 2.40 | 8% | 2.00 → 1.77 | 11% |
  | combine N5 | 4.05 → 3.71 | 8% | 3.39 → 3.15 | 7% |
  | expert w13 K6144 (per-expert) | 1.40 → 1.40 | 0% | 1.18 → 1.24 | −6% |
  | FlashInfer K2048 | 3.58 → 3.67 | −3% | 2.49 → 2.57 | −3% |
  | quant K6144 | 6.38 → 6.59 | −3% | 5.04 → 5.22 | −4% |

- **How far to trust it:**
  - With 2 timed sessions, ±5% is noise, so the small and negative rows mean "no measurable saving".
  - The arms differ in their coins too: plain must use the coin tree (`FC_COINS=tree`), and `nofw` uses seed coins. A coin costs little next to a proof, but this isn't a pure `--zk` toggle.
  - It's CPU only. On the H200, ZK extras were 56% of MoE GemmW's marginal device time, so the H200 share can be much larger than CPU's. STATIONARY's plain arm on `vy-spot-overlap` decides the 30% question.
  - It's a per-session ratio, so it reads the same for a week, a month and a year, and for a datacenter or the world's compute.
- **Against the 30% switch rule, on CPU:** w13 (0%) and w2 (18%) are below 30%. Only all-reduce and the registered MoE GemmW (which the fused cut replaces) reach it. The H200 verdict for w13 and w2 follows from STATIONARY's run, as does the fused blocks' own share. CPU's fused shares come from r20261107-235242-1fe5 and r20261107-235302-669b, below.
- **The cache's factor needs long-lived processes.** The cache lives in the process. It removes setup only when one verifier and one prover process prove many sessions of one type at one drawn count: the proof server's loop, the shape your busy reading assumes, and what STATIONARY measured on the H200 (10× and 8.3× on the fixed cost for w13 and w2).
  - SPOT's leg as wired today (`prove_leg.py`) starts a fresh serve and prove pair for each sampled member, so there the cache never hits.
  - So price it as a property of the proof server, not of today's leg script.

## 00:35Z from HILLCLIMB: the fused per-expert block on CPU: 0.59× (w2) and 0.75× (w13) of the unfused pair it replaces (measured on CPU)
- **Measured on CPU** in r20261107-235302-669b on cpu-13, 16 cores, cache on: 16 blocks a statement, median of 2 timed sessions, every session accepted.
  - Fused and unfused ran in the same job on the same node.
  - Staging passes only if every read inside the fused block opens correctly (`ok` = 1, checked against the expected outputs).
  - The fused types are `MoeExpertFp8ReadGemm(W)_v1` at the partition card's statics, with `READ = EmbeddingRead_v1{V=256, H=NK/2}`, both at `k_log` 25.

| fused block | e2e s: fused vs unfused pair | ratio | witness+PIOP ratio | plain arm ratio |
|---|---|---:|---:|---:|
| w13 (GEMM + read H3072) | 4.57 vs 6.07 | **0.75** | 0.77 | 0.69 |
| w2 (GEMM + read H4096) | 4.10 vs 6.92 | **0.59** | 0.60 | 0.71 |

- **What the unfused pair is.** It's the per-expert GEMM block (`k_log` 24) plus a read block (`k_log` 25) of the same row.
  - read H4096 isn't measured: its cost is scaled from H3072 by staged ANDs, 1.22×.
- **So the modeled price, the two marginals summed, overstates the fused block** by about 1.3× (w13) and 1.7× (w2) on CPU.
  - These are e2e ratios at 16 blocks, so the fixed part is included.
  - The H200's own number is STATIONARY's r20261108-000152-53c3 (fr-w13 and fr-w2 at 1024 and 512 blocks), by 01:30Z.
- **Likely conservative for the fused side.** Its sessions proved while two other jobs shared cpu-13 (00:02–00:17Z). The unfused sessions proved after those jobs ended.
- **The marginal per block and the full statement aren't measured yet.** The 2-block run (r20261107-235242-1fe5) proved at the same moments as the 16-block run, so the two-point fit is contaminated: pe-fw13 even came out negative.
  - Clean reruns prove each size one after the other, at 16 and then 64 blocks:
    - r20261108-003351-566e on cpu-13: fused w13 and w2, ETA about 01:25Z for w13 and 01:45Z for w2;
    - r20261108-003417-49b0 on cpu-1: unfused read, w13 and w2, ETA about 01:40Z.
  - The full statement, about 1024 blocks at `k_log` 25, will be fixed + 1024 × marginal, labelled extrapolated.
- **What `--zk` costs the fused blocks on CPU:** w13 19%, w2 −3% (plain is no faster). Both are below the 30% switch. The H200 verdict is STATIONARY's 53c3.
- **The AND counts don't compare across tools.**
  - stage.py counts read H3072 at 6,964,018 ANDs, where circuit-check (the partition card) says 4,974,722. The GEMMs agree (2,180,169 against 2,196,306).
  - Each fused block's staged count is exactly its GEMM's plus its read's: w13 is 9,144,187 against the card's 7,171,028, and w2 is 10,680,491 against 8,328,537.
  - Seconds are measured, so nothing above moves. But any figure priced per AND from the card's read counts is low by about 1.4×.
- **Per monitoring period:** these are ratios, so they read the same for a week, a month or a year, and for one datacenter or the world.

## 00:53Z from STATIONARY: the plain arm and the fused per-expert block on one H200

**Plain arm: on the prover's session, `--zk` costs 35% of pe-w2 and 28% of pe-w13.** So **w2 clears the 30% threshold and w13 doesn't.** The fused blocks that replace them are below it: 21% for fr-w13 and 23.5% for fr-w2.

- **The run.**
  - r20261108-000152-53c3 on vy-spot-overlap (1x H200), evidence art:f68065a805491863634a7a8a711b46cd9ba3505aa3ee287404dce87f9978af41.
  - Every session was accepted.
- **The comparison.** It is `dev` against `plaindev`:
  - Both run `prove --gpu` on the device's witness, at one prover, with the statement cache on and no verify-ahead.
  - `plaindev` drops `--zk` and runs on the challenger's coin tree (`FC_COINS=tree`), the tree `--zk` also draws from. So only `--zk` differs. HILLCLIMB's CPU pair (`nofw` against `plain`) differs in its coins too.
- **Noise.** The numbers are medians of 4 timed sessions after a warm one. Each arm's sessions agree within about ±3%, so 28% reads as 26-31%.

  | type | blocks | session s, zk → plain | `--zk` share of the session | GPU prove phase s, zk → plain | share | verifier s, zk → plain |
  |---|---:|---|---:|---|---:|---|
  | pe-w13 | 2048 | 3.09 → 2.21 | **28%** | 2.23 → 1.41 | 37% | 3.48 → 0.17 |
  | pe-w2 | 2048 | 2.98 → 1.95 | **35%** | 2.45 → 1.48 | 40% | 3.52 → 0.17 |
  | read (EmbeddingRead_v1 H3072, registered) | 1024 | 3.15 → 2.37 | 25% | 2.30 → 1.52 | 34% | 4.19 → 0.30 |
  | fr-w13 (fused, #7852) | 1024 | 3.65 → 2.89 | **21%** | 2.47 → 1.61 | 35% | 4.27 → 0.51 |
  | fr-w2 (fused, #7852) | 1024 | 3.73 → 2.85 | **23.5%** | 2.46 → 1.59 | 36% | 4.25 → 0.52 |

- **What the share depends on:**
  - **The GPU proving phase alone.** There `--zk` is 34-40% for every type. The rest of the session doesn't change with `--zk`: the witness (0.2-0.8 s), the wire and the setup.
  - **The verifier.** Zero-knowledge verification takes 3.5-4.3 s a statement, against 0.17-0.52 s plain.
    - At one prover without verify-ahead the prover waits for each verdict. So `--zk` is 55-66% of a cycle: 6.7-8.6 s against 2.3-3.8 s.
    - Here the verifier shares the prover's 20 CPUs. In SPOT it runs on the verifier's machine, so the session share is the proof server's price.
  - **Not measured: plain at K > 1.** With verify-ahead at K = 4, `dev`'s cycle (3.3 s) is longer than its session (2.3 s), and the zero-knowledge verifier (2.7 s a statement under that load) is the likely limit. So plain's overlapped gain may be larger than its session share.
- **Scope.** The shares and h below are per-statement ratios. They read the same for a week, a month or a year, and for a datacenter or the world's compute.

**Fused per-expert block: measured on the H200, so this price replaces the modeled one.**

- **The blocks.** Both have the statement cache on and run the `dev` arm. Each block is k_log 25, so a full statement is 1024 blocks.
  - fr-w13 is `MoeExpertFp8ReadGemm_v1{READ = EmbeddingRead_v1{V=256, H=3072}, K=6144, N=1, G=KB=128}` on the MmaV2 tile: 9.14 M ANDs a block.
  - fr-w2 is `MoeExpertFp8ReadGemmW_v1{READ H=4096, K=256, N=32, SWAP}` on the Qgmma tile: 10.68 M ANDs a block.

  | measure | fr-w13 | fr-w2 |
  |---|---|---|
  | full statement at one prover, session | 3.65 s = 3.57 ms a block, h 5.7e8 | 3.73 s = 3.64 ms a block, h 4.4e8 |
  | one-prover fit (1024 and 512 blocks), cache on | fixed 0.74 s + marginal 2.85 ms a block (h 4.6e8) | fixed 0.68 s + marginal 2.98 ms a block (h 3.6e8) |
  | best overlapped, cache and verify-ahead | 309 blocks/s at K = 4 on 256 blocks (h 5.2e8); 283 over the provers' whole span | 288 blocks/s at K = 4 on 256 blocks (h 4.2e8); 242 over the whole span |
  | one prover, cycle including the verdict | 120 blocks/s | 119 blocks/s |

- **The fixed cost.** It is measured with the cache on, so don't apply the cache's factor to it again.
- **w13 against the unfused pair it replaces, on the same run and basis.**
  - Per expert row, pe-w13 (1.51 ms a block) plus read H3072 (3.08 ms a block) is 4.58 ms, and fused is 3.57 ms: 0.78x.
  - The marginals: fused is 2.85 ms a block, against the modeled sum of 1.10 + 2.39 = 3.49 ms from the main overlap run's one-prover fits: 0.82x.
  - At each one's best overlapped throughput: the unfused pair proves 191 rows/s (pe-w13 at 897 and read at 243, in series), and fused proves 309: 1.6x.
- **w2.** Read at H = 4096 isn't measured, so there's no pair comparison yet. fr-w2 costs 3.64 ms a block.
- **Overlap gains less on fused blocks.** At one prover the session basis is already 280 and 275 blocks/s, and the best overlapped throughput is 309 and 288. Most of overlap's gain here is hiding the zero-knowledge verdict: the one-prover cycle gives 120 blocks/s.
- **This run's K = 8 isn't valid.**
  - The eight provers' warm sessions ended up to 15 s apart, and 4 timed sessions of about 3 s each never all overlapped. So the nominal 330 and 323 blocks/s overstate throughput.
  - A rerun with 12 timed sessions at K = 4 and 8 is r20261108-005139-36f3, due about 01:12Z. I'll append it.

**Housekeeping.**

- The main overlap run's evidence (art:886d9c2a…) now carries `source = cursor/proofs-stationary-9ca7@5f5c8e8616`, pushed. The store's vocabulary has no `commit` key.
- The code is on `cursor/proofs-stationary-9ca7` at 011d76a931:
  - the `zk_cost.py` arm `plaindev`;
  - the `stage.py` types fr-w13 and fr-w2, with #7852 merged;
  - the runner's `ST_ARMS1` and `ST_VERIFY_AHEAD`.

## 01:13Z from STATIONARY: the fused blocks' overlapped price, with 12 timed sessions; K = 4 is best, and host memory caps K = 8

**These rows replace the "best overlapped" row of my 00:53Z table.** Both fused blocks prove best at **K = 4 on 256 blocks**, with the statement cache and verify-ahead:

- **fr-w13: 329 blocks/s, h 4.9e8.** That is 318 blocks/s over the provers' whole span.
- **fr-w2: 307 blocks/s, h 3.9e8.** That is 297 blocks/s over the whole span.

- **The run.**
  - r20261108-005139-36f3 on vy-spot-overlap, evidence art:7fcd29ebbc33b065009296a4baaea5e14b61f48bc83987f2cc38b4013a79cf22. The source is the same as before (011d76a931).
  - Every point but one was accepted, each with 12 timed sessions. The provers' overlap windows were 27-40 s, so these throughputs are valid; the K = 8 points at 00:53Z were not.

  | type | blocks | K | blocks/s | over the span | h | GPU busy | session s | verifier s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | fr-w13 | 256 | 4 | **329** | 318 | 4.9e8 | 0.48 | 2.38 | 2.76 |
  | fr-w13 | 128 | 4 | 219 | 211 | 7.4e8 | 0.43 | 1.64 | 2.23 |
  | fr-w13 | 128 | 8 | 287 | 263 | 5.6e8 | 0.59 | 2.34 | 3.48 |
  | fr-w2 | 256 | 4 | **307** | 297 | 3.9e8 | 0.49 | 2.46 | 2.86 |
  | fr-w2 | 128 | 4 | 217 | 206 | 5.6e8 | 0.42 | 1.68 | 2.20 |
  | fr-w2 | 128 | 8 | out of memory | | | | | |

- **K = 8 is capped by host memory, not device memory.**
  - fr-w13 at K = 8 peaked at 231 GiB of the pod's 251 GB memory limit, and was slower than K = 4.
  - fr-w2 at K = 8 crossed the limit: the pod's cgroup recorded 2 out-of-memory kills, and the shared verifier died mid-session. No retry was made.
  - So K = 4 is the fused blocks' best on this node.
- **What changes against 00:53Z:**
  - **w13 against the unfused pair, at each one's best overlapped throughput.** Fused proves 329 rows/s; the pair proves 191 rows/s (pe-w13 at 897 and read at 243, in series). So fused is **1.7x** the pair, and 0.78x its time a row at one prover.
  - **h for the lever paths, on the H200 with the cache on:**
    - fr-w13: 5.7e8 at one prover (session basis), 4.6e8 on the marginal, and **4.9e8 at the best overlapped K**.
    - fr-w2: 4.4e8, 3.6e8 and **3.9e8**.
  - **The verifier is the likely limit.** At K = 4 the zero-knowledge verdict takes 2.8 s a statement, against 2.4 s for the session, and the GPU is about half busy. A plain verifier (0.5 s) would lift this. Plain at K > 1 is not measured.
- **Scope.** h is a per-statement ratio, the same for a week, a month or a year, and for a datacenter or the world's compute.
- **The pod.** I've launched nothing more. vy-spot-overlap's lease ends no sooner than 01:44Z, and it stops itself.
