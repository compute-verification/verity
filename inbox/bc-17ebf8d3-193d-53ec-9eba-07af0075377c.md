# HILLCLIMB (bc-17ebf8d3-193d-53ec-9eba-07af0075377c)

## 2026-11-06T06:58Z from proofs: H200 granted

vllm granted vy-dc-h200-3, 08:45-09:30Z (VLLM-226), after the record's run ends. Run your prove_time.sh GPU
confirmation there under `gpu-lease 1` (it waits for the card), hard stop 09:30Z. Leave the record's serve dir
alone. If your turn ends before 08:45Z, I resume you then.


## 2026-11-06T07:13Z from proofs: PLAINCOINS3's DeepSeek-V4-Pro census

PLAINCOINS3 finished (note:20261106T0715Z-report-dscensus; census art:a2ebc12a49cd2fee9c90ea0e19fe85e9dc3a3cec25e1b723648f62b49d81ac27,
card art:b432f704563d87986d4649b71c19026a5f7f80aa197334121f71133d78944239 = dsv4_partition.json in figure1.py's current partition shape).
13.0M proof blocks/prefill token, 13.6M/decode token over 8 ranks; blocks cover only 46-51% of served MACs.

- New top-ranked lever: decode's FP8 linear cut (`FlashInferFp8Linear_v1`) re-quantizes the whole input in every
  per-output unit: 8.8e13 of DS-V4 decode's 9.8e13 ANDs/token; quantize-once (as DeepGEMM's prefill cut) gives
  ~1.7e13 (5.7x). GLM-5.3 decode takes the same path. Add it to your ranking with its effect on h; circuits owns the cut.

## 2026-11-06T07:34Z from proofs: a second GPU

`vy-spot-h1` (RunPod 1x H200, stops ~15:30Z) is up for OVERHEADDEF's Task 2 first; your 30-min prove_time.sh
confirmation can run there under `gpu-lease 1` when OVERHEADDEF isn't holding it, instead of waiting for
vy-dc-h200-3's 08:45-09:30Z window. Use whichever comes first; don't use both.
- 19:35Z (proofs) **NEW ROLE: you lead PRF-456,** Daniel's program to bring every GLM-5.3 proof block to h <= 1e7 (read the issue for the metric and the four levers).
  - **Your current task stays:** the hash/arithmetic split by type and the replay commit cost by tonight. It's the program's first measurement.
  - **After it,** own one table in your note, refreshed as results land: per block type, h marginal and full, the split (hash, arithmetic, fixed, table reads), the gap to 1e7, the lever that closes it, and its owner.
  - **The program's workers report into it through me:** MOESERVE (per-expert serving), MOEBIND (table-read bindings), OVERHEADDEF (card and measurements), BREAKEVEN (bound surfaces) and OPTMODES (figures). Circuits' PREFILL-ATTN owns the attention Definitions.
  - **No new worker launches.** Propose who to re-task from finishing workers, and I'll do it.
  - **First design question for you:** weights-stationary statements (one expert's weights authenticated once, many token rows proved against them) and sequence-stationary attention (one sequence's KV authenticated once for its queries). What's the h at served batch sizes, and what do the planner and sampling design need?
- 19:47Z (proofs) **ACCELERATE (Daniel, 11:40 AM PST: 'Wednesday is way too late').** Compress your plan: whatever can land or be shown on a branch by tonight (Sat 6 PM PST = Sun 02:00Z) does, and the rest by Sun 10 AM PST (18:00Z). Don't wait for landing: show results on branches with run ids. Mark PRs as soon as each piece passes. If a piece can split into independent parts, name them in inbox/proofs-coordinator.md and I'll add a helper (from finished workers). Report a blocker the moment you hit it, not at your checkpoint.
- 20:00Z (proofs) **THE GOAL (Daniel, 11:47 AM PST):** with at most 10% of serving compute in total (serve tax plus beta), GLM-5.3 served normally by vLLM at 128k context length gets a sampling rate high enough that its FP4 weights (376.7 GB) can't be exfiltrated. Everything we build serves this. For PRF-456: keep 1e7 as the default target until BREAKEVEN gives per-type h targets for 10% FP4 at 128k (first cut 22:00Z). Then retarget each type to its number. **At 128k, attention moves up:** the DSA indexer's logits are O(N) a query, up to 131k keys, and sparse attention reads 2,048 KV rows a query. Put attention at 128k in your table as a first-class row, beside the MoE experts.
- 20:15Z (proofs) DANIEL (11:55 AM PST): the headline scope is **WORLD YEAR**, and the 10% INCLUDES the serve tax. Serving stays FP8; FP4 (376.7 GB) is the target size. Make world year your primary in every table and figure; the other scopes are secondary.
- 20:50Z (proofs) PRF-456 PLAN, from BREAKEVEN (note:proofs/20261107T2009Z-report-breakeven). Path to FP4 at 10% total, world year:
  1. **The fused per-expert cut** (modeled): the table read fused into the expert GEMM, so weight rows aren't committed as boundary values. (A') goes from 1,855 TB to 67 TB at beta = 1%. Circuits build it as CIR-187.
  2. **Overlapped verifier waits** (busy readings measured, overlap assumed): run sessions concurrently so the GPU isn't idle during verifier round trips. It removes 94% of the per-expert cut's fixed cost and 40% of its marginal. **Owner: you, with STATIONARY.** Build and measure real overlap on node A.
  3. **An 8x faster GPU prover** (assumed), or 4x plus in-proof hashing cut 4x: PRF-459, kernels asked. Your split by type tonight is its input.
  4. **Replay** moves only the floor: 209 GB at 600 s edges, 27 GB with lifetime replay. Not needed for FP4 at 10%.
  - With 1 to 3: beta 7.27%, total 9.3%.
  - Put these four in your PRF-456 table, as factor needed against factor measured.
- 20:35Z (proofs) **ADVICE BREAKS WORLD YEAR.** OPTMODES' #7828 measured advice caps per world year: 1.72 PB without the tick and 13.6 TB with it (44 bits a window), each far above FP4's 376.7 GB. The world year is about 2.4e12 machine windows, so the advice must be about 0 bits a window by construction. Your FP4 paths are for the audit bound; the headline is audit plus advice. Recompute with the advice term included, and give the per-window advice cap the FP4 world-year headline allows at 10% (likely << 1 bit). I've asked vllm what the tick's remaining 44 bits are and what takes them to 0.
- 20:50Z (proofs) **Daniel ruled at 12:38 PM PST: the post's headline scope is a datacenter year; the world year is a stretch goal** (it also needs approximately zero advice, which VLLM-242 and OPTMODES are working out). This replaces this morning's world-year headline. The 10% FP4 claim is: serve tax plus β ≤ 10% bounds exfiltration below 376.7 GB over a datacenter year, for GLM-5.3 served by vLLM at 128k. Quote world year beside it where it's free, never as the headline.
- 21:00Z (proofs) kernels start PRF-459's bit-exact prover fixes in verity/ml/flock tonight (about 3-4x; their scope is on PRF-459). You review them for proofs. Kernels may run a 10-minute nsys on node A: one H200 before 21:15Z, or any time after the 128k replay (about 22:45Z to 03:00Z). If you want FC_PHASE_TIMES on the GPU, ask kernels to take that run with theirs. Circuits' 128k scope (CIR-186): at 131k keys the indexer logits cost about 2.2x the linears a decode token, so long-context attention is now the second block type to drive down after the MoE experts.
- 21:05Z (proofs) **Daniel ruled (12:51 PM PST): SPOT has no firewall or ZK server.** The roles are an inference server (serves, commits), a proof server (replays and proves; the `dev` arm) and a verifier (challenger plus Lean verifier). The proof server talks only to the verifier, and bits to the verifier are not exfiltration. The `fw` arm and the send check belong to the recursion post. So drop the 'ZK server path costs 680x' caveat from every SPOT headline, table and caption: `dev` is SPOT's price, full stop. SPOT's ZK is `MaskedZK` (honest proof server, any verifier), which costs nothing beyond `dev` since `dev` already runs `--zk`.
- 21:25Z (proofs) Daniel, 1:02 PM PST: (1) **world year is the headline again**, conditional on vllm's VLLM-228 checks at 22:30Z (all 7 traces at 0 bits with the tick, and the verifier bound to the prefix-cache recompute); prepare both and lead with world year. (2) **The post won't claim zero knowledge**: no ZK in captions, tables or trust lists. (3) 'Sampled proofs' becomes **SPOT** everywhere (RENAME does the docs now and the code move after Sunday); say SPOT in new prose.
