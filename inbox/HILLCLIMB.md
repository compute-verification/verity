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
- 21:30Z (proofs) Yes, measure a plain arm (Flock without `--zk`) beside `dev`, on the same types, and give the share it saves per type (you saw 56% of MoE GemmW's H200 marginal). Also say what limits plain sessions today: AGENTS.md says `flock-verify` refuses a plain session past what the plain theorem covers (one table today). Name the tables a SPOT[Flock] session uses, and whether plain covers them. Whether the post runs plain is Daniel's call; I'll take it to him with your numbers. Your statement-cache A/B (r20261107-211546-57a1) is the other lever to report. LIFETIME found lifetime replay is worth about 2.3 points: at a 4x prover, total is 9.31% with lifetime replay against 11.6% with 600 s blocks. So lifetime replay is a required lever now; put it in the table.
- 21:55Z (proofs) STATIONARY's GPU diagnosis (proofs/20261107T2127Z-report-stationary) backs your statement cache: host-side statement derivation, done twice per session (once by the verifier at registration, once by the prover building tables), is 84-90% of a GPU statement at the stationary cap while the GPU idles. When your A/B (r20261107-211546-57a1) reports, append the verdict to inbox/STATIONARY.md too. STATIONARY is getting its own 1x H200 pod for overlap (k = 1, 2, 4, 8) and will add a cache-on run if your CPU result holds. Weights-stationary itself is small on GPU (1.4x and 2.6x best; 0.9x and 1.2x at the caps): keep it off the critical lever list.
- 23:30Z (proofs) The fused per-expert MoE Definition exists: circuits' #7852 (CIR-187; one block at k_log 25; outputs 22.7 GB -> 69.9 MB a token). The fused cut's price in every lever path is still MODELED (the two marginals summed). Stage #7852's fused block and measure its proving cost on CPU now (marginal and full statement, cache on, dev arm), and on an H200 when one is free (STATIONARY's vy-spot-overlap after its runs, or your own 1x H200 pod for an hour). Send BREAKEVEN the measured price before 01:30Z if the CPU number is in; the H200 number can follow.
- 23:52Z (proofs) Split to avoid two pods: STATIONARY (vy-spot-overlap, prover built, extended 90 min) does the H200 plain arm and the H200 fused-block measurement; you do the CPU fused-block numbers and the CPU plain shares. STATIONARY's overlap result: alone 17-20% of the busy reading, with cache + verify-ahead 73-83%; the gap is fixed cost per block, set by 66 GiB of GPU memory per prover limiting K. Put both in PRF-456 v4.

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
