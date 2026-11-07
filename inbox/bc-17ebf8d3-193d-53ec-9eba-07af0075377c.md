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
