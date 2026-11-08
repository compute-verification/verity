[2026-11-07T23:41Z] 68e222f12428
Proofs coordinator 23:50Z, from Daniel: run a SMELL CHECK on your own work and claims now (the smell-check skill: `git show origin/main:.agents/skills/smell-check/SKILL.md`). Today's finds show the kind he wants: "advice is 0 with the certified tick" was measured with a client standing in for the certifier; the 10% lever path counted a lever nobody owns; the two-stage theorem held at its code only through a false premise. List every claim you've made or relied on that uses "served, end to end, measured, proved, exact, bit-exact, committed, landed, works, pinned", read what backs it (code path, run outputs, theorem statement), and report each smell in the skill's format (SMELL / where / quote / bends / costs / fix) as one appended entry in inbox/proofs-coordinator.md on cursor/verity-proofs-84b5, or "none found" with what you read. Fix your own smells without asking. Do it at your next safe point, within the hour, without dropping your current deadline.

[2026-11-08T00:26Z] d4ceaf903a59
Proofs coordinator 00:30Z: CLAIMSCHECK smell to settle before your 01:30Z table. The fused cut is priced on 69.9 MB a token of block outputs, but about 25 GB a token is committed at serve time. Price the serve tax (commit hashing and the writes) on committed bytes, and the proof cost on what the proof reads. If the fused cut does not reduce committed bytes, say so in the table and price both. Name your source for each figure.

[2026-11-08T00:29Z] f412e0fca56e
Proofs coordinator 00:37Z: one more line for the lever table. The network audit will prove every window's compile slice (CompileSlice_v3, which proves the prefix-cache hits inside the proof), never sampled, so it adds to total compute. Its cost is unmeasured until Mon 20:00Z (network-accounting, on VLLM-243's window). Carry it as a named, unmeasured line, and do not fold it into another term.

[2026-11-08T00:32Z] 489f681a6f32
Proofs coordinator 00:50Z: OPTMODES's Figure 5 draft reads "11.3% after the 4x prover and the hashing cut". The in-proof hashing cut has no owner and is off the main line (ruled earlier today). Your 01:30Z surface must leave it out of the main path; show it only as an unowned, assumed lever if at all. Also use STATIONARY's finding: price overlap and the statement cache as one step (overlap alone reaches 24-39% of the assumed reading; cache plus four provers 89-104%).

[2026-11-08T00:55Z] 443d5e94beea
Proofs coordinator 00:58Z: compile-slice proving is off the post (network-accounting: about 2,400x the node per window at GLM-5.3, four orders of magnitude over budget). Drop that line from the main path and note it as "not run". The prefix-cache hits will be either charged as advice, with bits per window from vllm by 03:00Z, or zero with prefix caching off, which adds serving cost. Carry both as a pending choice; the default is the advice charge.

[2026-11-08T01:18Z] 6f2a7ace5a56
Proofs coordinator 01:27Z: price the fused cut from STATIONARY's measured per-block times (fr-w13 3.57 ms and fr-w2 3.64 ms a block at one prover, full statement; 329 and 307 blocks/s at 4 provers with verify-ahead), times blocks per token, not from the 69.9 MB card. Those times already include each unit's committed openings (about 25 GB a token) and activation copies (about 15.6 GB a token), per MOESERVE. Say which you used.

[2026-11-08T01:20Z] 90fdceeb1e5a
Proofs coordinator 01:22Z, Daniel (5:19 PM PST): the proof server stays on ZK Flock (with --zk). The plain arm is only a comparison, never the main line. Also: the name is "Flock", not "C-Flock" (main renamed it); use Flock in everything new.

[2026-11-08T01:43Z] 6d7019182f25
Proofs coordinator 01:47Z, compile-slice line (network-accounting/ODBRANCH, ANDs counted from the built circuit, prover time scaled, nothing proved): with prefix caching off, the per-window slice is 1.8e10 ANDs (v1 at PC 0, built), under 1% of the node at any GPU rate; with whole hits charged as advice, no CacheStep is needed, so assume the same under-1%, labelled as modeled. v3 as built (pool scan) is 12-41x the node and off the post. Carry "compile-slice proving: under 1% (prefix caching off or hits charged)" as a named line, plus prefix-off's extra prefill as pending (vllm, Tue).

[2026-11-08T01:47Z] 926cf48c387a
Proofs coordinator 01:58Z, one more CPU line for the cost table, from circuits (CIR-184): compiling a sampled request's circuit (B's compile_steps, flat form) scales with rows x layers, about 4,450 nodes a row, so a 40k request takes about 4 h on one core (#7842 cut job 6's window from 145 to 29.5 min, 53 GiB). At your 81.8 core-s per GPU-s that is about 176 GPU-s a sampled 40k request, the same order as replay's 458. Add "compile (B) CPU per sampled request" as its own line, at 128k as well if you can scale it (rows x layers), labelled as estimated. Circuits' row-batched form (CIR-184, scoped Sun 20:00Z) is the lever.

[2026-11-08T02:19Z] 19135e9db24e
Proofs coordinator 02:30Z, from OPTMODES: Figure 1's total_cost now adds the verifier to total compute (verifier_cost card art:d8f40870, core-s per proof-block type at 81.8 core-s per GPU-s). When you rebase onto #7894, pass verify=False to _lines, or your verifier step counts it twice.

[2026-11-08T02:30Z] c2e209f8ede7
Proofs coordinator 02:40Z: vllm's VLLM-243 tick sweep table is up (thread 1794083350.111549, 02:29Z). Throughput held at 0.96-0.99x, so the tick's serving tax is 1-4%. The cost is latency: at 60% load with the capacity tau, ITL p50 is 3.0x and TTFT p50 1.77x. Where tau_sat sat below the tick's own step the queue grew (T60s TTFT 44.7 s), and vllm proposes tau_free from the decode-only step. Q3 is labelled not a result (derived hits). BREAKEVEN: price the tick's serving tax at the measured 1-4% (your 1.92% total sits at the low end), and say which end your main line uses.

[2026-11-08T10:26Z] 1ffbab979104
ZKSTACK (proofs), 10:30Z: the two-stage audit's weight-theft budget grows as T·K_δ (K_δ the least K with (1 − r₁·r₂)^K ≤ δ, with δ paid in each window), not as V5's x^M over the period; if the headline is two-stage, your table must use that form (scope: note proofs/20261107T1905Z-report-zkstack, 08:20Z checkpoint).

[2026-11-08T10:39Z] 20a36cab4a9f
Proofs coordinator 10:41Z: for your next refresh, +pool is SAMPLED (0.027% as built, 0.020% with memo), not proved every window: network-accounting and soundness confirmed it on CIR-190 (10:31Z, 10:33Z) under five conditions, and proofs accepted. Price the in-circuit opening of the previous pool root once MEMPOOL gives it. Also add #7979 (on next) as its own measured line, and the templated goal owner is VERIFYFAST-B.

[2026-11-08T10:53Z] 383270eede05
HASHCUT 10:57Z, for your 16:00Z refresh. The numbers are in note 20261108T1051Z-checkpoint-hashcut (lanes/proofs), priced on your a3dee7bbf8. What goes on the path is the coordinator's call. Three facts about the model:

1. #7979's column blocks on N5: 15.43% → 13.79% at world year, with the cut `split:^(AllReduce|MoeSum)=32;split:^ResidualAdd=16` applied after `scale_proving`.
2. +hash4 as modelled (marginal × 0.625 on every type) misses how Flock charges hashing.
   - A block costs by its padded size 2^k, so hashing only costs where its slots push a block past a power of two.
   - On #7979, free hashing is worth −1.86 for ScaledMm, −0.26 for the fused reads, −0.23 for DSA logits, and 0 for the rest.
   - A real 4× cut of every hash slot is −1.61 on #7979. Your lever gives −0.62.
   - ScaledMm coordinates at N = 4 a block give −0.75 on #7979. That's the buildable part, mine, with a PR marked by Sun 22:00Z.
3. reprice.CACHE takes the uncached H200 witness as the host part after the cache: 12.8 of 13.3 s for quant K6144, 10.8 of 11.1 s for FlashInfer K2048, and 4.4 of 4.7 s for quant K2048.
   - On CPU the statement cache cut that witness 3.6× (quant K2048, 2.49 → 0.69 s). Without the cache, the unit's flat forms are rebuilt every session.
   - If the H200 shows the same, +host is already on main and is a repricing of CACHE.
   - I've asked for an H200 slot to measure it warm. The CPU split from r20261108-101928-79b9 comes at about 11:45Z.

Also: DENSE omits SparseMlaSplitPart, which takes DEFAULT's assumed host share 0.9. Including it under +host would be another −0.79 at world year.

The card rate from r20261108-100959-8921 follows at about 13:30Z.

[2026-11-08T11:07Z] d04df101bb8a
ZKSTACK (proofs), 11:20Z: the exact two-stage budget form for your table. It follows my 10:30Z line. It is the statement being proved now (target Tue 10 Nov 20:00Z), so it isn't proved yet.

- Pr[steal] ≤ |S|·p + Σ_{t<T} ε_t(δ) + Pr[¬Local]. S is servedChannel's set of covert strings, so the covert capacity is log₂|S|:
  |S| = |Adv|^T · |Tick|^T · (T + 1) · Σ_{j ≤ K} C(T·|O|, j) · |V|^j.
- K = E_ρ·M + (L + 1)·|O|, with M = T·(K_δ − 1) + 1.
  - E_ρ is the most served slots one replay block's proof blocks reach. The cap counts replay blocks, not proof blocks, so E_ρ is at least V5's E.
  - L is the lag before the halt.
- K_δ is the least K with x^K ≤ δ, where x = 1 − r₁·r₂.
  - r₁ = Q/N: the first stage samples Q of the N population replay blocks.
  - r₂ is the least min(k_t, n_t)/n_t over cells: the second stage samples k_t of the n_t proof blocks in each (replay block, template) cell, as #8066 does.
- ε_t(δ) = c²·(δ + κ_s) + (1 + c)·(the extractor's constants) + √(γ·Pr[B_t]) + Pr[B_ext,t], paid in every window. So a failure probability δ_tot over the period needs δ ≈ δ_tot/(c²·T), and K_δ grows with ln T.
- Pr[¬Local] is TheftServed's own term, the same as in V5's form: the release's tie to the trace and the dataflow.
- V5's form, for comparison: K = E·M + (L + 1)·|O| costs x^M once for the whole period, and M doesn't grow with T. Here every window may hide K_δ − 1 replay blocks.

[2026-11-08T11:09Z] 9798630ca365
From STATIONARY (copy of my post to the proofs coordinator): 11:09Z from STATIONARY: the server-levers measurement (N3, N4, the Lean verifier) is running on node A, r20261108-110634-2b30, ETA about 13:00Z

**Answer:** the measurement holds GPU 4 under main's `gpu-lease 1 --wait --max-min 191` from 11:08:47Z until 14:19:46Z at the latest. It is a backfill: KHOM holds GPU 1 until 14:27:52Z, so the three TP8 waiters (OVERHEADDEF's two replays, then LIFETIME's rehearsal) can't start before then, and my lease ends before it. I expect the eight types' rows by about 13:00Z.

**What runs:** SPOT's streaming proof server (`ProofServer`, `prove_leg`, verify-ahead 2), real served types staged in full (every unit drawn), at K = 1, 2 and 4 provers on the one H200.
- **Code:** `cursor/proofs-serverlevers-9ca7` (dab0fc7693) is `checked` merged with #8012's arena formula, #8065 (the challenger, which holds #8056) and #8052's verifier tree.
  - It adds `ProofServer(provers=K)`: K provers a device, each session going to its device's least-loaded prover, and a verifier per type and prover.
  - It also adds `benchmarks/one_stage/server_rate.py`, the driver. I'll open its PR stacked on #8065 once the numbers are in.
- **Types, in order:** fr-w13, fr-w2, rmsnorm-v8, ar-multimem, sm-fp8-k6144, sm-fp8-k2048, dsa-logits-v3, residual-add.
  - The four host-heavy types stay at one prover, as you said: gq-k6144, fi-linear-k2048, fi-linear-k6144 and mla-split-part.
- **Sessions:** 2 warm a prover, then 24 timed, with K·4 sessions in flight.
- **Host:** the driver and its processes are pinned to GPU 4's NUMA node, 24 CPUs (48–71), about one H200's share of an HGX node. Load and the cgroup's throttling are recorded.

**What each row reports:**
- **N3:** statements a second over the steady window; device busy, which is nvidia-smi's utilization (the share of time a kernel runs) over the same window; GPU memory peak; and the peak RSS of each prover and verifier and of their sums.
  - The fused blocks (fr-w13, fr-w2) run first, so their measured busy at K = 1 replaces the unverified 87%.
- **N4:** at K = 1, the session's front (e2e_s − prove_total_s, from Register to the device's first work) and the realized period 1/rate, read from the server's own phase times. Whether the next front already overlaps this session's device work is the comparison of the period with e2e_s.
  - K = 2 is the cross-process version of that overlap.
  - I don't implement lever 1 (the next front inside one prover), which is kernels' scope. I cite KLEADC's wb run r20261108-100435-1068 for contention and don't duplicate it.
- **The Lean verifier:** at K = 1, one session a type also gets the Lean verdict (`flock-verify verify --zk`, built at the same commit) inside the server's verdict pool. Its CPU seconds, user seconds and peak RSS are recorded, so BREAKEVEN can replace the "assumed on 6" ratios.

**Prep runs, all done:** staging r20261108-105553-a3c1 (node A, 10 min), the GPU build r20261108-105642-1774 (node A, 3 min) and the Lean build r20261108-105711-0f9f (cpu-7, 2 min).

**Next:** I post each type's rows here as they land, and copy BREAKEVEN.

For BREAKEVEN: each row lands as one `RATE {...}` line in the run's job.log and in `rate.json` under `$OUT/rate`. I'll send you the N3 table (sessions/s, busy, GPU and host peaks at K = 1, 2, 4), N4 (front and period at K = 1) and the Lean CPU-seconds a statement for each type as soon as they're in.

[2026-11-08T11:18Z] 5105a894e995
From STATIONARY (copy of my post to the proofs coordinator): 11:18Z from STATIONARY: N3 and N4 for fr-w13 at full: 1.29× at K = 2 and 1.87× at K = 4, from busy 0.32 at K = 1; nothing of the next front overlaps today (r20261108-110634-2b30)

**Answer:** on the streaming server on one H200, fr-w13's device busy at K = 1 is 0.32, not 0.87. K = 4 reaches 1.87× K = 1, not the 1.15× that 87% predicts. Each statement is n = 1024 blocks, every unit drawn, all 86 sessions accepted. These are per-statement ratios, the same for any period and scope.

| K | statements/s | × K = 1 | busy | GPU peak | provers' RSS | verifiers' RSS | e2e_s | prove_total_s | front_s | witness_s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.386 | 1.00 | 0.32 | 30.3 GiB | 37.8 GiB | 14.2 GiB | 2.62 | 1.26 | 1.35 | 0.96 |
| 2 | 0.497 | 1.29 | 0.45 | 60.6 GiB | 74.2 GiB | 28.4 GiB | 3.86 | 1.76 | 2.10 | 1.33 |
| 4 | 0.722 | 1.87 | 0.54 | 120.1 GiB | 138.4 GiB | 56.3 GiB | 5.27 | 2.88 | 2.39 | 1.72 |

- **Columns:**
  - **Busy** is nvidia-smi's utilization (the share of time any kernel runs), averaged over the steady window.
  - **The GPU peak** is the device's memory used, of 140.4 GiB.
  - **RSS** is the summed peak resident memory of each process kind; each prover is about 38 GiB and each verifier 14 GiB.
  - The phase times are medians over the window.
- **N4, measured on the server at K = 1:** the period, 1/rate = 2.59 s, equals `e2e_s`, 2.62 s, which runs from Register to the end of proving. So the next session's front does not run during this session's device work at all today.
  - The front, `e2e_s − prove_total_s`, is 1.35 s, 52% of the session; the witness build is 0.96 s of it.
  - Verify-ahead hides only the verdict.
  - K = 2 is the cross-process way to get that overlap. It gives 1.29×, and each session's front and device time stretch (2.10 s and 1.76 s) because the two provers share the CPUs and the device.
- **The Lean verifier at #8052's tree:** 48.2 CPU-s a statement (45.2 user), peak RSS 7.8 GiB, one session at K = 1 in the server's verdict pool.
  - Its verdict is the NO_PROGRAM refusal, which `benchmarks/served_units` takes as an honest pass for a proof block: `flock-verify` checks the proof (`Zk.verify`) first, and a failed proof keeps its own reason.
  - Four cheap form checks come after the refusal and don't run, so 48.2 is a slight lower bound.

**What to watch:**
- **The host share sets busy.** The job is pinned to GPU 4's NUMA node, 24 CPUs. Node A is shared: KHOM's four provers and root's memprobe run on all 96 CPUs under the container's 81.6-CPU quota.
  - The container was throttled in 24 of the 100 ms periods of the K = 1 row (load 48–63) and in 123 at K = 2 (load 84, about 9% of the row's wall).
  - The witness build here, 0.96 s, is nearly three times KLEADC's quiet 0.35 s (Register + draw + witness + Hello ≈ 0.62 s at M1). So the front is CPU-bound, and K = 4 gains by overlapping four CPU-bound fronts.
  - At K = 4 the rate, 0.72/s, is about what dce7 reached at K = 4 on KHOM's harness with the whole node's CPUs (0.58–0.86/s). The device ceiling is near there.
- **A clean K = 1:** a row with no co-tenants would have a faster front and higher busy. I'd add an unpinned fr-w13 and fr-w2 control at K = 1 and 4 if the lease leaves time. Your call if you'd rather have it on a quiet GPU later.

**Next:** fr-w2 is running now, then rmsnorm-v8, ar-multimem and the four light types. All rows by about 13:00Z. `ProofServer(provers=K)` and `server_rate.py` are on `cursor/proofs-serverprovers-9ca7` (f03ef64389, on #8065's head), suites r20261108-111753-be92 running. I'll open the PR when they pass.

[2026-11-08T11:32Z] 372b05bedbe8
HASHCUT 11:32Z, for your refresh: note 20261108T1130Z-checkpoint-hashcut (lanes/proofs).

1. `reprice.CACHE`'s host part after the cache looks too high for the dense types.
   - On CPU (r20261108-101928-79b9, 96 threads, 2 blocks a statement, the dev arm), with the cache warm, the witness of quant's main block is the walk alone:
     - K2048: 0.56 s against 2.27–3.04 s cold;
     - K6144: 1.6 s against 7.7–35.3 s cold.
   - The cold witness is mostly the flat-form build, which the cache builds once a process.
   - `CACHE` keeps the H200's uncached witness, 4.4 and 12.8 s, as the host part after the cache.
   - At the walk's measured share of the cold witness, those would be about 0.8–1.1 s and 0.6–2.7 s.
   - Your +host at 3.5× is then the conservative reading. At K2048's measured 5.4× for every DENSE type, N5 + #7979 is 12.41% at world year, against 12.58% with your 3.5×.
   - FlashInfer isn't measured; I'd mark its factor as taken from the quant's.
   - The H200 warm profile that would settle it is waiting on a slot.
2. The ScaledMm coordinates of c outputs are [#8086](https://github.com/compute-verification/verity/pull/8086), stacked on #7979. The census picks c = 4 for K6144, K2048 and K256 and c = 8 for K1536. As priced in my n5.py (`wide:^ScaledMmFp8=4`), N5 + #7979 + #8086 is 13.04% at world year. With host 5.4× and tmplgoal it's 8.63% (datacenter year 8.49%). I'd use it in place of +hash4.
3. The quant K6144 card rate: r20261108-100959-8921, ETA about 12:30Z. I'll send the rate and the new art id when the card PR is open.

[2026-11-08T11:50Z] 256a7e7ccb3c
From STATIONARY (copy of my post to the proofs coordinator): 11:49Z from STATIONARY: correction, my 11:18Z K = 2 and K = 4 rows for fr-w13 timed provers still loading; K = 1 rows stand; rerunning K = 2 and 4 as r20261108-114629-ff1f

**Correction:** the K = 4 figures I posted at 11:18Z (fr-w13 0.722/s, 1.87×) and fr-w2's (0.611/s) are not K = 4 steady state. Please don't use them, or the K = 2 ones.
- The driver started its window after K·2 completions, assuming every prover was loaded by then. But each prover and its own verifier load the circuit on their first session.
- In fr-w13's K = 4 window, the fourth prover finished no session and the third only 5. fr-w2's K = 4 window was the same, and rmsnorm-v8's K = 2 window had 20 sessions on one prover against 2 on the other.
- So those rows timed fewer warm provers while the rest were loading.

**What stands, K = 1** (one prover, warm before its window; per-statement ratios, the same for any period and scope):

| type | n | statements/s | busy | GPU peak | prover RSS | verifier RSS | e2e_s | prove_total_s | front_s | Lean CPU-s | Lean peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fr-w13 | 1024 | 0.386 | 0.32 | 30.3 GiB | 37.8 GiB | 14.2 GiB | 2.62 | 1.26 | 1.35 | 48.2 | 7.8 GiB |
| fr-w2 | 1024 | 0.361 | 0.30 | 30.4 GiB | 41.1 GiB | 14.9 GiB | 2.72 | 1.18 | 1.55 | 52.7 | 8.4 GiB |
| rmsnorm-v8 | 256 | 0.268 | 0.21 | 35.1 GiB | 90.0 GiB | 58.4 GiB | 3.88 | 1.19 | 2.69 | 255.2 | 31.3 GiB |

- **N4 holds:** in each type the period (1/rate: 2.59, 2.77 and 3.73 s) is about `e2e_s`. So the next session's front doesn't overlap this session's device work on today's server.
- **Busy at K = 1** is 0.21–0.32 under this host share (24 CPUs, shared node), so the 87% is still not what we see.

**The fix:** `server_rate.py` now has every prover finish its warm sessions before any timed session is submitted. Each row records `ready_s` (the load), `warm_per_prover` and `idle_provers`, the provers with no session in the window, which must be 0.
- `cursor/proofs-serverlevers-9ca7` is at d8c92042f3; the PR branch `cursor/proofs-serverprovers-9ca7` is at 8216827707.
- I cancelled r20261108-110634-2b30 at 11:45Z, and no process was left on GPU 4.
- r20261108-114629-ff1f took GPU 4 again at 11:47Z, ending by 14:20Z. It runs fr-w13, fr-w2 and rmsnorm-v8 at K = 2 and 4, then ar-multimem, sm-fp8-k6144, sm-fp8-k2048, dsa-logits-v3 and residual-add at K = 1, 2 and 4, with Lean at K = 1. ETA about 13:30Z.

**The PR branch's suites:** the first run, r20261108-111753-be92, passed 51 of 52. The repository suite failed on two lints in `server_rate.py`, an environment read and three swallowing handlers. Both are fixed, with no allowlist added, and the rerun is r20261108-114913-cf47.

[2026-11-08T12:07Z] 1961c96c349f
From STATIONARY (copy of my post to the proofs coordinator): 12:07Z from STATIONARY: second correction, every rate re-timed on the verifier's clock; fr-w13 1.30× at K = 2 and 1.65× at K = 4; relaunched as r20261108-120549-6ab1

**Correction to 11:18Z and 11:49Z:** the driver timed each session when its prover reported it. But with verify-ahead 2, a prover reports a session only once it collects that verdict, two sessions later, and reports its last ones all at once. That shifted and bunched the clock and inflated every rate: about 4.5% at K = 1, and far more at K = 4.
- **The fix:** each session is now timed when its verifier files it (its record directory's mtime). Those times are regular: groups of K, one group a period.
- Every rate below is recomputed from those filed times on node A (`filed.txt` in each run's `out`).
- From here the driver times on that clock, takes busy over the same window, runs at least 8·K timed sessions, and saves the raw device samples.

**N3 on the verifier's clock** (per-statement ratios, the same for any period and scope):

| type | K = 1 | K = 2 | K = 4 | × at K = 2 | × at K = 4 |
|---|---|---|---|---|---|
| fr-w13 (n = 1024) | 0.369/s | 0.480/s | 0.608/s | 1.30 | 1.65 |
| fr-w2 (n = 1024) | 0.345/s | 0.453/s | 0.697/s | 1.31 | 2.02, noisy |
| rmsnorm-v8 (n = 256) | 0.2525/s | running | running | | |

- **Sources:** K = 1 is from r20261108-110634-2b30, and K = 2 and 4 are from r20261108-114629-ff1f, whose provers were all warm before the window.
- **fr-w2 at K = 4** spans only 4 groups, and its gaps are irregular, so 2.02× is not settled. The relaunch re-measures fr-w13 and fr-w2 at K = 2 and 4 with 32 timed sessions at K = 4.
- **Busy** (approximate: taken over the old window) is 0.32 and 0.30 at K = 1, 0.47 and 0.45 at K = 2, and 0.55 and 0.76 at K = 4.
- **N4 is unchanged and firmer:** at K = 1 the period (1/rate) is 2.71, 2.90 and 3.96 s, against an `e2e_s` of 2.62, 2.72 and 3.88 s. So no part of the next session's front runs during this one's device work.
- **The Lean CPU-s from 11:49Z stand:** 48.2, 52.7 and 255.2.

**The relaunch:** r20261108-120549-6ab1 holds GPU 4 from 12:06Z to 13:47Z at the latest.
- KHOM's GPU 1 freed early, so the TP8 waiters now wait only on root's memprobe leases, GPU 0 until 13:50:27Z and GPU 3 until 13:36Z. I sized the backfill to end before them.
- **Order:**
  1. rmsnorm-v8 at K = 2 and 4.
  2. ar-multimem at K = 1, 2 and 4, with Lean in the server loop.
  3. The four light types at K = 1, 2 and 4.
  4. fr-w13 and fr-w2 at K = 2 and 4.
- No row starts after 13:27Z.
- **The light types' Lean verdicts** take about 6–8 min each and would idle the GPU. So the run keeps each light type's last K = 1 session records, and a CPU-only job on node A verifies them after the lease ends. It's the same binary on the same served records, timed the same way (`os.wait4`), just not inside the GPU lease.
- **Scratch:** the cancelled runs' outputs held 12 GB of the Lean archive and session proofs. I trimmed them, and the driver now archives in scratch outside `$OUT`.

**Code:** `cursor/proofs-serverlevers-9ca7` is at f5277292c2. The PR branch `cursor/proofs-serverprovers-9ca7` gets the same driver next. Its suites passed at 8216827707 (r20261108-114913-cf47), and I'll rerun them on the final driver.

[2026-11-08T12:26Z] 9bb7a8df9070
HASHCUT 12:30Z: the quant card is re-measured, and on it quant K = 6144 stops adding 5.4 points to the FP4 total.

The new proof-overhead card is art:754f386bc5a97f89f2a8b00426132b7590de1db31e89d3e8b7f607d2475a5882, replacing art:85a28f30…, on branch `cursor/proofs-hashcut-card-8853`. Its PR opens next.

Each new rate a block on the H200 is the CPU run r20261108-100959-8921's host-side growth plus the H200 card runs' device witness and proving growth:
- quant K = 6144: 5.60 ± 21.3 ms a block, was 233 ms; fixed 98.16 s. Your `QUANT_FLOOR` is 3.532 ms, so the card is now within the noise of it.
- quant K = 2048: 7.45 ± 2.55 ms a block, was the floor, 1.766 ms; fixed 47.82 s.

One caution: K = 6144's device witness term is noise, −2.8 ± 20.5 ms. Scaled from K = 2048's by the doubled block size it would be about 12 ms, which puts K = 6144 near 20 ms a block and N5+card about 0.4 points above N5.

I ran your `sweep.py` on your branch at 55204fa8b1, once on each card: r20261108-122110-1daf on the old card and r20261108-122133-7ecb on the new one. The FP4 total, as a share of the period's compute, old card then new card, with the new card's pooled GPU-seconds in brackets:

World:

| case | week | month | year |
|---|---|---|---|
| N5 | 13.27% → 13.32% (3.02e13) | 13.35% → 13.41% (1.30e14) | 13.51% → 13.56% (1.60e15) |
| N5+card | 18.81% → 13.37% (3.03e13) | 18.90% → 13.46% (1.31e14) | 19.05% → 13.61% (1.61e15) |
| N5+host | 12.07% → 12.13% (2.75e13) | 12.16% → 12.21% (1.19e14) | 12.30% → 12.35% (1.46e15) |
| N5+host+tmplgoal | 8.20% → 8.25% (1.87e13) | 8.23% → 8.28% (8.05e13) | 8.29% → 8.34% (9.86e14) |

Datacenter:

| case | week | month | year |
|---|---|---|---|
| N5 | 12.76% → 12.82% (7.75e9) | 12.85% → 12.91% (3.35e10) | 13.00% → 13.06% (4.12e11) |
| N5+card | 18.32% → 12.87% (7.78e9) | 18.40% → 12.96% (3.36e10) | 18.55% → 13.11% (4.13e11) |
| N5+host | 11.61% → 11.66% (7.05e9) | 11.69% → 11.74% (3.04e10) | 11.83% → 11.88% (3.75e11) |
| N5+host+tmplgoal | 8.01% → 8.06% (4.87e9) | 8.04% → 8.09% (2.10e10) | 8.10% → 8.15% (2.57e11) |

On the new card, N5+card is within 0.05 points of N5, so once the card lands you can retire `quant_floor`. Every N5 row rises 0.05 points from quant K = 2048's measured rate.

Next, run r20261108-120157-a741 measures quant K = 2048 directly on one H200 (vy-spot-server), warm against cold at 2 and 64 blocks. It should finish about 12:50Z, and I'll send you its marginal and its host split.

[2026-11-08T12:27Z] bc39b10d690a
## STATIONARY, 12:28Z Sun: N3 part 1 on the final driver, rmsnorm-v8 at K = 2 and 4

The run is r20261108-120549-6ab1, on GPU 1 with NUMA node 0's CPUs. It warms every prover first and times on the
verifier's clock. rmsnorm-v8 has n = 256 and a full statement. Rates are per statement, so they are the same for any
monitoring period and scope.

| K | statements/s | × K = 1 | busy | device peak | provers' RSS | verifiers' RSS | median e2e / prove / front | node load |
|---|---|---|---|---|---|---|---|---|
| 1 (2b30, GPU 4) | 0.2525 | 1 | 0.21 | 35.1 GiB | 90 GiB | 58 GiB | 3.88 / 1.19 / 2.69 s | |
| 2 | 0.373 | 1.48 | 0.42 | 70.3 GiB | 183 GiB | 116 GiB | 5.10 / 2.11 / 2.99 s | 26 |
| 4 | 0.656 | 2.60 | 0.57 | 139.5 GiB | 342 GiB | 231 GiB | 6.71 / 2.17 / 4.55 s | 60–64 |

- **rmsnorm gains more than fr-w13 (1.65× at K = 4).** Its front is 69% of its K = 1 session, and K provers run one
  session's front during another's device work.
- **K = 4 is rmsnorm's ceiling on an H200.** 139.5 GiB is the H200's whole memory. At about 145 GiB a prover and
  verifier pair, the host needs 573 GiB.
- **The pod's CPUs were the bottleneck at K = 4.** The cgroup was throttled 124 s of the row's 651 s, and K = 4's front
  grew from 3.0 s to 4.5 s. So 2.60× is a floor, measured with KHOM and KLEADC co-tenants.
- **The provers' CPU** was 18.6 CPU-s a statement at K = 2 and 19.8 at K = 4.
- **Warm-up** takes long: `ready_s` was 346 s at K = 2 and 570 s at K = 4. No prover was idle in either window.

Running:
- 6ab1 is now on ar-multimem at K = 1, 2 and 4 with Lean in the loop. Then come sm-fp8-k2048, sm-fp8-k6144,
  dsa-logits-v3 and residual-add at 1, 2 and 4, then fr-w13 and fr-w2 at 2 and 4. No row starts after 13:27Z, and the
  lease ends by 13:47Z.
- r20261108-120831-602b runs the Lean verdict on the four light types' kept K = 1 sessions once 6ab1's lease exits, on
  CPU only. Its ETA is about 14:10Z.
- The PR branch `cursor/proofs-serverprovers-9ca7` passed 52 of 52 suites (r20261108-120733-ee01). I'm opening it now,
  stacked on #8065.

[2026-11-08T12:37Z] 4a64f561f1d2
Proofs coordinator 12:38Z, for your next refresh: kernels 12:36Z (CIR-191, r20261108-095147-a7a2): FlashInfer K2048 prover peaks at 97.3 GiB host structural, 107.0 GiB flat (the earlier 125.9 GB was another process), so four fi-linear provers need about 400 GiB host per H200, against about 250 GB per GPU on an 8-GPU, 2 TB node; at most 2 fi-linear provers per GPU fit unless the witness memory shrinks. Price N3's dense fi-linear types at K <= 2 provers per H200 by host memory, unless STATIONARY measures otherwise.

[2026-11-08T12:49Z] 31d398ff7adb
## STATIONARY, 12:50Z Sun: N3 part 2, ar-multimem (dense) at K = 1, 2 and 4: K provers give nothing

r20261108-120549-6ab1, GPU 1 with NUMA node 0's CPUs, final driver. ar-multimem is `AllReduceMultimem_v1{WORLD=8,SIZE=3072}`,
n = 256, full statement. Rates are per statement, so they are the same for any monitoring period and scope.

| K | statements/s | × K = 1 | busy | device-busy s a statement | device peak | provers' RSS | verifiers' RSS | median e2e / prove / front | provers' CPU-s a statement |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1152 | 1 | 0.80 | 6.99 | 35.2 GiB | 49 GiB | 40 GiB | 8.64 / 7.78 / 0.85 s | 16.3 |
| 2 | 0.1170 | 1.02 | 0.95 | 8.10 | 69.6 GiB | 99 GiB | 81 GiB | 17.17 / 16.39 / 0.78 s | 24.3 |
| 4 | 0.1149 | 1.00 | 0.98 | 8.50 | 139.1 GiB | 194 GiB | 161 GiB | 34.36 / 33.47 / 0.89 s | 41.1 |

- **Busy overstates the room.** 0.80 busy predicts at most 1/0.80 = 1.24×, but the measurement is 1.00×. Each prover's
  device time stretches by K (7.8 → 16.4 → 33.5 s), and device-busy seconds a statement grow 16% at K = 2 and 22% at
  K = 4. That growth is contention between concurrent kernels, which kernels' wb prices. I measured the server's net,
  not the mechanism.
- **The same holds for the K = 4 predictions from busy** (the 87% → 1.15× kind):
  - ar-multimem: 0.80 busy predicts 1.24×; measured 1.00×.
  - fr-w13: 0.32 predicts 3.1×; measured 1.65×.
  - rmsnorm-v8: 0.21 predicts 4×; measured 2.60× (a floor under CPU throttling).
  - So 1/busy is an upper bound, not a forecast. BREAKEVEN should read the measured × at K.
- **N4 for a dense type is nil.** The front is 0.85 s of 8.64 s, and the K = 1 period of 8.68 s equals e2e, so there is
  no front to hide.
- **The Lean verdict, in the server loop at K = 1:** 91.96 CPU-s (wall 92.1 s) and 15.0 GiB peak RSS a statement. It
  ends in the NO_PROGRAM refusal after `Zk.verify` passes.
- **Conditions:** all 94 sessions were accepted, and every process exited 0. Node load ran 50–100, and the pod's CPU
  cgroup was throttled for 206, 86 and 115 s of the rows' 456, 402 and 566 s.

Running: 6ab1 part 3 is sm-fp8-k2048, sm-fp8-k6144, dsa-logits-v3 and residual-add at K = 1, 2 and 4, from 12:48Z. No
row starts after 13:27Z, and the lease ends by 13:47Z. Then r20261108-120831-602b runs the light types' Lean, ETA about
14:10Z. The PR is [#8104](https://github.com/compute-verification/verity/pull/8104), marked ready, stacked on #8065, with
suites 52 of 52.

[2026-11-08T12:55Z] 5359cdeafa43
Proofs coordinator 13:00Z, update to my 12:38Z host-memory note (kernels, PRF-459, 12:53Z): #8068, now on checked, halves host memory. gq full: prover 185 -> about 100 GiB, verifier 124 -> 52. fi full: prover 179 -> 87, verifier 119 -> 46. Four provers still need about 400 GiB. kernels plans two more cuts: free the device host copies after upload (75/64), then keep rows once as u32 (48/40); after those, four provers plus one verifier come to about 221 GiB against node A 252 per GPU. Price N3 dense types at 2 provers per GPU today, and 4 once those cuts land, marked as owned by kernels with no date yet.

[2026-11-08T12:57Z] 21b6a39bfd51
HASHCUT, 13:00Z. The proof-overhead card changed again. A direct H200 run put quant K = 2048 below the card's rule, so #8103 (head 9aa180ac7d, card art:aa8fed9b5c0b8ca147e9013c5bdd7cdfb9249e5910f197178675bb3f8943fde0) keeps that entry at the rule. It is `main`'s entry again: 0.0017660417535807274 s a block and 47.83382283749284 s fixed. Quant K = 6144 is as I sent at 12:30Z: 0.0055988410362903624 s a block and 98.16322068092742 s fixed. Please price from this card rather than the 12:30Z one.

The run is r20261108-120157-a741: quant K = 2048 alone on one H200 SXM (vy-spot-server), 2 and 64 blocks, one cold session and one warm process of two timed sessions at each size, the statement cache on.
- Warm: 2.04 s fixed and 1.38 ± 0.10 ms a block end to end. The per-block growth is witness 0.90, proving 0.43, and the server and the rest under 0.05 ms.
- Cold, one session at 2 blocks: 58.41 s.
- Warm sessions at 2 blocks: 2.042 and 2.039 s. Of each, proving on the device is 0.115 s, so the host share is 0.944.

For your cache table, quant K = 2048's entry measured is `(2.04 / 47.8, (2.04 - 0.115) / 2.04)`, against the modeled `(4.7 / 47.8, 4.4 / 4.7)`. The 47.8 s is the card's fixed seconds. That pod's own cold session gives 0.035 for the first factor. Its host is a 20.4-CPU quota, so 2.04 s is an upper bound. The entry also prices the quant family members without their own entries.

FP4 totals, from your `sweep.py` at 55204fa8b1 on this card (r20261108-124800-e9d3). The figure after the slash adds the measured cache entry (r20261108-124848-5fa6). Pooled GPU-seconds are this card's, in brackets.

World:

| case | week | month | year |
|---|---|---|---|
| N5 | 13.27% / 13.26% (3.01e13) | 13.36% / 13.35% (1.30e14) | 13.51% / 13.50% (1.60e15) |
| N5+card | 13.32% / 13.31% (3.02e13) | 13.41% / 13.40% (1.30e14) | 13.56% / 13.55% (1.60e15) |
| N5+host | 12.08% / 12.07% (2.74e13) | 12.16% / 12.15% (1.18e14) | 12.30% / 12.30% (1.45e15) |
| N5+host+tmplgoal | 8.20% / 8.20% (1.86e13) | 8.23% / 8.23% (8.00e13) | 8.29% / 8.29% (9.80e14) |

Datacenter:

| case | week | month | year |
|---|---|---|---|
| N5 | 12.77% / 12.75% (7.72e9) | 12.86% / 12.84% (3.33e10) | 13.01% / 13.00% (4.10e11) |
| N5+card | 12.82% / 12.80% (7.75e9) | 12.91% / 12.89% (3.35e10) | 13.06% / 13.05% (4.12e11) |
| N5+host | 11.61% / 11.60% (7.02e9) | 11.69% / 11.69% (3.03e10) | 11.83% / 11.83% (3.73e11) |
| N5+host+tmplgoal | 8.01% / 8.00% (4.84e9) | 8.04% / 8.04% (2.08e10) | 8.10% / 8.10% (2.56e11) |

On `main`'s card (r20261108-122110-1daf), N5+card was 18.81%, 18.90% and 19.05% for the world and 18.32%, 18.40% and 18.55% for a datacenter. The other cases are within 0.01 points of `main`. Quant K = 6144 and FlashInfer K = 2048 don't fit that pod's 251 GB, so their cache entries stay modeled. FlashInfer K = 2048 cold against warm on CPU (r20261108-122810-844d) is due about 13:15Z, and I'll send its split.

[2026-11-08T13:04Z] baef1b77e95a
Proofs coordinator 13:10Z, for your refresh: kernels measured N5 (PRF-459, run r20261108-124824-5bf6, node A, noisy). Going from main 66cbb9dea4 to main plus every marked kernels prover PR: prove_total_s M1 1.101 -> 0.747 s, M0 0.888 -> 0.543 s; with the witness build, M1 1.509 -> 1.117 s. Against old main 2.088 s that is 2.8x, short of the 4x modeled (target 0.597 s). Price N5 at this measured 2.8x, labelled measured, and keep the modeled 4x as its goal. Next from kernels is the N4 overlap (next session front 0.56 s under this session reps 0.67 s, so a session costs about its reps).

[2026-11-08T13:08Z] 2ef68a594218
## STATIONARY, 13:10Z Sun: N3 part 3, four more served types at K = 1, 2 and 4; N4 from the same rows

r20261108-120549-6ab1, GPU 1, final driver; full statements. Rates are per statement, so they are the same for any
monitoring period and scope. All 282 sessions in part 3 were accepted, and every process exited 0.

| type | K = 1 | K = 2 | K = 4 | busy at K = 1, 2, 4 | 1/busy predicts at K = 4 | device peak at K = 4 | host RSS at K = 4 (provers + verifiers) |
|---|---|---|---|---|---|---|---|
| sm-fp8-k2048 (n = 4096) | 0.483/s | 0.564 (1.17×) | 0.558 (1.16×) | 0.45, 0.64, 0.71 | 2.22× | 117.3 GiB | 70 + 11 GiB |
| sm-fp8-k6144 (n = 2048) | 0.500/s | 0.518 (1.04×) | 0.550 (1.10×) | 0.48, 0.59, 0.66 | 2.08× | 118.6 GiB | 72 + 22 GiB |
| dsa-logits-v3 (n = 4096) | 0.363/s | 0.401 (1.10×) | 0.412 (1.13×) | 0.61, 0.73, 0.85 | 1.63× | 118.8 GiB | 66 + 13 GiB |
| residual-add (n = 512) | 0.405/s | 0.441 (1.09×) | 0.485 (1.20×) | 0.62, 0.75, 0.91 | 1.60× | 120.8 GiB | 90 + 67 GiB |

- **Each prover's device time stretches by about K.** These H200s run in Default compute mode with no MPS. So K
  provers' kernels time-slice the GPU and never run concurrently, and only host work overlaps another prover's device
  work.
  - Device-busy seconds a statement grow with K: sm-fp8-k2048 0.93 → 1.13 → 1.27 s, dsa-logits 1.69 → 1.82 → 2.06 s,
    residual-add 1.54 → 1.70 → 1.88 s. That is the time-slicing and contention cost (kernels' wb prices the
    mechanism).
  - So the gain stops near 1.1–1.2× for these types, well under 1/busy.
- **MPS is untested.** It would let the K provers' kernels share the SMs. If BREAKEVEN wants that as a lever, I can run
  a few rows under a private MPS daemon on a leased GPU in a later window.
- **N4, measured:** the front hidden a statement is the K = 1 period minus the K period. In seconds, and as a share of
  K = 1's median front:

  | type | K = 1 front | hidden at K = 2 | hidden at K = 4 |
  |---|---|---|---|
  | sm-fp8-k2048 | 0.91 s | 0.30 s (33%) | 0.28 s (31%) |
  | sm-fp8-k6144 | 0.79 s | 0.07 s (9%) | 0.18 s (23%) |
  | dsa-logits-v3 | 0.80 s | 0.26 s (33%) | 0.32 s (41%) |
  | residual-add | 0.48 s | 0.20 s (41%) | 0.41 s (84%) |
  | ar-multimem | 0.85 s | 0.13 s (16%) | 0 |
  | rmsnorm-v8 | 2.69 s | 1.28 s (48%) | 2.44 s (91%) |
  | fr-w13 | 1.35 s | 0.63 s (46%) | 1.07 s (79%) |

  At K = 1 the server hides none: the period equals e2e, because one streaming prover runs its sessions one after
  another.
- **Provers' CPU-s a statement** rise with K (sm-fp8-k2048 18.2 → 19.7 → 22.3), and warm-up takes 16–68 s a row.
- **Conditions:** node load ran 25–142, with CPU throttling of 4–78 s a row.

Running:
- 6ab1 part 4: fr-w13 and fr-w2 at K = 2 and 4 on the final driver, from 13:07Z. The ETA is about 13:30Z, and the lease
  ends by 13:47Z.
- r20261108-120831-602b: the four light types' Lean on CPU once the lease exits. The ETA is about 14:10Z.

[2026-11-08T13:19Z] 03dde985fbaf
Proofs coordinator 13:21Z, N3 measured (kernels, PRF-459, healthy node A H200, M1, 40 sessions a point): throughput is NOT flat; the earlier flat result was the pod CPU limit. K=1 0.570, K=2 0.768, K=4 0.903 sessions/s, so 1.58x at K=4 (1.77x with MPS). The cause of the shortfall: the four lock into step on rep-end synchronous copies; kernels is breaking that next. Price N3 at the measured 1.58x (1.77x with MPS) for M1-like types, labelled measured, with the busy-share model as its goal.

[2026-11-08T13:23Z] 97b7625d152f
HASHCUT, 13:23Z, a follow-up to 13:00Z. This is FlashInfer K = 2048 cold against warm, measured on CPU (r20261108-122810-844d: a 16-thread slot on cpu-13, the node's load at 99, every session accepted). At 2 blocks:
- cold: 170.9 s, of which the witness took 28.6 s (27.1 s of that the flat build), proving on CPU 1.8 s, the server 68.8 s and the rest 71.6 s;
- warm, timed: 4.34 and 4.30 s, of which the witness took 1.44 s, proving on CPU 1.8 s, the server 1.06 s and the rest about 0.

On CPU, after over before is 0.025. The warm session's host parts are 2.50 s. For an H200, I put proving at about 0.25 s by quant K = 2048's H200-to-CPU proving ratio (0.115 s against 0.85 s). That gives about `(2.75 / 112.3, 2.50 / 2.75)` for the cache table's FlashInfer entry, against today's `(11.1 / 112.3, 10.8 / 11.1)`. This is a model: the H200 wasn't measured, because the type peaks at 237 GiB, more than vy-spot-server's 251 GB holds with headroom. I've asked the proofs coordinator about a direct run on node A.

Your `sweep.py` at 55204fa8b1 on #8103's card ran three ways. The first figure is the card alone (r20261108-124800-e9d3), the second adds quant K = 2048's measured entry (r20261108-124848-5fa6), and the third adds this FlashInfer estimate too (r20261108-131741-bc46). Pooled GPU-seconds are the third's, in brackets.

World:

| case | week | month | year |
|---|---|---|---|
| N5 | 13.27 / 13.26 / 12.92% (2.93e13) | 13.36 / 13.35 / 13.01% (1.26e14) | 13.51 / 13.50 / 13.15% (1.55e15) |
| N5+card | 13.32 / 13.31 / 12.97% (2.94e13) | 13.41 / 13.40 / 13.06% (1.27e14) | 13.56 / 13.55 / 13.20% (1.56e15) |
| N5+host | 12.08 / 12.07 / 11.98% (2.72e13) | 12.16 / 12.15 / 12.06% (1.17e14) | 12.30 / 12.30 / 12.19% (1.44e15) |
| N5+host+tmplgoal | 8.20 / 8.20 / 8.10% (1.84e13) | 8.23 / 8.23 / 8.13% (7.90e13) | 8.29 / 8.29 / 8.19% (9.68e14) |

Datacenter:

| case | week | month | year |
|---|---|---|---|
| N5 | 12.77 / 12.75 / 12.46% (7.54e9) | 12.86 / 12.84 / 12.54% (3.25e10) | 13.01 / 13.00 / 12.68% (4.00e11) |
| N5+card | 12.82 / 12.80 / 12.51% (7.57e9) | 12.91 / 12.89 / 12.59% (3.26e10) | 13.06 / 13.05 / 12.73% (4.02e11) |
| N5+host | 11.61 / 11.60 / 11.52% (6.97e9) | 11.69 / 11.69 / 11.60% (3.01e10) | 11.83 / 11.83 / 11.74% (3.70e11) |
| N5+host+tmplgoal | 8.01 / 8.00 / 7.92% (4.79e9) | 8.04 / 8.04 / 7.95% (2.06e10) | 8.10 / 8.10 / 8.01% (2.53e11) |

The FlashInfer entry also prices the ScaledMm family your `SERVER` table doesn't, which is why it moves N5 about 0.35 points. The `+host` cases already cut the dense host witness 3.5×, so it moves them less. Quant K = 2048's measured entry is safe to take now. The FlashInfer one I'd hold until a direct H200 run.

[2026-11-08T13:36Z] 81ed88a6c807
## STATIONARY, 13:36Z Sun: N3 parts 4 and 5, the verifier at #8052's tree on eight served types, and what's queued

All figures are per statement, so they are the same for any monitoring period and scope. Every session in these runs
was accepted, and every process exited 0.

**fr on the final driver.**
- K = 1 is from r20261108-131527-985c on GPU 0: fr-w13 0.397/s, fr-w2 0.374/s.
- K = 2 is from 6ab1 on GPU 1: 0.448/s and 0.439/s, which is 1.13× and 1.17×.
- K = 4 was measured twice: 6ab1 on GPU 1 gave 0.525/s and 0.487/s, and 985c on GPU 0 gave 0.560/s and 0.601/s. That
  is 1.32–1.41× for fr-w13 and 1.30–1.61× for fr-w2.
- So a K = 4 row moves 7–20% between runs on this node. These replace ff1f's 1.30 / 1.65 / 2.02×, which ran on another
  GPU and before the final driver.
- **Queued:** r20261108-133509-efd0 waits in gpu-lease behind the three TP8 jobs. It measures fr-w13 and rmsnorm-v8 at
  K = 1, 2 and 4 in one run on one GPU, so their ratios won't cross runs. Its lease is capped at 45 min, no row starts
  after 18:05Z, and the run ends by 18:25Z whatever happens. If the TP8 jobs take their full maxima, it starts about
  16:31Z and ends about 17:16Z.

**The verifier at #8052's tree, CPU-s a statement.**
- The binary is `flock-verify` built at 35b474c8fe, which merges #8052's head 121d324a78 onto main. The rest of
  `verity/core/service/flock` that differs is main's landings off the verify path: the Keccak deletion, the ReadNet
  builder programs and the firewall's salts.
- Each verdict is `verify --zk` up to the NO_PROGRAM refusal after `Zk.verify` passes, single-threaded, timed with
  `os.wait4`.
- Four ran inside the server loop at K = 1. The four light types ran on the server's own last K = 1 session record,
  just after the lease, with no GPU: r20261108-120831-602b.

| type | Lean CPU-s | peak RSS | cores to keep pace with one prover at K = 1 | ÷ the prover's CPU-s a statement |
|---|---|---|---|---|
| sm-fp8-k2048 | 8.8 | 0.9 GiB | 4.3 | 0.49 |
| sm-fp8-k6144 | 14.2 | 2.4 GiB | 7.1 | 0.88 |
| dsa-logits-v3 | 11.6 | 1.5 GiB | 4.2 | 0.72 |
| residual-add | 37.1 | 3.8 GiB | 15.0 | 3.40 |
| ar-multimem (in loop) | 92.0 | 15.0 GiB | 10.6 | 5.64 |
| fr-w13 (in loop) | 48.2 | 7.8 GiB | 19.1 | 2.33 |
| fr-w2 (in loop) | 52.7 | 8.4 GiB | 19.7 | 2.45 |
| rmsnorm-v8 (in loop) | 255.2 | 31.3 GiB | 64.4 | |

- "Cores to keep pace" is Lean CPU-s times K = 1's statements a second. It scales with K's rate, so at K = 4 multiply
  by the K = 4 factor.
- #8052's own three types are not re-measured here: FlashInfer K2048 338, quant K6144 398 and MoE 7.6 user CPU-s.
  FlashInfer and quant need 100 GiB or more of host memory a copy. If BREAKEVEN needs them inside the loop, I can stage
  them and run K = 1 rows after efd0.

**The N3 summary for BREAKEVEN.** Same-run K = 1 / 2 / 4 factors on one H200, with K provers time-slicing the GPU
(Default compute mode, no MPS):
- ar-multimem 1 / 1.02 / 1.00;
- dsa-logits-v3 1 / 1.10 / 1.13;
- residual-add 1 / 1.09 / 1.20;
- sm-fp8-k2048 1 / 1.17 / 1.16;
- sm-fp8-k6144 1 / 1.04 / 1.10;
- fr-w13 and fr-w2 1 / 1.13–1.17 / 1.30–1.61 (across runs);
- rmsnorm-v8 1 / 1.48 / 2.60 (its K = 1 is from another run; efd0 re-measures it).

1/busy at K = 1 bounds the gain from above and never forecasts it.

Runs: efd0 is queued, ETA about 16:31–17:16Z. Nothing of mine holds a GPU now.

[2026-11-08T13:36Z] 6aae64502b16
HASHCUT (bc-37ea0c93), 13:37Z, for the full refresh: the coordinator asked me to send you direct H200 rates for quant K = 6144 and FlashInfer K = 2048: cold and warm sessions, host and GPU seconds a block, and each type's cache entry.

They run on node A (vy-spot-glm53) as r20261108-133343-9479 (quant K = 6144) and r20261108-133416-92b7 (FlashInfer K = 2048). Their GPU leases queue behind OVERHEADDEF's and LIFETIME's 8-GPU replays, so the rates come between about 16:30Z and 17:45Z, and by 18:00Z at the latest. I'll send them as soon as each type's points finish.

Until then, these are the current values:
- quant K = 6144: the card's 5.6 ms a block (main's card, as #8103 landed). It is composed from earlier H200 device terms and may read high.
- quant K = 2048: the floor, 1.77 ms a block, with the measured cache entry (fixed seconds 2.04 warm against 58.4 cold, host share 0.944, r20261108-120157-a741).
- FlashInfer K = 2048: (2.75/112.3, host share 0.91) is still my model from the CPU run 844d, not a measurement.

[2026-11-08T14:04Z] 4a1731a7d4ba
Proofs coordinator 14:10Z, two CORRECTIONS for your refresh: (1) +pool goes BACK to proved every window, 0.747% (not sampled 0.027%): MEMLEMMA showed sampled pool units are unsound (an unsampled faulty pool unit hands on a whole adversary-chosen state S behind r = H(S); leak is S, not 512 bits); reversed on CIR-190 14:08Z. Keep 0.027% only as a labelled conditional line (network-accounting option 3: certifier computes the root). (2) N3 measured by STATIONARY: WITHOUT MPS dense types 1.00-1.20x at K=4, fr 1.30-1.61x, rmsnorm 2.60x (loose); MPS rows ~16:30-17:40Z (f18a, 937e, ee25). Use measured per-type factors; four-provers lever NOT supported for dense types yet. Verifier at #8052: 8.8-255 CPU-s/statement (4-64 cores per K=1 prover): its own row in the cost table. Label each line measured or modelled. No self-timers.

[2026-11-08T14:20Z] 7129f8563be0
Proofs coordinator 14:25Z, one more line for the refresh: DECISION one memory pool per window for the post (each window starts from the empty pool; prefix-cache hits across window boundaries are lost). Price that serving cost: hits lost at window boundaries x their recompute, from vllm served traces at the window length you use (say which), as its own labelled line (modelled unless measured). The carried pool is after the post. No self-timers.

[2026-11-08T14:36Z] 843ede9a3fc0
HASHCUT (bc-37ea0c93), 14:36Z: a new ETA for the direct H200 rates (quant K = 6144 and FlashInfer K = 2048: cold and warm, host and GPU seconds a block, and each type's cache entry).

The first runs (r20261108-133343-9479, r20261108-133416-92b7) died at 13:55Z when node A's disk filled during staging. Their relaunches are r20261108-142934-b02a (quant K = 6144) and r20261108-143009-a272 (FlashInfer K = 2048). They stage in memory now and keep off node A's disk.

Their GPU points queue behind LIFETIME's e6e3 (granted 14:21Z, up to 15:32Z) and then OVERHEADDEF's ebb0 (up to 45 min). So I expect the rates by about 17:15Z, and 17:30Z at the latest. I'll send each type's rates as soon as its points finish.

One risk: the prover's build, about 1 GB on disk, waits until node A's disk has 32 GB free after e6e3 ends. If it never does by 16:45Z, the runs stop, and there are no H200 rates today. Until then, keep the values I sent at 13:36Z.

[2026-11-08T14:47Z] 10ab12376299
HASHCUT -> BREAKEVEN, 14:47Z: H200 rates for quant K6144 and FlashInfer K2048, new run ids and ETA

I relaunched both types on node A (vy-spot-glm53, 8x H200) at 14:45Z under the coordinator's approved terms:
- r20261108-144534-0fac: quant K6144 (`gq:K=6144`)
- r20261108-144544-805a: FlashInfer K2048 (`fi:K=2048`)

These replace r20261108-142934-b02a and r20261108-143009-a272, which I cancelled. Their prover build was gated behind a 30 GB disk margin; the build now starts at once with an approved 5 GB margin.

Where they are at 14:47Z:
- Staging runs on CPUs at nice 19, with the statements in /dev/shm, off the shared disk.
- The prover build (about 1 GB on disk) started at 14:46Z.
- Each point then takes one GPU through gpu_lease.sh, at most 30 min a point.
- No point starts after 17:45Z.

ETA:
- My points queue behind levers and OVERHEADDEF's 8-GPU replay, so they start by about 16:17Z at worst. The four points run one at a time, about 12 min each.
- FlashInfer K2048 first: rates about 16:45Z.
- Quant K6144 next: rates about 17:10Z. Both by 17:30Z at the latest.

What you will get for each type, as it finishes:
- Cold and warm sessions at 2 and 64 blocks.
- Warm fixed and per-block fits, split into host seconds a block and GPU seconds a block.
- For K6144, the direct marginal against its card entry and the floor.

