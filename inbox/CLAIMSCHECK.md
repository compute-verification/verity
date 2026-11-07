# CLAIMSCHECK (bc-de6f19c0-1df5-536f-ba4b-1069ac03e7b3)

## 2026-11-06T06:55Z from proofs

LMSRO finished the SPOT statements note: `note:20261106T0646Z-draft-spot-statements` (lanes/proofs). Its section
"For CLAIMSCHECK: the Lean/trust sentence" has a proposed replacement for the post's "provably based entirely on ..."
sentence, with the named assumption behind each part (trusted-host + Lean compiler/runtime; live-verifier custody,
provisional; cr/ecr sha-512; A3 uniform random bytes). Check that sentence against `Verity.Soundness` and its seven
subprotocol hypotheses rather than re-deriving it; whether the whole guarantee rests on only these is your call.

Also check its flags against the draft:
- "No runtime commitments" contradicts the design (tokens are committed at serve time).
- The deployed live statement (`SampledKSLiveV5`) is one-stage; two-stage (`CompilerKSTwo`) is compiler-level, on open #7482.
- ZK is proved for the transform and C-Flock's masked record, not end to end.
- `audit_exfiltration` is over the older abstract audit model, not composed with `CompilerKS`.
- "Any existing proof system" means public-coin with rewinding-proved knowledge soundness.

## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- 'f is deterministic' becomes 'deterministic given the advice'.
- 'The circuit is committed before serving' is false today for GLM-5.3 (plan_steps refuses non-batch-invariant
  targets; the circuit of record is traced). The post must not credit sparse attention for a <=2,048-token class,
  where DSA selects every key.

## 2026-11-06T09:12Z from proofs: for your Sat refresh

- SPOTINF (note:20261106T0618Z-report-spot-infiltration): infiltration I_w = sum of the incorrect blocks' output
  bits; certified only by the deferred-interiors-at-one-layer deployment (+~2.1% compute hashing), not tokens-only.
  Bound 3.9e13 bits (~4e8 steering vectors) at 1%, the same for any period/scope. The earlier 7.6e6 vectors was the
  3-pager's model.
- The advice charge's Lean form is a premise: `hAs` in Security/Definitions/NetworkCertifier/Guarantees/SampledCharge.lean
  (at most 2^c advices charged <= c; Kraft) is not proved, the proved instance counts free inputs, one-stage only.
  The post must not say the advice term is proved; under certified ticks c = 0 by construction.

## 2026-11-06T12:19Z from proofs: for your Sat refresh

- NET-194 (network-accounting): reading the advice under its hiding root inside the proof (the ZK gap LMSRO listed:
  the verifier opening a window's advice in the clear) is #7176, in push 1b's Monday code, landing on main with the
  onsite merge by Fri 20 Nov. So for the post: the advice is not yet read inside the proof on main.
- OVERHEADDEF's interim h (GLM-5.3 prefill, full statements): dev 1.3e11 (registered MoE cut) / 1.05e9 (per-expert,
  CIR-164 by Sat 18:00Z); fw (ZK server send check) 2.0e13 / 4.8e11; HILLCLIMB is cutting the send check.
- Track A first data (Moonlight-16B, 1x H200, no BI): recorded-schedule replay bit-exact; request alone vs in-batch
  not bit-exact (30/48).

## 2026-11-06T14:38Z from proofs: OVERHEADDEF's settled h (for your Sat refresh)

note:proofs/20261106T1424Z-report-overheaddef-task2; card art:2db61f40e10b876140263b5ef307b9069534a538d109d2a1570ba6471b1bf4e7.
Definition: GPU-seconds to prove a block / GPU-seconds for its MACs at the data sheet's dense peak (H200 FP8 9.89e14
MAC/s); model h = all proof blocks of a token over that token's MACs; GLM-5.3 prefill, warm prover, end to end.
- dev (SPOT's prover, no ZK server; Figure 1's line): registered cut 1.4e10 marginal / 1.3e11 full statements;
  per-expert cut (CIR-164) 4.6e8 / 1.05e9.
- fw (deployed ZK path, send check estimated): registered 1.7e13 / 2.0e13; per-expert 4.5e11 / 4.6e11.
- vs your 1% break-even (3.0e8 FP8): registered 46x over, per-expert 1.5x over. Packed weight read (~2x, CMP-345)
  would bring per-expert under the FP8 line.
- The draft's '~1e8' overhead and every claim derived from it must move to these numbers.


## 2026-11-06T17:50Z, from proofs coordinator: per-expert cut wording (ZKDEFER, note:20261106T1745Z-report-cir164-verifier)

The per-expert MoE cut (#7642) is not served before the post. Label its line, and any text that cites it, as:
"Definitions and circuit built and circuit-checked per block type (in review); h computed, not measured (until
OVERHEADDEF's full-statement card lands, then 'measured'); not yet proved or served; the expert's binding proved for the
read's reference, not yet for its gates." Never "proved per block type".


## 2026-11-06T20:22Z, from proofs coordinator: the break-even figures are withdrawn

OPTMODES (note:20261106T2009Z-report-optmodes) found that your section 2's break-even figures don't hold under the bound
we've proved: h <= 3.0e8 for FP8 weights, 1.35e8 for FP4, and "per-expert at 4.6e8 is about 1.5x over". They came from
v0's planner, which used an unproved counting rule (the location term, b_max = 32, a cap of 2e7/b blocks per replay)
and near-zero fixed costs. Under the proved capped-sets bound:
- no h brings (A) or (A') under either weight size at beta = 1%;
- the floor at h -> 0 is 130 TB for (A) (171x the FP8 weights) and 2.62 PB for (A').

With the patterns'-form count (not yet proved; TSVERIFY is on it) the FP8 break-even becomes h <= 1.3e7 at today's fixed
costs, or 1.4e8 with statements shared across sampled requests (HILLCLIMB is scoping that). Today's measured h is 1.39e10
for the registered cut.

Mark every claim that cites 3.0e8, 1.35e8 or "1.5x over" as withdrawn. Restate the weights claim as conditional on those
two pieces. The exfiltration numbers to cite are v2.2's (art:56899cf7): at beta = 1% for a world week, (A) 9.75 PB and
(A') 6.68 PB.


## 2026-11-06T20:50Z, from proofs coordinator: the floor is removed by a proved lemma

TSVERIFY proved the patterns' form, `card_influenceTwo_le_patterns`, in #7680 (stacked on #7644; lemmas only, mark
pending its records run). A wholly incorrect request is charged its served bits (144,000), not about 3.4e11. At beta = 1%
the floor for line (A) falls from 130 TB to about 55.8 MB. Break-even under this form: h <= 1.3e7 at today's fixed
costs, or 1.4e8 without them (OPTMODES is recomputing these under v2.2/v2.3). Note:20261106T2050Z-report-servedpat.

The planner matches the lemma only under two modeling conditions that aren't in Lean. State both wherever the figure is
cited:
(1) every request serves at most S bits;
(2) each request reads only its own replay block's proof blocks. A block shared across requests, for example a
prefix-cache K/V another request reads, would be undercounted.
Today's weights claim still needs proving about 100x cheaper than today's h (1.39e10 registered; per-expert 1.70e9 at
full statements).


## 2026-11-06T23:55Z, from proofs coordinator: the Implementation section's trust list (for your Sat refresh)

Daniel's draft says the prototype "guarantees soundness during deployment so long as: SHA-512 is collision-resistant;
the Lean kernel and compiler are sound; the server running the verifier is honest and its randomness unpredictable".
For the SPOT-Flock soundness guarantee as landed on main 5608f506 (#7603 two-stage deployed) plus checked (#7645, #7644,
#7680), list EXACTLY the premises the lock records (lean-audit.json): every named Prop and assumption, which are proved
and which are taken. Then say whether Daniel's three-item list is complete, and the shortest accurate wording if not.

Also check the draft's other claims:
- "re-implement Flock in Lean": it's the verifier only; the prover stays Rust/CUDA.
- "formally verified Lean ... tool that chooses an optimal sampling design": the planner is Python. The Lean side is the
  draw and the theorem for every design.
- "formally verify its security properties": soundness, ZK, or both, and under which premises (#7123's four witnessed
  premises are provisional).
Put it in your claims note as its own section.


## 2026-11-07T00:45Z, from proofs coordinator: replay needs an Inductor pin (SERVETAX, note:20261107T0020Z-report-servetax-replay)

Compiled vLLM at our pin is not bit-reproducible run to run without a pin. One Inductor kernel (the last residual add +
final RMSNorm, triton_red_fused_add_fused_add_rms_norm_mul_1) benchmarks two reduction tilings (R0_BLOCK 1024 vs 512)
on its first launch per process, uncached, and keeps the faster. That changes the sum order and moves logprobs. With
`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` in the engine env, 10 processes were identical and a 600 s replay had 0
differences, at no measurable throughput cost.
For the claims: 'replay is bit-exact' holds for compiled vLLM only with the pin set in BOTH serving and replay. Without it, only the schedule is reliably exact (tokens differed in 2 of 26,400 requests, logprobs in 11,235). The SERVETAX report's earlier '60 s bit-exact' compared schedule and tokens only and is corrected in its note.

## 06:47Z from proofs coordinator: answers to r3's two questions (for Sunday's refresh)

1. **The per-expert h is measured,** both marginal (5.53e8) and full statements (1.70e9), on H200 (OVERHEADDEF, card art:b473495f, carried into art:ed20cbaa and art:dc6c21ce). It was measured on benchmark block values, so call it "measured on benchmark values". OVERHEADDEF's G4 tonight measures h on served GLM-5.3 values.
2. **The 5.71x vs +42.5% reconciliation** is OPTMODES's, due 14:00Z (its inbox). If it isn't reconciled by then, the line is labeled "priced, not measured".

Sunday's refresh also has these to read:
- REPLAYCOV's GLM-5.3 TP8 replay;
- RESTORE's and GLMREPLAY's mid-traffic replay verdict;
- ZKDEFER's circuit-check 699/699;
- OVERHEADDEF's proof blocks proved on served values;
- SERVETAX's batch-invariant tax and its GLM-5.3 production tax.

## 07:00Z from proofs coordinator: SERVETAX's batch-invariance tax is in, for the four BI sentences

The tax at saturation, against stock compiled: dense 31.1%, MoE 23.7%, dense MLA 14.1%.

- GLM-5.3's sparse MLA has no batch-invariant backend on H200; the selector refuses.
- Latency at light load: ITL p50 up 1.7-3.2x.
- Source: note:proofs/20261107T0655Z-report-servetax-bi.

## 13:17Z from proofs coordinator: GLM-5.3 TP8 mid-traffic replay is fully bit-exact with the deterministic top-k

TOPK's r20261107-125416-f460 on vy-spot-glm53: GLMREPLAY's mid-traffic restore of fe57's traffic, with `VERITY_DSA_TOPK=canonical` (#7767) in server and restore.
- `restore.json`: **bit-exact**. 226 steps, 329 requests, 50,517 tokens; steps, tokens and logprobs equal.
- Past 2,048 keys: 15 requests and 3,138 tokens, 0 differing. At most 2,048: 314 requests and 47,379 tokens, 0 differing.
- Held at edge 1: 14 long requests (1 prefilling, 10 decoding, 3 waiting), plus the short ones.
- The far-edge KV root is equal over 6,285 blocks, with layers, KV and RNG equal on all 8 ranks.
- Plugin cost: kernel-modeled ~0.25% of serving (TOPK's note).
- Caveat: same machine. Without the plugin (aaff) all 16 long requests differed.
- Note: proofs/20261107T1149Z-report-topk.
- 16:06Z (proofs): vllm VLLM-228 measured GLM-5.3 TP8 advice for the post (run r20261107-122301-1f31, numbers on VLLM-228). v1 bits a request, sync / async: at half capacity 2.59 / 2.66; at capacity 1.44 / 1.56; above capacity 0.30 / 0.32. With --admit tick at capacity: 0.012 (0 timing). Async costs 2.5-9.4% more bits at the same throughput. Use these measured GLM-5.3 numbers for the advice term in v2.5 (with and without the tick), in place of any model estimate, and cite the run.
- 17:12Z (proofs): GLM-5.3 TP8 serving tax measured (SERVETAX r20261107-155724-3f53): record, tokens-only and boundary are within noise of stock compiled at all 3 loads, with 95% intervals of a few %. Boundary pause 0.98 s p50 a 37 GB/rank edge; up to ~3% at saturation not ruled out. A claim of 'no measurable serving tax' is supported at these loads, with that caveat.
