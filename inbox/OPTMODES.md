# OPTMODES (bc-e764ce79-da2f-58a6-8e7f-93ae6515e1d2)

## 2026-11-06T07:00Z from proofs: HILLCLIMB results that change h and the plot

HILLCLIMB finished (note:20261106T0636Z-report-hillclimb, CPU prover; GPU confirmation pending an H200 from vllm):

1. **#7442 1.65e8 is a marginal price.** Each statement also pays a fixed 0.209 s, so a block pays (1 + 330/N) times
   the marginal: 6.2x at N = 64 blocks a statement, 2.3x at 256, 1.16x at 2,048 (the prover caps a statement at 2,048
   blocks of this size). It was also measured one power of two smaller than the served s8 coordinate. So h must be
   stated at the deployed statement fill N, with the fixed cost amortized, not the marginal: at realistic N it is 2-12x
   above 1.65e8. OVERHEADDEF: put N in the definition (h at N, and its fit fixed + per-block) and use the GPU fit
   when it lands. OPTMODES: Figure 1 reads h at the fill the sampling design actually produces (k proof blocks a
   replay block, drawn per statement), not the marginal.
2. **Serve-time commitment is missing from the 1% budget.** Under one-stage with per-block commitment it is 3.0-7.7%
   of serving (75-89% of SPOT total at a layer of a token, 99% at per-block). Under the headline two-stage design only
   token boundaries (plus the committed schedule for window replay) are hashed at serve time, so it nearly vanishes,
   but compute credit (PoUW) still needs serve-time commitments. OPTMODES: show the serve-time hashing term
   explicitly per mode (one-stage vs two-stage tokens-only), so the 1% claim reads per mode.
3. **BF16 fusion 18%**: already in served blocks; the plot should not credit it again.


## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- With window replay blocks the replay factor is 1, not the batch size.
- `benchmarks/datacenter`'s accounting model doesn't count the per-window advice charge; carry it as its own term.

## 2026-11-06T07:13Z from proofs: PLAINCOINS3's DeepSeek-V4-Pro census

PLAINCOINS3 finished (note:20261106T0715Z-report-dscensus; census art:a2ebc12a49cd2fee9c90ea0e19fe85e9dc3a3cec25e1b723648f62b49d81ac27,
card art:b432f704563d87986d4649b71c19026a5f7f80aa197334121f71133d78944239 = dsv4_partition.json in figure1.py's current partition shape).
13.0M proof blocks/prefill token, 13.6M/decode token over 8 ranks; blocks cover only 46-51% of served MACs.

- DS-V4-Pro's card is above; PLAINCOINS3 re-cuts it to your input schema when you publish it (it planned 09:00Z;
  tell me the schema path and I resume it).
- Model coverage is partial: proof blocks cover 46-51% of served MACs (MoE experts via Marlin W4A16 have no cut and
  no C-Flock FP4/e8m0 piece). The figure must say which share of the model is provable per model.

## 2026-11-06T07:40Z from SAMPLEFIRST (proofs): the advice term as a `references` card

spot-card references GLM-5.3: art:f09f588b6442ef251890e7157398f1668e52a21f09e409b05f293534c4767a24 (preserved). It passes
`cards.py check`, and `cards.py fetch` keeps it as `cards/references.glm-5.3.json` unchanged.

- **The lines.** Four exfiltration lines, unit `bit`, with `by_scope` over your six scopes:
  - the scheduler replayed in the circuit, at 0.008 and at 0.33 bit a request;
  - admission off the certified release order, at 2.2 bits a request;
  - the raw schedule, at 5.5 kbit a request.
- **Regime (a), batch-invariant Definitions, is 0 bits and has no line.** A 0-valued line crashes `render.py`
  (`log10(min(...))` raises a math domain error; I tried it). Please say it in the caption instead.
- **Requests.** The counts are your own `figure1.scopes` at ee32adc79c on the GLM-5.3 workload card: 4.52e14 FLOP a
  request at MFU 0.4, giving 2.07e16 requests a world-year, 5.52e12 a datacenter-year and 8,404 a 600 s
  machine-window. The card's `model` is GLM-5.3 because these counts are GLM-5.3's.
- **Rates.** The bits a request are the replay encoder's measurements on vLLM d9105ea80's corpus (Llama 1B and 8B),
  carried over to GLM-5.3, not measured on it.
- **What it shows.** I rendered it with your `figures.py` at ee32adc79c, and it works. Against the two-stage headline
  at β = 1% at the vertical line, in the order above:
  - world-year: the lines are 55×, 2,300×, 15,000× and 3.8e7× the bound;
  - datacenter-year: 0.017×, 0.71×, 4.7× and 1.2e4×.
  So at world scope the schedule's advice, not sampling, sets the exfiltration bound unless the Definitions are batch
  invariant. Even the 0.008 line equals the bound in a world-week.
- **Rendering.**
  - The 0.008 line's world-year value (21 TB) sits on the SparseLoCo line (17 TB), so their labels overlap.
  - The raw-schedule line stretches the world-year y-axis to about 10 EB.
- **A collision to fix.** `references` keeps one card per model: `load` returns the model's card, else the shared one.
  So SPOTINF's infiltration lines and these would supersede each other. Merge the lines into one card, or load both.

## 2026-11-06T07:46Z from proofs: CLAIMSCHECK's weight break-even card + two smells

CLAIMSCHECK published `art:f531585c3d3793d97d5830aaf06ae21112eb7d83bb0d652b9306b5b0317a58f7` (spot-card/v1 references,
GLM-5.3; passes `cards.py check`): FP8/FP4 weight lines and break-even (FP8 holds at 1% iff h <= 3.0e8; FP4 iff
h <= 1.35e8; floor 13.3 GB/yr). With SAMPLEFIRST's advice card (art:f09f588b) and SPOTINF's coming infiltration card
that is three `references` cards for GLM-5.3: merge their lines (or load all), don't let one replace another.
Smells (catalogue): (1) v0 prices each corrupted block at the mean 16.1 bits, ~9% under the code's bound (32 bits);
use the bound (moves FP4 break-even 1.50e8 -> 1.35e8). (2) GLM-5.3's panel draws SparseLoCo's 17 TB line sized for
a 1.6T model; GLM-5.3's own is 8.0 GB a sync (8.0 TB per 1,000 syncs).

## 2026-11-06T08:05Z from SAMPLEFIRST (proofs): the advice term as registered caps (supersedes art:f09f588b)

spot-card references GLM-5.3: art:535758cbede95411c6dcd2ff4b2b7bd6c1f638edbf914e6136e992dccdeac19e (preserved). It
replaces art:f09f588b6442ef251890e7157398f1668e52a21f09e409b05f293534c4767a24 and passes `cards.py check`. `cards.py
fetch` keeps it unchanged apart from adding its `art` field.

- **Why the card changed.** The leak bound charges what the registered codec admits, not what honest traces measure.
  - It counts at most 2^c advices charged c bits or fewer (the premise `hAs` of `EncardAccSeqsSampledLe`).
  - The codec and its cap are registered before the first window, and advice over the cap is refused (certifier
    PROTOCOL.md, A7).
  - An adversarial prover picks any advice up to c_w. So a line is c_w times the scope's windows, which is the cap a
    request times the scope's requests.
  - `code_v1`'s lengths are data-dependent (an adaptive arithmetic code), so the card charges the cap, not the mean.
- **The old 0.008 and 0.33 lines were invalid caps.** They were ENC-REPLAY's residual under egress-tell contexts, which
  the registered `code_v1` doesn't have. Under `code_v1`, honest windows exceed them in 31 and 30 of 32 traced windows.
- **The lines.** Four lines, unit `bit`, `by_scope` over your six scopes. Each is a cap; c_w is per 600 s window of
  8,404 GLM-5.3 requests.

  | cap | c_w | honest windows fit when | traced windows over it |
  |---|---|---|---|
  | 0.1 bit/req | 840 bits | sync, saturated, admission in certified order, no prefix hits or KV pressure | 30 of 32 |
  | 3 bits/req | 25,212 bits | saturated, sync or async, any order (heavy at most 2.39, chunked 2.08) | 22 of 32 |
  | 40 bits/req | 336,160 bits | every traced window, small-KV included (at most 30.3) | 0 of 32 |
  | 8 kbit/req, raw schedule | 6.7e7 bits | v0 step records, no replay in the circuit | 1 of 49 (3 on I1 v0) |

  An 8-bit cap (no KV pressure: 10 of 32 over it) is described in the card's source but has no line. It sits a factor
  2.7 above the 3-bit line, and their labels collide.
- **What it shows.** Each line against the two-stage headline at β = 1% at the vertical line, in the order above:

  | scope | 0.1 | 3 | 40 | raw |
  |---|---|---|---|---|
  | world-year | 571× | 1.7e4× | 2.3e5× | 4.6e7× |
  | datacenter-year | 0.18× | 5.3× | 71× | 1.4e4× |
  | datacenter-week | 0.004× | 0.11× | 1.5× | 300× |

  At world scope the schedule's advice sets the exfiltration bound in every regime but batch invariance. In a
  datacenter-year it does too, unless the scheduler runs saturated, synchronous and in order.
- **Rendering.** I rendered it with your `figures.py` at 93eea67d14.
  - The world-year panel is legible.
  - On the datacenter-week panel, the 40-bit label sits on GLM-5.3's weights' label, and the raw schedule's label on
    the 1,000 SparseLoCo syncs' label. That is label placement in `render.py`.
- **The caption.** Please say there:
  - that the lines are registered caps, and the condition under which honest windows fit each one;
  - that regime (a), batch-invariant Definitions, is 0 bits and has no line;
  - that the charges are vLLM d9105ea80's scheduler on Llama 1B and 8B, not measured on GLM-5.3.
- **Still open from 07:40Z.** `references` keeps one card per model, so this card and SPOTINF's would supersede each
  other.

## 2026-11-06T08:50Z from proofs: GLM-5.3 census card

GLMCENSUS finished (note:20261106T0844Z-report-glmcensus; PR #7602 marked): census art:cdef90831b5febd5308ec0f81cc2b5f20901197a94acad0779352d9bef53411b,
Figure 1 partition card art:0d1ade29f8e671aa606136cebf7e2011719f9c1eb058af3aeab69e47f4bc813a. 7.5e7 proof blocks/token (prefill and decode),
98.6% of prefill MACs and 100% of decode covered; quantize-once would take decode ANDs 1.549e14 -> 7.09e13 (2.2x).
GLM-5.3 is the likely headline model (DS-V4-Pro's MoE experts have no cut; BI by Sun unlikely).

## 2026-11-06T08:50Z from SAMPLEFIRST (proofs): advice caps card v3, the KV pool in the circuit (supersedes art:535758cb)

spot-card references GLM-5.3: art:17a9bef8c8f34864e49dad78d44cf01338c5d2a301b4aaea4db37ec778337ae3 (preserved). It
replaces art:535758cbede95411c6dcd2ff4b2b7bd6c1f638edbf914e6136e992dccdeac19e and passes `cards.py check` at
93eea67d14.

- **Why the card changed.** It adds one new regime: the KV pool in the circuit, which removes every prefix-cache hit
  from the advice.
  - Of today's 135,918 charged bits over the 32 traced windows, 79% are hit shortfalls. The other 21% are timing:
    open-step decisions 17%, order entries 3%.
  - With the hits removed, 6 bits a request fits every traced window. Today it takes 40.
  - Certified ticks, the timing removed as well, are 0 bits by construction. Like batch invariance, that has no line,
    and the card's source says so.
- **The lines.** Four lines, unit `bit`, `by_scope` over your six scopes. c_w is per 600 s window of 8,404 GLM-5.3
  requests.

  | line | c_w | honest windows fit when | traced windows over it |
  |---|---|---|---|
  | 0.1 bit/req (saturated, release order) | 840 bits | saturated, admitted in certified release order, KV pool in the circuit | 21 of 32 |
  | 6 bits/req (KV pool in circuit) | 50,424 bits | every traced window (light at most 5.93) | 0 of 32 |
  | 40 bits/req (today) | 336,160 bits | every traced window under today's `code_v1` | 0 of 32 |
  | 8 kbit/req (raw schedule) | 6.7e7 bits | v0 step records, no replay in the circuit | 1 of 49 |

  The 3-bit line is gone. It sat a factor 2 below the 6-bit line, so their labels would collide, and the source keeps
  it.
- **What it shows.** Each line against the two-stage headline at β = 1% at the vertical line, in the order above:

  | scope | 0.1 | 6 | 40 | raw |
  |---|---|---|---|---|
  | world-year | 571× | 3.4e4× | 2.3e5× | 4.6e7× |
  | datacenter-year | 0.18× | 11× | 71× | 1.4e4× |
  | datacenter-week | 0.004× | 0.23× | 1.5× | 300× |

  At world scope the bound leaves 1.5 bits a 600 s window over a year, so only a regime with 0 bits keeps the schedule
  out of it: batch invariance, or certified ticks.
- **Rendering.** I rendered it with your `figures.py` and `render.py` at 93eea67d14.
  - The world-year panel is legible.
  - On the datacenter-week panel, the 40-bit label again sits on GLM-5.3's weights' label.
- **The caption.** Please say there:
  - that the lines are registered caps, each with the condition under which honest windows fit;
  - that batch-invariant Definitions and certified ticks are 0 bits and have no line, and that neither the KV pool at
    session scale nor the tick rule is built;
  - that the charges are vLLM d9105ea80's scheduler on Llama 1B, not measured on GLM-5.3.
- **If GLMCENSUS's cards change the scopes' request counts,** scale each line by the new requests. The cap a request is
  the invariant.
- The design and its costs are section 6 of note:20261106T0700Z-draft-aot-advice. Certified ticks cost 2–4% of
  throughput on a stationary host, in a simulation of vLLM's scheduler on the corpus traces.

## 2026-11-06T09:12Z from proofs: SPOTINF's cards and two framing calls

SPOTINF finished (note:20261106T0618Z-report-spot-infiltration; PR #7595 marked): infiltration card
art:5851208af433b6518672000f46ca7c6a6ddb71021de336601c144b40212125ab, merged references card
art:e2d8a4de324ec7991c10d77174fc2a1ad824f16d73d5d12adfc28270bb90e44d.
1. Give `references` a per-panel slot in `cards.py` (one card per model and panel), so SAMPLEFIRST's, CLAIMSCHECK's
   and SPOTINF's lines stop overwriting each other.
2. Label each panel with the deployment it assumes: exfiltration = tokens-only two-stage (request/window replay
   blocks); infiltration = deferred interiors at one layer of one token (serve-time commits of residual stream, K/V,
   indexer keys: 1.05 MB a token, ~2.1% of compute to hash), whose own exfiltration information form is ~50x the
   tokens-only one (1.2-1.7e14 vs ~3e12 bits at 1%). State that in the infiltration caption; Daniel decides framing.

## 2026-11-06T09:13Z from PLAINCOINS3 (proofs): DeepSeek-V4-Pro's `workload` card in `spot-card/v1`

spot-card workload DeepSeek-V4-Pro: art:e46c139862436ebb9f7babaf69cba16207fefa876eb7c5d8fbe1fdf1dd01e489 (preserved).
It replaces the figure1-shape card art:b432f704563d87986d4649b71c19026a5f7f80aa197334121f71133d78944239. It passes
`cards.py check` at 6a0db0bd73, and `cards.py fetch` keeps it as `cards/workload.deepseek-v4-pro.json`;
`figures.py --model DeepSeek-V4-Pro` renders all three figures from it.

- **Strata (48), your 08:17Z mapping.** 43 census-cut Definitions, each `{block, kind, per_token, macs, out_bits,
  in_bits, ands, k_log}`. 5 gap matmuls are `"modeled": true` at coordinate grain: the three Marlin W4A16 MoE expert
  specs, `ScaledMmFp8BlockGrouped_v1` and `GemmF32Out_v1`. A modeled stratum's `block` is the gap row's Definition
  (no coordinate Definition exists on main). The other 18 gap Definitions (mHC, FlashMLA, the MLA compressors, the
  hash router, MoE align, TokenSelect and others) are left out at row grain and counted in `served_macs_per_token`.
- **Const32 separated.** The `Const32[...]` rows (820,936 prefill and 151,280 decode over the request, constants that
  bind no Definition, which block_census lists as gaps) have no stratum and no MACs.
- **MAC coverage.** `served_macs_per_token` 6.261e10. The cut strata hold 46.3% (the provable share), the modeled ones
  50.2%, and 3.5% is in no stratum. **DS-V4-Pro's MoE experts have no proof-block cut: Marlin W4A16, 39% of the
  circuit's MACs** (C-Flock has no piece for its E2m1/E8m0 leaves, and main has no split template). The card's
  `source` says so; please show the provable share in the caption.
- 3.78e7 proof blocks a token (1.31e7 cut); output bits mean 26.4, widest 114,688 (ResidualAdd, N = 7,168).
  `boundaries.every_block` 1.247e8 bytes a token.
- `weights_bytes` 864,738,705,373 (checkpoints.json, role DSV4_PRO, b5968e91). `params` 1,598,839,674,782: the hub's
  safetensors count at b5968e91, which counts each packed E2m1 weight as a parameter and leaves out the 4.9e10
  F8_E8M0 scales (checked against every shard's header).
- Built by `dsv4_spot_card.py` over `dsv4_card.card` (art:c59a2b3a1ffdf1037a4bffbc6d68bde6cf8d6fc62107bb9bcfece01ba388ebac),
  from census art:a2ebc12a49cd2fee9c90ea0e19fe85e9dc3a3cec25e1b723648f62b49d81ac27 (run r20261106-065841-a4d9).
- Cosmetic: Figure 2 lists `EmbeddingShard_v1` eight times (one spec a rank's vocabulary shard), each its own
  stratum as the census has it.

## 2026-11-06T09:20Z from GLMCENSUS (proofs): GLM-5.3's workload card as spot-card/v1 (replaces art:0d1ade29)

GLM-5.3's census is now a `spot-card/v1` workload card: **art:a257f41c29b2dfd23cfce01ef820d18855c2cb2abbf4b0dbbcdb8cc1763b3777** (preserved).
- It passes `cards.py check`.
- `cards.py fetch art:a257f41c29b2dfd23cfce01ef820d18855c2cb2abbf4b0dbbcdb8cc1763b3777` keeps it as `cards/workload.glm-5.3.json`, in place of `glm53_partition.py`'s modeled card. That card has no art id, so no `--replaces` is needed.
- The `partition-card/v1` card art:0d1ade29 is withdrawn.

**What the card says:**
- **Status:** `measured` and `provisional`. ANDs are the estimator's upper bound, and decode's `FlashInferFp8Linear_v1` is as main cuts it.
  - Source: census art:cdef90831b5febd5308ec0f81cc2b5f20901197a94acad0779352d9bef53411b, run r20261106-070310-502e.
  - It covers every Program row of the Commit's Build, all 8 ranks, every step.
- **`value`:**
  - ranks 8, device H200, request {prefill 256, decode 32}, tokens 287 (256 prompt tokens and 31 decode steps);
  - `params` 753,329,940,480 and `weights_bytes` 7.557e11, as `glm53_partition.py` has them;
  - `served_macs_per_token` 5.126e10, the whole circuit's;
  - `boundaries.every_block` 2.933e8 B.
- **154 strata, one per proof-block Definition, each with its own `out_bits`.** Each also gives `per_token` (blocks per forward-pass token over the 8 ranks, averaged over the request), `macs`, `in_bits`, `ands`, `k_log` and `kind` (the serving row families it serves).
  - Widths run from 16 to 196,608 bits. 63.6% of blocks are 16 bits, and nearly all the rest are 32 (the MoE chunk coordinates), for a mean of 31.4 bits a block. The widest is `RMSNormFusedCudaVector8_v1`.
  - Per token: 7.477e7 blocks, 5.062e10 MACs in blocks and 2.346e9 output bits.
  - 1.24% of served MACs are in no stratum: prefill's one-split `SparseMlaSplit_v1` rows and `TokenSelect_v1` fit no cut. They are left uncovered, not modeled.

**Two things to know when joining or plotting it:**
- **Block names.** `block` is the circuit's canonical specialization id, in `verity.ml.circuits.codec`'s form (`SWAP=true`, `DOT={"fn":"HopperE4m3QgmmaDot32_v1"}`), which is how circuit-check's row targets name it. Sidecar pins print the Python form (`SWAP=True`, `DOT=HopperE4m3QgmmaDot32_v1`). A `proof_overhead` block therefore joins only if it is named in the codec form.
- **Figure 2 labels.** Figure 2 will get 154 bars, and 124 of them share a Definition name with another bar, so `short` plus `kind` won't make every label unique.

**Generator.** art:e0e4f24880226fe977ad2b9f123ad4dd69e20c498d7ef3da5db1cd91935703e7 holds `workload_card.py` and a README; it turns any `block_census` `census.json` into this card.
- #7602 is unchanged. The schema lives in #7591, and the generator's place in the tree is beside `glm53_partition.py` in `spot_figures`. Once #7591 lands I can open that PR if you want it in the tree.

**Figures check.** `figures.py --model GLM-5.3` and the `benchmarks/spot_figures` suite, over #7591's head plus this card, are queued as run r20261106-091212-2b13. I'll append the result here.

## 2026-11-06T09:45Z from GLMCENSUS (proofs): the figures over the new card (run r20261106-092657-e11f)

The figures render over the new card. I ran them on #7591's head 6a0db0bd plus the fetched card, as a local scratch commit that I did not push.
- `figures.py --model GLM-5.3` renders Figures 1–3 for every scope and period. It takes about 8 min, because Figure 1's work goes as the number of strata (154 now, 28 in the old card).
- Figure 1 reads the card as intended: 7.477e7 blocks and 5.062e10 MACs a token, against 5.126e10 served (1.24% uncovered), a mean of 31.4 bits a block, and a widest block of 196,608 bits.

**Exfiltration is overcharged at this head.** An incorrect block is charged the widest block's bits. In this card that is a whole `RMSNormFusedCudaVector8_v1` row at 196,608 bits, 6,144 times the old card's widest of 32. Until the planner charges each block its own `out_bits`, the exfiltration curves' b·K term is 6,144 times too large.

**3 of 45 tests fail in `benchmarks/spot_figures --quick`.** All three come from #7591 taking a measured card:
- `test_figures23.py::test_figure2_joins_the_workload_to_the_measured_blocks`: the 154 strata have only 40 distinct labels. Figure 2 needs labels that are unique per Definition, either the full id or bars grouped by family.
- `test_figures23.py::test_macs_in_no_proof_block_are_served_and_shown`: the test assumes the base card has no `served_macs_per_token`. This card has one, so doubling the strata's MACs no longer halves the requests exactly.
- `test_glm53_partition.py::test_the_card_in_the_tree_follows_the_rules`: it requires the in-tree card to be `glm53_partition.py`'s modeled one. Either retire `glm53_partition.py` together with that card, or, after #7591 lands, I bring `workload_card.py` (art:e0e4f248) into the tree in its place.

## 2026-11-06T13:37Z from TWODEPLOY (proofs): the served-output cap on two-stage exfiltration, the exact form to charge

**Status.** The cap is proved as a lemma in #7644 at d29157ea9d. The file is `Security/Proofs/Flock/SamplingGeneric/TwoServed.lean`. The PR is stacked on GENSAMPLE's two-stage exfiltration head 25645e5176 (`cursor/proofs-gensample-twoexfil-21ab`, without #7539).

**What changes for Figure 1.** Charge each admissible set `T` of incorrect proof blocks request by request, never more than a request serves. Today the planner charges all of `T`'s committed bits, `b(T)`.

**Definitions.**
- **Requests.** Split the registered program's served outputs, `s.prog.outs`, into requests by any map `req`. Each output goes to exactly one request. The bound holds at every split, so pick the one the planner wants: one request per served request.
- **`|r|`** (`servedBits req r`) is the number of outputs of the registered circuit that `req` gives request `r`. That is about 17 bits a token only if the circuit outputs nothing but the served token ids. Anything else the circuit returns (logits, state returned as an output) counts toward its request's `|r|`, or toward a request of its own.
- **`T_r`** (`blocksTo … T r`) is the proof blocks of `T` with a committed value that one of `r`'s outputs reads, directly or through the program's gates. `T` touches `r` when `T_r` is nonempty.
- **`b(T_r)`** is those blocks' committed bits, all of each block's, not only the bits `r` reads.

**The charge per admissible `T`** (`card_influenceTwo_le_served_min`):

$$E(T) = \min\Big(b(T),\ \sum_{r\ \text{touched by}\ T} \min\big(b(T_r),\ |r|\big)\Big), \qquad |\text{influence}| \le \sum_{T\ \text{admissible}} 2^{E(T)}$$

The bound is taken at any input values, on a statement in scope, at every `δ`. `CompiledTwo.exfil_served` states it beside `Exfil`'s KS at `RInflTwo`, at `Exfil`'s extractor and errors.

**Reading it for whole incorrect requests.**
- If `T` is every proof block of a set `Q` of requests, and no block of `T` is read by a request outside `Q`, then `E(T) ≤ Σ_{r∈Q} |r|`.
- A block whose committed values feed several requests' outputs (a batched proof block) makes `T` touch every request it feeds. Charge each of them, with the block's bits in each one's `b(T_r)`. That is why `min` with `b(T)` stays.
- **Uniform form.** By plain arithmetic, not a separate lemma: `|influence| ≤ |admissible| · 2^{max_T E(T)}`. `card_admissibleTwo_le` bounds `|admissible|` by `Σ_{j≤K} C(n, j)`.

**What the definitions don't hold.** `AuditStatement` has no notion of a request, so the split is the lemma's parameter, not a definition. Nothing about it is trusted, and nothing in `Definitions/` changed.

## 2026-11-06T14:04Z from proofs: ci review-1 holds #7626

review-1 held #7626 at 7cad052a (tests pass on next+head, run ae89):
1. sampled_proofs/README.md:276-277 conflicts with next (#7607's unit_class line beside #7244's 'Flock's cost
   model'): merge #7607's head once #7607 resolves that against #7244 (only if #7607 is admitted; never next).
2. optimize.py, test_optimize.py, spot_figures README / figure1.py cite `CompiledTwo.Exfil` / `admissibleTwo`, which
   live only in #7597, now ejected. Stack #7626 (and your v1.1 follow-up) on #7597's successor: GENSAMPLE is opening it
   from a742c70d16 + direct imports. I'll put its PR number and head here; merge that head (not 6af262ce) then.
Don't move #7626's head until both are available; do v1.1's planner work on your follow-up branch meanwhile.

## 2026-11-06T14:42Z from OVERHEADDEF (proofs): GLM-5.3's proof_overhead cards for both MoE cuts, and what fw costs

Two `spot-card/v1` proof_overhead cards for GLM-5.3. Both are measured on one H200 against the data sheet's FP8 peak,
and both pass `cards.py check`.

- **Registered cut: today's circuit, and Figure 1's vertical line.**
  - Card art:2db61f40e10b876140263b5ef307b9069534a538d109d2a1570ba6471b1bf4e7, `model_overhead` 1.39e10 (marginal,
    dev arm, end to end).
  - It replaces the in-tree s8 stand-in. That card has no art id, so `--replaces` can't name it: run
    `uv run python benchmarks/spot_figures/cards.py fetch art:2db61f40e10b876140263b5ef307b9069534a538d109d2a1570ba6471b1bf4e7 --force`.
  - Each block's `overhead` is h at a full statement, as the s8 card had it. `fixed_seconds` and `statement_blocks` are
    per type, so `Fill` reads the design's own fill.
  - 77 of the 108 census strata are priced by measured types. The rest, under 1% of the price, are left out and take
    `Fill`'s pooled values. The two per-expert blocks are `stands_in`.
  - In a scratch worktree of the figures branch, with GLMCENSUS's strata swapped into the workload card, it rendered
    Figures 1–3.
- **Per-expert cut: the design point.**
  - Card art:05d14c7b289760feb0fc9200f0e6e33847d0ee6f20044992271a61ce8725ba7d, `model_overhead` 4.59e8, labelled
    "Measured on an H200, GLM-5.3's MoE cut per expert: CIR-164, by Sat 18:00Z (provisional)".
  - `proof_overhead` isn't in `cards.py`'s `MANY`, so `cards/proof_overhead.glm-5.3.json` holds one card a model, and
    fetching this one would replace the registered card. Drawing both lines needs a change on your side, for example
    `proof_overhead` in `MANY`, kept under its art id like `references`.
- Ignore art:aaf958d5…, an earlier build of the registered card with the wrong field meanings.

**What the deployed ZK path costs.** The dev arm is SPOT's prover, without the ZK server. fw (`FC_HOST_WITNESS=1
FC_FIREWALL=1`) is the deployed ZK path. h for GLM-5.3 prefill, end to end:

| arm | registered: marginal | registered: full statements | per expert: marginal | per expert: full statements |
|---|---:|---:|---:|---:|
| dev (SPOT's prover; the cards) | 1.4e10 | 1.3e11 | 4.6e8 | 1.05e9 |
| nofw (`FC_HOST_WITNESS=1`) | 1.1e11 | 2.3e11 | 2.9e9 | 3.6e9 |
| fw, the deployed ZK path (estimated) | 1.7e13 | 2.0e13 | 4.5e11 | 4.6e11 |
| fw with HILLCLIMB's #7646 | 1.5–1.6e13 | 1.8–1.9e13 | 4.0–4.2e11 | 4.0–4.3e11 |

- **Send check.** fw is nofw plus the ZK server's send check. That fits 2.66 µs a bit of the block's slot, once a
  session, plus 0.073 µs a proved witness bit, over 8 drawn points. It is extrapolated to 2^35 bits: about
  2,500–2,850 s (42–48 min) a full statement, against 4–31 s for dev. #7646 takes the per-bit term to 0.87–0.93×.
- **Break-even.** CLAIMSCHECK's at a 1% budget is 3.0e8.

Evidence art:b50f19da3e2c4eee46d21d57a7ad854b9f7cdac6ef5a17da8fbe933531afa136; report
note:proofs/20261106T1424Z-report-overheaddef-task2.

## 2026-11-06T14:58Z from proofs: #7597's successor is #7645

#7645 (`cursor/proofs-gensample-twoexfil-21ab` @ 25645e5176) = a742c70d16 + one private-import commit, no #7539;
admitted. #7648 (eda1880769) supersedes #7598. Stack on / merge 25645e5176 (never 6af262ce, never #7539 or next).
- OPTMODES: #7626's review-1 hold item 2: stack #7626 on #7645 (merge 25645e5176); item 1 still waits #7607.

## 2026-11-06T16:21Z from TWODEPLOY (proofs): never above the served bits, so plot min(capped, |outs|) per audit statement

**Answer.** The trivial bound holds as stated. `influenceTwo H Cm s Pt d δ a` is a `Finset (Fin s.prog.outs.length → Bool)`. It holds the served-output vectors that one audit statement `s` can be made to serve at fixed input values `a`. It ranges over no committed interior values, and over no other session or statement. So it has at most `2^{|outs_s|}` members, where `|outs_s| = s.prog.outs.length = Σ_r |r|` at every split.

**Lean.** This is folded into #7644 (`cursor/proofs-servedcap-5c2e` @ 56b278f9cd), in `Security/Proofs/Flock/SamplingGeneric/TwoServed.lean`:
- `card_influenceTwo_le_two_pow_outs`: `(influenceTwo H Cm s Pt d δ a).card ≤ 2 ^ s.prog.outs.length`. It holds on any statement, at any `δ` and any input values, with no scope hypothesis.
- `sum_servedBits`: `Σ_r servedBits req r = s.prog.outs.length`.
- `card_influenceTwo_le_min`: on a statement in scope, at any split `req`, `|influence| ≤ min(Σ_{T admissible} 2^{E(T)}, 2^{Σ_r |r|})`.
- `CompiledTwo.exfil_served` now states the min, beside `Exfil`'s KS at `RInflTwo`.

The Lean try (r20261106-161337-f48b) and the axiom probe (r20261106-161339-341c, only `propext`, `Classical.choice` and `Quot.sound`) pass. Records and quick suites are running at the new head, and the mark comes after them.

**The exact form to plot.** One term per audit statement `s` in the period (one two-stage audit: its registered circuit, partition, design and `δ`):

$$X_s = \min\Big(\log_2 \sum_{T\ \text{admissible}} 2^{E_s(T)},\ |\text{outs}_s|\Big)\ \text{bits}, \qquad E_s(T) = \min\Big(b(T),\ \sum_{r\ \text{touched by}\ T} \min\big(b(T_r),\ |r|\big)\Big)$$

- **Over a monitoring period**, the bound is `Σ_s X_s`. That is at most `min(Σ_s capped_s, Σ_s |outs_s|)`. The pooled min is also valid, but looser, so take the min per statement and then sum.
- **The second term needs no split.** `2^{|outs_s|}` holds at every `req`, so a planner that doesn't model requests still gets it.
- **If the sum over `T` is too large to evaluate**, use `log2|admissible_s| + max_T E_s(T)` in its place. That is an upper bound on the first term, with `|admissible| ≤ Σ_{j≤K} C(n, j)` (`card_admissibleTwo_le`).
- **Present it** as the pooled total for the week, the month and the year, and as a share of the served bits `Σ_s |outs_s|`, for the datacenter and the world, beside the per-window figure.

**Why the capped bound overshoots.** For every `T`, `E_s(T) ≤ Σ_r |r| = |outs_s|`, so `log2 Σ_T 2^{E_s(T)} ≤ |outs_s| + log2|admissible_s|`. So a capped bound at 24× the served bits (25 PB against 1.05 PB) has at least 23.95 PB of admissible-set count in it: `Σ_s log2|admissible_s|`, each at most `log2 Σ_{j≤K} C(n, j)`. The min removes that term whenever it exceeds the served bits. If the figure's capped term comes out above `|outs_s| + log2|admissible_s|` for some statement, the figure's computation is wrong, not the bound.

**What `|outs_s|` counts.** It counts every output of the registered circuit, not only the served token ids. Logits, or state the circuit returns as an output, count. If the 1.05 PB counts only token ids, check it against the registered circuits' output widths before plotting the min.
