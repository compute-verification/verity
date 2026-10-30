# SILICON: what multiply-accumulate hardware would make PoUW cheap and its premises strong

Daniel, 29 Oct 2:31 PM PDT: "suppose I was Nvidia ... what properties would a new way of doing multiply-add-accumulate
have to integrate best with noising, denoising, cryptography ... easy to verify, circuit-level properties about how the
MACs work ... integrate the hashing into the MAC, or substantively decrease the hashing ... a swarm, a preliminary
exploration of the design space." Academic: nobody builds it this year. The output is the map.

## Why hashing is 99% of the lead's credit (the problem this attacks)

- The lead's h = 127.5: honest hashing per stock MAC (HASH.md, §interim-1). The MMA has no hardness premise.
  - It is linear across steps.
  - Its noise is low-rank (a row walk plus a column walk: w = x + R_t(i) + C_t(class(i), j)), because coins cost
    bandwidth.
  - So a cheater shortcuts it (COUNTER's `decomp`). SHA-512 is the only thing with a cost premise, so we credit
    SHA-512.
- Software folds help: R80X runs SHA-512's rounds over the words, h ≈ 18 at rank 4a (FOLD-ROUNDS).
- The question here is which property of the MAC itself would make the useful work creditable, or cut the hashing to
  a seal per call.

## Trust model (fixed)

- **Trusted:** that the chip computes its published MAC function. That function is a circuit-level spec, testable by
  measurement as `verity/ml/circuits/silicon` is today.
- **Not trusted:** no keys, no remote attestation, no device secrets, nothing the device signs.
- **Cost premises** about the new unit ("the cheapest way to compute f on this chip is its MAC path") are ranked on
  HASH's ladder. A premise about a standard primitive (a SHA-512 or Keccak round) ranks above one about NVIDIA's own
  mixer.
- **Rules unchanged:** everything zero-knowledge, live coins only, SHA-512 the one hash. By the 21:15Z ruling, a
  mixing round that doesn't bind isn't a hash.

## Metrics, for every design

| metric | meaning |
|---|---|
| h | honest hashing per stock MAC, after the change |
| Δarea | gate equivalents per MAC, synthesized, and % of the baseline s8 and FP8 MACs below |
| throughput, latency | the MMA's rate and pipeline depth, before and after |
| premise, rank | what the credit rests on, on HASH's ladder |
| T1 | the gap a cheater gains, attacked with `decomp` and `pack` first |
| T3 | extra served bytes, e.g. coin bandwidth |
| proof cost | C-Flock gates (or an arithmetic proof system's constraints) per MAC |
| useful | runs today's int8, FP8 and NVFP4 models exactly? |

**Baseline**, owned by SI-ALG and published first (22:45Z):
- A 4-wide s8×s8→s32 dot-product MAC.
- An FP8 E4M3 MAC with FP32 accumulate.
- Both synthesized with yosys + ABC (`pip install yowasp-yosys` needs no root) to a generic gate library, with the
  script on the branch.

## Lanes (notes: `lanes/compute-research/<ts>-report-silicon-<lane>.md`)

- **SI-FOLD: the accumulate path.**
  - Mix the accumulator non-linearly with a standard round ρ: a SHA-512 round, a Keccak-f round, a ChaCha
    quarter-round, an AES round, an Ascon or Xoodoo round.
  - Placement: each k-step, each prefix word, or each tile output.
  - What the credit's premise becomes, the gate cost of ρ against a MAC, and whether h reaches about 0. Also the
    cheapest mixing whose premise is about a standard primitive, and what `decomp` does to each.
- **SI-ALG: arithmetic and formats.**
  - Prime-field MMA (Mersenne-31, BabyBear, Goldilocks, 2^61−1); binary tower fields, carry-less (Binius-style, which
    fits Boolean circuits); RNS; Kulisch or fixed-order exact FP accumulation; wide integer accumulators.
  - For each: area; noise (perfect one-time-pad hiding, exact cancellation, no (127, 1, 1) weights); Freivalds and
    Schwartz–Zippel soundness; proof cost per MAC; usefulness (int8 MMA embeds exactly in F_p for large p); and
    whether algebraic sketches become sound enough to replace the per-word hashing.
- **SI-NOISE: noise and sketches on chip.**
  - A seeded generator in the operand path: a public function of the verifier's live coin (Trivium or Grain,
    ChaCha, AES-CTR, a Keccak squeeze), giving full-rank per-element noise at zero bandwidth.
  - Does that kill `decomp`, and let NH-class sketches (h ≈ 1) or a design with no per-word hashing survive?
  - Hardware sketch units on the accumulator stream (NH, polynomial hash, GF(2^k) multiply), and the denoising term
    computed on chip. What hashing remains at minimum: one SHA-512 seal per tile, or per call?
- **SI-THEORY** (when a slot frees): what property of a MAC's function admits work credit under a standard premise.
  Literature: fine-grained PoUW (Ball, Rosen, Sabin, Vasudevan 2017), PoUW from matrix multiplication (check
  Komargodski, Schen, Weinstein), RandomX's hardware-native design, memory- and bandwidth-hard functions, and zk
  hardware (ZPrize, zkASICs).
- **SI-RED + SI-CARD** (when slots free): attack each proposal, and card the frontier of Δarea against h against
  premise rank.

## Times

- Baseline at 22:45Z.
- Each lane's preliminary note by 00:00Z, which the coordinator synthesizes for Daniel by 00:30Z (5:30 PM PDT).
- **Done-when:** a map of design classes with the dominant point in each (Pareto over Δarea, h and rank), each number
  marked checked, measured or estimated, and the one or two hardware properties that buy the most.

**22:15Z, SI-THEORY preliminary (note 20261029T2203Z-report-silicon-theory):**
- **The dominant property is a SHA-512 round inside the MMA unit,** folding accumulator words into internal chained state that SHA-512 seals. h falls to about 0.1–1, and the premise becomes a loose lower bound (going around the unit costs at least the unit, with about 14× margin), still rank 4a.
- **Second, closure:** only a full-K MMA issue may advance the fold. That credits the MACs in issue time, with no operand noise, so the network limits stop binding.
- **Next best:** a raw-word SHA-512 round instruction, which gives DistinctWrites credit only. Intel's VSHA512RNDS2 takes the word with its round constant already added, which `decomp` breaks.
- **On stock hardware** no standard premise credits MACs (KSW, arXiv 2504.09971, confirmed: it credits one unit a word).
- **Scope:** every credit is relative to the published chip.
- Area figures are estimates until SI-FOLD's synthesis.

**22:40Z, SI-FOLD preliminary (note 20261029T2230Z-report-silicon-fold; run r20261029-222434-a055):**
- **A fused split transcript.** The accumulator stays stock, so outputs and denoising are unchanged. Beside the write-back port, a transcript register absorbs every instruction's accumulator words through one Keccak-f[1600] round, starting at the call's live coin and never feeding the output. Only full-K credited instructions advance it, each absorbing its shape tag, and each tile is sealed once by SHA-512.
- **Results:** h ≈ 0.28; area 2.7% of an s8 MAC (1.0% of FP8), or 5.7% at a round every 15 words for margin 4 on sm_120; depth 8 against the MAC's 45; proof +1.56 ANDs a MAC against the lead's +65; T1 = 0 against `decomp`, `pack` and injection.
- **Premise:** no program computes the chained rounds for less than the MMA's cost of the words. Rank 4a; rank 2 in the round-structured model.
- **Why the split is forced:** in-line mixing breaks outputs (0/128 exact), and affine mixers are faked by `decomp` (48/48). Absorbing per tile only gives T1 98.4%; without the trigger rule, 50%.
- **Open:**
  - **the energy gap:** about 6% under a 700 W cap, because zero operands run faster, so this credits issues, not energy;
  - closure is argued, not enumerated;
  - the wiring isn't priced;
  - **Keccak or SHA-512 for the round** is a premise choice for Daniel (a SHA-512 round costs 21% of the s8 MAC every word, or 6–14% at margin 4).
- Re-targeted: on-chip rank-1 operand noise from the live coin, for power whitening with cheap exact denoising ((uvᵀ)B = u(vᵀB)); closure enumerated; the wiring.

**22:55Z, SI-ALG preliminary (note 20261029T2235Z-report-silicon-alg; runs r20261029-222851-e9b5 and r20261029-220924-1126):**
- **The best format is exact (Kulisch) accumulation:** a 4-wide FP8 dot product into a 64-bit exact fixed-point register. 841.7 GE a MAC (1.03× the s8 MAC, 0.37× FP8/FP32), 211.8 C-Flock ANDs a MAC against 924. It is exact and order-independent, so FP8 noise cancels exactly and FP8 can be checked by Freivalds. NVFP4 into the same register costs 309.3 GE.
- For integer work, a 48-bit accumulator (1.07×) or an M31 accumulate (1.40×) replaces today's 1/2-per-draw Freivalds check.
- **No format cuts h on its own.**
- **A constraint, checked by `split_check.py`:** a sketch in the accumulator's own ring splits under `decomp`, field-native and binary-field MMAs included. A carry-less GF(2^k) fold over words accumulated with carries does not split. So keep carries in the MAC, and don't adopt a binary-field (Binius-style) MMA for PoUW.
- The in-unit GF(2^32) fold (+13%, rank 5) is dominated by SI-FOLD's Keccak transcript (2.7%), and is kept as a fallback.
- Synthesized baselines: s8 MAC and FP8/FP32 MAC, yosys + ABC (`cursor/compute-research-silicon-alg-8e03`).

## §prelim (coordinator, 07:30Z Fri 30 Oct; due 00:30Z, delayed by the 23:22Z–07:06Z platform outage)

**Answer.** Two circuit-level properties of the MMA unit would let PoUW credit the multiply itself and drop per-word hashing:
1. **A transcript (sketch) register the MMA unit writes itself** on every credited accumulator write, never feeding the served accumulator, and sealed by SHA-512 once a tile.
2. **A non-linear standard-primitive step in what it absorbs:** a seeded PRF word from the verifier's live seed (SI-NOISE), or a Keccak-f round (SI-FOLD).

Together (synthesized netlists, not simulated):
- h falls from 127.5 to about 0.28, a cut of about 450×, for +2–3% of an s8 MAC's area (about 1% of FP8's);
- the premise is about a standard primitive (rank 4a), needed only to a factor of 2–7, against the lead's 0.008%;
- the coins fall from megabytes a token to hundreds of bytes (seeds), which also removes the certifier-bandwidth problem;
- proof cost is about +1 AND a MAC, against the lead's +65.

**Why this shape, and why two lanes reached it independently:**
- The served path must stay exactly bilinear. Any mixer a linear denoiser passes through is affine, and `decomp` fakes affine mixers (SI-FOLD: 48/48). Mixing in line breaks the outputs (0/128 exact).
- So the transcript has to be a separate path, the split.
- **Closure:** only full-K credited MMA issues may advance it, each binding its shape. A k16 issue under a k32 tag halves the cost (SI-THEORY, checked in `closure_ref.py`).

**The classes, with the dominant point in each:**

| class | dominant point | h | Δarea | premise (rank) | status |
|---|---|---|---|---|---|
| transcript or sketch register (split) | Keccak-p round, or PRF word + rotl-xor | 0.28 | +2.1–3.1% s8 | standard primitive (4a), factor 2–7 | synthesized; SI-RED attacking |
| same, SHA-512 as the primitive | SHA-512 PRF or round | 0.28 | +6–21% s8 | ShaWrites with 6.9× slack (4a, existing) | synthesized |
| in-unit GF(2^32) fold | carry-less over carried words | about 1 | +13% s8 | our construction (5) | dominated |
| formats alone | Kulisch exact FP8 accumulation | unchanged | 1.03× s8 | none | exact, order-independent FP8: exact noise cancellation, Freivalds for FP8 |
| raw-word SHA-512 round instruction | DistinctWrites credit only | ≥ 18 (R80X) | — | 4a | Intel's VSHA512RNDS2 adds K_t, so `decomp` breaks it; a new one must take the word raw |
| binary-field MMA (Binius-style) | — | — | — | — | **avoid**: sketches in the accumulator's own ring split under `decomp` (SI-ALG) |

**Open, being worked to 09:00Z:**
- **Energy gap, about 6%:** zero operands run faster under a power cap, so the design credits issues, not energy. On-chip rank-1 operand noise is being tested.
- Closure isn't yet enumerated over the instruction classes that write accumulators.
- The wiring tap is unpriced.
- SI-RED's attacks.
- **Keccak or SHA-512 inside the unit**, for Daniel. A mixer that doesn't bind isn't a hash (21:15Z), so the one-hash rule holds either way. But a cost premise on Keccak is a new standard-primitive premise. SHA-512 costs 3–8× the area and keeps the existing premise.

**Scope:**
- Credit is in work at a locked clock, as §2a judges, not in wall time. Co-issued ALUs could fake transcripts for other statements (SI-THEORY estimates 14–28% of wall clock).
- Every credit is relative to the published chip.

**Smell check:**
- Every area figure is a yosys + ABC netlist on a generic library, not simulated, against SI-ALG's synthesized baselines. The premise margins are estimates.
- "h ≈ 0.28" is the card's operand binding plus the seal. It is not a measured kernel.
- SI-NOISE's coin ratio used the stale 13.2 MB decode figure; the lead's real decode stream is 144.79 MB a token, so the saving is larger.

**08:00Z, SI-RED (note 20261030T0750Z-report-silicon-red; run r20261030-074057-61d1). All three proposals break as written; each has a checked fix that costs about nothing:**
1. **Sparse or int4 issue at the credited shape** (m16n8k32): a 2:4-sparse s8 or an int4 issue advances the transcript at half the cycles, T1 = 50%. It passes `closure_ref`'s verify and SI-FOLD's trigger; SI-FOLD's trigger refuses int4. **Fix:** advance and tag on the whole instruction descriptor (format, sparsity and shape), not on shape alone.
2. **Copy across tiles at zero data:** if a call's tiles share coins, one chain seals every zero-data tile, T1 = 1 − 1/N. **Fix:** start each tile's fold from its own state, H(coins ‖ tile index).
3. **SI-NOISE's sketch falls to double duty:** a hidden GEMM scaled so every accumulator ≡ 0 mod 2^b (int7×int8 at b = 1, int4×int4 at b = 8) makes the sketch split into hidden plus claimed. Over-credit is 31.6–48.4% of stock. **Fix:** absorb acc + n·φ with φ fixed by no rotation (φ = 2³¹ − 1; 0x01010101 fails).
- With all three fixes, the harness finds no over-credit. Still open:
  - the descriptor's field list must be measured on sm_120 and the H200;
  - φ is checked only against exact separation;
  - the seal's and the PRF's premises are unattacked.
- BRW: brw32 over ℤ/2³² fails from G 3. brw31 (F_p) and the hardware BRWs hold to G 31, except at zero data without per-tile starts.
- **Correction to §prelim:** the design needs these three rules to hold. §prelim's "T1 = 0" was the proposals' own claim, before attack.

**08:10Z, SI-NOISE final (note 20261029T2232Z-report-silicon-noise, status final; run r20261030-075845-30db):**
- SI-FOLD's split Keccak transcript (FT) and SI-NOISE's PRF sketch (SN) are one class.
  - FT dominates on cell area (−9.4 to −12.9 GE a MAC) and on premise: one Keccak-chain assumption against SN's four.
  - SN dominates on proof cost above margin 2.33, and may win on area if FT's tap needs its own channel (up to 23.6 GE a MAC), which layout decides.
  - They tie on h, energy and closure.
- **Recommendation:** FT with per-chain ids (SI-RED's copy fix) and the whole-descriptor tag (the sparse-issue fix). On the H200 that is a round every 32 words, margin 3.75, 22.2 GE a MAC = **2.71% of an s8 MAC**.
- **§2a credit theorems:** both have gap 0 against stock. cost(P) ≥ cost(H) + cost*(F, Z | A, B, A·Bᵀ). FT's hypotheses are the chain-id rule and the Keccak-chain assumption (ε ≈ 2⁻⁵¹²); SN's are seclusion, distinct ids, the PRF and no-separation with the 2³¹−1 multiplier.
- Under §2a, the energy gap (about 6%) is time, not work, so it falls outside the theorem. Closure needs the whole-descriptor tag only if issue time is ever judged.
- Corrections to its preliminary: no-separation was wrong (SI-RED); margins were 1.67× off; the Keccak lead was overstated. SI-FOLD's 0.90 margin a bit assumes the full 1,600-bit output; at rate 1,088 it is 1.32.
