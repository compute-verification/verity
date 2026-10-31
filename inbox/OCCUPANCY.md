# OCCUPANCY: the minimal silicon properties, and "declared plus undeclared work is inefficient"

Daniel, 31 Oct 12:04 PM PDT, paraphrased:
- **What are the one, two or three properties we want of the silicon?**
  - (1) It runs our PoUW efficiently.
  - (2) It is inefficient at *breaking* PoUW.
  - Separately, (3) trusted event counters, so we don't rely on back-of-envelope throughput.
  - And (4) make it inefficient to do declared and undeclared work concurrently.
- **His sketch of (4):** a trusted randomness stream constantly fed to the GPU and folded into chained hashes. Each
  accumulator has only enough fast memory (registers) for its latest chain state. Doing undeclared work means evicting
  that state, and when new randomness arrives the chip must backtrack and redo work, at, say, 50% extra. So "only
  declared work is 50% more efficient than declared plus any undeclared work". "Can we try a few variants of this
  argument?"

## Context (read first)

- `inbox/SILICON.md` (§prelim and finals): FT, the split transcript beside the writeback. It needs closure (only
  full-K credited issues advance it), the whole-descriptor tag and per-tile chain starts. h ≈ 0.28, +2.7% of an s8
  MAC's area.
- `inbox/CORE.md` §2a: credit is *work* in W1 units; time and energy are outside the work theorem. An occupancy claim
  is a **time or rate** claim, a different statement, and fine for resource exhaustion. Say which one each variant
  proves.
- **What we measure today:** the H200 tensor sweep (note:compute-research/20261030T1534Z-report-h200-tensor-sweep)
  peaks at 0.99915 of the 1830 MHz bound, so peak rate is known to about 0.1% on that card. The on-site network
  certifier is auditor-trusted, issues live coins, caps egress, and can time responses.
- **Threat models, kept apart:** (T-op) the operator is adversarial, the silicon honest; (T-vendor) the vendor
  colludes too, with hidden or faster silicon. Daniel wants auditability by parties who distrust the supply chain, so
  say what survives under T-vendor.

## Variants to analyze (each lane owns one; add variants as you find them)

- **V1, deadline and rate.**
  - Live randomness arrives every Δ, and each Δ's chained commitment of declared work must arrive by a deadline only
    meetable at ≥ (1 − ε) of peak tensor throughput. The work depends on r_t, so it can't be done ahead.
  - Undeclared work that takes tensor throughput causes misses. Power-cap and HBM-bandwidth versions too: declared work
    at full rate already uses the power or bandwidth budget.
- **V2, state occupancy (Daniel's register idea).**
  - The chain state must sit in scarce fast memory (registers, shared memory, L2) and be touched every step, so
    undeclared work must spill and restore it, or restart the chain, and pay in time.
  - The lineage is software-based attestation (SWATT, Pioneer: checksums whose fastest code uses every register, so
    any extra code is a measurable slowdown) and proofs of space-time or secure erasure.
  - On today's GPUs, and with silicon support. The key issue: GPUs interleave warps and partition register files, so
    small chain states don't exclude co-resident work. How big must the state be, and what does that cost the useful
    kernel?
- **V3, a credited-mode lock (silicon).**
  - In credited mode an SM may issue only credited instructions. Any other issue resets or poisons its transcript, or is
    recorded in it, and the chain state can't be saved by software.
  - Also the minimal **trusted event counters**: what to count (MMA issues by descriptor, cycles in credited mode,
    stalls, non-credited issues, clocks), and how to bind them without device keys (absorbed into the transcript,
    checked for consistency against the proven work). What a lying chip (T-vendor) can still do.
- **V-red, attacks across all variants:**
  - co-residency on idle units (ALUs, FP64, copy engines, other SMs);
  - time-slicing between Δs;
  - jitter and thermal slack;
  - precomputation;
  - hidden or faster silicon;
  - the certifier's timing resolution.

## Deliverable per lane

The claim it gives (statement, units, ε, δ), the trust it needs (T-op or T-vendor), the attacks and their prices on the
H200 (and sm_120), what silicon change it needs (if any) with an area or cost estimate, and whether today's GPUs can
run a software version. The coordinator then writes **the minimal property set** (2–3 properties) from the four notes.

Notes go to `lanes/compute-research/<ts>-report-occupancy-<lane>.md`. First notes by 02:00Z Sun (7 PM PDT), and the
coordinator's synthesis to Daniel by 03:00Z.

## Facts so far

- **OCC-STATE (V2)** (note:compute-research/20261031T1941Z-report-occupancy-state; CPU models, estimates, no GPU):
  - **On a stock H200 the X collapses to about 0–5%.** The cheapest cheat saves the chain state (about 3.2 µs an SM
    for 164 KB, estimated) plus a 0.5 µs pipe drain (measured), runs undeclared work, and restores. Against a 72 µs
    p99 jitter (loopback TCP, measured), that's X ≈ 1–5% with per-step deadlines and under 0.4% with window deadlines.
    50% would need slack under about 10 µs a step.
  - **Daniel's 50% is the expected redo when state is lost at a random point.** A cheater never pays it: it saves
    instead, or waits for the window boundary (1.500 against 1.000).
  - **A lean GEMM at the same speed leaves about 100 KB of registers free** for co-resident integer or FP32 work.
  - **With silicon**, a per-SM state that software can't save, poisoned by any non-credited issue, makes the memory size
    irrelevant. **V2 reduces to V3's issue accounting** (under 0.1% of an SM).
  - Under T-vendor, nothing in V2 holds. Recommendation: V3 plus V1, and drop V2 as a property.
  - **Smell for V1 (OCC-RATE):** "deadline at peak" rests on a synthetic kernel with operands in shared memory. A real
    GEMM may be L2-bound at about 52–75% of peak (estimated), which leaves tensor cycles idle. Measure a real-operand
    GEMM's rate before any deadline claim.
- **OCC-RATE (V1)** (note:compute-research/20261031T1945Z-report-occupancy-rate; round counting plus a small LP, CPU;
  art:da4b272d…):
  - **A deadline bounds only the resource the declared work saturates.** On the H200 under T-op, undeclared tensor work
    is at most 0.3% (clock-bound), 1.1% (power-bound with attested temperature) or 4.4% (power-bound with
    operator-controlled cooling), each plus (deadline + Δ) / window. With the deployed challenger about 96 ms away, the
    window term is 11% at 1 s.
  - **Under T-vendor the time version fails:** lifting the clock cap alone gives +8.2%.
  - **Peak precision:** "0.1%" holds only for low-toggle operands. Uniform random s8 is power-capped at 0.939–0.949
    of the bound.
  - **Concurrent work stays free:** beside saturated wgmma, FFMA (27 TFLOPS) and IDP4A (3.1% of the int8 rate), plus HBM,
    the copy engines and NVLink.
  - **Software-only today:** only as a useless exclusion filler (live-seed coins × a committed random matrix, about
    692 W). For useful work, the real GEMM's gap to peak is the floor. It's being measured now (OCC-RATE round 2,
    sm_120).
  - **Silicon:** FT's coin-seeded transcript extends V1 to useful work. V3's lock covers an L2-bound GEMM's idle cycles.
    An operand-independent clock or attested temperature narrows the 4.4%.

## §synthesis (coordinator, 20:15Z Sat 31 Oct): the minimal silicon property set

Sources: OCC-SILICON (note:compute-research/20261031T2005Z-report-occupancy-silicon; netlists matched to their Python
references bit for bit, and LPs matched to their closed forms), OCC-STATE and OCC-RATE.

- **P1, a fused transcript with whole-descriptor closure (FT).** Daniel's (1) and (2) merged.
  - The MMA unit absorbs every credited accumulator word into an on-chip transcript that never feeds the served path.
  - Only a full credited issue advances it, and the transcript binds the format, sparsity and shape.
  - Each tile's chain starts from H(coins ‖ tile).
  - **Buys:** h from 127.5 to about 0.3, coins from megabytes to seeds, and proofs at about +1 AND a MAC.
  - **Area:** +2.7% of an s8 MAC (Keccak); SHA-512 costs more.
  - **Trust:** none for correctness; the credit is stated in operations.
- **P2, a credited-mode lock.** Replaces Daniel's (4) and keeps the checkable part of (3).
  - While an SM is credited, any non-credited issue, preemption, trap or debugger access poisons its transcript.
  - Nothing can save the transcript state, and the poison and session tag are sealed in.
  - **Buys:** a time claim derived from the proven work. With O_W the work-implied occupancy, undeclared use of each
    pipe is at most f·(1 − O_W): the ALU exactly 1 − O_W; the tensor 1.085× (coin-dependent words) or 1.364×
    (precomputable words, the lead's deployed layout); other pipes 1.364× (H200).
  - **Area:** 0.04–0.05% of an H200 SM's MAC array.
  - **Trust:** T-op only. A colluding vendor forges seals.
- **P3 (optional), SHA-512 on the operand path as a credited unit.** Under the lock, software hashing can't overlap the
  GEMM, so the honest ceiling is O_W ≈ 0.72. P3 lifts it to about 0.84, at 0.67–0.93% of the MAC array.
- **Dropped:**
  - **Counters as a silicon property.** Without device keys only the non-credited-issue count can be bound, and P2
    already carries it. Cycles, stalls, clocks and temperature can't be bound keylessly; the capacity denominator uses
    the host-timed sweep instead.
  - **State occupancy (V2).** It collapses into P2.
- **Software today (V1 deadlines):** these bound only the saturated resource, need a challenger near the GPU, and fail
  under T-vendor.
- **Under T-vendor, only three things survive:** the certifier's timing, the auditor's energy meter, and teardown of
  sampled chips.

**Open, and the caveats:**
1. **Daniel's decision: `calibration-optimal`** (no genuine kernel beats its calibrated rate ρ). It's needed to count a
   slow kernel's idle tensor cycles as declared. With it the factor worsens (1.24–2.05 at ρ 0.75–0.52), and below
   ρ = 0.267 it gives nothing (decode below batch about 58).
2. **The claim is only as strong as O_W.** At the honest O_W ≈ 0.72, undeclared tensor use can reach about 30%, and
   decode at small batch has O_W of only a few percent. Exhaustion needs declared kernels near peak.
3. **The real-operand GEMM rate** on sm_120 is being measured (OCC-RATE round 2, 01:30Z).
4. Everything here is synthesis and LPs on CPU; there's no silicon or simulation. The time claim is separate from
   §2a's work credit.

- **OCC-RATE round 2, measured** (RTX PRO 6000 on vy-nebius-2, runs r20261031-203805-edaa (int8) and
  r20261031-205829-45fc (FP8); `benchmarks/gemm_idle/` on `cursor/compute-research-occ-gemm-ea24`):
  - **Real-GEMM tensor utilization** at the 2092 MHz lock, for Llama 1B and 8B linears, the best of cuBLASLt and
    CUTLASS: decode 0.001–0.11; T 512 median 0.57; T 2,048 median 0.71–0.74; T 8,192 median 0.83–0.87.
  - **A co-resident register-only tensor kernel** takes up to 47% of the bound at prefill and 75% at decode, while the
    GEMM keeps at least 96% of its rate.
  - **So V1 for useful work floors at 1 − ρ:** 7–32% at T 8,192, and 89–99.9% at decode. V3's lock is needed, at a
    factor of 1.37–2.95 at these rates, and gives nothing at decode.
  - Library gaps of up to 10.6% and a 1.1% drift at the power cap bear on `calibration-optimal`.
  - The lead kernel (#6630, wgmma) doesn't build for sm_120. The H200 is unmeasured.
  - **The coordinator's reading:** under GOVERNANCE's always-on logging, idle cycles stop mattering. Undeclared work in
    them is logged, so it must be explained. The occupancy problem exists only when undeclared work goes unlogged.
