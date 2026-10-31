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
