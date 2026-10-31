# GOVERNANCE: always-on compute logging, verified by inspectors against an adversarial chip designer

Daniel, 31 Oct 1:50 PM PDT, paraphrased:
- No credited mode. **100% of computation** goes through the transcript, as part of a global compute-governance regime.
- It must be **physically verifiable by random inspection and teardown**: not trusting the silicon, but verifying it.
- The chip designer is adversarial but has agreed to the plan.
- "Does something like this seem feasible? What properties does it give us, and how would international inspectors
  verify it?"

## The coordinator's framing (to confirm or break)

**Always-on logging.** Every arithmetic result the chip produces, tensor and other pipes, advances a transcript the
chip can't skip:
- MMA accumulator words absorbed as in FT (`inbox/SILICON.md`);
- other instruction classes at least counted, or hashed where it's cheap.

The transcript:
- depends on live randomness from the inspector's on-site device, so it can't be precomputed or replayed;
- leaves the chip only through a physical port wired to that device. There are no device keys; binding is physical
  and tamper-evident.

The operator must explain the transcript: sampled zero-knowledge proofs that each sampled segment belongs to a
registered, declared workload. Unexplained segments are violations.

**The claim:** undeclared compute ≤ ε (sampling, at confidence δ) + whatever capacity bypasses the logging physically.
Inspection bounds the second term.

**What inspectors do:**
- **Before deployment:** review of the agreed design (RTL, netlist, layout); mask audit at the fab; teardown of sampled
  wafers.
- **In deployment:**
  - inventory (every chip present, by serial and by location);
  - tamper-evident seals on each chip's transcript port and its wiring;
  - unannounced random pulls of deployed chips for teardown and conformance tests (known-answer programs, transcript
    completeness);
  - an energy cross-check (metered rack power against the transcript-implied work).
- **Ongoing:** verifying the sampled proofs.

**Why sampled teardown might work:** GPU compute is a regular array of identical SMs and MAC arrays. A bypass worth
anything must be replicated across many of them, so randomly inspecting a few SMs on randomly pulled chips catches it
with high probability, if logging is structurally unavoidable on the accumulator writeback path and visible in the
layout.

## Lanes (notes: `lanes/compute-research/<ts>-report-governance-<lane>.md`, due Sun 1 Nov 04:00Z)

- **GOV-HW, the always-on logging design.**
  - What exactly must log: tensor MMA words; ALU, FP32, FP64 and SFU ops (count against hash); memory-side compute;
    copy engines; the on-package CPU.
  - Where the transcript unit sits, so logging is **structurally unavoidable** (on the only write path from each MAC
    array to the register file).
  - The area, throughput and energy cost on an H200-class die, from SILICON's netlists.
  - The transcript's output rate and the sideband port.
  - What emulation routes remain: matmul on FP32 FMA, on IDP4A, on memory-side logic. Price each as leakage if
    non-tensor ops are only counted.
- **GOV-INSPECT, physical verification against an adversarial designer.**
  - The literature: hardware Trojan detection and its limits at 4 nm; SEM delayering cost and time; split
    manufacturing; mask audits; "hardware-enabled governance mechanisms" (CNAS "Secure, Governable Chips", flexHEG,
    RAND); IAEA safeguards practice (material accountancy, containment and surveillance, seals).
  - The sampling statistics: chips pulled × SMs imaged against bypass replication, and detection probability against
    cost.
  - What an adversarial designer can still hide (a hidden datapath, a trigger-activated bypass, off-die compute on the
    module or board), and which inspection catches each.
  - Energy accounting's power to detect hidden compute.
  - What has to be disclosed to inspectors, and what can stay the vendor's.
- **GOV-CLAIM, the end-to-end statement.**
  - The guarantee: units, ε, δ, and which terms come from sampling, inspection and inventory.
  - The proving burden for 100% coverage with sampled ZK proofs, and the operator's retention needs.
  - Privacy (weights and data stay hidden).
  - What fails and how it shows: a missed bypass, seal tampering, a chip swap.
  - A comparison with today's software PoUW (the lead) and with the credited-mode lock.

**Rules:** mark everything read, argued, estimated or measured. No Slack, Linear or PR. The coordinator synthesizes for
Daniel by 05:00Z Sun (10 PM PDT).

## Facts feeding the lanes

- **OCC-RATE measured** (sm_120) that real GEMMs leave 13–43% of the tensor pipe idle at prefill and nearly all of it at
  decode, and a co-resident kernel can use it while every deadline holds. Always-on logging sidesteps this: undeclared
  work in idle cycles is in the transcript, so it must be explained. The GOV lanes should state that advantage over
  the lock and deadline designs, and its premise: every tensor issue is logged.
- **GOV-HW** (note:compute-research/20261031T2151Z-report-governance-hw; SILICON's flow, netlists matched to their
  references; art:bb827600…):
  - **Always-on logging is buildable at under 1% of an H200-class die** (central 0.88%, range 0.39–2.06%), with no clock
    cost.
  - **It must hash every register-file write, not only the MMA words.** Counting the non-tensor pipes leaks 7.6–10.6%
    of s8 tensor capacity on the H200 (3.1% beside saturated declared work) and 37–49% on sm_120. Counts suffice only
    for add-only memory-side units and the copy engines.
  - **The keyed absorb binding fix** measures 63,047 GE, 7.53% of an s8 MAC (FT is 2.71%). It fits one clock and adds
    about 9% MAC toggle energy.
  - **Power:** at most about 56 W (tensor tap) and 60 W (vector tap) on the H200, as upper bounds.
  - **Placement:** in-line on each tensor core's write-back, and beside each sub-partition's register file with paired
    counters. There's no enable or mode bit, the coin register is write-only, and it's eight blocks an SM.
  - **Binding without keys:** a hidden live coin, the keyed absorb, and dedicated pins on a sealed link, with seals
    covering both directions. Swapping in another chip's transcript reduces to inventory.
  - **Decision for Daniel:** the keyed-Keccak premise.

## §synthesis (coordinator, 22:35Z Sat 31 Oct)

Sources (all three final):
- note:compute-research/20261031T2151Z-report-governance-hw
- note:compute-research/20261031T2059Z-report-governance-inspect
- note:compute-research/20261031T2132Z-report-governance-claim

**Verdict: feasible in principle.** The hardware is cheap, and the proofs are cheap. The bound is set by inspection and
by the host CPUs.

**The guarantee** (GOV-CLAIM): fleet-wide undeclared operations ≤ ε_s + ε_b + ε_e + ε_o, except with δ ≈ 10⁻³. For
100,000 H200-class GPUs at 50% load over a year:

| term | share of capacity | what sets it |
|---|---|---|
| ε_s, sampling | 1.4·10⁻⁶ | 27,712 draws a day |
| ε_b, an altered chip missed by teardown | 1.0% | 762–1,374 teardowns a year, $97–160M |
| ε_e, emulation on add-only memory units | < 0.1% | GOV-HW's tap on every register-file write |
| ε_o, off-chip compute (host CPUs, Grace) | 1.3–2.9% | not logged |
| **total** | **2.3–4.0%** | 1.3–3.0% with a fab-custody premise instead of sampled teardown |

Proving costs at most 0.5 GPU-hours and 3–8 CPU-minutes of Lean verification a day.

**The hardware** (GOV-HW): under 1% of the die (central 0.88%), with no clock cost. It must hash *every register-file
write*, not only the MMA words; counting the other pipes leaks 7.6–10.6% on the H200 and 37–49% on sm_120.
- **Binding is keyless:** a hidden live coin, a keyed absorb (7.53% of an s8 MAC), and dedicated pins on a sealed link
  with seals in both directions.
- **Nothing to switch off:** there is no mode or enable bit, and the coin register is write-only.

**Inspection is two jobs** (GOV-INSPECT):
1. **Design review:** is the disclosed design honest? It's exhaustive and computational, with no δ, and it's the single
   point of failure. GOV-HW's placement makes most of it structural, by requiring every driver of every register-file
   write to be a transcript output. The whole-die GDS and netlist go to an inspector-run information-barrier enclave.
   Human-readable disclosure covers only the governance blocks.
2. **Silicon conformance:** are the deployed chips that design? Masks against GDS, and sampled GDS-versus-SEM
   teardown.
   - Chips are the unit: hiding a bypass on part of the fleet needs a second mask set, so the cheater's best split is
     whole chips. Inspectors need ln(1/δ)/(c·q) chips a year, almost independent of fleet size.
   - **Cost:** $26M, $45M and $71M a year at 50%, 90% and 99% detection of a 1% bypass, or 3–8% of fleet cost.
     Sub-percent assurance needs fab custody instead.
   - **Triggered bypasses** defeat conformance tests but not imaging or formal review. **Dopant and analog Trojans** are
     the hard case; 100% stuck-at test coverage of the tap and contact-layer voltage contrast mitigate them.
   - **Energy metering** is a weak backstop, at best 5–20%.

**New burden:** with the register-file tap, a drawn segment is every register write of one warp, so honest operators
must replay it bit-exactly from recorded nondeterministic inputs. That affects pass rates, not soundness.

**Privacy gap:** the inspectors' device sees drawn segments in the clear, which can include weights. A two-party check
fixes that at about 6.7 GB of garbled circuit a draw.

**Decisions for Daniel:**
1. The keyed-Keccak premise. GOV recommends it; SHA-512 keeps the one-hash rule at about 2.5× the area.
2. Port seals in both directions, the weakest premise, with no public defeat rate against a state.
3. Sampled teardown against fab custody. Recommended: state both, and use custody plus deterrence teardowns.
4. Host CPUs as the stated floor (1.3–2.9%), or extend logging to them.
5. Privacy: accept clear segments at the inspectors' device, or pay for the two-party check.

**Smell check:**
- No teardown at 3–5 nm was measured; every teardown time and cost is an estimate from read figures.
- The catch rate q is unmeasured (it needs 29 blind trials per modification class).
- The area figures are generic-library netlists.
- The energy bound uses one H200 sweep.
