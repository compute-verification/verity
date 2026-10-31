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
