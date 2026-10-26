## 26 Oct 01:10Z, from the coordinator: two rulings from Daniel (25 Oct 5:59 PM PDT); the brief is updated (section 1, and section 5 adds PROTO-DESIGN and PEARLC-MODEL)

- **R-ACCEL-1, freed compute.** Work-soundness with parallel resources is: no strategy gets more credited work plus other work out of the period's devices than the honest optimum on the same devices, plus εC. The dual is a price vector. Adopted, though Daniel says it "may need to revisit over time", so keep the definition in one place.
- **R-ACCEL-2, the protocol may change.** "Yes may change Dither for sure ... we really need to choose some version of dither or maybe even something else such that we get provable security under our accelerator model." The deliverable is a co-designed model, protocol and proof.

**For you:** when you hit an obstacle (a threat you can't close for today's Dither), don't stop. Propose the smallest protocol change that removes it, with its honest overhead γ and its usefulness cost, then continue on the changed protocol and report both. PROTO-DESIGN (CMP-118) collects and ranks these. Pearl-C, whose noise sits inside the FP8 quantization, is being assessed by PEARLC-MODEL (CMP-119). Its bit-exact rounding kills Strassen but also kills rank counting, so if your route works without rank, say so.

## 26 Oct 01:42Z, from the coordinator: avoid "Activity task timed out"
Two ACCEL workers (O-IO and O-CHARGE) died 30–40 min into their first turn with "Activity task timed out", before pushing or writing a note, so their work was lost. Do these:
- **If you haven't yet,** write and sync your note now, even if it only holds your plan and findings so far, and sync it again at every result.
- **Keep every tool call short** (well under 10 min).
- **Read big files in slices.**
- **Push your branch early.**
- **End your turn** rather than run one very long step.

## 26 Oct 01:42Z, from the coordinator: what the old "¼ without CoincCharged" was
O-CHARGE found it in P2-FF r7/r8 (`lanes/compute-accounting/20261016T0724Z-draft-fullfloor-s-r7`, `20261016T0917Z-draft-fullfloor-s-r8`). The old Theorem O was output counting: every instruction that writes a 32-bit word pays at least 32 MACs, except copies. The ¼ is a price ratio, an LDS.128 copy at 8 MACs a word against 32 for any head at VF3e, not a register-file bandwidth bound. Your 4(g) analysis should start from it and say what changes at data-sheet rates.
