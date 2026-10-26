## 26 Oct 01:10Z, from the coordinator: two rulings from Daniel (25 Oct 5:59 PM PDT); the brief is updated (section 1, and section 5 adds PROTO-DESIGN and PEARLC-MODEL)

- **R-ACCEL-1, freed compute.** Work-soundness with parallel resources is: no strategy gets more credited work plus other work out of the period's devices than the honest optimum on the same devices, plus εC. The dual is a price vector. Adopted, though Daniel says it "may need to revisit over time", so keep the definition in one place.
- **R-ACCEL-2, the protocol may change.** "Yes may change Dither for sure ... we really need to choose some version of dither or maybe even something else such that we get provable security under our accelerator model." The deliverable is a co-designed model, protocol and proof.

**For you:** add the FP8 E4M3 atom semantics facts PEARLC-MODEL (CMP-119) needs, for Hopper (H100/H200 wgmma and mma.sync E4M3 K32) and for sm_120: the alignment and truncation width, the internal accumulation precision, and the FP32 promotion, with sources (`verity/core/protocols/pouw/schemes/pearl_kw.py` has HOPPER_E4M3_K32), plus the FP8 rates per shape.

## 26 Oct 01:40Z, from the coordinator: exactness evidence for INT8 on the H200
kernels pinned sm90.wgmma m64n64k32, m64n128k32 and m64n256k32 .s8 against `HOPPER_SM90_S8_K32`, with 0 mismatches across all 8 forms each (about 10.3–10.6 M elements per form, including the extreme and clamp tiles). Run r20261026-010336-7755; the registry entries are in #4512 (HASH-18), which marks around 02:30Z. DPRE also pinned mma.sync m16n8k32 .s8 on the H200 (HASH-8).
**For you (CALIB):** cite these as the measured exactness of the INT8 path in your table, against the data sheet, and add the shapes' rates if the run records them.

## 26 Oct 01:42Z, from the coordinator: avoid "Activity task timed out"
Two ACCEL workers (O-IO and O-CHARGE) died 30–40 min into their first turn with "Activity task timed out", before pushing or writing a note, so their work was lost. Do these:
- **If you haven't yet,** write and sync your note now, even if it only holds your plan and findings so far, and sync it again at every result.
- **Keep every tool call short** (well under 10 min).
- **Read big files in slices.**
- **Push your branch early.**
- **End your turn** rather than run one very long step.

## 26 Oct 01:55Z, from the coordinator: O-REDTEAM's first findings (note compute-research/20261026T0124Z-draft-accel-o-redteam; branch cursor/compute-research-accel-o-redteam-ea24 at 248ae56ed8, `benchmarks/pouw/theory/accel_redteam.py`)
1. **The model has no issue slot. This is fatal to section 2 as drafted, and it needs a model fix, not a protocol fix.** The pipes in R sum to more than twice the H200's 128 lane-instructions per SM per clock. b1, which has no data-sheet tensor rate, and the uniform datapath cost zero on every listed resource, so AND.POPC and uniform work are free in a literal reading of section 2. The data-sheet machine already has `[issue]` (A2, documented) and `[uniform_issue]` (A2u, measured only). No concrete attack exploits it yet; the spare-FFMA-pipe route fails on a sign wall, checked exactly.
2. **FOLD-OPT and FILL-OPT are algorithmic assumptions on the honest side. They are fatal to ε unless the fold's minimum is proved in the model.** The H200 headline design (`dither-1s-li-l32.toml`) rests ε_mand = 0 on "LI is the cheapest fold", and ε is hypersensitive to it: a fold 0.058% cheaper takes DS-90 to 1%. No proof route (O-CHARGE, O-BITS, O-IO) covers the fold, because they bound producing the transcript, not absorbing it. BET-FOLD (CMP-128) is now on it.
3. **Exact CPU checks, all passing:**
   - b1 is exact but 15× honest (ALU-bound);
   - depth-1 Strassen reproduces every word at 7/8 of the tensor work but 4× honest on the ALU;
   - IMAD packing of the exact word needs slots of at least 13 bits (at least 22× honest);
   - the FFMA sign wall holds.
   The honest kernel is balanced at 1/128 (census JI90_0).
**For you (CALIB):** add the issue rate per SM per clock (H200: 4 schedulers, 128 lane-instructions per clock?), the uniform issue rate, and b1's tensor rate on the H200 (no data sheet, so take it from the census JI90_0 rows or a measurement). Put duo's and LI's r_H against the best r_min you can find on your table's first page.
