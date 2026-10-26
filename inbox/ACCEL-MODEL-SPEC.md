## 26 Oct 01:10Z, from the coordinator: two rulings from Daniel (25 Oct 5:59 PM PDT); the brief is updated (section 1, and section 5 adds PROTO-DESIGN and PEARLC-MODEL)

- **R-ACCEL-1, freed compute.** Work-soundness with parallel resources is: no strategy gets more credited work plus other work out of the period's devices than the honest optimum on the same devices, plus εC. The dual is a price vector. Adopted, though Daniel says it "may need to revisit over time", so keep the definition in one place.
- **R-ACCEL-2, the protocol may change.** "Yes may change Dither for sure ... we really need to choose some version of dither or maybe even something else such that we get provable security under our accelerator model." The deliverable is a co-designed model, protocol and proof.

**For you:** no change to your task. Make sure P1 covers FP8 E4M3 tensor-core atoms (Hopper's E4M3 K32 atom with its alignment truncation, in `pearl_kw.py`, and sm_120's) as well as INT8 and BF16, since PEARLC-MODEL (CMP-119) will instantiate the model at Pearl-C.

## 26 Oct 01:40Z, from the coordinator: exactness evidence for INT8 on the H200
kernels pinned sm90.wgmma m64n64k32, m64n128k32 and m64n256k32 .s8 against `HOPPER_SM90_S8_K32`, with 0 mismatches across all 8 forms each (about 10.3–10.6 M elements per form, including the extreme and clamp tiles). Run r20261026-010336-7755; the registry entries are in #4512 (HASH-18), which marks around 02:30Z. DPRE also pinned mma.sync m16n8k32 .s8 on the H200 (HASH-8).
**For you (MODEL-SPEC):** P1's INT8 exact domain on the H200 instance can cite this evidence. Name the pinned semantics (`HOPPER_SM90_S8_K32`), not a prose claim.

## 26 Oct 01:42Z, from the coordinator: avoid "Activity task timed out"
Two ACCEL workers (O-IO and O-CHARGE) died 30–40 min into their first turn with "Activity task timed out", before pushing or writing a note, so their work was lost. Do these:
- **If you haven't yet,** write and sync your note now, even if it only holds your plan and findings so far, and sync it again at every result.
- **Keep every tool call short** (well under 10 min).
- **Read big files in slices.**
- **Push your branch early.**
- **End your turn** rather than run one very long step.
