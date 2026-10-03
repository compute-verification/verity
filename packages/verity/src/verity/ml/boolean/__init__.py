"""verity.ml.boolean: word arithmetic as Boolean circuits.

    forms     GF(2) linear forms over committed bits and the circuit builder that counts ANDs (XOR and NOT are free)
    fp        exact f32 / bf16 arithmetic on words of forms (rounding, conversions, the tensor-core steps)
    tc_step   the finite tensor-core transition (`verity.ml.tc` GroupSum.step) that `fp`'s steps wrap
    softmax   attention's FTZ f32 operations, strict compare, max and -inf guard
    scalar    the bf16 comparator, the selects, PTX max / min and the E4M3 cast as Boolean Definitions (and names the IEEE f32
              arithmetic of `elementwise` and the bf16 widening of `cast`)
    fp4       the NVFP4 / MXFP4 block-scaled step (64 E2M1 codes and their scale bytes per operand)
    universal protocol 2's universal unit: one Definition per size class that runs any hidden netlist given as a value

These builders are what C-Flock's lowering has always run; `verity_flock.gf2`, `.fp`, `.unit` and `.unit_fp4` are these modules,
and `verity_flock.tail_pieces` imports `softmax`'s and `scalar`'s.
"""
