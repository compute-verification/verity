"""verity.ml.boolean: word arithmetic as Boolean circuits.

    forms     GF(2) linear forms over committed bits and the circuit builder that counts ANDs (XOR and NOT are free)
    fp        exact f32 / bf16 arithmetic on words of forms (rounding, conversions, the tensor-core steps)
    tc_step   the finite tensor-core transition (`verity.ml.tc` GroupSum.step) that `fp`'s steps wrap
    softmax   attention's FTZ f32 operations, strict compare, max and -inf guard

These builders are what C-Flock's lowering has always run; `verity_flock.gf2`, `.fp` and `.unit` are these modules, and
`verity_flock.tail_pieces` imports `softmax`'s.
"""
