"""B-Ligero v2 units for the BF16 and FP8 tensor-core steps (lane relmin-lookup): the same relations as ``unit.py`` /
``fp8/relation.py`` -- the pinned instruction semantics of ``verity.ml.circuits.silicon.models`` -- compiled with the *public selection*
technique: everything the verifier can compute from the public operand words is a pin (the products, the group maximum of
the products, and the signed sum of the truncated products for every possible extra shift), so that the prover's witness
holds only what depends on the private accumulator: two shift amounts, the exact sum, and the normalised output.  See
``relation.py``."""
