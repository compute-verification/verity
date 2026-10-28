"""Python and CUDA: the provers, the Python versions of core's circuits (`circuits`, with `ops`, what models compute, and
`silicon`, what the GPUs actually compute), the commitments, crypto and randomness they use, the kernels, and the
circuits core's protocols' provers build (`protocols`), which their verifiers don't read. Lean is the definition and the
verifier of record, and the Python here mirrors it for the prover's speed, held to it by difftest. A verifier imports none of it; until each protocol's verifier of record is Lean, what its Python verifier reads
of it is on the soundness path (`SOUNDNESS_PATH` in `tests/test_boundaries.py`), which only shrinks. Nothing here imports
`verity.core` but what `ML_IMPORTS_CORE` names."""
