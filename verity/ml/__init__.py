"""Python and CUDA: the provers, the Python versions of core's circuits (`circuits`, with `ops`, what models compute, and
`silicon`, what the GPUs actually compute), the commitments, crypto and randomness they use, and the kernels. What core's
verifiers read of it is on the soundness path (`SOUNDNESS_PATH` in `tests/test_boundaries.py`), and nothing here
imports `verity.core` but what `ML_IMPORTS_CORE` names."""
