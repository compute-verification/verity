"""Python and CUDA: the provers, the Python versions of core's circuits (`circuits`, with `ops`, what models compute, and
`silicon`, what the GPUs actually compute), the commitments, crypto and randomness they use, and the kernels. Nothing here
affects soundness, and nothing here imports `verity.experimental` but C-Flock's tests, which `tests/test_boundaries.py`
names."""
