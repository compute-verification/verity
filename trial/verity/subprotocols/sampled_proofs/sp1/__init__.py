"""The SP1 construction: each drawn execution comes with a proof the auditor verifies. Heavy: its dependencies (here
`tomli_w`, standing in for the SP1 toolchain) come only with the `verity-sp1` distribution, whose `pyproject.toml` beside
this file declares them and registers the construction. Its source ships with `verity` and is never imported unless a
configuration names `sampled_proofs/sp1`.
"""
