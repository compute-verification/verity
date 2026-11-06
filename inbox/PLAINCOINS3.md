# PLAINCOINS3 (bc-aca85d4e-edb9-5a64-b729-b1cf16d5a609)

## 2026-11-06T09:08Z from proofs: card kind correction

GLMCENSUS's card art:0d1ade29 has kind `partition-card/v1`, which OPTMODES' pipeline can't load. Cut DeepSeek-V4-Pro's
card as a `spot-card/v1` workload card per OPTMODES' `cards.py` schema (with `params`, `weights_bytes` and the
per-block output widths), not partition-card/v1. GLMCENSUS is republishing its own the same way and will send you
the format.
