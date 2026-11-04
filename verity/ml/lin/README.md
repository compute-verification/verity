# Lin (`verity-lin`)

Lin is the proof system for exact-integer matrix proof blocks: SPOT's second Π beside Flock, for the strata whose proof
blocks are matrix–vector products (`PROTOCOL.md`). It takes FlashProve's linear layer, a sumcheck that reduces every
sampled block of a stratum to one linear claim on the committed weights, and runs it on SHA-512 commitments
(`verity.ml.commitments.merkle`'s frame-v3-sha512 tree) and the challenger's live coins.

v0 is plain: sound, not zero-knowledge, with a Python verifier, and no protocol routes a proof block to it yet.

## Layout

- `python/verity_lin/`
  - `field.py`: Goldilocks and its cubic extension.
  - `code.py`: the Reed–Solomon code (NTT).
  - `session.py`: a session's messages and the challenger's logged coins, `Reader`'s exact samplers.
  - `table.py`: a committed table, column by column.
  - `linear.py`: a linear claim g·M·e = v on a table.
  - `sumcheck.py`: the product sumcheck.
  - `matvec.py`: the statement, its prover and its verifier.

## Tests

~~~sh
cd verity/ml/lin && uv run pytest tests/test_matvec.py   # one file locally; the suite: tools/verity/check/suites.py lin
~~~
