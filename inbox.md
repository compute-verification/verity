[2026-11-12T22:14Z] 4e4f8a5ffb9c
LOG, Thu 12 Nov, 2:14 PM PST

**#9039's head is `ca6c4c2a5483` (pushed, no force), carrying MF1 and SF1 to SF3 and one merge of `main`, and every suite it touches passes there.**

- RAN (r20261112-215150-bce5, at `ca6c4c2a54`, base `5f0c008aa040`, `--changed 5f0c008a --fresh`): `4 suites: 4 passed (0 cached), 0 failed` (repository 356, grader 31, training_slowdown 623, verity-graph 246); the grader's slowdown-code tests `6 passed`; ruff `All checks passed!`; last line `suites rc 0 grader rc 0 ruff rc 0`.
- READ (git log): the commits on top of `5c793f5ee5`:
  - `34cfa7ec0b` SF2: `gates.row_dtype`, one reader that raises.
  - `84682f825b` SF1: `fp32_precision` first, then the legacy pair.
  - `c84bf35f2d` MF1: one way out, plus the M caveat in the docstring and README, the `adapted` test and the no-trace test rewritten.
  - `2bca248b2c` SF3: in-place ops named by torch's promotion.
  - `ca6c4c2a54` the merge.
- READ: the merge commit's subject line carries a `Co-authored-by` trailer a commit hook appended. It's cosmetic and stays, since rewriting it would need a force-push.
- SF4 and the SF3 line in the body wait on nine pair runs at `ca6c4c2a54`. One has ended (rc 0), two are running and six are waiting for a slot. I set the body when they end and send Shampoo's p again if it moved.
- QUERY: this is the head to merge once more.

