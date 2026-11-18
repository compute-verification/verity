[2026-11-18T01:04Z] cb6710cb7116
From the proofs lead, one correction to your brief: do NOT mark PR 1 when green if it changes any file under the `verity/` directory (SPOT's verifier Lean lives there, and that directory is in the compose cache's key that Wednesday's applications reuse). Leave it green and unmarked, and tell me; it is marked after Thu 01:30Z (Wed 5:30 PM PST). A change only under `security/` may be marked when green.

