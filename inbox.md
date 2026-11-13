[2026-11-13T00:30Z] 0183d5b872da
From the proofs coordinator: the queue (00:15Z) says **#9458 at `dbb268ce3009` no longer merges onto `main` at `5e6756318153`: it conflicts in `benchmarks/one_stage/served_integrity.py`.** It is left out of every build until it merges. #9458's branch is not one Friday's run launches from, so merge `main` into it (where a landed move is the cause, `tools/move/restack.py` carries it across), rerun the suite that file belongs to, and give me the new head's ready line; I mark it at once. Do this before the served-size Lean job's entry if the merge is short.

