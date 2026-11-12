[2026-11-12T23:43Z] 7130373fc93e
From the proofs coordinator. #9237 (`cursor/proofs-redraw-cc71` at `9d4aec0111fb`) no longer merges with `main`: one conflict, in `benchmarks/one_stage/a5.py` [GitHub's state of the pull request and ci's note of 2:40 PM PST; I did not run the merge].

ci advises taking `main` now, so that it is ready when the batch ahead of it lands (#9375's, about 9 to 10 PM PST tonight), and lands with NET-390's batch and #9075.

Do this before your other work: merge `origin/main` into the branch (never rebase), resolve `a5.py`, rerun what the merge touches (the one-stage suite's quick tier, and the Lean try or records run only if a Lean file or a lock moved), and write a ready line with the new head. I mark it at that head.

Second, from STAGECOINS's entry of 23:24Z (its NEEDS 5), for you and RUNCOINS: the run's coin check (`coinsReason`) reads the coins under the key the evidence carries and the family its segment names, while CUT's `firstDraws` reads them under the parameters' key. STAGECOINS needs RUNCOINS's `CheckReadsDraw` stated at the parameters' key: what the run accepted, `drawCoins` also accepts under the parameters' key, and the bytes read start its coins. Your change (the verifier's key into `coinsReason`) is what makes the two keys one. Say in your next entry whether, after your change, `coinsReason` reads under the parameters' key alone, and give RUNCOINS the lemma's name if you have one.

