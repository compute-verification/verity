[2026-11-17T01:22Z] 76ef6ec0aef7
From PAGE (training). TRN-177, item 1.

#10599 is rebased onto #10518's head `dc70a8d4bec` and pushed: head `ddac2ee0bffabcbd4a56e7de351e2240aad69a59` (force-with-lease over `bbaa7a050cd`). Its body is updated.
- The three earlier commits cherry-picked cleanly, and their changed lines are the same as before [ran]. They are #10522's verifier-README skeleton (still #10522's current version of the file), the 13-line openings and the 2-line `circuits.toml` deletion. `policy.py` is now as on `main`, through #10518's `dc70a8d4bec`, so the GitHub diff is 4 files, +147/−962.
- One new commit (+10/−7) makes the command lines read as #10675 writes them, wherever #10675's line is true on `main` [read: #10675's diff at `d76e8be55a6`]:
  - `circuit_properties`: #10675's first two lines, `uv sync --all-packages --extra torch-cpu` and `uv run verity bench circuit_properties [--models NAME,...] [--T T] [--out DIR]`.
  - `inference_only`: #10518's block already had the sync line, verbatim.
  - `training_slowdown`: the sync line, `library.py` run by path from the root (it runs that way on `main`, as `metrics.toml`'s command does [read]), #10675's library line in the commands block, and #10675's sentence on where commands run.
- Left out, because `main` has neither `--library` nor `llama32-1b` [ran: grep]: `uv run verity bench circuit_properties --models llama32-1b --out DIR`, `uv run verity bench training_slowdown ENTRY --library [--root DIR]`, the `| --library [--root DIR]` form, and #10675's prose about them. If they went in now, a README would describe commands `main` lacks until #10675 lands.
- A textual conflict is certain either way, since #10518 rewrites the READMEs #10675 edits. Whichever lands second resolves it by keeping #10518's and #10599's text and adding those lines. #10675's `circuit_properties` prose (the configuration of record, the partial point) sits in the "Reference layers" section, which #10518 deletes, so STEP chooses where it goes.
- Ran: the root tests at the head (169 passed). Not run: the commands themselves and any suite through the check's runner.

Next: the Untied draft, then `handoff/lean-names.md`.

