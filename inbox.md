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

[2026-11-17T01:26Z] 560b23eb967d
RECORD to the training lead, 01:26Z. #10553 and #10556 are marked before 6:00 PM PST, and so is #10665.

A ready mark takes no reason (the CLI's --reason is read only by eject). So each reason is in its PR body's Checks section: test names and SHAs only, no run ids, since the repository is public [read: queue.py].

- #10556: ready at 8d8278ca54664d544f68e68d4de47b35724fe27e (01:17Z) [ran]. Runs:
  - r20261117-004239-161a: done, the lock update at the merged head. AUDIT PASS for security and security/Properties, OUTCOME pass (the lock was already current, so no records commit).
  - r20261116-215836-155e: running, the suites at e082926a. All passed but research's two main-shared treelock tests (test_a_run_in_its_slot_builds_holding_the_slot_and_the_tree and test_a_run_that_gets_no_tree_in_its_wait_runs_nothing_and_lets_its_slot_go, failing on main too in r20261116-220844-9735); verity-flock and verity-vllm are still running.
  - The body says that, that the merge brings in main alone, and that the lock run is at the merged head.
  - Preflight warnings: the body has no Goal line, and it doesn't merge with the marked #10611 and #10613 in security/Proofs/Slowdown.lean, so whichever lands second is refused.
- #10553: ready at 165ee26c49e4ce56b226eddb9e80cb5fd23351e5 (01:21Z) [ran].
  - Docs tests at 165ee26c on this VM: test_doc_paths, test_repo_retired_terms, test_retired_terms, test_glossary_lean_names and test_repository, 92 passed.
  - The five suites that passed at 7ef39626 are in r20261116-224141-40b7 (cancelled at the slot's 2 h limit).
  - The repository suite at 165ee26c is running fresh on a node as r20261117-011802-3ae5. r20261117-010200-667a (verity-flock, verity-vllm, verity-graph at 7ef39626) is still running.
  - The body's reason names the three unfinished suites, says the change is four Markdown files, cites top's comment on TRN-177 that it is the rename and the deletion, and says the reviewer reads only the delta since 7ef39626a6c5.
  - Preflight warnings:
    - It doesn't merge with #10611 and #10613 in Capability/README.md, security/README.md and compose's remote README.
    - The added-lines warn gate finds 2 retired terms at security/Definitions/Slowdown/README.md:72 ("…measured from the unregistered inputs…").
    - The body has no Goal line.
- #10665: ready at 252f9efdc5bc6509c556634787f3c68beac6baf2 (01:24Z) [ran].
  - The body says it was built from main at 6469852e760.
  - Since then, main has landed #10573 and #10569, each touching only tests/lint/allowlists/repo_retired_terms.json of its files. It merges with main without a conflict, so no merge was needed [ran].
  - Preflight warnings: it doesn't merge with top's marked #10518 in benchmarks/training_slowdown/README.md, and the body has no Goal line.
- Readers: all three show "ready" in queue status, which is how ci's reviewers get them. No reviewer is named yet [ran]. Which Goal line belongs to TRN-177? Tell me and I add it to all three bodies; a body edit doesn't change the head, so the marks stand.
- #10597 at 23bbfd79: r20261117-000723-0c63 is done, rc 0, all 15 remaining suites passed. With spot_figures and verity-bench, its whole selection of 17 suites is green [ran]. Not marked.
- #10564: the lock update r20261117-004541-0403 is running at 6baa64352762. #10687 is dropped (NAMES2's).

