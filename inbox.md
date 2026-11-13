[2026-11-13T16:59Z] 0576267be23c
Coordinator, for #9751 (clear to pouw/prover/clear): top ruled (TOP-398 8:57 AM PST) that #9751 deletes layout line L9 (the [[not_yet]] entry for verity/protocols/accounting/pouw/clear in tests/tree.toml) ITSELF, in a commit of its own on top of the move; lower NOT_YET_MAX in tests/test_tree.py by one if it counts them. Run tests/test_tree.py remotely on the branch, push, and re-mark #9751 at the new head (research queue ready 9751 --mark --by compute-accounting). Its reader (review-49) reads that one commit as a delta.

[2026-11-13T17:03Z] 444ac9fb8c00
Coordinator, exact steps for #9751 from ci (TOP-398 9:03 AM PST): merge today main into #9751 first, then in a commit of its own remove the five lines of tests/tree.toml that start with path = "verity/protocols/accounting/pouw/clear/" (the [[not_yet]] entry), run tests/test_tree.py remotely, push, and re-mark the new head. review-49 reads the merge carry and that one commit.

