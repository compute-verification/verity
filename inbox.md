[2026-11-17T03:09Z] 18fef8da3b7b
RECORD to the training lead, 03:09Z.

- #10553: r20261117-010200-667a (the three suites at 7ef39626) is done, failed.
  - verity-flock 2164 passed (69 skipped) and verity-graph 248 passed [ran].
  - verity-vllm: 24021 passed and 2 failed, both in protocols/test_credit_gate.py: test_a_period_open_without_draw_k_is_refused and test_a_window_presented_outside_its_periods_close_or_out_of_order_is_refused. Both fail the same way on main at 6e3d3208536 on this VM, raising NotImplementedError "compute-accounting/pouw-verdict-from-main" [ran]. They are main's and compute-accounting's, not this Markdown change's.
  - So every suite #10553 selects has now passed at 7ef39626 or 165ee26c, except those two main-shared tests.
  - Its two phrases are left as you say.
- Main moved to 6e3d3208536 (#10650, #10681, and #10659, which deletes the two treelock tests). All four of our marked heads still merge with it cleanly [ran: git merge-tree]: #10553 165ee26c, #10556 8d8278ca, #10665 f1ec87b9, #10619 03b32d55.
- #10704, #10611 and our marked PRs were all open as of 03:08Z. The #10553 merge is prepared (48f52ff56da6) and waits for #10704.

