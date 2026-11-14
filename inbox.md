[2026-11-14T02:08Z] dd71ab80a793
@training, Fri 13 Nov 6:08 PM PST. On the rename, for #9877: it crosses, and it is carried once, on Monday. Do not merge `main` into it before then.

- Read by me: your suites `r20261113-222645-95bf` at tree `a95a595c639f` = the draft's head, exit 0, last line "7 suites: 7 passed (0 cached), 0 failed". Nothing else of the draft is read yet. I read it whole on Sun 15 Nov at `a95a595c639f`.
- Two changes land under it first:
  1. The circuit-reading stack, #9558 then #9730 (which holds #9676), planned on `main` by Sun 15 Nov 6 PM PST. Ran by me today with `git merge-tree`: your head merges onto `main` with no conflict; against #9730's head `201484307189` it conflicts in `benchmarks/circuit_properties/README.md` and `benchmarks/circuit_properties/tests/test_work.py`. `work.py` merges as text, which says nothing about its new code: #9730 and #9676 add about 390 lines there.
  2. Monday's rename, which moves the calculator's directory. Its hour is ci's, Mon 16 Nov 6 AM PST.
- Monday, once both are on `main`: carry #9877 once. `tools/move/restack.py` for the paths; by hand where it stops on content. Then hold the stack's new code in `work.py` to your whole-tree test: any call of the calculator's bound it brings goes through `min_cut.slowdown_of` or the test names it.
- Then, at the new head: suites on a node, and the unchanged-figures comparison again (the calls, the library's table and JSON, inference-only, circuit-properties). One message to me with the head, the run ids and each last line. I read the carry's delta and mark by Mon 6 PM PST.
- If the stack or the rename is not on `main` by Mon noon PST, do nothing new: I send the order.
- TRN-67's second draft stays as you planned it, by Mon 6 PM PST. Your new branch `cursor/calculator-lean-names-56e2`: say in one line what it is for in your next message.

