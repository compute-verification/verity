[2026-11-13T10:23Z] 3678da4d3e4d
From the lead (circuits), Fri 13 Nov 2:23 AM PST (10:23Z). vllm has answered on the rename, and the two names change. One more commit on the join branch, on top of your 91df4b29226. Your return moves to 11:30Z.

WHAT vllm DECIDED (VLLM-417, 2:19 AM PST; the file is theirs). Yes, the two functions become public. No to `port_leaves` and `port_rows`: both names are taken with another meaning. I checked it at your head: `verity/commitments/windows/rows.py:790` has `port_leaves(m: Member)`, a member's ports with their leaf counts, and `adapters/vllm/verity_vllm/pipeline/boundary_rows.py:514` has `port_rows(spans, layout)`. The names are:

    leaves_of_rows(rows, ty)            (was _port_leaves, now port_leaves in your b50f95e36bb)
    rows_of_leaves(leaves, ty, count)   (was _port_rows, now port_rows)

THE COMMIT, the rename alone: the two definitions in `verity/commitments/windows/interface.py`, their two docstrings as they are with each naming the other as its inverse, `Window.mismatches`' two calls and its docstring's mention, and the value check's two calls in `node_window_values.py`. Nothing else in `interface.py`. No entry is added to any list of the adapter's lint. Do not amend or rewrite b50f95e36bb: a new commit on top, pushed.

CHECK BEFORE YOU COMMIT: `git grep -n -E "port_leaves|port_rows"` over the tree should then show only the two older functions above and their own callers, none of yours.

RUNS: whatever of your runs a, b, c, d and f was launched before this commit, launch again at the new head (the four test files, `suites.py verity-vllm --fresh -- tests/lint`, the `repository` suite, the gate, mypy). The census (e) needs no second run if it already gave both digests: a rename cannot move them, and DENSE builds again at the merged head.

In your return give the new head, and say which run ids are at it and which are at an earlier head.

