[2026-11-15T21:40Z] b4903ecc5694
From your lead: the queue says your branch's bases #10354 and #10364 landed on main at 528eba7df639. Merge origin/main into your branch before your next push (first check `git log --grep="^Move:" --grep="^Layout-move:" <your merge base>..origin/main`: if a move commit lies between, carry with `python3 tools/move/restack.py BRANCH --main MOVE` first, then merge; a plain merge across a move is clean and wrong). Rerun the suites on the merged head.

