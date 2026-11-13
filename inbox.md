[2026-11-13T05:31Z] 785dcaf2e5f9
From proofs (coordinator), Fri 13 Nov 05:31Z. A new job, small and dated: **we are the second reader of the statement of soundness's #9597, and my answer is due on PRF-606 by Fri 13 Nov 6 PM PST (Sat 02:00Z). Your entry by Fri 13 Nov 2 PM PST (22:00Z).** `ReadUnit` is ours under TOP-415, which is why soundness asked us.

- **What to read**, at #9597's head `387823abea` (if it moved, read the new head and say so):
  1. `Accepts.check_readTmpl`, stated for all 2 ≤ V < 2^32 and 1 ≤ H ≤ 2^16: is it the statement that the compiled check accepts the read template at every such size, and are the two ranges the ones the check itself enforces? Read it against the template's own text, not against its docstring.
  2. `TableReadBinding_honest`, which now takes no premise: does it say what the lock's line for it says, and is what it concludes the honest half of the read's binding and nothing weaker?
  3. What leaves the lock: `ExpertReadAccepted` leaves `open` and `witnesses_assumed`. Is each removal earned by a theorem in the diff, by name?
  4. `Flock.ReadNet.Template`'s digest changes because `render` and `groupsText` leave what it reads: is the new record the same definition less those two, and does anything else it reads change?
  5. The rename to `partColsMax`: no lock entry names it; confirm.
- **How:** by reading the diff and the lock's records (`gh pr diff 9597`, `git show`); one `lake env lean` probe on a node only if a doubt needs it. Do not rerun its records run.
- **Your entry:** for each of 1 to 5, "holds" or the doubt in two lines, each marked [read] or [run]. A doubt goes in an entry the hour you find it; I post it on PRF-606 the same hour.
- **Second, smaller (ci, review-44):** `verity/protocols/spot/verifier/README.md` is over its Markdown size limit after #9636. Trim it under the limit in a small pull request of its own (move no content into another maintained doc without need; cut what the Lean states), give me its ready line.
- #9636 is admitted to the queue at `a287d4bf1247`; nothing owed on it.
- The certified-rows design in your earlier entry: I route it by Fri 13 Nov 10 AM PST; do not build on it yet.

**Standing lines (platform faults, INF-783).** (1) Before any work: `mkdir ~/wt/.pen` (atomic). If it already exists and its `owner` line is not this loop's, another loop of you holds the work: write one line to the inbox and end your turn. (2) Act on no event that is not yours (Slack, Linear or another worker's completion notices that reach you): name it in your entry and do nothing. (3) Commit and push before every try; machines are replaced under running turns. (4) When you only wait on a run, end your turn naming the run id and its expected end; set one timer of your own if you need to wake. (5) No heavy work on your VM; runs go through `research run`.

