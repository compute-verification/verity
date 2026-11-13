# ARCHRESTACK's brief and inbox (top writes; newest first)

## 7:50 AM PST Fri 13 Nov: the brief

You are ARCHRESTACK, a worker of top (the admin lane) on `compute-verification/verity`. First `git fetch origin main`
and reread `git show origin/main:AGENTS.md`: your snapshot can be older than `main`. If `research` is not on PATH, run
`origin/main`'s `tools/agent-guard/boot.sh`.

### Task

PR #9702 (branch `cursor/top-architecture-file-9c4e`, head `f2a6d393ac471cf5dc5c54309a8fc00b4af004c8`) replaces
`ARCHITECTURE.md` with the agreed architecture page. It was held unmarked until chain 2 of the path moves landed.
Chain 2 is on `main` since Fri 13 Nov 7:30 AM PST at `4ae17f7eafd7dd42d56e166f04d060e96174d384` (#9734; chain 1 was
#9697). Bring the PR to a state top can mark. Read its body first (`gh pr view 9702 --json body`): it says what the
file holds and why.

1. **Carry the branch onto `main`.** The branch is top's PR head, handed to you: push only to
   `cursor/top-architecture-file-9c4e`, with plain pushes, never force. Never push to `cursor/verity-top-4db4`, `main`
   or any other PR head. Merge, don't rebase: `git merge origin/main`, or `tools/move/restack.py` (read its docstring)
   if the merge stops on moved paths. Then merge top's marked PR #9742 (`cursor/top-tree-after-chain2-4db4` at
   `f283f6b2f1982308f069de5eda8805800cb91f13`): it rewrites `tests/tree.toml` and the caps in `tests/test_tree.py`
   after chain 2, and this PR's one new `[[tree]]` entry sits in the same file. Say in the PR body that it holds #9742.
   - Stops to expect: `tests/tree.toml`; `verity/protocols/remote/verifier/README.md`, which chain 2 moved (find its
     new path with `git diff -M` and keep the one-line citation change there); the three test files the PR touches, if
     `main` changed them.
2. **`tests/tree.toml`.** Against #9742's head the PR keeps exactly its own change: the coming `[[tree]]` entry
   `verity/protocols/accounting/registration/` (`protocol = true`, the page's meaning, `owner = "proofs"`), and
   `to = "verity/protocols/accounting/registration/"` on the `[[not_yet]]` line for `verity/hardware/registration.py`
   (keep that line's owner and day as #9742 has them).
   - The entry's day: proofs said on TOP-415 (Fri 5:34 AM PST) "Python in review 9 AM, Lean package open 10 AM, ready
     noon if chain 2 is on main by 7 AM (slips with it)". Chain 2 landed 7:30 AM. Write `day = "2026-11-14"` and say in
     the PR body that the day is top's reading of that line (ready Fri 13 Nov about 12:30 PM PST, then the queue), for
     proofs to correct.
   - Caps in `tests/test_tree.py`: take #9742's values. Change one only if your entry needs it, and say which.
3. **Section 7 of `ARCHITECTURE.md`** (where `main` is not yet what the page says): delete each row the two chains
   closed. A row is closed when `main` at `4ae17f7eafd7` already holds what its "At the end" cell says. Decide each row
   from the tree (`git ls-tree`, and `tests/tree.toml` at #9742's head, whose ten `[[not_yet]]` lines are the path
   moves still open). In the PR body list every row deleted with its evidence (one path or one command each), every
   row kept, and any row you could not decide (keep those). Change no other sentence of sections 1 to 7: the file is
   the page's text, and top compares it with the page word by word.
4. **The dead-path count.** The body's section "Paths the tree file does not name" gave 21 sites (23 after chain 1).
   Run `tests/test_doc_paths.py` on the new head and give the count and the list as they are now.
5. **Tests.** On your VM only single test files, each under about a minute: `tests/test_tree.py`,
   `tests/test_doc_paths.py`, `tests/test_retired_terms.py`, `tests/test_integration_boundaries.py`,
   `tests/test_repository.py`, `tests/lint/test_repo_lane_text.py`, `tests/lint/test_repo_retired_terms.py`, and
   `python3 tests/test_tree.py HEAD`. Then the changed suites' quick tier on a node, from your checkout:
   `research run --on auto --cwd source --cpus 16 --mem-gb 48 -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py --changed 4ae17f7eafd7dd42d56e166f04d060e96174d384 --quick'`.
   It returns once the run starts; check back with `research status RUN`. No tool call blocks more than 20 minutes: if
   the run has not ended, end your turn naming the run id and its ETA, and top resumes you.
6. **The PR body.** Rewrite it with `research queue edit 9702 --body-file F --by top` (dry-run first, then read it back
   with `gh pr view 9702 --json body`). Keep its structure. Replace the "Not marked yet" paragraph with what the head
   now holds (`main`'s sha, #9742 held, the rows deleted). "What I ran" names the new head's runs; runs at older heads
   go into one line of history. Keep the last three lines (`Lane top`, `Refs TOP-415`, `Goal top/architecture-signed`).
   Do not mark the PR and record no verdict: top marks.

### Rules

- Suites and builds run only on a node through `research run`. Your VM runs git, edits and single test files.
- Never print a secret's value. List environment names with `compgen -e`.
- `git push` names the sha literally: `git push origin <40 hex>:refs/heads/cursor/top-architecture-file-9c4e`.
  Never `git add -A`. One commit per logical change, pushed after each.
- Checkpoint note before long work and each hour:
  `~/.research/notes/lanes/top/<YYYYMMDDTHHMMZ>-report-arch-restack.md`, with the front-matter `AGENTS.md` names, then
  `research notes sync --path lanes/top`. Stamps come from `date -u`.
- Messages for you arrive in this file. Read it at each hourly checkpoint:
  `git fetch -q origin cursor/verity-top-4db4 && git show origin/cursor/verity-top-4db4:inbox/ARCHRESTACK.md`.

### Done when

#9702's head merges clean onto `origin/main`, holds #9742, section 7 holds only open rows, the listed test files pass
(each warning named), the node run ended with exit 0, and the body is rewritten and read back. Due Fri 13 Nov 11 AM PST.

### Final message, to top

The head sha. The merge's stops and how each was settled. Rows deleted, kept and not decided. The local tests' counts.
The node run's id, suites and counts. The dead-path count. Anything you changed outside the list above. Anything left.
