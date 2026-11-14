[2026-11-14T18:37Z] 03deb0f92c8c
circuits lead, Sat 14 Nov 18:40Z (10:40 AM PST). A NEW ORDER THAT REPLACES YOUR MARKING WORK. Daniel (TOP-424, 10:25 AM PST): the whole repository goes into the architecture's shape within 24 hours on one branch, `architecture`, tests after. Read TOP-424's description once (the Linear issue; read-only) before you act.

The branch's rules, all binding:
- `architecture` was cut from `main` 82f6b86e8f2d and is handed to every lane. To put work on it: `git fetch origin architecture`, merge `origin/architecture` into your branch (or your branch into a local `architecture`), and `git push origin HEAD:architecture`. NEVER rebase it, NEVER force. If the push is refused because the branch moved, fetch, merge and push again.
- There is no queue, mark, verdict or check on it. Before a push, every `.py` file you changed compiles: `python3 -m py_compile FILES`. Nothing else is asked: launch NO suites, NO runs, wait on none, and cancel any run of yours that only served a mark on `main`.
- Shared lists are NOT kept by hand on this branch: do not edit the lint allowlists' entries or counts, `tests/tree.toml`, any `lean-audit.json`, or a generated file. On a merge conflict in one, keep both sides' lines.
- No path move without the move tool (`tools/move/`); you move no path.
- `main` takes no such change from now. Mark nothing of this on `main`; a mark you hold on an unlanded change that goes to the branch is withdrawn (`research data label pr:N@FULLSHA ready false --by circuits`). You close no pull request.
- Unchanged: no subagents, timers or subscriptions; no secret, IP, machine name, hostname or URL in a commit or a comment; read the clock before writing a time; commit in small pieces and push after each.

YOURS: #10075 is on `main`. Merge #10160 (the checker's silent-excepts cures) and any other CIR-233 cure you have written into `architecture` now, as they stand, with no mark; cancel run r20261114-160918-db93 if it still waits. Finish the remaining CIR-233 cures directly on `architecture`. Then return with what you pushed and what is left.

