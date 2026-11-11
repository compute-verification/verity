[2026-11-11T06:02Z] 2d970449d2e0
coordinator 06:02Z: REGX's part 2 is pushed: cross 7c3e over f1a8ef0141ff5fa5242c0dcc71ee5a7650438391 on cursor/proofs-regx-part2-crossed-57da (PR #8820). Its lock is over layout.py's limit, so add one lock-only commit by merge.py; pass the parent's Restack-move commits as --move-commit. Details: git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bc-72117543-e1a2-5bc9-8347-c90d795d129c.md | tail -n 12

