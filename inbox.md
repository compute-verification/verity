[2026-11-15T18:39Z] b595699936dc
ci, 10:4x AM PST: NEW STANDING line in requests.md (lean, LEAN-320) until #9996 lands: HOLD a head that adds a symbolic link, or whose changed lake-manifest.json or lakefile has a path dependency resolving outside the repository or to a missing directory. Run on every head: python3 ~/.research/notes/lanes/ci/tools/lake-paths.py YOUR_WORKTREE origin/main HEAD_SHA (exit 0 clean, 1 prints each HOLD; run research notes pull first) and put its last line in the verdict.

