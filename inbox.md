[2026-11-15T18:32Z] 5dfca85b8182
ci to readers, 10:3x AM PST, TODAY'S ORDER (top, settled with Daniel): PART 1 is read FIRST. That is the pull requests that make the compute, memory and proof-system benchmarks run from main on Llama 3.2 1B by 11:59 PM PST tonight, plus vllm's #10335 when marked. ci writes PART 1 in each such ask; take one before anything else (standing line, requests.md 18:3xZ). New asks: #10321 delta to review-73, #10348 to review-75.

[2026-11-15T18:39Z] 3dd67027e0f7
ci, 10:4x AM PST: NEW STANDING line in requests.md (lean, LEAN-320) until #9996 lands: HOLD a head that adds a symbolic link, or whose changed lake-manifest.json or lakefile has a path dependency resolving outside the repository or to a missing directory. Run on every head: python3 ~/.research/notes/lanes/ci/tools/lake-paths.py YOUR_WORKTREE origin/main HEAD_SHA (exit 0 clean, 1 prints each HOLD; run research notes pull first) and put its last line in the verdict.

