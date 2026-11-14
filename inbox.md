[2026-11-14T15:46Z] 602494cc047f
STEP, Sat 14 Nov 7:46 AM PST. #9863 is marked at b5f9048a2a14fe9c42b6756049768cc455192a81 [RAN]. The queue's reply: `#9863 is ready at b5f9048a2a14 (pr:9863@b5f9048a2a14…)`.
- Its preflight merged the head onto `main` at a116c5614312, and checked nothing: `tests:adapters/vllm/tests/lint not checked: still running when the 120 s budget ran out`.
- The draft flag is still on [RAN, `gh pr view`]. Nothing I have can clear it: `gh` is read-only here, and `research queue` has no command for it.
- The queue never reads the flag [READ, `tools/research` on `origin/main`]. My recommendation: leave it, or clear it from your side. If I hear nothing in 30 minutes, I'll leave it.
- Next, I'm opening the backend-switch change (Refs TRN-164), stacked on #9863's branch.

