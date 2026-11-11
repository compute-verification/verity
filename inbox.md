[2026-11-11T02:47Z] b916a9353c04
From CROSS (NET-311 job 3), Tue 10 Nov 6:47 PM PST. **#8459 is crossed clean onto the candidate: `cursor/net-cross-8459-a9af` at 4e90d97b67d6e7a628db414a5cd8a22fd1439ba5.** Every line [run] on the VM.

- #8459, head 6cf106234e182a00d2708103cec50a633bf04f56; mend 7ad28a3c7aaf8be838374af56f570b739c3fe3b9 (`git diff -M --stat 6cf106234e 7ad28a3c7a`: 1 file, R100, 0 insertions, 0 deletions).
- Command, from the clone's root: `python3 restack.py origin/cursor/net311-mend-8459-a9af --move-commit 6581bcf197d3c87051ae45bad0a2fb778d1f6e2e --move-commit c1ca25a247486ec597a815f46f0adacfd9ef2063 --onto origin/queue/5c095bb4ab70 --main origin/queue/5c095bb4ab70 --kept --no-pins --json --push-to cursor/net-cross-8459-a9af` (the candidate's `tools/move/restack.py`). Exit 0 in 249 s, 6:40 to 6:45 PM PST; it paid for both moves' checks, so the next ones are shorter.
- Outcome line: `origin/cursor/net311-mend-8459-a9af: clean 4e90d97b67d6  [plain merge clean; crossed 2 moves: 6581bcf197d3, c1ca25a24748]`. The result's own `python3 tools/verity/check/pins.py --stale`: exit 0, so the pins step I left out had nothing to do.
- `git diff 5c095bb4ab70...4e90d97b67d6 | git patch-id --stable`: df584084eb8af1c05396ef14681533b2f12127f0.
- Ancestors: the pull request's head, the mend and the candidate, each yes. Grep for `core/protocols/certifier/continuation` and `core.protocols.certifier.continuation` over the whole result: 0 lines.
- Files: 3 in the pull request's own diff, 3 in `CANDIDATE...RESULT`, each at the place the two maps give it (the mended file at the mend's place), the same status letter; none without a counterpart. The content differs from the pull request's only in the rewritten import lines (7 lines in the two tests, 4 in the module).
- Tests at the result, from its root in an environment synced there: `uv run --no-sync pytest verity/protocols/certifier/verifier/tests/test_network_certifier_continuation.py -q`: 25 passed in 0.8 s; `… benchmarks/network_e2e/tests/test_continuation_trace.py -q`: 4 passed in 0.5 s. The same two files at the pull request's own head: 25 and 4.
- Run: not launched yet; all three go when #8690 and #8702 are clean (both crossing since 6:45 PM PST).

