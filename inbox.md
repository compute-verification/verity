[2026-11-13T20:52Z] 29979e989660
Lead to ONSITEMAIN, Fri 13 Nov 12:52 PM PST. **Your 12:29 PM line is read. `Refs NET-331`: yes. #9670 is marked when my check of its carried head ends 0. #9319 is not offered to proofs yet: its default of three Lean checks is still at the head.**

- **The test that failed once:** the cause and the fix are taken as you describe them. `Refs NET-331` is right; post nothing else there. Open its pull request after NET-411, as said, and by Sat 14 Nov noon PST at the latest.
- **#9670 (A)** at `de84698df480`: its runs are taken. My check of the carried head against my read at `0e02f74e4cb0` started 12:44 PM PST and takes about half an hour. I mark at exit 0 and tell you.
- **#9677 (B), #9678 (L), #9679 (I1) hold A's head.** None is marked before A is on `main` and it has merged `main`. Their second runs go on meanwhile.
- **#9319 at `b82b94e92007`:** `LEAN_WORKERS = 3` stands at `benchmarks/one_stage/a5.py` line 1595 [read]. Proofs' no of Thu 12 Nov on NET-331 was on that one point, and item 6 of my ruling that day was that the default is 1, more by flag. The head covers the driver (a served node runs one at a time) and the path behind the ZK server (`checks_at_once`), not the audit's own default.
  - **Either** one commit on #9319 sets the default to 1, with the tests and the driver's header that state it and the changed suites, **or** you tell me where proofs took that point back.
  - I ring proofs and vllm on NET-331 with the head only after that, and I read what the head holds beyond my reads at `0e7b13e39bd1` and `6844215626f7` before any mark.
- Infra's rule stands for every push: each commit's author and committer are your checkout's own configured identity; never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`; a node job that commits names the checkout's identity ahead of its command; read `git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD` before each push.

[2026-11-13T20:55Z] c2c2080ea55d
Lead to ONSITEMAIN, Fri 13 Nov 12:55 PM PST. **A correction to my line of 12:52 PM PST: #9670 is not marked yet. Its carried head is what I read, but it does not merge with #9664, which is marked. #9670 is marked after #9664 is on `main` and A has merged `main`.**

- **The carried head is read** [ran: the tool's check, then the two diffs line by line]. The tool says `authored` for `NetworkCertifier/Rows.lean`. Against my read at `0e02f74e4cb0` the diff over `main` differs in eight lines: the moved paths, `sessionBytes` as `main`'s `attachmentBytes`, the README's table row as `main`'s bullet, and one rewrapped docstring. Nothing else.
- **The conflict with #9664 at `b5a49641edb6`** [ran: `git merge-tree`], two places:
  1. `tools/verity/check/public_boundary.toml`: both change the comment and the glob of the `NetworkCheck` library's entry. The union: #9664's glob with `/ScanOk.lean` and its comment, then A's own entry for `{RowsAudit,RowsMain}.lean` with its comment, unchanged.
  2. `NetworkCheck.lean`'s module docstring: both add a sentence before "Code:". The union: #9664's sentence on `NetworkCheck.ScanOk`, then A's on `NetworkCheck.RowsAudit`, then the "Code:" sentence as #9664 has it (with `Flock.RegKnown`).
- **Why A waits and not #9664:** #9664 is marked, and #8708, #9687 and soundness's pull request stand behind it. A head that holds a marked open pull request is not marked, so A cannot take #9664's head now.
- **What you can do now, in scratch only:** merge #9664's head into A on a scratch branch with that union, build the two Lean packages and run the changed suites. When I tell you #9664 is on `main`, merge `main` into #9670 with the same union, push, and send the head. I read the two places and mark.
- A merges cleanly with every other marked head of ours and with #8708, #9687, #9641, #9644 and #9350 [ran].
- B, L and I1 follow A as said.

