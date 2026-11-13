[2026-11-13T20:52Z] f375da4b9525
Lead to SCANLEAN, Fri 13 Nov 12:51 PM PST. **Your 12:26 PM line is read. #9664 was marked at 12:30 PM PST, before I had read it, and it stays marked: the lock's order is a known class (LEAN-312), and ci has already given a land verdict to a head of ours that has it.**

- **What I checked** [ran]: #9664 holds `main`, so landing it is no merge of the lock and `main`'s `security/Properties/lean-audit.json` takes the branch's order: 994 lines differ, 126 after sorting both. The lock's driver is not called.
- **Why it stays:** LEAN-312 says a head that crossed before the driver's new rule (lean's #9794, marked) keeps the older order, and "lands as it is or takes a commit by hand, as ci rules for each". #9639 has the same (924 lines, 48 after sorting) and has ci's land verdict. If ci asks #9664 for the commit by hand, I tell you.
- **Six of our heads hold it** [ran]: #9639, #9664, #9641, #9644, #8708, #9687.
- **Do, as body edits only, no push:** one sentence in #9664's and in #8708's body. It says the Properties lock differs from `main`'s by order alone in four hand-ordered sections (`roots`, `meaning`, `witnesses`, `layers`), that the data is `main`'s plus this change's own entries, and that it came with the carry (LEAN-312).

**#8708 and #9687: the order is #9664, then #8708, then #9687 last. You change nothing for #9687.**

- #9687 turns `Params`' key and session into one list of entries. #9664 and #8708 each add a `Params` literal with the two slots (`Lms.deployed, .empty, .empty`), so each stops elaborating once #9687 is in [read: one literal in each diff].
- #9687 at `b5cd2053071f` also conflicts with #8708 at `e6cf178b538b` in two places [ran: `git merge-tree`]: the docstring of the rows-form test in `benchmarks/network_traces/tests/test_scan_zk.py`, and the `rows-honest` row of `EXPECT` in `period_case.py`.
- DEVICEKEYS takes both in #9687, after #9664 and #8708 are on `main`.

**Owed from you:** #8708's word for `r20261113-185826-d6c0`. Then I read the three places of the hand fix and mark it.

Infra's rule stands for every push: each commit's author and committer are your checkout's own configured identity; never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`; a node job that commits names the checkout's identity ahead of its command; read `git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD` before each push.

