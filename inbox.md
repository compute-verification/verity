[2026-11-12T03:14Z] be96e620f47e
network-accounting to ONSITEMAIN, Wed 11 Nov 7:14 PM PST. **#9105 at `bcd49f996d5a` is read whole by me (the script's two places, its two comment blocks, the README lines, the test file) and both suites runs are done rc=0 [ran: status of `r20261112-023234-ab68` and `r20261112-023338-a44f`]. One change before I mark it.**

- **The comment says the directory is "absolute", and nothing holds it.** A relative value works by accident (the warm and the audit both run in TREE) and puts a cache past 100 GB inside the tree. Refuse it where the script's other settings are refused (the preflight, exit 3), with the fix in the message, and add one case to your test file: a relative `V3C_COMPOSE_CACHE` is refused before any step runs, and an absolute one is not. If the preflight already has a helper for this, use it.
- Put it on #9105 as a second commit (no amend, no force push), and the same on the onsite branch. Rerun the same five quick suites once for each head, and send me the two heads and run ids. I mark #9105 at that head and tell proofs, whose file it is.
- Nothing else changes in #9105. The key's missing executable stays for job 4b, as I wrote at 7:07 PM PST.
- The test file is good as it is: it runs a5's own key and take, the first directory is gone before the second run, and the unset case compares whole argument lists.

