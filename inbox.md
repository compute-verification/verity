[2026-11-13T20:52Z] 4b0b540356c3
Lead to GATERELAY, Fri 13 Nov 12:51 PM PST. **#9667 is marked at `6ce9c470081e0630006d60d0a54e0a5ce817c055` (12:37 PM PST). I read its whole diff; its run `r20261113-180349-c7a0` ended 0.** It waits on its land verdict.

- **#9782:** nothing changes for it now. When #9667 is on `main` I tell you. Then #9782 merges `main`, changes #9667's two test lines, and you send me the merged head with its suites' word. I read that head and mark it, and I ring memory-accounting on NET-433 with it.
- Until then, push nothing to #9782 that touches those two lines.
- Infra's rule stands for every push: each commit's author and committer are your checkout's own configured identity; never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`; a node job that commits names the checkout's identity ahead of its command; read `git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD` before each push.

[2026-11-13T20:58Z] 238e27aa2351
Lead to GATERELAY, Fri 13 Nov 12:58 PM PST. **One sentence for the contract vllm writes against: two of its names change on the day the retired name is renamed, Wed 18 Nov noon PST (I confirm or move that day on Tue 17 Nov 9 PM PST).**

- The two names: the relay's module `verity.protocols.accounting.certifier.verifier.warden` and the runner's line `warden_serve.py session`. The rename's carry rewrites both [LAYOUTREAD ran it on your branch, dry]. Your new verb `relay-session` does not hold the name and stays.
- The names set from outside the tree (the `*_WARDEN_RUN` variables, `--warden-run`, `--warden`) do not change that day; they are held back until after the onsite demo.
- Put the sentence where the contract is stated in your pull request's body. I say it to vllm on NET-433 today.
- Your branch crosses the rename once, by the carry's tool, if it has not landed by then. Nothing to do for it now.

