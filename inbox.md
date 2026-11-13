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

[2026-11-13T21:04Z] 0598965dc4f7
Lead to GATERELAY, Fri 13 Nov 1:04 PM PST. **infra corrected itself on NET-435 (12:36 PM PST): it pins a commit on the gate's machine only once that commit is on `main`. `next` has not moved since Mon 9 Nov and never holds our pull request. So the pin follows the landing, and the pull request's day now sets the pin's day.**

- **In the body, under "For the pin":** the sha infra gets is the commit of `main` that holds the pull request. I send it; you do not. Write no sha there, and do not say "on `next`".
- **One question, answered in your next line: what is the earliest hour the pull request can open, unmarked, with the verb, the runner's `session`, the piece and their tests?** The plan says Mon 16 Nov noon PST. A landing takes a day or more in the queue as it runs today, so a pull request marked Monday evening is pinned Wednesday at the earliest. If it can open Sat 14 Nov with the contract's code, proofs and I read it over the weekend and Monday morning, and I mark it Monday.
- Do not cut anything to open earlier, and do not open it half built. If Mon 16 Nov noon is the earliest honest hour, say so and I plan the pin from that.
- Nothing else in your work changes. #9782 still waits for #9667 on `main` (ci gave #9667 its land verdict at 12:51 PM PST; it is not on `main` yet).

The rule for every push stands: each commit's author and committer are your checkout's own configured identity; never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`; a node job that commits names the checkout's identity ahead of its command; read `git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD` before each push.

