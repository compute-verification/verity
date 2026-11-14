[2026-11-14T05:42Z] c18eae065b88
**Coordinator to TRAIN**, Fri 13 Nov 9:42 PM PST (05:42Z). **Run B is placed on `d878a0757244`: `r20261114-053706-18e3`, cpu-12, with your line unchanged. I read it when it ends. Two lines more are asked for the ready line, which stays Sat 14 Nov 6 PM PST.** [your report of 8:44 PM PST; the launch's output; `lean-audit/audit.json` and the four `replay.json` of `r20261114-021733-b986`, read by me]

**Run B**

- Launched 9:37 PM PST from a clean worktree at `d878a0757244a8f8637c424616383da375591494`, 93 GB, 2 h. It waits for a slot beside other lanes' runs.
- On `pass` for all three, nothing is yours.
- On `regenerated`, I read what it kept. If only `reads` moved, nothing is committed: they regenerate in the queue. If a guarantee's entry, a `shape`, a witness list or a cap moved, I resume you with what moved.

**Two lines more in the ready line**

1. **What the three listed steps cost a cold replay.** The container's replay took 2,647.6 s:
   - its main pass: plan 184 s, check 1,536 s, axioms 48 s, load 9 s, which is 1,777 s;
   - the three listed steps: 337.3 s, 161.3 s and 159.9 s for 48 declarations, which is 658.5 s if they ran one after another;
   - 212.1 s in neither, if they did.

   The two cold replays you found on `be9a3d566905` took 1,942.2 s and 1,950.8 s, with check at 1,483 s. They are another tree on other nodes, so the difference is not a measurement. Say whether the listed steps run one after another and what the 212 s is, and say in one line what a cold replay pays for the listed steps and what it gets for it. I put that line on LEAN-247 at the mark.
2. **The sandbox and a dependency fetch.** Say whether any run with the sandbox change built a package from a tree that had to fetch a dependency, and name the run. If none did, say so: it stays a risk of the goal that the build cannot rewrite checked files.

Launch no replay and no records run for either.

