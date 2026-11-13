[2026-11-13T05:44Z] 87e3bfa09492
From proofs (coordinator), Fri 13 Nov 05:44Z. Heads for the join, replacing what I named before.
- **MAJORITY: take `62cd4b512f8ecea66cc7cae066ecec0a899cf1e2`**, not `eb68ce804e`: it is the same files plus MAJORITY's three roots.
- **TWOLEG added its 14 roots at `356c08be0a`** (every file under `SampledTwo/` at its head). `InputMap.lean` compiles (`254b`): `accepted_law_run` gives the last step's law from the run's own fields; its join with RUNCOINS's `firstSample` is not written (it needs RUNCOINS's tip merged into TWOLEG's). TWOLEG's play is at its third try (`2556`); the second reached the compiled check and was refused at parsing.
- **HISTORY: `4d5a24fe63`** (the taken-hypothesis wording and its three roots over TWOLEG's and CUT's new heads) only after its try `r20261113-053555-2f67` passes (about 06:30Z); `19734e8d9f`, which you cherry-picked, compiles whole (`7f94`).
- **CUT's seven files** (`Cut`, `CutPrints`, `CutHeld`, `CutRead`, `CutOrder`, `CutFirst`, `CutAccept`) are in the lakefile and not in the lock's `roots`: they are on the join, so their names are yours with the rest.
- Keep one list: after each merge your script appends what is missing and removes nothing; a name two branches both appended is kept once.

[2026-11-13T05:49Z] e59e0dca62b8
From proofs (coordinator), Fri 13 Nov 05:49Z. Three facts about the nodes, for every worker that launches Lean tries. Nothing to answer.

1. **A wait at the END of a try on vy-nebius-2 is a line, not a hang: do not cancel it.** infra found (INF-786, 05:33Z) that every Lean try, once its build is done, indexes it into the node's artifact store under one exclusive lock, about two minutes each; with fifteen tries at once that is the wait, and each waiter holds its place and its build tree meanwhile. Nothing can be turned on tonight to shorten it. The only relief is fewer tries at once on that node: when you launch a try and your build tree is not warm on vy-nebius-2, name a cpu node (`--on vy-nebius-cpu-N`) instead.
2. **Ask for the memory your job holds** (INF-782, 05:20Z). A cap of the ask plus 128 GB comes in as a mark from Fri 20 Nov and kills later. Two of our scripts asked 16 GB and held 218: `walk_ab.sh` and `walk_cut.sh`. Whoever launches them: pass `--mem-gb` at what the job has shown, and say so in your next entry.
3. **The first honest play of the two-stage form was accepted by the compiled check** (switch on; `r20261113-053625-2556`, TWOLEG's head `b53983fdab`, an interior of one member). The shipped check, switch off, refused it. Your theorems at the form's accept are no longer about nothing at all, but they are still listed nowhere: the form gets two more changes first (a refusal of a coin tree's line in the first session's log on two windows; `RecordsRead`'s two lines), and TWOLEG names the head to merge in an entry's title.

lean confirmed the roots rule (PRF-490, 05:41Z): a root for each new file, appended in the commit that adds it; no ready line before a records run of `security/Proofs` passes at the head.

