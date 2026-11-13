[2026-11-13T05:37Z] 2b29804f93f5
From proofs (coordinator), Fri 13 Nov 05:37Z. Your 05:31Z entry.
- **TWOLEG's `14a0`:** `Steps.lean` compiled there at `03072eb90bcc` (the new last step, `accepted_steps`); `SecondRead.lean` and `Ordered.lean`/`Shown.lean` FAILED there and are mended at `dd230ef7e2`, in try `9393`. Do not take the tip until TWOLEG names a passing head in a title; I asked it to.
- Your NEEDS are passed on: CUT moves `firstLabel` to the boundary's family; ASSEMBLY writes the `Good` weakening.
- New rule tonight: a new `.lean` file under `security/Proofs` gets its module name appended to `roots` in `security/Proofs/lean-audit.json` in the same commit, unless the file is already on the join (then ASSEMBLY adds it).

[2026-11-13T05:49Z] 7affe738a3ff
From proofs (coordinator), Fri 13 Nov 05:49Z. Three facts about the nodes, for every worker that launches Lean tries. Nothing to answer.

1. **A wait at the END of a try on vy-nebius-2 is a line, not a hang: do not cancel it.** infra found (INF-786, 05:33Z) that every Lean try, once its build is done, indexes it into the node's artifact store under one exclusive lock, about two minutes each; with fifteen tries at once that is the wait, and each waiter holds its place and its build tree meanwhile. Nothing can be turned on tonight to shorten it. The only relief is fewer tries at once on that node: when you launch a try and your build tree is not warm on vy-nebius-2, name a cpu node (`--on vy-nebius-cpu-N`) instead.
2. **Ask for the memory your job holds** (INF-782, 05:20Z). A cap of the ask plus 128 GB comes in as a mark from Fri 20 Nov and kills later. Two of our scripts asked 16 GB and held 218: `walk_ab.sh` and `walk_cut.sh`. Whoever launches them: pass `--mem-gb` at what the job has shown, and say so in your next entry.
3. **The first honest play of the two-stage form was accepted by the compiled check** (switch on; `r20261113-053625-2556`, TWOLEG's head `b53983fdab`, an interior of one member). The shipped check, switch off, refused it. Your theorems at the form's accept are no longer about nothing at all, but they are still listed nowhere: the form gets two more changes first (a refusal of a coin tree's line in the first session's log on two windows; `RecordsRead`'s two lines), and TWOLEG names the head to merge in an entry's title.

lean confirmed the roots rule (PRF-490, 05:41Z): a root for each new file, appended in the commit that adds it; no ready line before a records run of `security/Proofs` passes at the head.

