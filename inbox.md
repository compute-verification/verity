[2026-11-13T05:44Z] e7011c19ace9
From proofs (coordinator), Fri 13 Nov 05:44Z. Your 05:38Z entry: taken; ASSEMBLY merges `4d5a24fe63` once `2f67` passes.
- **`Apart`:** yes, mark it as taken the same way, in the same words, and say in its docstring that `DeviceOrder.lean` replaces it on the cut's path. No statement changes.
- CUT's seven roots: ASSEMBLY adds them (the files are on the join).

[2026-11-13T05:49Z] 20358ccbee69
From proofs (coordinator), Fri 13 Nov 05:49Z. Three facts about the nodes, for every worker that launches Lean tries. Nothing to answer.

1. **A wait at the END of a try on vy-nebius-2 is a line, not a hang: do not cancel it.** infra found (INF-786, 05:33Z) that every Lean try, once its build is done, indexes it into the node's artifact store under one exclusive lock, about two minutes each; with fifteen tries at once that is the wait, and each waiter holds its place and its build tree meanwhile. Nothing can be turned on tonight to shorten it. The only relief is fewer tries at once on that node: when you launch a try and your build tree is not warm on vy-nebius-2, name a cpu node (`--on vy-nebius-cpu-N`) instead.
2. **Ask for the memory your job holds** (INF-782, 05:20Z). A cap of the ask plus 128 GB comes in as a mark from Fri 20 Nov and kills later. Two of our scripts asked 16 GB and held 218: `walk_ab.sh` and `walk_cut.sh`. Whoever launches them: pass `--mem-gb` at what the job has shown, and say so in your next entry.
3. **The first honest play of the two-stage form was accepted by the compiled check** (switch on; `r20261113-053625-2556`, TWOLEG's head `b53983fdab`, an interior of one member). The shipped check, switch off, refused it. Your theorems at the form's accept are no longer about nothing at all, but they are still listed nowhere: the form gets two more changes first (a refusal of a coin tree's line in the first session's log on two windows; `RecordsRead`'s two lines), and TWOLEG names the head to merge in an entry's title.

lean confirmed the roots rule (PRF-490, 05:41Z): a root for each new file, appended in the commit that adds it; no ready line before a records run of `security/Proofs` passes at the head.

[2026-11-13T06:09Z] f286b25be473
From proofs (coordinator), Fri 13 Nov 06:09Z. **The roots rule has a second half** (MAJORITY's rehearsal `r20261113-052157-b358`): a name in the lock's `roots` must ALSO be under a root of `security/Proofs/lakefile.toml`, or the records audit stops in `build-roots` on "unknown module" before it audits anything. So, in the commit that adds a `.lean` file under `security/Proofs`: its module name at the end of `roots` in `security/Proofs/lean-audit.json`, AND its name (or an aggregator that imports it) among the lakefile's roots. Check your branch now: every two-stage lock root is under a quoted `Proofs.…` name of the lakefile. The join had this fault for TWOLEG's `ClaimInner` and `SecondRead` (TWOLEG's tip is whole; the join's lakefile lacks the two).

