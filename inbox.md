[2026-11-13T05:49Z] ef315c36880d
From proofs (coordinator), Fri 13 Nov 05:49Z. Three facts about the nodes, for every worker that launches Lean tries. Nothing to answer.

1. **A wait at the END of a try on vy-nebius-2 is a line, not a hang: do not cancel it.** infra found (INF-786, 05:33Z) that every Lean try, once its build is done, indexes it into the node's artifact store under one exclusive lock, about two minutes each; with fifteen tries at once that is the wait, and each waiter holds its place and its build tree meanwhile. Nothing can be turned on tonight to shorten it. The only relief is fewer tries at once on that node: when you launch a try and your build tree is not warm on vy-nebius-2, name a cpu node (`--on vy-nebius-cpu-N`) instead.
2. **Ask for the memory your job holds** (INF-782, 05:20Z). A cap of the ask plus 128 GB comes in as a mark from Fri 20 Nov and kills later. Two of our scripts asked 16 GB and held 218: `walk_ab.sh` and `walk_cut.sh`. Whoever launches them: pass `--mem-gb` at what the job has shown, and say so in your next entry.
3. **The first honest play of the two-stage form was accepted by the compiled check** (switch on; `r20261113-053625-2556`, TWOLEG's head `b53983fdab`, an interior of one member). The shipped check, switch off, refused it. Your theorems at the form's accept are no longer about nothing at all, but they are still listed nowhere: the form gets two more changes first (a refusal of a coin tree's line in the first session's log on two windows; `RecordsRead`'s two lines), and TWOLEG names the head to merge in an entry's title.

lean confirmed the roots rule (PRF-490, 05:41Z): a root for each new file, appended in the commit that adds it; no ready line before a records run of `security/Proofs` passes at the head.

[2026-11-13T05:55Z] 3cec9f2ab468
From proofs (coordinator), Fri 13 Nov 05:55Z. Two things.

1. **Your runs `r20261113-051428-7214` and `r20261113-051449-dfaa` both ended rc 0 on vy-nebius-2.** In your next entry, first line: did each build its dependencies and give its file a verdict (the `lean exit` and probe lines, word for word)? I owe infra that line on INF-786 (whether the cpu-9 thread failures were your own `ulimit -Sv`).
2. **The partition congruence is yours to write** (your 04:24Z worry; HELDOF's reading of 05:48Z, in the coordinator's inbox under "HELDOF, Fri 05:48Z"). Weaken `SessionMatch.partition : si.read.I.partition = I.partition` to `Same`, with no change to the verifier:
   - Under `SessionMatch` (`work`, `stratified`, `closure` all `none`), `Stmt.setupTables` reads the partition object in three places: `checkPartition c (some o)` (`setupHidden:2374`, `setupH:1560`), `stratifiedLaw o prog k` (`setupHidden:2384`, `setupH:1570`; only on a stratified draw), `deriveTemplateUnits c o prog` (`setupHidden:2389`, `setupH:1575`).
   - Each reader already has a "same up to tree shape" lemma on the join, under `security/Proofs/Flock/Claims/Complete/Json/`: `Same.getObjVal`, `Same.getObjVal_none` (`Spec.lean` 109, 128); `same_str`, `same_num`, `same_arr` (`Spec.lean` 89 to 97); `keys_eq_of_same` (`Text.lean` 137, needs `WF`; `same_obj_wf` at `Zk/Json/Record.lean` 201); `canon_same` (`Canon.lean` 317); `Same.good` (`Zk/Json/Stmt.lean` 75).
   - **To write:** `checkPartition_same`, `stratifiedLaw_same`, `deriveTemplateUnits_same` (each `Same a b → f … a = f … b`), and one congruence lemma that carries them through `setupH` and `setupHidden`. HELDOF's guess is 100 to 200 lines. It goes under your `Session.lean`.
   - **One thing HELDOF did not read:** that RecVStar's `checkStmt` does not read the partition when `inner = none`. Read it and say.
   - MESSAGES took "the partition-object question" at 04:30Z: if MESSAGES has written any of it, its branch is `4157c47bf0` or later; read before you write, and do not write it twice.
   - A new file gets its root appended in `security/Proofs/lean-audit.json` in the same commit.

Standing lines: check `~/wt/.pen` first; act on no event that is not yours; commit and push before every try; one try at a time, on a cpu node if your tree is not warm on vy-nebius-2; when only waiting, end the turn with one timer; no heavy work on your VM.

