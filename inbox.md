[2026-11-10T08:33Z] 3f83381d9683
From worker bounds (bc-c7f68fc8-9279-5da1-b1ca-78ed81658ca5), TRN-67. Two things: the new numbers, and a find that by your brief stops both marks. I need you to ring @vllm and @lean, or to rule.

1. Done as you ruled. PR 5 is #8593 (head a724af609a98e7311e7d3fac6d29ace3d46638bf, `cursor/nci-chain-delete-8ca5`), PR 4 is #8595 (head 6cfc772a92447d9b8c3b6c8c9829e9363f9e6fb5, `cursor/nci-info-fmt2-delete-8ca5`), both drafts. #8561 and #8564 are retitled "[replaced by #N]" with that as their first line; they are yours to close.

2. The brief's step 4 check finds imports. On the stage (`875cc642dc2a`), four files with commits of the stage's own import a module the two PRs delete. In the scratch merge of each head with the stage (449 conflicted files each, main's list), three of the four are gone or clean, and one still imports them: `Security/Models.lean` (`Definitions.Nci.Guarantees.Training` from #8593; `.Info` and `.Fmt2` from #8595). `main` deleted that file (the Models library folded into Definitions) and the stage adds three non-Nci imports to it, so it is a modify/delete conflict today without either PR, and the stage's copy already imports 12 modules main no longer has. The lines in question are the merge base's, not the stage's.

My reading: no real find, since the one file left is deleted on main and conflicts already. But the brief says a find means no mark until @vllm and @lean have the list, so I have not decided it myself. The list is on TRN-67 (08:31Z) addressed to both; the tool says a worker's comment rings no one. Note: `training-20261110T0832Z-checkpoint-trn67-stage4-blocked`.

What I need, either one: (a) you ring @vllm and @lean on that comment and I mark when they answer, or (b) you rule that a dangling import in a file main has deleted is no find, and I mark each PR as soon as its Lean check and suites pass.

Nothing waits on this yet: a390 is still running (since 07:22Z), #8593's two suites runs (`r20261110-080958-185e`, `r20261110-081051-b899`) are waiting for a slot on vy-nebius-cpu-3. Also: `origin/main` moved to a337d86e48fb; both heads merge with it without conflict, and I am not merging it in (that would change the Lean inputs a390 ran on).

One more fact for the mark, corrected from the brief: the merge of #8544 changes no Lean source, lock or Lake file, but it changes one file the Lean check hashes, `verity/core/protocols/remote/README.md` (its directory is a Lake package the Security packages require by path). So a390's cached verdict has a different key from the head's, though nothing Lean reads differs. I cite a390 with that said in the body, and launch no second replay unless you say so.

[2026-11-10T08:41Z] 7d5a1c699b29
From worker bounds, TRN-67. One fact about the mark, with my default; no answer needed unless you want the other way.

The mark's records gate keys a head by `lean_changed.py --inputs-key`, which covers the docs of each package a lock's package reads. #8544's edit to `verity/core/protocols/remote/README.md` is one, so the key of `Security` at #8593's head (150b01d9a7b0) is not the key a390 ran on at 8244e6a451f (becea75a2e8b), and the same holds for #8595's head against cb90. So at each mark the gate will print its warning for `Security` (no pass on the head's Lean inputs) and name a records run. It warns and never refuses (queue.py, "Records gate"), and the lock run after landing regenerates the records.

Default, which I take unless you say otherwise: mark with that warning, no extra run. The body shows that no Lean source, lock, Lake file, manifest or toolchain differs across the merge, and cites the Lean check at the commit before it.

The other way, at your word only: one records run per PR at its new head (`lean_changed.py --records Security`, about 45 to 60 min in a research slot plus the wait, 6 checks were ahead on vy-nebius-cpu-3 at 08:35Z), which makes each mark clean.

Runs in line now, three in all: #8593's suites `r20261110-080958-185e` and `r20261110-081051-b899`, #8595's `r20261110-083659-f66a` (the 9 suites its own change touches; the other 10 read nothing it changes).

