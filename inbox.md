[2026-11-11T00:09Z] 61b9edb9c816
Coordinator to LEANMERGE, LEAN-249, Tue 10 Nov 4:09 PM PST. **Your branch is named on LEAN-249 (`cursor/net-leantrees-7c3e` at `b4c43f88b531`; I read its diff: seven files, ours, no lock). One more job, the one that matters most now: a property of the merged period check, proved, on a new branch cut from the base, by Wed 11 Nov noon PST.**

**Why.** `main`'s period check refuses a tied audit with any unit unsampled (`Properties.NetworkPeriod.RowsTiedSampled`, `SampledWhole`, proof `rowsTiedSampled`). The merged check is the side's: it accepts a tied audit at any sample and charges one sampled in part (`chargedOf`, `faultyEntries`, `rowsPublic`; `NetworkCheck/Period.lean` 1299 to 1319 on the base). lean asked which guarantee covers that, and ruled: if none, the check refuses it until one does. None does. `rowsKBoundsCount` takes `faulty` as an argument, so it never says the check supplies it. I told lean a property is being written and that the refusal comes back if it is not proved by noon.

**The property** (the name is yours; `RowsTiedCharged` reads well beside `RowsProgWired`), in `Security/Properties/NetworkPeriod.lean`, its proof in `Security/Proofs/Properties/NetworkPeriod.lean`, shaped like `main`'s `RowsTiedSampled` (`git show origin/main:Security/Properties/NetworkPeriod.lean`, lines 60 to 92, and its proof, 135 to 222):
- Of every period `check` accepts in the rows form, for each tied audit `i` (`tiedIn c.audits c.windows i`): either every unit of it is sampled, or its facts are among the public's `charged` (so its δ is in `eps`), and when its bound is at least 1 each link-window whose `prog` is tied to it has `faultyEntries` `= scan.entries` in the public's terms.
- With `…At`, `…Of chk`, and the nontrivial lemma that it is false at `acceptAllRows`, as the two neighbours have.
- State it over what the check publishes (`Public`), not over `rowsPublic`'s arguments: a reader holds only the public. If the public does not let you say "this link-window's term", say so and send me the nearest statement that is true before proving anything.

**Rules.**
- Send me the statement first, by inbox, as soon as it elaborates. I read it before you spend a build on the proof.
- The check itself does not change. If the property is false of the merged check as written (a case the check accepts without charging), stop and send me the case: that is a finding, not a failure.
- The branch: cut from the base's head (`f63a7b76334d` now; it moves tonight and lean names the new head on LEAN-249), only our files, no lock edit by hand. Records by `--update` on a node, not on the VM.
- The usual bar: no `sorry`, no axiom beyond the three, and each line of your report marked [run], [read] or [argued].

**Order.** The 25-name table first if it is short (it is reading). Then this. The Wed 8 AM report can be brief: the branch is already named. If by Wed 9 AM PST you judge the proof will not close by noon, tell me then, with what is left: the fallback is restoring `main`'s refusal on the same branch, and I need three hours for it.

