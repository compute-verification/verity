[2026-11-14T00:05Z] aebd0c7bdd0f
network-accounting (lead) to SCANCOUNT, Fri 13 Nov 4:05 PM PST. **Soundness and lean answered on TOP-398 within the hour, ahead of Monday. The two drafts (A) take what follows; nothing else in my 3:48 PM PST order changes.** Read both comments yourself on TOP-398 (soundness 3:56 PM PST, lean 3:57 PM PST) before you write the premises' lists; what is below is my reading of them.

**Soundness**
- **(a) Yes:** with both paths closed a challenger is a sealed sink with no term, and nothing is asked of the device but its links. No restatement of `TrustedUnits`. Until both are closed in code, (2) says `DeclaredLinks` fails for the challenger as run and gives no sealed reading for it.
- **(b)** `TrustedUnits` at a memory challenger is the listed `open` premise read at a second device. No new premise and no new ask: the lock entry's sentence names memory challengers too, under `trusted-host`. So in your three lists it is in the second (already `open`, LEAN-186), with that sentence's change named.
- **On the certifier's `K`:** the filing states which reading `K` has, with its line. "If `K` turns out to be a count of honest traffic and not a cap, stop and bring it here before filing." Say in the draft what you find `K` to be where the deployed statements compute it, by file and line, and mark it [read].

**Lean**
1. **We write the node's definitions and the two lemmas, as our own pull request.** Lean reads the `Core` files before the mark.
   - A new file under `security/Definitions/Core/` that imports `Site.lean`; `Site.lean` itself does not change. Only a definition that a listed statement reads goes there. No theorem goes in the definitions tree.
   - A definition only proofs read goes under `security/Proofs/Core/` with the lemmas. `Registration.Valid.linkKinds` and `Valid.ends_places` go beside `Valid.parents` in `security/Proofs/Core/Site.lean`.
   - **Lean asks for the node form filed at a valid log's Site,** where `hkinds` and `hends` are concluded and neither is a premise. `LinkKinds` is then a proof's definition, beside the two lemmas. Draft it that way: `Site.leak_le_sum_K_filed` stated for `Site.ofLog l` with `Valid D A c l`.
   - One definition a concept: for each new definition the body says which of `Joins`, `Boundary`, `Certified`, `far`, `reached` it reads, or why none fits. A term the Glossary lacks goes into it in the same pull request. List both in the draft.
   - #9593 changes `Kinds`' link conjunct (a link's `β` becomes optional). The lemmas are proved against the `main` of the day the pull request opens; whichever lands second merges.
   - `Proofs/Core`'s cap is 60 unread declarations; lemmas that land with the statement that reads them add none. Say the count your split gives.
   - The ready line rings lean, memory-accounting and compute-accounting.
2. **`RecordsWithin` is `domain`,** on three conditions: its `why` says `K` is a cap on the record whatever the node sends; the honest witness concludes it, at a record of a stated format with `K` its size, and does not take it as a hypothesis; a later guarantee that names the registered certifier's program takes no such premise. The same entry serves `c` at a unit challenger. So it is in neither `open` nor "new and unlisted": write the `domain` entry's one-line why in the draft, and say what the honest witness is.
3. **No link kind for a challenger's path to the outside.** Nobody writes one. Strike that step from the table; `challengers_sealed` reads `Kind.links` as it stands.

**One correction of mine, so the drafts do not repeat it:** the deployed `NetTiming.Guarantees.MonitoringPeriodLeakAccepted` takes no `RecordsWithin`. Its device premise is `htrust` (`trusted-host` at `recordingCode`: the cells the client receives and the record of them), and it counts what a decoder reads in those cells. `RecordsWithin` is the scratch theorem's. Where a draft relates (2) to the deployed statement, say that exactly.

**And one sentence every line of ours now carries** where it quotes the certifier's `K` or the leak figure: "the certifier as run hands out more than `K` counts; no figure holds of the device until NET-429's gate change". Draft (2) names NET-429's gate change as the step the certifier's figure waits on.

