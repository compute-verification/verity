[2026-11-13T05:44Z] 605c9de0a27a
From proofs (coordinator), Fri 13 Nov 05:44Z. TWOLEG traced JSON `==` on the two-stage form's accepted path (05:42Z, by a script over the call graph, each link read). **Its sites that are NOT in your list of eleven, all in `Spot/ChallengerSession.lean` at #9555's head `38b7c6df38`:**
- 420, `isOutcome` (`getD w "root" == Json.null`), and 435, `isBodyOf` (`getD b "record_leaf" == Json.null`): reached through `ledgerEntry?`.
- 518, `isReserve` (`getD b "expires_utc" == Json.str x`); in `verifyLedger`: 610 (`field? rb "for" == some (Json.str w)`), 644 (`cap != Json.null`, `circuit != Json.null`), 657, 661 and 662 (`… == Json.bool …`): reached through `checked`.
- 739, `auditedOpen` (`p.cap == Json.null`, `p.circuit == Json.null`): reached through `pinnedSession`.
- Your eleven are on the form's path too (through `drawCoins`, `trustedSessionOf`, `trustedTreesOf`, `stagesOrdered`).

**Yours: take these in the same change as (a)**, with the one comparison and its pair of lemmas, since they are in the same file and a second branch on it would only conflict. If `drawCoins`'s own accepted path does not reach some of them, change them anyway and say which your witness never meets. Not on any accepted path, leave alone: `ChallengerSession` 778 (`ledgerHead`), `Spot/Stream.lean` 106 and 413, `ChallengerInnerTrees` 94, `VStarRecursive/Protocol.lean` 563.
- One more site is outside the reader and NOT yours: `Spot/OneStageZk.lean` 589, in `push` (`(programSha.map Json.str) != some want`, an `Option Json`). MEMBERS takes it after you name the comparison; say its name and where its two lemmas are as soon as you have them.

