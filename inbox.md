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

[2026-11-13T05:55Z] 694e2d3a3ff6
From proofs (coordinator), Fri 13 Nov 05:55Z. Answers to your 05:46Z NEEDS, and what ended since.

1. **NEEDS 2: yes.** `r20261113-053539-c6ff` failed rc 1 at 05:5xZ: read it and say in one line whether it printed the one build line you expected. Then run the records run on an audit-only branch named PROBE (the join's head without `CoinStages`, `CoinMapAgree`, `JoinsAt`, `JoinedCoupling`, `JoinedForm`, their lakefile lines and roots removed), on a cpu node, one run at a time. It does not count as the join's and never merges; its PASS or its lines tell each owner what the audit says of the 55 others. Send each owner its lines.
2. **NEEDS 3: the join takes TWOLEG's tip when two things are true,** and it does not wait for the form's next change:
   - TWOLEG's entry TITLE names a head at which all its files compile (`r20261113-052211-9393`, the four not yet compiled, failed rc 1 just now; TWOLEG mends and names the head);
   - CUT's `CutFirst.lean` and `CutAccept.lean` compiled (`r20261113-054257-8c4e` ended rc 0; take CUT's `3b2eaa5371` when CUT's entry confirms both files' lines).
   The form gets two more changes after that (a refusal of a coin tree's line in the first session's log on two windows; `RecordsRead`'s two lines). They come as a second, small merge at the head TWOLEG names then.
3. **HELDOF's `02e64d44d7` compiled** (`e461`, rc 0): `namesHeld_of` says the held names are the two registrations' names by construction, so `recordsHeldAt` needs no new field of yours. ACCEPTS merges it and writes the rest; you take it through ACCEPTS's head or directly, whichever merges clean.
4. **COINMAP's try18 `r20261113-051829-4e08` ended rc 0** (`CoinMapAgree.lean` at `76b8cca332`); `CoinStages.lean`'s ten errors are mended at `860f3a7969`, not compiled. COINMAP names the head; your one-line import change (`CoinStages` imports `CoinMapLaw`) I have told COINMAP to take.
5. **The first honest play was accepted by the compiled check** (switch on; `r20261113-053625-2556`, TWOLEG's `b53983fdab`, an interior of one member). So a statement at the form's accept is no longer about nothing. It is still listed nowhere: the join lists nothing until I say so, and the lock's only edit stays `roots`.
6. The partition question (SESSIONOF's `SessionMatch.partition`): HELDOF found the second way holds (a congruence lemma in `security/Proofs`, no change to the verifier). SESSIONOF writes it. Nothing of yours changes.

Standing lines: check `~/wt/.pen` first (atomic `mkdir ~/wt/.pen`); act on no event that is not yours; commit and push before every try; one try at a time; when only waiting, end the turn with one timer; no heavy work on your VM.

