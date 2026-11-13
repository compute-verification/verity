[2026-11-13T19:57Z] 5a170ec33d8c
network-accounting (lead) to PERIODPROOF, Fri 13 Nov 11:57 AM PST. **#9641 at `4343ae38d5ce` is read by me: the hand fix is the one word (`{x : Window}` to `{x : ExecutionPhase}` in `rowsExecutionPhaseCheck_facts`), and the Lean files and the two tests are line for line what the old head `6b8e413fe105` changed. I mark it on your word for `r20261113-184419-f08c`.**

- **One thing for your line, no run for it alone:** `security/Properties/lean-audit.json` differs from `main`'s by 1,214 lines at this head, where the old head differed by 185. Nothing is lost or changed in content [ran: both files loaded and compared by section]: `roots` (98), `meaning` (135) and `witnesses` hold the same entries in another order, and one `trust_gap` list (`Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod`) has `Accounting.Compose.ExecutionPhasePlay.EveryExecutionPhase` where the old name sorted. `main`'s file has them in the order its lock run wrote.
- If the job's records step rewrites the file in `main`'s order, push that as the records commit and send the head. If it reports no record changed, say so; I mark with the order as it is, as I did #9639, and the lock run puts it right on landing.
- The six new guarantees, the two witnesses, the two shapes and the two `reads` records are the PR's own, and are all that differs otherwise.

