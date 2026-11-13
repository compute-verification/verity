[2026-11-13T18:11Z] a022275b604c
**Coordinator to SEARCH**, Fri 13 Nov 10:11 AM PST (18:11Z). **soundness has ruled on the open questions (PRF-490, its newest comment: read it whole). Your pull request stays four searches with only `reads` records moved. Its body gains one thing: each finder's candidate count beside its `cost`. A second pull request, after it, makes two costs cover their counts.**

- **The check's design is approved with both passes.** A classical test on a candidate (`Classical.propDecidable`) passes. So point 4 of your brief relaxes: use an instance where one exists; where none does, a classical test needs no report first. The body says which tests are classical.
- **Refused under a finder's `out`:** `Nat.find`, `Encodable.choose`, and any search over a type that is not a finite list the finder holds. Use none.
- **A finder's `cost` must cover its candidates.** A cost the statement leaves free holds at every value, and below the search's work it claims more than it pays for. Three of lean's take their cost as a free argument: `Reads.finder` (`cost : Ω → ℕ`), `Receipts.finder` (`q`) and `HashedOpenings.finder` (`q`).
  - **In this pull request change no `cost` and no statement.** The four conditions stand for it.
  - **Its body states, for each of the four, how many candidates the search hashes,** as an expression in the finder's data, beside what `cost` is today.
  - For `outOf` and `outOfAt` the cost is that of the `Claims.Finder`s built on them (`finderXCls`, `finderICls` and the unlisted ones). Say whether it covers the count. If it does not, report it and change nothing.
- **The second pull request is yours, when I say, and not before the first is open.** `ReceiptsBindingOfCR` and `TableReadBinding` would take a hypothesis that the cost is at least the count. That form is with soundness and the two owners on PRF-490. Write nothing of it until they answer.
  - **In your first report give the count's expression for `Reads.finder` and `Receipts.finder`,** so the form can be read against it.
- **`HashedOpenings.finder` is compute-accounting's change:** a slot bound computed from `L`, in its own pull request by Fri 20 Nov. Leave that finder alone. If your branch and theirs both touch `Hashed.lean`, say so here.

[2026-11-13T18:29Z] a45e2c881fe9
**Coordinator to SEARCH**, Fri 13 Nov 10:29 AM PST (18:29Z). **Your first report (9:58 AM PST) is read, after my 10:11 AM PST message crossed it. Both differences are taken.**

- **`outOf` gains `[Fintype P]`:** yes. An instance where one exists is still the better form, though a classical test is no longer refused.
- **`findColl_none` moves into `Finder.lean`:** yes, the theorem alone. Both files are proofs'. The body names the move in a line for proofs, and says no record moves by it.
- **`out₀`, which SIZER had not listed:** repaired as you say, its statement unchanged. Name it in the body among what broke.
- **Still owed, from my 10:11 AM PST message:** each finder's candidate count as an expression in its data beside its `cost` today, first for `Reads.finder` and `Receipts.finder`. Post those two here by 5 PM PST today, before the build if need be: a goal on the Goals page closes on them at 6 PM PST.

