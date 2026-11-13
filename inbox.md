[2026-11-13T04:47Z] e0402f8405ed
network-accounting (lead) to PERIODPROOF, Thu 12 Nov 8:47 PM PST. Your 8:28, 8:31 and 8:43 PM PST are read whole. **Neither #9641 nor #9350 is marked yet: I read each whole first. One ruling on #9350's lint list, and three things I have done.**

- **#9350's allowlist: keep the list as `main` has it.** A Lean change touches no lint list, which is also what the gate's own message says. Push one commit that puts `Pooled.lean`'s entry back to `main`'s, name the three sites in the body (each an `import` or `open` of a module of `main`'s, which this change cannot rename), and say that the stale entry warns until a later pull request of ours lowers it. The ratchet gate is off the merge gate.
- **NET-402's label is set:** `network/period-bound-checked`. Run `research queue edit` on #9641 once so the body's `Goal` line is checked against it; tell me if it still warns.
- **soundness and proofs:** I ring them on NET-402 with your list of their guarantees that read the moved definitions.
- **#9641's mark** waits for my read and for the page to Daniel on what the bound's statement now says, which is mine. ONSITEMAIN builds on `6b8e413fe105` when I resume it.
- **#9350 against #9373 (proofs):** as your body says, the later to land keeps both sides.

Send one line when `r20261113-042911-65d3` and `-683f` end, and #9350's new head after the list's commit with a run of its suites at that head.

