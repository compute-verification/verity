[2026-11-13T05:12Z] d088c300b2c5
From proofs (coordinator), Fri 13 Nov 05:12Z. Answers to your 05:05Z entry.

- **`RunsAs/Within.lean`: delete it**, as you propose. ACCEPTS's `readsWithin` (`SampledTwo/OfRun.lean`) is the one that stays. Say in the commit which run compiled yours (`49aa`), so the record is not lost.
- **`coinsCode … = none`:** write it at the join when you restate `Steps.lean` over TWOLEG's `run`, as you say. I told CUT of the two-liner; if CUT asks for it in its file, I will tell you.
- **`inputOf`'s map** (`p.replay`, `q`, `p`'s counts to the run's `it.replay`, `it.params.verifier.query`, `i.rule`'s `k`): given to TWOLEG at 05:12Z, with your `firstSample` and `selectSecond_law` named as what it joins to.
- **The aggregator's probe covering nothing:** do send the try of `Within.lean`'s successors for their axiom lines (`Steps`, `Coins`, `Proved`, `Top`, `Played`); the join reads `accepted_steps`, `runAcceptsPlay` and `spotRunAcceptsPlay`, and none has a probe line at `d3cb96c006`.
- TWOLEG's honest play `r20261113-044454-29e7` did not accept: the driver did not find the form's case file, so the form never ran. Nothing of yours; do not restate over TWOLEG's `run` until it names a head that compiled.

[2026-11-13T05:49Z] 16d1ccdcf2d5
From proofs (coordinator), Fri 13 Nov 05:49Z. Three facts about the nodes, for every worker that launches Lean tries. Nothing to answer.

1. **A wait at the END of a try on vy-nebius-2 is a line, not a hang: do not cancel it.** infra found (INF-786, 05:33Z) that every Lean try, once its build is done, indexes it into the node's artifact store under one exclusive lock, about two minutes each; with fifteen tries at once that is the wait, and each waiter holds its place and its build tree meanwhile. Nothing can be turned on tonight to shorten it. The only relief is fewer tries at once on that node: when you launch a try and your build tree is not warm on vy-nebius-2, name a cpu node (`--on vy-nebius-cpu-N`) instead.
2. **Ask for the memory your job holds** (INF-782, 05:20Z). A cap of the ask plus 128 GB comes in as a mark from Fri 20 Nov and kills later. Two of our scripts asked 16 GB and held 218: `walk_ab.sh` and `walk_cut.sh`. Whoever launches them: pass `--mem-gb` at what the job has shown, and say so in your next entry.
3. **The first honest play of the two-stage form was accepted by the compiled check** (switch on; `r20261113-053625-2556`, TWOLEG's head `b53983fdab`, an interior of one member). The shipped check, switch off, refused it. Your theorems at the form's accept are no longer about nothing at all, but they are still listed nowhere: the form gets two more changes first (a refusal of a coin tree's line in the first session's log on two windows; `RecordsRead`'s two lines), and TWOLEG names the head to merge in an entry's title.

lean confirmed the roots rule (PRF-490, 05:41Z): a root for each new file, appended in the commit that adds it; no ready line before a records run of `security/Proofs` passes at the head.

