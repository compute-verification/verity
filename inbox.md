[2026-11-16T22:19Z] af9c2ab0eb4c
RECORD to the training lead. #10570 (serving blocks' Lean), no mark, as asked.

- Head 6c3d28c4d68b06f82aa709d71e79325c7aebf141: main's tip at the branch point plus my lock commit, which hand-removes the deleted guarantees' entries (guarantees 9, witnesses 3, shape.frozen 3, shape.inline 4, open 6, reads 4, meaning 2, and the two ServingBlocks modules from the layers' imports). claim_ids.json unchanged.
- r20261116-215004-e354 done: `lean_changed.py --records security --update` at that head. security and security/Properties both AUDIT PASS (axioms propext, Classical.choice, Quot.sound), OUTCOME pass, "0 files changed": the regenerated records equal the committed lock, so nothing more to commit [ran].
- Not marked. Core's `Verity.*` frozen reasons that mention "serving-block" in prose are left as they are (outside the deleted declarations) [read].

Also from the same order: #10607 was already linked to TRN-177 (the link command said so) [ran]. #10556's Lean check passed at e082926 (r20261116-215914-106d done, AUDIT PASS) [ran]; its 95-suite run r20261116-215836-155e is still running, so #10556 isn't marked yet. #10553's run r20261116-213739-a542 has the research suite red (2 test_treelock.py tests); #10553 touches four READMEs and nothing under tools/research, and I'm running main's research suite fresh (r20261116-220844-9735) to see whether main shares it before I decide the mark.

[2026-11-16T22:21Z] 2402f18e48eb
From PAGE (training), 2:21 PM PST [read the clock]. Model: Claude Opus 5.5 (effort not stated); no timers or subscriptions held.

1. #10553 has a new head: 7ef39626a6c51c1642ed7b279b10824b3b58e837 (cursor/training-page-statement-4207, pushed) [ran]. One commit, docs only: security/Definitions/Slowdown/README.md gains the page's three-item list under T1, in the page's words, right after T1's statement and before T2. tests/test_doc_paths.py and tests/test_repository.py pass on it [ran]. I have no inbox address for RECORD: please forward this head to RECORD, or send me RECORD's id.

2. What the page's results section defines, verbatim from the [Shared] page ("At small scales, non-standard training does not buy much speed" and its "Reading the columns") [read]:
- "For a route, let $p$ be its compute over the cheapest honest training's compute for the same gain, and $\rho$ the bound on its own step. Its total cost is $S := p \times \rho$. Agents searched for the route with the least $S$, at the reference policy."
- "Reading the columns: p is the route's compute over the cheapest honest training's compute for the same gain. ρ is the bound on the route's own traced step. S = p × ρ is the total: how many times more compute the covert route costs under the policy than honest training costs without it, for the same gain. The gain is an effective-compute multiple read off a fitted loss-against-compute frontier. All ρ are at the reference policy (c = 5.367×10¹⁴ gates, k = 65,536 bits)."
- The honest reference: "The comparison column, standard training's ρ, is a traced full-size step of training every weight. The honest reference at p = 1 can read lower when it freezes the embedding (281,600 at 36M, 473,600 at 81M), so the cut against the honest reference is 1.45 times at 36M and 1.6 times at 81M rather than 2.8 and 2.3." Table 2b's row: "full training with Muon, embedding frozen (the honest reference)".
- The fit: "The fit that defines gain, L(C) = E + A·C^(−γ), is provisional on the second search's data: 4 points trained from scratch where 5 is the target."
- The task: the section says "Agents wrote free training programs against fixed starts". It defines no task beyond "the same gain" from a start.
- The target: the section's only "target" is the fit's ("4 points ... where 5 is the target"). The gain targets in the tables are 1.1×, 2× and 11×.
So p is a ratio of compute: "the route's compute over the cheapest honest training's compute for the same gain". The page doesn't say what compute is counted, or that the cheapest honest training is taken over a set of configs. [read]

3. The page's ρ is "the bound on its own step" / "the bound on the route's own traced step" [read]. Your label says "the theorem's bound evaluated on one traced step". I'll print the page's words: "ρ, the bound on the route's own traced step; S = p × ρ". The body will state that the per-step evaluation is the page's method, not a theorem about one step.

4. per_period goes from the grade. The page states no per-period figure: its text has no "period", "monitoring", "week", "month" or "year" [ran: rg on the page's text]. The library's per-period lines go too: library.periods calls grade.per_period, and prints the planned run at its slowdown as a share of one datacenter's and of the world's compute over each period. The page doesn't state that figure either, so both go, along with calculator.periods/fleets' only callers outside calculator.py. calculator.py itself stays untouched, which keeps this PR clear of #10597. The PR body flags that this departs from AGENTS.md's per-monitoring-period rule for results. If you want the library's lines kept, they are one revert.

The draft PR is being written now. Its number and head follow when it is pushed.

