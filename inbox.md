[2026-11-16T20:27Z] 466e21065e18
From PAGE to the training lead, 12:30 PM PST. Answer to your 12:25 and status on your 12:27.

**The rename hand-in crowds me out: please give it to another worker.** The reason is not time. This VM has lost its injected Cursor secrets: `research doctor` finds none of the 24 in this shell or in a fresh tmux login shell, the "exec-daemon restart" case. That blocks three things here:
1. `research run` (no RUNPOD/Nebius keys): no remote suites and no run ids;
2. the notes push (no RESEARCH_NOTES_TOKEN): nothing reaches `lanes/training/handins/`;
3. the Linear app link (`research queue link`).
A hand-in needs 1 and 2. GitHub push, `gh` reads and the Linear MCP still work. Per doctor, a new agent VM gets the secrets at boot. If you'd rather keep me on it, resume me on a fresh VM.

**Done:**
- Check 1's README is open as draft #10553 (https://github.com/compute-verification/verity/pull/10553): `Lane training` first line, `Refs TRN-177` last, unmarked, +149/−375. It replaces `security/README.md`'s "The security argument" (the four parts, coverage, LDL, the counting assemblies) with a pointer, and keeps 13 status rows. It is not linked in Linear's app (no credentials here).
- Its checks: the four doc tests that read it ran here (test_doc_paths, test_retired_terms, lint/test_repo_retired_terms, test_glossary_lean_names: 75 passed). The three suites `suites.py --changed origin/main` selects (repository, verity-graph, the PoUW verifier's) have NOT run, for reason 1.
- The map, merged with the supplement, is posted on TRN-177. The one list of guarantees follows in a minute.

**New in the map, goes up:** Core states the older argument too. It lists 5 guarantees: `Verity.Guarantees.Counting` (Theorem 3, with `_honest`, `_nontrivial` and `CountingSum`) and `Verity.Guarantees.BlockDescription` (Lemma 1), in `Definitions/Core/Formalism`. Only Capability, ServingBlocks and formalism v2's `Slowdown` read them, and the page's statement doesn't. They are another lane's.

**Next:** (ii) by reading `Slowdown/Remote.lean` and the two Remote guarantees, then what `formalism.py`'s importers take. PR 2 can be written and pushed from here, but its Lean check can't run here (reason 1). So I will push it as a draft and say so, unless you tell me to stop at the reading.

