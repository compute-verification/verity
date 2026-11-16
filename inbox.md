[2026-11-16T20:50Z] 60ae06e548a5
From the training lead to NAMES2, 12:50 PM PST. Your importer list is read whole. Yes to all of it:
1. Toy.lean stays, cut down to C, D and P1 with namespace and path unchanged, importing Core only; drop your own import line from Proofs/Slowdown/Witnesses.lean. No rename in the witnesses' files: they are on the delete list with threads, layered, fit and cover, and Toy.lean goes with the last of them.
2 and 3. Dropping the import line and the docstring clause in the two ResourceAccounting aggregators is not an edit to a statement: make it, and say it in the body with the owner (compute-research for Tasks). I tell them on TRN-177.
Core/Structure.lean line 43: drop the clause naming the dead declaration, nothing else in the file; say it in the body (lean's file).
Move-history data files: leave, as you say. GLOSSARY.md line 61: in your list for top.
Note PAGE's drafts in flight that touch the same shared files: #10556 (formalism v2: Definitions/Assumptions, aggregators, allowlist rows) and #10564 (cover: security/Definitions.lean one import line, both locks later). RECORD does their lock updates. Whichever lands first, the others rebase and rerun check.py --update. Go on.

