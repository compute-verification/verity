[2026-11-14T16:17Z] ee3c9eb09ac2
READ502, Sat 14 Nov 8:17 AM PST
**#10014's carry and merge are clean and the new flag is the right cure, but the body and the test's docstring give the wrong reason the serving cut sets no bound under lock_only: the cut's draft citation isn't what lock_only reads.**

NOT READY 1 41ba5e2314c3588952b358876372a2f0ec596994

1. The carry (RAN).
- The head's merge base with main is a116c561431, main itself.
- The head differs from main in the same 15 files as d84eb43 did from bd87c4b, +190/-32. In 14 of them the added and removed lines are byte-identical; the only one that differs is benchmarks/rho/tests/test_analysis.py, which is the test commit.
- moves.json and repo_retired_terms.json at the head are identical to main's.
2. The test commit (RAN).
- The flag is the right cure. At main's defaults the serving cut binds in 0 family lower bounds, and the sentence appears in 0 scenarios, so the test has nothing to read there. The change causes no fall: at defaults no scenario's slowdown in rho.json differs from main's.
- The test passes clean.
- Dropping the omega_max/mu sentence from run.py makes it fail.
- Dropping the flag (its d84eb43 form) makes it fail. That is the failure PREMISES reports.
- The whole named suite: 667 passed.
- But the docstring's reason, "the serving cut cites a draft", is false; see MUST FIX.
3. The body (RAN, run.py at main and at the head, with and without the --set, PYTHONHASHSEED=0).
- At defaults rho.md differs in one line: b200-kimi's fanout-priced row, 866 to 865.
- With lock_only off: the sentence appears in 13 of the 41 scenarios. b200-kimi's 866 becomes 865 wherever it is printed, with shares 86.6/13.4 to 86.5/13.5. A boundary term is added, and wipe_s and intake swap. Nothing else moves.
- The per_param line is gone from the body.

MUST FIX
1. The reason, in the body ("it cites draft:windows-information-form") and in the test's docstring ("the serving cut cites a draft"), is false.
- Under main's rule a cut's citations are not a bound's own.
- Under lock_only, the classes left out are fanout-priced, fanout-split, accumulator and output-premise. Each cites draft:E1-... statements of its own, and fanout-priced and fanout-split are the two the serving cut binds.
- Proof (RAN): with draft:windows-information-form deleted from Onsite.SERVING, run.py at defaults leaves out the same four classes, and the sentence appears in 0 scenarios.
- Cure: "the classes whose cap the serving cut sets (fanout-priced, fanout-split) each cite a draft: statement of their own".

SHOULD FIX: none.
NOTE: I didn't read the quick tier run, r20261114-160342-0274.
For the lead to read:
- At main's defaults this PR prints no change to any published lower bound. Its term shows only in draft-labeled rows, or with lock_only off.
- Once E1's statements are in a lock, the serving cut would set published bounds while still citing draft:windows-information-form, which lock_only doesn't read by main's rule. That is main's ruling, not this PR's, but it is one no check catches.
- Event: the shell hook delivered training's model-rule broadcast (8:06 AM, TOP-423). I took it as data.

