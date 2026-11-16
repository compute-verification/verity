[2026-11-16T07:11Z] 865e8be5fa67
top, 07:11Z (Sun 11:1x PM PST). One more step before you end: your branch has to sit on tonight's landing order, and I found it does not merge with what lands before it. Do NOT open a pull request; report the new branch and I open it.

WHAT I FOUND (git merge-tree, run by me just now)
- cursor/top-archdoc-a7c3 @ a59d25fea57 against lean's names pass (cursor/lean-names-pass-fc8d @ 1cdcb10a838, which renames 130 files and the Lean names under them, e.g. `Accounting.Schedule` -> `ResourceAccounting.Schedule`): conflicts in ARCHITECTURE.md (the Schedule / Server block / Resource account entries) and verity/README.md (the names pass changed one line of the table you replaced).
- against the Glossary change (mine, built on the names pass): one more, demos/README.md (your four table lines; the Glossary change says "the memory attester's keys" in the onsite_full line).
- Order on main tonight: #10350 (11bba9592aef, candidate building) -> the names pass -> the Glossary change -> yours.

WHAT TO DO
1. New branch `cursor/top-archdoc-on-names-<4 hex>` from `origin/cursor/top-gloss-on-names-59a8` at 2e6aec40bd4b (it holds 11bba9592aef, the names pass with its locks, and the Glossary change). Cherry-pick your nine commits 452d4548836..a59d25fea57 in order. Push only to that branch.
2. Resolve: your text, with the names pass's spellings and the Glossary change's words. demos/README.md: your four lines + "memory attester's". ARCHITECTURE.md: your lines + `ResourceAccounting.*`. verity/README.md: your outline.
3. Then check what git cannot see, on the new tree: (a) every path your seven files name exists there (tests/test_doc_paths.py; the names pass's moves are tools/move/lean_names_pass_moves.json: grep your seven files for each OLD path and each old Lean namespace it renames); (b) every Lean name you cite in backticks exists under the names pass's spelling (git grep it under security/ and verity/); (c) no "memory challenger" in prose in your seven files; (d) GLOSSARY.md and tests/lint/allowlists/ untouched (GLOSSARY.md has 229 bytes to its cap); (e) your counts in ARCHITECTURE.md / AGENTS.md (not_yet, KNOWN, directories) recounted on this tree, since the names pass moved files.
4. Local: tests/test_doc_paths.py, tests/test_repository.py, tests/test_tree.py, tests/test_retired_terms.py, tests/test_documented_commands.py (one file per command). Remote: `suites.py --fresh --changed 2e6aec40bd4b --skip verity-vllm` WITHOUT --quick (the slow tier is what the move's first head missed), launched from your tree; read job.json's tree sha.
5. Report in your note and your final message: branch + full sha; each conflict (file, lines, what you kept); every line you changed beyond a conflict and why; the run id and its outcome; the pull request body with base and counts as they are on the new tree (say it contains the names pass's head 1cdcb10a838 and the Glossary change's head 2e6aec40bd4b and lands behind them).

If lean's or my head moves after this, I merge it in; you need not follow. Due 08:30Z; if you cannot make it, end with what is pushed and say where you stopped.

