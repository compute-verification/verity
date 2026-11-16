[2026-11-16T21:21Z] 4686830f7e9a
From PAGE (training), 1:21 PM PST [read the clock]. #10553 has a new head: 7881f0f7c4eebc1762927ec4b6b9c5e27c6be544 (cursor/training-page-statement-4207, pushed) [ran].
The one commit, docs only, 2 files, +7/-9:
- verity/protocols/compose/verifier/remote/README.md: all four citations of security/README.md's parts, not only line 132:
  - line 132's sentence citing "part N" now says "Training's slowdown is stated in `security/Definitions/Slowdown/README.md`."
  - line 321 "(`security/README.md`, part 4)" now cites `security/Definitions/Slowdown/README.md`.
  - line 372: "; `security/README.md`, part 2, lists them" went; it only cited.
  - line 449's clause now says "what a block's c and k count is in `security/Definitions/Slowdown/README.md` ("Inference circuits")".
  Line 558's "`security/README.md`: the definitions, properties and proofs trees ..." cites no part, so I left it.
- security/Definitions/Slowdown/Capability/README.md line 8: "How the bounds enter the security argument is `security/README.md`, part 2." is now "Training's statement is `security/Definitions/Slowdown/README.md`.", and line 9's "(`security/README.md`, Start here)" is now "(`Slowdown.Cut.Guarantees.HE`)": #10553 removed the "Start here" block that named the min-cut theorem.
[ran] tests/test_doc_paths.py and tests/test_retired_terms.py on the new head: 34 passed.
Whole PR now: 4 files, +157/-385. Back to PR 5.

