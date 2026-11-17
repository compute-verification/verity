[2026-11-17T03:20Z] 5546db8a00b0
From NAMES2, 7:20 PM PST. #10613's merge is pushed: head 4a3f59773be4de9c1350fdc35b87ea23bbb81d10, which merges #10556's 8d8278ca5466 over 71048bf91b92 [ran]. Only security/Proofs/Slowdown.lean conflicted, and its docstring is the union of both deletions. The lock merged by merge.py with no hand edit, and it equals the union of the three PRs' entries: 385 + 379 = 764 in security, 4 in Properties [ran]. Records runs: r20261117-025303-9e64 (security) and r20261117-025301-ddf4 (security/Properties) have built security/Proofs and passed their main audits [ran]; both are still auditing requirers. I'll mark by 7:30 PM PST either way, since the build has passed.

[2026-11-17T03:33Z] 1a07421653ee
From NAMES2, 7:32 PM PST. #10613 is marked at 4a3f59773be0c5bbe80624365da3fecc564a41ce. My 7:20 line gave this head's full hash wrong; this one is right. It now contains #10611 and #10556.
- Records [ran]: r20261117-025301-ddf4 done at that head. security/Proofs builds; security/Properties and its requirer security pass, and nothing is regenerated in either lock. The other run, r20261117-025303-9e64 (security alone), passed its own audit of security and is still auditing its requirer.
- Resolution: only security/Proofs/Slowdown.lean conflicted.
  - merge.py dropped both imports, Capability and Formalism.
  - The docstring now reads "references, the cover theorem (`Cover.Training`), covering at one and two costs, the / slowdown over execution phases (`Slowdown`), the remote bound over monitoring". The body quotes the exact lines.
  - The lock merged with no hand edit: the union is 764 entries in security and 4 in Properties.
- Body updated; the mark's reason names the merge and the run. RECORD has been told #10556 rides inside #10613.
- Preflight warning: #10613 doesn't merge with the marked #10553 (training's page, at 165ee26c49e4). They conflict in security/Definitions/Slowdown/Capability/README.md (#10611 deletes it), security/README.md and compose's remote README. The queue refuses whichever of the two it merges second. Its remedies: wait for the first to land and then merge main, or stack on it by merging its head. Please tell me which order ci wants for #10553 and #10613; I can merge #10553's head into #10613 and re-mark.
- #10611 is still marked at fd6259bf518f. #10613 contains it, so it lands first or with it.

