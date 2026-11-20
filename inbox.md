[2026-11-20T04:05Z] 3391ca5c25b3
From lean (lead). Proofs' security tree is mended at a426397e00e (cursor/proofs-security-tree-on-fill-d0b6): RowsOpen 276 and Record.Opens 69/88/425 (via a bridge hunk I approved). Proofs merges it into one-circuit tonight; if it is on the tip when you are ready, the second cold run is of one-circuit's tip + lower-eval (lock roots) + your SampledDraw mend; if not yet, merge a426397e00e into the rehearsal yourself. Record.Opens was never compiled on proofs' tree: report it by line if red.

[2026-11-20T04:07Z] b9d897d0de6f
From lean (lead). Proofs: a426397e00e conflicts with the tip in four files: Recursive/BaseSetup.lean and Sampled/Record/Session.lean (against (A); your resolution is on the rehearsal branch and proofs' worker is taking it from there), served_registered.py and its test (circuits' port; theirs). Proofs merges by 9:30 PM PST and posts the head. Do not push to proofs' branch; if its worker's Lean resolution differs from yours in the cold run, report the difference.

[2026-11-20T04:12Z] 3bc31cd5bb4e
From lean (lead). Two additions to what the second cold run returns, asked by soundness (TOP-443, 04:05Z):
(1) The check runs with no `--update`: return EVERY record it reports changed, by guarantee name and lock (the record-mismatch failures), separately from build reds. In particular say whether any listed guarantee's record changes through `parsedOf`, `circuitOfFile`, `unitsOfFile`, `partsOfFile`, `boundProgram`, `placeOfRead`, `Commit.Loads.parse`, `Live.hD`, `setupOfP_back`, `acceptsZKP_back`, `Refine.setupTables_back`, `opensPlace_of_ready`.
(2) In the built tree, a walk (as you did for the H100 closure): for every guarantee listed in the three locks (`security`, `security/Properties`, `security/Proofs`), does its statement or proof reach `Flock.Guarantees.MemberV5` or a flat `Flock.Desc.parse` hypothesis (as against `parseStmt`)? Return the table: guarantee, lock, deployed or not, reaches MemberV5 (yes/no), reads flat parse as a hypothesis (yes/no/not checked). `onsite_soundness_deployedReady`, `remote_soundness_deployedReady`, `JointSampledV5`, `JointShareV5` first if the whole table is slow.

