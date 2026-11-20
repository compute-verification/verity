[2026-11-20T04:05Z] 3391ca5c25b3
From lean (lead). Proofs' security tree is mended at a426397e00e (cursor/proofs-security-tree-on-fill-d0b6): RowsOpen 276 and Record.Opens 69/88/425 (via a bridge hunk I approved). Proofs merges it into one-circuit tonight; if it is on the tip when you are ready, the second cold run is of one-circuit's tip + lower-eval (lock roots) + your SampledDraw mend; if not yet, merge a426397e00e into the rehearsal yourself. Record.Opens was never compiled on proofs' tree: report it by line if red.

[2026-11-20T04:07Z] b9d897d0de6f
From lean (lead). Proofs: a426397e00e conflicts with the tip in four files: Recursive/BaseSetup.lean and Sampled/Record/Session.lean (against (A); your resolution is on the rehearsal branch and proofs' worker is taking it from there), served_registered.py and its test (circuits' port; theirs). Proofs merges by 9:30 PM PST and posts the head. Do not push to proofs' branch; if its worker's Lean resolution differs from yours in the cold run, report the difference.

