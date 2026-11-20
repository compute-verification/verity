[2026-11-20T02:02Z] e40ff342a791
From the lead: proofs MERGED the header mend into one-circuit (bddb9b43bfc, 27 files; the road head it ran on was e1e053e0dd9). So your registered-read change now targets one-circuit directly: merge origin/one-circuit into your branch (or rebase onto it; say which), make sure headerHeldAt there is the merged one and say what changed in it under you, and give me your head and git diff --shortstat origin/one-circuit, so it is a merge candidate for one-circuit itself. Everything else in your job stands.

