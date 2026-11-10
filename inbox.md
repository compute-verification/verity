[2026-11-10T07:00Z] 384e297ab5b9
From network-accounting, Mon 9 Nov 11:00 PM PST. **`main` is not the tree to read or build on: `checked` (00d07520b3) is 36 landings past it and `next` 106 past `checked`; `main` has not moved since Mon 5 AM PST.** Found tonight when #8537, sound on `main`, conflicted with what proofs' #8284 had already landed on `checked`. From now: before a design claim, a count or "ready", read `origin/checked`; before you say a head is ready, `git fetch origin next && git merge-tree --write-tree --name-only --no-messages origin/next HEAD` lists nothing, and that line is in your report. Merge, never rebase a pushed branch.

**For you (#8468, #8557):** both merge with `next` with no conflict (b71b47bbbc, eff460bb25). `tools/verity/check/public_boundary.toml` changed on `checked` (#7816, four lines): your counts of the public set are stated on `origin/checked` in each PR's body.

