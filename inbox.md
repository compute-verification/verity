[2026-11-11T07:41Z] 9bfb95b4fbda
LOCKORDER (bc-bf53488b-39a2-55e7-b5d1-b8e021c5a988), NET-333, Tue 10 Nov 11:41 PM PT. Branch: a choice of mine you may want to reverse. Nothing is pushed yet.

- My run was given the branch name `cursor/verity-network-accounting-b51f`. [run] `git ls-remote origin`: it exists on origin at `6f7ddc8d55`, 19 commits not on `main` (NET-118's period maps: "Move by tools/move/period_net_map.toml" ... "Records of the put-back", last 10 Nov 02:09 UTC), merge base 2002 commits behind `main`. [run] `git merge-tree --write-tree origin/main 6f7ddc8d55`: conflicts in four Lean files, GLOSSARY.md and the lean-proofs skill.
- So "work on the one you were given" cannot be had with "cut from origin/main", "never force-push" and "never push to a branch that is not yours" together: it takes a force-push over those 19 commits, or stacking NET-333 on them.
- What I do instead: work on `cursor/net333-records-pass-1883`, cut from `origin/main` `8201dacff26e`, the name your brief gives first. This departs from your parenthesis and from the name my run was given; it is my call, not something either granted.
- If you want another branch, say so by inbox. I read the inbox before my first push and before the PR.

