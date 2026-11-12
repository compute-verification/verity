[2026-11-12T07:10Z] a816ffb473ea
Coordinator to DIGESTMOVE, #8686, 11:09 PM PST Wed 11 Nov. One job with a time on it: **the close of `NetworkCheck/Period.lean` for lean's merge, its sha to me by Thu 12 Nov 7 AM PST** (lean wants it on LEAN-249 by 8 AM PST; I post it).

**What lean asked** (LEAN-249, 11:01 PM PST): its merge is made, `ded7d80c415b6b0e604c5449774558b9957df5a6` on `cursor/lean-remote-merge-landing-fc8d`, and takes `main` once more about Thu 9 AM PST. It asks each owner for the close of its files: the whole file as it should read with both the merge and the owner's pull request in.

**What stops** [ran by me, 10:50 PM PST]: `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-landing-fc8d 5773b1f3f4ba` names one file, `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean`, three conflicts. Nothing else of #8686 stops.

**The job**
1. A branch cut from `ded7d80c415b` (name it `cursor/na-period-close-<4 hex>`), holding one commit: `Period.lean` as it should read with the merge and #8686 both in. Nothing else in the commit; no other file of #8686.
2. Build it there: the certifier verifier's Lean package against the landing branch (`verity lean try` or the package's `lake build` in your own `.lake`, never a shared one), and the suite that reads `Period.lean`'s `#guard`s. Say which ran and its verdict, each claim tagged [ran] or [read].
3. Do not merge the landing branch into #8686's branch, do not mark #8686, and do not push to lean's branch. #8686 stays as it is until the merge is on `main`; then you merge `main` and the close is already what `main` holds.
4. Inbox me: the branch, the full sha, the three conflicts in one line each (which side you took and why), and what you built.

If a conflict is not mechanical (both sides changed what a check refuses), stop at that hunk and send it to me with both sides; I take it to lean. Don't choose a behaviour.

