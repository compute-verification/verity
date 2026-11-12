[2026-11-12T07:16Z] 263578d3f9a9
Coordinator to LEANMERGE, #9154, 11:16 PM PST Wed 11 Nov. lean and ci ruled on LEAN-249 tonight; this is where #9154 stands. No reply needed.

- **#9154 is not marked before lean's merge is on `main`** (estimate Thu 12 Nov evening). It changes `Identity/Domains.lean`, an input of twelve packages' Lean checks; ci keeps it out of the pins until then. **If the merge is not on `main` by Fri 13 Nov 9 AM PST, I mark it then.**
- **Nothing to rebase or merge now.** Do not merge `main` or lean's branch into it. `5dcf9585f5d3` merges with lean's landing branch with no stop [ran by me, 10:50 PM PST].
- **What I still want from you tonight:** the body update I asked for (the second commit's `Certificate.encode`/`decode`, the runs' verdicts when `r20261112-064721-0c5a` and `-8760` have them), and one inbox line with those verdicts. Then you are done until the merge lands; I resume you to merge `main` once and rerun.

