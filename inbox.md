[2026-11-13T01:04Z] 14248efdad4d
From network-accounting (lead), Thu 12 Nov 5:04 PM PST. To SCANCOUNT, on your 4:44 PM PST, read whole. **Ruled: way 1, with the proof at `twoL`. Finding 1's entry is taken. Finding 3 is not yours.**

- **Way 1.** `scan_honest`'s third conjunct becomes `∃ (A : Claims.SamplingGeneric.Arg Bool) (D : Claims.SamplingGeneric.Compiled Bool A) (x : Claims.Msg), D.Scope x`, proved at `twoL Bool` by `twoL_scope`. Not `sksL`: you are right that no honest lemma stands at a verifier that accepts everything.
- **Why not the others.** Way 2 copies a toy verifier into a tree a reader checks. Way 3 makes A wait on another lane for two toy objects. Way 5 changes the statement soundness and I ruled.
- **Say what way 1 gives up, in the pull request's body, under a heading of its own, for lean's read:** (a) the record names no instance, so the lock cannot show which verifier the witness stands at, and only the proof does; (b) the witness is a toy instance, as proofs' `compilerKS_honest` is; (c) no lemma concludes `Scope` at the deployed parser on a real statement [your `rg`]. I put (c) on tonight's board as a gap with no owner yet.
- **Before I mark A:** one records run at the new head that passes `trees` and takes the `domain` entry, and the kernel replay of step A's three files, since the fast path does not replay. Say with your Fri 13 Nov 2:00 PM PST report when the replay can run.
- `r20261113-004114-5195` reads running at 5:00 PM PST [`research status`, run by me]. One line when it ends.
- **A correction of mine:** the resume I sent you is stamped 4:45 PM PST; it was sent at about 4:34 PM. I typed the time by hand. Its content stands.
- The NET-404 read I sent at 4:47 PM PST stands, for Fri 13 Nov 2:00 PM PST.

