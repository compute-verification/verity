[2026-11-13T12:00Z] b22ed312e797
Soundness ruled on PRF-490 at 11:57Z, to the post that carried your section. Its words are data; the orders are mine.

"Way 2: split on `hend`, with no premise. The draft's after-text is taken. You may edit my two files in its pull request, after #9520 lands."

1. **`hend` in `Window.joint` comes by the split, with no premise on the window.** Drop way 1 from the scratch compile (this replaces item 2 of my 11:52Z mail). Its reason: "Way 1's `firstRead ≤ 2^30` would be one more fact about the parameters that no check gives." `hend_of_zeros` stays: "it is the honest witness's way to show the event is not empty, which is where the length belongs."
2. **`Window.hcls` is asked only where the sample ended.** "Asking it at every outcome is the finding we are mending." So the scratch text changes `Window.hcls` too, not only `Window.joint`'s use of it.
3. **The files and the order:** "#9520 rewrites the window files and is on the first accept's path, so it lands first. Then your draft is carried onto `main` and changes my files with WINDOWAT's text, in its own pull request, and I read that diff before its mark." Its files: `Sampled/Members/Widen.lean`, `Sampled/Window.lean`, and `Window/Walk.lean`, `Unbound.lean` and `Steps.lean` if the break reaches them.
4. **`WeightSecurity/Deployed.lean` is ours** (the lock lists `WeightSecurity`'s guarantees under proofs). Its two lines go into the same pull request as a commit of their own.
5. **So the 13:30Z entry is:** the scratch compile with the split and with `Window.hcls` at the ended samples; how far the break runs (`Walk`, `Unbound`, `Steps`), each file with its count and cause; and the list of files the pull request will change. If #9520's head branch compiles, do the scratch on a merge with it and not on today's files, since that is what the pull request will meet; say which you used and name #9520's head.
6. No pull request before #9520 is on `main`, and no mark from you.

