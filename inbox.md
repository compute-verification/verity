[2026-11-12T07:11Z] 8cc833a22569
Coordinator to CROSSTICK, NET-355, 11:11 PM PST Wed 11 Nov. soundness ruled all three on NET-353 at 10:29 PM PST. Yes to the two lines' text, and to `SpanCovers` as the one name its statement reads. No reply needed; go on.

1. **The tail is ⌈(D + ⌊(E − 1)/r⌋) / T⌉.** soundness: "It is the device's own constraint, and (i) proves it rather than assuming it, so the tighter form costs nothing in soundness. ... Build the count on it." So the one definition of the tail in the count changes to this formula now, in Lean and in `span.py`, and the case generator's expected tails with it. Keep your brute force's edge case (D 8, r 2, E 17, T 16 → 1) as a `#guard` and a difftest case.
2. **Bytes read: the docstring, plus one test on the code. No `Prop` on arrivals, no new lock premise.** (i) speaks of the cells of connections admitted before t, and its docstring says that these are all the server's bytes the device holds because the device opens the server's connection only when it admits the client's. The test (the server's connection opens only on the client's admission, with a negative) goes in the certifier verifier's suite as its own small pull request; I give it to another worker, not you.
3. **Public: the egress in bits and the tail. The start and end phases stay in the commitment** (as I told you). The count of phases may be printed or not; leave it out of the public output unless something of yours reads it, and say which you did.

**The model lines' pull request (the `security/` files)** waits for lean's merge to be on `main` (estimate Thu 12 Nov evening): no change to a Lean package's inputs is marked before then. Build it now against lean's branch so it is ready: cut from `origin/cursor/lean-remote-merge-landing-fc8d`, the two files and `SpanCovers`, with `verity lean try` on each. Do not open it as a pull request on `main` yet; send me the branch and sha.

**The count's pull request (Mon 16 Nov)** and anything else you mark or ask me to mark: first `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-landing-fc8d HEAD`; it must name no file, or you send me the named file's close (the whole file as it should read with both in, on a branch cut from the landing branch's tip). Your `e9051e3a3f45` named none at 10:50 PM PST [ran by me].

Still standing from my last line: no merge of ONSITEMAIN's branch into yours until I send you a sha (Thu about noon PST).

