[2026-11-13T10:41Z] 701f8d195802
training, Fri 13 Nov, 2:41 AM PST. TRN-145, your 2:38 AM answer.
**Agreed: the column case stays word-level in this PR. Three things go with it, and the hash sweep is a follow-up I decide on from your before/after.**

1. **Count it apart.** `Reader.read_by` counts the column case as its own kind, by words by design, not as a fallback. The tiny test then holds rows and selects at zero fallbacks and states the column count (16 on tiny). A fallback must keep meaning "the form was not met".
2. **Time it.** The before/after gives the seconds spent in `softmax_rows`' column case, beside the whole reading, at s(2,256,512), tiny(2,1024,64) and s-t20. That number decides the follow-up.
3. **Say it in the body.** The equivalence statement names what is read by structure (rows, selects) and what is still read by words (columns: time in ctx² a product node, nothing kept past the call).

The follow-up, if the number asks for it: your sweep is sound as you state it, since equal columns give equal hashes, so "pairwise different" is claimed only where every hash differs and a tie goes to the words. Use a fixed hash of the row and the constant, nothing a process seeds. Its own PR.

Noted: 54ec read s-t20 in 12,460.4 s at 124.9 GiB under #9676 and is in the pass. Send its end.

