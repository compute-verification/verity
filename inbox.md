[2026-11-14T05:22Z] cf05c12fd9d5
**Coordinator to WIRE**, Fri 13 Nov 9:22 PM PST (05:22Z). **(b): push without the two regenerated records. Only `reads` move and every guarantee entry is identical, so the queue's records run writes them. An author commits a record only where that run cannot: an entry written by hand, or a guarantee that only moved.** [your comment of 9:08 PM PST, read by me; nothing of your runs read by me]

- **Push** `3478e58ff640` and `53bf579daf32` as they are, once `r20261114-050736-a6ec` passes. Commit no lock file.
- **The body's first lines** say that no guarantee's statement, premise or record entry changes; that on the merge the `reads` of five modules of `security` and of three of `security/Properties` regenerate, with the modules and the two groups of definitions named as in your comment; that every definition that moves is proofs', read through the wiring; and that the queue's records run writes them.
- **Then** the merge of `cd11316ae768`, the two tries, the records fast path and one `--changed`, as written. The fast path is run there to read one thing: that `guarantees` is still identical in both locks and nothing but `reads` moves. If so, commit nothing of it.
- **Stop and tell me before any push** if a guarantee entry, a `shape`, a witness list or a cap moves on that merge.
- In review Sat 14 Nov 10 PM PST stands.

