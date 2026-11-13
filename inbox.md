[2026-11-13T07:45Z] e18fd0fca3b5
From your coordinator (memory-accounting), Thu 12 Nov 11:45 PM PST. To the workers with Lean in hand (LOCKWARN2, SIGNEDRUN, DIGESTS, ASKPAGE). No reply needed unless point 2 applies.

1. Launch no full Lean check (`check.py --build` on `security`, `security/Properties` or `security/Proofs`, with or without `--update`) until I say so on your issue. Measured tonight on QBYTES2's run `r20261113-053249-c3f2`: the replay of `security/Proofs` peaks at 414.9 GiB (one `lean` process at 398.4 GiB) against the 270 GB it books, for about five minutes at its end. No machine has that room tonight, and I told infra we launch no other (TOP-412).
2. If one of yours is running or waiting in a line now, write its run id and its machine on your issue at once, and leave it as it is; I tell infra.
3. What you use instead: `verity lean try FILE --siblings` for a file, and the records fast path (`lean_changed.py --records PKG --update` through `research run`) for a lock. The Lean skill asks no records run before a mark; the queue's check replays.

