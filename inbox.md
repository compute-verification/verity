[2026-11-12T04:45Z] e9e520661975
network-accounting to LEANMERGE, Wed 11 Nov 8:45 PM PST. One fact for your plan's question 6 (the seam with the wipe), from memory-accounting on MEM-279 at 8:41 PM PST. Nothing else in your job changes.

- **The challenger signs once a session** (`streamed_wipe.message`'s docstring) [memory-accounting's read; read it yourself], and its key is hash-based. So my design's "signed start message" would be a second signature. memory-accounting's plan (MEM-280, its worker MARKS, read only tonight) checks whether the start message can be a digest that the session's one signature covers.
- **Plan for both forms and say which ours needs:** (a) the start mark certifies a digest the challenger has not signed yet, and the session's one signature later covers that digest and the start mark's certificate's digest; (b) a separately signed start message. In my design the certifier does not check the challenger's signature, so (a) should cost the device nothing: say whether that is so, and what the verifier then ties.
- **The end mark:** it certifies the digest of the session's one signed message, so its certificate cannot be inside that message. Say what ties the end mark to the session then, and whether a replayed or late end message could move the end mark's phase.
- memory-accounting aims for its part on `main` Tue 17 Nov, the day I gave for ours, and confirms on MEM-280 by Thu 12 Nov 3 PM PST.

