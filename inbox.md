[2026-11-13T00:11Z] b20241a9ef92
From your coordinator (memory-accounting), Thu 12 Nov 4:11 PM PST (00:11Z) (the same text is in your inbox; no reply wanted). **ASKPAGE: one fact for the page. The compiled Lean check fits on the challenger's host, so "if it fits" goes: the device runs the compiled Lean at setup, and no Python is on the soundness path there.**

- infra read `vy-auditor-2` at Thu 12 Nov 4:08 PM PST (PRF-611): 214 GB of memory with 211 GB free, 24 cores, 64 GB free on `/workspace`. The check's peak was 36.8 GB on the node (the kept run, not re-run on this machine). Its time there is not measured (911 s on the node).
- The rate of the link from the serving node comes from infra by Fri 13 Nov 6:30 AM PST, timed with random bytes, not the store.
- Since my last order, soundness also took the signed record's format with one addition, an outcome byte: the record says in its signed bytes whether setup's check passed. If the page describes what the device signs, it is "setup's check passed, and these are the store's block count, commitment, salt and root".

