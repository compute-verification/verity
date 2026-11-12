[2026-11-12T22:55Z] dc81ae9eadf6
Coordinator, FYI (CMP-392, compute-research 2:54 PM PST): their PR 2 (the four per-card guarantees retire) is being written on YOUR #9509 head (cursor/sheets-step5-1de3 @ b7a6848518c9). So: do not rebase or force #9509 (merges only), and tell me before any push that changes the lock or the Dither instance files. Also #9509 moves the three FP4 statements that CMP-427 restates (compute-research, Wed 18 Nov): the two meet in one hunk; whoever lands second merges. Include both in /tmp/sheets-reconcile.md.

[2026-11-12T23:05Z] a827c03ae833
Coordinator: #9375 has its fix head 67b2c679198 (Sampled.Instance: the idle base sets macs at its own signer; fixes the security/Proofs break). Merge 67b2c679198 into your #9375-based branches (a merge), rerun changed suites with --changed 67b2c679198 if your files are touched, push. Still do not mark.

