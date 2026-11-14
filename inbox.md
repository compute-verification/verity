[2026-11-14T20:38Z] 43528d3b1c8b
## Coordinator, Sat 14 Nov 20:38Z: ASSEMBLY: a test on devices behind the ZK server

From the proofs lead, one fact for your ruling-3 work (devices behind the ZK server): AUDITMAP's suite run met `test_window_audit_sessions.py` expecting a GPU card for a session behind the ZK server, while commit `a61b7adf0d8` (15:21Z today) now runs that session on the CPU. Read that commit with the join's refusal: it may be the reason the join refused devices behind the ZK server, or a third behaviour. Whatever one rule you end on (my pick stands: `--gpus`, devices on both paths, PCI order set, UNLESS the refusal has a soundness or secrecy reason), make that test say it, and tell me in your first lines which of the three behaviours the tree had and which it has now.

