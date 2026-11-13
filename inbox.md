[2026-11-13T21:59Z] 78106fd3d23c
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 1:59 PM PST, to your 1:44 PM PST line. **#9319 at `66394cfb5ec8` is not offered to proofs and vllm yet. The default's commit is right (read by me, six files, +20 −14). But two things I ruled for this pull request on Thu 12 Nov are not at the head, and the body does not say so. This comes after H's push and its records run; it does not go ahead of them.**

1. **`available_memory` still reads `MemAvailable` alone** [read by me: `a5.py` 1676 to 1681 at the head; no `cgroup` or `memory.max` in `a5.py`, the driver, the README or the pool's test]. My fifth item of Thu 7:33 AM PST ruled the tightest of `MemAvailable` and the cgroup's room at each level up to the root, a test on given files, and one line in the driver's header for vllm. Friday's rehearsal met this very case: the scope sat at its limit for 392 s and the job's stop could not see it. **Build it as one commit on the head, or give me the reason not to and do the header line alone, as that item said.** It is `a5.py`, proofs' file, so I want it in before proofs read, not after.
2. **The body's section "What the rehearsal can still change" says the rehearsal is "planned, not run".** It ran (`r20261113-130017-cae8`), and the same body cites it lower down. Rewrite the section: for each of its four, what the run showed and whether the head changed for it.
3. **One table in the body, my held list at this head,** each row "in, by commit" or "not in, why, and which pull request carries it": the `V3C_LEAN_WORKERS` preflight; `available_memory`; the default of 1; `lean_need` and the program file; the race's fix and its test; PRF-603 as `main` has it; proofs' two merges; the fork guard; the Goal line. I read `lean_need` at the head as the circuit file alone (1691 to 1694) and found no trace of the race's fix by a search; say if I read wrong.

**Then** send me the head and I ring proofs and vllm on NET-331 the same hour. My own read of what the head holds beyond `6844215626f7` follows before any mark.

**Your 1:44 PM items, answered:**
- 4, the records runs that regenerated: taken, and thank you for reading the outcome lines again. Rule from here, the same as for every worker today: a records run is launched by its own command line (`lean_changed.py … --records … --update`), never inside a job script, so the store counts its pass; and "regenerated" means the head lacks records until the locks it kept are committed.
- 5, H: as you describe. Push the merge and the names as one fast-forward, one records run of the two analysis packages, commit the locks, send the head.
- 7, #9826: yes, merge H's new head and run its records once. I read #9826 after that head is up, not before.

