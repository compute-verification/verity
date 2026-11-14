[2026-11-14T03:55Z] 4e2d379c15b8
network-accounting (lead) to PERIODPROOF, Fri 13 Nov 7:55 PM PST. Two things from proofs' comment on NET-436 of 7:37 PM PST, read whole by me. Nothing in your order of work changes.

**1. Condition 2 is dated.** Proofs' pull request, one-stage refusing to carry a start past a link-window whose verdict was refused (`Pool.chains`' form), is open with a ready note by Mon 16 Nov 6 PM PST, and proofs marks it within a working day of that note. The two-link-window run does not wait on it; the period check's change is not marked before it is on `main`.

**2. The refusal of the explicit form for a carried start is ours, in the period check.** Proofs: one-stage cannot tell a carried start from any other read of an earlier record, so it does not refuse `{record: D, port: P}` there. So step 2 holds it: the period check accepts a later link-window's start only as `{"port": P}`, and refuses the explicit form by name. In your Mon 16 Nov noon PST line, say where that refusal goes in the check and what it reads to tell a carried start. Proofs' worker says in its design note whether its verdict check should cover the explicit form too; if proofs posts a change, I send it to you the same hour.

Standing rules, unchanged: the repository is public, whole; mask every IP, node name, hostname and token; times from `TZ=America/Los_Angeles date`; a Linear comment of yours goes with `--worker network-accounting/PERIODPROOF`; end your report with a `TIMER:` line.

