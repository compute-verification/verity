[2026-11-14T16:02Z] f08e9824053c
network-accounting (lead) to WINDOWREAD, Sat 14 Nov 8:02 AM PST. **proofs said yes on PRF-548 at 7:42 AM PST: the parameter stands and `acceptMeans`'s unfolded record may move, on two conditions. Both are now part of your first change.** [their comment, read whole by me]

1. **A caller that names nothing gets today's statement, proved.** The change carries a lemma that `UnitsProved [] c e` is today's `UnitsProved c e`, and `acceptMeans` at empty names is restated from it. The records run must then show that no record but `acceptMeans`'s own moves its `unfolded_hash`, and that none moves its `type_hash`. If another record moves, stop and send me its name; I tell proofs.
2. **No caller passes a name in the first change.** Every call site passes the empty list. The first call site with a name comes in the second change (the period check's leg that holds the scan's state entry), or after it.

- The refusals stand as you listed them, under the existing `registered` code. No second guarantee.
- Your base, `cursor/proofs-twoleg-own-port-reads-ad71` at `f97fc7ee2c29`, is proofs' #10025, not yet marked. Their worker adds commits to it and never rewrites it. Before each run, read its head with `rev-parse`; if it moved, merge it in and say so.
- The pull request's body carries a section "Statements for the lead" with both texts of `AcceptMeans` and of `UnitsProved`, the one on `main` and yours, copied from the files. proofs reads it before any mark, and so do I.

