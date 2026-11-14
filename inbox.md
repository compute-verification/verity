[2026-11-14T08:32Z] e142083d3608
PREMISES, Sat 14 Nov 12:32 AM PST. **periods.py task: DampenerRemotePeriod is not the sum over servers and windows (it is one pooled inequality over the period), so neither option fits as written; I recommend citing it on the remote rows' "not scaled" line, where its terms are, and building nothing until you answer.**

The difference (READ: security/Definitions/Nci/Guarantees/RemotePeriod.lean on main, benchmarks/rho/periods.py and rho/envelopes.py):
- The statement has no per-server or per-window bound and no sum over servers. Windows enter only as coin rounds: sum cap_t <= C, and the failure terms p/eps', J(1-q)^(K+1) and sum_t ks_t. Its only sum over blocks is over declared days, n_i blocks a day at CountHE's form.
- periods.py's "scaled" line is the server block (one server over the period) times the fleet count. RAN: in all four remote rows (reference-remote-h200 and -sampled, one datacenter and the world, a year) its share is 0.0%. The rate is the "not scaled" line: routes 99.7%, 0% and 99.7%, plus the unregistered pool; sampled, one datacenter: capacity 99.9%.
- Term by term, those routes are the statement's terms, each a share of the fleet's capacity (ctx.capacity is the whole fleet's, so summing per-server shares equals the pooled share):
  - undecl.host + undecl.gpu + undecl.clock = r-hat
  - skip.pouw = eps_W W_cert/C
  - skip.certs = eps', with its delta window_delta_coin/eps_agg = p/eps'
  - break.lag = J Delta/H
  - break.escape = J K B/C
  - exfil.incorrect = chi
  - unregistered = s
- Where they differ (INFERRED, not traced to a number):
  - (a) The escape: the code takes K = luck ln(1/p_evade)/q and reports delta = p_evade. The statement's tail at that K is J(1-q)^(K+1), about J p_evade, so its delta is J times larger unless luck (declared 8) covers it.
  - (b) sum_t ks_t grows with the number of windows; the code's crypto delta is per envelope, not summed over windows.
  - (c) The inference cut: the statement's n_i is each day's verdict count, while the code's n_S is certified work over c; nothing ties the two. min-cut's HE has CountHE's per-block form (READ), but the cover class has no place in the statement, and it sets sampled, one datacenter (RAN).
- So the five premises are premises of the pooled terms (PouwPerWindow: eps'; SampledPerWindow and IdentitySchedule: the escape and delta; UnmonitoredAtMost: s; CheatingAtMost: chi), not of the server-block sum.

Recommendation: for a remote scenario, the "not scaled" line cites DampenerRemotePeriod for the routes and the pool, and lists the five with their lock status (open, TRN-167). They are not in data/premises.toml; adding them there would put them in the ranking, which I would not do here. Differences (a) to (c) go in the docstring as where the code and the statement differ. The "scaled" line stays "scaled" and cites ServingBlockBoundRemote per server (after #10014): no statement sums it over servers. Unanswered in 30 minutes, I build this.

Also:
- #9999: READCALC gave READY at ea8f747 with 0 must-fixes. Its notes:
  - INFERRED claims. I edited the body (the head did not move): the server-block line now says no cut on main cites ServingBlockBound yet (#10014 does); the period-form line now says what the reading above found; "network accounting read all ten" is marked as their answer relayed by you.
  - The first commit's subject ("15 to network accounting") is false at the head. Fixing it needs a rebase, which I may not do; I leave it, and the head and body are right.
- Runs (RAN, all waiting, about 50 min to a slot): lock records r20261114-082245-872d and lock suites r20261114-082250-da8e, both at ea8f747; gap quick tier r20261114-082719-73de at 2800c4b. I mark each PR when its runs are done at its head.
- Seen addressed to @training: none.

