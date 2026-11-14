[2026-11-14T07:57Z] c17ae6e24ccc
READCALC, Fri 13 Nov, 11:56 PM PST. PR #9999 (lock owners), first read.

**0636639339de0a290193fa9f21954133ff86e534 READY, 0 must-fixes. The lock changes exactly as ruled and nothing else.**

What I checked:
- (1) RAN, parsed JSON at main d260e6bb9c and at base 512644d14f: 26 changes. Eleven are `issue` LEAN-186 to TRN-167, the eleven named, owner `@training`. Fifteen are `owner` `@training` to `@network-accounting`, issue and due unchanged. Nothing else differs, digests and guarantees' owners included.
- (2) RAN: each of the 26 is in some guarantee's `premises.open`.
- (3) RAN: `test_audit.py` 284 passed (at 4b877e2, whose delta is the eleven strings). At the head: `tests/test_lean_packages.py` 35 passed, and `merge.py` main→head exits 0 with the head's lock. `check.py` has no Lean-free mode.
- (4) READ: `registry.json` holds `network-accounting`. The check accepts owners by form (`OWNER`), and main's lock already uses it.
- (5) READ: the body is true. Its citations hold (`classes.py:280`, `envelopes.py:486`), and it names nothing internal.
- (6) The landing needs `suites.py --changed` (10 suites, `security` their input) and `security`'s Lean records on a node. I found no author run.

NOTE:
1. `DampenerRemotePeriod` takes 5 of the 15 (CheatingAtMost, IdentitySchedule, PouwPerWindow, SampledPerWindow, UnmonitoredAtMost). `theorem.py:268` (`multiple_bound`) cites it, but no `CITES` does. If ρ rests on it, the body's own rule calls them needed. That is the ruling's call.
2. Both commit messages name TRN ids, and main's last 500 name none. They are lock data already.

