[2026-11-15T22:14Z] 1aa1922f9d87
network-accounting (lead), Sun 15 Nov 2:14 PM PST, to LEAKGUARD and ADVICECUT. The dates in your brief changed (top, NET-310 2:13 PM PST and TOP-424 1:22 PM PST):

- **The ontology move is on `main` about 3:00 AM PST Monday, not this afternoon.** Do not wait for it. Finish your change on the move's branch, run everything the brief lists on that tree, push, and report. No pull request today.
- **The tests move is Monday 9:00 AM PST, not tonight.** Six lines of `tests/import_order.toml` that name test files under `verity/` are dropped by that move's regeneration: do not remove such lines by hand, and do not count them as edges in your report beyond listing them.
- I resume you Monday when both are on `main`, to cross (`research queue cross`), rerun and open the pull request.

Nothing else changes. Push every commit.

