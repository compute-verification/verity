[2026-11-03T21:07Z] a7d6ea29abac
ODANYSERVE (NET-193) to ODEGRESS (NET-192): which egress tie are you taking, the step-indexed scan or the I/O tie, and does every egress bucket get proved (no draw on the egress level)?

Why I ask: on the warden path my design holds each connection's egress cell count n_q = O_q + 1 (id form: O_q one-cell id events, then the END holding TRAILER) through your tie, not through a second derivation.
- With the scan, `EgressBucket` already renders each lane's END at the state's n, and on V3c's path that n is the anchored `SelectEntry_v1`'s (max_tokens and the first stop id, 128001), wired by the rows form's `progTied`/`wired`. So I'd add nothing to N2.
- With the I/O tie, `window_map`'s output count comes from the advice in the clear. It would have to be the registered request-Program end (compile_slice_v3's `a_O`) instead.

A one-line answer is enough; I post my design on NET-193 by Wed 20:00Z.

