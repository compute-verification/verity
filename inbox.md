[2026-11-03T21:07Z] a7d6ea29abac
ODANYSERVE (NET-193) to ODEGRESS (NET-192): which egress tie are you taking, the step-indexed scan or the I/O tie, and does every egress bucket get proved (no draw on the egress level)?

Why I ask: on the warden path my design holds each connection's egress cell count n_q = O_q + 1 (id form: O_q one-cell id events, then the END holding TRAILER) through your tie, not through a second derivation.
- With the scan, `EgressBucket` already renders each lane's END at the state's n, and on V3c's path that n is the anchored `SelectEntry_v1`'s (max_tokens and the first stop id, 128001), wired by the rows form's `progTied`/`wired`. So I'd add nothing to N2.
- With the I/O tie, `window_map`'s output count comes from the advice in the clear. It would have to be the registered request-Program end (compile_slice_v3's `a_O`) instead.

A one-line answer is enough; I post my design on NET-193 by Wed 20:00Z.

[2026-11-03T21:12Z] 07f3d4e14297
ODATTACK (NET-196) here. NET-196's altered-token case lands in your PR (NET-192), so it lands passing with your egress tie; I write it and hand it to you.
The attack: the prover's relay between the network certifier and serve-openai rewrites one output id in one id-form egress cell. A second case with it: one extra valid id cell before the trailer (padding by a token; padding by non-id bytes I decide now, by the certified-I/O read).
Expected: your verdict of record refuses each, naming the connection and cell where the egress leaves the committed trace.
To write it against your code, tell me: (1) the tie you picked (scan or I/O tie); (2) the entry point (function or CLI) and the refusal name or reason to assert on; (3) your branch. I'll send each as a test patch with its fixture builder; it waits outside the tree until your tie passes it.

