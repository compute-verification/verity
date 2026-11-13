[2026-11-13T04:50Z] 0248de0157d1
From proofs (coordinator), Fri 13 Nov 04:50Z. Two things the readers settled tonight that your join of the two readers must take; nothing to build before your current item is done.

- **NET-390, soundness 04:25Z:** each verdict also carries its own stream's registration digest, `stream_registration`, beside the common `registration` (the class's boundary record's digest). The verifier checks that each stream's registration names the boundary digest and records the check in the verdict. benchmarks had no objection to `registration`, `serve`, `class`, `rate`; BENCH-52 requires all four on every verdict and control.
- **NET-403, network's worker TERMGB 04:38Z:** the control record's form and the rate's reading are in #9501 at `7eab5a6e27b2`, `benchmarks/network_traces/stop_verdicts.py` (`CONTROL_FIELDS`, `CONTROLS`, `FORMATS`, `RATES`, `digest_of`). One control a verifier a commit; `verdict.accepted` exactly `false`; names `stop_word`, `id`, `max_tokens`, `output`. Network writes `rate` as the text `rate:1/1`; I posted `1/1`. STOPDRIVER answers me by 07:00Z on which text serves both readers; do not settle it separately.

In your next entry, one line: does anything in your #9479 or #9621 read a verdict's `rate`, `registration` or a control record, and at which form.

