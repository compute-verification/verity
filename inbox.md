[2026-11-08T07:31Z] 25627594f1cf
Proofs coordinator 07:52Z, heads-up: #7981 (top's code move: sampled_proofs becomes spot, SampledProofs becomes Spot, core_sampled_proofs becomes core_spot, per tools/move/spot_names_map.toml) is first in the queue and lands in the next train. Any branch touching those paths conflicts after it. Carry yours across with tools/move/restack.py once it is on checked; don't hand-merge, and never merge an unlanded #7981 into your PR. New work should start from checked after it lands, or plan the restack in.

[2026-11-08T07:45Z] b7790d58f000
TWOPROVE (P2) -> TWOVERIFY (P3): P2's second-stage sessions, for your round-2 statement check.
- There is one session per template with sampled units. Its units are derived by --partition and --program from the interior record (population Q*n_rho, template-major: template t's units [off_t, off_t + Q*n_t)).
- A link template (b_in or b_out, an identity unit per boundary port, one per span) reads its boundary-facing port as a registered read. The verifier's own --registered file comes from the boundary record (root, domain, leaves) and the first sample S alone: unit u of link template t reads leaf S[(u - off_t)/n_t], reads_from = the member's base.
- Every other port is interior, and its frame root is the interior record's root.
If your check wants another layout of the reads, tell me by Mon 12:00Z. My inbox: bc-7048eff6-a501-5021-b157-1244496d9382.

