# Backends: rules for the proof campaign

Read the root `AGENTS.md` first. This file is the policy shared by every proof
backend and by the campaign-side numerical work under `backends/numerical/`.

## What is fixed

- The target is `verity.verification.target.FIRST`. Lanes do not change it.
  The backend transition unit is one 16-product update with arbitrary finite
  `c`; the Verity VU is `GemmCoordinate<K>` from `c_0 = 0` plus the epilogue.
- Candidates must agree on profile, input/output bitstrings, supported domain,
  the exact function and VU endpoints. Auxiliary witness layouts are theirs.
- Semantic disagreements go into `fixtures/discrepancy_log.json` (an entry
  with an owner), then to the coordinator. Changing a reference
  implementation resolves nothing by itself.
- Proof gadgets (selectors, quotient/remainder, leading-bit, lookup layouts,
  auxiliary columns) live with the backend or in `backends/numerical/`, not in
  `verity.verification`, unless one proves to be a durable backend-independent
  obligation representation.

## Labels every number carries

~~~text
proof_class     ARITHMETIC_DIAGNOSTIC | NON_ZK_PROOF_DIAGNOSTIC | NON_ZK_PROOF | COMPLETE_ZK_BACKEND | NO_PROOF
authentication  included | excluded
~~~

Never compare across `proof_class`. A non-ZK kernel is a diagnostic; a
backend is complete only when its interactive, private, hash-based
composition is implemented and its soundness accounting is written.
Guest-cycle shares are not proving-time shares: measure wall time, memory,
shards and fixed costs directly.

## Workloads and the decision

~~~text
one group               debugging
16-product update       first complete integrated checker
K=1536 GemmCoordinate   the architecture decision (handoff section 9.5)
~~~

Nothing is selected as the winner on the small cases. The decision is the
lowest complete numerical-proving cost at matched soundness, privacy and
domain on the K=1536 VU. A batch may be split into independently committed
sub-batches; the reported soundness error is then the union bound over them
(`verity_numerical.security.accounting`) and the per-VU cost includes every
sub-batch. Security parameters are stated per run (`security.target` in the
result fingerprint); the protocol documents are `backends/gkr/PROTOCOL.md` and
`backends/direct/PROTOCOL.md`, and the red team reviews their transcripts
before any implementation claims a proof class above diagnostic. Report the six timing buckets (witness,
encoding/commitment, arithmetic, lookup, ZK-additional, serialization), peak
memory, bytes moved, transcript bytes, rounds, verifier work.

## How work is done

- All measured work is a recorded `research` run on a pod; proving and GPU
  probes never run on a laptop or the control pod. `--exclusive` on shared
  pods. Agents do not create or terminate pods; the coordinator does, and the
  campaign's pods (`vy-*`) are under their own budget cap.
- Commit before `--source .`; ignored files are not shipped.
- A lane's exit gate is tests + recorded run ids + the concrete unresolved
  items. `main` is the integration point; another lane's branch is not
  canonical.
- Rust for anything whose speed is the result; Python for references,
  checkers and adversarial search. CPU baselines first, GPU after parity.
